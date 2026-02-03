#!/usr/bin/env python3
"""
Extract sparse SAE vectors from preprocessed CSV files.
For each sentence (headline + lede), extracts activations, then latent vectors,
averages them across the sentence, and saves as sparse matrix.

Optimized for batch processing.

-- Example usage --
# Auto batch size (recommended)
python extract_sae_vectors.py

# Manual batch size
python extract_sae_vectors.py --batch-size 16

# If VRAM is tight, keep SAE on CPU
python extract_sae_vectors.py --sae-on-cpu --batch-size 8

# With shorter sequences (faster, less memory)
python extract_sae_vectors.py --max-length 256 --batch-size 16
"""

import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from functools import partial
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, vstack as sparse_vstack
from pathlib import Path
import gc
from tqdm import tqdm
import argparse
import json


# SAE Configuration
LAYER = 22  # options are {9, 17, 22, 29}
WIDTH = "16k"   # options are {16k, 65k, 262k}
L0 = "medium"  # options are {small, medium, big}


class JumpReLUSAE(nn.Module):
    def __init__(self, d_in, d_sae, affine_skip_connection=False):
        super().__init__()
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        if affine_skip_connection:
            self.affine_skip_connection = nn.Parameter(torch.zeros(d_in, d_in))
        else:
            self.affine_skip_connection = None

    def encode(self, input_acts):
        """Encode activations. Supports batched input [batch, seq, d_in]."""
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.w_dec + self.b_dec

    def forward(self, x):
        acts = self.encode(x)
        recon = self.decode(acts)
        if self.affine_skip_connection is not None:
            return recon + x @ self.affine_skip_connection
        return recon


def gather_acts_hook_batched(mod, inputs, outputs, cache: dict, key: str):
    """Hook function to store batched activations."""
    # outputs[0] has shape [batch, seq_len, d_model]
    cache[key] = outputs[0]
    return outputs


def gather_residual_activations_batched(model, target_layer, input_ids, attention_mask):
    """Gather residual activations from a specific layer for a batch."""
    cache = {}
    handle = model.model.language_model.layers[target_layer].register_forward_hook(
        partial(gather_acts_hook_batched, cache=cache, key="resid_post")
    )
    try:
        with torch.no_grad():
            _ = model.forward(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return cache["resid_post"]


def format_prompt(user_prompt: str) -> str:
    """Format prompt for Gemma-3."""
    return f"""<start_of_turn>user
{user_prompt}<end_of_turn>
<start_of_turn>model
"""


def extract_sae_vectors_batch(model, sae, tokenizer, target_layer, texts, device, max_length=512):
    """
    Extract sparse SAE vectors for a batch of sentences.
    Returns averaged latent vectors across all tokens (excluding padding).
    
    Args:
        model: The language model
        sae: The SAE model
        tokenizer: The tokenizer
        target_layer: Which layer to extract from
        texts: List of text strings
        device: Device for model
        max_length: Maximum sequence length
    
    Returns:
        numpy array of shape [batch_size, d_sae]
    """
    # Format and batch tokenize
    prompts = [format_prompt(text) for text in texts]
    
    # Tokenize with padding
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True
    )
    
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    
    # Get activations for the batch
    with torch.no_grad():
        # Shape: [batch, seq_len, d_model]
        target_acts = gather_residual_activations_batched(
            model, target_layer, input_ids, attention_mask
        )
        
        # Get SAE device
        sae_device = next(sae.parameters()).device
        
        # Move activations to SAE device if needed
        if target_acts.device != sae_device:
            target_acts = target_acts.to(sae_device)
        
        # Encode through SAE - shape: [batch, seq_len, d_sae]
        sae_acts = sae.encode(target_acts.to(torch.float32))
        
        # Average across sequence length, excluding padding tokens
        # attention_mask shape: [batch, seq_len]
        # Expand mask for broadcasting: [batch, seq_len, 1]
        mask_expanded = attention_mask.to(sae_device).unsqueeze(-1).float()
        
        # Masked sum and count
        masked_acts = sae_acts * mask_expanded
        sum_acts = masked_acts.sum(dim=1)  # [batch, d_sae]
        count = mask_expanded.sum(dim=1).clamp(min=1)  # [batch, 1]
        
        # Average (excluding padding)
        avg_acts = sum_acts / count  # [batch, d_sae]
    
    # Move to CPU and convert to numpy
    result = avg_acts.cpu().numpy()
    
    # Clean up
    del input_ids, attention_mask, target_acts, sae_acts, masked_acts, sum_acts, count, avg_acts
    
    return result


def process_csv_file(csv_path, output_path, model, sae, tokenizer, target_layer, device, 
                     batch_size=8, max_length=512, save_interval=1000):
    """
    Process a CSV file and extract SAE vectors for each row using batch processing.
    
    Args:
        csv_path: Path to input CSV
        output_path: Base path for output files
        model: The language model
        sae: The SAE model
        tokenizer: The tokenizer
        target_layer: Which layer to extract from
        device: Device for model
        batch_size: Number of sentences to process at once
        max_length: Maximum sequence length
        save_interval: Save intermediate results every N rows
    """
    print(f"\nProcessing: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df):,}")
    
    # Combine headline + lede into sentences
    df['sentence'] = df.apply(
        lambda row: f"{row['headline']} {row['lede']}" if pd.notna(row['headline']) and pd.notna(row['lede']) 
        else (str(row['headline']) if pd.notna(row['headline']) else ''),
        axis=1
    )
    
    # Filter out empty sentences
    df = df[df['sentence'].str.strip() != ''].copy()
    print(f"Rows with valid sentences: {len(df):,}")
    
    # Get sentences and labels
    sentences = df['sentence'].tolist()
    best_labels = df['best'].tolist()
    
    # Get SAE dimension for zero vector fallback
    d_sae = sae.w_enc.shape[1]
    
    # Process in batches
    all_vectors = []
    all_labels = []
    num_batches = (len(sentences) + batch_size - 1) // batch_size
    
    print(f"Processing {len(sentences):,} sentences in {num_batches:,} batches (batch_size={batch_size})...")
    
    pbar = tqdm(total=len(sentences), desc="Extracting SAE vectors")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sentences))
        
        batch_texts = sentences[start_idx:end_idx]
        batch_labels = best_labels[start_idx:end_idx]
        
        try:
            # Extract SAE vectors for the batch
            batch_vectors = extract_sae_vectors_batch(
                model, sae, tokenizer, target_layer, batch_texts, device, max_length
            )
            all_vectors.append(batch_vectors)
            all_labels.extend(batch_labels)
            
        except Exception as e:
            print(f"\nError processing batch {batch_idx}: {e}")
            # Use zero vectors as fallback for this batch
            fallback = np.zeros((len(batch_texts), d_sae))
            all_vectors.append(fallback)
            all_labels.extend(batch_labels)
        
        pbar.update(len(batch_texts))
        
        # Periodic cache clearing
        if (batch_idx + 1) % 10 == 0:
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    
    pbar.close()
    
    # Concatenate all vectors
    print("Concatenating vectors...")
    sae_array = np.vstack(all_vectors)
    
    # Convert to sparse matrix
    print("Converting to sparse matrix...")
    sparse_matrix = csr_matrix(sae_array)
    
    # Best labels array
    best_array = np.array(all_labels).reshape(-1, 1)
    
    # Save outputs
    print(f"Saving to {output_path}...")
    output_dir = Path(output_path).parent
    output_stem = Path(output_path).stem
    
    # Save sparse matrix
    sparse_path = output_dir / f"{output_stem}_sparse.npz"
    np.savez_compressed(sparse_path, 
                       data=sparse_matrix.data,
                       indices=sparse_matrix.indices,
                       indptr=sparse_matrix.indptr,
                       shape=sparse_matrix.shape)
    
    # Save best labels
    best_path = output_dir / f"{output_stem}_best.npy"
    np.save(best_path, best_array)
    
    # Save metadata
    nnz = sparse_matrix.nnz
    size = sparse_matrix.shape[0] * sparse_matrix.shape[1]
    sparsity = 1.0 - (nnz / size) if size > 0 else 0.0
    
    metadata = {
        'num_rows': len(df),
        'num_features': sae_array.shape[1],
        'sparse_matrix_shape': list(sparse_matrix.shape),
        'sparsity': sparsity,
        'nnz': int(nnz),
        'batch_size': batch_size,
        'max_length': max_length,
        'layer': target_layer,
    }
    
    metadata_path = output_dir / f"{output_stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved sparse matrix: {sparse_path}")
    print(f"Saved best labels: {best_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Sparsity: {sparsity:.4f}")
    print(f"Non-zero elements: {nnz:,}")
    
    # Clean up
    del sae_array, all_vectors
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return sparse_matrix, best_array


def load_model_and_sae(device='cuda', use_quantization=False, sae_on_cpu=False):
    """
    Load model and SAE with memory optimizations.
    
    Note: User reported that 4-bit quantization actually uses MORE VRAM (7GB vs 6.3GB),
    so we default to no quantization but with optimizations.
    """
    print("Loading model and SAE...")
    torch.set_grad_enabled(False)
    
    # Try to use quantization if requested
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.float16,
        )
        print("Loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-4b-it",
            device_map='auto',
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )
    else:
        print("Loading model without quantization (using optimizations)...")
        # Use optimizations to save VRAM
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-4b-it",
            device_map='auto',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'  # Pad on right for causal LM

    # # Wait 10 sec
    # time.sleep(10)

    # # Print VRAM usage
    # if device == 'cuda' and torch.cuda.is_available():
    #     allocated = torch.cuda.memory_allocated() / 1024**3
    #     reserved = torch.cuda.memory_reserved() / 1024**3
    #     print(f"VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    # Try to unload vision module if it exists
    if hasattr(model, 'vision_model'):
        print("Unloading vision module...")
        del model.vision_model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Also try to unload any other unused components
    for attr in ['vision', 'image_processor', 'visual', 'vision_tower']:
        if hasattr(model, attr):
            print(f"Unloading {attr} component...")
            delattr(model, attr)
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    # Check model structure for multi-modal components
    if hasattr(model, 'model'):
        for attr in ['vision_tower', 'mm_projector', 'image_newline']:
            if hasattr(model.model, attr):
                print(f"Unloading model.{attr} component...")
                delattr(model.model, attr)
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
    
    # # Wait 10 sec
    # time.sleep(10)

    # # Print VRAM usage
    # if device == 'cuda' and torch.cuda.is_available():
    #     allocated = torch.cuda.memory_allocated() / 1024**3
    #     reserved = torch.cuda.memory_reserved() / 1024**3
    #     print(f"VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    # Load SAE
    print("Loading SAE...")
    path_to_params = hf_hub_download(
        repo_id="google/gemma-scope-2-4b-it",
        filename=f"resid_post/layer_{LAYER}_width_{WIDTH}_l0_{L0}/params.safetensors",
    )
    
    params = load_file(path_to_params)
    d_model, d_sae = params["w_enc"].shape
    sae = JumpReLUSAE(d_model, d_sae)
    sae.load_state_dict(params)
    
    # Keep SAE on CPU or GPU based on option
    if sae_on_cpu:
        print("Keeping SAE on CPU to save VRAM...")
        sae.to('cpu')
    elif device == 'cuda':
        try:
            sae.to(device)
        except RuntimeError as e:
            print(f"Warning: Could not move SAE to {device}, keeping on CPU: {e}")
            sae.to('cpu')
    else:
        sae.to(device)
    
    sae.eval()
    
    # # Wait 10 sec
    # time.sleep(10)

    print(f"Model loaded. SAE dimensions: {d_model} -> {d_sae}")
    print(f"SAE device: {next(sae.parameters()).device}")
    
    # Clear cache after loading
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Print VRAM usage
    if device == 'cuda' and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    return model, sae, tokenizer


def estimate_batch_size(device='cuda'):
    """
    Estimate a safe batch size based on available VRAM.
    """
    if device != 'cuda' or not torch.cuda.is_available():
        return 4  # Conservative default for CPU
    
    # Get available VRAM
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated() / 1024**3
    available = total_mem - allocated
    
    # Estimate: ~0.5GB per batch item for activations + SAE encoding at max_length=512
    # Be conservative
    estimated_per_item = 0.3  # GB
    estimated_batch_size = max(1, int(available / estimated_per_item))
    
    # Cap at reasonable values
    estimated_batch_size = min(estimated_batch_size, 32)
    
    print(f"Available VRAM: {available:.2f}GB, estimated batch size: {estimated_batch_size}")
    
    return estimated_batch_size


def main():
    parser = argparse.ArgumentParser(description='Extract SAE vectors from preprocessed CSV files')
    parser.add_argument('--input-dir', type=str, default='data/cleaned_archive',
                       help='Directory containing preprocessed CSV files')
    parser.add_argument('--output-dir', type=str, default='data/sae_vectors',
                       help='Output directory for sparse matrices')
    parser.add_argument('--use-quantization', action='store_true',
                       help='Use 4-bit quantization (default: False, as it may use more VRAM)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--sae-on-cpu', action='store_true',
                       help='Keep SAE on CPU to save VRAM (slower but uses less VRAM)')
    parser.add_argument('--batch-size', type=int, default=0,
                       help='Batch size (0 = auto-estimate based on VRAM)')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length for tokenization')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and SAE
    model, sae, tokenizer = load_model_and_sae(
        device=args.device, 
        use_quantization=args.use_quantization,
        sae_on_cpu=args.sae_on_cpu
    )
    
    # Estimate or use provided batch size
    if args.batch_size <= 0:
        batch_size = estimate_batch_size(args.device)
    else:
        batch_size = args.batch_size
    
    print(f"Using batch size: {batch_size}")
    
    # Process each CSV file
    input_dir = Path(args.input_dir)
    csv_files = [
        'confirmatory_preprocessed.csv',
        'exploratory-preprocessed.csv',
        'holdout-preprocessed.csv',
    ]
    
    for csv_file in csv_files:
        csv_path = input_dir / csv_file
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping...")
            continue
        
        output_path = output_dir / f"{Path(csv_file).stem}_sae_vectors"
        
        try:
            process_csv_file(
                csv_path, 
                output_path, 
                model, 
                sae, 
                tokenizer, 
                LAYER, 
                args.device,
                batch_size=batch_size,
                max_length=args.max_length
            )
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
