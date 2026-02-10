#!/usr/bin/env python3
"""
Extract sparse SAE delta vectors from preprocessed CSV files.
For each test_id, extracts best and worst headlines (by CTR), passes through LLM,
masks instruction tokens, encodes through SAE, max-pools per sentence,
then creates delta vectors: best-worst (label=1) and worst-best (label=0).

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

# Our current best:
python extract_sae_vectors.py --sae-on-cpu --batch-size 5 --max-length 256 --minibatch-size 5000
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
    # outputs is tensor [batch, seq_len, d_model] for this model
    # (for some models it's a tuple, so we handle both)
    if isinstance(outputs, tuple):
        cache[key] = outputs[0]
    else:
        cache[key] = outputs
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


# Instruction prefix/suffix for content token masking
INSTRUCTION_PREFIX = "<start_of_turn>user\n"
INSTRUCTION_SUFFIX = "<end_of_turn>\n<start_of_turn>model\n"


def get_content_token_mask(tokenizer, texts, input_ids, device):
    """
    Create a mask that is 1 for content tokens (headline) and 0 for instruction tokens.
    
    Args:
        tokenizer: The tokenizer
        texts: Original text strings (without prompt formatting)
        input_ids: Tokenized input_ids tensor [batch, seq_len]
        device: Device for the mask tensor
    
    Returns:
        Tensor of shape [batch, seq_len] with 1s for content tokens, 0s for instruction tokens
    """
    batch_size, seq_len = input_ids.shape
    
    # Tokenize prefix to get its length (same for all samples)
    prefix_tokens = tokenizer(INSTRUCTION_PREFIX, add_special_tokens=True)['input_ids']
    prefix_len = len(prefix_tokens)
    
    # Tokenize suffix to get its length
    suffix_tokens = tokenizer(INSTRUCTION_SUFFIX, add_special_tokens=False)['input_ids']
    suffix_len = len(suffix_tokens)
    
    # Create mask for each sample
    content_mask = torch.zeros(batch_size, seq_len, device=device)
    
    for i, text in enumerate(texts):
        # Tokenize just the content to get its length
        content_tokens = tokenizer(text, add_special_tokens=False)['input_ids']
        content_len = len(content_tokens)
        
        # Content starts after prefix, ends before suffix
        content_start = prefix_len
        content_end = prefix_len + content_len
        
        # Set mask to 1 for content tokens
        if content_end <= seq_len:
            content_mask[i, content_start:content_end] = 1.0
    
    return content_mask


def get_best_worst_pairs(df):
    """
    Extract best and worst headline pairs per test_id based on CTR.
    
    Args:
        df: DataFrame with columns: clickability_test_id, headline, ctr
    
    Returns:
        List of tuples: (best_headline, worst_headline, test_id)
    """
    pairs = []
    
    for test_id, group in df.groupby('clickability_test_id'):
        # Need at least 2 different headlines
        if len(group) < 2:
            continue
        
        # Get best (max CTR) and worst (min CTR) headlines
        best_idx = group['ctr'].idxmax()
        worst_idx = group['ctr'].idxmin()
        
        # Skip if same headline (tie in CTR)
        if best_idx == worst_idx:
            continue
        
        best_headline = group.loc[best_idx, 'headline']
        worst_headline = group.loc[worst_idx, 'headline']
        
        # Skip if either is empty/NaN
        if pd.isna(best_headline) or pd.isna(worst_headline):
            continue
        if str(best_headline).strip() == '' or str(worst_headline).strip() == '':
            continue
            
        pairs.append((str(best_headline), str(worst_headline), test_id))
    
    return pairs


def extract_sae_vectors_batch(model, sae, tokenizer, target_layer, texts, device, max_length=512):
    """
    Extract sparse SAE vectors for a batch of sentences.
    Returns max-pooled latent vectors across content tokens only (excluding padding and instruction tokens).
    
    Args:
        model: The language model
        sae: The SAE model
        tokenizer: The tokenizer
        target_layer: Which layer to extract from
        texts: List of text strings (raw headlines, not formatted)
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
        # Step 1: Get LLM activations - Shape: [batch, seq_len, d_model]
        target_acts = gather_residual_activations_batched(
            model, target_layer, input_ids, attention_mask
        )
        
        # Get SAE device
        sae_device = next(sae.parameters()).device
        
        # Move activations to SAE device if needed
        if target_acts.device != sae_device:
            target_acts = target_acts.to(sae_device)
        
        # Step 2: Create content token mask (excludes instruction tokens and padding)
        content_mask = get_content_token_mask(tokenizer, texts, input_ids, sae_device)
        combined_mask = content_mask * attention_mask.to(sae_device).float()
        
        # Step 3: Extract ONLY content token activations (remove instruction/padding)
        batch_size, seq_len = input_ids.shape
        d_model = target_acts.shape[-1]
        
        content_counts = combined_mask.sum(dim=1).long()  # [batch] - tokens per sample
        
        # Flatten and extract only content tokens
        flat_mask = combined_mask.view(-1).bool()  # [batch * seq_len]
        flat_acts = target_acts.view(-1, d_model)  # [batch * seq_len, d_model]
        content_acts = flat_acts[flat_mask]  # [total_content_tokens, d_model]
        
        # Step 4: Encode ONLY content activations through SAE (much smaller!)
        content_sae_acts = sae.encode(content_acts.to(torch.float32))  # [total_content_tokens, d_sae]
        d_sae = content_sae_acts.shape[-1]
        
        # # OLD: Average across sequence length, excluding padding tokens
        # # attention_mask shape: [batch, seq_len]
        # # Expand mask for broadcasting: [batch, seq_len, 1]
        # mask_expanded = attention_mask.to(sae_device).unsqueeze(-1).float()
        # 
        # # Masked sum and count
        # masked_acts = sae_acts * mask_expanded
        # sum_acts = masked_acts.sum(dim=1)  # [batch, d_sae]
        # count = mask_expanded.sum(dim=1).clamp(min=1)  # [batch, 1]
        # 
        # # Average (excluding padding)
        # avg_acts = sum_acts / count  # [batch, d_sae]
        
        # Step 5: Max pool per sample (split back by sample)
        max_acts = torch.zeros(batch_size, d_sae, device=sae_device)
        offset = 0
        for i, count in enumerate(content_counts):
            count_val = count.item()
            if count_val > 0:
                sample_acts = content_sae_acts[offset:offset + count_val]  # [count, d_sae]
                max_acts[i] = sample_acts.max(dim=0)[0]
            # else: max_acts[i] stays zero
            offset += count_val

        if torch.isnan(target_acts).any() or torch.isinf(target_acts).any():
            print("NaNs/Infs in target_acts!")
        if torch.isnan(content_sae_acts).any() or torch.isinf(content_sae_acts).any():
            print("NaNs/Infs in content_sae_acts!")
    
    # Move to CPU and convert to numpy
    result = max_acts.cpu().numpy()
    
    # Clean up
    del input_ids, attention_mask, target_acts, content_mask, combined_mask, content_acts, content_sae_acts, max_acts
    # del input_ids, attention_mask, target_acts, sae_acts, masked_acts, sum_acts, count, avg_acts
    
    return result


def save_minibatch(vectors_list, labels_list, output_dir, output_stem, minibatch_idx, d_sae):
    """
    Save a minibatch of vectors and labels to disk.
    
    Args:
        vectors_list: List of numpy arrays (batches of vectors)
        labels_list: List of labels
        output_dir: Output directory
        output_stem: Base name for output files
        minibatch_idx: Index of this minibatch
        d_sae: SAE dimension (for validation)
    
    Returns:
        Paths to saved files
    """
    if not vectors_list:
        return None, None
    
    # Concatenate vectors
    sae_array = np.vstack(vectors_list)
    best_array = np.array(labels_list).reshape(-1, 1)
    
    # Convert to sparse matrix
    sparse_matrix = csr_matrix(sae_array)
    
    # Save minibatch files
    sparse_path = output_dir / f"{output_stem}_sparse_minibatch_{minibatch_idx:03d}.npz"
    best_path = output_dir / f"{output_stem}_best_minibatch_{minibatch_idx:03d}.npy"
    
    np.savez_compressed(sparse_path, 
                       data=sparse_matrix.data,
                       indices=sparse_matrix.indices,
                       indptr=sparse_matrix.indptr,
                       shape=sparse_matrix.shape)
    
    np.save(best_path, best_array)
    
    return sparse_path, best_path


# def combine_minibatches(output_dir, output_stem, num_minibatches):
#     """
#     Combine all saved minibatches into final output files.
    
#     Args:
#         output_dir: Output directory
#         output_stem: Base name for output files
#         num_minibatches: Number of minibatches to combine
    
#     Returns:
#         Combined sparse matrix and best labels array
#     """
#     print(f"Combining {num_minibatches} minibatches...")
    
#     sparse_matrices = []
#     best_arrays = []
    
#     for i in range(num_minibatches):
#         sparse_path = output_dir / f"{output_stem}_sparse_minibatch_{i}.npz"
#         best_path = output_dir / f"{output_stem}_best_minibatch_{i}.npy"
        
#         if not sparse_path.exists() or not best_path.exists():
#             print(f"Warning: Minibatch {i} files not found, skipping...")
#             continue
        
#         # Load minibatch
#         sparse_data = np.load(sparse_path)
#         sparse_matrix = csr_matrix(
#             (sparse_data['data'], sparse_data['indices'], sparse_data['indptr']),
#             shape=sparse_data['shape']
#         )
#         best_array = np.load(best_path)
        
#         sparse_matrices.append(sparse_matrix)
#         best_arrays.append(best_array)
    
#     if not sparse_matrices:
#         raise ValueError("No minibatches found to combine!")
    
#     # Combine sparse matrices
#     print("Stacking sparse matrices...")
#     combined_sparse = sparse_vstack(sparse_matrices)
    
#     # Combine best arrays
#     combined_best = np.vstack(best_arrays)
    
#     return combined_sparse, combined_best


def process_csv_file(csv_path, output_path, model, sae, tokenizer, target_layer, device, 
                     batch_size=8, max_length=512, minibatch_size=5000):
    """
    Process a CSV file and extract SAE delta vectors for best/worst headline pairs.
    For each test_id, extracts best and worst headlines by CTR, computes delta vectors.
    Creates two examples per pair: best-worst (label=1) and worst-best (label=0).
    
    Args:
        csv_path: Path to input CSV
        output_path: Base path for output files
        model: The language model
        sae: The SAE model
        tokenizer: The tokenizer
        target_layer: Which layer to extract from
        device: Device for model
        batch_size: Number of pairs to process at once (GPU batch)
        max_length: Maximum sequence length
        minibatch_size: Number of delta examples after which to save and clear memory
    """
    print(f"\nProcessing: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df):,}")
    
    # # OLD: Combine headline + lede into sentences
    # df['sentence'] = df.apply(
    #     lambda row: f"{row['headline']} {row['lede']}" if pd.notna(row['headline']) and pd.notna(row['lede']) 
    #     else (str(row['headline']) if pd.notna(row['headline']) else ''),
    #     axis=1
    # )
    # 
    # # Filter out empty sentences
    # df = df[df['sentence'].str.strip() != ''].copy()
    # print(f"Rows with valid sentences: {len(df):,}")
    # 
    # # Get sentences and labels
    # sentences = df['sentence'].tolist()
    # best_labels = df['best'].tolist()
    
    # NEW: Get best/worst headline pairs per test_id
    pairs = get_best_worst_pairs(df)
    print(f"Found {len(pairs):,} best/worst headline pairs")
    
    # Get SAE dimension for zero vector fallback
    d_sae = sae.w_enc.shape[1]
    
    # Setup output paths
    output_dir = Path(output_path).parent
    output_stem = Path(output_path).stem
    
    # Process in batches with minibatch saving
    current_minibatch_vectors = []  # Accumulated delta vectors for current minibatch
    current_minibatch_labels = []    # Accumulated labels for current minibatch
    examples_processed = 0           # Total delta examples processed in current minibatch
    minibatch_idx = 0                # Current minibatch index
    
    # Each pair produces 2 examples (best-worst and worst-best)
    total_examples = len(pairs) * 2
    num_batches = (len(pairs) + batch_size - 1) // batch_size
    
    print(f"Processing {len(pairs):,} pairs in {num_batches:,} batches (batch_size={batch_size})...")
    print(f"Will produce {total_examples:,} delta examples (2 per pair)")
    print(f"Minibatch size: {minibatch_size:,} examples")
    
    pbar = tqdm(total=len(pairs), desc="Extracting delta vectors")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(pairs))
        
        batch_pairs = pairs[start_idx:end_idx]
        batch_size_actual = len(batch_pairs)
        
        # Separate best and worst headlines
        best_headlines = [p[0] for p in batch_pairs]
        worst_headlines = [p[1] for p in batch_pairs]
        
        try:
            # Extract SAE vectors for best headlines
            best_vectors = extract_sae_vectors_batch(
                model, sae, tokenizer, target_layer, best_headlines, device, max_length
            )
            
            # Extract SAE vectors for worst headlines
            worst_vectors = extract_sae_vectors_batch(
                model, sae, tokenizer, target_layer, worst_headlines, device, max_length
            )
            
            # Compute delta vectors
            # best - worst with label 1
            delta_best_worst = best_vectors - worst_vectors
            # worst - best with label 0
            delta_worst_best = worst_vectors - best_vectors
            
            # Interleave: for each pair, add both deltas
            for i in range(batch_size_actual):
                current_minibatch_vectors.append(delta_best_worst[i:i+1])  # Keep as 2D
                current_minibatch_labels.append(1)
                current_minibatch_vectors.append(delta_worst_best[i:i+1])
                current_minibatch_labels.append(0)
            
            examples_processed += batch_size_actual * 2
            
        except Exception as e:
            print(f"\nError processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            # Use zero vectors as fallback for this batch
            for _ in range(batch_size_actual):
                fallback = np.zeros((1, d_sae))
                current_minibatch_vectors.append(fallback)
                current_minibatch_labels.append(1)
                current_minibatch_vectors.append(fallback)
                current_minibatch_labels.append(0)
            examples_processed += batch_size_actual * 2
        
        pbar.update(batch_size_actual)
        
        # Check if we've processed enough examples to save a minibatch
        if examples_processed >= minibatch_size:
            print(f"\nSaving minibatch {minibatch_idx} ({examples_processed:,} examples)...")
            sparse_path, best_path = save_minibatch(
                current_minibatch_vectors,
                current_minibatch_labels,
                output_dir,
                output_stem,
                minibatch_idx,
                d_sae
            )
            
            if sparse_path:
                print(f"  Saved: {sparse_path.name}, {best_path.name}")
            
            # Clear current minibatch from memory
            del current_minibatch_vectors, current_minibatch_labels
            current_minibatch_vectors = []
            current_minibatch_labels = []
            examples_processed = 0
            minibatch_idx += 1
            
            # Aggressive cleanup
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # Periodic cache clearing (every 10 batches)
        if (batch_idx + 1) % 10 == 0:
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    
    pbar.close()
    
    # Save any remaining vectors in the last minibatch
    if current_minibatch_vectors:
        print(f"\nSaving final minibatch {minibatch_idx} ({examples_processed:,} examples)...")
        sparse_path, best_path = save_minibatch(
            current_minibatch_vectors,
            current_minibatch_labels,
            output_dir,
            output_stem,
            minibatch_idx,
            d_sae
        )
        if sparse_path:
            print(f"  Saved: {sparse_path.name}, {best_path.name}")
        minibatch_idx += 1
        del current_minibatch_vectors, current_minibatch_labels
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()


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
            dtype=torch.bfloat16,
        )
    else:
        print("Loading model without quantization (using optimizations)...")
        # Use optimizations to save VRAM
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-4b-it",
            device_map='auto',
            dtype=torch.bfloat16,
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
    parser.add_argument('--minibatch-size', type=int, default=5000,
                       help='Number of sentences after which to save and clear memory (default: 5000)')
    
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
                max_length=args.max_length,
                minibatch_size=args.minibatch_size
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
