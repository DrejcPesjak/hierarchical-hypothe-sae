#!/usr/bin/env python3
"""
Extract sparse SAE context+diff vectors with weights from preprocessed CSV files.
For each test_id, samples random headline pairs, passes through LLM,
masks instruction tokens, encodes through SAE, max-pools per sentence,
then creates context vectors (mean of pair) and diff vectors (first - second),
along with better-label and three weight variants (CTR diff, logit diff, beta-prior).

Optimized for batch processing.

-- Example usage --
# Auto batch size (recommended)
python extract_sae_vectors_meandiffweight.py

# Manual batch size
python extract_sae_vectors_meandiffweight.py --batch-size 16

# If VRAM is tight, keep SAE on CPU
python extract_sae_vectors_meandiffweight.py --sae-on-cpu --batch-size 8

# With shorter sequences (faster, less memory)
python extract_sae_vectors_meandiffweight.py --max-length 256 --batch-size 16

# All pairs (no sampling limit)
python extract_sae_vectors_meandiffweight.py --max-pairs 0

# Our current best:
python extract_sae_vectors_meandiffweight.py --sae-on-cpu --batch-size 5 --max-length 256 --minibatch-size 5000
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
from itertools import combinations


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


def sample_pairs_within_test_id(df, max_pairs=10, seed=42):
    """
    Sample random headline pairs within each test_id.
    
    Args:
        df: DataFrame with columns: clickability_test_id, headline, ctr, clicks, impressions
        max_pairs: Maximum pairs per test_id (None or 0 = all pairs)
        seed: Random seed for reproducibility
    
    Returns:
        List of dicts with keys: test_id, id_first, id_second, headline_first, headline_second,
        ctr_first, ctr_second, clicks_first, clicks_second, impressions_first, impressions_second
    """
    rng = np.random.RandomState(seed)
    all_pairs = []
    
    for test_id, group in df.groupby('clickability_test_id'):
        group = group.reset_index(drop=True)  # local indices 0, 1, 2, ...
        n = len(group)
        if n < 2:
            continue
        
        # All C(N, 2) unique pairs
        pair_indices = list(combinations(range(n), 2))
        
        # Filter out pairs where CTR is identical or either headline is empty/NaN
        valid_pairs = []
        for i, j in pair_indices:
            if group.loc[i, 'ctr'] == group.loc[j, 'ctr']:
                continue
            h_i, h_j = group.loc[i, 'headline'], group.loc[j, 'headline']
            if pd.isna(h_i) or pd.isna(h_j):
                continue
            if str(h_i).strip() == '' or str(h_j).strip() == '':
                continue
            valid_pairs.append((i, j))
        
        if not valid_pairs:
            continue
        
        # Sample up to max_pairs (None or 0 means take all)
        if max_pairs and len(valid_pairs) > max_pairs:
            indices = rng.choice(len(valid_pairs), size=max_pairs, replace=False)
            valid_pairs = [valid_pairs[idx] for idx in sorted(indices)]
        
        # Randomize order within each pair
        for i, j in valid_pairs:
            if rng.random() < 0.5:
                first, second = i, j
            else:
                first, second = j, i
            
            all_pairs.append({
                'test_id': test_id,
                'id_first': first,
                'id_second': second,
                'headline_first': str(group.loc[first, 'headline']),
                'headline_second': str(group.loc[second, 'headline']),
                'ctr_first': float(group.loc[first, 'ctr']),
                'ctr_second': float(group.loc[second, 'ctr']),
                'clicks_first': int(group.loc[first, 'clicks']),
                'clicks_second': int(group.loc[second, 'clicks']),
                'impressions_first': int(group.loc[first, 'impressions']),
                'impressions_second': int(group.loc[second, 'impressions']),
            })
    
    return all_pairs


def compute_weights(ctr_a, ctr_b, clicks_a, clicks_b, imp_a, imp_b, eps=1e-6):
    """
    Compute three weight variants for a headline pair.
    
    Args:
        ctr_a, ctr_b: Click-through rates for first and second headline
        clicks_a, clicks_b: Raw click counts
        imp_a, imp_b: Raw impression counts
        eps: Epsilon for logit clipping
    
    Returns:
        (weight_diff, weight_logit, weight_beta)
    """
    # Weight 1: absolute CTR difference
    weight_diff = abs(ctr_a - ctr_b)
    
    # Weight 2: absolute logit difference with epsilon clamp
    p_a = np.clip(ctr_a, eps, 1 - eps)
    p_b = np.clip(ctr_b, eps, 1 - eps)
    logit_a = np.log(p_a / (1 - p_a))
    logit_b = np.log(p_b / (1 - p_b))
    weight_logit = abs(logit_a - logit_b)
    
    # Weight 3: Jeffreys Beta(0.5, 0.5) posterior mean + sqrt(min impressions)
    alpha = 0.5
    beta = 0.5
    p_a_beta = (clicks_a + alpha) / (imp_a + alpha + beta)
    p_b_beta = (clicks_b + alpha) / (imp_b + alpha + beta)
    p_a_beta = np.clip(p_a_beta, eps, 1 - eps)
    p_b_beta = np.clip(p_b_beta, eps, 1 - eps)
    logit_a_beta = np.log(p_a_beta / (1 - p_a_beta))
    logit_b_beta = np.log(p_b_beta / (1 - p_b_beta))
    weight_beta = abs(logit_a_beta - logit_b_beta) * np.sqrt(min(imp_a, imp_b))
    
    return weight_diff, weight_logit, weight_beta


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


def save_minibatch(vectors_list, meta_dict, output_dir, output_stem, minibatch_idx, d_sae):
    """
    Save a minibatch of context+diff vectors and metadata to disk.
    
    Args:
        vectors_list: List of numpy arrays, each [1, 2*d_sae] (context|diff concatenated)
        meta_dict: Dict of lists with keys: test_id, id_first, id_second, ctr_first, ctr_second,
                   better_label, weight_diff, weight_logit, weight_beta
        output_dir: Output directory
        output_stem: Base name for output files
        minibatch_idx: Index of this minibatch
        d_sae: SAE dimension (for validation)
    
    Returns:
        Paths to saved files (sparse_path, meta_path)
    """
    if not vectors_list:
        return None, None
    
    # Concatenate vectors and convert to sparse
    sae_array = np.vstack(vectors_list)
    sparse_matrix = csr_matrix(sae_array)
    
    # Save sparse vectors
    sparse_path = output_dir / f"{output_stem}_sparse_minibatch_{minibatch_idx:03d}.npz"
    np.savez_compressed(sparse_path,
                       data=sparse_matrix.data,
                       indices=sparse_matrix.indices,
                       indptr=sparse_matrix.indptr,
                       shape=sparse_matrix.shape)
    
    # Save metadata
    meta_path = output_dir / f"{output_stem}_meta_minibatch_{minibatch_idx:03d}.npz"
    np.savez(meta_path,
             test_id=np.array(meta_dict['test_id'], dtype=object),
             id_first=np.array(meta_dict['id_first'], dtype=np.int32),
             id_second=np.array(meta_dict['id_second'], dtype=np.int32),
             ctr_first=np.array(meta_dict['ctr_first'], dtype=np.float64),
             ctr_second=np.array(meta_dict['ctr_second'], dtype=np.float64),
             better_label=np.array(meta_dict['better_label'], dtype=np.int32),
             weight_diff=np.array(meta_dict['weight_diff'], dtype=np.float64),
             weight_logit=np.array(meta_dict['weight_logit'], dtype=np.float64),
             weight_beta=np.array(meta_dict['weight_beta'], dtype=np.float64))
    
    return sparse_path, meta_path


def process_csv_file(csv_path, output_path, model, sae, tokenizer, target_layer, device, 
                     batch_size=8, max_length=512, minibatch_size=5000, max_pairs=10, seed=42):
    """
    Process a CSV file and extract SAE context+diff vectors for sampled headline pairs.
    For each test_id, samples random pairs, computes context (mean) and diff vectors,
    plus better-labels and three weight variants.
    
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
        minibatch_size: Number of examples after which to save and clear memory
        max_pairs: Maximum pairs to sample per test_id (None or 0 = all pairs)
        seed: Random seed for reproducibility
    """
    print(f"\nProcessing: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df):,}")
    
    # Sample pairs within test_ids
    pairs = sample_pairs_within_test_id(df, max_pairs=max_pairs, seed=seed)
    print(f"Sampled {len(pairs):,} headline pairs")
    
    # Get SAE dimension for zero vector fallback
    d_sae = sae.w_enc.shape[1]
    
    # Setup output paths
    output_dir = Path(output_path).parent
    output_stem = Path(output_path).stem
    
    # Process in batches with minibatch saving
    current_minibatch_vectors = []   # Accumulated [context|diff] vectors
    current_minibatch_meta = {       # Accumulated metadata
        'test_id': [], 'id_first': [], 'id_second': [],
        'ctr_first': [], 'ctr_second': [], 'better_label': [],
        'weight_diff': [], 'weight_logit': [], 'weight_beta': []
    }
    examples_in_minibatch = 0
    minibatch_idx = 0
    
    num_batches = (len(pairs) + batch_size - 1) // batch_size
    
    print(f"Processing {len(pairs):,} pairs in {num_batches:,} batches (batch_size={batch_size})...")
    print(f"Will produce {len(pairs):,} examples (1 per pair)")
    print(f"Minibatch size: {minibatch_size:,} examples")
    
    pbar = tqdm(total=len(pairs), desc="Extracting context+diff vectors")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(pairs))
        
        batch_pairs = pairs[start_idx:end_idx]
        batch_size_actual = len(batch_pairs)
        
        # Separate first and second headlines
        first_headlines = [p['headline_first'] for p in batch_pairs]
        second_headlines = [p['headline_second'] for p in batch_pairs]
        
        try:
            # Extract SAE vectors for first headlines
            first_vectors = extract_sae_vectors_batch(
                model, sae, tokenizer, target_layer, first_headlines, device, max_length
            )
            
            # Extract SAE vectors for second headlines
            second_vectors = extract_sae_vectors_batch(
                model, sae, tokenizer, target_layer, second_headlines, device, max_length
            )
            
            # Compute context (mean) and diff vectors
            context_vectors = (first_vectors + second_vectors) / 2.0
            diff_vectors = first_vectors - second_vectors
            
            # Concatenate [context | diff] per pair -> [batch, 2*d_sae]
            concat_vectors = np.hstack([context_vectors, diff_vectors])
            
            # Compute labels and weights for each pair
            for i in range(batch_size_actual):
                p = batch_pairs[i]
                
                better_label = 1 if p['ctr_first'] > p['ctr_second'] else 0
                w_diff, w_logit, w_beta = compute_weights(
                    p['ctr_first'], p['ctr_second'],
                    p['clicks_first'], p['clicks_second'],
                    p['impressions_first'], p['impressions_second']
                )
                
                current_minibatch_vectors.append(concat_vectors[i:i+1])
                current_minibatch_meta['test_id'].append(p['test_id'])
                current_minibatch_meta['id_first'].append(p['id_first'])
                current_minibatch_meta['id_second'].append(p['id_second'])
                current_minibatch_meta['ctr_first'].append(p['ctr_first'])
                current_minibatch_meta['ctr_second'].append(p['ctr_second'])
                current_minibatch_meta['better_label'].append(better_label)
                current_minibatch_meta['weight_diff'].append(w_diff)
                current_minibatch_meta['weight_logit'].append(w_logit)
                current_minibatch_meta['weight_beta'].append(w_beta)
            
            examples_in_minibatch += batch_size_actual
            
        except Exception as e:
            print(f"\nError processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            # Use zero vectors as fallback for this batch
            for i in range(batch_size_actual):
                p = batch_pairs[i]
                fallback = np.zeros((1, 2 * d_sae))
                
                better_label = 1 if p['ctr_first'] > p['ctr_second'] else 0
                w_diff, w_logit, w_beta = compute_weights(
                    p['ctr_first'], p['ctr_second'],
                    p['clicks_first'], p['clicks_second'],
                    p['impressions_first'], p['impressions_second']
                )
                
                current_minibatch_vectors.append(fallback)
                current_minibatch_meta['test_id'].append(p['test_id'])
                current_minibatch_meta['id_first'].append(p['id_first'])
                current_minibatch_meta['id_second'].append(p['id_second'])
                current_minibatch_meta['ctr_first'].append(p['ctr_first'])
                current_minibatch_meta['ctr_second'].append(p['ctr_second'])
                current_minibatch_meta['better_label'].append(better_label)
                current_minibatch_meta['weight_diff'].append(w_diff)
                current_minibatch_meta['weight_logit'].append(w_logit)
                current_minibatch_meta['weight_beta'].append(w_beta)
            examples_in_minibatch += batch_size_actual
        
        pbar.update(batch_size_actual)
        
        # Check if we should save a minibatch
        if examples_in_minibatch >= minibatch_size:
            print(f"\nSaving minibatch {minibatch_idx} ({examples_in_minibatch:,} examples)...")
            sparse_path, meta_path = save_minibatch(
                current_minibatch_vectors,
                current_minibatch_meta,
                output_dir,
                output_stem,
                minibatch_idx,
                d_sae
            )
            
            if sparse_path:
                print(f"  Saved: {sparse_path.name}, {meta_path.name}")
            
            # Clear current minibatch from memory
            del current_minibatch_vectors, current_minibatch_meta
            current_minibatch_vectors = []
            current_minibatch_meta = {
                'test_id': [], 'id_first': [], 'id_second': [],
                'ctr_first': [], 'ctr_second': [], 'better_label': [],
                'weight_diff': [], 'weight_logit': [], 'weight_beta': []
            }
            examples_in_minibatch = 0
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
        print(f"\nSaving final minibatch {minibatch_idx} ({examples_in_minibatch:,} examples)...")
        sparse_path, meta_path = save_minibatch(
            current_minibatch_vectors,
            current_minibatch_meta,
            output_dir,
            output_stem,
            minibatch_idx,
            d_sae
        )
        if sparse_path:
            print(f"  Saved: {sparse_path.name}, {meta_path.name}")
        minibatch_idx += 1
        del current_minibatch_vectors, current_minibatch_meta
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
                       help='Number of examples after which to save and clear memory (default: 5000)')
    parser.add_argument('--max-pairs', type=int, default=10,
                       help='Maximum pairs to sample per test_id (0 = all pairs, default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for pair sampling (default: 42)')
    
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
        
        # Convert max_pairs: 0 means None (all pairs)
        max_pairs = args.max_pairs if args.max_pairs > 0 else None
        
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
                minibatch_size=args.minibatch_size,
                max_pairs=max_pairs,
                seed=args.seed
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
