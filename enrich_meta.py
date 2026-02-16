# %% import libraries
import numpy as np
import pandas as pd
from pathlib import Path

# %% configuration
EPS = 1e-6
ALPHA = 0.5   # Jeffreys prior: Beta(0.5, 0.5)
BETA = 0.5

batch_dir = Path("./data/sae_vectors")
csv_path = Path("./data/cleaned_archive/confirmatory_preprocessed.csv")

# %% load CSV and build lookup
print("Loading CSV...")
df = pd.read_csv(csv_path)
print(f"CSV rows: {len(df):,}")

# Build lookup: (test_id, local_idx) -> {clicks, impressions, ctr}
# Replicates the groupby + reset_index(drop=True) from extraction script
print("Building lookup table...")
lookup = {}
for test_id, group in df.groupby('clickability_test_id'):
    group = group.reset_index(drop=True)
    for local_idx in range(len(group)):
        row = group.loc[local_idx]
        lookup[(test_id, local_idx)] = {
            'clicks': int(row['clicks']),
            'impressions': int(row['impressions']),
            'ctr': float(row['ctr']),
        }

print(f"Lookup entries: {len(lookup):,}")

# %% find meta files
meta_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_meta_minibatch_*.npz"))
print(f"\nFound {len(meta_files)} meta files to enrich")

# %% process each meta batch
for meta_path in meta_files:
    print(f"\nProcessing: {meta_path.name}")
    m = np.load(meta_path, allow_pickle=True)
    
    test_ids = m['test_id']
    id_first = m['id_first']
    id_second = m['id_second']
    n = len(test_ids)
    
    # Allocate new arrays
    clicks_first = np.zeros(n, dtype=np.int32)
    clicks_second = np.zeros(n, dtype=np.int32)
    impressions_first = np.zeros(n, dtype=np.int32)
    impressions_second = np.zeros(n, dtype=np.int32)
    target = np.zeros(n, dtype=np.float64)
    weight = np.zeros(n, dtype=np.float64)
    
    n_miss = 0
    for i in range(n):
        tid = str(test_ids[i])
        idf = int(id_first[i])
        ids = int(id_second[i])
        
        info_a = lookup.get((tid, idf))
        info_b = lookup.get((tid, ids))
        
        if info_a is None or info_b is None:
            n_miss += 1
            continue
        
        c_a, imp_a = info_a['clicks'], info_a['impressions']
        c_b, imp_b = info_b['clicks'], info_b['impressions']
        
        clicks_first[i] = c_a
        clicks_second[i] = c_b
        impressions_first[i] = imp_a
        impressions_second[i] = imp_b
        
        # # Jeffreys Beta posterior mean
        # p_a_beta = (c_a + ALPHA) / (imp_a + ALPHA + BETA)
        # p_b_beta = (c_b + ALPHA) / (imp_b + ALPHA + BETA)
        # p_a_beta = np.clip(p_a_beta, EPS, 1 - EPS)
        # p_b_beta = np.clip(p_b_beta, EPS, 1 - EPS)
        
        # Basic clipped logits
        p_a = np.clip(info_a['ctr'], EPS, 1 - EPS)
        p_b = np.clip(info_b['ctr'], EPS, 1 - EPS)
        
        logit_a = np.log(p_a / (1 - p_a))
        logit_b = np.log(p_b / (1 - p_b))
        
        # target = signed logit diff (first - second), matches diff vector direction
        target[i] = logit_a - logit_b
        
        # weight = sqrt(min impressions)
        weight[i] = np.sqrt(min(imp_a, imp_b))
    
    if n_miss > 0:
        print(f"  WARNING: {n_miss} rows could not be looked up")
    
    # Save enriched meta
    out_name = meta_path.name.replace("_meta_minibatch_", "_1meta_minibatch_")
    out_path = batch_dir / out_name
    
    # Copy all original fields + add new ones
    np.savez(out_path,
             # original fields
             test_id=test_ids,
             id_first=id_first,
             id_second=id_second,
             ctr_first=m['ctr_first'],
             ctr_second=m['ctr_second'],
             better_label=m['better_label'],
             weight_diff=m['weight_diff'],
             weight_logit=m['weight_logit'],
             weight_beta=m['weight_beta'],
             # new fields
             clicks_first=clicks_first,
             clicks_second=clicks_second,
             impressions_first=impressions_first,
             impressions_second=impressions_second,
             target_logit_diff=target,
             weight_sqrt_min_impr=weight,
    )
    
    print(f"  Saved: {out_name}")
    print(f"  target range: [{target.min():.4f}, {target.max():.4f}], "
          f"mean: {target.mean():.4f}, std: {target.std():.4f}")
    print(f"  weight range: [{weight.min():.2f}, {weight.max():.2f}], "
          f"mean: {weight.mean():.2f}")

# %% verify
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)
out_files = sorted(batch_dir.glob("confirmatory_preprocessed_sae_vectors_1meta_minibatch_*.npz"))
print(f"Output files: {len(out_files)}")

m = np.load(out_files[0], allow_pickle=True)
print(f"Keys: {list(m.keys())}")
print(f"\nFirst 5 rows:")
for i in range(5):
    print(f"  tid={m['test_id'][i]}, "
          f"first={m['id_first'][i]}, second={m['id_second'][i]}, "
          f"clicks=({m['clicks_first'][i]},{m['clicks_second'][i]}), "
          f"impr=({m['impressions_first'][i]},{m['impressions_second'][i]}), "
          f"target={m['target_logit_diff'][i]:.4f}, "
          f"weight={m['weight_sqrt_min_impr'][i]:.2f}, "
          f"better={m['better_label'][i]}")

