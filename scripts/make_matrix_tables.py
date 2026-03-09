import pandas as pd
import re

df = pd.read_csv('results/master_results.csv')
df['mode'] = df['mode'].replace({'zero_shot': 'Zero-Shot', 'fine_tuned': 'Fine-Tuned'})
df['quantization'] = df['quantization'].fillna('none')

def extract_dims(filename, default_dim):
    match = re.search(r'_(\d+)_(binary|fp32|int8)\.json', filename)
    if match: return int(match.group(1))
    return default_dim

df['dims'] = df.apply(lambda row: extract_dims(row['source_file'], row['dims']), axis=1)
df['dims'] = df['dims'].fillna(0).astype(int)

# Filter to just fine-tuned for the ablation matrices to keep it clean. Group by model, dims, quantization taking first item if duplicated.
ft_df = df[df['mode'] == 'Fine-Tuned'].copy()
ft_df = ft_df.groupby(['model', 'dims', 'quantization']).first().reset_index()

models_to_matrix = ['gte_modernbert_base', 'nomic_v15', 'bge_small', 'minilm_l6']

metrics = [
    ('overall_mrr_at_10', 'MRR@10'),
    ('latency_p50', 'Latency p50 (ms)'),
    ('index_size_mb', 'Index Size (MB)')
]

def format_val(val, metric):
    if pd.isna(val): return "-"
    if 'mrr' in metric: return f"{val:.3f}"
    if 'latency' in metric: return f"{val:.2f}"
    if 'size' in metric: return f"{val:.1f}"
    return str(val)

print("## Appendix: Complete Ablation Matrices\n")
print("To understand the direct tradeoff space across all limits, here are exact boundary grids measuring precision against dimensionality.\n")

for metric_col, metric_name in metrics:
    print(f"### {metric_name} grid\n")
    
    print("<div style=\"display: flex; justify-content: space-between;\">\n")
    
    for model in models_to_matrix:
        model_df = ft_df[ft_df['model'] == model]
        if model_df.empty: continue
        
        pivot = model_df.pivot(index='dims', columns='quantization', values=metric_col)
        pivot = pivot.sort_index(ascending=False)
        
        cols = ['fp32', 'int8', 'binary']
        cols = [c for c in cols if c in pivot.columns]
        
        print(f"<div>\n")
        print(f"**Model:** `{model}` ({metric_name})\n")
        
        header = "| Dims | " + " | ".join(cols) + " |"
        sep    = "|:---|" + "|".join([":---:"] * len(cols)) + "|"
        print(header)
        print(sep)
        
        for dims, row in pivot.iterrows():
            vals = [format_val(row.get(c), metric_col) for c in cols]
            print(f"| **{dims}** | " + " | ".join(vals) + " |")
        print("\n</div>\n")
        
    print("</div>\n<br>\n")

