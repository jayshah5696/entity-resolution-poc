import pandas as pd
import re

df = pd.read_csv('results/master_results.csv')

df['mode'] = df['mode'].replace({'zero_shot': 'Zero-Shot', 'fine_tuned': 'Fine-Tuned'})
df['quantization'] = df['quantization'].fillna('none')

def extract_dims(filename, default_dim):
    match = re.search(r'_(\d+)_(binary|fp32|int8)\.json', filename)
    if match:
        return int(match.group(1))
    return default_dim

df['dims'] = df.apply(lambda row: extract_dims(row['source_file'], row['dims']), axis=1)
df['dims'] = df['dims'].fillna(0).astype(int)

df_emb = df[df['model'] != 'bm25_baseline']

max_mrr = df_emb['overall_mrr_at_10'].max()

pivot_mrr = df_emb.pivot_table(index=['model', 'mode', 'dims'], columns='quantization', values='overall_mrr_at_10', aggfunc='first').reset_index()
pivot_size = df_emb.pivot_table(index=['model', 'mode', 'dims'], columns='quantization', values='index_size_mb', aggfunc='first').reset_index()

pivot_mrr = pivot_mrr.sort_values(by=['model', 'mode', 'dims'], ascending=[True, False, False])

def fmt_mrr(val):
    if pd.isna(val): return "-"
    s = f"{val:.3f}"
    if val >= max_mrr:
        return f"**{s}** 🏆"
    elif val > 0.900:
        return f"**{s}**"
    return s

def fmt_size(val):
    if pd.isna(val): return "-"
    if val < 500:
        return f"*{val:.1f}MB* ⚡"
    return f"{val:.1f}MB"

print("| Model | Mode | Dims | FP32 (MRR \\| Size) | INT8 (MRR \\| Size) | Binary (MRR \\| Size) |")
print("|-------|------|------|--------------------|--------------------|----------------------|")

for _, row in pivot_mrr.iterrows():
    model = row['model']
    mode = row['mode']
    dims = row['dims']
    
    sz_row = pivot_size[(pivot_size['model']==model) & (pivot_size['mode']==mode) & (pivot_size['dims']==dims)].iloc[0]
    
    fp32_str = f"{fmt_mrr(row.get('fp32'))} \\| {fmt_size(sz_row.get('fp32'))}" if not pd.isna(row.get('fp32')) else "-"
    int8_str = f"{fmt_mrr(row.get('int8'))} \\| {fmt_size(sz_row.get('int8'))}" if not pd.isna(row.get('int8')) else "-"
    bin_str = f"{fmt_mrr(row.get('binary'))} \\| {fmt_size(sz_row.get('binary'))}" if not pd.isna(row.get('binary')) else "-"
    
    print(f"| `{model}` | {mode} | {dims} | {fp32_str} | {int8_str} | {bin_str} |")

