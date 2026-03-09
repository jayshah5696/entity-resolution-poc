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

# Create pivot tables
pivot_mrr = df_emb.pivot_table(index=['model', 'mode', 'dims'], columns='quantization', values='overall_mrr_at_10', aggfunc='first').reset_index()
pivot_r10 = df_emb.pivot_table(index=['model', 'mode', 'dims'], columns='quantization', values='overall_recall_at_10', aggfunc='first').reset_index()
pivot_size = df_emb.pivot_table(index=['model', 'mode', 'dims'], columns='quantization', values='index_size_mb', aggfunc='first').reset_index()

# Sort descending by dims
pivot_mrr = pivot_mrr.sort_values(by=['model', 'mode', 'dims'], ascending=[True, False, False])

def fmt_val(val):
    if pd.isna(val): return "-"
    return f"{val:.3f}"

def fmt_size(val):
    if pd.isna(val): return "-"
    return f"{val:.1f}MB"

out = []
out.append("## Appendix: Comprehensive Ablation Grids")
out.append("The following tables present the full ablation results for Matryoshka dimensions against quantization levels, grouping the key metrics side-by-side to allow for direct evaluation of compression tradeoffs.")
out.append("")

models = ['bge_small', 'minilm_l6']
for model in models:
    out.append(f"### Model: `{model}` (Fine-Tuned)")
    out.append("| Dimensions | FP32 (MRR | R@10 | Size) | INT8 (MRR | R@10 | Size) | Binary (MRR | R@10 | Size) |")
    out.append("|---|---|---|---|")
    
    sub = pivot_mrr[(pivot_mrr['model'] == model) & (pivot_mrr['mode'] == 'Fine-Tuned')]
    for _, row in sub.iterrows():
        d = row['dims']
        
        r10_row = pivot_r10[(pivot_r10['model']==model) & (pivot_r10['mode']=='Fine-Tuned') & (pivot_r10['dims']==d)].iloc[0]
        sz_row = pivot_size[(pivot_size['model']==model) & (pivot_size['mode']=='Fine-Tuned') & (pivot_size['dims']==d)].iloc[0]
        
        fp32_str = f"**{fmt_val(row.get('fp32'))}** \\| {fmt_val(r10_row.get('fp32'))} \\| {fmt_size(sz_row.get('fp32'))}" if not pd.isna(row.get('fp32')) else "-"
        int8_str = f"**{fmt_val(row.get('int8'))}** \\| {fmt_val(r10_row.get('int8'))} \\| {fmt_size(sz_row.get('int8'))}" if not pd.isna(row.get('int8')) else "-"
        bin_str = f"**{fmt_val(row.get('binary'))}** \\| {fmt_val(r10_row.get('binary'))} \\| {fmt_size(sz_row.get('binary'))}" if not pd.isna(row.get('binary')) else "-"
        
        out.append(f"| {d}D | {fp32_str} | {int8_str} | {bin_str} |")
    
    out.append("\n")

print('\n'.join(out))
