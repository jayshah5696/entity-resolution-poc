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

# Sort
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

models = ['gte_modernbert_base', 'nomic_v15', 'bge_small', 'minilm_l6']
for model in models:
    for mode in ['Fine-Tuned', 'Zero-Shot']:
        sub = pivot_mrr[(pivot_mrr['model'] == model) & (pivot_mrr['mode'] == mode)]
        if sub.empty:
            continue
            
        out.append(f"### Model: `{model}` ({mode})")
        out.append("| Dimensions | FP32 (MRR | R@10 | Size) | INT8 (MRR | R@10 | Size) | Binary (MRR | R@10 | Size) |")
        out.append("|---|---|---|---|")
        
        for _, row in sub.iterrows():
            d = row['dims']
            
            # Use filters to get corresponding R@10 and Size safely
            r10_sub = pivot_r10[(pivot_r10['model']==model) & (pivot_r10['mode']==mode) & (pivot_r10['dims']==d)]
            sz_sub = pivot_size[(pivot_size['model']==model) & (pivot_size['mode']==mode) & (pivot_size['dims']==d)]
            
            if r10_sub.empty or sz_sub.empty:
                continue
                
            r10_row = r10_sub.iloc[0]
            sz_row = sz_sub.iloc[0]
            
            quants = ['fp32', 'int8', 'binary']
            cells = []
            for q in quants:
                mrr = row.get(q)
                r10 = r10_row.get(q)
                sz = sz_row.get(q)
                
                if not pd.isna(mrr):
                    cells.append(f"**{fmt_val(mrr)}** \\| {fmt_val(r10)} \\| {fmt_size(sz)}")
                else:
                    cells.append("-")
            
            out.append(f"| {d}D | {' | '.join(cells)} |")
        out.append("\n")

print('\n'.join(out))
