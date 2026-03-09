import pandas as pd
import re

df = pd.read_csv('results/master_results.csv')

# Clean up
df['mode'] = df['mode'].replace({'zero_shot': 'Zero-Shot', 'fine_tuned': 'Fine-Tuned'})
df['quantization'] = df['quantization'].fillna('none')

def extract_dims(filename, default_dim):
    match = re.search(r'_(\d+)_(binary|fp32|int8)\.json', filename)
    if match:
        return int(match.group(1))
    return default_dim

df['dims'] = df.apply(lambda row: extract_dims(row['source_file'], row['dims']), axis=1)
df['dims'] = df['dims'].fillna(0).astype(int)

# Filter to fine-tuned grid models
df_ft = df[(df['mode'] == 'Fine-Tuned') & (df['model'].isin(['bge_small', 'minilm_l6']))]

metrics = [
    ('overall_mrr_at_10', 'Overall MRR@10', '{:.3f}'),
    ('overall_recall_at_10', 'Overall Recall@10', '{:.3f}'),
    ('overall_recall_at_1', 'Overall Recall@1', '{:.3f}'),
    ('index_size_mb', 'Index Size (MB)', '{:.1f}'),
    ('latency_p50', 'Latency p50 (ms)', '{:.2f}')
]

models = [
    ('bge_small', 'BGE-Small (Fine-Tuned)'),
    ('minilm_l6', 'MiniLM-L6 (Fine-Tuned)')
]

quants = ['fp32', 'int8', 'binary']
dims_list = [384, 256, 128, 64]

markdown_output = []
markdown_output.append("## Appendix: Comprehensive Ablation Grids\n")
markdown_output.append("The following tables present the full ablation results for Matryoshka dimensions against quantization levels, grouped by metric. This grid format allows for direct evaluation of compression tradeoffs.\n")

for metric_col, metric_name, fmt in metrics:
    markdown_output.append(f"### {metric_name}\n")
    
    # We will put the tables side-by-side using HTML/Markdown trickery or just sequential
    # Sequential is safer for markdown rendering
    
    for model_id, model_name in models:
        markdown_output.append(f"**{model_name}**\n")
        markdown_output.append("| Dimensions | FP32 | INT8 | Binary |")
        markdown_output.append("|------------|------|------|--------|")
        
        for d in dims_list:
            row_vals = []
            for q in quants:
                val = df_ft[(df_ft['model'] == model_id) & (df_ft['dims'] == d) & (df_ft['quantization'] == q)]
                if len(val) > 0:
                    v = val.iloc[0][metric_col]
                    row_vals.append(fmt.format(v))
                else:
                    row_vals.append("-")
            
            markdown_output.append(f"| {d}D | {row_vals[0]} | {row_vals[1]} | {row_vals[2]} |")
        markdown_output.append("\n")

print('\n'.join(markdown_output))
