# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "pandas",
#     "seaborn",
# ]
# ///

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

df = pd.read_csv('results/master_results.csv')

# Clean up mode names for plots
df['mode'] = df['mode'].replace({'zero_shot': 'Zero-Shot', 'fine_tuned': 'Fine-Tuned'})
df['quantization'] = df['quantization'].fillna('fp32')

# Fix dims by extracting from source_file if available
def extract_dims(filename, default_dim):
    match = re.search(r'_(\d+)_(binary|fp32|int8)\.json', filename)
    if match:
        return int(match.group(1))
    return default_dim

df['dims'] = df.apply(lambda row: extract_dims(row['source_file'], row['dims']), axis=1)

os.makedirs('results/plots', exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. MRR@10 by Model and Mode (Base Dims)
plt.figure(figsize=(10, 6))
base_dims = df[((df['model'] == 'gte_modernbert_base') & (df['dims'] == 768)) |
               ((df['model'] == 'bge_small') & (df['dims'] == 384)) |
               ((df['model'] == 'minilm_l6') & (df['dims'] == 384)) |
               ((df['model'] == 'nomic_v15') & (df['dims'] == 768))]
sns.barplot(data=base_dims, x='model', y='overall_mrr_at_10', hue='mode', palette='muted')

bm25_row = df[df['model']=='bm25_baseline'].iloc[0]
bm25_mrr = bm25_row['overall_mrr_at_10']
plt.axhline(y=bm25_mrr, color='r', linestyle='--', linewidth=2, label=f'BM25 Baseline ({bm25_mrr:.3f})')

plt.title('Overall MRR@10: Zero-Shot vs Fine-Tuned (Uncompressed Index)', fontsize=14, fontweight='bold')
plt.ylabel('Mean Reciprocal Rank @ 10', fontsize=12)
plt.xlabel('Model Architecture', fontsize=12)
plt.legend(title='Training Mode')
plt.ylim(0.7, 1.0)
plt.tight_layout()
plt.savefig('results/plots/mrr_overall.png', dpi=300)
plt.close()

# 2. Compression Ablation - All 4 models
models = ['gte_modernbert_base', 'nomic_v15', 'bge_small', 'minilm_l6']
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
axes = axes.flatten()

for i, model in enumerate(models):
    ax = axes[i]
    model_ft = df[(df['model'] == model) & (df['mode'] == 'Fine-Tuned')].sort_values('dims')
    if model_ft.empty:
        model_ft = df[(df['model'] == model)].sort_values('dims')
        title_suffix = "(All data)"
    else:
        title_suffix = "(Fine-Tuned)"
        
    sns.lineplot(data=model_ft, x='dims', y='overall_mrr_at_10', hue='quantization', style='quantization',
                 markers=True, dashes=False, linewidth=2, ax=ax, palette='deep')
    
    ax.axhline(y=bm25_mrr, color='r', linestyle='--', alpha=0.4)
    ax.set_title(f'{model} {title_suffix}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dimensions')
    ax.set_ylabel('Overall MRR@10')

plt.suptitle('Compression Ablation across All Models (MRR@10)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('results/plots/all_models_ablation.png', dpi=300)
plt.close()

# 3. Latency vs MRR@10 Pareto Frontier - All models
plt.figure(figsize=(12, 7))
all_variants = df[df['model'].isin(models) & (df['model'] != 'bm25_baseline')]

sns.scatterplot(data=all_variants, x='latency_p50', y='overall_mrr_at_10', 
                hue='model', style='mode', size='dims', sizes=(20, 200), alpha=0.7, palette='viridis')

bm25_lat = bm25_row['latency_p50']
plt.axhline(y=bm25_mrr, color='red', linestyle='--', alpha=0.5, label='BM25 MRR')
plt.axvline(x=bm25_lat, color='red', linestyle=':', alpha=0.5, label='BM25 Latency')

plt.title('Pareto Frontier: Latency vs MRR@10 (All Models & Configurations)', fontsize=14, fontweight='bold')
plt.xlabel('Latency p50 (ms)', fontsize=12)
plt.ylabel('Overall MRR@10', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('results/plots/latency_pareto_all.png', dpi=300)
plt.close()

# 4. Bucket Heatmap - Best variants of each model
buckets = ['pristine', 'missing_firstname', 'missing_email_company', 'typo_name', 'domain_mismatch', 'swapped_attributes']
heatmap_data = []
labels = []

# BM25
heatmap_data.append([bm25_row[f"{b}_mrr_at_10"] for b in buckets])
labels.append('BM25 Baseline')

for model in models:
    best_run = df[df['model'] == model].sort_values('overall_mrr_at_10', ascending=False).iloc[0]
    heatmap_data.append([best_run[f"{b}_mrr_at_10"] for b in buckets])
    labels.append(f'{model} (Best)')

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", xticklabels=buckets, yticklabels=labels, vmin=0.3, vmax=1.0)
plt.title('MRR@10 by Category: Model Comparison (Best Configs)', fontsize=14, fontweight='bold')
plt.xticks(rotation=20, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig('results/plots/bucket_heatmap_all.png', dpi=300)
plt.close()

print("All-model enhanced plots generated.")
