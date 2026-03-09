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

bm25_mrr = df[df['model']=='bm25_baseline']['overall_mrr_at_10'].iloc[0]
plt.axhline(y=bm25_mrr, color='r', linestyle='--', linewidth=2, label=f'BM25 Baseline ({bm25_mrr:.3f})')

plt.title('Overall MRR@10: Zero-Shot vs Fine-Tuned (Uncompressed Index)', fontsize=14, fontweight='bold')
plt.ylabel('Mean Reciprocal Rank @ 10', fontsize=12)
plt.xlabel('Model Architecture', fontsize=12)
plt.legend(title='Training Mode')
plt.ylim(0.7, 1.0)
plt.tight_layout()
plt.savefig('results/plots/mrr_overall.png', dpi=300)
plt.close()

# 2. BGE Small - Dimensionality & Quantization Grid (MRR@10)
plt.figure(figsize=(9, 6))
bge_ft = df[(df['model'] == 'bge_small') & (df['mode'] == 'Fine-Tuned')].sort_values('dims')
sns.lineplot(data=bge_ft, x='dims', y='overall_mrr_at_10', hue='quantization', style='quantization',
             markers=True, dashes=False, linewidth=2.5, markersize=10, palette='deep')

plt.axhline(y=bm25_mrr, color='r', linestyle='--', alpha=0.6, label='BM25 Baseline')
plt.title('Compression Ablation: BGE-Small Fine-Tuned (MRR@10)', fontsize=14, fontweight='bold')
plt.xlabel('Matryoshka Dimensions', fontsize=12)
plt.ylabel('Overall MRR@10', fontsize=12)
plt.xticks([64, 128, 256, 384])
plt.legend(title='Quantization', loc='lower right')
plt.tight_layout()
plt.savefig('results/plots/bge_ablation.png', dpi=300)
plt.close()

# 3. Latency vs MRR@10 Pareto Frontier for fine-tuned compressed indexes
plt.figure(figsize=(10, 6))
ft_variants = df[(df['mode'] == 'Fine-Tuned') & (df['model'].isin(['bge_small', 'minilm_l6']))]

sns.scatterplot(data=ft_variants, x='latency_p50', y='overall_mrr_at_10', 
                hue='model', style='quantization', s=150, palette='colorblind')

# Annotate points with dimensions
for i in range(ft_variants.shape[0]):
    row = ft_variants.iloc[i]
    plt.text(row['latency_p50']+0.2, row['overall_mrr_at_10'] - 0.002, f"{int(row['dims'])}D", fontsize=9)

bm25_lat = df[df['model']=='bm25_baseline']['latency_p50'].iloc[0]
plt.axhline(y=bm25_mrr, color='red', linestyle='--', alpha=0.5, label='BM25 MRR')
plt.axvline(x=bm25_lat, color='red', linestyle=':', alpha=0.5, label='BM25 Latency')

plt.title('Pareto Frontier: Latency vs MRR@10 (Fine-Tuned Models)', fontsize=14, fontweight='bold')
plt.xlabel('Latency p50 (ms)', fontsize=12)
plt.ylabel('Overall MRR@10', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('results/plots/latency_pareto.png', dpi=300)
plt.close()

# 4. Bucket Heatmap
bm25 = df[df['model'] == 'bm25_baseline'].iloc[0]
gte_zs = df[(df['model'] == 'gte_modernbert_base') & (df['mode'] == 'Zero-Shot')].iloc[0]
gte_ft = df[(df['model'] == 'gte_modernbert_base') & (df['mode'] == 'Fine-Tuned')].iloc[0]
bge_ft_base = df[(df['model'] == 'bge_small') & (df['mode'] == 'Fine-Tuned') & (df['dims'] == 384) & (df['quantization'] == 'fp32')].iloc[0]

buckets = ['pristine', 'missing_firstname', 'missing_email_company', 'typo_name', 'domain_mismatch', 'swapped_attributes']
heatmap_data = []
labels = ['BM25 Baseline', 'GTE ModernBERT (Zero-Shot)', 'GTE ModernBERT (Fine-Tuned)', 'BGE Small (Fine-Tuned, 384D, fp32)']

for row in [bm25, gte_zs, gte_ft, bge_ft_base]:
    heatmap_data.append([row[f"{b}_mrr_at_10"] for b in buckets])

plt.figure(figsize=(11, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", xticklabels=buckets, yticklabels=labels, vmin=0.3, vmax=1.0)
plt.title('MRR@10 by Corruption Category', fontsize=14, fontweight='bold')
plt.xticks(rotation=20, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('results/plots/bucket_heatmap.png', dpi=300)
plt.close()

print("Enhanced plots generated.")
