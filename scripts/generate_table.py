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

df = df.sort_values(by=['model', 'mode', 'dims', 'quantization'], ascending=[True, False, False, True])

cols = ['model', 'mode', 'dims', 'quantization', 'index_size_mb', 'latency_p50', 'overall_recall_at_10', 'overall_mrr_at_10']
df_sub = df[cols].copy()

df_sub['overall_recall_at_10'] = df_sub['overall_recall_at_10'].apply(lambda x: f"{x:.3f}")
df_sub['overall_mrr_at_10'] = df_sub['overall_mrr_at_10'].apply(lambda x: f"{x:.3f}")
df_sub['index_size_mb'] = df_sub['index_size_mb'].apply(lambda x: f"{x:.1f}" if x > 0 else "-")
df_sub['latency_p50'] = df_sub['latency_p50'].apply(lambda x: f"{x:.2f}")
df_sub['dims'] = df_sub['dims'].apply(lambda x: str(x) if x > 0 else "-")

print(df_sub.to_markdown(index=False))
