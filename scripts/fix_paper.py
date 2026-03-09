import re

with open('paper.md', 'r') as f:
    content = f.read()

# Make sure image paths are relative to where the markdown is viewed
# Since images are in results/plots/, they should be referenced as such in paper.md if it's in the root
content = content.replace('![Overall MRR@10](plots/mrr_overall.png)', '![Overall MRR@10](results/plots/mrr_overall.png)')
content = content.replace('![Bucket Heatmap](plots/bucket_heatmap.png)', '![Bucket Heatmap](results/plots/bucket_heatmap.png)')
content = content.replace('![Dimensionality Ablation: BGE-Small](plots/bge_ablation.png)', '![Dimensionality Ablation: BGE-Small](results/plots/bge_ablation.png)')
content = content.replace('![Latency vs MRR Pareto Frontier](plots/latency_pareto.png)', '![Latency vs MRR Pareto Frontier](results/plots/latency_pareto.png)')

with open('paper.md', 'w') as f:
    f.write(content)
print("Fixed image paths in paper.md")
