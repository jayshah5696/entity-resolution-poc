import os

images = [
    'results/plots/mrr_overall.png',
    'results/plots/bucket_heatmap.png', 
    'results/plots/bge_ablation.png',
    'results/plots/latency_pareto.png'
]

for img in images:
    if not os.path.exists(img):
        print(f"Warning: Image {img} not found.")
    else:
        print(f"Image {img} found!")
