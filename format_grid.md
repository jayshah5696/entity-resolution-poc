## Appendix: Comprehensive Ablation Grids
The following tables present the full ablation results for Matryoshka dimensions against quantization levels, grouping the key metrics side-by-side to allow for direct evaluation of compression tradeoffs.

### Model: `bge_small` (Fine-Tuned)
| Dimensions | FP32 (MRR | R@10 | Size) | INT8 (MRR | R@10 | Size) | Binary (MRR | R@10 | Size) |
|---|---|---|---|
| 384D | **0.898** \| 0.932 \| 1616.8MB | **0.906** \| 0.941 \| 1726.2MB | **0.906** \| 0.940 \| 1680.4MB |
| 256D | **0.896** \| 0.930 \| 1161.3MB | **0.907** \| 0.942 \| 1207.1MB | **0.906** \| 0.942 \| 1176.6MB |
| 128D | **0.885** \| 0.921 \| 665.2MB | - | - |


### Model: `minilm_l6` (Fine-Tuned)
| Dimensions | FP32 (MRR | R@10 | Size) | INT8 (MRR | R@10 | Size) | Binary (MRR | R@10 | Size) |
|---|---|---|---|
| 384D | **0.895** \| 0.928 \| 1616.8MB | **0.902** \| 0.935 \| 1726.2MB | **0.902** \| 0.934 \| 1680.4MB |
| 256D | **0.892** \| 0.925 \| 1161.3MB | **0.901** \| 0.933 \| 1207.1MB | **0.900** \| 0.933 \| 1176.6MB |
| 128D | **0.881** \| 0.916 \| 665.2MB | **0.902** \| 0.934 \| 688.0MB | **0.898** \| 0.931 \| 672.8MB |
| 64D | **0.855** \| 0.890 \| 417.1MB | **0.897** \| 0.929 \| 428.5MB | **0.886** \| 0.920 \| 420.9MB |


