## Appendix: Comprehensive Ablation Grids

The following tables present the full ablation results for Matryoshka dimensions against quantization levels, grouped by metric. This grid format allows for direct evaluation of compression tradeoffs.

### Overall MRR@10

**BGE-Small (Fine-Tuned)**

| Dimensions | FP32 | INT8 | Binary |
|------------|------|------|--------|
| 384D | 0.898 | 0.906 | 0.906 |
| 256D | 0.896 | 0.907 | 0.906 |
| 128D | 0.885 | - | - |
| 64D | - | - | - |


**MiniLM-L6 (Fine-Tuned)**

| Dimensions | FP32 | INT8 | Binary |
|------------|------|------|--------|
| 384D | 0.895 | 0.902 | 0.902 |
| 256D | 0.892 | 0.901 | 0.900 |
| 128D | 0.881 | 0.902 | 0.898 |
| 64D | 0.855 | 0.897 | 0.886 |


### Overall Recall@10

**BGE-Small (Fine-Tuned)**

| Dimensions | FP32 | INT8 | Binary |
|------------|------|------|--------|
| 384D | 0.932 | 0.941 | 0.940 |
| 256D | 0.930 | 0.942 | 0.942 |
| 128D | 0.921 | - | - |
| 64D | - | - | - |


**MiniLM-L6 (Fine-Tuned)**

| Dimensions | FP32 | INT8 | Binary |
|------------|------|------|--------|
| 384D | 0.928 | 0.935 | 0.934 |
| 256D | 0.925 | 0.933 | 0.933 |
| 128D | 0.916 | 0.934 | 0.931 |
| 64D | 0.890 | 0.929 | 0.920 |


### Overall Recall@1

**BGE-Small (Fine-Tuned)**

| Dimensions | FP32 | INT8 | Binary |
|------------|------|------|--------|
| 384D | 0.882 | 0.889 | 0.890 |
| 256D | 0.880 | 0.890 | 0.889 |
| 128D | 0.869 | - | - |
| 64D | - | - | - |


**MiniLM-L6 (Fine-Tuned)**

| Dimensions | FP32 | INT8 | Binary |
|------------|------|------|--------|
| 384D | 0.879 | 0.886 | 0.886 |
| 256D | 0.876 | 0.886 | 0.884 |
| 128D | 0.866 | 0.886 | 0.882 |
| 64D | 0.838 | 0.882 | 0.871 |


### Index Size (MB)

**BGE-Small (Fine-Tuned)**

| Dimensions | FP32 | INT8 | Binary |
|------------|------|------|--------|
| 384D | 1616.8 | 1726.2 | 1680.4 |
| 256D | 1161.3 | 1207.1 | 1176.6 |
| 128D | 665.2 | - | - |
| 64D | - | - | - |


**MiniLM-L6 (Fine-Tuned)**

| Dimensions | FP32 | INT8 | Binary |
|------------|------|------|--------|
| 384D | 1616.8 | 1726.2 | 1680.4 |
| 256D | 1161.3 | 1207.1 | 1176.6 |
| 128D | 665.2 | 688.0 | 672.8 |
| 64D | 417.1 | 428.5 | 420.9 |


### Latency p50 (ms)

**BGE-Small (Fine-Tuned)**

| Dimensions | FP32 | INT8 | Binary |
|------------|------|------|--------|
| 384D | 10.63 | 17.53 | 17.22 |
| 256D | 17.81 | 17.46 | 18.40 |
| 128D | 17.86 | - | - |
| 64D | - | - | - |


**MiniLM-L6 (Fine-Tuned)**

| Dimensions | FP32 | INT8 | Binary |
|------------|------|------|--------|
| 384D | 8.06 | 14.37 | 13.78 |
| 256D | 8.52 | 6.91 | 6.71 |
| 128D | 6.51 | 6.79 | 6.57 |
| 64D | 6.33 | 6.58 | 6.37 |


