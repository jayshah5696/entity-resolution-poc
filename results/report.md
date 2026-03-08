# Experiment Results

Generated: 2026-03-08 23:31 UTC

Experiments loaded: 10

## 1. Summary

| Experiment | Model | Mode | Serialization | R@10 Overall | MRR@10 | Latency p50 ms |
|------------|-------|------|---------------|:------------:|:------:|:--------------:|
| 001 | bm25_baseline | zero_shot | pipe | 0.958 | 0.917 | 3.25 |
| 002 | gte_modernbert_base | zero_shot | pipe | 0.941 | 0.891 | 18.74 |
| bge_small_base | bge_small | zero_shot | pipe | 0.894 | 0.844 | 11.69 |
| bge_small_ft | bge_small | fine_tuned | pipe | 0.932 | 0.898 | 10.63 |
| gte_modernbert_base_base | gte_modernbert_base | zero_shot | pipe | 0.941 | 0.891 | 24.19 |
| gte_modernbert_base_ft | gte_modernbert_base | fine_tuned | pipe | 0.966 | 0.917 | 24.12 |
| minilm_l6_base | minilm_l6 | zero_shot | pipe | 0.876 | 0.840 | 8.47 |
| minilm_l6_ft | minilm_l6 | fine_tuned | pipe | 0.928 | 0.895 | 8.06 |
| nomic_v15_base | nomic_v15 | zero_shot | pipe | 0.882 | 0.850 | 11.64 |
| nomic_v15_ft | nomic_v15 | fine_tuned | pipe | 0.817 | 0.795 | 11.86 |

## 2. Per-Bucket Recall@10

| Model | Serialization | pristine | missing_firstname | missing_email_company | typo_name | domain_mismatch | swapped_attributes |
|-------|---------------|:---: | :---: | :---: | :---: | :---: | :---:|
| bm25_baseline | pipe | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 |
| gte_modernbert_base | pipe | 1.000 | 0.919 | 0.747 | 0.994 | 0.990 | 0.999 |
| bge_small | pipe | 1.000 | 0.905 | 0.597 | 0.963 | 0.900 | 1.000 |
| bge_small | pipe | 1.000 | 0.985 | 0.623 | 0.984 | 1.000 | 1.000 |
| gte_modernbert_base | pipe | 1.000 | 0.919 | 0.747 | 0.994 | 0.990 | 0.999 |
| gte_modernbert_base | pipe | 1.000 | 0.997 | 0.798 | 1.000 | 1.000 | 0.999 |
| minilm_l6 | pipe | 1.000 | 0.853 | 0.485 | 0.922 | 0.996 | 0.999 |
| minilm_l6 | pipe | 1.000 | 0.976 | 0.617 | 0.976 | 1.000 | 1.000 |
| nomic_v15 | pipe | 0.997 | 0.930 | 0.488 | 0.966 | 0.916 | 0.997 |
| nomic_v15 | pipe | 0.979 | 0.959 | 0.152 | 0.948 | 0.895 | 0.967 |

## 3. Delta vs BM25 Baseline (R@10 per bucket)

| Model | Serialization | pristine | missing_firstname | missing_email_company | typo_name | domain_mismatch | swapped_attributes | Overall |
|-------|---------------|:---: | :---: | :---: | :---: | :---: | :---:|:-------:|
| gte_modernbert_base | pipe | +0.0pp | -8.1pp | -0.4pp | -0.6pp | -1.0pp | -0.1pp | -1.7pp |
| bge_small | pipe | +0.0pp | -9.5pp | -15.3pp | -3.7pp | -10.0pp | +0.0pp | -6.4pp |
| bge_small | pipe | +0.0pp | -1.5pp | -12.8pp | -1.6pp | +0.0pp | -0.0pp | -2.6pp |
| gte_modernbert_base | pipe | +0.0pp | -8.1pp | -0.4pp | -0.6pp | -1.0pp | -0.1pp | -1.7pp |
| gte_modernbert_base | pipe | +0.0pp | -0.3pp | +4.7pp | -0.0pp | +0.0pp | -0.1pp | +0.7pp |
| minilm_l6 | pipe | +0.0pp | -14.7pp | -26.6pp | -7.8pp | -0.4pp | -0.1pp | -8.3pp |
| minilm_l6 | pipe | +0.0pp | -2.4pp | -13.4pp | -2.4pp | +0.0pp | +0.0pp | -3.0pp |
| nomic_v15 | pipe | -0.3pp | -7.0pp | -26.3pp | -3.4pp | -8.4pp | -0.3pp | -7.6pp |
| nomic_v15 | pipe | -2.1pp | -4.1pp | -59.9pp | -5.2pp | -10.5pp | -3.3pp | -14.2pp |

## 4. Latency

| Model | Mode | p50 ms | p95 ms | p99 ms | Index MB |
|-------|------|:------:|:------:|:------:|:--------:|
| bm25_baseline | zero_shot | 3.25 | 3.83 | 3.98 | 143.7 |
| gte_modernbert_base | zero_shot | 18.74 | 26.92 | 31.39 | 3105.3 |
| bge_small | zero_shot | 11.69 | 14.81 | 17.03 | 1616.8 |
| bge_small | fine_tuned | 10.63 | 13.93 | 15.94 | 1616.8 |
| gte_modernbert_base | zero_shot | 24.19 | 33.88 | 38.67 | 3105.3 |
| gte_modernbert_base | fine_tuned | 24.12 | 33.26 | 37.04 | 3105.3 |
| minilm_l6 | zero_shot | 8.47 | 9.97 | 12.52 | 1616.8 |
| minilm_l6 | fine_tuned | 8.06 | 11.07 | 11.70 | 1616.8 |
| nomic_v15 | zero_shot | 11.64 | 14.09 | 16.18 | 3105.3 |
| nomic_v15 | fine_tuned | 11.86 | 14.89 | 17.05 | 3104.9 |

## 5. Key Findings

### Best Model per Bucket (by R@10)

- **pristine**: `bm25_baseline` (exp 001) with R@10=1.000
- **missing_firstname**: `bm25_baseline` (exp 001) with R@10=1.000
- **missing_email_company**: `gte_modernbert_base` (exp gte_modernbert_base_ft) with R@10=0.798
- **typo_name**: `bm25_baseline` (exp 001) with R@10=1.000
- **domain_mismatch**: `bm25_baseline` (exp 001) with R@10=1.000
- **swapped_attributes**: `bm25_baseline` (exp 001) with R@10=1.000

### Largest Gains over BM25 (Overall R@10)

- `gte_modernbert_base` (exp gte_modernbert_base_ft): +0.7pp overall R@10 vs BM25
- `gte_modernbert_base` (exp 002): -1.7pp overall R@10 vs BM25
- `gte_modernbert_base` (exp gte_modernbert_base_base): -1.7pp overall R@10 vs BM25
- `bge_small` (exp bge_small_ft): -2.6pp overall R@10 vs BM25
- `minilm_l6` (exp minilm_l6_ft): -3.0pp overall R@10 vs BM25
