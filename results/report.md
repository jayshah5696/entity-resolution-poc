# Experiment Results

Generated: 2026-03-09 03:40 UTC

Experiments loaded: 53

## 1. Summary

| Experiment | Model | Mode | Serialization | R@10 Overall | MRR@10 | Latency p50 ms |
|------------|-------|------|---------------|:------------:|:------:|:--------------:|
| 001 | bm25_baseline | zero_shot | pipe | 0.958 | 0.917 | 3.25 |
| 002 | gte_modernbert_base | zero_shot | pipe | 0.941 | 0.891 | 18.74 |
| bge_small_base | bge_small | zero_shot | pipe | 0.894 | 0.844 | 11.69 |
| bge_small_base_128_binary | bge_small | zero_shot | pipe | 0.889 | 0.844 | 10.14 |
| bge_small_base_128_fp32 | bge_small | zero_shot | pipe | 0.848 | 0.798 | 10.10 |
| bge_small_base_128_int8 | bge_small | zero_shot | pipe | 0.902 | 0.860 | 10.26 |
| bge_small_base_256_binary | bge_small | zero_shot | pipe | 0.907 | 0.861 | 10.21 |
| bge_small_base_256_fp32 | bge_small | zero_shot | pipe | 0.882 | 0.831 | 10.27 |
| bge_small_base_256_int8 | bge_small | zero_shot | pipe | 0.915 | 0.872 | 10.43 |
| bge_small_base_384_binary | bge_small | zero_shot | pipe | 0.912 | 0.868 | 10.37 |
| bge_small_base_384_fp32 | bge_small | zero_shot | pipe | 0.895 | 0.846 | 10.30 |
| bge_small_base_384_int8 | bge_small | zero_shot | pipe | 0.917 | 0.875 | 10.55 |
| bge_small_base_64_binary | bge_small | zero_shot | pipe | 0.841 | 0.797 | 16.44 |
| bge_small_base_64_fp32 | bge_small | zero_shot | pipe | 0.771 | 0.710 | 10.13 |
| bge_small_base_64_int8 | bge_small | zero_shot | pipe | 0.870 | 0.828 | 10.04 |
| bge_small_ft | bge_small | fine_tuned | pipe | 0.932 | 0.898 | 10.63 |
| bge_small_ft_128_fp32 | bge_small | fine_tuned | pipe | 0.921 | 0.885 | 17.86 |
| bge_small_ft_256_binary | bge_small | fine_tuned | pipe | 0.942 | 0.906 | 18.40 |
| bge_small_ft_256_fp32 | bge_small | fine_tuned | pipe | 0.930 | 0.896 | 17.81 |
| bge_small_ft_256_int8 | bge_small | fine_tuned | pipe | 0.942 | 0.907 | 17.46 |
| bge_small_ft_384_binary | bge_small | fine_tuned | pipe | 0.940 | 0.906 | 17.22 |
| bge_small_ft_384_fp32 | bge_small | fine_tuned | pipe | 0.932 | 0.898 | 17.91 |
| bge_small_ft_384_int8 | bge_small | fine_tuned | pipe | 0.941 | 0.906 | 17.53 |
| gte_modernbert_base_base | gte_modernbert_base | zero_shot | pipe | 0.941 | 0.891 | 24.19 |
| gte_modernbert_base_ft | gte_modernbert_base | fine_tuned | pipe | 0.966 | 0.917 | 24.12 |
| minilm_l6_base | minilm_l6 | zero_shot | pipe | 0.876 | 0.840 | 8.47 |
| minilm_l6_base_128_binary | minilm_l6 | zero_shot | pipe | 0.866 | 0.833 | 13.93 |
| minilm_l6_base_128_fp32 | minilm_l6 | zero_shot | pipe | 0.839 | 0.799 | 7.97 |
| minilm_l6_base_128_int8 | minilm_l6 | zero_shot | pipe | 0.877 | 0.844 | 13.32 |
| minilm_l6_base_256_binary | minilm_l6 | zero_shot | pipe | 0.882 | 0.847 | 7.35 |
| minilm_l6_base_256_fp32 | minilm_l6 | zero_shot | pipe | 0.867 | 0.830 | 7.53 |
| minilm_l6_base_256_int8 | minilm_l6 | zero_shot | pipe | 0.887 | 0.852 | 7.98 |
| minilm_l6_base_384_binary | minilm_l6 | zero_shot | pipe | 0.887 | 0.852 | 8.87 |
| minilm_l6_base_384_fp32 | minilm_l6 | zero_shot | pipe | 0.876 | 0.840 | 9.95 |
| minilm_l6_base_384_int8 | minilm_l6 | zero_shot | pipe | 0.891 | 0.856 | 8.45 |
| minilm_l6_base_64_binary | minilm_l6 | zero_shot | pipe | 0.839 | 0.802 | 13.66 |
| minilm_l6_base_64_fp32 | minilm_l6 | zero_shot | pipe | 0.780 | 0.713 | 12.96 |
| minilm_l6_base_64_int8 | minilm_l6 | zero_shot | pipe | 0.860 | 0.827 | 13.42 |
| minilm_l6_ft | minilm_l6 | fine_tuned | pipe | 0.928 | 0.895 | 8.06 |
| minilm_l6_ft_128_binary | minilm_l6 | fine_tuned | pipe | 0.931 | 0.898 | 6.57 |
| minilm_l6_ft_128_fp32 | minilm_l6 | fine_tuned | pipe | 0.916 | 0.881 | 6.51 |
| minilm_l6_ft_128_int8 | minilm_l6 | fine_tuned | pipe | 0.934 | 0.902 | 6.79 |
| minilm_l6_ft_256_binary | minilm_l6 | fine_tuned | pipe | 0.933 | 0.900 | 6.71 |
| minilm_l6_ft_256_fp32 | minilm_l6 | fine_tuned | pipe | 0.925 | 0.892 | 8.52 |
| minilm_l6_ft_256_int8 | minilm_l6 | fine_tuned | pipe | 0.933 | 0.901 | 6.91 |
| minilm_l6_ft_384_binary | minilm_l6 | fine_tuned | pipe | 0.934 | 0.902 | 13.78 |
| minilm_l6_ft_384_fp32 | minilm_l6 | fine_tuned | pipe | 0.928 | 0.895 | 13.81 |
| minilm_l6_ft_384_int8 | minilm_l6 | fine_tuned | pipe | 0.935 | 0.902 | 14.37 |
| minilm_l6_ft_64_binary | minilm_l6 | fine_tuned | pipe | 0.920 | 0.886 | 6.37 |
| minilm_l6_ft_64_fp32 | minilm_l6 | fine_tuned | pipe | 0.890 | 0.855 | 6.33 |
| minilm_l6_ft_64_int8 | minilm_l6 | fine_tuned | pipe | 0.929 | 0.897 | 6.58 |
| nomic_v15_base | nomic_v15 | zero_shot | pipe | 0.882 | 0.850 | 11.64 |
| nomic_v15_ft | nomic_v15 | fine_tuned | pipe | 0.817 | 0.795 | 11.86 |

## 2. Per-Bucket Recall@10

| Model | Serialization | pristine | missing_firstname | missing_email_company | typo_name | domain_mismatch | swapped_attributes |
|-------|---------------|:---: | :---: | :---: | :---: | :---: | :---:|
| bm25_baseline | pipe | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 |
| gte_modernbert_base | pipe | 1.000 | 0.919 | 0.747 | 0.994 | 0.990 | 0.999 |
| bge_small | pipe | 1.000 | 0.905 | 0.597 | 0.963 | 0.900 | 1.000 |
| bge_small | pipe | 1.000 | 0.912 | 0.543 | 0.958 | 0.923 | 1.000 |
| bge_small | pipe | 1.000 | 0.910 | 0.391 | 0.947 | 0.838 | 0.999 |
| bge_small | pipe | 1.000 | 0.912 | 0.603 | 0.959 | 0.938 | 1.000 |
| bge_small | pipe | 1.000 | 0.900 | 0.644 | 0.959 | 0.937 | 1.000 |
| bge_small | pipe | 1.000 | 0.903 | 0.552 | 0.960 | 0.877 | 1.000 |
| bge_small | pipe | 1.000 | 0.904 | 0.675 | 0.962 | 0.947 | 1.000 |
| bge_small | pipe | 1.000 | 0.896 | 0.677 | 0.959 | 0.942 | 1.000 |
| bge_small | pipe | 1.000 | 0.912 | 0.594 | 0.965 | 0.897 | 1.000 |
| bge_small | pipe | 1.000 | 0.901 | 0.694 | 0.963 | 0.947 | 1.000 |
| bge_small | pipe | 1.000 | 0.924 | 0.346 | 0.931 | 0.845 | 0.999 |
| bge_small | pipe | 1.000 | 0.897 | 0.191 | 0.833 | 0.707 | 0.999 |
| bge_small | pipe | 1.000 | 0.924 | 0.467 | 0.948 | 0.883 | 0.999 |
| bge_small | pipe | 1.000 | 0.985 | 0.623 | 0.984 | 1.000 | 1.000 |
| bge_small | pipe | 1.000 | 0.980 | 0.560 | 0.987 | 1.000 | 1.000 |
| bge_small | pipe | 1.000 | 0.982 | 0.687 | 0.984 | 1.000 | 1.000 |
| bge_small | pipe | 1.000 | 0.987 | 0.608 | 0.986 | 1.000 | 1.000 |
| bge_small | pipe | 1.000 | 0.982 | 0.686 | 0.984 | 1.000 | 1.000 |
| bge_small | pipe | 1.000 | 0.984 | 0.672 | 0.986 | 1.000 | 1.000 |
| bge_small | pipe | 1.000 | 0.982 | 0.626 | 0.982 | 1.000 | 1.000 |
| bge_small | pipe | 1.000 | 0.985 | 0.679 | 0.984 | 1.000 | 1.000 |
| gte_modernbert_base | pipe | 1.000 | 0.919 | 0.747 | 0.994 | 0.990 | 0.999 |
| gte_modernbert_base | pipe | 1.000 | 0.997 | 0.798 | 1.000 | 1.000 | 0.999 |
| minilm_l6 | pipe | 1.000 | 0.853 | 0.485 | 0.922 | 0.996 | 0.999 |
| minilm_l6 | pipe | 1.000 | 0.855 | 0.440 | 0.909 | 0.995 | 0.998 |
| minilm_l6 | pipe | 1.000 | 0.853 | 0.308 | 0.882 | 0.991 | 0.997 |
| minilm_l6 | pipe | 1.000 | 0.864 | 0.487 | 0.914 | 0.996 | 0.998 |
| minilm_l6 | pipe | 1.000 | 0.853 | 0.525 | 0.917 | 0.996 | 0.998 |
| minilm_l6 | pipe | 1.000 | 0.851 | 0.442 | 0.913 | 0.996 | 0.998 |
| minilm_l6 | pipe | 1.000 | 0.851 | 0.556 | 0.921 | 0.996 | 0.999 |
| minilm_l6 | pipe | 1.000 | 0.855 | 0.550 | 0.924 | 0.996 | 0.998 |
| minilm_l6 | pipe | 1.000 | 0.863 | 0.474 | 0.923 | 0.995 | 0.998 |
| minilm_l6 | pipe | 1.000 | 0.855 | 0.570 | 0.924 | 0.996 | 0.998 |
| minilm_l6 | pipe | 1.000 | 0.879 | 0.296 | 0.870 | 0.993 | 0.998 |
| minilm_l6 | pipe | 1.000 | 0.816 | 0.170 | 0.723 | 0.974 | 0.998 |
| minilm_l6 | pipe | 1.000 | 0.876 | 0.401 | 0.893 | 0.994 | 0.998 |
| minilm_l6 | pipe | 1.000 | 0.976 | 0.617 | 0.976 | 1.000 | 1.000 |
| minilm_l6 | pipe | 1.000 | 0.974 | 0.634 | 0.978 | 1.000 | 1.000 |
| minilm_l6 | pipe | 1.000 | 0.973 | 0.548 | 0.975 | 1.000 | 1.000 |
| minilm_l6 | pipe | 1.000 | 0.979 | 0.647 | 0.979 | 1.000 | 1.000 |
| minilm_l6 | pipe | 1.000 | 0.974 | 0.649 | 0.975 | 1.000 | 1.000 |
| minilm_l6 | pipe | 1.000 | 0.976 | 0.597 | 0.976 | 1.000 | 1.000 |
| minilm_l6 | pipe | 1.000 | 0.979 | 0.641 | 0.978 | 1.000 | 1.000 |
| minilm_l6 | pipe | 1.000 | 0.978 | 0.648 | 0.978 | 1.000 | 1.000 |
| minilm_l6 | pipe | 1.000 | 0.976 | 0.620 | 0.975 | 1.000 | 1.000 |
| minilm_l6 | pipe | 1.000 | 0.979 | 0.654 | 0.975 | 1.000 | 1.000 |
| minilm_l6 | pipe | 1.000 | 0.977 | 0.564 | 0.977 | 1.000 | 1.000 |
| minilm_l6 | pipe | 1.000 | 0.974 | 0.393 | 0.975 | 1.000 | 1.000 |
| minilm_l6 | pipe | 1.000 | 0.978 | 0.620 | 0.978 | 1.000 | 1.000 |
| nomic_v15 | pipe | 0.997 | 0.930 | 0.488 | 0.966 | 0.916 | 0.997 |
| nomic_v15 | pipe | 0.979 | 0.959 | 0.152 | 0.948 | 0.895 | 0.967 |

## 3. Delta vs BM25 Baseline (R@10 per bucket)

| Model | Serialization | pristine | missing_firstname | missing_email_company | typo_name | domain_mismatch | swapped_attributes | Overall |
|-------|---------------|:---: | :---: | :---: | :---: | :---: | :---:|:-------:|
| gte_modernbert_base | pipe | +0.0pp | -8.1pp | -0.4pp | -0.6pp | -1.0pp | -0.1pp | -1.7pp |
| bge_small | pipe | +0.0pp | -9.5pp | -15.3pp | -3.7pp | -10.0pp | +0.0pp | -6.4pp |
| bge_small | pipe | +0.0pp | -8.8pp | -20.8pp | -4.2pp | -7.7pp | -0.0pp | -6.9pp |
| bge_small | pipe | +0.0pp | -9.0pp | -36.0pp | -5.3pp | -16.2pp | -0.1pp | -11.1pp |
| bge_small | pipe | +0.0pp | -8.8pp | -14.7pp | -4.1pp | -6.2pp | -0.0pp | -5.6pp |
| bge_small | pipe | +0.0pp | -10.0pp | -10.6pp | -4.1pp | -6.3pp | -0.0pp | -5.1pp |
| bge_small | pipe | +0.0pp | -9.7pp | -19.8pp | -4.0pp | -12.3pp | -0.0pp | -7.7pp |
| bge_small | pipe | +0.0pp | -9.6pp | -7.6pp | -3.8pp | -5.3pp | +0.0pp | -4.4pp |
| bge_small | pipe | +0.0pp | -10.4pp | -7.3pp | -4.1pp | -5.8pp | +0.0pp | -4.6pp |
| bge_small | pipe | +0.0pp | -8.8pp | -15.6pp | -3.5pp | -10.3pp | +0.0pp | -6.4pp |
| bge_small | pipe | +0.0pp | -9.9pp | -5.6pp | -3.7pp | -5.3pp | +0.0pp | -4.1pp |
| bge_small | pipe | +0.0pp | -7.6pp | -40.5pp | -6.9pp | -15.5pp | -0.1pp | -11.8pp |
| bge_small | pipe | +0.0pp | -10.3pp | -55.9pp | -16.7pp | -29.3pp | -0.1pp | -18.7pp |
| bge_small | pipe | +0.0pp | -7.6pp | -28.4pp | -5.2pp | -11.7pp | -0.1pp | -8.8pp |
| bge_small | pipe | +0.0pp | -1.5pp | -12.8pp | -1.6pp | +0.0pp | -0.0pp | -2.6pp |
| bge_small | pipe | +0.0pp | -2.0pp | -19.0pp | -1.3pp | +0.0pp | +0.0pp | -3.7pp |
| bge_small | pipe | +0.0pp | -1.8pp | -6.4pp | -1.6pp | +0.0pp | +0.0pp | -1.6pp |
| bge_small | pipe | +0.0pp | -1.3pp | -14.2pp | -1.4pp | +0.0pp | +0.0pp | -2.8pp |
| bge_small | pipe | +0.0pp | -1.8pp | -6.4pp | -1.6pp | +0.0pp | +0.0pp | -1.6pp |
| bge_small | pipe | +0.0pp | -1.6pp | -7.8pp | -1.4pp | +0.0pp | +0.0pp | -1.8pp |
| bge_small | pipe | +0.0pp | -1.8pp | -12.4pp | -1.8pp | +0.0pp | +0.0pp | -2.7pp |
| bge_small | pipe | +0.0pp | -1.5pp | -7.2pp | -1.6pp | +0.0pp | +0.0pp | -1.7pp |
| gte_modernbert_base | pipe | +0.0pp | -8.1pp | -0.4pp | -0.6pp | -1.0pp | -0.1pp | -1.7pp |
| gte_modernbert_base | pipe | +0.0pp | -0.3pp | +4.7pp | -0.0pp | +0.0pp | -0.1pp | +0.7pp |
| minilm_l6 | pipe | +0.0pp | -14.7pp | -26.6pp | -7.8pp | -0.4pp | -0.1pp | -8.3pp |
| minilm_l6 | pipe | +0.0pp | -14.5pp | -31.0pp | -9.1pp | -0.5pp | -0.2pp | -9.2pp |
| minilm_l6 | pipe | +0.0pp | -14.7pp | -44.2pp | -11.8pp | -0.9pp | -0.3pp | -12.0pp |
| minilm_l6 | pipe | +0.0pp | -13.6pp | -26.4pp | -8.6pp | -0.4pp | -0.2pp | -8.2pp |
| minilm_l6 | pipe | +0.0pp | -14.7pp | -22.5pp | -8.3pp | -0.4pp | -0.2pp | -7.7pp |
| minilm_l6 | pipe | +0.0pp | -14.9pp | -30.9pp | -8.7pp | -0.4pp | -0.2pp | -9.2pp |
| minilm_l6 | pipe | +0.0pp | -14.9pp | -19.4pp | -7.9pp | -0.4pp | -0.1pp | -7.1pp |
| minilm_l6 | pipe | +0.0pp | -14.5pp | -20.0pp | -7.6pp | -0.4pp | -0.2pp | -7.1pp |
| minilm_l6 | pipe | +0.0pp | -13.7pp | -27.6pp | -7.7pp | -0.5pp | -0.2pp | -8.3pp |
| minilm_l6 | pipe | +0.0pp | -14.5pp | -18.1pp | -7.6pp | -0.4pp | -0.2pp | -6.8pp |
| minilm_l6 | pipe | +0.0pp | -12.1pp | -45.4pp | -13.0pp | -0.7pp | -0.2pp | -11.9pp |
| minilm_l6 | pipe | +0.0pp | -18.4pp | -58.1pp | -27.7pp | -2.6pp | -0.2pp | -17.9pp |
| minilm_l6 | pipe | +0.0pp | -12.4pp | -34.9pp | -10.7pp | -0.6pp | -0.2pp | -9.8pp |
| minilm_l6 | pipe | +0.0pp | -2.4pp | -13.4pp | -2.4pp | +0.0pp | +0.0pp | -3.0pp |
| minilm_l6 | pipe | +0.0pp | -2.6pp | -11.6pp | -2.2pp | +0.0pp | +0.0pp | -2.7pp |
| minilm_l6 | pipe | +0.0pp | -2.7pp | -20.3pp | -2.5pp | -0.0pp | +0.0pp | -4.2pp |
| minilm_l6 | pipe | +0.0pp | -2.1pp | -10.4pp | -2.1pp | -0.0pp | +0.0pp | -2.4pp |
| minilm_l6 | pipe | +0.0pp | -2.6pp | -10.2pp | -2.5pp | -0.0pp | +0.0pp | -2.6pp |
| minilm_l6 | pipe | +0.0pp | -2.4pp | -15.3pp | -2.4pp | -0.0pp | +0.0pp | -3.4pp |
| minilm_l6 | pipe | +0.0pp | -2.1pp | -11.0pp | -2.2pp | +0.0pp | +0.0pp | -2.6pp |
| minilm_l6 | pipe | +0.0pp | -2.2pp | -10.3pp | -2.2pp | -0.0pp | +0.0pp | -2.4pp |
| minilm_l6 | pipe | +0.0pp | -2.4pp | -13.0pp | -2.5pp | +0.0pp | +0.0pp | -3.0pp |
| minilm_l6 | pipe | +0.0pp | -2.1pp | -9.6pp | -2.5pp | +0.0pp | +0.0pp | -2.4pp |
| minilm_l6 | pipe | +0.0pp | -2.3pp | -18.7pp | -2.3pp | -0.0pp | +0.0pp | -3.9pp |
| minilm_l6 | pipe | +0.0pp | -2.6pp | -35.7pp | -2.5pp | -0.0pp | +0.0pp | -6.8pp |
| minilm_l6 | pipe | +0.0pp | -2.2pp | -13.0pp | -2.2pp | -0.0pp | +0.0pp | -2.9pp |
| nomic_v15 | pipe | -0.3pp | -7.0pp | -26.3pp | -3.4pp | -8.4pp | -0.3pp | -7.6pp |
| nomic_v15 | pipe | -2.1pp | -4.1pp | -59.9pp | -5.2pp | -10.5pp | -3.3pp | -14.2pp |

## 4. Latency

| Model | Mode | p50 ms | p95 ms | p99 ms | Index MB |
|-------|------|:------:|:------:|:------:|:--------:|
| bm25_baseline | zero_shot | 3.25 | 3.83 | 3.98 | 143.7 |
| gte_modernbert_base | zero_shot | 18.74 | 26.92 | 31.39 | 3105.3 |
| bge_small | zero_shot | 11.69 | 14.81 | 17.03 | 1616.8 |
| bge_small | zero_shot | 10.14 | 12.99 | 14.52 | 672.8 |
| bge_small | zero_shot | 10.10 | 12.72 | 14.25 | 665.2 |
| bge_small | zero_shot | 10.26 | 13.00 | 14.50 | 688.0 |
| bge_small | zero_shot | 10.21 | 13.03 | 14.55 | 1176.6 |
| bge_small | zero_shot | 10.27 | 12.87 | 14.16 | 1161.3 |
| bge_small | zero_shot | 10.43 | 13.19 | 14.76 | 1207.1 |
| bge_small | zero_shot | 10.37 | 13.08 | 14.43 | 1680.4 |
| bge_small | zero_shot | 10.30 | 12.97 | 14.45 | 1657.5 |
| bge_small | zero_shot | 10.55 | 13.51 | 15.36 | 1726.2 |
| bge_small | zero_shot | 16.44 | 18.37 | 19.43 | 420.9 |
| bge_small | zero_shot | 10.13 | 12.64 | 14.12 | 417.1 |
| bge_small | zero_shot | 10.04 | 12.75 | 14.20 | 428.5 |
| bge_small | fine_tuned | 10.63 | 13.93 | 15.94 | 1616.8 |
| bge_small | fine_tuned | 17.86 | 20.33 | 22.59 | 665.2 |
| bge_small | fine_tuned | 18.40 | 21.58 | 25.48 | 1176.6 |
| bge_small | fine_tuned | 17.81 | 20.12 | 21.51 | 1161.3 |
| bge_small | fine_tuned | 17.46 | 19.41 | 21.49 | 1207.1 |
| bge_small | fine_tuned | 17.22 | 19.95 | 22.19 | 1680.4 |
| bge_small | fine_tuned | 17.91 | 19.62 | 20.52 | 1657.5 |
| bge_small | fine_tuned | 17.53 | 19.09 | 20.12 | 1726.2 |
| gte_modernbert_base | zero_shot | 24.19 | 33.88 | 38.67 | 3105.3 |
| gte_modernbert_base | fine_tuned | 24.12 | 33.26 | 37.04 | 3105.3 |
| minilm_l6 | zero_shot | 8.47 | 9.97 | 12.52 | 1616.8 |
| minilm_l6 | zero_shot | 13.93 | 17.70 | 25.73 | 672.8 |
| minilm_l6 | zero_shot | 7.97 | 11.05 | 17.57 | 665.2 |
| minilm_l6 | zero_shot | 13.32 | 17.87 | 23.43 | 688.0 |
| minilm_l6 | zero_shot | 7.35 | 10.75 | 13.85 | 1176.6 |
| minilm_l6 | zero_shot | 7.53 | 10.84 | 14.72 | 1161.3 |
| minilm_l6 | zero_shot | 7.98 | 11.47 | 14.08 | 1207.1 |
| minilm_l6 | zero_shot | 8.87 | 12.46 | 19.18 | 1680.4 |
| minilm_l6 | zero_shot | 9.95 | 13.41 | 19.89 | 1657.5 |
| minilm_l6 | zero_shot | 8.45 | 11.10 | 13.06 | 1726.2 |
| minilm_l6 | zero_shot | 13.66 | 17.54 | 22.36 | 420.9 |
| minilm_l6 | zero_shot | 12.96 | 17.05 | 20.41 | 417.1 |
| minilm_l6 | zero_shot | 13.42 | 17.16 | 19.90 | 428.5 |
| minilm_l6 | fine_tuned | 8.06 | 11.07 | 11.70 | 1616.8 |
| minilm_l6 | fine_tuned | 6.57 | 9.98 | 10.58 | 672.8 |
| minilm_l6 | fine_tuned | 6.51 | 10.08 | 10.72 | 665.2 |
| minilm_l6 | fine_tuned | 6.79 | 10.25 | 10.90 | 688.0 |
| minilm_l6 | fine_tuned | 6.71 | 10.19 | 10.92 | 1176.6 |
| minilm_l6 | fine_tuned | 8.52 | 12.65 | 15.01 | 1161.3 |
| minilm_l6 | fine_tuned | 6.91 | 10.28 | 10.88 | 1207.1 |
| minilm_l6 | fine_tuned | 13.78 | 18.50 | 24.76 | 1680.4 |
| minilm_l6 | fine_tuned | 13.81 | 17.47 | 22.65 | 1657.5 |
| minilm_l6 | fine_tuned | 14.37 | 17.88 | 22.88 | 1726.2 |
| minilm_l6 | fine_tuned | 6.37 | 9.98 | 10.60 | 420.9 |
| minilm_l6 | fine_tuned | 6.33 | 9.90 | 10.44 | 417.1 |
| minilm_l6 | fine_tuned | 6.58 | 10.08 | 10.81 | 428.5 |
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
- `bge_small` (exp bge_small_ft_256_binary): -1.6pp overall R@10 vs BM25
- `bge_small` (exp bge_small_ft_256_int8): -1.6pp overall R@10 vs BM25
- `gte_modernbert_base` (exp 002): -1.7pp overall R@10 vs BM25
- `gte_modernbert_base` (exp gte_modernbert_base_base): -1.7pp overall R@10 vs BM25
