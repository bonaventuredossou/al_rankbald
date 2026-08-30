# RankBALD: Ranking-Aligned Active Evaluation for Language Models

**Bonaventure F. P. Dossou, Jackie Chi Kit Cheung**

COLM 2026

## Abstract

Language model evaluation faces a structural challenge: as performance gaps between state-of-the-art models narrow, static evaluation sets require increasingly large budgets to yield stable, reliable model comparisons. Adaptive evaluation can improve measurement efficiency by sequentially selecting informative test items, but existing approaches primarily focus on estimating latent parameters that describe model capabilities. This focus overlooks the fact that experimenters may have diverse downstream evaluation goals, such as to produce a relative ordering of models. We introduce a framework for adaptive evaluation in the Item Response Theory (IRT) setting that is compatible with both capability assessment and model-ranking objectives. Within this framework, ranking reliability depends on resolving uncertainty near the boundaries of pairwise orderings rather than minimizing overall posterior variance. To operationalize this insight, we draw on Bayesian active learning (BAL), a family of methods that select examples by maximizing expected information gain under a posterior belief. BAL was originally developed for information-efficient parameter identification. We adapt BAL to the IRT evaluation setting, giving rise to what we call active evaluation: sequential item selection driven by principled uncertainty reduction over model abilities and their relative ordering. We develop a family of acquisition criteria that increasingly align with the ranking objective, culminating in RankBALD, which directly maximizes the mutual information between candidate test items and the pairwise model-ordering variables. Experiments across six English benchmarks from the Open LLM Leaderboard and five African-language benchmarks show that, compared with ability-estimation baselines, ranking-aligned acquisition consistently improves ranking validity and reduces evaluation variance. These results demonstrate that aligning item selection with the downstream evaluation objective substantially improves the efficiency and reliability of active evaluation.

---

## Repository Structure

```
AL_RANKBALD/
├── code/
│   ├── al_for_eval_irt_new.py   # Main runner: Open LLM Leaderboard experiments
│   └── run_afribench.py         # AfroBench runner: African-language experiments
├── data/
│   └── afribench/
│       ├── benchmarks/          # Raw item-level response data (source)
│       │   ├── afrimgsm.csv      # columns: item_id, model_1, model_2, ...
│       │   ├── afrimmlu.csv
│       │   ├── afrixnli.csv
│       │   ├── belebele.csv
│       │   └── sib.csv
│       ├── irt_models/          # Pre-fitted 2PL IRT parameters (derived from benchmarks/)
│       │   ├── afrimgsm_theta.json
│       │   ├── afrimgsm.csv
│       │   ├── afrimmlu_theta.json
│       │   ├── afrimmlu.csv
│       │   ├── afrixnli_theta.json
│       │   ├── afrixnli.csv
│       │   ├── belebele_theta.json
│       │   ├── belebele.csv
│       │   ├── sib_theta.json
│       │   └── sib.csv
│       └── lm_eval_results/     # Per-family splits derived from benchmarks/
│           ├── gemini_gemma.csv  # gemma-3-27b-it, gemini-2.0-flash, gemini-3-pro-preview
│           ├── gpt.csv           # gpt-5-2025-08-07
│           └── qwen.csv          # Qwen3.5-27B
└── outputs/
    ├── afribench_table2_final.json                    # AfroBench main results (Table 4)
    ├── afribench_table3_fixed100_by_benchmark.json    # AfroBench per-benchmark breakdown
    ├── afribench_table4_fixed100_by_family.json       # AfroBench per-model-family breakdown
    ├── custom_code_results_setup1.json                # Open LLM Leaderboard main results (Table 1)
    ├── custom_code_results_table3_fixed100_by_benchmark_setup1.json  # Per-benchmark breakdown (Table 2)
    └── custom_code_results_table4_fixed100_by_lm_setup1.json        # Per-LM breakdown (Table 3)
```

---

## Installation

```bash
git clone https://github.com/bonaventuredossou/rankbald
cd rankbald
pip install -r requirements.txt
```

---

## Reproducing Paper Results

### Open LLM Leaderboard Experiments (Tables 1, 2, 3)

Data is automatically downloaded from the AllenAI fluid-benchmarking HuggingFace
repository (`allenai/fluid-benchmarking`) on first run. No manual download required.

```bash
python code/al_for_eval_irt_new.py \
    --methods random_ability,fluid_benchmarking,bald,bald_weighted,var_bald,rank_bald \
    --eval_sizes 10,50,100,500 \
    --out_prefix outputs/custom_code_results_setup1 \
    --seed 0 \
    --device cuda
```

### AfroBench Experiments (Table 4)

IRT parameters and LM evaluation results for AfroBench are included in `data/afribench/`
and were obtained directly from the AfroBench authors (Ojo et al., 2025).

```bash
python code/run_afribench.py \
    --methods random_ability,fluid_benchmarking,bald,bald_weighted,var_bald,rank_bald \
    --eval_sizes 10,50,100,500 \
    --out_prefix outputs/afribench \
    --seed 0
```

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--methods` | all methods | Comma-separated list of acquisition strategies |
| `--eval_sizes` | `10,50,100,500` | Evaluation budgets to sweep |
| `--lms` | all LMs | Comma-separated list of language models |
| `--benchmarks` | all benchmarks | Comma-separated list of benchmarks |
| `--out_prefix` | `outputs/output` | Output filename prefix |
| `--seed` | `0` | Random seed |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |
| `--alpha` | `1.0` | Variance reduction weight for VarBALD |
| `--debug` | `False` | Enable verbose debug output |

---

## Acquisition Methods

| Method key | Paper name | Description |
|---|---|---|
| `random_ability` | Random-IRT | Random item selection with IRT-based ability estimation |
| `fluid_benchmarking` | Fluid | Maximum Fisher Information (Hofmann et al., 2025) |
| `bald` | BALD | BALD adapted to the IRT evaluation setting |
| `bald_weighted` | DiffBALD | Difficulty-aware extension of BALD |
| `var_bald` | VarBALD | Variance-aware extension of DiffBALD |
| `rank_bald` | RankBALD | Pairwise ordering uncertainty maximization (primary contribution) |

---

## Evaluation Metrics

| Metric | Direction | Description |
|---|---|---|
| Validity | lower is better | Average rank distance across capability-matched benchmark pairs |
| Variance | lower is better | Normalized total variation of ability trajectories over checkpoints |
| Saturation | higher is better | Spearman rank correlation between checkpoint order and ability estimates |

---

## Results

### Table 1: Performance across acquisition strategies as the number of items per benchmark increases

**Validity (rank distance, lower is better)**

| Method | 10 | 50 | 100 | 500 |
|---|---|---|---|---|
| Random-IRT | 59.45 | 51.52 | 45.67 | 36.09 |
| Fluid Benchmarking | 46.89 | 36.91 | 36.25 | 34.92 |
| BALD | 49.36 | 38.25 | 36.55 | 34.94 |
| VarBALD | 47.56 | 37.22 | 36.43 | 34.77 |
| DiffBALD | 48.88 | 37.51 | 36.37 | 34.83 |
| **RankBALD** | **35.30** | **29.10** | **24.23** | **22.45** |

**Variance (total variation, lower is better)**

| Method | 10 | 50 | 100 | 500 |
|---|---|---|---|---|
| Random-IRT | 19.38 | 16.00 | 13.77 | 12.50 |
| Fluid Benchmarking | 14.30 | 6.95 | 6.06 | 4.94 |
| BALD | 23.64 | 9.66 | 8.09 | 6.25 |
| VarBALD | 14.96 | 8.48 | 6.82 | 5.02 |
| DiffBALD | 15.25 | 8.34 | 6.88 | 5.03 |
| **RankBALD** | **12.42** | **6.09** | **5.81** | **3.98** |

**Saturation (rank correlation, higher is better)**

| Method | 10 | 50 | 100 | 500 |
|---|---|---|---|---|
| Random-IRT | 0.428 | 0.642 | 0.702 | 0.836 |
| Fluid Benchmarking | 0.708 | 0.831 | 0.850 | 0.875 |
| BALD | 0.538 | 0.779 | 0.818 | 0.861 |
| VarBALD | 0.650 | 0.811 | 0.837 | 0.872 |
| DiffBALD | 0.657 | 0.809 | 0.841 | 0.873 |
| **RankBALD** | **0.728** | **0.849** | **0.861** | **0.877** |

---

### Table 2: Benchmark-specific evaluation at 100 items

**Validity (rank distance, lower is better)**

| Method | ARC | GSM | HS | MMLU | WG |
|---|---|---|---|---|---|
| Random-IRT | 25.48 | 36.55 | 15.61 | 41.39 | 26.35 |
| Fluid | 12.44 | 19.31 | 6.05 | 42.38 | 14.64 |
| BALD | 14.55 | 24.17 | 6.60 | 45.80 | 21.92 |
| VarBALD | 12.97 | 21.11 | 6.38 | 42.87 | 17.27 |
| DiffBALD | 12.92 | 21.07 | 6.36 | 42.83 | 16.66 |
| **RankBALD** | **12.39** | **19.10** | **5.53** | **38.77** | **12.68** |

**Variance (total variation, lower is better)**

| Method | ARC | GSM | HS | MMLU | WG |
|---|---|---|---|---|---|
| Random-IRT | 7.92 | 17.50 | 6.48 | 22.00 | 10.44 |
| Fluid | 3.27 | 9.09 | 2.04 | 6.32 | 5.81 |
| BALD | 3.58 | 11.85 | 2.26 | 7.98 | 9.90 |
| VarBALD | 3.49 | 10.03 | 2.16 | 6.50 | 7.85 |
| DiffBALD | 3.45 | 10.38 | 2.15 | 6.69 | 7.86 |
| **RankBALD** | **3.07** | **8.82** | **2.01** | **5.55** | **4.33** |

**Saturation (rank correlation, higher is better)**

| Method | ARC | GSM | HS | MMLU | WG |
|---|---|---|---|---|---|
| Random-IRT | 0.821 | 0.713 | 0.819 | 0.555 | 0.780 |
| Fluid | 0.951 | 0.864 | 0.985 | 0.668 | 0.925 |
| BALD | 0.933 | 0.834 | 0.979 | 0.616 | 0.863 |
| VarBALD | 0.941 | 0.853 | 0.982 | 0.651 | 0.909 |
| DiffBALD | 0.944 | 0.855 | 0.983 | 0.651 | 0.918 |
| **RankBALD** | **0.958** | **0.877** | **0.990** | **0.718** | **0.934** |

Benchmarks: ARC = ARC Challenge, GSM = GSM8K, HS = HellaSwag, WG = WinoGrande. TruthfulQA excluded following the Fluid Benchmarking protocol.

---

### Table 3: Language-model-specific evaluation at 100 items

Model abbreviations: A7B = Amber-7B, K2 = K2-65B, O1 = OLMo1-7B, O2 = OLMo2-7B, P3 = Pythia-2.8B, P7 = Pythia-6.9B.

**Validity (rank distance, lower is better)**

| Method | A7B | K2 | O1 | O2 | P3 | P7 |
|---|---|---|---|---|---|---|
| Random-IRT | 39.74 | 31.77 | 55.58 | 44.48 | 51.95 | 51.10 |
| Fluid | 24.75 | 28.65 | 41.61 | 41.86 | 35.22 | 36.13 |
| BALD | 27.34 | 28.53 | 49.47 | 50.30 | 36.63 | 39.72 |
| VarBALD | 26.69 | 27.58 | 42.83 | 41.92 | 35.17 | 38.31 |
| DiffBALD | 26.88 | 27.85 | 43.01 | 41.27 | 35.90 | 37.75 |
| **RankBALD** | **23.28** | 28.38 | **40.48** | **38.67** | **31.58** | **35.09** |

**Variance (total variation, lower is better)**

| Method | A7B | K2 | O1 | O2 | P3 | P7 |
|---|---|---|---|---|---|---|
| Random-IRT | 11.47 | 15.29 | 9.37 | 13.50 | 19.99 | 12.98 |
| Fluid | 5.54 | 7.14 | 5.83 | 6.83 | 6.55 | 4.46 |
| BALD | 5.77 | 13.20 | 6.43 | 9.25 | 7.33 | 6.57 |
| VarBALD | 5.47 | 9.41 | 6.05 | 8.09 | 7.06 | 4.86 |
| DiffBALD | 5.45 | 9.36 | 6.17 | 8.08 | 7.22 | 4.99 |
| **RankBALD** | **4.91** | **6.03** | **5.08** | **5.43** | **4.79** | **4.03** |

**Saturation (rank correlation, higher is better)**

| Method | A7B | K2 | O1 | O2 | P3 | P7 |
|---|---|---|---|---|---|---|
| Random-IRT | 0.672 | 0.660 | 0.775 | 0.630 | 0.713 | 0.759 |
| Fluid | 0.819 | 0.892 | 0.905 | 0.802 | 0.813 | 0.868 |
| BALD | 0.788 | 0.838 | 0.886 | 0.734 | 0.811 | 0.853 |
| VarBALD | 0.802 | 0.864 | 0.895 | 0.786 | 0.813 | 0.863 |
| DiffBALD | 0.810 | 0.871 | 0.898 | 0.790 | 0.814 | 0.865 |
| **RankBALD** | **0.822** | **0.897** | **0.908** | **0.808** | **0.839** | **0.876** |

---

### Table 4: AfroBench evaluation across three experimental axes

**Validity (rank distance, lower is better)**

| Method | 10 | 50 | 100 | 500 | GSM | MMLU | XNLI | Bele | SIB | Gem/Gemma | GPT | Qwen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random-IRT | 0.900 | 0.800 | 0.600 | 0.800 | 0.400 | 0.800 | 0.000 | 0.400 | 0.400 | 0.800 | 0.400 | 0.000 |
| Fluid | 0.900 | 0.800 | 0.800 | 0.700 | 0.200 | 0.200 | 0.400 | 0.200 | 0.200 | 0.667 | 0.000 | 0.000 |
| BALD | 0.700 | 0.600 | 1.000 | 1.200 | 0.800 | 0.600 | 0.200 | 1.200 | 0.200 | 1.467 | 0.000 | 0.400 |
| VarBALD | 0.800 | 0.800 | 0.800 | 0.800 | 0.000 | 0.000 | 0.000 | 0.000 | 0.400 | 0.133 | 0.400 | 0.000 |
| DiffBALD | 0.600 | 0.800 | 0.800 | 0.800 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.400 | 0.400 | 0.000 |
| **RankBALD** | 0.667 | **0.333** | **0.333** | **0.333** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.133** | -- | -- |

**Variance (total variation, lower is better)**

| Method | 10 | 50 | 100 | 500 | GSM | MMLU | XNLI | Bele | SIB | Gem/Gemma | GPT | Qwen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random-IRT | 5.076 | 5.076 | 5.076 | 5.076 | 1.500 | 1.500 | 1.500 | 2.814 | 1.500 | 1.763 | -- | -- |
| Fluid | 2.024 | 2.024 | 2.024 | 2.024 | 1.500 | 1.500 | 1.500 | 5.441 | 1.500 | 2.288 | -- | -- |
| BALD | 1.875 | 1.875 | 1.875 | 1.875 | 1.553 | 1.500 | 1.500 | 1.500 | 1.500 | 1.511 | -- | -- |
| VarBALD | 5.768 | 5.768 | 5.768 | 5.768 | 1.500 | 1.500 | 1.500 | 4.831 | 1.500 | 2.166 | -- | -- |
| DiffBALD | 8.028 | 8.028 | 8.028 | 8.028 | 1.500 | 1.500 | 1.500 | 4.652 | 1.500 | 2.130 | -- | -- |
| **RankBALD** | **1.455** | **1.455** | **1.455** | **1.455** | 1.500 | 1.500 | 1.500 | **1.500** | 1.500 | **1.456** | -- | -- |

**Saturation (rank correlation, higher is better)**

| Method | 10 | 50 | 100 | 500 | GSM | MMLU | XNLI | Bele | SIB | Gem/Gemma | GPT | Qwen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random-IRT | 0.703 | 0.703 | 0.703 | 0.703 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.900 | -- | -- |
| Fluid | 0.677 | 0.677 | 0.677 | 0.677 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.900 | -- | -- |
| BALD | 0.637 | 0.637 | 0.637 | 0.637 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | -- | -- |
| VarBALD | 0.450 | 0.450 | 0.450 | 0.450 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.900 | -- | -- |
| DiffBALD | 0.597 | 0.597 | 0.597 | 0.597 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.900 | -- | -- |
| **RankBALD** | **0.740** | **0.740** | **0.740** | **0.740** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 0.900 | -- | -- |

Model families: Gemini/Gemma = gemma-3-27b-it -> gemini-2.0-flash -> gemini-3-pro-preview (S=3). GPT = gpt-5-2025-08-07 (S=1). Qwen = Qwen3.5-27B (S=1). "--" denotes undefined values for singleton families where variance and saturation require at least two checkpoints. Benchmarks: GSM = AfriGSM, MMLU = AfriMMLU, XNLI = AfriXNLI, Bele = Belebele, SIB = SIB-200.

---

## Statistical Significance

We report standard errors and paired t-test results over 6 LM-level observations at budget=100.

**Validity (rank distance, lower is better)**

| Method | Mean +/- SE | 95% CI | vs RankBALD |
|---|---|---|---|
| Random-IRT | 45.67 +/- 3.64 | [38.5, 52.8] | p=0.005 |
| Fluid | 36.25 +/- 2.81 | [30.7, 41.8] | p=0.021 |
| BALD | 36.55 +/- 4.03 | [28.6, 44.4] | p=0.017 |
| VarBALD | 36.43 +/- 2.85 | [30.9, 42.0] | p=0.015 |
| DiffBALD | 36.37 +/- 2.76 | [31.0, 41.8] | p=0.013 |
| RankBALD | 24.23 +/- 2.65 | [19.0, 29.5] | -- |

**Variance (total variation, lower is better)**

| Method | Mean +/- SE | 95% CI | vs RankBALD |
|---|---|---|---|
| Random-IRT | 13.77 +/- 1.49 | [10.9, 16.7] | p=0.002 |
| Fluid | 6.06 +/- 0.40 | [5.3, 6.9] | p=0.005 |
| BALD | 8.09 +/- 1.13 | [5.9, 10.3] | p=0.022 |
| VarBALD | 6.82 +/- 0.70 | [5.5, 8.2] | p=0.013 |
| DiffBALD | 6.88 +/- 0.68 | [5.5, 8.2] | p=0.010 |
| RankBALD | 5.05 +/- 0.27 | [4.5, 5.6] | -- |

**Saturation (rank correlation, higher is better)**

| Method | Mean +/- SE | 95% CI | vs RankBALD |
|---|---|---|---|
| Random-IRT | 0.702 +/- 0.024 | [0.655, 0.749] | p=0.0004 |
| Fluid | 0.850 +/- 0.017 | [0.817, 0.883] | p=0.066 (ns) |
| BALD | 0.818 +/- 0.021 | [0.778, 0.858] | p=0.006 |
| VarBALD | 0.837 +/- 0.019 | [0.800, 0.874] | p=0.001 |
| DiffBALD | 0.841 +/- 0.019 | [0.804, 0.878] | p=0.002 |
| RankBALD | 0.861 +/- 0.017 | [0.828, 0.894] | -- |

RankBALD's improvements are statistically significant on validity (p=0.021 vs Fluid) and variance (p=0.005 vs Fluid) at budget=100. On saturation, RankBALD improves significantly over all BALD variants but the gap with Fluid does not reach significance at this budget (p=0.066), consistent with the smaller numerical difference in Table 1 (0.861 vs 0.850).

---

## Citation

```bibtex
@inproceedings{dossou2026rankbald,
  title     = {RankBALD: Ranking-Aligned Active Evaluation for Language Models},
  author    = {Bonaventure F. P. Dossou and Jackie Chi Kit Cheung},
  booktitle = {Third Conference on Language Modeling},
url={https://openreview.net/forum?id=WGF7BPL7WF},
  year      = {2026}
}
```
