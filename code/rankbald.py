"""
RankBALD: Ranking-Aligned Active Evaluation for Language Models
================================================================
Bonaventure F. P. Dossou, Jackie Chi Kit Cheung
COLM 2026

This module implements the full experimental pipeline for RankBALD and all
baseline acquisition strategies evaluated in the paper. It provides:

Dataset utilities
-----------------
- Loading IRT parameters and LM evaluation results from the AllenAI
  fluid-benchmarking HuggingFace repository.
- Filtering and preprocessing per benchmark and checkpoint.

IRT utilities
-------------
- 2PL logistic response model (sigmoid, Fisher information, Bernoulli entropy).
- MAP ability estimation with Laplace approximation.
- Entropy-adaptive posterior sampling (scale mixture of Gaussians).

Acquisition strategies
----------------------
- Random-IRT: random item selection with IRT-based ability estimation.
- Fluid Benchmarking: Maximum Fisher Information (MFI) item selection
  (Hofmann et al., 2025).
- BALD: Bayesian Active Learning by Disagreement adapted to the IRT setting.
- DiffBALD: difficulty-aware extension of BALD.
- VarBALD: variance-aware extension of DiffBALD.
- RankBALD: pairwise ordering uncertainty maximization (primary contribution).

Evaluation metrics
------------------
- Validity: average rank distance across capability-matched benchmark pairs.
- Variance: normalized total variation of ability trajectories over checkpoints.
- Saturation: Spearman rank correlation between checkpoint order and ability.

Output
------
The end-to-end runner writes:
- A JSON summary file with aggregated metrics per method and budget.
- A JSONL file with one record per (benchmark, LM, checkpoint).

Usage
-----
    python rankbald.py \\
        --methods random_ability,fluid_benchmarking,bald,var_bald,rank_bald \\
        --eval_sizes 10,50,100,500 \\
        --out_prefix outputs/output \\
        --seed 0

See README.md for full usage instructions and reproduction steps.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from scipy.stats import rankdata, spearmanr
from torch import nn
from torch.optim import Adam

# =============================================================================
# DEBUGGING
# =============================================================================

# Debug verbosity -- set via --debug CLI flag (default: False).
# When True, prints timing spans, DataFrame previews, and array stats.
DEBUG = False
DEBUG_EVERY_S = 1
DEBUG_MAX_ITEMS_PRINT = 10


def _dbg(msg: str) -> None:
    """Print a debug message immediately (flush=True)."""
    print(f"[DEBUG] {msg}", flush=True)


@contextmanager
def _dbg_span(name: str) -> Any:
    """
    Context manager that prints ENTER/EXIT timing (and stacktrace on error).

    This is used heavily to make long experiment loops debuggable when running
    on large benchmarks or GPUs.
    """
    t0 = time.time()
    _dbg(f"ENTER {name}")
    try:
        yield
        dt = time.time() - t0
        _dbg(f"EXIT  {name} ({dt:.3f}s)")
    except Exception as e:
        dt = time.time() - t0
        _dbg(f"EXC   {name} ({dt:.3f}s): {type(e).__name__}: {e}")
        _dbg(traceback.format_exc())
        raise


def _dbg_df(df: pd.DataFrame, name: str) -> None:
    """Print shape and small preview of a DataFrame."""
    _dbg(f"{name}: shape={df.shape} index_name={df.index.name} cols={len(df.columns)}")
    try:
        _dbg(f"{name}: head_index={list(df.index[:min(len(df.index), DEBUG_MAX_ITEMS_PRINT)])}")
        _dbg(f"{name}: head_cols={list(df.columns[:min(len(df.columns), DEBUG_MAX_ITEMS_PRINT)])}")
    except Exception:
        pass


def _dbg_arr(a: np.ndarray, name: str) -> None:
    """Print shape/dtype and NaN count (for floating arrays)."""
    nan_count = "NA"
    if isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.floating):
        nan_count = int(np.isnan(a).sum())
    _dbg(
        f"{name}: shape={getattr(a, 'shape', None)} dtype={getattr(a, 'dtype', None)} nan_count={nan_count}"
    )


def _dbg_minmax(a: np.ndarray, name: str) -> None:
    """Print min/max over finite values of an array (if any)."""
    try:
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            _dbg(f"{name}: no finite values")
            return
        _dbg(f"{name}: min={float(finite.min()):.6f} max={float(finite.max()):.6f}")
    except Exception as e:
        _dbg(f"{name}: minmax failed: {e}")


def _dbg_mem_torch(tag: str) -> None:
    """
    Print CUDA memory usage before/after forcing GC + cache clear.
    Helpful for tracking memory bloat inside big loops.
    """
    if not torch.cuda.is_available():
        return
    try:
        _dbg(
            f"{tag} BEFORE: cuda_mem_alloc={int(torch.cuda.memory_allocated())} "
            f"cuda_mem_reserved={int(torch.cuda.memory_reserved())}"
        )
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        _dbg(
            f"{tag} AFTER: cuda_mem_alloc={int(torch.cuda.memory_allocated())} "
            f"cuda_mem_reserved={int(torch.cuda.memory_reserved())}"
        )
    except Exception:
        pass


# =============================================================================
# CONFIG
# =============================================================================

HF_REPO_ID = "allenai/fluid-benchmarking"
LM_EVAL_RESULTS_PATH = "data/lm_eval_results/{}.csv"
IRT_MODELS_PATH = "data/irt_models/{}.csv"

LMS = [
    "amber-7b",
    "k2-65b",
    "olmo1-7b",
    "olmo2-7b",
    "pythia-7b",
    "pythia-3b",
]

BENCHMARKS = [
    "arc_challenge",
    "gsm8k",
    "hellaswag",
    "truthfulqa_mc2",
    "winogrande",
    "mmlu",
]

METHODS = [
    "random_accuracy",
    "random_ability",
    "fluid_benchmarking",
]

IRT_METHODS = [
    "random_ability",
    "fluid_benchmarking",
]

ESTIMATION_METHOD_IRT = "map"

N_SAMPLES_LIST = (
    list(range(1, 10))
    + list(range(10, 100, 10))
    + list(range(100, 600, 100))
)


# =============================================================================
# DATA LOADING
# =============================================================================

def checkpoint_sort_key(name: str) -> int:
    """
    Sort key for checkpoint column names like "ckpt_000", "ckpt_010", etc.
    Falls back to -1 if no number found.
    """
    nums = re.findall(r"\d+", str(name))
    return int(nums[-1]) if nums else -1


def load_irt_model(repo_id: str, filename: str) -> pd.DataFrame:
    """
    Load a pre-fit IRT model CSV from the HF dataset repo.

    Parameters
    ----------
    repo_id : str
        HuggingFace dataset repo id.
    filename : str
        Relative path inside the dataset repo.

    Returns
    -------
    pd.DataFrame
        IRT dataframe (index is item_id; columns include a/b under some name).
    """
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    df = pd.read_csv(path, index_col=0)
    _dbg_df(df, f"Loaded IRT model: {filename}")
    return df


def load_lm_eval_results(repo_id: str, filename: str, binary: bool = True) -> pd.DataFrame:
    """
    Load per-item evaluation scores for one LM from the HF dataset repo.

    If `binary=True`, threshold scores at 0.5 and return {0,1}.

    Returns
    -------
    pd.DataFrame
        Rows are item ids, columns are checkpoints.
    """
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    eval_results = pd.read_csv(path, index_col=0)
    _dbg_df(eval_results, f"Loaded LM eval results: {filename}")
    return eval_results.ge(0.5).astype(int) if binary else eval_results


def load_open_llm_leaderboard_results() -> Dict[str, Any]:
    """
    Load Open LLM Leaderboard results JSON (used to initialize start_ability per benchmark).

    Returns
    -------
    dict
        Mapping benchmark -> fields including "ability".
    """
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="data/open_llm_leaderboard_results.json",
        repo_type="dataset",
    )
    with open(path, "r") as f:
        results = json.load(f)
    _dbg(f"Loaded Open LLM Leaderboard results: {len(results)} benchmarks")
    return results


# =============================================================================
# BENCHMARK INDEXING
# =============================================================================

def id2benchmark(item_id: str) -> str:
    """
    Extract benchmark name from item id.

    Convention:
      - item_id like "<benchmark>_<...>_<qid>"
      - For mmlu, items start with "mmlu_*" and are normalized to benchmark="mmlu"
    """
    benchmark = "_".join(item_id.split("_")[:-1])
    return "mmlu" if benchmark.startswith("mmlu") else benchmark


def filter_benchmark(lm_eval_results: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """
    Filter an eval-results dataframe to keep only rows belonging to one benchmark.
    """
    mask = lm_eval_results.index.map(lambda x: id2benchmark(x) == benchmark)
    out = lm_eval_results[mask]
    _dbg(f"Filtered benchmark {benchmark}: {out.shape[0]} rows kept")
    return out


# =============================================================================
# IRT CORE UTILITIES
# =============================================================================

def sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """
    Numerically-stable sigmoid.
    """
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def bernoulli_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Bernoulli entropy H(p) in nats (elementwise).
    """
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def fisher_information(theta: float, a: np.ndarray, b: np.ndarray, D: float = 1.0) -> np.ndarray:
    """
    Fisher information for 2PL at theta, per item (a,b).
    """
    z = D * a * (theta - b)
    P = sigmoid_stable(z)
    return (D**2) * (a**2) * (P * (1.0 - P))


# =============================================================================
# ABILITY ESTIMATION (MAP/MLE)
# =============================================================================

def ability_estimate(
    y: np.ndarray,  # (t,) 0/1
    a: np.ndarray,  # (t,)
    b: np.ndarray,  # (t,)
    *,
    method: Literal["map", "mle", "MAP", "MLE"] = "map",
    D: float = 1.0,
    mu0: float = 0.0,
    sigma0: float = 1.0,
    theta0: Optional[float] = None,
    theta_range: Tuple[float, float] = (-4.0, 4.0),
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    1D ability estimation for 2PL given selected item responses.

    This solves for theta that maximizes:
      - MAP: log p(y | theta) + log N(theta; mu0, sigma0^2)
      - MLE: log p(y | theta)

    Uses Newton updates with mild backtracking + bisection fallback.

    Parameters
    ----------
    y, a, b : np.ndarray
        Selected-item vectors, all 1D and same length.
    method : {"map","mle"}
        Estimation type (case-insensitive).
    D : float
        IRT scaling (often 1.0).
    mu0, sigma0 : float
        Prior parameters for MAP. sigma0 must be positive if MAP.
    theta0 : Optional[float]
        Warm start theta. If None, uses mu0.
    theta_range : (float, float)
        Hard clipping range for theta.
    tol : float
        Score tolerance stopping criterion.
    max_iter : int
        Max Newton iterations.

    Returns
    -------
    float
        Estimated theta in [theta_range[0], theta_range[1]].
    """
    method_l = method.lower()
    if method_l not in {"map", "mle"}:
        raise ValueError("method must be 'map' or 'mle'.")

    y = np.asarray(y, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if y.ndim != 1 or a.ndim != 1 or b.ndim != 1:
        raise ValueError("y/a/b must be 1D.")
    if not (y.shape[0] == a.shape[0] == b.shape[0]):
        raise ValueError(f"shape mismatch: y={y.shape}, a={a.shape}, b={b.shape}")

    low, high = float(theta_range[0]), float(theta_range[1])
    if low >= high:
        raise ValueError("theta_range must have low < high.")

    if y.size == 0:
        th0 = mu0 if theta0 is None else float(theta0)
        return float(np.clip(th0, low, high))

    inv_sigma2 = 0.0
    if method_l == "map":
        if sigma0 <= 0:
            raise ValueError("sigma0 must be positive for MAP.")
        inv_sigma2 = 1.0 / (sigma0 * sigma0)

    def score(theta: float) -> float:
        z = D * a * (theta - b)
        P = sigmoid_stable(z)
        prior_term = (mu0 - theta) * inv_sigma2
        likelihood_term = D * np.sum(a * (y - P))
        return prior_term + likelihood_term

    def score_prime(theta: float) -> float:
        z = D * a * (theta - b)
        P = sigmoid_stable(z)
        PQ = P * (1.0 - P)
        prior_term = -inv_sigma2
        likelihood_term = -(D**2) * np.sum((a * a) * PQ)
        return prior_term + likelihood_term

    theta = mu0 if theta0 is None else float(theta0)
    theta = float(np.clip(theta, low, high))
    # _dbg(f"ability_estimate init theta={theta}")
    for _ in range(max_iter):
        T = score(theta)
        if np.abs(T) < tol:
            # _dbg(f"ability_estimate converged iter={it} theta={theta}")
            return theta

        Tp = score_prime(theta)
        if not np.isfinite(Tp) or Tp == 0.0:
            # _dbg(f"ability_estimate break nonfinite Tp iter={it} Tp={Tp}")
            break

        step = -T / Tp
        new_theta = theta + step
        if new_theta < low or new_theta > high or not np.isfinite(new_theta):
            new_theta = float(np.clip(new_theta, low, high))

        # backtracking-ish: shrink if score got worse
        T_abs = abs(T)
        for _bt in range(15):
            T_new = score(new_theta)
            if abs(T_new) < T_abs or not np.isfinite(T_new):
                break
            new_theta = 0.5 * (new_theta + theta)

        theta = new_theta

    # bisection fallback if sign change
    sL = score(low)
    sH = score(high)
    if sL * sH <= 0:
        lo, hi = low, high
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            sM = score(mid)
            if abs(sM) < tol:
                # _dbg(f"ability_estimate bisection converged theta={mid}")
                return mid
            if sL * sM > 0:
                lo, sL = mid, sM
            else:
                hi = mid
        # _dbg(f"ability_estimate bisection end theta={0.5*(lo+hi)}")
        return 0.5 * (lo + hi)
    # _dbg(f"ability_estimate fallback theta={ret}")
    return high if (sL > 0 and sH > 0) else low


# =============================================================================
# FLUID BENCHMARKING (MFI SELECTION)
# =============================================================================

def select_mfi(theta: float, irt_model: np.ndarray, used_mask: np.ndarray, D: float = 1.0) -> int:
    """
    Select next item by Maximum Fisher Information (MFI).

    Parameters
    ----------
    theta : float
        Current ability estimate.
    irt_model : np.ndarray
        Shape (J,2) with columns [a,b].
    used_mask : np.ndarray
        Boolean mask (J,) indicating items already administered.
    D : float
        IRT scaling.

    Returns
    -------
    int
        Index of selected item.
    """
    a = irt_model[:, 0]
    b = irt_model[:, 1]
    fi = fisher_information(theta, a, b, D=D)
    fi_masked = np.where(~used_mask, fi, -np.inf)
    idx = int(np.argmax(fi_masked))
    if not np.isfinite(fi_masked[idx]):
        raise RuntimeError("No available items to select. All items administered?")
    _dbg(f"select_mfi chose idx={idx} fi={float(fi_masked[idx])}")
    return idx


def random_accuracy(lm_responses: np.ndarray, sample_idxes: np.ndarray) -> float:
    """
    Empirical accuracy on a fixed subset of items.
    """
    return float(np.mean(lm_responses[sample_idxes]))


def random_ability(
    lm_responses: np.ndarray,
    irt_model: np.ndarray,
    sample_idxes: np.ndarray,
    estimation_method: str = "map",
    *,
    mu0: float = 0.0,
    sigma0: float = 1.0,
    theta0: Optional[float] = None,
    theta_range: Tuple[float, float] = (-4.0, 4.0),
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Ability estimate on a fixed subset using selected-only estimator.

    This avoids building a full-length vector with NaNs and is faster.
    """
    idx = np.asarray(sample_idxes, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError(f"sample_idxes must be 1D; got {idx.shape}")

    y = np.asarray(lm_responses, dtype=np.float64)[idx]
    a = np.asarray(irt_model, dtype=np.float64)[idx, 0]
    b = np.asarray(irt_model, dtype=np.float64)[idx, 1]

    return float(
        ability_estimate(
            y,
            a,
            b,
            method=estimation_method,
            mu0=float(mu0),
            sigma0=float(sigma0),
            theta0=theta0,
            theta_range=theta_range,
            tol=tol,
            max_iter=max_iter,
        )
    )


def run_fluid_benchmarking(
    *,
    lm_responses: np.ndarray,
    irt_model: np.ndarray,
    start_ability: float = 0.0,
    n_max: int = 100,
    method: Literal["map", "mle", "MAP", "MLE"] = "map",
    D: float = 1.0,
    mu0: float = 0.0,
    sigma0: float = 1.0,
    theta_range: Tuple[float, float] = (-4.0, 4.0),
    tol: float = 1e-6,
    max_iter: int = 100,
    estimator: Optional[Callable[..., float]] = None,
) -> Dict[str, Any]:
    """
    Fluid Benchmarking sequential engine (MFI acquisition + MAP/MLE updates).

    Differences vs a "running-vector with NaNs" implementation:
      - tracks only selected indices
      - ability estimation always uses selected items only

    Returns
    -------
    dict
        {"abilities_fb": List[float], "items_fb": List[int]}
    """
    if estimator is None:
        estimator = ability_estimate  # expects y,a,b

    n_items = irt_model.shape[0]
    used_mask = np.zeros(n_items, dtype=bool)

    items: List[int] = []
    abilities: List[float] = []

    a_all = np.asarray(irt_model[:, 0], dtype=np.float64)
    b_all = np.asarray(irt_model[:, 1], dtype=np.float64)
    y_all = np.asarray(lm_responses, dtype=np.float64)

    # first item
    idx0 = select_mfi(float(start_ability), irt_model, used_mask, D=D)
    used_mask[idx0] = True
    items.append(int(idx0))

    # first theta update
    idx = np.asarray(items, dtype=np.int64)
    th = float(
        estimator(
            y_all[idx],
            a_all[idx],
            b_all[idx],
            method=method,
            D=D,
            mu0=mu0,
            sigma0=sigma0,
            theta0=float(start_ability),
            theta_range=theta_range,
            tol=tol,
            max_iter=max_iter,
        )
    )
    abilities.append(th)
    _dbg(f"run_fluid_benchmarking step=1 idx={idx0} theta={th}")

    while len(items) < n_max and len(items) < n_items:
        idx_next = select_mfi(float(abilities[-1]), irt_model, used_mask, D=D)
        used_mask[idx_next] = True
        items.append(int(idx_next))

        idx = np.asarray(items, dtype=np.int64)
        th = float(
            estimator(
                y_all[idx],
                a_all[idx],
                b_all[idx],
                method=method,
                D=D,
                mu0=mu0,
                sigma0=sigma0,
                theta0=float(abilities[-1]),
                theta_range=theta_range,
                tol=tol,
                max_iter=max_iter,
            )
        )
        abilities.append(th)

        if DEBUG and (len(items) <= 3 or len(items) % max(1, n_max // 10) == 0):
            _dbg(
                f"run_fluid_benchmarking step={len(items)} idx={idx_next} theta={th} "
                f"used={len(items)}/{n_items}"
            )

    _dbg(f"run_fluid_benchmarking done steps={len(items)}")
    return {"abilities_fb": abilities, "items_fb": items}


def fluid_benchmarking(
    lm_responses: np.ndarray,
    irt_model: np.ndarray,
    start_ability: float,
    n_max: int,
    estimation_method: str = "map",
) -> Tuple[List[float], List[int]]:
    """
    Convenience wrapper returning (abilities, items) for Fluid Benchmarking.
    """
    res = run_fluid_benchmarking(
        lm_responses=lm_responses,
        irt_model=irt_model,
        start_ability=start_ability,
        n_max=n_max,
        method=estimation_method,
    )
    return res["abilities_fb"], res["items_fb"]


# =============================================================================
# METRICS
# =============================================================================

def empirical_accuracy_on_items(U_row: np.ndarray, item_idxes: List[int]) -> float:
    """Mean of U_row over selected items."""
    if len(item_idxes) == 0:
        return float("nan")
    return float(np.mean(U_row[item_idxes]))


def projected_perf_on_items(theta: float, irt_model: np.ndarray, item_idxes: List[int]) -> float:
    """
    Mean predicted probability on selected items under 2PL:
      mean_j sigmoid(a_j*(theta-b_j))
    """
    if len(item_idxes) == 0:
        return float("nan")
    a = irt_model[item_idxes, 0]
    b = irt_model[item_idxes, 1]
    return float(np.mean(sigmoid_stable(a * (theta - b))))


def mean_rank_distance(scores_A: np.ndarray, scores_B: np.ndarray) -> float:
    """
    Mean absolute distance between ranks induced by scores_A and scores_B.
    Higher means ranking disagreement.
    """
    valid = ~np.isnan(scores_A) & ~np.isnan(scores_B)
    if np.sum(valid) < 2:
        return np.nan
    rank_A = rankdata(-scores_A[valid], method="average")
    rank_B = rankdata(-scores_B[valid], method="average")
    return float(np.mean(np.abs(rank_A - rank_B)))


def normalized_total_variation(scores: np.ndarray) -> float:
    """
    Normalized total variation along a curve, relative to endpoint range.
    Used as a rough "smoothness/oscillation" proxy.
    """
    valid = ~np.isnan(scores)
    valid_scores = scores[valid]
    n = len(valid_scores)
    if n < 2:
        return np.nan

    abs_diffs = np.abs(np.diff(valid_scores))
    tv_sum = np.sum(abs_diffs)
    range_abs = np.abs(valid_scores[-1] - valid_scores[0])

    MIN_RANGE = 0.01
    if range_abs < MIN_RANGE:
        return np.nan

    tv = (n / (n - 1)) * (tv_sum / range_abs)
    return float(tv)


def saturation_monotonicity(scores: np.ndarray) -> float:
    """
    Saturation monotonicity = |Spearman rho| between step index and score.
    Higher => more monotonic.
    """
    valid = ~np.isnan(scores)
    if np.sum(valid) < 2:
        return np.nan
    order = np.arange(len(scores))[valid]
    valid_scores = scores[valid]
    rho, _ = spearmanr(order, valid_scores)
    return float(np.abs(rho))


def projected_fullQ(theta: float, irt_model: np.ndarray, D: float = 1.0) -> float:
    """
    Mean predicted probability over ALL items.
    """
    if irt_model.shape[0] == 0:
        return float("nan")
    a = irt_model[:, 0]
    b = irt_model[:, 1]
    z = D * a * (theta - b)
    return float(np.mean(sigmoid_stable(z)))


def pack_round_metrics(U_row: np.ndarray, irt_model: np.ndarray, items: List[int], theta: float) -> Dict[str, float]:
    """
    Metrics on the administered set Q_n.
    Returns: empirical, projected, abs_error.
    """
    emp = empirical_accuracy_on_items(U_row, items)
    proj = projected_perf_on_items(theta, irt_model, items)
    return {"empirical": emp, "projected": proj, "abs_error": abs(proj - emp)}


def pack_round_metrics_full(
    U_row: np.ndarray,
    irt_model: np.ndarray,
    items_n: List[int],
    theta: float,
    D: float = 1.0,
) -> Dict[str, float]:
    """
    Metrics both on Q_n and on full benchmark Q.

    Returns
    -------
    dict with keys:
      - empirical_on_Qn
      - projected_on_Qn
      - abs_error_on_Qn
      - projected_on_fullQ
      - abs_error_fullQ
    """
    emp_qn = empirical_accuracy_on_items(U_row, items_n)
    proj_qn = projected_perf_on_items(theta, irt_model, items_n)
    err_qn = abs(proj_qn - emp_qn)

    proj_full = projected_fullQ(theta, irt_model, D=D)
    true_full = float(np.nanmean(U_row))
    err_full = abs(proj_full - true_full)

    return {
        "empirical_on_Qn": emp_qn,
        "projected_on_Qn": proj_qn,
        "abs_error_on_Qn": err_qn,
        "projected_on_fullQ": proj_full,
        "abs_error_fullQ": err_full,
    }


# =============================================================================
# THETA SAMPLING + LAPLACE VARIANCE (used by BALD-family + RankBALD)
# =============================================================================

def entropy_gaussian_1d(var: float) -> float:
    """Differential entropy (nats) of N(0,var)."""
    return 0.5 * np.log(2.0 * np.pi * np.e * max(var, 1e-12))


def k_from_entropy(H: float, k_min: float = 1.0, k_max: float = 4.0) -> float:
    """
    Map entropy to a tail multiplier k (monotone, clipped).
    """
    return float(np.clip(1.0 + H, k_min, k_max))


def sample_theta_entropy_mixture(rng: np.random.Generator, theta_hat: float, var: float, n: int) -> np.ndarray:
    """
    Entropy-adaptive Gaussian mixture sampling for theta:
      - half samples from N(theta_hat, var)
      - half from wider tail N(theta_hat, (k*sqrt(var))^2)
    """
    H = entropy_gaussian_1d(var)
    k = k_from_entropy(H, 2.0, 6.0)
    n1 = n // 2
    n2 = n - n1
    return np.concatenate(
        [
            rng.normal(theta_hat, np.sqrt(var), size=n1),
            rng.normal(theta_hat, k * np.sqrt(var), size=n2),
        ]
    )


def theta_var_laplace(
    theta_hat: float,
    a: np.ndarray,
    b: np.ndarray,
    *,
    method: Literal["map", "mle", "MAP", "MLE"] = "map",
    D: float = 1.0,
    sigma0: float = 1.0,
    min_var: float = 2.0,
    max_var: float = 6.0,
) -> float:
    """
    Laplace variance approximation at theta_hat using selected items only:
      var ≈ 1 / (I(theta_hat) + prior_curvature)

    For MAP with N(mu0, sigma0^2), prior_curvature = 1/sigma0^2.

    Returns clipped var in [min_var, max_var] (paper-aligned).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.size == 0:
        base = (sigma0 * sigma0) if method.lower() == "map" else max_var
        return float(np.clip(base, min_var, max_var))

    z = D * a * (float(theta_hat) - b)
    P = sigmoid_stable(z)
    fisher = float(np.sum((D * D) * (a * a) * P * (1.0 - P)))

    prior_curv = 0.0
    if method.lower() == "map":
        prior_curv = 1.0 / max((sigma0 * sigma0), 1e-12)

    var = 1.0 / max(fisher + prior_curv, 1e-12)
    return float(np.clip(var, min_var, max_var))


# =============================================================================
# BALD-FAMILY ITEM SELECTION (per-checkpoint sequential)
# =============================================================================

def select_bald_item(theta_samples: np.ndarray, irt_model: np.ndarray, used_mask: np.ndarray) -> int:
    """
    BALD selection among unused items.

    Score per candidate item j:
      I(Y_j; theta) ≈ H(E[p]) - E[H(p)]
    where expectation is approximated by theta_samples.
    """
    with _dbg_span("select_bald_item"):
        idxs = np.where(~used_mask)[0]
        _dbg(f"select_bald_item: candidates={idxs.size} samples={theta_samples.size}")

        a = irt_model[idxs, 0][:, None]
        b = irt_model[idxs, 1][:, None]
        th = theta_samples[None, :]

        p = sigmoid_stable(a * (th - b))
        p_bar = p.mean(axis=1)
        bald = bernoulli_entropy(p_bar) - bernoulli_entropy(p).mean(axis=1)

        sel = int(idxs[int(np.argmax(bald))])
        _dbg(f"select_bald_item: chose idx={sel}")
        return sel


def select_bald_weighted_item(theta_samples: np.ndarray, irt_model: np.ndarray, used_mask: np.ndarray) -> int:
    """
    BALD with a difficulty-extremes penalty (posterior-averaged).

    score = bald * mean_t exp(-0.5*(b-theta)^2)
    """
    with _dbg_span("select_bald_weighted_item"):
        idxs = np.where(~used_mask)[0]
        _dbg(f"select_bald_weighted_item: candidates={idxs.size} samples={theta_samples.size}")

        a = irt_model[idxs, 0][:, None]
        b = irt_model[idxs, 1][:, None]
        th = theta_samples[None, :]

        p = sigmoid_stable(a * (th - b))
        p_bar = p.mean(axis=1)
        bald = bernoulli_entropy(p_bar) - bernoulli_entropy(p).mean(axis=1)

        difficulty_penalty = np.exp(-0.5 * (b - th) ** 2).mean(axis=1)
        score = bald * difficulty_penalty

        sel = int(idxs[int(np.argmax(score))])
        _dbg(f"select_bald_weighted_item: chose idx={sel}")
        return sel


def select_var_bald_item(
    theta_samples: np.ndarray,
    irt_model: np.ndarray,
    used_mask: np.ndarray,
    alpha: float,
) -> int:
    """
    Var-BALD (variance-aware) selection among unused items.

    Your implementation:
      - compute bald
      - approximate variance reduction under Laplace-ish update
      - score = bald * (1 + alpha * var_red_norm)
      - apply same difficulty-extremes penalty
    """
    with _dbg_span("select_var_bald_item"):
        idxs = np.where(~used_mask)[0]
        _dbg(f"select_var_bald_item: candidates={idxs.size} samples={theta_samples.size} alpha={alpha}")

        a = irt_model[idxs, 0][:, None]
        b = irt_model[idxs, 1][:, None]
        th = theta_samples[None, :]

        p = sigmoid_stable(a * (th - b))
        p_bar = p.mean(axis=1)
        bald = bernoulli_entropy(p_bar) - bernoulli_entropy(p).mean(axis=1)

        sigma2 = float(np.var(theta_samples))
        sigma2 = max(sigma2, 1e-8)

        fisher_jt = (a**2) * p * (1 - p)
        I_bar = fisher_jt.mean(axis=1)

        sigma_new2 = 1.0 / np.maximum((1.0 / sigma2) + I_bar, 1e-8)
        var_red = sigma2 - sigma_new2
        var_red_norm = var_red / (sigma2 + 1e-12)

        score = bald * (1.0 + alpha * var_red_norm)

        difficulty_penalty = np.exp(-0.5 * (b - th) ** 2).mean(axis=1)
        score *= difficulty_penalty

        sel = int(idxs[int(np.argmax(score))])
        _dbg(f"select_var_bald_item: chose idx={sel}")
        return sel


def select_batchbald_item(theta_samples: np.ndarray, irt_model: np.ndarray, used_mask: np.ndarray) -> int:
    """
    BatchBALD-style sequential setting: reduces to BALD in your code.
    """
    with _dbg_span("select_batchbald_item"):
        return select_bald_item(theta_samples, irt_model, used_mask)


def select_first_index_by_method(
    method: str,
    irt_model: np.ndarray,
    alpha: float,
    seed: int,
    start_ability: float = 0.0,
) -> int:
    """
    Select the first item using the same policy being evaluated.

    For Fluid: MFI at start_ability.
    For BALD-family: sample theta from entropy mixture, then pick accordingly.
    """
    with _dbg_span(f"select_first_index_by_method(method={method}, seed={seed})"):
        rng = np.random.default_rng(seed)
        used_mask = np.zeros(irt_model.shape[0], dtype=bool)

        if method == "fluid_benchmarking":
            idx = select_mfi(start_ability, irt_model, used_mask, D=1.0)
            _dbg(f"select_first_index_by_method: chose idx={idx} (mfi @ theta={start_ability})")
            return idx

        theta_samples = sample_theta_entropy_mixture(rng, float(start_ability), 1.0, 64)

        if method == "bald":
            return select_bald_item(theta_samples, irt_model, used_mask)
        if method == "var_bald":
            return select_var_bald_item(theta_samples, irt_model, used_mask, alpha=alpha)
        if method == "batchbald":
            return select_batchbald_item(theta_samples, irt_model, used_mask)
        if method == "bald_weighted":
            return select_bald_weighted_item(theta_samples, irt_model, used_mask)

        raise ValueError(method)


def run_fluidstyle_custom_engine(
    lm_responses: np.ndarray,
    irt_model: np.ndarray,
    n_max: int,
    method: Literal["bald", "var_bald", "batchbald", "bald_weighted"],
    alpha: float,
    seed: int,
    start_ability: float = 0.0,
) -> Dict[str, Any]:
    """
    Per-checkpoint sequential engine for BALD-family methods.

    Uses:
      - item selection via method
      - MAP ability estimation via `ability_estimate` on selected items only
      - theta posterior sampling via entropy-adaptive mixture, with Laplace variance approx.

    Returns
    -------
    dict
        {"items": List[int], "abilities": List[float]}
    """
    with _dbg_span(f"run_fluidstyle_custom_engine(method={method}, n_max={n_max}, seed={seed})"):
        rng = np.random.default_rng(seed)

        J = irt_model.shape[0]
        used_mask = np.zeros(J, dtype=bool)

        a_all = irt_model[:, 0].astype(np.float64, copy=False)
        b_all = irt_model[:, 1].astype(np.float64, copy=False)
        y_all = np.asarray(lm_responses, dtype=np.float64)

        items: List[int] = []
        abilities: List[float] = []
        theta = float(start_ability)

        # First item
        j0 = select_first_index_by_method(method, irt_model, alpha=alpha, seed=seed, start_ability=start_ability)
        used_mask[j0] = True
        items.append(int(j0))

        # First MAP update
        idx = np.asarray(items, dtype=np.int64)
        theta = float(
            ability_estimate(
                y_all[idx],
                a_all[idx],
                b_all[idx],
                method="map",
                mu0=0.0,
                sigma0=1.0,
                theta0=float(start_ability),
                theta_range=(-4.0, 4.0),
                tol=1e-6,
                max_iter=100,
            )
        )
        abilities.append(theta)
        _dbg(f"run_fluidstyle_custom_engine: step=1 idx={j0} theta={theta}")

        while len(items) < min(n_max, J):
            idx = np.asarray(items, dtype=np.int64)
            a_sel = a_all[idx]
            b_sel = b_all[idx]

            var = theta_var_laplace(
                float(theta),
                a_sel,
                b_sel,
                method="map",
                sigma0=1.0,
                min_var=2.0,
                max_var=6.0,
            )
            theta_samples = sample_theta_entropy_mixture(rng, float(theta), float(var), 64)

            if method == "bald":
                idx_next = select_bald_item(theta_samples, irt_model, used_mask)
            elif method == "var_bald":
                idx_next = select_var_bald_item(theta_samples, irt_model, used_mask, alpha=alpha)
            elif method == "bald_weighted":
                idx_next = select_bald_weighted_item(theta_samples, irt_model, used_mask)
            else:
                idx_next = select_batchbald_item(theta_samples, irt_model, used_mask)

            used_mask[idx_next] = True
            items.append(int(idx_next))

            idx = np.asarray(items, dtype=np.int64)
            theta = float(
                ability_estimate(
                    y_all[idx],
                    a_all[idx],
                    b_all[idx],
                    method="map",
                    mu0=0.0,
                    sigma0=1.0,
                    theta0=float(theta),
                    theta_range=(-4.0, 4.0),
                    tol=1e-6,
                    max_iter=100,
                )
            )
            abilities.append(theta)

            if DEBUG and (len(items) <= 3 or len(items) % max(1, n_max // 10) == 0):
                _dbg(f"run_fluidstyle_custom_engine: step={len(items)} idx={idx_next} theta={theta} used={len(items)}/{J}")

        _dbg(f"run_fluidstyle_custom_engine: done steps={len(items)}")
        return {"items": items, "abilities": abilities}


# =============================================================================
# RANKBALD UTILITIES
# =============================================================================

def mi_binary_vs_pair_many(
    z: np.ndarray,          # (T,) {0,1}
    y_s: np.ndarray,        # (Jc,T) {0,1}
    y_t: np.ndarray,        # (Jc,T) {0,1}
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Vectorized mutual information between a binary variable z and a 4-state pair (y_s,y_t).

    z is length T.
    y_s and y_t are (Jc, T) and represent Monte-Carlo samples for each candidate item.

    Returns
    -------
    np.ndarray
        mi: (Jc,) mutual information in nats, clipped to >=0.
    """
    z = z.astype(np.uint8, copy=False)
    y_s = y_s.astype(np.uint8, copy=False)
    y_t = y_t.astype(np.uint8, copy=False)

    Jc, T = y_s.shape
    if z.shape != (T,) or y_t.shape != (Jc, T):
        raise ValueError(f"shape mismatch: z={z.shape}, y_s={y_s.shape}, y_t={y_t.shape}")

    y = (y_s << 1) | y_t                       # (Jc,T) in {0..3}
    code = (z[None, :] << 2) | y               # (Jc,T) in {0..7}

    row_base = (np.arange(Jc, dtype=np.int64) * 8)[:, None]
    flat = (row_base + code.astype(np.int64)).ravel()

    counts_flat = np.bincount(flat, minlength=Jc * 8).astype(np.float64)
    counts = counts_flat.reshape(Jc, 8).reshape(Jc, 2, 4)

    joint = counts / float(T)
    pz = joint.sum(axis=2, keepdims=True)     # (Jc,2,1)
    py = joint.sum(axis=1, keepdims=True)     # (Jc,1,4)

    denom = (pz * py) + eps
    mi = np.sum(joint * (np.log(joint + eps) - np.log(denom)), axis=(1, 2))
    return np.maximum(mi, 0.0)


def compute_bar_pi_across_ckpts(
    theta_samples_by_ckpt: np.ndarray,  # (S, T)
    b_all: np.ndarray,                  # (Jc,)
    *,
    sigma_pi: float,
) -> np.ndarray:
    """
    Difficulty-aware weight averaged over posterior samples:
      pi_{sj}(theta) = exp( -0.5 * ((b_j - theta)/sigma_pi)^2 )
      bar_pi[s,j]    = mean_t pi_{sj}(theta_s^{(t)})

    Parameters
    ----------
    theta_samples_by_ckpt : np.ndarray
        Shape (S, T) theta samples per checkpoint.
    b_all : np.ndarray
        Candidate item difficulty vector (Jc,).
    sigma_pi : float
        Positive bandwidth (often derived from median posterior std).

    Returns
    -------
    np.ndarray
        bar_pi of shape (S, Jc).
    """
    if sigma_pi <= 0:
        raise ValueError(f"sigma_pi must be > 0; got {sigma_pi}")

    th = theta_samples_by_ckpt[:, None, :].astype(np.float64, copy=False)  # (S,1,T)
    b = b_all[None, :, None].astype(np.float64, copy=False)               # (1,Jc,1)

    diff = (b - th) / float(sigma_pi)
    return np.exp(-0.5 * diff * diff).mean(axis=2)


def select_rank_bald_item_pairwise_fast(
    theta_samples_by_ckpt: np.ndarray,  # (S,Ttheta)
    irt_model: np.ndarray,              # (J,2) [a,b]
    used_mask: np.ndarray,              # (J,)
    pairs: np.ndarray,                  # (P,2) indices over S
    rng: np.random.Generator,
    *,
    normalize: bool = True,
    norm_eps: float = 1e-12,
    use_distance_weighting: bool = True,
    debug: bool = False,
    debug_topk: int = 5,
) -> int:
    """
    Pairwise RankBALD selection (fast path).

    Score per candidate item j:
      S_j = (sum_{(s,t)} omega(|s-t|) * bar_pi[s,j]bar_pi[t,j] * I(Z_{s,t}; (U_s,U_t)))
            / (sum_{(s,t)} omega(|s-t|) * bar_pi[s,j]bar_pi[t,j] + eps)

    where:
      - Z_{s,t} = 1[theta_s > theta_t] is binary per MC sample
      - (U_s,U_t) are sampled binary responses for checkpoints s and t
      - bar_pi weights items by "relevance" to checkpoint ability and item difficulty

    Returns
    -------
    int
        Selected global item index in [0, J).
    """
    if theta_samples_by_ckpt.ndim != 2:
        raise ValueError(f"theta_samples_by_ckpt must be 2D (S,T); got {theta_samples_by_ckpt.shape}")
    if irt_model.ndim != 2 or irt_model.shape[1] != 2:
        raise ValueError(f"irt_model must be (J,2); got {irt_model.shape}")
    if used_mask.ndim != 1 or used_mask.shape[0] != irt_model.shape[0]:
        raise ValueError(f"used_mask must be (J,) matching irt_model; got {used_mask.shape} vs J={irt_model.shape[0]}")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must be (P,2); got {pairs.shape}")

    idxs = np.where(~used_mask)[0]
    if idxs.size == 0:
        raise ValueError("No available candidate items (all used).")

    S, Ttheta = theta_samples_by_ckpt.shape
    if np.any(pairs < 0) or np.any(pairs >= S):
        raise ValueError(f"pairs contains invalid indices for S={S}")

    a_all = irt_model[idxs, 0].astype(np.float64, copy=False)
    b_all = irt_model[idxs, 1].astype(np.float64, copy=False)
    Jc = idxs.size
    P = pairs.shape[0]

    sigma_pi = float(np.median(np.std(theta_samples_by_ckpt.astype(np.float64, copy=False), axis=1))) + 1e-12

    if debug and (DEBUG if "DEBUG" in globals() else False):
        _dbg(
            f"select_rank_bald_item_pairwise_fast: candidates={Jc} S={S} Ttheta={Ttheta} pairs={P} "
            f"normalize={normalize} sigma_pi={sigma_pi:.4g} dist_weight={use_distance_weighting}"
        )

    bar_pi = compute_bar_pi_across_ckpts(theta_samples_by_ckpt, b_all, sigma_pi=sigma_pi)

    th = theta_samples_by_ckpt.astype(np.float64, copy=False)[:, None, :]  # (S,1,T)
    a = a_all[None, :, None]                                               # (1,Jc,1)
    b = b_all[None, :, None]                                               # (1,Jc,1)
    logits = a * (th - b)                                                  # (S,Jc,T)
    p = sigmoid_stable(logits)                                             # (S,Jc,T)
    y = rng.binomial(n=1, p=np.clip(p, 1e-6, 1 - 1e-6)).astype(np.uint8)    # (S,Jc,T)

    z_pairs = (theta_samples_by_ckpt[pairs[:, 0], :] > theta_samples_by_ckpt[pairs[:, 1], :]).astype(np.uint8)

    if use_distance_weighting:
        dmax = S - 1
        if dmax <= 1:
            omega = np.ones((P,), dtype=np.float64)
        else:
            d = np.abs(pairs[:, 0].astype(np.int64) - pairs[:, 1].astype(np.int64)).astype(np.float64)
            omega = 1.0 - (d - 1.0) / float(dmax - 1)
            omega = np.clip(omega, 0.0, 1.0).astype(np.float64)
    else:
        omega = None

    scores = np.zeros((Jc,), dtype=np.float64)
    mass = np.zeros((Jc,), dtype=np.float64) if normalize else None

    for k in range(P):
        s = int(pairs[k, 0])
        t = int(pairs[k, 1])

        mi_vec = mi_binary_vs_pair_many(z_pairs[k], y[s], y[t])  # (Jc,)
        w = (bar_pi[s] * bar_pi[t]).astype(np.float64, copy=False)

        if omega is not None:
            w = w * omega[k]

        scores += w * mi_vec
        if normalize:
            mass += w

    if normalize:
        scores = scores / (mass + float(norm_eps))

    best_jpos = int(np.argmax(scores))
    best_idx = int(idxs[best_jpos])

    if debug and (DEBUG if "DEBUG" in globals() else False):
        top_idx = np.argsort(scores)[::-1][: max(1, min(debug_topk, Jc))]
        _dbg("select_rank_bald_item_pairwise_fast: top candidates by score:")
        for jpos in top_idx:
            _dbg(f"  idx={int(idxs[jpos])} score={float(scores[jpos]):.6e}")
        _dbg(f"select_rank_bald_item_pairwise_fast: chose idx={best_idx} best_score={float(scores[best_jpos]):.6e}")

    return best_idx


def run_rank_bald_curve_engine(
    U_full: np.ndarray,          # (S, J)
    irt_model: np.ndarray,       # (J, 2)
    n_max: int,
    seed: int,
    start_ability: float = 0.0,
    n_theta_samples: int = 64,
    *,
    debug: bool = False,
    debug_every: int = 10,
    n_pair_samples: Optional[int] = None,
    map_mu0: float = 0.0,
    map_sigma0: float = 1.0,
    theta_range: Tuple[float, float] = (-4.0, 4.0),
    min_var: float = 2.0,
    max_var: float = 6.0,
) -> Dict[str, Any]:
    """
    Setup-1 RankBALD engine (fixed IRT model).

    - Selects a shared acquisition order across checkpoints.
    - Updates each checkpoint ability theta_s via MAP using the selected items only.

    Returns
    -------
    dict
        {"items": List[int], "abilities": List[List[float]]}
        abilities[t][s] = theta for checkpoint s after selecting t+1 items.
    """
    with _dbg_span(f"run_rank_bald_curve_engine_fast(n_max={n_max}, seed={seed})"):
        rng = np.random.default_rng(seed)

        if U_full.ndim != 2:
            raise ValueError(f"U_full must be 2D (S,J); got {U_full.shape}")
        if irt_model.ndim != 2 or irt_model.shape[1] != 2:
            raise ValueError(f"irt_model must be (J,2); got {irt_model.shape}")

        S, J = U_full.shape
        if irt_model.shape[0] != J:
            raise ValueError(f"irt_model J mismatch: irt_model.shape[0]={irt_model.shape[0]} vs U_full J={J}")

        used_mask = np.zeros(J, dtype=bool)
        theta = np.full(S, float(start_ability), dtype=np.float64)

        abilities_hist: List[List[float]] = []
        items: List[int] = []

        all_pairs = np.array([(s, t) for s in range(S) for t in range(s + 1, S)], dtype=np.int32)
        if n_pair_samples is not None and all_pairs.shape[0] > int(n_pair_samples):
            pairs = all_pairs[rng.choice(all_pairs.shape[0], size=int(n_pair_samples), replace=False)]
        else:
            pairs = all_pairs

        if debug or (("DEBUG" in globals()) and DEBUG):
            _dbg(f"rank_bald_engine: S={S} J={J} n_max={n_max} Ttheta={n_theta_samples} pairs={pairs.shape[0]}")

        target_steps = min(int(n_max), int(J))
        a_all = irt_model[:, 0].astype(np.float64, copy=False)
        b_all = irt_model[:, 1].astype(np.float64, copy=False)

        while len(items) < target_steps:
            step = len(items) + 1
            if (debug or (("DEBUG" in globals()) and DEBUG)) and (step <= 3 or step % max(1, int(debug_every)) == 0):
                _dbg(f"rank_bald_engine: step={step}/{target_steps} selected_so_far={len(items)}")

            theta_samples_by_ckpt = np.empty((S, n_theta_samples), dtype=np.float64)

            if len(items) == 0:
                base_var = float(np.clip(map_sigma0 * map_sigma0, min_var, max_var))
                for s in range(S):
                    theta_samples_by_ckpt[s] = sample_theta_entropy_mixture(
                        rng, float(theta[s]), float(base_var), int(n_theta_samples)
                    )
            else:
                items_np = np.asarray(items, dtype=np.int64)
                a_sel = a_all[items_np]
                b_sel = b_all[items_np]
                for s in range(S):
                    var = theta_var_laplace(
                        float(theta[s]),
                        a_sel,
                        b_sel,
                        method="map",
                        sigma0=float(map_sigma0),
                        min_var=float(min_var),
                        max_var=float(max_var),
                    )
                    theta_samples_by_ckpt[s] = sample_theta_entropy_mixture(
                        rng, float(theta[s]), float(var), int(n_theta_samples)
                    )

            j = select_rank_bald_item_pairwise_fast(
                theta_samples_by_ckpt,
                irt_model,
                used_mask,
                pairs,
                rng,
                debug=debug and (step <= 2),
                debug_topk=5,
            )

            used_mask[j] = True
            items.append(int(j))

            items_np = np.asarray(items, dtype=np.int64)
            a_sel = a_all[items_np]
            b_sel = b_all[items_np]
            Y_sel = U_full[:, items_np].astype(np.float64, copy=False)

            for s in range(S):
                theta[s] = float(
                    ability_estimate(
                        Y_sel[s],
                        a_sel,
                        b_sel,
                        method="map",
                        mu0=float(map_mu0),
                        sigma0=float(map_sigma0),
                        theta0=float(theta[s]),
                        theta_range=theta_range,
                        tol=1e-6,
                        max_iter=100,
                    )
                )

            abilities_hist.append([float(x) for x in theta])

            if (debug or (("DEBUG" in globals()) and DEBUG)) and (step <= 3 or step % max(1, int(debug_every)) == 0):
                _dbg(
                    f"rank_bald_engine: step={step}/{target_steps} chose_item={j} used={int(np.sum(used_mask))}/{J} "
                    f"theta(min/mean/max)=({theta.min():.3f},{theta.mean():.3f},{theta.max():.3f})"
                )

        return {"items": items, "abilities": abilities_hist}


# =============================================================================
# CURVE CHOICE (for validity/variance/saturation analysis)
# =============================================================================

def curve_for_metrics(method: str, payload: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    """
    Choose which curve is used for downstream metrics.

    - random_accuracy: empirical curve
    - ability-based methods: projected curve

    Returns
    -------
    (curve, curve_name)
    """
    if method == "random_accuracy":
        return np.asarray(payload["empirical_curve"], dtype=float), "empirical"
    return np.asarray(payload["projected_curve"], dtype=float), "projected"


# =============================================================================
# RUNNER
# =============================================================================

def run_all(
    *,
    query_size: int,
    batch_size: int,
    lms: List[str],
    benchmarks: List[str],
    methods: List[str],
    eval_sizes: List[int],
    alpha: float,
    device: str,
    seed: int,
    out_prefix: str,
) -> None:
    """
    Full experiment runner — benchmark-first loop order.

    Writes:
      - {out_prefix}.jsonl   one record per (benchmark, LM, checkpoint)
      - {out_prefix}.json    aggregated metrics per method and budget
    """
    with _dbg_span("run_all"):
        _dbg(f"run_all: query_size={query_size} batch_size={batch_size} seed={seed} device={device}")
        _dbg(f"run_all: lms={lms}")
        _dbg(f"run_all: benchmarks={benchmarks}")
        _dbg(f"run_all: methods={methods}")
        _dbg(f"run_all: eval_sizes={eval_sizes}")

        if query_size != 1 or batch_size != 1:
            raise ValueError("Sequential only: query_size and batch_size must both be 1.")

        open_llm_leaderboard_results = load_open_llm_leaderboard_results()
        _dbg("Loaded Open LLM Leaderboard results")

        jsonl_path = f"{out_prefix}.jsonl"
        out: Dict[str, Any] = {
            "lms": lms,
            "benchmarks": benchmarks,
            "methods": methods,
            "eval_sizes": eval_sizes,
            "results": {},
            "summary": {},
        }

        with open(jsonl_path, "w") as jsonl_file:
            for bench in tqdm.tqdm(benchmarks, desc="Benchmarks", total=len(benchmarks), position=0):
                with _dbg_span(f"bench_loop(bench={bench})"):
                    start_ability = float(np.mean(list(open_llm_leaderboard_results[bench]["ability"].values())))
                    _dbg(f"[{bench}] start_ability = {start_ability:.3f}")

                    irt_df = load_irt_model(HF_REPO_ID, IRT_MODELS_PATH.format(bench))
                    irt_df.index = irt_df.index.astype(str)

                    samples_dict = None  # fixed random subsets per benchmark

                    for lm in tqdm.tqdm(lms, desc=f"{bench} • LMs", total=len(lms), position=1):
                        print(f"LM={lm} • bench={bench}")
                        with _dbg_span(f"lm_loop(lm={lm}, bench={bench})"):
                            eval_df = load_lm_eval_results(HF_REPO_ID, LM_EVAL_RESULTS_PATH.format(lm), binary=True)
                            eval_df = eval_df[sorted(eval_df.columns, key=checkpoint_sort_key)]

                            bench_t = filter_benchmark(eval_df, bench)
                            _dbg_df(bench_t, "bench_t")
                            if bench_t.shape[0] == 0:
                                _dbg(f"Skipping {lm} on {bench}: no items")
                                continue

                            items = bench_t.index.astype(str).tolist()
                            U_full = bench_t.to_numpy(dtype=float).T  # (S,J)
                            S, J = U_full.shape
                            _dbg(f"{lm} on {bench}: items={len(items)} S={S} J={J}")

                            irt_model = None
                            items_aligned = [it for it in items if it in set(irt_df.index)]
                            _dbg(f"Aligned items: {len(items_aligned)} / {len(items)}")
                            if len(items_aligned) == 0:
                                _dbg(f"No aligned items for {lm} on {bench}, skipping")
                                continue

                            bench_t = bench_t.loc[items_aligned]
                            U_full = bench_t.to_numpy(dtype=float).T
                            items = items_aligned
                            J = U_full.shape[1]

                            cols_lower = {c.lower(): c for c in irt_df.columns}
                            a_col = cols_lower.get("a") or cols_lower.get("disc") or cols_lower.get("discrimination")
                            b_col = cols_lower.get("b") or cols_lower.get("diff") or cols_lower.get("difficulty")
                            if a_col is None or b_col is None:
                                raise ValueError(
                                    f"Could not find IRT columns for a/b in irt_df for bench={bench}. "
                                    f"Columns: {list(irt_df.columns)}"
                                )

                            irt_model = np.stack(
                                [
                                    irt_df.loc[items, a_col].to_numpy(dtype=float),
                                    irt_df.loc[items, b_col].to_numpy(dtype=float),
                                ],
                                axis=1,
                            )
                            _dbg_arr(irt_model, "irt_model")
                            if DEBUG:
                                _dbg_minmax(irt_model[:, 0], "irt_model.a")
                                _dbg_minmax(irt_model[:, 1], "irt_model.b")

                            # fixed random subsets once per benchmark (depends on J)
                            if samples_dict is None:
                                random.seed(seed)
                                samples_dict = {}
                                n_items = J
                                for n_samples in N_SAMPLES_LIST:
                                    if n_samples > n_items:
                                        raise ValueError(
                                            f"Number of samples={n_samples} > number of items={n_items} "
                                            f"for benchmark {bench}."
                                        )
                                    samples_dict[n_samples] = np.array(
                                        random.sample(range(n_items), n_samples), dtype=int
                                    )
                                _dbg(
                                    f"[{bench}] Fixed random subsets generated for {len(samples_dict)} sizes: "
                                    f"{list(samples_dict.keys())} (J={J})"
                                )

                            n_max = max(samples_dict.keys()) if len(samples_dict) else 0

                            out["results"].setdefault(lm, {})
                            out["results"][lm].setdefault(bench, {})

                            true_curve = U_full.mean(axis=1).astype(float)
                            out["results"][lm][bench]["true_curve_full_accuracy"] = true_curve.tolist()

                            # JSONL lines (per checkpoint)
                            jsonl_lines: List[Dict[str, Any]] = []
                            for s in range(S):
                                jsonl_lines.append(
                                    {
                                        "benchmark": bench,
                                        "lm": lm,
                                        "checkpoint": f"ckpt_{s:03d}",
                                        "full_accuracy": float(true_curve[s]),
                                        "full_ability": None,
                                        "abilities_fb": [],
                                        "items_fb": [],
                                    }
                                )

                            out["results"][lm][bench].setdefault("random_accuracy", {})
                            out["results"][lm][bench].setdefault("random_ability", {})
                            out["results"][lm][bench].setdefault("rank_bald", {})

                            # ---------------------------
                            # Random baselines
                            # ---------------------------
                            for n_samples in N_SAMPLES_LIST:
                                emp = np.zeros(S, dtype=float)
                                proj = np.zeros(S, dtype=float)
                                err = np.zeros(S, dtype=float)
                                proj_full = np.zeros(S, dtype=float)
                                err_full = np.zeros(S, dtype=float)

                                idx = samples_dict[n_samples]
                                items_n = idx.tolist()

                                if "random_accuracy" in methods:
                                    for s in range(S):
                                        score = float(random_accuracy(U_full[s], idx))
                                        jsonl_lines[s][f"random_accuracy_{n_samples}"] = score
                                        emp[s] = score
                                        proj[s] = np.nan
                                        err[s] = np.nan

                                    out["results"][lm][bench]["random_accuracy"][str(n_samples)] = {
                                        "empirical_curve": emp.tolist(),
                                        "projected_curve": proj.tolist(),
                                        "abs_error_curve": err.tolist(),
                                        "abs_error_mean": float(np.mean(err)),
                                        "abs_error_median": float(np.median(err)),
                                        "projected_full_curve": proj_full.tolist(),
                                        "abs_error_full_curve": err_full.tolist(),
                                        "abs_error_full_mean": float(np.mean(err_full)),
                                        "abs_error_full_median": float(np.median(err_full)),
                                    }

                                if "random_ability" in methods:
                                    for s in range(S):
                                        th = random_ability(U_full[s], irt_model, idx, estimation_method="map")
                                        jsonl_lines[s][f"random_ability_{n_samples}"] = float(th)

                                        m = pack_round_metrics(U_full[s], irt_model, items_n, float(th))
                                        emp[s] = m["empirical"]
                                        proj[s] = m["projected"]
                                        err[s] = m["abs_error"]

                                    out["results"][lm][bench]["random_ability"][str(n_samples)] = {
                                        "empirical_curve": emp.tolist(),
                                        "projected_curve": proj.tolist(),
                                        "abs_error_curve": err.tolist(),
                                        "abs_error_mean": float(np.mean(err)),
                                        "abs_error_median": float(np.median(err)),
                                        "projected_full_curve": proj_full.tolist(),
                                        "abs_error_full_curve": err_full.tolist(),
                                        "abs_error_full_mean": float(np.mean(err_full)),
                                        "abs_error_full_median": float(np.median(err_full)),
                                    }

                            full_idx = np.arange(J, dtype=int)
                            for s in range(S):
                                th_full = random_ability(U_full[s], irt_model, full_idx, estimation_method="map")
                                jsonl_lines[s]["full_ability"] = float(th_full)

                            # setup1-only fixed-IRT rank_bald
                            if "rank_bald" in methods:
                                res_rb = run_rank_bald_curve_engine(
                                    U_full=U_full,
                                    irt_model=irt_model,
                                    n_max=n_max,
                                    seed=seed,
                                    start_ability=start_ability,
                                    n_theta_samples=64,
                                )

                                emp_curve = {n: np.zeros(S, dtype=float) for n in eval_sizes}
                                proj_curve = {n: np.zeros(S, dtype=float) for n in eval_sizes}
                                err_curve = {n: np.zeros(S, dtype=float) for n in eval_sizes}
                                proj_full_curve = {n: np.zeros(S, dtype=float) for n in eval_sizes}
                                err_full_curve = {n: np.zeros(S, dtype=float) for n in eval_sizes}

                                items_rb = res_rb["items"]
                                theta_hist_rb = res_rb["abilities"]

                                final_items = [None] * S
                                for s in range(S):
                                    jsonl_lines[s]["abilities_rank_bald"] = [float(theta_hist_rb[t][s]) for t in range(len(theta_hist_rb))]
                                    jsonl_lines[s]["items_rank_bald"] = [int(i) for i in items_rb]

                                    final_items[s] = items_rb[:n_max]
                                    for n in eval_sizes:
                                        n_eff = min(n, len(items_rb))
                                        items_n = items_rb[:n_eff]
                                        abilities_s = [float(theta_hist_rb[t][s]) for t in range(len(theta_hist_rb))]
                                        theta_n = float(abilities_s[n_eff - 1]) if n_eff > 0 else start_ability

                                        m = pack_round_metrics(U_full[s], irt_model, items_n, theta_n)
                                        emp_curve[n][s] = m["empirical"]
                                        proj_curve[n][s] = m["projected"]
                                        err_curve[n][s] = m["abs_error"]

                                        m_full = pack_round_metrics_full(U_full[s], irt_model, items_n, theta_n)
                                        proj_full_curve[n][s] = m_full["projected_on_fullQ"]
                                        err_full_curve[n][s] = m_full["abs_error_fullQ"]

                                for n in eval_sizes:
                                    out["results"][lm][bench]["rank_bald"][str(n)] = {
                                        "empirical_curve": emp_curve[n].tolist(),
                                        "projected_curve": proj_curve[n].tolist(),
                                        "abs_error_curve": err_curve[n].tolist(),
                                        "abs_error_mean": float(np.mean(err_curve[n])),
                                        "abs_error_median": float(np.median(err_curve[n])),
                                        "projected_full_curve": proj_full_curve[n].tolist(),
                                        "abs_error_full_curve": err_full_curve[n].tolist(),
                                        "abs_error_full_mean": float(np.mean(err_full_curve[n])),
                                        "abs_error_full_median": float(np.median(err_full_curve[n])),
                                    }

                                out["results"][lm][bench]["rank_bald"]["final_acquired_items_per_checkpoint"] = final_items

                            # Remaining methods
                            methods_rem = [m for m in methods if m not in {"random_ability", "random_accuracy", "rank_bald"}]

                            for method in tqdm.tqdm(methods_rem, desc=f"{lm} • {bench} • methods", position=2, total=len(methods_rem)):
                                with _dbg_span(f"method_loop(method={method})"):
                                    _dbg(f"adaptive: n_max={n_max} S={S} J={J}")
                                    out["results"][lm][bench].setdefault(method, {})

                                    emp_curve = {n: np.zeros(S, dtype=float) for n in eval_sizes}
                                    proj_curve = {n: np.zeros(S, dtype=float) for n in eval_sizes}
                                    err_curve = {n: np.zeros(S, dtype=float) for n in eval_sizes}
                                    proj_full_curve = {n: np.zeros(S, dtype=float) for n in eval_sizes}
                                    err_full_curve = {n: np.zeros(S, dtype=float) for n in eval_sizes}

                                    final_items = [None] * S

                                    for s in range(S):
                                        with _dbg_span(f"checkpoint(s={s}/{S-1})"):
                                                if method == "fluid_benchmarking":
                                                    abilities, items_sel = fluid_benchmarking(
                                                        lm_responses=U_full[s],
                                                        irt_model=irt_model,
                                                        start_ability=start_ability,
                                                        n_max=n_max,
                                                        estimation_method="map",
                                                    )
                                                    jsonl_lines[s]["abilities_fb"] = [float(a) for a in abilities]
                                                    jsonl_lines[s]["items_fb"] = [int(i) for i in items_sel]
                                                else:
                                                    res = run_fluidstyle_custom_engine(
                                                        lm_responses=U_full[s],
                                                        irt_model=irt_model,
                                                        n_max=n_max,
                                                        method=method,  # type: ignore[arg-type]
                                                        alpha=alpha,
                                                        seed=seed + s,
                                                        start_ability=start_ability,
                                                    )
                                                    abilities = res["abilities"]
                                                    items_sel = res["items"]
                                                    jsonl_lines[s][f"abilities_{method}"] = [float(a) for a in abilities]
                                                    jsonl_lines[s][f"items_{method}"] = [int(i) for i in items_sel]

                                                final_items[s] = items_sel[:n_max]
                                                for n in eval_sizes:
                                                    n_eff = min(n, len(items_sel))
                                                    items_n = items_sel[:n_eff]
                                                    theta_n = float(abilities[n_eff - 1]) if n_eff > 0 else start_ability

                                                    m = pack_round_metrics(U_full[s], irt_model, items_n, theta_n)
                                                    emp_curve[n][s] = m["empirical"]
                                                    proj_curve[n][s] = m["projected"]
                                                    err_curve[n][s] = m["abs_error"]

                                                    m_full = pack_round_metrics_full(U_full[s], irt_model, items_n, theta_n)
                                                    proj_full_curve[n][s] = m_full["projected_on_fullQ"]
                                                    err_full_curve[n][s] = m_full["abs_error_fullQ"]

                                        for n in eval_sizes:
                                            out["results"][lm][bench][method][str(n)] = {
                                                "empirical_curve": emp_curve[n].tolist(),
                                                "projected_curve": proj_curve[n].tolist(),
                                                "abs_error_curve": err_curve[n].tolist(),
                                                "abs_error_mean": float(np.mean(err_curve[n])),
                                                "abs_error_median": float(np.median(err_curve[n])),
                                                "projected_full_curve": proj_full_curve[n].tolist(),
                                                "abs_error_full_curve": err_full_curve[n].tolist(),
                                                "abs_error_full_mean": float(np.mean(err_full_curve[n])),
                                                "abs_error_full_median": float(np.median(err_full_curve[n])),
                                            }

                                        out["results"][lm][bench][method]["final_acquired_items_per_checkpoint"] = final_items

                            # per-LLM average error storage
                            per_llm_errors = out["summary"].setdefault("per_llm_errors", {})
                            bench_dict = per_llm_errors.setdefault(bench, {})

                            for method in methods:
                                if method not in bench_dict:
                                    bench_dict[method] = {"eval_sizes": eval_sizes[:]}
                                method_dict = bench_dict[method]

                                method_res = out["results"][lm][bench].get(method, {})
                                errors_list: List[float] = []
                                for n in eval_sizes:
                                    payload = method_res.get(str(n), {})
                                    abs_curve = payload.get("abs_error_curve", [])
                                    mean_err = float(np.nanmean(abs_curve)) if abs_curve else float("nan")
                                    errors_list.append(mean_err)

                                method_dict[lm] = errors_list
                                _dbg(f"Stored per_llm_errors {bench}/{method}/{lm}: {errors_list}")

                            # write JSONL
                            for s in range(S):
                                jsonl_file.write(json.dumps(jsonl_lines[s], separators=(",", ":")) + "\n")

            out_path = f"{out_prefix}.json"
            with _dbg_span(f"write_json({out_path})"):
                _dbg(f"writing out_path={out_path}")
                with open(out_path, "w") as f:
                    json.dump(out, f, indent=2)

            _dbg(f"JSONL output written to {jsonl_path}")


def main() -> None:
    """
    CLI entrypoint.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_size", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_prefix", type=str, default="outputs/output")
    ap.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable verbose debug output (timing spans, DataFrame previews).",
    )
    ap.add_argument("--lms", type=str, default=",".join(LMS))
    ap.add_argument("--benchmarks", type=str, default=",".join(BENCHMARKS))
    ap.add_argument(
        "--methods",
        type=str,
        default="random_ability,fluid_benchmarking,bald_weighted,bald,var_bald,rank_bald",
    )
    ap.add_argument("--eval_sizes", type=str, default="10,50,100,500")

    args = ap.parse_args()

    if args.query_size != 1 or args.batch_size != 1:
        raise ValueError("Sequential only: query_size and batch_size must both be 1.")

    # Wire CLI flags into module-level globals
    global DEBUG
    DEBUG = args.debug

    lms = [x.strip() for x in args.lms.split(",") if x.strip()]
    benchmarks = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    eval_sizes = [int(x.strip()) for x in args.eval_sizes.split(",") if x.strip()]

    run_all(
        query_size=args.query_size,
        batch_size=args.batch_size,
        lms=lms,
        benchmarks=benchmarks,
        methods=methods,
        eval_sizes=eval_sizes,
        alpha=args.alpha,
        device=args.device,
        seed=args.seed,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()