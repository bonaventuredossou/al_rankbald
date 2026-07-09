"""
Fluid Benchmarking runner adapted for AfriBench.

Differences from the original AllenAI script:
  - Loads lm_eval_results and irt_models from local files instead of HuggingFace
  - id2benchmark: afrimgsm_amh_0 -> afrimgsm  (strips lang + number suffix)
  - start_ability computed from fitted theta (mean b from IRT model)
  - LMS / BENCHMARKS / paths point to afribench/data/
  - rank_bald uses run_rank_bald_curve_engine (shared acquisition across checkpoints)

Usage
-----
    python run_afribench.py \
        --data_dir   ./data \
        --lms        gpt-5-2025-08-07,gemini-3-pro-preview \
        --benchmarks afrimgsm,afrimmlu,afrixnli,belebele,sib \
        --methods    random_accuracy,random_ability,fluid_benchmarking,rank_bald,bald,var_bald,bald_weighted \
        --eval_sizes 10,50,100,500 \
        --out_prefix results/afribench_setup1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr


# =============================================================================
# DEBUG
# =============================================================================

# Debug verbosity -- set via --debug CLI flag (default: False).
DEBUG = False
DEBUG_EVERY_S = 1
DEBUG_MAX_ITEMS_PRINT = 10


def _dbg(msg: str) -> None:
    print(f"[DEBUG] {msg}", flush=True)

@contextmanager
def _dbg_span(name: str) -> Any:
    t0 = time.time()
    _dbg(f"ENTER {name}")
    try:
        yield
        _dbg(f"EXIT  {name} ({time.time()-t0:.3f}s)")
    except Exception as e:
        _dbg(f"EXC   {name} ({time.time()-t0:.3f}s): {type(e).__name__}: {e}")
        _dbg(traceback.format_exc())
        raise

def _dbg_df(df, name):
    _dbg(f"{name}: shape={df.shape} cols={list(df.columns[:DEBUG_MAX_ITEMS_PRINT])}")

def _dbg_arr(a, name):
    nan_count = int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else "NA"
    _dbg(f"{name}: shape={a.shape} dtype={a.dtype} nan_count={nan_count}")

def _dbg_minmax(a, name):
    try:
        f = a[np.isfinite(a)]
        _dbg(f"{name}: min={float(f.min()):.6f} max={float(f.max()):.6f}" if f.size else f"{name}: no finite values")
    except Exception as e:
        _dbg(f"{name}: minmax failed: {e}")

def _safe_nanmean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.nanmean(a)) if np.any(np.isfinite(a)) else float("nan")

def _safe_nanmedian(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.nanmedian(a)) if np.any(np.isfinite(a)) else float("nan")

def _dbg_mem_torch(tag):
    if not torch.cuda.is_available():
        return
    try:
        import gc; gc.collect(); torch.cuda.empty_cache()
        _dbg(f"{tag}: cuda_mem_alloc={int(torch.cuda.memory_allocated())}")
    except Exception:
        pass


# =============================================================================
# CONFIG
# =============================================================================

LMS = [
    "gemini_gemma",
    "gpt",
    "qwen",
]

BENCHMARKS = [
    "afrimgsm",
    "afrimmlu",
    "afrixnli",
    "belebele",
    "sib",
]

BENCHMARK_FILE_MAP = {
    "afrimgsm": "afrimgsm.csv",
    "afrimmlu": "afrimmlu.csv",
    "afrixnli": "afrixnli.csv",
    "belebele": "belebele.csv",
    "sib":      "sib.csv",
}

N_SAMPLES_LIST = (
    list(range(1, 10))
    + list(range(10, 100, 10))
    + list(range(100, 600, 100))
)


# =============================================================================
# LOCAL DATA LOADING
# =============================================================================

def checkpoint_sort_key(name: str) -> int:
    nums = re.findall(r"\d+", str(name))
    return int(nums[-1]) if nums else -1

def load_irt_model_local(data_dir: str, benchmark: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "irt_models", f"{benchmark}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"IRT model not found: {path}\nRun fit_irt.py first.")
    df = pd.read_csv(path, index_col=0)
    _dbg_df(df, f"Loaded IRT model: {path}")
    return df

def load_lm_eval_results_local(data_dir: str, lm: str, binary: bool = True) -> pd.DataFrame:
    path = os.path.join(data_dir, "lm_eval_results", f"{lm}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"LM eval results not found: {path}\nRun pivot_to_per_model.py first.")
    df = pd.read_csv(path, index_col=0)
    df = df[sorted(df.columns, key=checkpoint_sort_key)]
    _dbg_df(df, f"Loaded LM eval: {path}")
    return df.ge(0.5).astype(int) if binary else df

def compute_start_abilities(data_dir: str, benchmarks: List[str]) -> Dict[str, float]:
    """
    Read mean fitted theta from data_dir/irt_models/<bench>_theta.json.
    Clipped to [-2, 2] to avoid starting MFI at an extreme of the theta scale.
    Falls back to 0.0 if file not found.
    """
    abilities = {}
    for bench in benchmarks:
        path = os.path.join(data_dir, "irt_models", f"{bench}_theta.json")
        if not os.path.exists(path):
            _dbg(f"compute_start_abilities: {path} not found, defaulting to 0.0 "
                 "(re-run fit_irt.py to generate theta files)")
            abilities[bench] = 0.0
            continue
        with open(path) as f:
            theta_dict = json.load(f)
        abilities[bench] = float(np.clip(theta_dict["mean"], -2.0, 2.0))
        _dbg(f"start_ability[{bench}] = {abilities[bench]:.3f} "
             f"(mean theta clipped to [-2,2], models={[k for k in theta_dict if k != 'mean']})")
    return abilities


# =============================================================================
# BENCHMARK INDEXING
# =============================================================================

def id2benchmark(item_id: str) -> str:
    """afrimgsm_amh_0 -> afrimgsm  |  sib_amh_0 -> sib"""
    parts = item_id.split("_")
    return "_".join(parts[:-2])

def filter_benchmark(lm_eval_results: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    mask = lm_eval_results.index.map(lambda x: id2benchmark(x) == benchmark)
    out  = lm_eval_results[mask]
    _dbg(f"Filtered benchmark={benchmark}: {out.shape[0]} rows")
    return out


# =============================================================================
# IRT CORE
# =============================================================================

def sigmoid_stable(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

def bernoulli_entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def fisher_information(theta, a, b, D=1.0):
    z = D * a * (theta - b)
    P = sigmoid_stable(z)
    return (D**2) * (a**2) * (P * (1.0 - P))

def ability_estimate(y, a, b, *, method="map", D=1.0, mu0=0.0, sigma0=1.0,
                     theta0=None, theta_range=(-4.0, 4.0), tol=1e-6, max_iter=100):
    y = np.asarray(y, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    low, high = float(theta_range[0]), float(theta_range[1])
    if y.size == 0:
        return float(np.clip(mu0 if theta0 is None else theta0, low, high))
    inv_sigma2 = (1.0 / sigma0**2) if method.lower() == "map" else 0.0

    def score(th):
        P = sigmoid_stable(D * a * (th - b))
        return (mu0 - th) * inv_sigma2 + D * np.sum(a * (y - P))

    def score_prime(th):
        P = sigmoid_stable(D * a * (th - b))
        return -inv_sigma2 - (D**2) * np.sum(a**2 * P * (1 - P))

    theta = float(np.clip(mu0 if theta0 is None else theta0, low, high))
    for _ in range(max_iter):
        T = score(theta)
        if abs(T) < tol:
            return theta
        Tp = score_prime(theta)
        if not np.isfinite(Tp) or Tp == 0.0:
            break
        new_theta = float(np.clip(theta - T / Tp, low, high))
        for _ in range(15):
            if abs(score(new_theta)) < abs(T):
                break
            new_theta = 0.5 * (new_theta + theta)
        theta = new_theta

    sL, sH = score(low), score(high)
    if sL * sH <= 0:
        lo, hi = low, high
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            sM  = score(mid)
            if abs(sM) < tol:
                return mid
            if sL * sM > 0:
                lo, sL = mid, sM
            else:
                hi = mid
        return 0.5 * (lo + hi)
    return high if (sL > 0 and sH > 0) else low


# =============================================================================
# FLUID BENCHMARKING
# =============================================================================

def select_mfi(theta, irt_model, used_mask, D=1.0):
    fi        = fisher_information(theta, irt_model[:, 0], irt_model[:, 1], D=D)
    fi_masked = np.where(~used_mask, fi, -np.inf)
    idx       = int(np.argmax(fi_masked))
    if not np.isfinite(fi_masked[idx]):
        raise RuntimeError("No available items.")
    return idx

def random_accuracy(lm_responses, sample_idxes):
    return float(np.mean(lm_responses[sample_idxes]))

def random_ability(lm_responses, irt_model, sample_idxes, estimation_method="map",
                   *, mu0=0.0, sigma0=1.0, theta0=None,
                   theta_range=(-4.0, 4.0), tol=1e-6, max_iter=100):
    idx = np.asarray(sample_idxes, dtype=np.int64)
    return float(ability_estimate(
        np.asarray(lm_responses)[idx],
        np.asarray(irt_model)[idx, 0],
        np.asarray(irt_model)[idx, 1],
        method=estimation_method, mu0=mu0, sigma0=sigma0,
        theta0=theta0, theta_range=theta_range, tol=tol, max_iter=max_iter,
    ))

def fluid_benchmarking(lm_responses, irt_model, start_ability, n_max, estimation_method="map"):
    n_items   = irt_model.shape[0]
    used_mask = np.zeros(n_items, dtype=bool)
    a_all     = irt_model[:, 0].astype(np.float64)
    b_all     = irt_model[:, 1].astype(np.float64)
    y_all     = np.asarray(lm_responses, dtype=np.float64)

    idx0      = select_mfi(float(start_ability), irt_model, used_mask)
    used_mask[idx0] = True
    items     = [int(idx0)]
    th        = float(ability_estimate(y_all[[idx0]], a_all[[idx0]], b_all[[idx0]],
                                       method=estimation_method, theta0=float(start_ability)))
    abilities = [th]

    while len(items) < n_max and len(items) < n_items:
        idx_next = select_mfi(float(abilities[-1]), irt_model, used_mask)
        used_mask[idx_next] = True
        items.append(int(idx_next))
        idx = np.asarray(items, dtype=np.int64)
        th  = float(ability_estimate(y_all[idx], a_all[idx], b_all[idx],
                                     method=estimation_method, theta0=float(abilities[-1])))
        abilities.append(th)

    return abilities, items


# =============================================================================
# METRICS
# =============================================================================

def empirical_accuracy_on_items(U_row, item_idxes):
    return float("nan") if not item_idxes else float(np.mean(U_row[item_idxes]))

def projected_perf_on_items(theta, irt_model, item_idxes):
    if not item_idxes:
        return float("nan")
    a, b = irt_model[item_idxes, 0], irt_model[item_idxes, 1]
    return float(np.mean(sigmoid_stable(a * (theta - b))))

def projected_fullQ(theta, irt_model, D=1.0):
    if irt_model.shape[0] == 0:
        return float("nan")
    return float(np.mean(sigmoid_stable(D * irt_model[:, 0] * (theta - irt_model[:, 1]))))

def pack_round_metrics(U_row, irt_model, items, theta):
    emp  = empirical_accuracy_on_items(U_row, items)
    proj = projected_perf_on_items(theta, irt_model, items)
    return {"empirical": emp, "projected": proj, "abs_error": abs(proj - emp)}

def pack_round_metrics_full(U_row, irt_model, items_n, theta, D=1.0):
    emp_qn    = empirical_accuracy_on_items(U_row, items_n)
    proj_qn   = projected_perf_on_items(theta, irt_model, items_n)
    proj_full = projected_fullQ(theta, irt_model, D=D)
    true_full = float(np.nanmean(U_row))
    return {
        "empirical_on_Qn":    emp_qn,
        "projected_on_Qn":    proj_qn,
        "abs_error_on_Qn":    abs(proj_qn   - emp_qn),
        "projected_on_fullQ": proj_full,
        "abs_error_fullQ":    abs(proj_full  - true_full),
    }


# =============================================================================
# BALD-FAMILY
# =============================================================================

def entropy_gaussian_1d(var):
    return 0.5 * np.log(2.0 * np.pi * np.e * max(var, 1e-12))

def sample_theta_entropy_mixture(rng, theta_hat, var, n):
    H  = entropy_gaussian_1d(var)
    k  = float(np.clip(1.0 + H, 2.0, 6.0))
    n1 = n // 2
    return np.concatenate([
        rng.normal(theta_hat, np.sqrt(var), size=n1),
        rng.normal(theta_hat, k * np.sqrt(var), size=n - n1),
    ])

def theta_var_laplace(theta_hat, a, b, *, method="map", D=1.0, sigma0=1.0,
                      min_var=2.0, max_var=6.0):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return float(np.clip((sigma0**2) if method.lower() == "map" else max_var, min_var, max_var))
    P          = sigmoid_stable(D * a * (float(theta_hat) - b))
    fisher     = float(np.sum((D**2) * a**2 * P * (1 - P)))
    prior_curv = (1.0 / max(sigma0**2, 1e-12)) if method.lower() == "map" else 0.0
    return float(np.clip(1.0 / max(fisher + prior_curv, 1e-12), min_var, max_var))

def select_bald_item(theta_samples, irt_model, used_mask):
    idxs = np.where(~used_mask)[0]
    a    = irt_model[idxs, 0][:, None]
    b    = irt_model[idxs, 1][:, None]
    th   = theta_samples[None, :]
    p    = sigmoid_stable(a * (th - b))
    bald = bernoulli_entropy(p.mean(axis=1)) - bernoulli_entropy(p).mean(axis=1)
    return int(idxs[int(np.argmax(bald))])

def select_bald_weighted_item(theta_samples, irt_model, used_mask):
    idxs  = np.where(~used_mask)[0]
    a     = irt_model[idxs, 0][:, None]
    b     = irt_model[idxs, 1][:, None]
    th    = theta_samples[None, :]
    p     = sigmoid_stable(a * (th - b))
    bald  = bernoulli_entropy(p.mean(axis=1)) - bernoulli_entropy(p).mean(axis=1)
    score = bald * np.exp(-0.5 * (b - th) ** 2).mean(axis=1)
    return int(idxs[int(np.argmax(score))])

def select_var_bald_item(theta_samples, irt_model, used_mask, alpha):
    idxs         = np.where(~used_mask)[0]
    a            = irt_model[idxs, 0][:, None]
    b            = irt_model[idxs, 1][:, None]
    th           = theta_samples[None, :]
    p            = sigmoid_stable(a * (th - b))
    bald         = bernoulli_entropy(p.mean(axis=1)) - bernoulli_entropy(p).mean(axis=1)
    sigma2       = max(float(np.var(theta_samples)), 1e-8)
    I_bar        = ((a**2) * p * (1 - p)).mean(axis=1)
    var_red_norm = (sigma2 - 1.0 / np.maximum(1.0/sigma2 + I_bar, 1e-8)) / (sigma2 + 1e-12)
    score        = bald * (1.0 + alpha * var_red_norm) * np.exp(-0.5 * (b - th)**2).mean(axis=1)
    return int(idxs[int(np.argmax(score))])

def run_fluidstyle_custom_engine(lm_responses, irt_model, n_max, method, alpha, seed,
                                  start_ability=0.0):
    rng       = np.random.default_rng(seed)
    J         = irt_model.shape[0]
    used_mask = np.zeros(J, dtype=bool)
    a_all     = irt_model[:, 0].astype(np.float64)
    b_all     = irt_model[:, 1].astype(np.float64)
    y_all     = np.asarray(lm_responses, dtype=np.float64)
    items, abilities = [], []

    theta_samples = sample_theta_entropy_mixture(rng, float(start_ability), 1.0, 64)
    if method == "bald":
        j0 = select_bald_item(theta_samples, irt_model, used_mask)
    elif method == "var_bald":
        j0 = select_var_bald_item(theta_samples, irt_model, used_mask, alpha)
    else:  # bald_weighted
        j0 = select_bald_weighted_item(theta_samples, irt_model, used_mask)
    used_mask[j0] = True
    items.append(int(j0))
    theta = float(ability_estimate(y_all[[j0]], a_all[[j0]], b_all[[j0]],
                                   theta0=float(start_ability)))
    abilities.append(theta)

    while len(items) < min(n_max, J):
        idx     = np.asarray(items, dtype=np.int64)
        var     = theta_var_laplace(theta, a_all[idx], b_all[idx])
        th_samp = sample_theta_entropy_mixture(rng, theta, float(var), 64)
        if method == "bald":
            idx_next = select_bald_item(th_samp, irt_model, used_mask)
        elif method == "var_bald":
            idx_next = select_var_bald_item(th_samp, irt_model, used_mask, alpha)
        else:  # bald_weighted
            idx_next = select_bald_weighted_item(th_samp, irt_model, used_mask)
        used_mask[idx_next] = True
        items.append(int(idx_next))
        idx   = np.asarray(items, dtype=np.int64)
        theta = float(ability_estimate(y_all[idx], a_all[idx], b_all[idx], theta0=theta))
        abilities.append(theta)

    return {"items": items, "abilities": abilities}


# =============================================================================
# RANK BALD  (shared acquisition across checkpoints, fixed IRT)
# =============================================================================

def mi_binary_vs_pair_many(z, y_s, y_t, eps=1e-12):
    z   = z.astype(np.uint8, copy=False)
    y_s = y_s.astype(np.uint8, copy=False)
    y_t = y_t.astype(np.uint8, copy=False)
    Jc, T    = y_s.shape
    y        = (y_s << 1) | y_t
    code     = (z[None, :] << 2) | y
    row_base = (np.arange(Jc, dtype=np.int64) * 8)[:, None]
    flat     = (row_base + code.astype(np.int64)).ravel()
    counts   = np.bincount(flat, minlength=Jc * 8).astype(np.float64).reshape(Jc, 2, 4)
    joint    = counts / float(T)
    pz       = joint.sum(axis=2, keepdims=True)
    py       = joint.sum(axis=1, keepdims=True)
    mi       = np.sum(joint * (np.log(joint + eps) - np.log(pz * py + eps)), axis=(1, 2))
    return np.maximum(mi, 0.0)

def select_rank_bald_item(theta_samples_by_ckpt, irt_model, used_mask, pairs, rng):
    """
    Pairwise RankBALD selection.
    theta_samples_by_ckpt : (S, T)
    """
    idxs  = np.where(~used_mask)[0]
    S, T  = theta_samples_by_ckpt.shape
    Jc    = idxs.size
    a_all = irt_model[idxs, 0].astype(np.float64)
    b_all = irt_model[idxs, 1].astype(np.float64)

    sigma_pi = float(np.median(np.std(theta_samples_by_ckpt.astype(np.float64), axis=1))) + 1e-12
    th       = theta_samples_by_ckpt[:, None, :].astype(np.float64)   # (S,1,T)
    b        = b_all[None, :, None]                                    # (1,Jc,1)
    bar_pi   = np.exp(-0.5 * ((b - th) / sigma_pi) ** 2).mean(axis=2) # (S,Jc)

    th2      = theta_samples_by_ckpt.astype(np.float64)[:, None, :]   # (S,1,T)
    a        = a_all[None, :, None]                                    # (1,Jc,1)
    p        = sigmoid_stable(a * (th2 - b))                          # (S,Jc,T)
    y        = rng.binomial(n=1, p=np.clip(p, 1e-6, 1-1e-6)).astype(np.uint8)

    z_pairs  = (theta_samples_by_ckpt[pairs[:, 0]] >
                theta_samples_by_ckpt[pairs[:, 1]]).astype(np.uint8)  # (P,T)

    dmax   = S - 1
    d      = np.abs(pairs[:, 0].astype(np.int64) - pairs[:, 1].astype(np.int64)).astype(np.float64)
    omega  = np.ones(len(pairs)) if dmax <= 1 else np.clip(1.0 - (d - 1.0) / float(max(dmax-1, 1)), 0, 1)

    scores = np.zeros(Jc)
    mass   = np.zeros(Jc)
    for k in range(len(pairs)):
        s, t   = int(pairs[k, 0]), int(pairs[k, 1])
        mi_vec = mi_binary_vs_pair_many(z_pairs[k], y[s], y[t])
        w      = bar_pi[s] * bar_pi[t] * omega[k]
        scores += w * mi_vec
        mass   += w

    scores = scores / (mass + 1e-12)
    return int(idxs[int(np.argmax(scores))])

def run_rank_bald_curve_engine(U_full, irt_model, n_max, seed, start_ability=0.0,
                                n_theta_samples=64, n_pair_samples=None,
                                map_mu0=0.0, map_sigma0=1.0,
                                theta_range=(-4.0, 4.0), min_var=2.0, max_var=6.0):
    """
    Setup-1 RankBALD: shared item acquisition across all checkpoints (S).
    U_full    : (S, J)
    irt_model : (J, 2)
    Returns   : {"items": List[int], "abilities": List[List[float]]}
                abilities[t][s] = theta_s after t+1 items selected
    """
    with _dbg_span(f"run_rank_bald_curve_engine(n_max={n_max}, seed={seed})"):
        rng       = np.random.default_rng(seed)
        S, J      = U_full.shape
        used_mask = np.zeros(J, dtype=bool)
        theta     = np.full(S, float(start_ability), dtype=np.float64)
        a_all     = irt_model[:, 0].astype(np.float64)
        b_all     = irt_model[:, 1].astype(np.float64)

        all_pairs = np.array([(s, t) for s in range(S) for t in range(s+1, S)], dtype=np.int32)

        # S=1: no checkpoint pairs exist — rank_bald requires S>=2, skip
        if len(all_pairs) == 0:
            _dbg(f"rank_bald: S={S} < 2, no checkpoint pairs available. "
                 "rank_bald requires at least 2 checkpoints/models — skipping.")
            return {"items": [], "abilities": []}

        if n_pair_samples is not None and all_pairs.shape[0] > n_pair_samples:
            all_pairs = all_pairs[rng.choice(all_pairs.shape[0], size=n_pair_samples, replace=False)]

        items: List[int] = []
        abilities_hist: List[List[float]] = []

        target = min(int(n_max), int(J))

        while len(items) < target:
            # Sample theta posteriors per checkpoint
            theta_samples = np.empty((S, n_theta_samples), dtype=np.float64)
            if len(items) == 0:
                base_var = float(np.clip(map_sigma0**2, min_var, max_var))
                for s in range(S):
                    theta_samples[s] = sample_theta_entropy_mixture(
                        rng, float(theta[s]), base_var, n_theta_samples)
            else:
                items_np = np.asarray(items, dtype=np.int64)
                for s in range(S):
                    var = theta_var_laplace(float(theta[s]), a_all[items_np], b_all[items_np],
                                            method="map", sigma0=map_sigma0,
                                            min_var=min_var, max_var=max_var)
                    theta_samples[s] = sample_theta_entropy_mixture(
                        rng, float(theta[s]), var, n_theta_samples)

            j = select_rank_bald_item(theta_samples, irt_model, used_mask, all_pairs, rng)
            used_mask[j] = True
            items.append(int(j))

            # Update theta for each checkpoint using MAP
            items_np = np.asarray(items, dtype=np.int64)
            Y_sel    = U_full[:, items_np].astype(np.float64)
            for s in range(S):
                theta[s] = float(ability_estimate(
                    Y_sel[s], a_all[items_np], b_all[items_np],
                    method="map", mu0=map_mu0, sigma0=map_sigma0,
                    theta0=float(theta[s]), theta_range=theta_range,
                ))

            abilities_hist.append([float(x) for x in theta])

            if DEBUG and (len(items) <= 3 or len(items) % max(1, n_max//10) == 0):
                _dbg(f"rank_bald step={len(items)}/{target} item={j} "
                     f"theta(min/mean/max)=({theta.min():.3f},{theta.mean():.3f},{theta.max():.3f})")

        return {"items": items, "abilities": abilities_hist}


# =============================================================================
# RUNNER
# =============================================================================

def run_all(*, data_dir, lms, benchmarks, methods, eval_sizes, alpha, device, seed, out_prefix):
    with _dbg_span("run_all"):
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

        start_abilities = compute_start_abilities(data_dir, benchmarks)

        out = {
            "lms": lms, "benchmarks": benchmarks,
            "methods": methods, "eval_sizes": eval_sizes,
            "results": {}, "summary": {},
        }

        jsonl_path = f"{out_prefix}.jsonl"
        with open(jsonl_path, "w") as jsonl_file:

            for bench in tqdm.tqdm(benchmarks, desc="Benchmarks"):
                with _dbg_span(f"bench={bench}"):
                    start_ability = start_abilities.get(bench, 0.0)
                    _dbg(f"[{bench}] start_ability={start_ability:.3f}")

                    irt_df = load_irt_model_local(data_dir, bench)
                    irt_df.index = irt_df.index.astype(str)

                    samples_dict = None

                    for lm in tqdm.tqdm(lms, desc=f"{bench} LMs", leave=False):
                        with _dbg_span(f"lm={lm} bench={bench}"):
                            eval_df = load_lm_eval_results_local(data_dir, lm, binary=True)
                            bench_t = filter_benchmark(eval_df, bench)
                            if bench_t.shape[0] == 0:
                                _dbg(f"Skipping {lm}/{bench}: no items")
                                continue

                            items         = bench_t.index.astype(str).tolist()
                            items_aligned = [it for it in items if it in set(irt_df.index)]
                            if not items_aligned:
                                _dbg(f"No aligned items for {lm}/{bench}, skipping")
                                continue

                            bench_t = bench_t.loc[items_aligned]
                            items   = items_aligned
                            U_full  = bench_t.to_numpy(dtype=float).T   # (S, J)
                            S, J    = U_full.shape
                            _dbg(f"{lm}/{bench}: S={S} J={J}")

                            cols_lower = {c.lower(): c for c in irt_df.columns}
                            a_col      = cols_lower.get("a") or cols_lower.get("disc")
                            b_col      = cols_lower.get("b") or cols_lower.get("diff")
                            if a_col is None or b_col is None:
                                raise ValueError(f"Cannot find a/b in IRT model for {bench}")
                            irt_model = np.stack([
                                irt_df.loc[items, a_col].to_numpy(dtype=float),
                                irt_df.loc[items, b_col].to_numpy(dtype=float),
                            ], axis=1)

                            if samples_dict is None:
                                random.seed(seed)
                                samples_dict = {}
                                for n_samples in N_SAMPLES_LIST:
                                    if n_samples > J:
                                        break
                                    samples_dict[n_samples] = np.array(
                                        random.sample(range(J), n_samples), dtype=int)
                                _dbg(f"[{bench}] random subsets: {list(samples_dict.keys())}")

                            n_max      = max(samples_dict.keys()) if samples_dict else 0
                            true_curve = U_full.mean(axis=1).astype(float)

                            out["results"].setdefault(lm, {})
                            out["results"][lm].setdefault(bench, {})
                            out["results"][lm][bench]["true_curve_full_accuracy"] = true_curve.tolist()

                            # JSONL lines — one per checkpoint
                            jsonl_lines = [{
                                "benchmark":    bench,
                                "lm":           lm,
                                "checkpoint":   f"ckpt_{s:03d}",
                                "full_accuracy": float(true_curve[s]),
                                "full_ability": None,
                                "abilities_fb": [],
                                "items_fb":     [],
                            } for s in range(S)]

                            # Initialise rank_bald keys so they're always present
                            if "rank_bald" in methods:
                                for s in range(S):
                                    jsonl_lines[s]["abilities_rank_bald"] = []
                                    jsonl_lines[s]["items_rank_bald"]     = []

                            for key in ("random_accuracy", "random_ability"):
                                out["results"][lm][bench].setdefault(key, {})

                            # ── Random baselines ──────────────────────────
                            for n_samples, idx in samples_dict.items():
                                items_n   = idx.tolist()
                                emp       = np.zeros(S)
                                proj      = np.full(S, np.nan)
                                err       = np.full(S, np.nan)
                                proj_full = np.full(S, np.nan)
                                err_full  = np.full(S, np.nan)

                                if "random_accuracy" in methods:
                                    for s in range(S):
                                        score = float(random_accuracy(U_full[s], idx))
                                        jsonl_lines[s][f"random_accuracy_{n_samples}"] = score
                                        emp[s] = score
                                    out["results"][lm][bench]["random_accuracy"][str(n_samples)] = {
                                        "empirical_curve":       emp.tolist(),
                                        "projected_curve":       proj.tolist(),
                                        "abs_error_curve":       err.tolist(),
                                        "abs_error_mean":        _safe_nanmean(err),
                                        "abs_error_median":      _safe_nanmedian(err),
                                        "projected_full_curve":  proj_full.tolist(),
                                        "abs_error_full_curve":  err_full.tolist(),
                                        "abs_error_full_mean":   _safe_nanmean(err_full),
                                        "abs_error_full_median": _safe_nanmedian(err_full),
                                    }

                                if "random_ability" in methods:
                                    for s in range(S):
                                        th = random_ability(U_full[s], irt_model, idx)
                                        jsonl_lines[s][f"random_ability_{n_samples}"] = float(th)
                                        m  = pack_round_metrics(U_full[s], irt_model, items_n, th)
                                        emp[s]  = m["empirical"]
                                        proj[s] = m["projected"]
                                        err[s]  = m["abs_error"]
                                    out["results"][lm][bench]["random_ability"][str(n_samples)] = {
                                        "empirical_curve":       emp.tolist(),
                                        "projected_curve":       proj.tolist(),
                                        "abs_error_curve":       err.tolist(),
                                        "abs_error_mean":        _safe_nanmean(err),
                                        "abs_error_median":      _safe_nanmedian(err),
                                        "projected_full_curve":  proj_full.tolist(),
                                        "abs_error_full_curve":  err_full.tolist(),
                                        "abs_error_full_mean":   _safe_nanmean(err_full),
                                        "abs_error_full_median": _safe_nanmedian(err_full),
                                    }

                            # Full-benchmark ability
                            full_idx = np.arange(J, dtype=int)
                            for s in range(S):
                                jsonl_lines[s]["full_ability"] = float(
                                    random_ability(U_full[s], irt_model, full_idx))

                            # ── rank_bald (shared acquisition) ────────────
                            if "rank_bald" in methods:
                                with _dbg_span(f"rank_bald bench={bench} lm={lm}"):
                                    res_rb = run_rank_bald_curve_engine(
                                        U_full=U_full,
                                        irt_model=irt_model,
                                        n_max=n_max,
                                        seed=seed,
                                        start_ability=start_ability,
                                        n_theta_samples=64,
                                    )
                                    items_rb     = res_rb["items"]
                                    abilities_rb = res_rb["abilities"]  # List[List[float]] (t, S)

                                    if not items_rb:
                                        _dbg("rank_bald returned no items (S<2) — skipping metrics")
                                    else:
                                        out["results"][lm][bench].setdefault("rank_bald", {})
                                        emp_c  = {n: np.zeros(S) for n in eval_sizes}
                                        proj_c = {n: np.zeros(S) for n in eval_sizes}
                                        err_c  = {n: np.zeros(S) for n in eval_sizes}
                                        pf_c   = {n: np.zeros(S) for n in eval_sizes}
                                        ef_c   = {n: np.zeros(S) for n in eval_sizes}

                                        for s in range(S):
                                            ab_s = [abilities_rb[t][s] for t in range(len(abilities_rb))]
                                            jsonl_lines[s]["abilities_rank_bald"] = [float(x) for x in ab_s]
                                            jsonl_lines[s]["items_rank_bald"]     = [int(i) for i in items_rb]

                                            for n in eval_sizes:
                                                n_eff   = min(n, len(items_rb))
                                                items_n = items_rb[:n_eff]
                                                theta_n = float(ab_s[n_eff-1]) if n_eff > 0 else start_ability
                                                m       = pack_round_metrics(U_full[s], irt_model, items_n, theta_n)
                                                mf      = pack_round_metrics_full(U_full[s], irt_model, items_n, theta_n)
                                                emp_c[n][s]  = m["empirical"]
                                                proj_c[n][s] = m["projected"]
                                                err_c[n][s]  = m["abs_error"]
                                                pf_c[n][s]   = mf["projected_on_fullQ"]
                                                ef_c[n][s]   = mf["abs_error_fullQ"]

                                        for n in eval_sizes:
                                            out["results"][lm][bench]["rank_bald"][str(n)] = {
                                                "empirical_curve":       emp_c[n].tolist(),
                                                "projected_curve":       proj_c[n].tolist(),
                                                "abs_error_curve":       err_c[n].tolist(),
                                                "abs_error_mean":        float(np.mean(err_c[n])),
                                                "abs_error_median":      float(np.median(err_c[n])),
                                                "projected_full_curve":  pf_c[n].tolist(),
                                                "abs_error_full_curve":  ef_c[n].tolist(),
                                                "abs_error_full_mean":   float(np.mean(pf_c[n])),
                                                "abs_error_full_median": float(np.median(pf_c[n])),
                                            }
                                        out["results"][lm][bench]["rank_bald"]["final_acquired_items"] = items_rb

                            # ── Per-checkpoint adaptive methods ───────────
                            methods_adaptive = [m for m in methods
                                                if m not in {"random_accuracy", "random_ability",
                                                             "rank_bald"}]

                            for method in tqdm.tqdm(methods_adaptive,
                                                    desc=f"{lm}/{bench} methods", leave=False):
                                out["results"][lm][bench].setdefault(method, {})
                                emp_c  = {n: np.zeros(S) for n in eval_sizes}
                                proj_c = {n: np.zeros(S) for n in eval_sizes}
                                err_c  = {n: np.zeros(S) for n in eval_sizes}
                                pf_c   = {n: np.zeros(S) for n in eval_sizes}
                                ef_c   = {n: np.zeros(S) for n in eval_sizes}
                                final_items = [None] * S

                                for s in range(S):
                                    if method == "fluid_benchmarking":
                                        abilities, items_sel = fluid_benchmarking(
                                            U_full[s], irt_model, start_ability, n_max)
                                        jsonl_lines[s]["abilities_fb"] = [float(a) for a in abilities]
                                        jsonl_lines[s]["items_fb"]     = [int(i) for i in items_sel]
                                    else:
                                        res = run_fluidstyle_custom_engine(
                                            U_full[s], irt_model, n_max,
                                            method=method, alpha=alpha,
                                            seed=seed+s, start_ability=start_ability)
                                        abilities  = res["abilities"]
                                        items_sel  = res["items"]
                                        jsonl_lines[s][f"abilities_{method}"] = [float(a) for a in abilities]
                                        jsonl_lines[s][f"items_{method}"]     = [int(i) for i in items_sel]

                                    final_items[s] = items_sel[:n_max]
                                    for n in eval_sizes:
                                        n_eff   = min(n, len(items_sel))
                                        items_n = items_sel[:n_eff]
                                        theta_n = float(abilities[n_eff-1]) if n_eff > 0 else start_ability
                                        m       = pack_round_metrics(U_full[s], irt_model, items_n, theta_n)
                                        mf      = pack_round_metrics_full(U_full[s], irt_model, items_n, theta_n)
                                        emp_c[n][s]  = m["empirical"]
                                        proj_c[n][s] = m["projected"]
                                        err_c[n][s]  = m["abs_error"]
                                        pf_c[n][s]   = mf["projected_on_fullQ"]
                                        ef_c[n][s]   = mf["abs_error_fullQ"]

                                for n in eval_sizes:
                                    out["results"][lm][bench][method][str(n)] = {
                                        "empirical_curve":       emp_c[n].tolist(),
                                        "projected_curve":       proj_c[n].tolist(),
                                        "abs_error_curve":       err_c[n].tolist(),
                                        "abs_error_mean":        float(np.mean(err_c[n])),
                                        "abs_error_median":      float(np.median(err_c[n])),
                                        "projected_full_curve":  pf_c[n].tolist(),
                                        "abs_error_full_curve":  ef_c[n].tolist(),
                                        "abs_error_full_mean":   float(np.mean(ef_c[n])),
                                        "abs_error_full_median": float(np.median(ef_c[n])),
                                    }
                                out["results"][lm][bench][method]["final_acquired_items_per_checkpoint"] = final_items

                            for s in range(S):
                                jsonl_file.write(json.dumps(jsonl_lines[s], separators=(",", ":")) + "\n")

        out_path = f"{out_prefix}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        _dbg(f"JSON  -> {out_path}")
        _dbg(f"JSONL -> {jsonl_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   type=str, default="./data")
    ap.add_argument("--lms",        type=str, default=",".join(LMS))
    ap.add_argument("--benchmarks", type=str, default=",".join(BENCHMARKS))
    ap.add_argument("--methods",    type=str,
                    default="random_ability,fluid_benchmarking,"
                            "rank_bald,bald,var_bald,bald_weighted")
    ap.add_argument("--eval_sizes", type=str,   default="10,50,100,500")
    ap.add_argument("--alpha",      type=float, default=1.0)
    ap.add_argument("--device",     type=str,   default="cpu")
    ap.add_argument("--seed",       type=int,   default=0)
    ap.add_argument("--out_prefix", type=str,   default="outputs/afribench")
    ap.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable verbose debug output.",
    )
    args = ap.parse_args()

    global DEBUG
    DEBUG = args.debug

    run_all(
        data_dir   = args.data_dir,
        lms        = [x.strip() for x in args.lms.split(",")        if x.strip()],
        benchmarks = [x.strip() for x in args.benchmarks.split(",") if x.strip()],
        methods    = [x.strip() for x in args.methods.split(",")    if x.strip()],
        eval_sizes = [int(x)    for x in args.eval_sizes.split(",") if x.strip()],
        alpha      = args.alpha,
        device     = args.device,
        seed       = args.seed,
        out_prefix = args.out_prefix,
    )

if __name__ == "__main__":
    main()