from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tandem.diagnostics import estimate_iso1_voq_arrival_rates
from tandem.policies import build_experiment_v3, build_iso1_iso2_lambda_policy_v3
from tandem.rate_optimizer import iso2_lambda_params_v3
from tandem.simulator import TandemAoISimulatorV3
from experiments.deterministic_heterogeneous_comparison import (
    NETWORK_ORDER as DETERMINISTIC_NETWORK_ORDER,
    RAW_WEIGHTS,
    P_GOOD_TO_BAD,
    deterministic_networks,
)


POLICY_ORDER = (
    "Joint FGMW",
    "iso1 + iso2",
    "iso1 + iso2-lambda",
    "Downstream-Aware MW",
    "Greedy",
    "Uniform",
)
CONFIG_ORDER = ("aligned", "neutral", "conflict")
DEFAULT_OUTPUT_CSV = Path("results/quick_heterogeneous_comparison.csv")


def random_weight_profile(N: int, seed: int = 7, ratio: float = 100.0) -> np.ndarray:
    """Draw log-uniform weights on [1, ratio] and normalize by their mean."""
    if N <= 0:
        raise ValueError("N must be positive.")
    if ratio <= 1.0:
        raise ValueError("ratio must be greater than 1.")
    rng = np.random.default_rng(seed)
    w = np.exp(rng.uniform(0.0, np.log(float(ratio)), size=int(N)))
    return w / w.mean()


def make_channel_configs_from_weights(
    w: Sequence[float],
    p_levels: Sequence[float],
    seed: int = 11,
) -> Dict[str, np.ndarray]:
    """Pair channel reliabilities with weights in aligned, neutral, and conflict regimes."""
    w = np.asarray(w, float)
    p_levels = np.asarray(p_levels, float)
    if w.ndim != 1 or p_levels.ndim != 1 or w.size != p_levels.size:
        raise ValueError("w and p_levels must be one-dimensional arrays of the same length.")
    if np.any((p_levels <= 0.0) | (p_levels > 1.0)):
        raise ValueError("All p_levels must lie in (0, 1].")

    high_weight_order = np.argsort(-w)
    p_desc = np.sort(p_levels)[::-1]
    p_asc = np.sort(p_levels)

    aligned = np.empty_like(p_desc)
    aligned[high_weight_order] = p_desc

    conflict = np.empty_like(p_asc)
    conflict[high_weight_order] = p_asc

    rng = np.random.default_rng(seed)
    neutral = rng.permutation(p_levels)

    return {
        "aligned": aligned,
        "neutral": neutral,
        "conflict": conflict,
    }


def _metric_row(result: Mapping[str, object]) -> Dict[str, float]:
    return {
        "weighted_dest_aoi": float(result["weighted_dest_aoi"]),
        "stage1_used_frac": float(result["s1_used_frac"]),
        "stage2_used_frac": float(result["s2_used_frac"]),
        "stage2_idle_empty_frac": float(result["s2_idle_empty_frac"]),
        "total_VOQ_arrival_rate": float(np.asarray(result["VOQ_arrival_rate"]).sum()),
        "total_delivery_rate": float(np.asarray(result["stage2_delivery_rate"]).sum()),
        "total_overwrite_rate": float(np.asarray(result["overwrite_rate"]).sum()),
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.asarray(values, float).mean())


def _standard_error(values: Sequence[float]) -> float:
    values = np.asarray(values, float)
    if values.size <= 1:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(values.size))


def run_policy_mean(
    policy,
    *,
    N: int,
    L: int,
    p: Sequence[float],
    mu: float,
    w: Sequence[float],
    seeds: Iterable[int],
    K: int,
    warmup: int,
) -> Dict[str, float]:
    """Run one policy over paired seeds and summarize post-warmup diagnostics."""
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one seed.")
    per_seed = []
    for seed in seeds:
        result = TandemAoISimulatorV3(N, L, p, mu, w, seed=int(seed)).run(
            policy,
            K=int(K),
            warmup=int(warmup),
        )
        per_seed.append(_metric_row(result))

    keys = per_seed[0].keys()
    summary = {key: _mean([row[key] for row in per_seed]) for key in keys}
    summary["weighted_dest_aoi_se"] = _standard_error(
        [row["weighted_dest_aoi"] for row in per_seed]
    )
    return summary


def _profile_dataframe(w: np.ndarray, channel_configs: Mapping[str, np.ndarray], L: int) -> pd.DataFrame:
    rows = []
    for i, weight in enumerate(w):
        row = {"source": i, "w": float(weight)}
        for name in CONFIG_ORDER:
            p_i = float(channel_configs[name][i])
            row[f"p_{name}"] = p_i
            row[f"c_{name}"] = float((L - 1.0) + 1.0 / p_i)
        rows.append(row)
    return pd.DataFrame(rows)


def _deterministic_profile_dataframe(networks: Mapping[str, Mapping[str, np.ndarray]], L: int) -> pd.DataFrame:
    w = RAW_WEIGHTS
    rows = []
    for i, weight in enumerate(w):
        row = {"source": i, "w": float(weight)}
        for name in DETERMINISTIC_NETWORK_ORDER:
            p_i = float(networks[name]["p"][i])
            row[f"p_{name}"] = p_i
            row[f"c_{name}"] = float((L - 1.0) + 1.0 / p_i)
        rows.append(row)
    return pd.DataFrame(rows)


def quick_random_heterogeneous_comparison(
    *,
    N: int = 8,
    L: int = 3,
    mu: float = 0.25,
    weight_seed: int = 7,
    channel_seed: int = 11,
    seeds: Sequence[int] = (0, 1, 2),
    pilot_seeds: Sequence[int] = (100, 101, 102, 103, 104),
    K: int = 20_000,
    warmup: int = 2_000,
    K_pilot: int = 50_000,
    warmup_pilot: int = 5_000,
    ratio: float = 100.0,
    p_levels: Sequence[float] = (0.95, 0.85, 0.70, 0.55, 0.40, 0.28, 0.18, 0.10),
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
) -> Dict[str, object]:
    """Run the deterministic aggressive raw-weight heterogeneous quick comparison.

    The function name is retained for notebook/backward compatibility, but the
    default quick demo no longer samples random log-uniform normalized weights.
    It uses the canonical deterministic raw weights and the det_aligned /
    det_conflict channel pairings.
    """
    seeds = tuple(seeds)
    pilot_seeds = tuple(pilot_seeds)
    if int(N) != RAW_WEIGHTS.size:
        raise ValueError("The deterministic quick comparison uses N=8.")
    w = RAW_WEIGHTS.copy()
    networks = deterministic_networks()

    rows = []
    for config_name in DETERMINISTIC_NETWORK_ORDER:
        p = networks[config_name]["p"]
        c = (L - 1.0) + 1.0 / p
        corr_wc = float(np.corrcoef(w, c)[0, 1])
        policies, params = build_experiment_v3(N, L, p, mu, w)
        lambda_est = estimate_iso1_voq_arrival_rates(
            N,
            L,
            p,
            mu,
            w,
            pilot_seeds=pilot_seeds,
            K_pilot=K_pilot,
            warmup_pilot=warmup_pilot,
        )
        lambda_hat = lambda_est["lambda_hat"]
        policies = dict(policies)
        policies["iso1 + iso2-lambda"] = build_iso1_iso2_lambda_policy_v3(
            N, L, p, mu, w, lambda_hat
        )
        iso2_lambda = iso2_lambda_params_v3(N, L, p, mu, w, lambda_hat)
        joint_dual = params["joint"]["dual"]
        qd_relaxed = params["iso2"]["qd"]
        qd_lambda = iso2_lambda["qd"]

        config_rows = []
        for policy_name in POLICY_ORDER:
            summary = run_policy_mean(
                policies[policy_name],
                N=N,
                L=L,
                p=p,
                mu=mu,
                w=w,
                seeds=seeds,
                K=K,
                warmup=warmup,
            )
            row = {
                "config": config_name,
                "network": config_name,
                "policy": policy_name,
                "N": int(N),
                "L": int(L),
                "mu": float(mu),
                "weight_seed": np.nan,
                "channel_seed": np.nan,
                "K": int(K),
                "warmup": int(warmup),
                "num_seeds": int(len(seeds)),
                "pilot_K": int(K_pilot),
                "pilot_warmup": int(warmup_pilot),
                "num_pilot_seeds": int(len(pilot_seeds)),
                "w": list(map(float, w)),
                "p": list(map(float, p)),
                "corr_w_c": corr_wc,
                "lambda_link": float(joint_dual["lambda_link"]),
                "nu_edge": float(joint_dual["nu_edge"]),
                "lambda_hat_min": float(lambda_est["min_lambda"]),
                "lambda_hat_max": float(lambda_est["max_lambda"]),
                "lambda_hat_sum": float(lambda_est["sum_lambda"]),
                "qd_iso2_relaxed_min": float(qd_relaxed.min()),
                "qd_iso2_relaxed_max": float(qd_relaxed.max()),
                "qd_iso2_relaxed_sum": float(qd_relaxed.sum()),
                "qd_iso2_lambda_min": float(qd_lambda.min()),
                "qd_iso2_lambda_max": float(qd_lambda.max()),
                "qd_iso2_lambda_sum": float(qd_lambda.sum()),
                **summary,
            }
            config_rows.append(row)

        iso_mean = next(
            row["weighted_dest_aoi"]
            for row in config_rows
            if row["policy"] == "iso1 + iso2"
        )
        iso_lambda_mean = next(
            row["weighted_dest_aoi"]
            for row in config_rows
            if row["policy"] == "iso1 + iso2-lambda"
        )
        for row in config_rows:
            row["gap_vs_iso_pct"] = 100.0 * (row["weighted_dest_aoi"] / iso_mean - 1.0)
            row["gap_vs_iso_lambda_pct"] = 100.0 * (
                row["weighted_dest_aoi"] / iso_lambda_mean - 1.0
            )
            rows.append(row)

    results = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    return {
        "results": results,
        "profiles": _deterministic_profile_dataframe(networks, L),
        "weights": w,
        "channels": {name: networks[name]["p"].copy() for name in DETERMINISTIC_NETWORK_ORDER},
        "pilot_seeds": pilot_seeds,
        "csv_path": output_path,
    }


if __name__ == "__main__":
    out = quick_random_heterogeneous_comparison()
    print(out["results"].round(4).to_string(index=False))
    print(f"\nWrote {out['csv_path']}")
