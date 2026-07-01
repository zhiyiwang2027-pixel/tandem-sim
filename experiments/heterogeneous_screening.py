from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.quick_heterogeneous_comparison import (
    CONFIG_ORDER,
    make_channel_configs_from_weights,
    random_weight_profile,
    run_policy_mean,
)
from tandem.diagnostics import estimate_iso1_voq_arrival_rates
from tandem.policies import build_experiment_v3, build_iso1_iso2_lambda_policy_v3
from tandem.rate_optimizer import iso2_lambda_params_v3


DEFAULT_OUTPUT_CSV = Path("results/heterogeneous_screening.csv")
POLICY_ORDER = (
    "Joint FGMW",
    "iso1 + iso2",
    "iso1 + iso2-lambda",
    "Downstream-Aware MW",
    "Greedy",
    "Uniform",
)
P_LEVELS = (0.95, 0.85, 0.70, 0.55, 0.40, 0.28, 0.18, 0.10)
WEIGHT_RATIOS = (10.0, 50.0, 200.0)
L_VALUES = (3, 5)
MU_VALUES = (0.08, 0.12, 0.18, 0.25, 0.4, 0.7)
EVAL_SEEDS = (0, 1, 2)
PILOT_SEEDS = (100, 101, 102)


def kkt_regime(lambda_link: float, nu_edge: float, tol: float = 1e-6) -> str:
    link = float(lambda_link) > tol
    edge = float(nu_edge) > tol
    if link and edge:
        return "both"
    if link:
        return "link-only"
    if edge:
        return "edge-only"
    return "slack"


def screening_networks(
    *,
    N: int,
    weight_ratio: float,
    weight_seed: int = 7,
    channel_seed: int = 11,
    p_levels: Sequence[float] = P_LEVELS,
) -> Dict[str, object]:
    w = random_weight_profile(N, seed=weight_seed, ratio=weight_ratio)
    channels = make_channel_configs_from_weights(w, p_levels, seed=channel_seed)
    return {"w": w, "channels": channels}


def iter_screening_settings(
    *,
    N: int = 8,
    weight_ratios: Sequence[float] = WEIGHT_RATIOS,
    alignments: Sequence[str] = CONFIG_ORDER,
    L_values: Sequence[int] = L_VALUES,
    mu_values: Sequence[float] = MU_VALUES,
    weight_seed: int = 7,
    channel_seed: int = 11,
    p_levels: Sequence[float] = P_LEVELS,
) -> Iterator[Dict[str, object]]:
    for weight_ratio in weight_ratios:
        network = screening_networks(
            N=N,
            weight_ratio=weight_ratio,
            weight_seed=weight_seed,
            channel_seed=channel_seed,
            p_levels=p_levels,
        )
        for alignment in alignments:
            p = network["channels"][alignment]
            for L in L_values:
                for mu in mu_values:
                    yield {
                        "N": int(N),
                        "L": int(L),
                        "mu": float(mu),
                        "weight_ratio": float(weight_ratio),
                        "alignment": alignment,
                        "weight_seed": int(weight_seed),
                        "channel_seed": int(channel_seed),
                        "w": np.asarray(network["w"], float),
                        "p": np.asarray(p, float),
                    }


def _summaries(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, float)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "sum": float(arr.sum()),
    }


def run_setting(
    setting: Mapping[str, object],
    *,
    seeds: Iterable[int],
    pilot_seeds: Iterable[int],
    K: int,
    warmup: int,
    K_pilot: int,
    warmup_pilot: int,
    screening_mode: str,
) -> list[Dict[str, object]]:
    seeds = tuple(seeds)
    pilot_seeds = tuple(pilot_seeds)
    N = int(setting["N"])
    L = int(setting["L"])
    mu = float(setting["mu"])
    p = np.asarray(setting["p"], float)
    w = np.asarray(setting["w"], float)
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

    joint_dual = params["joint"]["dual"]
    lambda_link = float(joint_dual["lambda_link"])
    nu_edge = float(joint_dual["nu_edge"])
    qd_relaxed = _summaries(params["iso2"]["qd"])
    qd_lambda = _summaries(iso2_lambda_params_v3(N, L, p, mu, w, lambda_hat)["qd"])
    lambda_summary = _summaries(lambda_hat)

    rows = []
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
        rows.append({
            "screening_mode": screening_mode,
            "policy": policy_name,
            "weighted_dest_aoi": summary["weighted_dest_aoi"],
            "weighted_dest_aoi_se": summary["weighted_dest_aoi_se"],
            "corr_w_c": corr_wc,
            "weight_ratio": float(setting["weight_ratio"]),
            "alignment": setting["alignment"],
            "L": L,
            "mu": mu,
            "lambda_link": lambda_link,
            "nu_edge": nu_edge,
            "kkt_regime": kkt_regime(lambda_link, nu_edge),
            "stage1_used_frac": summary["stage1_used_frac"],
            "stage2_used_frac": summary["stage2_used_frac"],
            "stage2_idle_empty_frac": summary["stage2_idle_empty_frac"],
            "total_VOQ_arrival_rate": summary["total_VOQ_arrival_rate"],
            "total_delivery_rate": summary["total_delivery_rate"],
            "total_overwrite_rate": summary["total_overwrite_rate"],
            "lambda_hat_min": lambda_summary["min"],
            "lambda_hat_max": lambda_summary["max"],
            "lambda_hat_sum": lambda_summary["sum"],
            "qd_iso2_relaxed_sum": qd_relaxed["sum"],
            "qd_iso2_lambda_sum": qd_lambda["sum"],
            "N": N,
            "weight_seed": int(setting["weight_seed"]),
            "channel_seed": int(setting["channel_seed"]),
            "K": int(K),
            "warmup": int(warmup),
            "num_seeds": int(len(seeds)),
            "pilot_K": int(K_pilot),
            "pilot_warmup": int(warmup_pilot),
            "num_pilot_seeds": int(len(pilot_seeds)),
        })

    iso = next(row["weighted_dest_aoi"] for row in rows if row["policy"] == "iso1 + iso2")
    iso_lambda = next(row["weighted_dest_aoi"] for row in rows if row["policy"] == "iso1 + iso2-lambda")
    for row in rows:
        row["gap_vs_iso_pct"] = 100.0 * (row["weighted_dest_aoi"] / iso - 1.0)
        row["gap_vs_iso_lambda_pct"] = 100.0 * (row["weighted_dest_aoi"] / iso_lambda - 1.0)
    return rows


def run_heterogeneous_screening(
    *,
    mode: str,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    N: int = 8,
    weight_ratios: Sequence[float] = WEIGHT_RATIOS,
    alignments: Sequence[str] = CONFIG_ORDER,
    L_values: Sequence[int] = L_VALUES,
    mu_values: Sequence[float] = MU_VALUES,
    seeds: Sequence[int] = EVAL_SEEDS,
    pilot_seeds: Sequence[int] = PILOT_SEEDS,
    p_levels: Sequence[float] = P_LEVELS,
    K: int = 10_000,
    warmup: int = 1_000,
    K_pilot: int = 20_000,
    warmup_pilot: int = 2_000,
) -> pd.DataFrame:
    if mode == "quick":
        weight_ratios = tuple(weight_ratios[:1])
        alignments = ("aligned", "conflict")
        L_values = tuple(L_values[:1])
        mu_values = (0.12, 0.4)
        seeds = tuple(seeds[:1])
        pilot_seeds = tuple(pilot_seeds[:1])
        K, warmup = 1_000, 100
        K_pilot, warmup_pilot = 1_500, 150
    elif mode != "full":
        raise ValueError("mode must be 'quick' or 'full'.")

    seeds = tuple(seeds)
    pilot_seeds = tuple(pilot_seeds)
    rows = []
    for setting in iter_screening_settings(
        N=N,
        weight_ratios=weight_ratios,
        alignments=alignments,
        L_values=L_values,
        mu_values=mu_values,
        p_levels=p_levels,
    ):
        rows.extend(
            run_setting(
                setting,
                seeds=seeds,
                pilot_seeds=pilot_seeds,
                K=K,
                warmup=warmup,
                K_pilot=K_pilot,
                warmup_pilot=warmup_pilot,
                screening_mode=mode,
            )
        )

    df = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heterogeneous tandem screening.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Run a small fast smoke screening.")
    group.add_argument("--full", action="store_true", help="Run the full screening grid.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_CSV), help="CSV output path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = "full" if args.full else "quick"
    df = run_heterogeneous_screening(mode=mode, output_csv=args.output)
    print(f"mode={mode}")
    print(f"rows={len(df)}")
    print(f"wrote={args.output}")
    summary = (
        df[df["policy"].isin(["Joint FGMW", "iso1 + iso2-lambda", "Downstream-Aware MW", "Greedy"])]
        .groupby(["policy", "alignment"])["gap_vs_iso_lambda_pct"]
        .mean()
        .round(3)
    )
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
