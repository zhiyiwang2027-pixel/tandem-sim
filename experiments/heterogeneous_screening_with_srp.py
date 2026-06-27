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

from experiments.heterogeneous_screening import (
    EVAL_SEEDS,
    L_VALUES,
    MU_VALUES,
    PILOT_SEEDS,
    P_LEVELS,
    WEIGHT_RATIOS,
    iter_screening_settings,
    kkt_regime,
)
from experiments.quick_heterogeneous_comparison import CONFIG_ORDER, run_policy_mean
from tandem.diagnostics import estimate_iso1_voq_arrival_rates
from tandem.policies import (
    build_experiment_v3,
    build_iso1_iso2_lambda_policy_v3,
    build_srp_iso_policy_v3,
    build_srp_tandem_lb_policy_v3,
)
from tandem.rate_optimizer import (
    iso1_srp_params_v3,
    iso2_lambda_params_v3,
    rate_to_srp_alpha,
    rate_to_srp_beta,
)


DEFAULT_OUTPUT_CSV = Path("results/heterogeneous_screening_with_srp.csv")
POLICY_ORDER_WITH_SRP = (
    "Joint FGMW",
    "iso1 + iso2",
    "iso1 + iso2-lambda",
    "SRP-iso",
    "SRP-tandem-LB",
    "Greedy",
    "Uniform",
)


def _summaries(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, float)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "sum": float(arr.sum()),
    }


def _blank_srp_metadata() -> Dict[str, object]:
    return {
        "srp_alpha_sum": np.nan,
        "srp_beta_sum": np.nan,
        "srp_stage1_target_sum": np.nan,
        "srp_stage2_target_sum": np.nan,
        "srp_iso1_nu": np.nan,
        "srp_iso1_link_binds": np.nan,
    }


def _srp_metadata(
    *,
    alpha: Sequence[float],
    beta: Sequence[float],
    stage1_target: Sequence[float],
    stage2_target: Sequence[float],
    iso1_nu: float = np.nan,
    iso1_link_binds: object = np.nan,
) -> Dict[str, object]:
    return {
        "srp_alpha_sum": float(np.asarray(alpha, float).sum()),
        "srp_beta_sum": float(np.asarray(beta, float).sum()),
        "srp_stage1_target_sum": float(np.asarray(stage1_target, float).sum()),
        "srp_stage2_target_sum": float(np.asarray(stage2_target, float).sum()),
        "srp_iso1_nu": float(iso1_nu) if np.isfinite(iso1_nu) else np.nan,
        "srp_iso1_link_binds": iso1_link_binds,
    }


def iter_srp_screening_settings(
    *,
    N: int = 8,
    weight_ratios: Sequence[float] = WEIGHT_RATIOS,
    alignments: Sequence[str] = CONFIG_ORDER,
    L_values: Sequence[int] = L_VALUES,
    mu_values: Sequence[float] = MU_VALUES,
    p_levels: Sequence[float] = P_LEVELS,
) -> Iterator[Dict[str, object]]:
    yield from iter_screening_settings(
        N=N,
        weight_ratios=weight_ratios,
        alignments=alignments,
        L_values=L_values,
        mu_values=mu_values,
        p_levels=p_levels,
    )


def run_setting_with_srp(
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
    iso2_lambda = iso2_lambda_params_v3(N, L, p, mu, w, lambda_hat)
    iso1_srp = iso1_srp_params_v3(N, L, p, w)

    policies = dict(policies)
    policies["iso1 + iso2-lambda"] = build_iso1_iso2_lambda_policy_v3(
        N, L, p, mu, w, lambda_hat
    )
    policies["SRP-iso"] = build_srp_iso_policy_v3(N, L, p, mu, w, lambda_hat)
    policies["SRP-tandem-LB"] = build_srp_tandem_lb_policy_v3(
        N, L, p, mu, w, joint_params=params["joint"]
    )

    qd_joint = params["joint"]["qd"]
    alpha_joint = rate_to_srp_alpha(qd_joint, p, L)
    beta_joint = rate_to_srp_beta(qd_joint, mu)
    beta_iso = rate_to_srp_beta(iso2_lambda["qd"], mu)

    srp_metadata = {
        "SRP-iso": _srp_metadata(
            alpha=iso1_srp["alpha_srp_iso1"],
            beta=beta_iso,
            stage1_target=iso1_srp["q_srp_iso1"],
            stage2_target=iso2_lambda["qd"],
            iso1_nu=float(iso1_srp["nu_iso1_srp"]),
            iso1_link_binds=bool(iso1_srp["link_binds"]),
        ),
        "SRP-tandem-LB": _srp_metadata(
            alpha=alpha_joint,
            beta=beta_joint,
            stage1_target=qd_joint,
            stage2_target=qd_joint,
        ),
    }

    joint_dual = params["joint"]["dual"]
    lambda_link = float(joint_dual["lambda_link"])
    nu_edge = float(joint_dual["nu_edge"])
    qd_relaxed = _summaries(params["iso2"]["qd"])
    qd_lambda = _summaries(iso2_lambda["qd"])
    lambda_summary = _summaries(lambda_hat)

    rows = []
    for policy_name in POLICY_ORDER_WITH_SRP:
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
        }
        row.update(_blank_srp_metadata())
        row.update(srp_metadata.get(policy_name, {}))
        rows.append(row)

    iso = next(row["weighted_dest_aoi"] for row in rows if row["policy"] == "iso1 + iso2")
    iso_lambda = next(row["weighted_dest_aoi"] for row in rows if row["policy"] == "iso1 + iso2-lambda")
    for row in rows:
        row["gap_vs_iso_pct"] = 100.0 * (row["weighted_dest_aoi"] / iso - 1.0)
        row["gap_vs_iso_lambda_pct"] = 100.0 * (row["weighted_dest_aoi"] / iso_lambda - 1.0)
    return rows


def run_heterogeneous_screening_with_srp(
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
    for setting in iter_srp_screening_settings(
        N=N,
        weight_ratios=weight_ratios,
        alignments=alignments,
        L_values=L_values,
        mu_values=mu_values,
        p_levels=p_levels,
    ):
        rows.extend(
            run_setting_with_srp(
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
    parser = argparse.ArgumentParser(description="Run heterogeneous tandem screening with SRP baselines.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Run a small fast smoke screening.")
    group.add_argument("--full", action="store_true", help="Run the full screening grid.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_CSV), help="CSV output path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = "full" if args.full else "quick"
    df = run_heterogeneous_screening_with_srp(mode=mode, output_csv=args.output)
    print(f"mode={mode}")
    print(f"rows={len(df)}")
    print(f"wrote={args.output}")
    summary = (
        df[df["policy"].isin(["Joint FGMW", "SRP-iso", "SRP-tandem-LB", "Greedy"])]
        .groupby(["policy", "alignment"])["gap_vs_iso_lambda_pct"]
        .mean()
        .round(3)
    )
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
