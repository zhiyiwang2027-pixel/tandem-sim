from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tandem.diagnostics import estimate_iso1_voq_arrival_rates
from tandem.policies import (
    build_experiment_v3,
    build_iso1_iso2_lambda_policy_v3,
    build_srp_iso_policy_v3,
    build_srp_tandem_lb_policy_v3,
)
from tandem.rate_optimizer import iso2_lambda_params_v3
from tandem.simulator import TandemAoISimulatorV3


DEFAULT_OUTPUT_CSV = Path("results/deterministic_heterogeneous_comparison.csv")
RAW_WEIGHTS = np.array([8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625], dtype=float)
P_GOOD_TO_BAD = np.array([0.95, 0.85, 0.70, 0.55, 0.40, 0.28, 0.18, 0.10], dtype=float)
NETWORK_ORDER = ("det_aligned", "det_conflict")
L_VALUES = (3, 5)
MU_VALUES = (0.12, 0.25, 0.40)
POLICY_ORDER = (
    "Joint FGMW",
    "iso1 + iso2-lambda",
    "Downstream-Aware MW",
    "Greedy",
    "SRP-iso",
    "SRP-tandem-LB",
)


def deterministic_networks() -> Dict[str, Dict[str, np.ndarray]]:
    """Return the two deterministic raw-weight heterogeneous networks."""
    return {
        "det_aligned": {
            "w": RAW_WEIGHTS.copy(),
            "p": P_GOOD_TO_BAD.copy(),
        },
        "det_conflict": {
            "w": RAW_WEIGHTS.copy(),
            "p": P_GOOD_TO_BAD[::-1].copy(),
        },
    }


def _json_array(values: Sequence[float]) -> str:
    return json.dumps([float(value) for value in values])


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


def _standard_error(values: Sequence[float]) -> float:
    values = np.asarray(values, float)
    if values.size <= 1:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(values.size))


def _run_policy_mean_with_debt(
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
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one seed.")

    rows = []
    debt_rows = []
    for seed in seeds:
        result = TandemAoISimulatorV3(N, L, p, mu, w, seed=int(seed)).run(
            policy,
            K=int(K),
            warmup=int(warmup),
        )
        rows.append(_metric_row(result))
        diag = policy.diagnostics() if hasattr(policy, "diagnostics") else {}
        bs = diag.get("bs", {})
        es = diag.get("es", {})
        debt_rows.append({
            "final_upstream_debt": float(bs.get("final_U_sum", np.nan)),
            "avg_upstream_debt": float(bs.get("avg_U_sum", np.nan)),
            "final_downstream_debt": float(es.get("final_D_sum", np.nan)),
            "avg_downstream_debt": float(es.get("avg_D_sum", np.nan)),
        })

    keys = rows[0].keys()
    summary = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    summary["weighted_dest_aoi_se"] = _standard_error([row["weighted_dest_aoi"] for row in rows])
    for key in debt_rows[0]:
        vals = np.asarray([row[key] for row in debt_rows], float)
        summary[key] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan")
    return summary


def _build_policies_with_lambda(N, L, p, mu, w, *, pilot_seeds, K_pilot, warmup_pilot):
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
    policies = dict(policies)
    policies["iso1 + iso2-lambda"] = build_iso1_iso2_lambda_policy_v3(
        N, L, p, mu, w, lambda_est["lambda_hat"]
    )
    policies["SRP-iso"] = build_srp_iso_policy_v3(
        N, L, p, mu, w, lambda_est["lambda_hat"]
    )
    policies["SRP-tandem-LB"] = build_srp_tandem_lb_policy_v3(
        N, L, p, mu, w, joint_params=params["joint"]
    )
    return policies, params, lambda_est


def run_setting(
    *,
    network: str,
    L: int,
    mu: float,
    seeds: Sequence[int],
    K: int,
    warmup: int,
    pilot_seeds: Sequence[int],
    K_pilot: int,
    warmup_pilot: int,
    mode: str,
) -> list[Dict[str, object]]:
    networks = deterministic_networks()
    if network not in networks:
        raise ValueError(f"Unknown deterministic network {network!r}.")

    w = networks[network]["w"]
    p = networks[network]["p"]
    N = int(w.size)
    L = int(L)
    mu = float(mu)
    c = (L - 1.0) + 1.0 / p
    corr_wc = float(np.corrcoef(w, c)[0, 1])
    policies, params, lambda_est = _build_policies_with_lambda(
        N,
        L,
        p,
        mu,
        w,
        pilot_seeds=pilot_seeds,
        K_pilot=K_pilot,
        warmup_pilot=warmup_pilot,
    )
    iso2_lambda = iso2_lambda_params_v3(N, L, p, mu, w, lambda_est["lambda_hat"])

    rows = []
    for policy_name in POLICY_ORDER:
        summary = _run_policy_mean_with_debt(
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
            "mode": mode,
            "network": network,
            "policy": policy_name,
            "L": L,
            "mu": mu,
            "N": N,
            "w": _json_array(w),
            "p": _json_array(p),
            "corr_w_c": corr_wc,
            "K": int(K),
            "warmup": int(warmup),
            "num_seeds": int(len(seeds)),
            "pilot_K": int(K_pilot),
            "pilot_warmup": int(warmup_pilot),
            "num_pilot_seeds": int(len(pilot_seeds)),
            "lambda_hat_sum": float(lambda_est["sum_lambda"]),
            "qd_iso2_lambda_sum": float(iso2_lambda["qd"].sum()),
            "lambda_link": float(params["joint"]["dual"]["lambda_link"]),
            "nu_edge": float(params["joint"]["dual"]["nu_edge"]),
            **summary,
        })

    iso_lambda = next(row["weighted_dest_aoi"] for row in rows if row["policy"] == "iso1 + iso2-lambda")
    for row in rows:
        row["gap_vs_iso_lambda_pct"] = 100.0 * (row["weighted_dest_aoi"] / iso_lambda - 1.0)
    return rows


def run_deterministic_heterogeneous_comparison(
    *,
    mode: str,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    L_values: Sequence[int] = L_VALUES,
    mu_values: Sequence[float] = MU_VALUES,
    networks: Sequence[str] = NETWORK_ORDER,
    seeds: Sequence[int] = (0, 1, 2),
    pilot_seeds: Sequence[int] = (100, 101, 102),
    K: int = 20_000,
    warmup: int = 2_000,
    K_pilot: int = 50_000,
    warmup_pilot: int = 5_000,
) -> pd.DataFrame:
    if mode == "quick":
        seeds = tuple(seeds[:1])
        pilot_seeds = tuple(pilot_seeds[:1])
        K, warmup = 1_500, 150
        K_pilot, warmup_pilot = 2_000, 200
    elif mode != "full":
        raise ValueError("mode must be 'quick' or 'full'.")

    rows = []
    for network in networks:
        for L in L_values:
            for mu in mu_values:
                rows.extend(
                    run_setting(
                        network=network,
                        L=int(L),
                        mu=float(mu),
                        seeds=tuple(seeds),
                        K=K,
                        warmup=warmup,
                        pilot_seeds=tuple(pilot_seeds),
                        K_pilot=K_pilot,
                        warmup_pilot=warmup_pilot,
                        mode=mode,
                    )
                )

    df = pd.DataFrame(rows)
    duplicated = df.duplicated(["network", "L", "mu", "policy"])
    if bool(duplicated.any()):
        raise RuntimeError("Deterministic comparison produced duplicate setting-policy rows.")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic raw-weight heterogeneous comparison.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Run a fast smoke comparison.")
    group.add_argument("--full", action="store_true", help="Run the longer comparison.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_CSV), help="CSV output path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = "full" if args.full else "quick"
    df = run_deterministic_heterogeneous_comparison(mode=mode, output_csv=args.output)
    print(f"mode={mode}")
    print(f"rows={len(df)}")
    print(f"wrote={args.output}")
    summary = (
        df[df["policy"].isin(POLICY_ORDER)]
        .groupby(["network", "policy"])["gap_vs_iso_lambda_pct"]
        .mean()
        .round(3)
    )
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
