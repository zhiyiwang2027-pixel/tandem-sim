from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def build_run_results(sim, K: int, warmup: int) -> Dict[str, object]:
    """Build post-warmup accounting and resource diagnostics for a completed run."""
    Keff = K - warmup
    c = sim._mc
    r = sim.resource
    s1_used = r["s1_gate_attempt"] + r["s1_locked"]
    s2_used = r["s2_start"] + r["s2_busy_continuation"]

    out = dict(
        N=sim.N, L=sim.L, mu=sim.mu, p=sim.p.copy(), w=sim.w.copy(),
        weighted_dest_aoi=sim._wh / c,
        weighted_bsside_age=sim._wA / c,
        per_source_dest_aoi=sim._hacc / c,
        per_source_bsside_age=sim._Aacc / c,
        VOQ_occupancy=sim._Vacc / c,
        avg_fresh_gap_Q=sim._Qacc / c,
        stage1_attempt_rate=sim.post["attempts"] / Keff,
        stage1_success_rate=sim.post["successes"] / Keff,
        VOQ_arrival_rate=sim.post["arrivals"] / Keff,
        stage2_start_rate=sim.post["starts"] / Keff,
        stage2_delivery_rate=sim.post["completions"] / Keff,
        overwrite_rate=sim.post["overwrites"] / Keff,
        same_slot_refill_rate=sim.post["same_slot_refills"] / Keff,
        s1_used_frac=s1_used / Keff,
        s1_gate_attempt_frac=r["s1_gate_attempt"] / Keff,
        s1_locked_frac=r["s1_locked"] / Keff,
        s1_idle_frac=r["s1_idle"] / Keff,
        s2_used_frac=s2_used / Keff,
        s2_start_frac=r["s2_start"] / Keff,
        s2_busy_continuation_frac=r["s2_busy_continuation"] / Keff,
        s2_idle_empty_frac=r["s2_idle_empty"] / Keff,
        s2_idle_nonempty_frac=r["s2_idle_nonempty"] / Keff,
        total_attempts=sim.total["attempts"].copy(),
        total_successes=sim.total["successes"].copy(),
        total_arrivals=sim.total["arrivals"].copy(),
        total_starts=sim.total["starts"].copy(),
        total_completions=sim.total["completions"].copy(),
        min_gap=sim.min_gap,
        max_bridge_violation=sim.max_bridge_violation,
    )
    if sim._series is not None:
        out["series"] = {key: np.asarray(value) for key, value in sim._series.items()}
    return out


def validate_run_result(r: Dict[str, object], K: int, L: int):
    """Finite-horizon checks using lifetime counts, not post-warmup counters."""
    attempts = np.asarray(r["total_attempts"])
    successes = np.asarray(r["total_successes"])
    arrivals = np.asarray(r["total_arrivals"])
    completions = np.asarray(r["total_completions"])
    if np.any(completions > arrivals):
        raise AssertionError("A source completed more packets than arrived from an empty initial pipeline.")
    if attempts.sum() + (L - 1) * successes.sum() > K + L:
        raise AssertionError("Stage-1 finite-horizon resource accounting failed.")
    if float(r["max_bridge_violation"]) > 1e-9:
        raise AssertionError("Structural bridge failed in the measured interval.")


def paired_replications(
    policy_a,
    policy_b,
    N,
    L,
    p,
    mu,
    w,
    *,
    seeds: Iterable[int],
    K: int,
    warmup: int,
) -> Dict[str, object]:
    """Paired common-random-number comparison for two policies."""
    from .simulator import TandemAoISimulatorV3

    rows = []
    for seed in seeds:
        ra = TandemAoISimulatorV3(N, L, p, mu, w, seed=seed).run(policy_a, K, warmup)
        rb = TandemAoISimulatorV3(N, L, p, mu, w, seed=seed).run(policy_b, K, warmup)
        rows.append((ra["weighted_dest_aoi"], rb["weighted_dest_aoi"]))
    values = np.asarray(rows, float)
    diff = values[:, 0] - values[:, 1]
    n = len(diff)
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return {
        "values": values,
        "mean_a": float(values[:, 0].mean()),
        "mean_b": float(values[:, 1].mean()),
        "mean_difference_a_minus_b": float(diff.mean()),
        "standard_error_difference": se,
        "approx_95pct_CI_difference": (
            float(diff.mean() - 1.96 * se),
            float(diff.mean() + 1.96 * se),
        ) if n > 1 else (np.nan, np.nan),
    }


def estimate_iso1_voq_arrival_rates(
    N,
    L,
    p,
    mu,
    w,
    *,
    pilot_seeds=(100, 101, 102, 103, 104),
    K_pilot=50_000,
    warmup_pilot=5_000,
) -> Dict[str, object]:
    """Estimate VOQ arrival rates induced by isolated Stage-1 MW.

    The Stage-2 controller is only used to keep the simulated tandem state valid.
    With the separated RNG streams and current slot convention, the isolated
    Stage-1 arrival process is the object being estimated here.
    """
    from .policies import ComposedV3, IsoBSV3, UniformESV3
    from .rate_optimizer import iso1_params_v3
    from .simulator import TandemAoISimulatorV3

    pilot_seeds = tuple(int(seed) for seed in pilot_seeds)
    if not pilot_seeds:
        raise ValueError("pilot_seeds must contain at least one seed.")

    iso1 = iso1_params_v3(N, L, p, w)
    policy = ComposedV3(IsoBSV3(iso1), UniformESV3(), "iso1 pilot + uniform Stage-2")
    rows = []
    for seed in pilot_seeds:
        result = TandemAoISimulatorV3(N, L, p, mu, w, seed=seed).run(
            policy,
            K=int(K_pilot),
            warmup=int(warmup_pilot),
        )
        rows.append(np.asarray(result["VOQ_arrival_rate"], float))

    values = np.asarray(rows, float)
    lambda_hat = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0]) if values.shape[0] > 1 else np.full(N, np.nan)
    return {
        "lambda_hat": lambda_hat,
        "standard_error": se,
        "sum_lambda": float(lambda_hat.sum()),
        "min_lambda": float(lambda_hat.min()),
        "max_lambda": float(lambda_hat.max()),
        "per_seed": values,
        "metadata": {
            "pilot_seeds": pilot_seeds,
            "K_pilot": int(K_pilot),
            "warmup_pilot": int(warmup_pilot),
            "stage1_controller": "IsoBSV3",
            "stage2_controller": "UniformESV3",
        },
    }
