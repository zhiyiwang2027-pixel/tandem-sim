from __future__ import annotations

import numpy as np

from tests.notebook_test_utils import load_notebook_env


def _config():
    return (
        5,
        3,
        np.array([0.9, 0.7, 0.5, 0.35, 0.2]),
        0.35,
        np.array([1.0, 1.5, 2.0, 4.0, 7.0]),
    )


def test_same_seed_reproduces_trajectory_and_rates_exactly():
    env = load_notebook_env()
    sim_cls = env["TandemAoISimulatorV3"]
    N, L, p, mu, w = _config()
    policies, _ = env["build_experiment_v3"](N, L, p, mu, w)

    first = sim_cls(N, L, p, mu, w, seed=42).run(
        policies["Joint FGMW"],
        K=2_000,
        warmup=200,
        record_series=True,
    )
    second = sim_cls(N, L, p, mu, w, seed=42).run(
        policies["Joint FGMW"],
        K=2_000,
        warmup=200,
        record_series=True,
    )

    scalar_keys = ["weighted_dest_aoi", "weighted_bsside_age", "s1_used_frac", "s2_used_frac"]
    for key in scalar_keys:
        assert first[key] == second[key]
    vector_keys = [
        "VOQ_arrival_rate",
        "stage2_delivery_rate",
        "stage1_attempt_rate",
        "stage1_success_rate",
        "VOQ_occupancy",
    ]
    for key in vector_keys:
        np.testing.assert_array_equal(first[key], second[key])
    for key in ["h", "Aq", "Q", "V"]:
        np.testing.assert_array_equal(first["series"][key], second["series"][key])


def test_stage1_path_is_reproducible_under_different_stage2_policy_rng_use():
    env = load_notebook_env()
    sim_cls = env["TandemAoISimulatorV3"]
    N, L, p, mu, w = _config()
    policies, _ = env["build_experiment_v3"](N, L, p, mu, w)

    mw = sim_cls(N, L, p, mu, w, seed=8).run(
        policies["iso1 + iso2"],
        K=3_000,
        warmup=0,
        record_series=True,
    )
    srp = sim_cls(N, L, p, mu, w, seed=8).run(
        policies["iso1 + WC-SRP2"],
        K=3_000,
        warmup=0,
        record_series=True,
    )

    np.testing.assert_array_equal(mw["series"]["Aq"], srp["series"]["Aq"])
    np.testing.assert_array_equal(mw["total_attempts"], srp["total_attempts"])
    np.testing.assert_array_equal(mw["total_successes"], srp["total_successes"])
    np.testing.assert_array_equal(mw["total_arrivals"], srp["total_arrivals"])
