from __future__ import annotations

import math

import numpy as np

from tests.notebook_test_utils import (
    load_notebook_env,
    run_measured_window,
    stage1_pending,
    stage2_in_service,
)


def _representative_window():
    env = load_notebook_env()
    N, L = 4, 3
    p = np.array([0.9, 0.7, 0.45, 0.25])
    mu = 0.4
    w = np.array([1.0, 2.0, 4.0, 8.0])
    policies, _ = env["build_experiment_v3"](N, L, p, mu, w)
    window = run_measured_window(
        env,
        policies["Joint FGMW"],
        N=N,
        L=L,
        p=p,
        mu=mu,
        w=w,
        seed=11,
        K=12_000,
        warmup=1_200,
    )
    return env, window, p, mu


def test_post_warmup_per_source_conservation():
    _, window, _, _ = _representative_window()
    sim = window.sim
    n = sim.N
    post = sim.post

    s1_start = stage1_pending(window.start, n)
    s1_end = stage1_pending(window.end, n)
    np.testing.assert_array_equal(
        s1_start + post["successes"],
        post["arrivals"] + s1_end,
        err_msg="Stage-1 success/arrival conservation failed over the measured window.",
    )

    voq_start = window.start.V.astype(np.int64)
    voq_end = window.end.V.astype(np.int64)
    np.testing.assert_array_equal(
        voq_start + post["arrivals"] - post["overwrites"] - post["starts"],
        voq_end,
        err_msg="VOQ occupancy conservation failed over the measured window.",
    )

    service_start = stage2_in_service(window.start, n)
    service_end = stage2_in_service(window.end, n)
    np.testing.assert_array_equal(
        service_start + post["starts"],
        post["completions"] + service_end,
        err_msg="Stage-2 in-service conservation failed over the measured window.",
    )

    np.testing.assert_array_equal(
        voq_start + service_start + post["arrivals"] - post["overwrites"] - post["completions"],
        voq_end + service_end,
        err_msg="Total post-Stage-1 packet conservation failed over the measured window.",
    )


def test_stage1_resource_accounting_and_boundary_adjusted_capacity_diagnostic():
    _, window, p, _ = _representative_window()
    sim = window.sim
    post = sim.post
    resource = sim.resource
    keff = window.keff

    assert (
        resource["s1_gate_attempt"] + resource["s1_locked"] + resource["s1_idle"]
    ) == keff
    assert int(post["attempts"].sum()) == resource["s1_gate_attempt"]
    assert resource["s1_gate_attempt"] + resource["s1_locked"] <= keff

    s1_start = stage1_pending(window.start, sim.N)
    s1_end = stage1_pending(window.end, sim.N)
    boundary_adjusted_arrivals = post["arrivals"] + s1_end - s1_start
    np.testing.assert_array_equal(boundary_adjusted_arrivals, post["successes"])

    attempts = post["attempts"].astype(float)
    successes = post["successes"].astype(float)
    expected_successes = attempts * p
    sigma = np.sqrt(attempts * p * (1.0 - p))
    assert np.all(np.abs(successes - expected_successes) <= 6.0 * sigma + 6.0)

    c = (sim.L - 1.0) + 1.0 / p
    boundary_adjusted_capacity_proxy = float(c @ boundary_adjusted_arrivals) / keff
    assert math.isfinite(boundary_adjusted_capacity_proxy)


def test_stage2_resource_accounting_and_completion_binomial_check():
    _, window, _, mu = _representative_window()
    sim = window.sim
    post = sim.post
    resource = sim.resource
    keff = window.keff

    assert (
        resource["s2_start"]
        + resource["s2_busy_continuation"]
        + resource["s2_idle_empty"]
        + resource["s2_idle_nonempty"]
    ) == keff
    assert int(post["starts"].sum()) == resource["s2_start"]
    busy_slots = resource["s2_start"] + resource["s2_busy_continuation"]
    assert busy_slots <= keff

    completions = int(post["completions"].sum())
    expected = busy_slots * mu
    sigma = math.sqrt(busy_slots * mu * (1.0 - mu))
    assert abs(completions - expected) <= 6.0 * sigma + 6.0
