from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from tandem.diagnostics import estimate_iso1_voq_arrival_rates
from tandem.policies import build_experiment_v3, build_iso1_iso2_lambda_policy_v3
from tandem.rate_optimizer import solve_capped_edge_rate_kkt
from tandem.simulator import TandemAoISimulatorV3


def _objective(q, w, g):
    return float(np.sum(w / (2.0 * q) + g * q))


def _objective_jac(q, w, g):
    return -w / (2.0 * q * q) + g


def _run_capped_scipy(w, g, lambda_cap, edge_cap):
    n = len(w)
    x0 = np.minimum(0.5 * lambda_cap, 0.5 * edge_cap / n)
    return minimize(
        lambda q: _objective(q, w, g),
        jac=lambda q: _objective_jac(q, w, g),
        x0=x0,
        method="trust-constr",
        bounds=Bounds(np.full(n, 1e-12), lambda_cap),
        constraints=[
            LinearConstraint(
                np.ones((1, n)),
                np.array([-np.inf]),
                np.array([edge_cap]),
            )
        ],
        options={"gtol": 1e-11, "xtol": 1e-11, "maxiter": 1000},
    )


def test_capped_edge_optimizer_feasibility_and_scipy_agreement():
    rng = np.random.default_rng(51)
    for _ in range(5):
        n = int(rng.integers(2, 7))
        mu = float(rng.uniform(0.08, 0.6))
        w = np.exp(rng.uniform(np.log(0.4), np.log(12.0), n))
        g = w * (1.0 - mu) / mu**2
        lambda_cap = rng.uniform(0.015, 0.18, n)

        q, info = solve_capped_edge_rate_kkt(w, g, lambda_cap, edge_cap=mu)
        assert np.all(q > 0.0)
        assert np.all(q <= lambda_cap + 1e-10)
        assert float(q.sum()) <= mu + 1e-10
        assert info["nu_edge"] >= -1e-12
        assert info["cap_active_count"] >= 0
        assert np.all(np.asarray(info["cap_dual_like"]) >= -1e-12)

        opt = _run_capped_scipy(w, g, lambda_cap, mu)
        assert opt.success, opt.message
        np.testing.assert_allclose(_objective(q, w, g), opt.fun, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(q, opt.x, rtol=2e-4, atol=2e-6)


def test_lambda_estimator_returns_finite_nonnegative_rates():
    N, L, mu = 4, 3, 0.4
    p = np.array([0.9, 0.8, 0.7, 0.6])
    w = np.array([1.0, 1.5, 2.0, 4.0])
    est = estimate_iso1_voq_arrival_rates(
        N,
        L,
        p,
        mu,
        w,
        pilot_seeds=(100, 101),
        K_pilot=2_000,
        warmup_pilot=200,
    )
    lam = est["lambda_hat"]
    se = est["standard_error"]
    assert lam.shape == (N,)
    assert se.shape == (N,)
    assert np.all(np.isfinite(lam))
    assert np.all(lam >= 0.0)
    assert np.isfinite(est["sum_lambda"])
    assert est["min_lambda"] >= 0.0
    assert est["max_lambda"] >= est["min_lambda"]


def test_iso1_iso2_lambda_policy_runs_successfully():
    N, L, mu = 5, 3, 0.3
    p = np.array([0.95, 0.7, 0.5, 0.3, 0.18])
    w = np.array([1.0, 1.2, 2.0, 4.0, 7.0])
    est = estimate_iso1_voq_arrival_rates(
        N,
        L,
        p,
        mu,
        w,
        pilot_seeds=(100, 101),
        K_pilot=3_000,
        warmup_pilot=300,
    )
    policy = build_iso1_iso2_lambda_policy_v3(N, L, p, mu, w, est["lambda_hat"])
    result = TandemAoISimulatorV3(N, L, p, mu, w, seed=0).run(
        policy,
        K=2_000,
        warmup=200,
        validate=True,
    )
    assert np.isfinite(result["weighted_dest_aoi"])
    assert float(result["stage2_delivery_rate"].sum()) <= mu + 0.1


def test_existing_joint_and_iso_relaxed_anchor_results_are_unchanged():
    N, L, mu = 6, 3, 0.6
    p = np.array([0.6] * N)
    w = np.array([1.0, 1.0, 1.0, 4.0, 4.0, 4.0])
    policies, _ = build_experiment_v3(N, L, p, mu, w)

    joint = TandemAoISimulatorV3(N, L, p, mu, w, seed=0).run(
        policies["Joint FGMW"],
        K=30_000,
        warmup=3_000,
        validate=True,
    )
    iso = TandemAoISimulatorV3(N, L, p, mu, w, seed=0).run(
        policies["iso1 + iso2"],
        K=30_000,
        warmup=3_000,
        validate=True,
    )

    np.testing.assert_allclose(joint["weighted_bsside_age"], 31.391999999999708, atol=1e-12)
    np.testing.assert_allclose(joint["weighted_dest_aoi"], 38.286506172839019, atol=1e-12)
    np.testing.assert_allclose(joint["VOQ_arrival_rate"].sum(), 0.273703703703704, atol=1e-12)
    np.testing.assert_allclose(joint["stage2_delivery_rate"].sum(), 0.273703703703704, atol=1e-12)

    np.testing.assert_allclose(iso["weighted_bsside_age"], 31.450808641974945, atol=1e-12)
    np.testing.assert_allclose(iso["weighted_dest_aoi"], 38.3891172839501, atol=1e-12)
    np.testing.assert_allclose(iso["VOQ_arrival_rate"].sum(), 0.27337037037037, atol=1e-12)
    np.testing.assert_allclose(iso["stage2_delivery_rate"].sum(), 0.273407407407407, atol=1e-12)
