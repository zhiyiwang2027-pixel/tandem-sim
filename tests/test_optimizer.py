from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from tests.notebook_test_utils import load_notebook_env


def _objective(q, w, g):
    return float(np.sum(w / (2.0 * q) + g * q))


def _objective_jac(q, w, g):
    return -w / (2.0 * q * q) + g


def _run_scipy_minimize(w, g, *, c=None, link_cap=1.0, edge_cap=None):
    n = len(w)
    caps = []
    if c is not None:
        caps.append(link_cap / float(np.sum(c)))
    if edge_cap is not None:
        caps.append(edge_cap / n)
    x0 = np.full(n, 0.5 * min(caps) if caps else 1.0)

    rows = []
    upper = []
    if c is not None:
        rows.append(c)
        upper.append(link_cap)
    if edge_cap is not None:
        rows.append(np.ones(n))
        upper.append(edge_cap)

    constraints = []
    if rows:
        constraints.append(
            LinearConstraint(
                np.vstack(rows),
                np.full(len(rows), -np.inf),
                np.asarray(upper, dtype=float),
            )
        )

    return minimize(
        lambda q: _objective(q, w, g),
        jac=lambda q: _objective_jac(q, w, g),
        x0=x0,
        method="trust-constr",
        bounds=Bounds(np.full(n, 1e-12), np.full(n, np.inf)),
        constraints=constraints,
        options={"gtol": 1e-11, "xtol": 1e-11, "maxiter": 1000},
    )


def test_joint_optimizer_feasibility_complementarity_and_scipy_agreement():
    env = load_notebook_env()
    solve_rate_kkt = env["solve_rate_kkt"]

    cases = [
        (
            np.array([1.0, 2.0, 4.0]),
            np.array([0.9, 0.55, 0.25]),
            3,
            0.35,
        ),
        (
            np.array([1.0, 1.0, 3.0, 8.0]),
            np.array([0.95, 0.7, 0.4, 0.12]),
            4,
            0.18,
        ),
        (
            np.array([0.5, 2.5]),
            np.array([1.0, 0.3]),
            2,
            1.0,
        ),
    ]

    for w, p, L, mu in cases:
        c = (L - 1.0) + 1.0 / p
        g = w * (1.0 - mu) / mu**2
        q, info = solve_rate_kkt(w, g, link_c=c, link_cap=1.0, edge_cap=mu)

        assert np.all(q > 0.0)
        assert float(c @ q) <= 1.0 + 1e-9
        assert float(q.sum()) <= mu + 1e-9
        assert info["lambda_link"] >= -1e-12
        assert info["nu_edge"] >= -1e-12
        assert abs(info["lambda_link"] * (1.0 - float(c @ q))) <= 1e-7
        assert abs(info["nu_edge"] * (mu - float(q.sum()))) <= 1e-7

        stationarity = -w / (2.0 * q * q) + g + info["lambda_link"] * c + info["nu_edge"]
        np.testing.assert_allclose(stationarity, np.zeros_like(q), atol=1e-7, rtol=1e-7)

        opt = _run_scipy_minimize(w, g, c=c, edge_cap=mu)
        assert opt.success, opt.message
        np.testing.assert_allclose(_objective(q, w, g), opt.fun, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(q, opt.x, rtol=5e-5, atol=5e-7)


def test_edge_only_optimizer_feasibility_complementarity_and_scipy_agreement():
    env = load_notebook_env()
    solve_rate_kkt = env["solve_rate_kkt"]

    rng = np.random.default_rng(123)
    for _ in range(6):
        n = int(rng.integers(2, 7))
        mu = float(rng.uniform(0.08, 1.0))
        w = np.exp(rng.uniform(np.log(0.2), np.log(20.0), n))
        g = w * (1.0 - mu) / mu**2

        q, info = solve_rate_kkt(w, g, edge_cap=mu)
        assert np.all(q > 0.0)
        assert float(q.sum()) <= mu + 1e-9
        assert info["nu_edge"] >= -1e-12
        assert abs(info["nu_edge"] * (mu - float(q.sum()))) <= 1e-7

        stationarity = -w / (2.0 * q * q) + g + info["nu_edge"]
        np.testing.assert_allclose(stationarity, np.zeros_like(q), atol=1e-7, rtol=1e-7)

        opt = _run_scipy_minimize(w, g, edge_cap=mu)
        assert opt.success, opt.message
        np.testing.assert_allclose(_objective(q, w, g), opt.fun, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(q, opt.x, rtol=5e-5, atol=5e-7)
