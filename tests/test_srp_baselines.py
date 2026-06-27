from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from experiments.heterogeneous_screening_with_srp import run_heterogeneous_screening_with_srp
from tandem.policies import build_srp_iso_policy_v3, build_srp_tandem_lb_policy_v3
from tandem.rate_optimizer import (
    joint_params_v3,
    rate_to_srp_alpha,
    rate_to_srp_beta,
    solve_iso1_srp_rate_kkt,
)
from tandem.simulator import TandemAoISimulatorV3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_srp_alpha_mapping_round_trip():
    L = 3
    p = np.array([0.9, 0.6, 0.35])
    q = np.array([0.045, 0.035, 0.015])

    alpha = rate_to_srp_alpha(q, p, L)
    denom = 1.0 - (L - 1.0) * float(q.sum())

    assert alpha.sum() <= 1.0 + 1e-10
    np.testing.assert_allclose(alpha * p * denom, q, rtol=1e-12, atol=1e-12)


def test_srp_alpha_feasibility_rejects_link_infeasible_rates():
    p = np.array([0.9, 0.6, 0.35])
    q = np.array([0.2, 0.15, 0.12])

    with pytest.raises(ValueError, match="link-feasible"):
        rate_to_srp_alpha(q, p, L=3)


def test_srp_beta_feasibility():
    q = np.array([0.03, 0.04, 0.02])
    beta = rate_to_srp_beta(q, mu=0.12)

    assert beta.sum() <= 1.0 + 1e-10
    np.testing.assert_allclose(beta, q / 0.12)

    with pytest.raises(ValueError, match="edge capacity"):
        rate_to_srp_beta(np.array([0.08, 0.07]), mu=0.12)


def test_iso1_srp_optimizer_returns_feasible_rates():
    L = 3
    p = np.array([0.95, 0.55, 0.28, 0.10])
    w = np.array([3.0, 1.0, 0.8, 0.2])

    q, info = solve_iso1_srp_rate_kkt(L, p, w)
    c = (L - 1.0) + 1.0 / p
    alpha = rate_to_srp_alpha(q, p, L)

    assert np.all(np.isfinite(q))
    assert np.all(q > 0.0)
    assert float(c @ q) <= 1.0 + 1e-9
    assert alpha.sum() <= 1.0 + 1e-8
    assert info["nu_iso1_srp"] >= 0.0


def test_srp_policy_smoke_simulation_runs():
    N, L, mu = 4, 3, 0.25
    p = np.array([0.95, 0.7, 0.4, 0.18])
    w = np.array([2.0, 1.0, 0.7, 0.3])
    lambda_cap = np.array([0.04, 0.035, 0.025, 0.015])

    policies = [
        build_srp_iso_policy_v3(N, L, p, mu, w, lambda_cap),
        build_srp_tandem_lb_policy_v3(N, L, p, mu, w, joint_params_v3(N, L, p, mu, w)),
    ]
    for policy in policies:
        result = TandemAoISimulatorV3(N, L, p, mu, w, seed=0).run(
            policy,
            K=800,
            warmup=100,
            validate=True,
        )
        assert np.isfinite(result["weighted_dest_aoi"])
        assert float(result["s1_used_frac"]) >= 0.0


def test_srp_quick_function_smoke_run(tmp_path):
    output = tmp_path / "screening_with_srp.csv"
    df = run_heterogeneous_screening_with_srp(
        mode="quick",
        output_csv=output,
        N=4,
        weight_ratios=(10.0,),
        L_values=(3,),
        seeds=(0,),
        pilot_seeds=(100,),
        p_levels=(0.95, 0.7, 0.4, 0.1),
    )

    assert output.exists()
    assert len(df) == 28
    assert set(df["policy"]) == {
        "Joint FGMW",
        "iso1 + iso2",
        "iso1 + iso2-lambda",
        "SRP-iso",
        "SRP-tandem-LB",
        "Greedy",
        "Uniform",
    }
    assert np.all(np.isfinite(df["weighted_dest_aoi"]))
    assert np.all(np.isfinite(df["gap_vs_iso_lambda_pct"]))


def test_srp_quick_cli_does_not_modify_mw_only_screening_csv(tmp_path):
    mw_only = Path("results/heterogeneous_screening.csv")
    before = _sha256(mw_only)
    output = tmp_path / "screening_with_srp_cli.csv"

    completed = subprocess.run(
        [
            sys.executable,
            "experiments/heterogeneous_screening_with_srp.py",
            "--quick",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    after = _sha256(mw_only)
    assert before == after
    assert "mode=quick" in completed.stdout
    assert output.exists()
    df = pd.read_csv(output)
    assert len(df) == 28
