from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.heterogeneous_screening import run_heterogeneous_screening
from tandem.policies import (
    DownstreamAwareMWBSV3,
    DownstreamAwareMWESV3,
    ComposedV3,
    GreedyBSV3,
    GreedyESV3,
    UniformBSV3,
    build_policies_v3,
)
from tandem.rate_optimizer import joint_params_v3
from tandem.simulator import TandemAoISimulatorV3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dummy_sim(**kwargs):
    base = dict(
        N=2,
        L=3,
        mu=0.25,
        p=np.array([0.9, 0.4]),
        w=np.array([100.0, 1.0]),
        h=np.array([10.0, 11.0]),
        Aq=np.array([2.0, 3.0]),
        V=np.array([True, True]),
        J=-1,
        ZS=0.0,
        rng_policy_bs=np.random.default_rng(0),
        rng_policy_es=np.random.default_rng(1),
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_greedy_stage1_uses_unweighted_aq_not_weighted_aq():
    sim = _dummy_sim(
        Aq=np.array([2.0, 3.0]),
        w=np.array([100.0, 1.0]),
    )

    assert GreedyBSV3(sim.w).bs(sim) == 1


def test_greedy_stage2_uses_unweighted_h_among_nonempty_not_weighted_h_or_q():
    sim = _dummy_sim(
        h=np.array([100.0, 90.0]),
        Aq=np.array([99.0, 0.0]),
        V=np.array([True, True]),
        w=np.array([1.0, 10.0]),
    )

    assert GreedyESV3(sim.w).es(sim) == 0


def test_damw_uses_joint_lower_bound_qtar_and_no_xtar():
    N, L, mu = 3, 3, 0.25
    p = np.array([0.9, 0.5, 0.2])
    w = np.array([1.0, 2.0, 4.0])
    policies, params = build_policies_v3(N, L, p, mu, w)
    damw = policies["Downstream-Aware MW"]

    np.testing.assert_allclose(damw.bs_ctrl.qtar, params["joint"]["qd"])
    np.testing.assert_allclose(damw.es_ctrl.qtar, joint_params_v3(N, L, p, mu, w)["qd"])
    assert not hasattr(damw.bs_ctrl, "xtar")
    assert not hasattr(damw.es_ctrl, "xtar")


def test_damw_stage1_index_is_debt_plus_theta_covered_copy_age_over_link_cost():
    sim = _dummy_sim(
        N=3,
        L=3,
        p=np.array([0.9, 0.6, 0.3]),
        w=np.array([1.0, 2.0, 3.0]),
        h=np.array([20.0, 18.0, 10.0]),
        Aq=np.array([4.0, 8.0, 3.0]),
        V=np.array([True, False, True]),
        J=2,
        ZS=6.0,
    )
    qtar = np.array([0.05, 0.10, 0.20])
    controller = DownstreamAwareMWBSV3(sim.L, sim.p, sim.w, qtar)
    controller.U = np.array([0.3, 0.2, 0.1])

    Q = sim.V * (sim.h - sim.Aq)
    Y = np.array([0.0, 0.0, sim.h[2] - sim.ZS])
    C = sim.h - np.maximum(Q, Y)
    theta = sim.w / qtar
    expected = (controller.U + theta * C) / ((sim.L - 1.0) + 1.0 / sim.p)

    np.testing.assert_allclose(controller.covered_copy_age(sim), C)
    np.testing.assert_allclose(controller.bs_index(sim), expected)


def test_damw_stage2_index_is_downstream_debt_plus_theta_waiting_gap():
    sim = _dummy_sim(
        h=np.array([20.0, 15.0]),
        Aq=np.array([5.0, 7.0]),
        V=np.array([True, False]),
        w=np.array([2.0, 4.0]),
    )
    qtar = np.array([0.05, 0.2])
    controller = DownstreamAwareMWESV3(sim.w, qtar)
    controller.D = np.array([0.4, 0.7])

    Q = sim.V * (sim.h - sim.Aq)
    expected = controller.D + (sim.w / qtar) * Q

    np.testing.assert_allclose(controller.es_index(sim), expected)


def test_damw_upstream_debt_uses_successful_stage1_commitment_indicator():
    N, L, mu = 1, 3, 0.5
    p = np.array([1.0])
    w = np.array([1.0])
    qtar = np.array([0.1])
    bs = DownstreamAwareMWBSV3(L, p, w, qtar)
    policy = ComposedV3(bs, GreedyESV3(), "test DA-MW upstream debt")

    TandemAoISimulatorV3(N, L, p, mu, w, seed=0).run(policy, K=3, warmup=0)

    np.testing.assert_allclose(bs.U, np.array([0.2]), atol=1e-12)


def test_damw_downstream_debt_uses_destination_delivery_indicator():
    N, L, mu = 1, 1, 1.0
    p = np.array([1.0])
    w = np.array([1.0])
    qtar = np.array([0.1])
    es = DownstreamAwareMWESV3(w, qtar)
    policy = ComposedV3(UniformBSV3(), es, "test DA-MW downstream debt")

    TandemAoISimulatorV3(N, L, p, mu, w, seed=0).run(policy, K=2, warmup=0)

    np.testing.assert_allclose(es.D, np.array([0.0]), atol=1e-12)


def test_downstream_aware_mw_smoke_simulation_runs():
    N, L, mu = 4, 3, 0.25
    p = np.array([0.95, 0.7, 0.4, 0.18])
    w = np.array([2.0, 1.0, 0.7, 0.3])
    policies, _ = build_policies_v3(N, L, p, mu, w)

    result = TandemAoISimulatorV3(N, L, p, mu, w, seed=0).run(
        policies["Downstream-Aware MW"],
        K=800,
        warmup=100,
        validate=True,
    )

    assert np.isfinite(result["weighted_dest_aoi"])
    assert result["s1_used_frac"] >= 0.0


def test_damw_quick_smoke_output_does_not_modify_full_cached_csvs(tmp_path):
    full_paths = [
        Path("results/heterogeneous_screening.csv"),
        Path("results/heterogeneous_screening_with_srp.csv"),
    ]
    before = {path: _sha256(path) for path in full_paths}
    output = tmp_path / "damw_debt_quick.csv"

    df = run_heterogeneous_screening(
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
    assert {"Greedy", "Downstream-Aware MW"}.issubset(set(df["policy"]))
    after = {path: _sha256(path) for path in full_paths}
    assert before == after
