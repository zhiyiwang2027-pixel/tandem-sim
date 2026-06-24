from __future__ import annotations

import warnings

import numpy as np

from tests.notebook_test_utils import load_notebook_env, oracle_for_stage


def _format_diagnostic(label, sim, diag):
    deltas = ", ".join(
        f"{action}: {delta:.6g}"
        for action, delta in sorted(diag["deltas"].items())
    )
    return (
        f"{label} drift-oracle diagnostic mismatch at slot {sim.k}: "
        f"implemented={diag['implemented']}, oracle={diag['oracle_actions']}, "
        f"B={sim.B}, rBS={sim.rBS}, J={sim.J}, V={sim.V.astype(int).tolist()}, "
        f"h={np.round(sim.h, 4).tolist()}, Aq={np.round(sim.Aq, 4).tolist()}, "
        f"delta_by_action={{ {deltas} }}"
    )


def test_one_step_drift_oracle_diagnostics_warn_instead_of_failing():
    env = load_notebook_env()
    sim_cls = env["TandemAoISimulatorV3"]
    N, L = 4, 3
    p = np.array([0.9, 0.65, 0.4, 0.2])
    mu = 0.35
    w = np.array([1.0, 2.0, 4.0, 8.0])
    policies, params = env["build_experiment_v3"](N, L, p, mu, w)
    sim = sim_cls(N, L, p, mu, w, seed=5)
    sim.reset(horizon=80)

    mismatches = []
    inspected = 0
    for _ in range(35):
        if sim.k in {4, 8, 12, 18, 26, 34}:
            for policy_name, coeff_name in [
                ("Joint FGMW", "joint"),
                ("iso1 + iso2", "iso1"),
            ]:
                ctrl = policies[policy_name].bs_ctrl
                implemented = ctrl.bs(sim) if sim.B == -1 else -1
                diag = oracle_for_stage(
                    sim,
                    params[coeff_name],
                    stage="bs",
                    implemented_action=implemented,
                    fixed_other_action=-1,
                )
                inspected += 1
                if diag["implemented"] not in diag["oracle_actions"]:
                    mismatches.append(_format_diagnostic(f"{policy_name} BS", sim, diag))

            for policy_name, coeff_name in [
                ("Joint FGMW", "joint"),
                ("iso1 + iso2", "iso2"),
            ]:
                ctrl = policies[policy_name].es_ctrl
                implemented = ctrl.es(sim) if sim.J == -1 else -1
                diag = oracle_for_stage(
                    sim,
                    params[coeff_name],
                    stage="es",
                    implemented_action=implemented,
                    fixed_other_action=-1,
                )
                inspected += 1
                if diag["implemented"] not in diag["oracle_actions"]:
                    mismatches.append(_format_diagnostic(f"{policy_name} ES", sim, diag))

        sim.step(policies["Joint FGMW"])

    for message in mismatches[:6]:
        warnings.warn(message, UserWarning)

    assert inspected > 0
