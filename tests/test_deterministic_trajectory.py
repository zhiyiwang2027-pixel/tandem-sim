from __future__ import annotations

import numpy as np

from tests.notebook_test_utils import load_notebook_env


class AlwaysServeOneSource:
    def decide(self, sim):
        u_bs = 0 if sim.B == -1 else -1
        u_es = 0 if sim.J == -1 and sim.V[0] else -1
        return u_bs, u_es


EXPECTED_PRE_STEP = {
    1: {
        "h": [2.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        "Aq": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "V": [False, True, True, True, True, True],
        "B": [-1, -1, -1, -1, -1, -1],
        "rBS": [0, 0, 0, 0, 0, 0],
        "J": [-1, -1, -1, -1, -1, -1],
        "ZS": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    2: {
        "h": [3.0, 4.0, 5.0, 4.0, 5.0, 4.0],
        "Aq": [2.0, 3.0, 2.0, 3.0, 2.0, 3.0],
        "V": [False, False, True, False, True, False],
        "B": [-1, 0, -1, 0, -1, 0],
        "rBS": [0, 1, 0, 1, 0, 1],
        "J": [-1, -1, -1, -1, -1, -1],
        "ZS": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    3: {
        "h": [4.0, 5.0, 6.0, 7.0, 5.0, 6.0, 7.0],
        "Aq": [3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0],
        "V": [False, False, False, True, False, False, True],
        "B": [-1, 0, 0, -1, 0, 0, -1],
        "rBS": [0, 2, 1, 0, 2, 1, 0],
        "J": [-1, -1, -1, -1, -1, -1, -1],
        "ZS": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
}


def test_current_slot_convention_pre_step_trajectory_for_single_source():
    env = load_notebook_env()
    sim_cls = env["TandemAoISimulatorV3"]
    policy = AlwaysServeOneSource()

    for L, expected in EXPECTED_PRE_STEP.items():
        sim = sim_cls(1, L, [1.0], 1.0, [1.0], seed=1)
        sim.reset(horizon=len(expected["h"]))
        observed = {key: [] for key in expected}
        for _ in range(len(expected["h"])):
            observed["h"].append(float(sim.h[0]))
            observed["Aq"].append(float(sim.Aq[0]))
            observed["V"].append(bool(sim.V[0]))
            observed["B"].append(int(sim.B))
            observed["rBS"].append(int(sim.rBS))
            observed["J"].append(int(sim.J))
            observed["ZS"].append(float(sim.ZS))
            sim.assert_state_invariants()
            sim.step(policy)

        for key, expected_values in expected.items():
            if key in {"h", "Aq", "ZS"}:
                np.testing.assert_allclose(observed[key], expected_values)
            else:
                assert observed[key] == expected_values


def test_current_slot_convention_single_source_averages():
    env = load_notebook_env()
    sim_cls = env["TandemAoISimulatorV3"]
    for L in (1, 2, 3):
        result = sim_cls(1, L, [1.0], 1.0, [1.0], seed=1).run(
            AlwaysServeOneSource(),
            K=500,
            warmup=50,
            validate=True,
        )
        np.testing.assert_allclose(result["weighted_bsside_age"], 1.5 * L - 0.5)
        np.testing.assert_allclose(result["weighted_dest_aoi"], 1.5 * L + 1.5)
