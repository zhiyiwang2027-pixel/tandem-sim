from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tandem_combined.ipynb"
CORE_CELLS = (2, 4, 6, 8)


@lru_cache(maxsize=1)
def load_notebook_env() -> Dict[str, Any]:
    """Execute only the notebook cells that define reusable simulator code."""
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))
    nb = json.loads(NOTEBOOK.read_text())
    env: Dict[str, Any] = {"__name__": "tandem_notebook_under_test"}
    for cell_no in CORE_CELLS:
        source = "".join(nb["cells"][cell_no - 1].get("source", []))
        source = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("%")
        )
        exec(compile(source, f"{NOTEBOOK.name}:cell-{cell_no}", "exec"), env)
    return env


@dataclass(frozen=True)
class StateSnapshot:
    h: np.ndarray
    Aq: np.ndarray
    V: np.ndarray
    B: int
    rBS: int
    J: int
    ZS: float


@dataclass(frozen=True)
class WindowRun:
    sim: Any
    start: StateSnapshot
    end: StateSnapshot
    K: int
    warmup: int

    @property
    def keff(self) -> int:
        return self.K - self.warmup


def snapshot(sim: Any) -> StateSnapshot:
    return StateSnapshot(
        h=sim.h.copy(),
        Aq=sim.Aq.copy(),
        V=sim.V.copy(),
        B=int(sim.B),
        rBS=int(sim.rBS),
        J=int(sim.J),
        ZS=float(sim.ZS),
    )


def source_indicator(index: int, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int64)
    if index != -1:
        out[int(index)] = 1
    return out


def stage1_pending(state: StateSnapshot, n: int) -> np.ndarray:
    return source_indicator(state.B, n)


def stage2_in_service(state: StateSnapshot, n: int) -> np.ndarray:
    return source_indicator(state.J, n)


def run_measured_window(
    env: Dict[str, Any],
    policy: Any,
    *,
    N: int,
    L: int,
    p: Iterable[float],
    mu: float,
    w: Iterable[float],
    seed: int,
    K: int,
    warmup: int,
) -> WindowRun:
    """Run the simulator while capturing exact warmup and final states."""
    sim_cls = env["TandemAoISimulatorV3"]
    sim = sim_cls(N, L, p, mu, w, seed=seed)
    sim.reset(horizon=K)
    start = None
    for k in range(K):
        if k == warmup:
            start = snapshot(sim)
        sim._collect = k >= warmup
        sim.step(policy)
    sim._collect = False
    if start is None:
        raise AssertionError("warmup state was not captured")
    return WindowRun(sim=sim, start=start, end=snapshot(sim), K=K, warmup=warmup)


class FixedActionPolicy:
    def __init__(self, bs_action: int, es_action: int):
        self.bs_action = int(bs_action)
        self.es_action = int(es_action)

    def decide(self, sim: Any) -> Tuple[int, int]:
        return self.bs_action, self.es_action


def clone_sim_state(sim: Any) -> Any:
    sim_cls = type(sim)
    clone = sim_cls(sim.N, sim.L, sim.p.copy(), sim.mu, sim.w.copy(), seed=sim.seed)
    clone.reset()
    clone.h = sim.h.copy()
    clone.Aq = sim.Aq.copy()
    clone.V = sim.V.copy()
    clone.B, clone.rBS = int(sim.B), int(sim.rBS)
    clone.J, clone.ZS = int(sim.J), float(sim.ZS)
    clone.k = int(sim.k)
    return clone


def simulate_forced_one_step(
    sim: Any,
    *,
    bs_action: int,
    es_action: int,
    gate_success: bool | None,
    service_success: bool | None,
) -> Any:
    clone = clone_sim_state(sim)

    if gate_success is not None:
        clone._gate_uniform = lambda source: 0.0 if gate_success else 1.0
    if service_success is not None:
        clone._service_uniform = lambda: 0.0 if service_success else 1.0

    clone.step(FixedActionPolicy(bs_action, es_action))
    return clone


def outcome_probabilities(sim: Any, bs_action: int, es_action: int) -> List[Tuple[float, bool | None, bool | None]]:
    if sim.B == -1 and bs_action != -1:
        gate_cases = [(float(sim.p[bs_action]), True), (1.0 - float(sim.p[bs_action]), False)]
    else:
        gate_cases = [(1.0, None)]

    service_attempted = sim.J != -1 or (sim.J == -1 and es_action != -1)
    if service_attempted:
        service_cases = [(float(sim.mu), True), (1.0 - float(sim.mu), False)]
    else:
        service_cases = [(1.0, None)]

    cases = []
    for gp, gate_success in gate_cases:
        for sp, service_success in service_cases:
            prob = gp * sp
            if prob > 0.0:
                cases.append((prob, gate_success, service_success))
    return cases


def lyapunov_score(sim: Any, coeffs: Dict[str, Any]) -> float:
    Q, Y = sim.gaps()
    score = 0.0
    if "vA" in coeffs:
        score += float((coeffs["vA"] * sim.Aq).sum())
    if "vQ" in coeffs:
        score += float((coeffs["vQ"] * Q).sum())
    if "vY" in coeffs:
        score += float((coeffs["vY"] * Y).sum())
    if "vh" in coeffs:
        score += float((coeffs["vh"] * sim.h).sum())
    return score


def expected_one_step_score(
    sim: Any,
    coeffs: Dict[str, Any],
    *,
    bs_action: int,
    es_action: int,
) -> float:
    total = 0.0
    for prob, gate_success, service_success in outcome_probabilities(sim, bs_action, es_action):
        nxt = simulate_forced_one_step(
            sim,
            bs_action=bs_action,
            es_action=es_action,
            gate_success=gate_success,
            service_success=service_success,
        )
        total += prob * lyapunov_score(nxt, coeffs)
    return float(total)


def oracle_for_stage(
    sim: Any,
    coeffs: Dict[str, Any],
    *,
    stage: str,
    implemented_action: int,
    fixed_other_action: int = -1,
    atol: float = 1e-9,
) -> Dict[str, Any]:
    current = lyapunov_score(sim, coeffs)
    if stage == "bs":
        candidates = [-1] if sim.B != -1 else [-1, *range(sim.N)]
        action_scores = {
            action: expected_one_step_score(
                sim,
                coeffs,
                bs_action=action,
                es_action=fixed_other_action,
            ) - current
            for action in candidates
        }
    elif stage == "es":
        candidates = [-1] if sim.J != -1 else [-1, *np.flatnonzero(sim.V).astype(int).tolist()]
        action_scores = {
            action: expected_one_step_score(
                sim,
                coeffs,
                bs_action=fixed_other_action,
                es_action=action,
            ) - current
            for action in candidates
        }
    else:
        raise ValueError(f"unknown stage {stage!r}")

    best = min(action_scores.values())
    oracle_actions = sorted(
        action for action, score in action_scores.items()
        if score <= best + atol
    )
    return {
        "implemented": int(implemented_action),
        "oracle_actions": oracle_actions,
        "deltas": action_scores,
    }
