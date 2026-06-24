from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .diagnostics import build_run_results

# =====================================================================
# Corrected simulator and diagnostics
# =====================================================================

class TandemAoISimulatorV3:
    """Tandem simulator with coherent initialization and auditable counters."""

    _EVENT_NAMES = ("attempts", "successes", "arrivals", "starts", "completions")

    def __init__(self, N, L, p, mu, w, seed=0, h_init=None):
        self.N = int(N)
        self.L = int(L)
        self.mu = float(mu)
        self.p = np.asarray(p, float).copy()
        self.w = np.asarray(w, float).copy()
        if self.p.shape != (self.N,) or self.w.shape != (self.N,):
            raise ValueError("p and w must have shape (N,).")
        if np.any((self.p <= 0.0) | (self.p > 1.0)):
            raise ValueError("Each p_i must lie in (0,1].")
        if not (0.0 < self.mu <= 1.0) or self.L < 1:
            raise ValueError("Require mu in (0,1] and integer L>=1.")
        self.seed = int(seed)
        self.h_init = h_init
        self.reset()

    def reset(self, horizon: Optional[int] = None):
        N, L = self.N, self.L
        seed_seq = np.random.SeedSequence(self.seed)
        ss_gate, ss_service, ss_bs, ss_es = seed_seq.spawn(4)
        self._rng_gate = np.random.default_rng(ss_gate)
        self._rng_service = np.random.default_rng(ss_service)
        self.rng_policy_bs = np.random.default_rng(ss_bs)
        self.rng_policy_es = np.random.default_rng(ss_es)
        # Backward-compatible policy RNG. Patched policies below use stage-specific RNGs.
        self.rng = self.rng_policy_es

        self._gate_u = None
        self._service_u = None
        if horizon is not None:
            self._gate_u = self._rng_gate.random((int(horizon), N))
            self._service_u = self._rng_service.random(int(horizon))

        if self.h_init is None:
            # Coherent empty-pipeline state: bridge holds with equality.
            self.h = np.full(N, float(L + 1))
        else:
            self.h = np.asarray(self.h_init, float).copy()
            if self.h.shape != (N,):
                raise ValueError("h_init must have shape (N,).")
        self.Aq = np.full(N, float(L))
        self.V = np.zeros(N, dtype=bool)
        self.B, self.rBS = -1, 0
        self.J, self.ZS = -1, 0.0
        self.k = 0

        self.total = {name: np.zeros(N, dtype=np.int64) for name in self._EVENT_NAMES}
        self.post = {name: np.zeros(N, dtype=np.int64) for name in self._EVENT_NAMES}
        self.post["overwrites"] = np.zeros(N, dtype=np.int64)
        self.post["same_slot_refills"] = np.zeros(N, dtype=np.int64)

        self.resource = {
            "s1_gate_attempt": 0, "s1_locked": 0, "s1_idle": 0,
            "s2_start": 0, "s2_busy_continuation": 0,
            "s2_idle_empty": 0, "s2_idle_nonempty": 0,
        }
        self._collect = False
        self._mc = 0
        self._wh = self._wA = 0.0
        self._hacc = np.zeros(N)
        self._Aacc = np.zeros(N)
        self._Vacc = np.zeros(N)
        self._Qacc = np.zeros(N)
        self.min_gap = np.inf
        self.max_bridge_violation = -np.inf
        self._series = None

    def _event(self, name: str, idx: int):
        self.total[name][idx] += 1
        if self._collect:
            self.post[name][idx] += 1

    def _gate_uniform(self, source: int) -> float:
        if self._gate_u is not None:
            return float(self._gate_u[self.k, source])
        return float(self._rng_gate.random())

    def _service_uniform(self) -> float:
        if self._service_u is not None:
            return float(self._service_u[self.k])
        return float(self._rng_service.random())

    def gaps(self) -> Tuple[np.ndarray, np.ndarray]:
        Q = self.V * (self.h - self.Aq)
        Y = np.zeros(self.N)
        if self.J != -1:
            Y[self.J] = self.h[self.J] - self.ZS
        return Q, Y

    def assert_state_invariants(self, tol: float = 1e-10):
        Q, Y = self.gaps()
        if (self.B == -1) != (self.rBS == 0):
            raise AssertionError("B/rBS consistency failed.")
        if self.B != -1 and not (self.L > 1 and 1 <= self.rBS <= self.L - 1):
            raise AssertionError("Invalid locked-link countdown.")
        if self.J == -1 and abs(self.ZS) > tol:
            raise AssertionError("Idle edge must have ZS=0.")
        if np.any(self.Aq < self.L - tol):
            raise AssertionError("Aq fell below L.")
        if np.any(Q < -tol) or np.any(Y < -tol):
            raise AssertionError("A freshness gap became negative.")
        bridge_violation = self.h - (self.Aq + Q + Y + 1.0)
        if np.max(bridge_violation) > tol:
            raise AssertionError(
                f"Structural bridge violated by {np.max(bridge_violation):.3e}."
            )
        if np.any(self.h - self.Aq < 1.0 - tol):
            raise AssertionError("Expected h_i >= Aq_i+1 under this convention.")

    @staticmethod
    def _validate_action(action, N: int, name: str) -> int:
        if isinstance(action, np.integer):
            action = int(action)
        if not isinstance(action, int) or action < -1 or action >= N:
            raise AssertionError(f"{name} action must be -1 or an index in [0,N).")
        return action

    def step(self, policy):
        N, L, mu = self.N, self.L, self.mu
        h, Aq, V = self.h, self.Aq, self.V
        B, rBS, J, ZS = self.B, self.rBS, self.J, self.ZS

        uBS, uES = policy.decide(self)
        uBS = self._validate_action(uBS, N, "Stage-1")
        uES = self._validate_action(uES, N, "Stage-2")
        if B != -1:
            uBS = -1
        if J != -1:
            uES = -1
        if uES != -1 and not V[uES]:
            raise AssertionError("Stage-2 selected an empty VOQ.")

        if self._collect:
            if B == -1:
                self.resource["s1_gate_attempt" if uBS != -1 else "s1_idle"] += 1
            else:
                self.resource["s1_locked"] += 1
            if J == -1:
                if uES != -1:
                    self.resource["s2_start"] += 1
                elif V.any():
                    self.resource["s2_idle_nonempty"] += 1
                else:
                    self.resource["s2_idle_empty"] += 1
            else:
                self.resource["s2_busy_continuation"] += 1

        # Stage 1
        arrival_idx = -1
        nB, nrBS = B, rBS
        if B == -1:
            if uBS != -1:
                self._event("attempts", uBS)
                if self._gate_uniform(uBS) < self.p[uBS]:
                    self._event("successes", uBS)
                    if L == 1:
                        arrival_idx = uBS
                        nB, nrBS = -1, 0
                    else:
                        nB, nrBS = uBS, L - 1
                else:
                    nB, nrBS = -1, 0
        else:
            if rBS == 1:
                arrival_idx = B
                nB, nrBS = -1, 0
            else:
                nB, nrBS = B, rBS - 1

        # Stage 2
        removed_idx = delivered = -1
        h_reset = 0.0
        nJ, nZS = J, ZS
        if J == -1:
            if uES != -1:
                served_age = float(Aq[uES])
                removed_idx = uES
                self._event("starts", uES)
                if self._service_uniform() < mu:
                    delivered = uES
                    h_reset = served_age + 2.0
                    self._event("completions", uES)
                    nJ, nZS = -1, 0.0
                else:
                    nJ, nZS = uES, served_age + 1.0
        else:
            if self._service_uniform() < mu:
                delivered = J
                h_reset = float(ZS + 2.0)
                self._event("completions", J)
                nJ, nZS = -1, 0.0
            else:
                nJ, nZS = J, float(ZS + 1.0)

        # Next-slot VOQ state
        Aq += 1.0
        if removed_idx != -1:
            V[removed_idx] = False
        if arrival_idx != -1:
            if self._collect:
                if V[arrival_idx]:
                    self.post["overwrites"][arrival_idx] += 1
                if removed_idx == arrival_idx:
                    self.post["same_slot_refills"][arrival_idx] += 1
            self._event("arrivals", arrival_idx)
            Aq[arrival_idx] = float(L)
            V[arrival_idx] = True

        # Next-slot destination state
        h += 1.0
        if delivered != -1:
            h[delivered] = h_reset

        self.B, self.rBS, self.J, self.ZS = nB, nrBS, nJ, nZS
        self.k += 1

    def run(
        self,
        policy,
        K: int,
        warmup: int,
        *,
        record_series: bool = False,
        validate: bool = False,
    ) -> Dict[str, object]:
        K, warmup = int(K), int(warmup)
        if not (0 <= warmup < K):
            raise ValueError("Require 0 <= warmup < K.")
        self.reset(horizon=K)
        if record_series:
            self._series = {"h": [], "Aq": [], "Q": [], "V": []}

        for k in range(K):
            self._collect = k >= warmup
            if validate:
                self.assert_state_invariants()
            if self._collect:
                Q, Y = self.gaps()
                self._mc += 1
                self._wh += float((self.w * self.h).sum() / self.N)
                self._wA += float((self.w * self.Aq).sum() / self.N)
                self._hacc += self.h
                self._Aacc += self.Aq
                self._Vacc += self.V
                self._Qacc += Q
                self.min_gap = min(self.min_gap, float(np.min(self.h - self.Aq)))
                self.max_bridge_violation = max(
                    self.max_bridge_violation,
                    float(np.max(self.h - (self.Aq + Q + Y + 1.0))),
                )
                if record_series:
                    self._series["h"].append(self.h.copy())
                    self._series["Aq"].append(self.Aq.copy())
                    self._series["Q"].append(Q.copy())
                    self._series["V"].append(self.V.copy())
            self.step(policy)

        self._collect = False
        if validate:
            self.assert_state_invariants()
        return build_run_results(self, K, warmup)

