from __future__ import annotations

from typing import Tuple

import numpy as np

from .rate_optimizer import (
    iso1_params_v3,
    iso2_lambda_params_v3,
    iso2_params_v3,
    joint_params_v3,
    lb_bsside_v3,
    lb_dest_iso2_relaxed_v3,
    lb_dest_joint_v3,
)

# =====================================================================
# Controllers with stage-specific policy RNG and unbiased tie-breaking
# =====================================================================

def _random_argmax(values: np.ndarray, feasible: np.ndarray, rng) -> Tuple[int, float]:
    masked = np.where(feasible, values, -np.inf)
    vmax = float(np.max(masked))
    if not np.isfinite(vmax):
        return -1, vmax
    ties = np.flatnonzero(feasible & np.isclose(values, vmax, rtol=1e-12, atol=1e-12))
    return int(rng.choice(ties)), vmax


class JointBSV3:
    def __init__(self, P): self.P = P
    def bs(self, sim):
        P, Aq = self.P, sim.Aq
        W = P["p"] * (P["vP"] * Aq - (P["L"] - 1.0) / P["L"] * (P["vR"] * Aq).sum())
        i, vmax = _random_argmax(W, np.ones(sim.N, bool), sim.rng_policy_bs)
        return i if vmax > 0.0 else -1


class IsoBSV3:
    def __init__(self, P): self.P = P
    def bs(self, sim):
        P, Aq = self.P, sim.Aq
        W = P["p"] * (P["vP"] * Aq - (P["L"] - 1.0) * (P["w"] * Aq).sum())
        i, vmax = _random_argmax(W, np.ones(sim.N, bool), sim.rng_policy_bs)
        return i if vmax > 0.0 else -1


def _es_maxweight_v3(sim, weight, vE):
    Q = sim.V * (sim.h - sim.Aq)
    common = (1.0 - sim.mu) * float((vE * Q).sum())
    W = weight * Q - common
    i, vmax = _random_argmax(W, sim.V, sim.rng_policy_es)
    return i if vmax > 0.0 else -1


class JointESV3:
    def __init__(self, P):
        self.P = P
        self.weight = P["mu"] * P["vh"] + (1.0 - P["mu"]) * P["vY"] - P["vQ"]
    def es(self, sim): return _es_maxweight_v3(sim, self.weight, self.P["vE"])


class IsoESV3:
    def __init__(self, P):
        self.P = P
        self.weight = P["A"] * P["w"] / P["qd"]
    def es(self, sim): return _es_maxweight_v3(sim, self.weight, self.P["vE"])


class SRPBSV3:
    def __init__(self, qd, p):
        a = np.asarray(qd, float) / np.asarray(p, float)
        self.alpha = a / a.sum()
    def bs(self, sim): return int(sim.rng_policy_bs.choice(sim.N, p=self.alpha))


class TheoremSRPESV3:
    """Sample all sources by beta and idle when the sampled VOQ is empty."""
    def __init__(self, qd):
        b = np.asarray(qd, float)
        self.beta = b / b.sum()
    def es(self, sim):
        i = int(sim.rng_policy_es.choice(sim.N, p=self.beta))
        return i if sim.V[i] else -1


class WorkConservingSRPESV3:
    """Age-blind randomized service, renormalized over currently nonempty VOQs."""
    def __init__(self, qd):
        b = np.asarray(qd, float)
        self.beta = b / b.sum()
    def es(self, sim):
        occ = np.flatnonzero(sim.V)
        if occ.size == 0:
            return -1
        prob = self.beta[occ]
        prob = prob / prob.sum()
        return int(sim.rng_policy_es.choice(occ, p=prob))


class UniformBSV3:
    def bs(self, sim): return int(sim.rng_policy_bs.integers(sim.N))


class UniformESV3:
    def es(self, sim):
        occ = np.flatnonzero(sim.V)
        return -1 if occ.size == 0 else int(sim.rng_policy_es.choice(occ))


class GreedyBSV3:
    def __init__(self, w): self.w = np.asarray(w, float)
    def bs(self, sim):
        i, _ = _random_argmax(self.w * sim.Aq, np.ones(sim.N, bool), sim.rng_policy_bs)
        return i


class GreedyESV3:
    def __init__(self, w): self.w = np.asarray(w, float)
    def es(self, sim):
        i, _ = _random_argmax(self.w * sim.h, sim.V, sim.rng_policy_es)
        return i


class ComposedV3:
    def __init__(self, bs_ctrl, es_ctrl, name=""):
        self.bs_ctrl, self.es_ctrl, self.name = bs_ctrl, es_ctrl, name
    def decide(self, sim):
        return (
            self.bs_ctrl.bs(sim) if sim.B == -1 else -1,
            self.es_ctrl.es(sim) if sim.J == -1 else -1,
        )


def build_policies_v3(N, L, p, mu, w, *, allow_uncertified_L1=False):
    if L == 1 and not allow_uncertified_L1:
        raise ValueError(
            "The supplied joint/isolated Stage-1 derivations use L>1 recursions. "
            "Use L>=2, or explicitly set allow_uncertified_L1=True for an empirical extension."
        )
    jp = joint_params_v3(N, L, p, mu, w)
    i1 = iso1_params_v3(N, L, p, w)
    i2 = iso2_params_v3(N, L, p, mu, w)
    pol = {
        "Joint FGMW": ComposedV3(JointBSV3(jp), JointESV3(jp), "Joint FGMW"),
        "iso1 + iso2": ComposedV3(IsoBSV3(i1), IsoESV3(i2), "iso1 + iso2"),
        "iso1 + WC-SRP2": ComposedV3(IsoBSV3(i1), WorkConservingSRPESV3(i2["qd"]), "iso1 + WC-SRP2"),
        "iso1 + theorem-SRP2": ComposedV3(IsoBSV3(i1), TheoremSRPESV3(i2["qd"]), "iso1 + theorem-SRP2"),
        "SRP1 + iso2": ComposedV3(SRPBSV3(i1["qd"], p), IsoESV3(i2), "SRP1 + iso2"),
        "Uniform": ComposedV3(UniformBSV3(), UniformESV3(), "Uniform"),
        "Greedy": ComposedV3(GreedyBSV3(w), GreedyESV3(w), "Greedy"),
    }
    return pol, dict(joint=jp, iso1=i1, iso2=i2)


def build_iso1_iso2_lambda_policy_v3(N, L, p, mu, w, lambda_cap):
    """Build the pilot-estimated lambda-aware isolated MW baseline."""
    i1 = iso1_params_v3(N, L, p, w)
    i2_lambda = iso2_lambda_params_v3(N, L, p, mu, w, lambda_cap)
    return ComposedV3(IsoBSV3(i1), IsoESV3(i2_lambda), "iso1 + iso2-lambda")


def build_experiment_v3(N, L, p, mu, w, *, allow_uncertified_L1=False):
    """Build policies, Lyapunov parameters, and all lower-bound scalars."""
    policies, params = build_policies_v3(
        N, L, p, mu, w, allow_uncertified_L1=allow_uncertified_L1
    )
    params["lb_bsside"] = lb_bsside_v3(N, L, p, w, params["iso1"])
    params["lb_dest_joint"] = lb_dest_joint_v3(N, L, p, mu, w, params["joint"])
    params["lb_dest_iso2_relaxed"] = lb_dest_iso2_relaxed_v3(
        N, L, p, mu, w, params["iso2"]
    )
    return policies, params
