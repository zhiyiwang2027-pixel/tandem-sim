from __future__ import annotations

from typing import Tuple

import numpy as np

from .rate_optimizer import (
    iso1_params_v3,
    iso1_srp_params_v3,
    iso2_lambda_params_v3,
    iso2_params_v3,
    joint_params_v3,
    lb_bsside_v3,
    lb_dest_iso2_relaxed_v3,
    lb_dest_joint_v3,
    rate_to_srp_alpha,
    rate_to_srp_beta,
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


class AlphaSRPBSV3:
    """Stationary randomized Stage-1 policy with explicit idle probability."""
    def __init__(self, alpha):
        alpha = np.asarray(alpha, float)
        if alpha.ndim != 1 or np.any(~np.isfinite(alpha)) or np.any(alpha < -1e-10):
            raise ValueError("alpha must be a finite nonnegative vector.")
        alpha = np.maximum(alpha, 0.0)
        alpha_sum = float(alpha.sum())
        if alpha_sum > 1.0 + 1e-8:
            raise ValueError("alpha probabilities cannot sum above one.")
        if alpha_sum > 1.0:
            alpha = alpha / alpha_sum
            alpha_sum = 1.0
        self.alpha = alpha
        self.prob = np.append(alpha, max(0.0, 1.0 - alpha_sum))
        self.prob = self.prob / self.prob.sum()

    def bs(self, sim):
        choice = int(sim.rng_policy_bs.choice(sim.N + 1, p=self.prob))
        return -1 if choice == sim.N else choice


class WorkConservingBetaSRPESV3:
    """Work-conserving tandem SRP Stage-2 rule using beta as priority weights."""
    def __init__(self, beta):
        beta = np.asarray(beta, float)
        if beta.ndim != 1 or np.any(~np.isfinite(beta)) or np.any(beta < -1e-10):
            raise ValueError("beta must be a finite nonnegative vector.")
        beta = np.maximum(beta, 0.0)
        beta_sum = float(beta.sum())
        if beta_sum > 1.0 + 1e-8:
            raise ValueError("beta probabilities cannot sum above one.")
        if beta_sum > 1.0:
            beta = beta / beta_sum
        self.beta = beta

    def es(self, sim):
        occ = np.flatnonzero(sim.V)
        if occ.size == 0:
            return -1
        weights = self.beta[occ]
        total = float(weights.sum())
        if total <= 0.0:
            return int(sim.rng_policy_es.choice(occ))
        return int(sim.rng_policy_es.choice(occ, p=weights / total))


class UniformBSV3:
    def bs(self, sim): return int(sim.rng_policy_bs.integers(sim.N))


class UniformESV3:
    def es(self, sim):
        occ = np.flatnonzero(sim.V)
        return -1 if occ.size == 0 else int(sim.rng_policy_es.choice(occ))


class GreedyBSV3:
    """Unweighted Stage-1 greedy: serve the largest beginning-of-slot A^Q."""
    def __init__(self, w=None): self.w = None
    def bs(self, sim):
        i, _ = _random_argmax(sim.Aq, np.ones(sim.N, bool), sim.rng_policy_bs)
        return i


class GreedyESV3:
    """Unweighted Stage-2 greedy: serve the nonempty VOQ with largest h."""
    def __init__(self, w=None): self.w = None
    def es(self, sim):
        i, _ = _random_argmax(sim.h, sim.V, sim.rng_policy_es)
        return i


class WeightedGreedyBSV3:
    """Legacy weighted greedy kept only for explicit debugging comparisons."""
    def __init__(self, w): self.w = np.asarray(w, float)
    def bs(self, sim):
        i, _ = _random_argmax(self.w * sim.Aq, np.ones(sim.N, bool), sim.rng_policy_bs)
        return i


class WeightedGreedyESV3:
    """Legacy weighted greedy kept only for explicit debugging comparisons."""
    def __init__(self, w): self.w = np.asarray(w, float)
    def es(self, sim):
        i, _ = _random_argmax(self.w * sim.h, sim.V, sim.rng_policy_es)
        return i


def _freshness_gaps_for_downstream_aware_mw(sim):
    Q = sim.V * (sim.h - sim.Aq)
    Y = np.zeros(sim.N)
    if sim.J != -1:
        Y[sim.J] = float(sim.h[sim.J] - sim.ZS)
    return Q, Y, np.maximum(Q, Y)


class DownstreamAwareMWBSV3:
    """Debt-based Stage-1 DA-MW.

    Stage 1 uses the covered-copy age C_i(k), not only the current waiting
    packet, because a refresh started now reaches the VOQ only after the
    Stage-1 locked run. U_i(k) is an upstream refresh debt relative to the
    lower-bound-guided delivery target q_i^tar. The only Stage-1 cost used is
    the gated deterministic effective cost L-1+1/p_i.
    """
    def __init__(self, L, p, w, qtar, *, q_floor=1e-12):
        self.L = int(L)
        self.p = np.asarray(p, float)
        self.w = np.asarray(w, float)
        self.qtar = np.asarray(qtar, float)
        if self.p.shape != self.w.shape or self.qtar.shape != self.w.shape:
            raise ValueError("p, w, and qtar must have the same shape.")
        if np.any(~np.isfinite(self.qtar)) or np.any(self.qtar <= q_floor):
            raise ValueError("DA-MW requires strictly positive lower-bound target rates qtar.")
        self.c = (self.L - 1.0) + 1.0 / self.p
        self.theta = self.w / self.qtar
        self.reset()

    def reset(self):
        self.U = np.zeros_like(self.qtar, dtype=float)
        self._debt_sum = 0.0
        self._debt_samples = 0

    def covered_copy_age(self, sim):
        _, _, S = _freshness_gaps_for_downstream_aware_mw(sim)
        return sim.h - S

    def bs_index(self, sim):
        return (self.U + self.theta * self.covered_copy_age(sim)) / self.c

    def bs(self, sim):
        i, vmax = _random_argmax(self.bs_index(sim), np.ones(sim.N, bool), sim.rng_policy_bs)
        return i if vmax > 0.0 else -1

    def observe_slot(self, sim, *, stage1_commit_idx: int = -1, delivered_idx: int = -1):
        d = np.zeros_like(self.qtar)
        if stage1_commit_idx != -1:
            d[int(stage1_commit_idx)] = 1.0
        self.U = np.maximum(self.U + self.qtar - d, 0.0)
        self._debt_sum += float(self.U.sum())
        self._debt_samples += 1

    def debt_diagnostics(self):
        avg = self._debt_sum / self._debt_samples if self._debt_samples else 0.0
        return {"final_U_sum": float(self.U.sum()), "avg_U_sum": float(avg)}


class DownstreamAwareMWESV3:
    """Debt-based Stage-2 DA-MW.

    Stage 2 uses immediate destination AoI reduction Q_i(k), while D_i(k)
    prevents starving sources whose deliveries lag q_i^tar. The destination
    delivery indicator c_i(k) is the simulator's realized edge completion.
    """
    def __init__(self, w, qtar, *, q_floor=1e-12):
        self.w = np.asarray(w, float)
        self.qtar = np.asarray(qtar, float)
        if self.qtar.shape != self.w.shape:
            raise ValueError("w and qtar must have the same shape.")
        if np.any(~np.isfinite(self.qtar)) or np.any(self.qtar <= q_floor):
            raise ValueError("DA-MW requires strictly positive lower-bound target rates qtar.")
        self.theta = self.w / self.qtar
        self.reset()

    def reset(self):
        self.D = np.zeros_like(self.qtar, dtype=float)
        self._debt_sum = 0.0
        self._debt_samples = 0

    def es_index(self, sim):
        Q, _, _ = _freshness_gaps_for_downstream_aware_mw(sim)
        return self.D + self.theta * Q

    def es(self, sim):
        i, vmax = _random_argmax(self.es_index(sim), sim.V, sim.rng_policy_es)
        return i if vmax > 0.0 else -1

    def observe_slot(self, sim, *, stage1_commit_idx: int = -1, delivered_idx: int = -1):
        c = np.zeros_like(self.qtar)
        if delivered_idx != -1:
            c[int(delivered_idx)] = 1.0
        self.D = np.maximum(self.D + self.qtar - c, 0.0)
        self._debt_sum += float(self.D.sum())
        self._debt_samples += 1

    def debt_diagnostics(self):
        avg = self._debt_sum / self._debt_samples if self._debt_samples else 0.0
        return {"final_D_sum": float(self.D.sum()), "avg_D_sum": float(avg)}


class ComposedV3:
    def __init__(self, bs_ctrl, es_ctrl, name=""):
        self.bs_ctrl, self.es_ctrl, self.name = bs_ctrl, es_ctrl, name
    def decide(self, sim):
        return (
            self.bs_ctrl.bs(sim) if sim.B == -1 else -1,
            self.es_ctrl.es(sim) if sim.J == -1 else -1,
        )
    def reset(self):
        for ctrl in (self.bs_ctrl, self.es_ctrl):
            if hasattr(ctrl, "reset"):
                ctrl.reset()
    def observe_slot(self, sim, *, stage1_commit_idx: int = -1, delivered_idx: int = -1):
        for ctrl in (self.bs_ctrl, self.es_ctrl):
            if hasattr(ctrl, "observe_slot"):
                ctrl.observe_slot(
                    sim,
                    stage1_commit_idx=stage1_commit_idx,
                    delivered_idx=delivered_idx,
                )
    def diagnostics(self):
        out = {}
        for prefix, ctrl in (("bs", self.bs_ctrl), ("es", self.es_ctrl)):
            if hasattr(ctrl, "debt_diagnostics"):
                out[prefix] = ctrl.debt_diagnostics()
        return out


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
        "Greedy": ComposedV3(GreedyBSV3(), GreedyESV3(), "Greedy"),
        "Downstream-Aware MW": ComposedV3(
            DownstreamAwareMWBSV3(L, p, w, jp["qd"]),
            DownstreamAwareMWESV3(w, jp["qd"]),
            "Downstream-Aware MW",
        ),
    }
    return pol, dict(joint=jp, iso1=i1, iso2=i2)


def build_iso1_iso2_lambda_policy_v3(N, L, p, mu, w, lambda_cap):
    """Build the pilot-estimated lambda-aware isolated MW baseline."""
    i1 = iso1_params_v3(N, L, p, w)
    i2_lambda = iso2_lambda_params_v3(N, L, p, mu, w, lambda_cap)
    return ComposedV3(IsoBSV3(i1), IsoESV3(i2_lambda), "iso1 + iso2-lambda")


def build_srp_iso_policy_v3(N, L, p, mu, w, lambda_cap):
    """Build the SRP-iso simulation baseline."""
    i1_srp = iso1_srp_params_v3(N, L, p, w)
    i2_lambda = iso2_lambda_params_v3(N, L, p, mu, w, lambda_cap)
    beta = rate_to_srp_beta(i2_lambda["qd"], mu)
    return ComposedV3(
        AlphaSRPBSV3(i1_srp["alpha_srp_iso1"]),
        WorkConservingBetaSRPESV3(beta),
        "SRP-iso",
    )


def build_srp_tandem_lb_policy_v3(N, L, p, mu, w, joint_params=None):
    """Build the SRP baseline induced by the tandem lower-bound target rates."""
    jp = joint_params_v3(N, L, p, mu, w) if joint_params is None else joint_params
    alpha = rate_to_srp_alpha(jp["qd"], p, L)
    beta = rate_to_srp_beta(jp["qd"], mu)
    return ComposedV3(
        AlphaSRPBSV3(alpha),
        WorkConservingBetaSRPESV3(beta),
        "SRP-tandem-LB",
    )


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
