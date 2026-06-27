from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# =====================================================================
# Reliable convex rate solver
# =====================================================================

def _bisect_decreasing(
    fun,
    target: float,
    *,
    lo: float = 0.0,
    hi: float = 1.0,
    rtol: float = 1e-12,
    atol: float = 1e-13,
    maxiter: int = 250,
) -> float:
    """Solve fun(x)=target for a continuous decreasing function."""
    flo = float(fun(lo) - target)
    if flo <= 0.0:
        return lo

    fhi = float(fun(hi) - target)
    for _ in range(maxiter):
        if fhi <= 0.0:
            break
        hi *= 2.0
        fhi = float(fun(hi) - target)
    else:
        raise RuntimeError("Could not bracket a decreasing KKT root.")

    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fmid = float(fun(mid) - target)
        if fmid > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= atol + rtol * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


def solve_rate_kkt(
    w: Sequence[float],
    g_lin: Sequence[float],
    *,
    link_c: Optional[Sequence[float]] = None,
    link_cap: float = 1.0,
    edge_cap: Optional[float] = None,
    feasibility_tol: float = 1e-9,
) -> Tuple[np.ndarray, Dict[str, float]]:
    r"""Solve

        min_q sum_i [w_i/(2q_i) + g_i q_i]

    subject to ``link_c @ q <= link_cap`` (optional),
    ``sum(q) <= edge_cap`` (optional), and q_i>0.

    KKT gives

        q_i = sqrt(w_i / (2(g_i + lambda*link_c_i + nu))).

    The two nonnegative multipliers are found by nested monotone bisection.
    """
    w = np.asarray(w, dtype=float)
    g = np.asarray(g_lin, dtype=float)
    if w.ndim != 1 or g.shape != w.shape:
        raise ValueError("w and g_lin must be one-dimensional and equally sized.")
    if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("All weights must be finite and strictly positive.")
    if np.any(~np.isfinite(g)) or np.any(g < 0.0):
        raise ValueError("All linear slopes must be finite and nonnegative.")

    c = None if link_c is None else np.asarray(link_c, dtype=float)
    if c is not None and (c.shape != w.shape or np.any(~np.isfinite(c)) or np.any(c <= 0.0)):
        raise ValueError("link_c must be finite, positive, and have the same shape as w.")
    if c is not None and link_cap <= 0.0:
        raise ValueError("link_cap must be positive.")
    if edge_cap is not None and edge_cap <= 0.0:
        raise ValueError("edge_cap must be positive.")

    def q_of(lam: float, nu: float) -> np.ndarray:
        penalty = g + nu
        if c is not None:
            penalty = penalty + lam * c
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(w / (2.0 * penalty))

    def edge_adjusted(lam: float) -> Tuple[np.ndarray, float]:
        q0 = q_of(lam, 0.0)
        if edge_cap is None or (np.all(np.isfinite(q0)) and q0.sum() <= edge_cap):
            return q0, 0.0
        nu = _bisect_decreasing(lambda x: float(q_of(lam, x).sum()), edge_cap)
        return q_of(lam, nu), nu

    q, nu = edge_adjusted(0.0)
    lam = 0.0
    if c is not None and (not np.all(np.isfinite(q)) or float(c @ q) > link_cap):
        lam = _bisect_decreasing(
            lambda x: float(c @ edge_adjusted(x)[0]),
            link_cap,
        )
        q, nu = edge_adjusted(lam)

    link_use = np.nan if c is None else float(c @ q)
    edge_use = float(q.sum())
    if np.any(~np.isfinite(q)) or np.any(q <= 0.0):
        raise RuntimeError("Rate solver returned a nonpositive or nonfinite rate.")
    if c is not None and link_use > link_cap + feasibility_tol:
        raise RuntimeError(f"Link constraint violation: {link_use-link_cap:.3e}")
    if edge_cap is not None and edge_use > edge_cap + feasibility_tol:
        raise RuntimeError(f"Edge constraint violation: {edge_use-edge_cap:.3e}")

    stationarity = -w / (2.0 * q * q) + g + nu
    if c is not None:
        stationarity += lam * c
    scale = max(1.0, float(np.max(g + nu + (0.0 if c is None else lam * c))))
    if float(np.max(np.abs(stationarity))) > 1e-7 * scale:
        raise RuntimeError("KKT stationarity residual is too large.")

    info = {
        "lambda_link": float(lam),
        "nu_edge": float(nu),
        "link_usage": link_use,
        "edge_usage": edge_use,
    }
    return q, info


def solve_capped_edge_rate_kkt(
    w: Sequence[float],
    g_lin: Sequence[float],
    lambda_cap: Sequence[float],
    *,
    edge_cap: float,
    feasibility_tol: float = 1e-9,
) -> Tuple[np.ndarray, Dict[str, object]]:
    r"""Solve an edge-only capped rate program.

        min_q sum_i [w_i/(2q_i) + g_i q_i]

    subject to ``0 < q_i <= lambda_cap_i`` and ``sum_i q_i <= edge_cap``.

    KKT gives ``q_i(nu)=min(lambda_i, sqrt(w_i/(2(g_i+nu))))``.
    The nonnegative edge multiplier ``nu`` is found by monotone bisection
    when the edge constraint binds.
    """
    w = np.asarray(w, dtype=float)
    g = np.asarray(g_lin, dtype=float)
    lam_cap = np.asarray(lambda_cap, dtype=float)
    if w.ndim != 1 or g.shape != w.shape or lam_cap.shape != w.shape:
        raise ValueError("w, g_lin, and lambda_cap must be one-dimensional and equally sized.")
    if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("All weights must be finite and strictly positive.")
    if np.any(~np.isfinite(g)) or np.any(g < 0.0):
        raise ValueError("All linear slopes must be finite and nonnegative.")
    if np.any(~np.isfinite(lam_cap)) or np.any(lam_cap <= 0.0):
        raise ValueError("All lambda caps must be finite and strictly positive.")
    if edge_cap <= 0.0:
        raise ValueError("edge_cap must be positive.")

    def q_of(nu: float) -> np.ndarray:
        penalty = g + nu
        with np.errstate(divide="ignore", invalid="ignore"):
            q_unconstrained = np.sqrt(w / (2.0 * penalty))
        return np.minimum(lam_cap, q_unconstrained)

    q0 = q_of(0.0)
    if np.all(np.isfinite(q0)) and float(q0.sum()) <= edge_cap:
        nu = 0.0
        q = q0
    else:
        nu = _bisect_decreasing(lambda x: float(q_of(x).sum()), edge_cap)
        q = q_of(nu)

    edge_use = float(q.sum())
    cap_slack = lam_cap - q
    cap_active = cap_slack <= 1e-10 * np.maximum(1.0, lam_cap)
    cap_dual_like = np.where(
        cap_active,
        np.maximum(w / (2.0 * lam_cap * lam_cap) - g - nu, 0.0),
        0.0,
    )

    if np.any(~np.isfinite(q)) or np.any(q <= 0.0):
        raise RuntimeError("Capped rate solver returned a nonpositive or nonfinite rate.")
    if np.any(q > lam_cap + feasibility_tol):
        raise RuntimeError("Capped rate solver violated a lambda cap.")
    if edge_use > edge_cap + feasibility_tol:
        raise RuntimeError(f"Edge constraint violation: {edge_use-edge_cap:.3e}")

    interior = ~cap_active
    if np.any(interior):
        stationarity = -w[interior] / (2.0 * q[interior] * q[interior]) + g[interior] + nu
        scale = max(1.0, float(np.max(g[interior] + nu)))
        if float(np.max(np.abs(stationarity))) > 1e-7 * scale:
            raise RuntimeError("Capped KKT stationarity residual is too large.")

    info: Dict[str, object] = {
        "nu_edge": float(nu),
        "edge_usage": edge_use,
        "edge_cap": float(edge_cap),
        "edge_slack": float(edge_cap - edge_use),
        "lambda_cap": lam_cap.copy(),
        "cap_slack": cap_slack,
        "cap_active": cap_active,
        "cap_active_count": int(cap_active.sum()),
        "cap_active_frac": float(cap_active.mean()),
        "cap_dual_like": cap_dual_like,
        "cap_dual_like_sum": float(cap_dual_like.sum()),
    }
    return q, info


def _clip_tiny_negative(values: Sequence[float], *, name: str, tol: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    if np.any(arr < -tol):
        raise ValueError(f"{name} contains entries below the numerical tolerance.")
    return np.maximum(arr, 0.0)


def rate_to_srp_alpha(
    q: Sequence[float],
    p: Sequence[float],
    L: int,
    *,
    tol: float = 1e-8,
) -> np.ndarray:
    r"""Map feasible Stage-1 VOQ arrival rates to stationary SRP probabilities.

    The mapping is

        alpha_i = q_i / [p_i * (1 - (L-1) sum_j q_j)].

    It is valid when ``sum_i ((L-1)+1/p_i) q_i <= 1``.
    """
    q = _clip_tiny_negative(q, name="q", tol=tol)
    p = np.asarray(p, dtype=float)
    if q.ndim != 1 or p.shape != q.shape:
        raise ValueError("q and p must be one-dimensional and equally sized.")
    if np.any(~np.isfinite(p)) or np.any((p <= 0.0) | (p > 1.0)):
        raise ValueError("Each p_i must lie in (0,1].")
    if int(L) < 1:
        raise ValueError("L must be at least 1.")

    c = (int(L) - 1.0) + 1.0 / p
    link_usage = float(c @ q)
    if link_usage > 1.0 + tol:
        raise ValueError(f"SRP alpha mapping requires link-feasible q; usage={link_usage:.12g}.")

    denom = 1.0 - (int(L) - 1.0) * float(q.sum())
    if denom <= 0.0:
        raise ValueError("SRP alpha denominator is nonpositive.")

    alpha = _clip_tiny_negative(q / (p * denom), name="alpha", tol=tol)
    alpha_sum = float(alpha.sum())
    if alpha_sum > 1.0 + tol:
        raise ValueError(f"SRP alpha probabilities sum to {alpha_sum:.12g}.")
    if alpha_sum > 1.0:
        alpha = alpha / alpha_sum
    return alpha


def rate_to_srp_beta(
    q: Sequence[float],
    mu: float,
    *,
    tol: float = 1e-8,
) -> np.ndarray:
    r"""Map feasible Stage-2 target delivery rates to SRP beta weights."""
    q = _clip_tiny_negative(q, name="q", tol=tol)
    mu = float(mu)
    if q.ndim != 1:
        raise ValueError("q must be one-dimensional.")
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be positive.")
    if float(q.sum()) > mu + tol:
        raise ValueError("Stage-2 target rates exceed the edge capacity.")

    beta = _clip_tiny_negative(q / mu, name="beta", tol=tol)
    beta_sum = float(beta.sum())
    if beta_sum > 1.0 + tol:
        raise ValueError(f"SRP beta probabilities sum to {beta_sum:.12g}.")
    if beta_sum > 1.0:
        beta = beta / beta_sum
    return beta


def solve_iso1_srp_rate_kkt(
    L: int,
    p: Sequence[float],
    w: Sequence[float],
    *,
    feasibility_tol: float = 1e-9,
) -> Tuple[np.ndarray, Dict[str, object]]:
    r"""Solve the exact isolated Stage-1 SRP rate program.

    With ``a_i = L-1+1/p_i`` and
    ``kappa = L(L-1)/2 * sum_i w_i``, KKT gives

        q_i(nu) = sqrt(w_i / (kappa + nu a_i)).
    """
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    L = int(L)
    if p.ndim != 1 or w.shape != p.shape:
        raise ValueError("p and w must be one-dimensional and equally sized.")
    if np.any(~np.isfinite(p)) or np.any((p <= 0.0) | (p > 1.0)):
        raise ValueError("Each p_i must lie in (0,1].")
    if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("All weights must be finite and strictly positive.")
    if L < 1:
        raise ValueError("L must be at least 1.")

    c = (L - 1.0) + 1.0 / p
    kappa = 0.5 * L * (L - 1.0) * float(w.sum())

    def q_of(nu: float) -> np.ndarray:
        penalty = kappa + float(nu) * c
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(w / penalty)

    q0 = q_of(0.0)
    if np.all(np.isfinite(q0)) and float(c @ q0) <= 1.0:
        nu = 0.0
        q = q0
    else:
        nu = _bisect_decreasing(lambda x: float(c @ q_of(x)), 1.0)
        q = q_of(nu)

    link_usage = float(c @ q)
    if np.any(~np.isfinite(q)) or np.any(q <= 0.0):
        raise RuntimeError("Isolated Stage-1 SRP solver returned a nonpositive or nonfinite rate.")
    if link_usage > 1.0 + feasibility_tol:
        raise RuntimeError(f"Isolated Stage-1 SRP link violation: {link_usage-1.0:.3e}")

    alpha = rate_to_srp_alpha(q, p, L)
    info: Dict[str, object] = {
        "nu_iso1_srp": float(nu),
        "link_usage": link_usage,
        "link_slack": float(1.0 - link_usage),
        "link_binds": bool(abs(1.0 - link_usage) <= 1e-8),
        "alpha_sum": float(alpha.sum()),
        "kappa": float(kappa),
    }
    return q, info


def joint_params_v3(N, L, p, mu, w):
    p = np.asarray(p, float)
    w = np.asarray(w, float)
    c = (L - 1.0) + 1.0 / p
    g = w * (1.0 - mu) / mu**2
    qd, dual = solve_rate_kkt(w, g, link_c=c, link_cap=1.0, edge_cap=mu)
    a = float((qd / p).sum())
    b = float(qd.sum())
    ell = a + (L - 1.0) * b
    delta = np.maximum(b / (mu * qd) + (1.0 - mu) / mu - (2.0 - mu), 0.0)
    return dict(
        kind="joint", qd=qd, a=a, b=b, ell=ell, delta=delta,
        p=p.copy(), w=w.copy(), c=c, L=int(L), mu=float(mu), dual=dual,
        vR=L * w, vP=ell * w / qd, vA=ell * w / qd,
        vE=w / mu, vQ=(1.0 - mu) / mu * w,
        vY=delta * w, vh=((2.0 - mu) / mu + delta) * w,
    )


def iso1_params_v3(N, L, p, w):
    p = np.asarray(p, float)
    w = np.asarray(w, float)
    c = (L - 1.0) + 1.0 / p
    qd = np.sqrt(w / c) / np.sqrt(w * c).sum()
    b = float(qd.sum())
    return dict(
        kind="iso1", qd=qd, b=b, p=p.copy(), w=w.copy(), c=c,
        L=int(L), vR=L * w,
        vP=(1.0 + (L - 1.0) * b) * w / qd,
        vA=(1.0 + (L - 1.0) * b) * w / qd,
    )


def iso1_srp_params_v3(N, L, p, w):
    """Exact isolated Stage-1 stationary randomized policy tuning."""
    p = np.asarray(p, float)
    w = np.asarray(w, float)
    if p.shape != (int(N),) or w.shape != (int(N),):
        raise ValueError("p and w must have shape (N,).")
    c = (int(L) - 1.0) + 1.0 / p
    q, info = solve_iso1_srp_rate_kkt(L, p, w)
    alpha = rate_to_srp_alpha(q, p, L)
    return dict(
        kind="iso1_srp",
        qd=q,
        q_srp_iso1=q,
        alpha_srp_iso1=alpha,
        alpha=alpha,
        b=float(q.sum()),
        p=p.copy(),
        w=w.copy(),
        c=c,
        L=int(L),
        dual=info,
        nu_iso1_srp=float(info["nu_iso1_srp"]),
        link_binds=bool(info["link_binds"]),
    )


def iso2_params_v3(N, L, p, mu, w):
    """Lambda-agnostic relaxed Stage-2 certificate: only sum_i q_i <= mu."""
    w = np.asarray(w, float)
    g = w * (1.0 - mu) / mu**2
    qd, dual = solve_rate_kkt(w, g, edge_cap=mu)
    b = float(qd.sum())
    A = mu + (1.0 - mu) * b / mu
    return dict(
        kind="iso2_relaxed", qd=qd, b=b, A=A, w=w.copy(), L=int(L),
        mu=float(mu), dual=dual,
        vE=w / mu, vQ=(1.0 - mu) / mu * w,
        vY=(A / qd - 1.0 + (1.0 - mu)**2 / mu) * w,
        vh=(A / qd + (3.0 - 4.0 * mu + mu**2) / mu) * w,
    )


def iso2_lambda_params_v3(N, L, p, mu, w, lambda_cap):
    """Pilot-estimated lambda-aware isolated Stage-2 tuning."""
    w = np.asarray(w, float)
    lambda_cap = np.asarray(lambda_cap, float)
    g = w * (1.0 - mu) / mu**2
    qd, dual = solve_capped_edge_rate_kkt(w, g, lambda_cap, edge_cap=mu)
    b = float(qd.sum())
    A = mu + (1.0 - mu) * b / mu
    return dict(
        kind="iso2_lambda",
        lambda_cap=lambda_cap.copy(),
        qd=qd,
        b=b,
        A=A,
        w=w.copy(),
        L=int(L),
        mu=float(mu),
        dual=dual,
        vE=w / mu,
        vQ=(1.0 - mu) / mu * w,
        vY=(A / qd - 1.0 + (1.0 - mu)**2 / mu) * w,
        vh=(A / qd + (3.0 - 4.0 * mu + mu**2) / mu) * w,
    )

# ---------------------------------------------------------------------
# Lower-bound helpers and a convenience experiment factory
# ---------------------------------------------------------------------
def lb_bsside_v3(N, L, p, w, P=None):
    """Universal BS-side/VOQ-age lower bound, normalized by N."""
    P = iso1_params_v3(N, L, p, w) if P is None else P
    q = P["qd"]
    return float((P["w"] * (L - 0.5 + 1.0 / (2.0 * q))).sum() / N)


def lb_dest_joint_v3(N, L, p, mu, w, P=None):
    """Joint tandem destination-AoI lower bound, normalized by N."""
    P = joint_params_v3(N, L, p, mu, w) if P is None else P
    q = P["qd"]
    return float((P["w"] * (1.0 / (2.0 * q) + L + 1.5 + (1.0 - mu) * q / mu**2)).sum() / N)


def lb_dest_iso2_relaxed_v3(N, L, p, mu, w, P=None):
    """Edge-only lambda-agnostic relaxed lower bound, normalized by N."""
    P = iso2_params_v3(N, L, p, mu, w) if P is None else P
    q = P["qd"]
    return float((P["w"] * (1.0 / (2.0 * q) + L + 1.5 + (1.0 - mu) * q / mu**2)).sum() / N)
