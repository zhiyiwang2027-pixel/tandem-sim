from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TandemConfig:
    N: int
    L: int
    p: np.ndarray
    mu: float
    w: np.ndarray


def make_symmetric_network(p, w_high, L, mu, N):
    N = int(N)
    half = N // 2
    return (
        N,
        int(L),
        np.full(N, float(p)),
        float(mu),
        np.r_[np.ones(half), np.full(N - half, float(w_high))],
    )


def make_config(N: int, L: int, p: Sequence[float], mu: float, w: Sequence[float]) -> TandemConfig:
    return TandemConfig(
        N=int(N),
        L=int(L),
        p=np.asarray(p, float),
        mu=float(mu),
        w=np.asarray(w, float),
    )
