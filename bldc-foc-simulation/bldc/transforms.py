"""Clarke/Park coordinate transforms (amplitude-invariant)."""
from __future__ import annotations

import math


def clarke(ia: float, ib: float, ic: float) -> tuple[float, float]:
    # Amplitude-invariant Clarke (2/3 factor)
    ialpha = (2.0 / 3.0) * (ia - 0.5 * ib - 0.5 * ic)
    ibeta = (2.0 / 3.0) * (math.sqrt(3.0) / 2.0) * (ib - ic)
    return ialpha, ibeta


def park(ialpha: float, ibeta: float, theta: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    id_ = ialpha * c + ibeta * s
    iq = -ialpha * s + ibeta * c
    return id_, iq


def inv_park(vd: float, vq: float, theta: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    valpha = vd * c - vq * s
    vbeta = vd * s + vq * c
    return valpha, vbeta


def inv_clarke(valpha: float, vbeta: float) -> tuple[float, float, float]:
    va = valpha
    vb = -0.5 * valpha + (math.sqrt(3.0) / 2.0) * vbeta
    vc = -0.5 * valpha - (math.sqrt(3.0) / 2.0) * vbeta
    return va, vb, vc
