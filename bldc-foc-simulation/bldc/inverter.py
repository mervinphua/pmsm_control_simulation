"""Three-phase inverter with space vector PWM (SVPWM)."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ThreePhaseInverter:
    Vdc: float = 48.0

    def _sector(self, valpha: float, vbeta: float) -> int:
        # Determine SVPWM sector (1..6) from alpha-beta reference
        angle = math.atan2(vbeta, valpha)
        if angle < 0.0:
            angle += 2.0 * math.pi
        sector = int(angle // (math.pi / 3.0)) + 1
        if sector > 6:
            sector = 6
        return sector

    def svpwm(self, valpha: float, vbeta: float) -> tuple[float, float, float]:
        # Min/max common-mode injection SVPWM (equivalent to sector-based method)
        va = valpha
        vb = -0.5 * valpha + (math.sqrt(3.0) / 2.0) * vbeta
        vc = -0.5 * valpha - (math.sqrt(3.0) / 2.0) * vbeta

        vmax = max(va, vb, vc)
        vmin = min(va, vb, vc)
        vcm = 0.5 * (vmax + vmin)

        va -= vcm
        vb -= vcm
        vc -= vcm

        # Map phase voltages to duty in [0, 1]
        da = 0.5 + va / self.Vdc
        db = 0.5 + vb / self.Vdc
        dc = 0.5 + vc / self.Vdc

        da = min(1.0, max(0.0, da))
        db = min(1.0, max(0.0, db))
        dc = min(1.0, max(0.0, dc))
        return da, db, dc

    def apply(self, da: float, db: float, dc: float) -> tuple[float, float, float]:
        # Phase voltages referenced to the DC-bus midpoint
        va = (da - 0.5) * self.Vdc
        vb = (db - 0.5) * self.Vdc
        vc = (dc - 0.5) * self.Vdc
        return va, vb, vc

    def sector(self, valpha: float, vbeta: float) -> int:
        return self._sector(valpha, vbeta)
