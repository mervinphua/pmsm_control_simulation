"""BLDC/PMSM motor model in the synchronous dq frame."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BLDCMotor:
    Rs: float = 0.5
    Ld: float = 0.0015
    Lq: float = 0.0015
    Ke: float = 0.01
    J: float = 0.0001
    B: float = 0.0001
    P: int = 4
    TL: float = 0.0

    id: float = 0.0
    iq: float = 0.0
    omega_m: float = 0.0
    theta_e: float = 0.0

    def step(self, vd: float, vq: float, dt: float) -> dict[str, float]:
        # dq-frame voltage equations
        did_dt = (vd - self.Rs * self.id + self.P * self.omega_m * self.Lq * self.iq) / self.Ld
        diq_dt = (vq - self.Rs * self.iq - self.P * self.omega_m * (self.Ld * self.id + self.Ke)) / self.Lq

        # Electromagnetic torque (with reluctance term)
        Te = 1.5 * self.P * (self.Ke * self.iq + (self.Ld - self.Lq) * self.id * self.iq)

        # Mechanical dynamics
        domega_dt = (Te - self.B * self.omega_m - self.TL) / self.J
        dtheta_e_dt = self.P * self.omega_m

        # Forward Euler integration
        self.id += did_dt * dt
        self.iq += diq_dt * dt
        self.omega_m += domega_dt * dt
        self.theta_e = (self.theta_e + dtheta_e_dt * dt) % (2.0 * math.pi)

        return {
            "id": self.id,
            "iq": self.iq,
            "omega_m": self.omega_m,
            "theta_e": self.theta_e,
            "Te": Te,
        }
