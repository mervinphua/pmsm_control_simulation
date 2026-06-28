"""Field-oriented control (FOC) with cascade PI loops."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PIController:
    Kp: float
    Ki: float
    out_min: float = -math.inf
    out_max: float = math.inf
    integral: float = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def update(self, error: float, dt: float) -> float:
        # Tentative output with current integral
        u_unsat = self.Kp * error + self.Ki * (self.integral + error * dt)
        # Saturate
        u = max(self.out_min, min(self.out_max, u_unsat))
        # Conditional integration (anti-windup): integrate only if not pushing further into saturation
        if u_unsat == u or (u_unsat > self.out_max and error < 0.0) or (u_unsat < self.out_min and error > 0.0):
            self.integral += error * dt
        return u


@dataclass
class FOCController:
    # Motor electrical parameters used for gain selection (not decoupling, kept simple)
    Rs: float = 0.5
    Ld: float = 0.0015
    Lq: float = 0.0015
    Ke: float = 0.01
    # Voltage limit (magnitude of dq voltage vector)
    Vmax: float = 12.0
    # Current limit for iq reference from speed loop
    Iq_max: float = 20.0
    # Gains (tuned for ~1000 rad/s current loop, ~50 rad/s speed loop)
    Kp_i: float = 1.5
    Ki_i: float = 500.0
    Kp_w: float = 0.01
    Ki_w: float = 0.1
    # id reference (SPM BLDC)
    id_ref: float = 0.0

    speed_pi: PIController = field(init=False)
    id_pi: PIController = field(init=False)
    iq_pi: PIController = field(init=False)

    def __post_init__(self) -> None:
        self.speed_pi = PIController(self.Kp_w, self.Ki_w, -self.Iq_max, self.Iq_max)
        # Per-axis voltage limit is the full Vmax; final circle saturation handles coupling
        self.id_pi = PIController(self.Kp_i, self.Ki_i, -self.Vmax, self.Vmax)
        self.iq_pi = PIController(self.Kp_i, self.Ki_i, -self.Vmax, self.Vmax)

    def reset(self) -> None:
        self.speed_pi.reset()
        self.id_pi.reset()
        self.iq_pi.reset()

    def update(
        self,
        omega_ref: float,
        omega_m: float,
        id_: float,
        iq: float,
        theta_e: float,
        dt: float,
    ) -> tuple[float, float]:
        # Speed loop -> iq reference
        iq_ref = self.speed_pi.update(omega_ref - omega_m, dt)
        # Current loops
        vd = self.id_pi.update(self.id_ref - id_, dt)
        vq = self.iq_pi.update(iq_ref - iq, dt)
        # Voltage circle limit
        vmag = math.hypot(vd, vq)
        if vmag > self.Vmax and vmag > 0.0:
            scale = self.Vmax / vmag
            vd *= scale
            vq *= scale
        return vd, vq
