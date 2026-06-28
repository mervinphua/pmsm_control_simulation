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


# ============================================================
# Conventional FCS-MPCC (uses motor parameters in prediction)
# Reference: Zhang et al., IEEE TIE, 2024, Eq.(1)-(7)
# ============================================================

@dataclass
class ConventionalMPCCController:
    """
    Conventional FCS-MPCC - uses motor parameters R, L, psi_f
    in the prediction model. This is the BASELINE that the paper
    compares MPF-MPCC against.
    
    When parameters are wrong, prediction is inaccurate and 
    performance degrades → this is the key weakness that MPF-MPCC solves.
    """
    Ts: float                   # Control period [s]
    Udc: float = 24.0           # DC-bus voltage [V]
    P: int = 4                  # Pole pairs
    
    # --- Motor parameters (USED in prediction model) ---
    Rs: float = 0.5             # Stator resistance [ohm]
    Ls: float = 0.0015          # Stator inductance (Ld=Lq for SPMSM) [H]
    psi_f: float = 0.01         # Permanent magnet flux linkage [Wb]
    
    # --- Speed PI ---
    Kp_w: float = 0.01
    Ki_w: float = 0.1
    Iq_max: float = 20.0
    Vmax: float = 12.0
    
    def __post_init__(self) -> None:
        self.speed_pi = PIController(self.Kp_w, self.Ki_w, -self.Iq_max, self.Iq_max)
        
        # 8 switching states
        self._switch_states = [
            (0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,1,1),(0,0,1),(1,0,1),(1,1,1)]
        
        # Previously applied voltage (for delay compensation)
        self._prev_vd = 0.0
        self._prev_vq = 0.0
    
    @staticmethod
    def _switch_to_dq(sw, Udc, theta_e):
        sa, sb, sc = sw
        va = (2*sa - sb - sc) / 3.0 * Udc
        vb = (2*sb - sa - sc) / 3.0 * Udc
        vc = (2*sc - sa - sb) / 3.0 * Udc
        v_alpha = (2/3)*(va - 0.5*vb - 0.5*vc)
        v_beta  = (2/3)*(math.sqrt(3)/2*vb - math.sqrt(3)/2*vc)
        cos_t, sin_t = math.cos(theta_e), math.sin(theta_e)
        vd =  v_alpha*cos_t + v_beta*sin_t
        vq = -v_alpha*sin_t + v_beta*cos_t
        return vd, vq
    
    def update(self, omega_ref, omega_m, id_k, iq_k, theta_e, dt):
        omega_e = self.P * omega_m
        Ts = self.Ts
        
        # Speed loop PI
        iq_ref = self.speed_pi.update(omega_ref - omega_m, dt)
        id_ref = 0.0
        
        # --- Eq.(2)+(5): D = [[1-RTs/L, +Tsωe], [-Tsωe, 1-RTs/L]]
        d11 = 1.0 - self.Rs * Ts / self.Ls
        d12 = +Ts * omega_e
        d21 = -Ts * omega_e
        d22 = 1.0 - self.Rs * Ts / self.Ls
        
        id_k1 = d11*id_k + d12*iq_k + (Ts/self.Ls)*self._prev_vd
        iq_k1 = d21*id_k + d22*iq_k + (Ts/self.Ls)*self._prev_vq - (Ts*omega_e*self.psi_f/self.Ls)
        
        # --- Eq.(6)-(7): Predict for all 8 states and minimize cost ---
        g_min = float('inf')
        best_vd = 0.0; best_vq = 0.0
        
        for sw in self._switch_states:
            ud_sw, uq_sw = self._switch_to_dq(sw, self.Udc, theta_e)
            
            # Eq.(6): predict i(k+2) using motor parameters R, L, psi_f
            id_pred = d11*id_k1 + d12*iq_k1 + (Ts/self.Ls)*ud_sw
            iq_pred = d21*id_k1 + d22*iq_k1 + (Ts/self.Ls)*uq_sw - (Ts*omega_e*self.psi_f/self.Ls)
            
            # Eq.(7): cost function
            g = (id_ref - id_pred)**2 + (iq_ref - iq_pred)**2
            
            if g < g_min:
                g_min = g
                best_vd = ud_sw
                best_vq = uq_sw
        
        self._prev_vd = best_vd
        self._prev_vq = best_vq
        
        vmag = math.hypot(best_vd, best_vq)
        if vmag > self.Vmax and vmag > 0.0:
            scale = self.Vmax/vmag
            best_vd *= scale; best_vq *= scale
        
        return best_vd, best_vq


# ============================================================
# MPF-MPCC (Motor-Parameter-Free Model Predictive Current Control)
# Reference: Zhang et al., IEEE TIE, Vol.71, No.6, June 2024
# ============================================================

@dataclass
class MPFMPCCController:
    """
    Motor-Parameter-Free Model Predictive Current Controller.
    
    Replaces PI current loops with an MPC algorithm that uses only
    current differences and voltage differences to predict future
    currents - no motor parameters (R, L, psi_f) are needed.
    
    The speed loop still uses a PI controller to generate iq_ref.
    """
    # --- Fixed parameters ---
    Ts: float                        # Control period [s] (e.g., 1/15000)
    Udc: float = 24.0                # DC-bus voltage [V]
    P: int = 4                       # Pole pairs (for omega_e calculation)
    ctrl_freq: float = 15000.0       # MPC control frequency [Hz] (matches paper)
    
    # --- Speed PI gains ---
    Kp_w: float = 0.01
    Ki_w: float = 0.1
    Iq_max: float = 20.0             # Current limit [A]
    
    # --- Voltage limit ---
    Vmax: float = 12.0               # Max dq voltage magnitude [V]
    
    def __post_init__(self) -> None:
        # MPC effective control period (paper uses 15 kHz)
        self._ctrl_Ts = 1.0 / self.ctrl_freq  # ~66.7 μs
        
        # Speed loop PI (only speed - current loops replaced by MPC)
        self.speed_pi = PIController(self.Kp_w, self.Ki_w, -self.Iq_max, self.Iq_max)
        
        # --- History storage (2-step memory for current & voltage) ---
        self.id_prev = 0.0           # id(k-1)
        self.iq_prev = 0.0           # iq(k-1)
        self.ud_prev = 0.0           # ud(k-1)
        self.uq_prev = 0.0           # uq(k-1)
        self.ud_prev2 = 0.0          # ud(k-2), for M calculation (Eq.22)
        
        # --- Balance factor state (Eq.11, 22, 31) ---
        self.Ed_integral = 0.0       # ∫ Ed·dt
        self.Ed_prev = 0.0           # Ed(k-1)
        self.M = 0.1                 # Integral gain factor (Eq.22)
        # α₀: paper uses 10·Tc for THEIR motor; we scale to OUR motor's L/R ratio
        self.alpha = 0.05  # ≈Ts/L (80us/0.0015H) for 12.5kHz control
        
        # --- 8 switching states of 2L-VSI ---
        # (Eq.3 in paper)  S ∈ {000, 100, 110, 010, 011, 001, 101, 111}
        self._switch_states = [
            (0, 0, 0),   # V0  - zero vector
            (1, 0, 0),   # V1  
            (1, 1, 0),   # V2  
            (0, 1, 0),   # V3  
            (0, 1, 1),   # V4  
            (0, 0, 1),   # V5  
            (1, 0, 1),   # V6  
            (1, 1, 1),   # V7  - zero vector
        ]
        
        # --- Previous optimal switch state (for delay compensation) ---
        self._prev_switch = (0, 0, 0)  # Start with zero vector
        
    def reset(self) -> None:
        """Reset all internal state."""
        self.speed_pi.reset()
        self.id_prev = 0.0
        self.iq_prev = 0.0
        self.ud_prev = 0.0
        self.uq_prev = 0.0
        self.ud_prev2 = 0.0
        self.Ed_integral = 0.0
        self.Ed_prev = 0.0
        self.M = 0.1
        self.alpha = 0.1
        self._prev_switch = (0, 0, 0)
    
    # ----------------------------------------------------------------
    #  Static helpers  (switch state ↔ dq voltage)
    # ----------------------------------------------------------------
    @staticmethod
    def _switch_to_abc(sw: tuple[int, int, int], Udc: float):
        """Convert switch state to three-phase voltages."""
        sa, sb, sc = sw
        va = (2.0 * sa - sb - sc) / 3.0 * Udc
        vb = (2.0 * sb - sa - sc) / 3.0 * Udc
        vc = (2.0 * sc - sa - sb) / 3.0 * Udc
        return va, vb, vc
    
    @staticmethod
    def _switch_to_dq(sw: tuple[int, int, int], Udc: float, theta_e: float):
        """Convert switch state directly to dq-frame voltage (Eq.4)."""
        va, vb, vc = MPFMPCCController._switch_to_abc(sw, Udc)
        # Clarke (amplitude-invariant)
        v_alpha = (2.0 / 3.0) * (va - 0.5 * vb - 0.5 * vc)
        v_beta  = (2.0 / 3.0) * (math.sqrt(3.0) / 2.0 * vb - math.sqrt(3.0) / 2.0 * vc)
        # Park
        cos_t = math.cos(theta_e)
        sin_t = math.sin(theta_e)
        vd =  v_alpha * cos_t + v_beta * sin_t
        vq = -v_alpha * sin_t + v_beta * cos_t
        return vd, vq
    
    # ----------------------------------------------------------------
    #  MPF-MPCC core update (called every control period)
    # ----------------------------------------------------------------
    def update(
        self,
        omega_ref: float,       # Reference mechanical speed [rad/s]
        omega_m: float,         # Actual mechanical speed [rad/s]
        id_k: float,            # id(k) - d-axis current
        iq_k: float,            # iq(k) - q-axis current
        ud_k: float,            # ud(k) - d-axis voltage (applied this period)
        uq_k: float,            # uq(k) - q-axis voltage (applied this period)
        theta_e: float,         # Electrical angle [rad]
        dt: float,              # Time step (= sim dt, may differ from ctrl_Ts)
    ) -> tuple[float, float]:
        """
        Main MPF-MPCC update. Returns optimal Vd, Vq for the next period.
        """
        omega_e = self.P * omega_m   # Electrical angular velocity [rad/s]
        Tc = self._ctrl_Ts           # Effective control period
        
        # ============================================================
        # Step 1: Speed loop PI → iq_ref  (uses sim dt)
        # ============================================================
        iq_ref = self.speed_pi.update(omega_ref - omega_m, dt)
        id_ref = 0.0   # MTPA: id=0 control for SPMSM
        
        # ============================================================
        # Step 2: Compute current & voltage differences
        #   ΔIdq(k) = Idq(k) - Idq(k-1)
        #   ΔUdq(k) = Udq(k) - Udq(k-1)
        # ============================================================
        delta_id = id_k - self.id_prev
        delta_iq = iq_k - self.iq_prev
        delta_ud = ud_k - self.ud_prev
        delta_uq = uq_k - self.uq_prev
        
        # ============================================================
        # Step 3: One-step delay compensation (Eq.24-25)
        #   Predict Idq(k+1) using CURRENT voltage & current differences
        #   ΔIdq(k+1)|S(k+1) = ΔIdq(k) + α·ΔUdq(k) + β·|ΔIdq(k)|·ΔIdq(k)⁻¹
        #   where β·|ΔI|·ΔI⁻¹ = [Tc·ωe·Δiq,  -Tc·ωe·Δid]ᵀ
        # ============================================================
        delta_id_k1 = delta_id + self.alpha * delta_ud + Tc * omega_e * delta_iq
        delta_iq_k1 = delta_iq + self.alpha * delta_uq - Tc * omega_e * delta_id
        
        # Compensated current at (k+1) (Eq.24)
        id_k1 = id_k + delta_id_k1
        iq_k1 = iq_k + delta_iq_k1
        
        # ============================================================
        # Step 4: Enumerate 8 switching states (Eq.26-27, 7)
        #   For each state S, predict Idq(k+2) and evaluate cost g
        #   ΔIdq(k+2)|S(k+2) = ΔIdq(k+1) + α·ΔUdq_S + β·|ΔIdq(k+1)|·ΔIdq(k+1)⁻¹
        #   where ΔUdq_S = U_S(k+1) − U(k)
        # ============================================================
        g_min = float('inf')
        best_vd = 0.0
        best_vq = 0.0
        
        for sw in self._switch_states:
            # 4a. dq voltage of this switch state
            ud_sw, uq_sw = self._switch_to_dq(sw, self.Udc, theta_e)
            
            # 4b. Voltage diff: candidate state minus current applied voltage
            delta_ud_sw = ud_sw - ud_k
            delta_uq_sw = uq_sw - uq_k
            
            # 4c. Predicted current change ΔIdq(k+2) (Eq.27)
            #     Using compensated differences ΔIdq(k+1) as the base
            delta_id_k2 = delta_id_k1 + self.alpha * delta_ud_sw + Tc * omega_e * delta_iq_k1
            delta_iq_k2 = delta_iq_k1 + self.alpha * delta_uq_sw - Tc * omega_e * delta_id_k1
            
            # 4d. Predicted (k+2) current (Eq.26)
            id_pred = id_k1 + delta_id_k2
            iq_pred = iq_k1 + delta_iq_k2
            
            # 4e. Cost function (Eq.7)
            g = (id_ref - id_pred) ** 2 + (iq_ref - iq_pred) ** 2
            
            if g < g_min:
                g_min = g
                best_vd = ud_sw
                best_vq = uq_sw
        
        # ============================================================
        # Step 5: Update balance factor α  (Eq.22, 31)
        # ============================================================
        Ed = id_ref - id_k
        self.Ed_integral += Ed * Tc  # integrate over CONTROL period
        
        # Optimal gain factor M (Eq.22): M = ³√(4·Δud_prev / (Ed+Ed_prev)²)
        denom = (Ed + self.Ed_prev) ** 2
        if denom > 1e-10:
            numer = abs(self.ud_prev2)
            if numer > 1e-10:
                self.M = (4.0 * numer / denom) ** (1.0 / 3.0)
        # Clamp M to prevent divergence
        self.M = max(0.001, min(self.M, 100.0))
        
        # Update α (Eq.31): α = Tc / (0.1 + M·∫Ed·dt)
        self.alpha = Tc / (0.1 + self.M * abs(self.Ed_integral) + 1e-10)
        # Keep α within reasonable bounds
        self.alpha = max(0.02, min(self.alpha, 10.0))
        
        # ============================================================
        # Step 6: Save history for next period
        # ============================================================
        self.ud_prev2 = delta_ud       # save Δud(k) for next M calc
        self.id_prev = id_k
        self.iq_prev = iq_k
        self.ud_prev = ud_k
        self.uq_prev = uq_k
        self.Ed_prev = Ed
        
        # Voltage circle limit
        vmag = math.hypot(best_vd, best_vq)
        if vmag > self.Vmax and vmag > 0.0:
            scale = self.Vmax / vmag
            best_vd *= scale
            best_vq *= scale
        
        return best_vd, best_vq
