"""Parameter mismatch experiments: demonstrate MPF-MPCC advantage over PI-FOC.

Key idea: PI-FOC needs motor parameters for tuning (R, L affect optimal Kp/Ki).
When motor parameters CHANGE (aging, temperature, manufacturing variance),
PI performance degrades because gains are no longer optimal.
MPF-MPCC uses NO parameters and is immune to mismatch.

Reference: Zhang et al., IEEE TIE, Vol.71, No.6, June 2024
"""
from __future__ import annotations

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from bldc.controller import FOCController, MPFMPCCController, ConventionalMPCCController
from bldc.inverter import ThreePhaseInverter
from bldc.motor import BLDCMotor
from bldc.transforms import clarke, inv_clarke, inv_park, park

RPM_TO_RADS = 2.0 * math.pi / 60.0

# Motors with different parameter sets
MOTOR_NOMINAL = dict(Rs=0.5, Ld=0.0015, Lq=0.0015, Ke=0.01, J=0.0001, B=0.0001, P=4)
MOTOR_2X_R_L = dict(Rs=1.0, Ld=0.003, Lq=0.003, Ke=0.01, J=0.0001, B=0.0001, P=4)  # 2R, 2L
MOTOR_2X_ALL = dict(Rs=1.0, Ld=0.003, Lq=0.003, Ke=0.02, J=0.0001, B=0.0001, P=4)  # 2R, 2L, 2Ke


def run_foc(
    dt: float, t_end: float, omega_ref_rpm: np.ndarray, TL_arr: np.ndarray,
    ctrl_type: str,  # "PI" or "MPC"
    motor_kwargs: dict,
) -> dict[str, np.ndarray]:
    """Run FOC simulation with given motor parameters."""
    n = len(omega_ref_rpm)
    motor = BLDCMotor(**motor_kwargs) if motor_kwargs else BLDCMotor()
    inverter = ThreePhaseInverter()

    if ctrl_type == "PI":
        ctrl = FOCController()
    elif ctrl_type == "CMPC":
        # Conventional FCS-MPCC (uses motor parameters from kwargs OR defaults)
        Rs = motor_kwargs.get("Rs", 0.5) if motor_kwargs else 0.5
        Ls = motor_kwargs.get("Ld", 0.0015) if motor_kwargs else 0.0015
        psi_f = motor_kwargs.get("Ke", 0.01) if motor_kwargs else 0.01
        ctrl = ConventionalMPCCController(Ts=dt, Udc=inverter.Vdc, Rs=Rs, Ls=Ls, psi_f=psi_f)
    else:
        ctrl = MPFMPCCController(Ts=dt, Udc=inverter.Vdc)

    t = np.arange(n) * dt
    omega_log = np.zeros(n); id_log = np.zeros(n); iq_log = np.zeros(n)
    vd_log = np.zeros(n); vq_log = np.zeros(n); Te_log = np.zeros(n)
    vd_act_prev = 0.0; vq_act_prev = 0.0

    for k in range(n):
        omega_ref = omega_ref_rpm[k] * RPM_TO_RADS
        id_, iq = motor.id, motor.iq

        if ctrl_type == "PI" or ctrl_type == "CMPC":
            vd_c, vq_c = ctrl.update(omega_ref, motor.omega_m, id_, iq, motor.theta_e, dt)
        else:
            vd_c, vq_c = ctrl.update(omega_ref, motor.omega_m, id_, iq,
                                      vd_act_prev, vq_act_prev, motor.theta_e, dt)

        va, vb = inv_park(vd_c, vq_c, motor.theta_e)
        da, db, dc = inverter.svpwm(va, vb)
        va_p, vb_p, vc_p = inverter.apply(da, db, dc)
        va_act, vb_act = clarke(va_p, vb_p, vc_p)
        vd_act, vq_act = park(va_act, vb_act, motor.theta_e)

        motor.TL = float(TL_arr[k])
        s = motor.step(vd_act, vq_act, dt)

        omega_log[k] = s["omega_m"] / RPM_TO_RADS
        id_log[k] = s["id"]; iq_log[k] = s["iq"]
        vd_log[k] = vd_c; vq_log[k] = vq_c; Te_log[k] = s["Te"]
        vd_act_prev = vd_act; vq_act_prev = vq_act

    return {"t": t, "omega_m": omega_log, "omega_ref": omega_ref_rpm,
            "id": id_log, "iq": iq_log, "vd": vd_log, "vq": vq_log, "Te": Te_log}


def plot2x2(results: dict, title: str, filename: str):
    """4-panel comparison: PI@nominal | MPC@nominal | PI@mismatch | MPC@mismatch."""
    keys = [
        ("PI_nominal", "PI (Nominal)"),
        ("MPC_nominal", "MPF-MPCC (Nominal)"),
        ("PI_mismatch", "PI (Param Mismatch)"),
        ("MPC_mismatch", "MPF-MPCC (Param Mismatch)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax_idx, (key, label) in enumerate(keys):
        ax = axes[ax_idx // 2, ax_idx % 2]
        d = results[key]
        ax.plot(d["t"], d["omega_ref"], "k--", lw=0.8, label="Ref")
        ax.plot(d["t"], d["omega_m"], lw=1.2, label=label)
        ax.set_ylabel("Speed [rpm]"); ax.set_title(label)
        ax.legend(loc="best", fontsize=8); ax.grid(True)
        ax.set_xlabel("Time [s]")

    fig.suptitle(title, fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    print(f"  Saved: {filename}")
    plt.close(fig)


def plot_zoom(results: dict, title: str, filename: str, t_start: float, t_end: float):
    """Zoom-in comparison of dq currents during specific time window."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # PI nominal
    d = results["PI_nominal"]
    mask = (d["t"] >= t_start) & (d["t"] <= t_end)
    ax = axes[0, 0]
    ax.plot(d["t"][mask], d["id"][mask], "C0", lw=1.0, label="id")
    ax.plot(d["t"][mask], d["iq"][mask], "C1", lw=1.0, label="iq")
    ax.set_title("PI (Nominal Params)"); ax.legend(loc="best", fontsize=8)
    ax.grid(True); ax.set_ylabel("Current [A]")

    # MPC nominal
    d = results["MPC_nominal"]
    mask = (d["t"] >= t_start) & (d["t"] <= t_end)
    ax = axes[0, 1]
    ax.plot(d["t"][mask], d["id"][mask], "C0", lw=1.0, label="id")
    ax.plot(d["t"][mask], d["iq"][mask], "C1", lw=1.0, label="iq")
    ax.set_title("MPF-MPCC (Nominal Params)"); ax.legend(loc="best", fontsize=8)
    ax.grid(True); ax.set_ylabel("Current [A]")

    # PI mismatch
    d = results["PI_mismatch"]
    mask = (d["t"] >= t_start) & (d["t"] <= t_end)
    ax = axes[1, 0]
    ax.plot(d["t"][mask], d["id"][mask], "C0", lw=1.0, label="id")
    ax.plot(d["t"][mask], d["iq"][mask], "C1", lw=1.0, label="iq")
    ax.set_title("PI (Param Mismatch 2R,2L)"); ax.legend(loc="best", fontsize=8)
    ax.grid(True); ax.set_ylabel("Current [A]"); ax.set_xlabel("Time [s]")

    # MPC mismatch
    d = results["MPC_mismatch"]
    mask = (d["t"] >= t_start) & (d["t"] <= t_end)
    ax = axes[1, 1]
    ax.plot(d["t"][mask], d["id"][mask], "C0", lw=1.0, label="id")
    ax.plot(d["t"][mask], d["iq"][mask], "C1", lw=1.0, label="iq")
    ax.set_title("MPF-MPCC (Param Mismatch 2R,2L)"); ax.legend(loc="best", fontsize=8)
    ax.grid(True); ax.set_ylabel("Current [A]"); ax.set_xlabel("Time [s]")

    fig.suptitle(title, fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    print(f"  Saved: {filename}")
    plt.close(fig)


# ================================================================
#  Experiment A: Resistance & Inductance doubled
#  Motor has 2R, 2L but PI was tuned for R, L
# ================================================================

def experiment_param_mismatch_RL():
    """Motor: 2R, 2L. PI was tuned for original params."""
    print("\n=== Experiment A: Parameter Mismatch (2R, 2L) ===")
    dt, t_end = 1e-5, 1.0
    n = int(t_end / dt)
    tt = np.arange(n) * dt
    # Longer profile: start at 0, ramp to 1000 at 0.2s, step to 2000 at 0.5s
    omega_ref = np.where(tt < 0.2, 0.0, np.where(tt < 0.5, 1000.0, 2000.0))
    TL = np.zeros(n)

    results = {}
    # Nominal motor → PI & MPC
    print("  PI @ nominal...")
    results["PI_nominal"] = run_foc(dt, t_end, omega_ref, TL, "PI", MOTOR_NOMINAL)
    print("  MPC @ nominal...")
    results["MPC_nominal"] = run_foc(dt, t_end, omega_ref, TL, "MPC", MOTOR_NOMINAL)

    # Mismatched motor (2R, 2L) → PI & MPC
    print("  PI @ mismatch...")
    results["PI_mismatch"] = run_foc(dt, t_end, omega_ref, TL, "PI", MOTOR_2X_R_L)
    print("  MPC @ mismatch...")
    results["MPC_mismatch"] = run_foc(dt, t_end, omega_ref, TL, "MPC", MOTOR_2X_R_L)

    plot2x2(results,
            "Exp A: Parameter Mismatch (2R, 2L) — Speed Response",
            "expA_mismatch_speed.png")
    plot_zoom(results,
              "Exp A: Parameter Mismatch (2R, 2L) — dq Current Zoom (t=0.45~0.55s)",
              "expA_mismatch_current.png", 0.45, 0.55)
    return results


# ================================================================
#  Experiment B: All parameters doubled (2R, 2L, 2Ke)
#  Simulates severe motor aging / manufacturing variance
# ================================================================

def experiment_param_mismatch_all():
    """Motor: 2R, 2L, 2Ke. Full parameter mismatch."""
    print("\n=== Experiment B: Parameter Mismatch (2R, 2L, 2Ke) ===")
    dt, t_end = 1e-5, 1.0
    n = int(t_end / dt)
    tt = np.arange(n) * dt
    omega_ref = np.where(tt < 0.2, 0.0, np.where(tt < 0.5, 1000.0, 2000.0))
    TL = np.zeros(n)

    results = {}
    print("  PI @ nominal...")
    results["PI_nominal"] = run_foc(dt, t_end, omega_ref, TL, "PI", MOTOR_NOMINAL)
    print("  MPC @ nominal...")
    results["MPC_nominal"] = run_foc(dt, t_end, omega_ref, TL, "MPC", MOTOR_NOMINAL)

    print("  PI @ severe mismatch...")
    results["PI_mismatch"] = run_foc(dt, t_end, omega_ref, TL, "PI", MOTOR_2X_ALL)
    print("  MPC @ severe mismatch...")
    results["MPC_mismatch"] = run_foc(dt, t_end, omega_ref, TL, "MPC", MOTOR_2X_ALL)

    plot2x2(results,
            "Exp B: Severe Mismatch (2R, 2L, 2Ke) — Speed Response",
            "expB_mismatch_speed.png")
    plot_zoom(results,
              "Exp B: Severe Mismatch (2R, 2L, 2Ke) — dq Current Zoom (t=0.45~0.55s)",
              "expB_mismatch_current.png", 0.45, 0.55)
    return results


# ================================================================
#  Experiment C: Load disturbance under parameter mismatch
#  Most representative of real-world scenario
# ================================================================

def experiment_load_under_mismatch():
    """
    Constant speed 1500 rpm, load step 0 → 0.05 N·m at t=0.3s.
    Motor has 2R, 2L.
    Compare PI vs MPC speed drop and recovery.
    """
    print("\n=== Experiment C: Load Disturbance + Param Mismatch ===")
    dt, t_end = 1e-5, 1.0
    n = int(t_end / dt)
    tt = np.arange(n) * dt
    omega_ref = np.full(n, 1500.0)
    TL = np.where(tt < 0.5, 0.0, 0.05)

    results = {}
    print("  PI @ mismatch...")
    results["PI_mismatch"] = run_foc(dt, t_end, omega_ref, TL, "PI", MOTOR_2X_R_L)
    print("  MPC @ mismatch...")
    results["MPC_mismatch"] = run_foc(dt, t_end, omega_ref, TL, "MPC", MOTOR_2X_R_L)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    d_pi = results["PI_mismatch"]
    d_mpc = results["MPC_mismatch"]

    # Speed zoom around load step
    ax = axes[0]
    t_mask = (d_pi["t"] >= 0.48) & (d_pi["t"] <= 0.65)
    ax.plot(d_pi["t"][t_mask], d_pi["omega_ref"][t_mask], "k--", lw=0.8, label="Ref (1500 rpm)")
    ax.plot(d_pi["t"][t_mask], d_pi["omega_m"][t_mask], "C0", lw=1.5, label="PI (2R,2L mismatch)")
    ax.plot(d_mpc["t"][t_mask], d_mpc["omega_m"][t_mask], "C1", lw=1.5, label="MPF-MPCC")
    ax.set_ylabel("Speed [rpm]"); ax.set_title("Load Disturbance with Parameter Mismatch (2R,2L)")
    ax.legend(loc="best"); ax.grid(True)

    # iq current zoom
    ax = axes[1]
    ax.plot(d_pi["t"][t_mask], d_pi["iq"][t_mask], "C0", lw=1.2, label="PI iq")
    ax.plot(d_mpc["t"][t_mask], d_mpc["iq"][t_mask], "C1", lw=1.2, label="MPF-MPCC iq")
    ax.set_ylabel("iq [A]"); ax.set_xlabel("Time [s]")
    ax.set_title("q-axis Current Response to Load Step")
    ax.legend(loc="best"); ax.grid(True)

    fig.tight_layout()
    fig.savefig("expC_load_mismatch.png", dpi=120)
    print("  Saved: expC_load_mismatch.png")
    plt.close(fig)

    return results


# ================================================================

def main():
    experiment_param_mismatch_RL()
    experiment_param_mismatch_all()
    experiment_load_under_mismatch()
    experiment_fcs_comparison()
    print("\n=== All mismatch experiments complete! ===")


# ================================================================
#  Experiment D: THE KEY COMPARISON (matches paper's main result)
#  Conventional FCS-MPCC vs MPF-MPCC under parameter mismatch
#  BOTH use FCS (8 switching states) → FAIR comparison
#  CMPC uses R,L,ψf in prediction → degrades with mismatch
#  MPF-MPCC uses NO parameters → immune to mismatch
# ================================================================

def experiment_fcs_comparison():
    """
    Fair comparison: both controllers use FCS (8 discrete states).
    Conventional MPCC needs parameters → degrades when wrong.
    MPF-MPCC doesn't → doesn't degrade.
    """
    print("\n=== Experiment D: FCS-MPCC vs MPF-MPCC (FAIR comparison) ===")
    dt, t_end = 1e-5, 1.0
    n = int(t_end / dt)
    tt = np.arange(n) * dt
    omega_ref = np.where(tt < 0.2, 0.0, np.where(tt < 0.5, 1000.0, 2000.0))
    TL = np.zeros(n)

    results = {}

    print("  CMPC @ nominal (correct params)...")
    results["CMPC_nominal"] = run_foc(dt, t_end, omega_ref, TL, "CMPC", MOTOR_NOMINAL)
    print("  MPF-MPCC @ nominal...")
    results["MPF_nominal"] = run_foc(dt, t_end, omega_ref, TL, "MPC", MOTOR_NOMINAL)
    print("  PI @ nominal...")
    results["PI_nominal"] = run_foc(dt, t_end, omega_ref, TL, "PI", MOTOR_NOMINAL)

    print("  CMPC @ mismatch (prediction uses WRONG params)...")
    results["CMPC_mismatch"] = run_foc(dt, t_end, omega_ref, TL, "CMPC", MOTOR_2X_R_L)
    print("  MPF-MPCC @ mismatch (no params needed)...")
    results["MPF_mismatch"] = run_foc(dt, t_end, omega_ref, TL, "MPC", MOTOR_2X_R_L)

    # --- Plot: 2-row comparison ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    
    labels = [
        (0,0,"PI_nominal","PI-FOC (SVPWM)"),
        (0,1,"CMPC_nominal","Conv. FCS-MPCC (nom.)"),
        (0,2,"MPF_nominal","MPF-MPCC (nom.)"),
        (1,0,"PI_nominal","PI-FOC (baseline)"),
        (1,1,"CMPC_mismatch","Conv. FCS-MPCC (2R,2L)"),
        (1,2,"MPF_mismatch","MPF-MPCC (2R,2L)"),
    ]
    
    for row, col, key, title in labels:
        ax = axes[row, col]
        d = results[key]
        ax.plot(d["t"], d["omega_ref"], "k--", lw=0.7, label="Ref")
        ax.plot(d["t"], d["omega_m"], lw=1.2)
        ax.set_ylabel("Speed [rpm]"); ax.set_title(title, fontsize=9)
        ax.grid(True); ax.set_xlabel("Time [s]")
    
    fig.suptitle("Exp D: Conventional FCS-MPCC vs MPF-MPCC — Parameter Mismatch (2R,2L)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig("expD_fcs_comparison.png", dpi=120)
    print("  Saved: expD_fcs_comparison.png")
    plt.close(fig)

    # --- dq current zoom ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (key, title) in zip(axes, [
        ("PI_nominal", "PI-FOC (SVPWM, baseline)"),
        ("CMPC_mismatch", "Conv. FCS-MPCC (param mismatch)"),
        ("MPF_mismatch", "MPF-MPCC (no params)"),
    ]):
        d = results[key]
        mask = (d["t"] >= 0.45) & (d["t"] <= 0.55)
        ax.plot(d["t"][mask], d["id"][mask], lw=1.0, label="id")
        ax.plot(d["t"][mask], d["iq"][mask], lw=1.0, label="iq")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="best", fontsize=8); ax.grid(True)
        ax.set_ylabel("Current [A]"); ax.set_xlabel("Time [s]")
    
    fig.suptitle("Exp D: dq Currents under Parameter Mismatch", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig("expD_fcs_current.png", dpi=120)
    print("  Saved: expD_fcs_current.png")
    plt.close(fig)

    return results


if __name__ == "__main__":
    main()
