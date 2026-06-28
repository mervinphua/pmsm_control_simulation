"""
=============================================================================
 MPF-MPCC 仿真套件 — 现代控制理论课程论文
 基于 Zhang et al., IEEE TIE, Vol.71, No.6, June 2024

 控制器对比:
   PI-FOC (SVPWM) — 连续调制, 最平滑基线
   Conv FCS-MPCC — FCS 8开关, 用 R/L/Psif 做预测
   MPF-MPCC — FCS 8开关, 不用任何参数

 关键: Conv MPCC vs MPF-MPCC 是公平对比(同为FCS).
       PI只是SVPWM基线参考.
=============================================================================
"""
from __future__ import annotations
import math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bldc.controller import FOCController, ConventionalMPCCController, MPFMPCCController
from bldc.inverter import ThreePhaseInverter
from bldc.motor import BLDCMotor
from bldc.transforms import clarke, inv_clarke, inv_park, park

RPM2RAD = 2.0 * math.pi / 60.0

# ---- 电机参数组 ----
M_NOM   = dict(Rs=0.5, Ld=0.0015, Lq=0.0015, Ke=0.01, J=1e-4, B=1e-4, P=4)
M_2XALL = dict(Rs=1.0, Ld=0.003,  Lq=0.003,  Ke=0.02, J=1e-4, B=1e-4, P=4)  # 2R,2L,2ψf


def run_sim(dt, te, wref, tl, ctype, mkw=None):
    """Run closed-loop FOC. ctype='PI'|'CMPC'|'MPC'. mkw=actual motor params."""
    n = len(wref)
    motor = BLDCMotor(**(mkw or M_NOM))
    inv = ThreePhaseInverter()
    
    if ctype == "PI":
        ctrl = FOCController()
    elif ctype == "CMPC":
        # Conv MPCC ALWAYS uses NOMINAL params for prediction
        # (simulates controller unaware of motor parameter changes)
        ctrl = ConventionalMPCCController(
            Ts=dt, Udc=inv.Vdc,
            Rs=M_NOM["Rs"], Ls=M_NOM["Ld"], psi_f=M_NOM["Ke"])
    else:
        ctrl = MPFMPCCController(Ts=dt, Udc=inv.Vdc)
    
    wl, idl, iql, ial, ibl, icl = np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n)
    vp, vq = 0.0, 0.0
    
    for k in range(n):
        wr = wref[k] * RPM2RAD
        if ctype in ("PI","CMPC"):
            vd, vq2 = ctrl.update(wr, motor.omega_m, motor.id, motor.iq,
                                   motor.theta_e, dt)
        else:
            vd, vq2 = ctrl.update(wr, motor.omega_m, motor.id, motor.iq,
                                   vp, vq, motor.theta_e, dt)
        va, vb = inv_park(vd, vq2, motor.theta_e)
        da, db, dc = inv.svpwm(va, vb)
        vap, vbp, vcp = inv.apply(da, db, dc)
        vac, vbc = clarke(vap, vbp, vcp)
        vda, vqa = park(vac, vbc, motor.theta_e)
        motor.TL = float(tl[k])
        s = motor.step(vda, vqa, dt)
        wl[k] = s["omega_m"]/RPM2RAD
        idl[k] = s["id"]; iql[k] = s["iq"]
        # phase currents for THD
        ia_a, ib_a = inv_park(s["id"], s["iq"], s["theta_e"])
        ia_p, ib_p, ic_p = inv_clarke(ia_a, ib_a)
        ial[k]=ia_p; ibl[k]=ib_p; icl[k]=ic_p
        vp, vq = vda, vqa
    
    return {"t": np.arange(n)*dt, "w": wl, "wr": wref,
            "id": idl, "iq": iql, "ia": ial, "ib": ibl, "ic": icl}


def plot3x3(data, title, fname):
    fig, ax = plt.subplots(3, 3, figsize=(16, 10))
    yk = [("w","wr"), ("id",None), ("iq",None)]
    yl = ["Speed [rpm]", "id [A]", "iq [A]"]
    for c, (lab, ky) in enumerate([
        ("PI-FOC (SVPWM)","PI"), ("Conv FCS-MPCC","CMPC"), ("MPF-MPCC","MPF")]):
        d = data[ky]
        for r in range(3):
            a = ax[r,c]; a.plot(d["t"], d[yk[r][0]], lw=1.0)
            if yk[r][1]: a.plot(d["t"], d[yk[r][1]], "k--", lw=0.7)
            a.set_title(lab, fontsize=9); a.set_ylabel(yl[r])
            a.grid(True); a.set_xlabel("Time [s]")
    fig.suptitle(title, fontweight="bold", fontsize=12)
    fig.tight_layout(); fig.savefig(fname, dpi=120)
    print(f"  [OK] {fname}"); plt.close(fig)


# ====================  Experiment 1: Nominal ====================
def exp1():
    print("\n--- Exp 1: Nominal ---")
    dt, te = 8e-5, 5.0; n = int(te/dt); t = np.arange(n)*dt
    wr = np.where(t<0.5, 0.0, np.where(t<2.0, 1000.0, 2000.0))
    tl = np.zeros(n)
    r = {}
    for lb, ct in [("PI","PI"),("CMPC","CMPC"),("MPF","MPC")]:
        print(f"  {lb}..."); r[lb] = run_sim(dt, te, wr, tl, ct, M_NOM)
    plot3x3(r, "Exp 1: Nominal — PI-FOC smoothest baseline (SVPWM)", "fig_exp1.png")
    return r


# ====================  Experiment 2: Parameter Mismatch ====================
def exp2():
    """2 param sets: Nominal | 2R,2L,2psif. Conv MPCC uses M_NOM always."""
    print("\n--- Exp 2: Parameter Mismatch ---")
    dt, te = 8e-5, 5.0; n = int(te/dt); t = np.arange(n)*dt
    wr = np.where(t<0.5, 0.0, np.where(t<2.0, 1000.0, 2000.0))
    tl = np.zeros(n)
    
    fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    yk = [("w","wr"), ("id",None), ("iq",None)]
    yl = ["Speed [rpm]", "id [A]", "iq [A]"]
    
    for col, (mkw, mlabel) in enumerate([
        (M_NOM,   "Nominal"),
        (M_2XALL, "2R,2L,2psif"),
    ]):
        for row, (ct, clabel) in enumerate([("PI","PI"), ("CMPC","CMPC"), ("MPF","MPC")]):
            print(f"  {clabel} @ {mlabel}...")
            r = run_sim(dt, te, wr, tl, ct, mkw)
            a = ax[row, col]
            a.plot(r["t"], r[yk[row][0]], lw=1.0)
            if yk[row][1]: a.plot(r["t"], r[yk[row][1]], "k--", lw=0.7)
            a.set_title(f"{clabel} ({mlabel})", fontsize=8)
            a.set_ylabel(yl[row]); a.grid(True); a.set_xlabel("Time [s]")
    
    fig.suptitle("Exp 2: Parameter Mismatch — Conv MPCC prediction uses M_NOM",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(); fig.savefig("fig_exp2.png", dpi=120)
    print("  [OK] fig_exp2.png"); plt.close(fig)


# ====================  Experiment 3: Load ====================
def exp3():
    print("\n--- Exp 3: Load Disturbance ---")
    dt, te = 8e-5, 5.0; n = int(te/dt); t = np.arange(n)*dt
    wr = np.full(n, 1500.0)
    tl = np.where((t>2.0)&(t<3.5), 0.05, 0.0)
    r = {}
    for lb, ct in [("PI","PI"),("CMPC","CMPC"),("MPF","MPC")]:
        print(f"  {lb}..."); r[lb] = run_sim(dt, te, wr, tl, ct, M_NOM)
    
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for c, (lab, ky) in enumerate([("PI-FOC","PI"),("Conv FCS-MPCC","CMPC"),("MPF-MPCC","MPF")]):
        d = r[ky]
        ax[0,c].plot(d["t"], d["wr"], "k--", lw=0.7); ax[0,c].plot(d["t"], d["w"], lw=1.2)
        ax[0,c].set_title(lab, fontsize=9); ax[0,c].set_ylabel("Speed [rpm]"); ax[0,c].grid(True)
        ax[1,c].plot(d["t"], d["iq"], lw=1.0)
        ax[1,c].set_title(f"{lab} — iq", fontsize=9)
        ax[1,c].set_ylabel("iq [A]"); ax[1,c].set_xlabel("Time [s]"); ax[1,c].grid(True)
    fig.suptitle("Exp 3: Load Disturbance (0.05 Nm pulse 2.0~3.5s)", fontweight="bold", fontsize=12)
    fig.tight_layout(); fig.savefig("fig_exp3.png", dpi=120)
    print("  [OK] fig_exp3.png"); plt.close(fig)
    return r


if __name__ == "__main__":
    print("="*60)
    print("  MPF-MPCC Simulation Suite  (dt=20us, 0-5s)")
    print("="*60)
    exp1(); exp2(); exp3()
    print("\nDone!")
