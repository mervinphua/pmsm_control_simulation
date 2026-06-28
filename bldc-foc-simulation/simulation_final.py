"""Final comparison: PI-FOC vs Conv FCS-MPCC vs MPF-MPCC.

KEY INSIGHT:
- PI-FOC (SVPWM):  smoothest waveform under ALL conditions (continuous modulation)
                   but needs R,L,Ke for decoupling → id error under mismatch
- Conv FCS-MPCC:   FCS-based (8 states only → ripple)
                   uses R,L,Psif in prediction → DEGRADES severely under mismatch
- MPF-MPCC:        FCS-based (8 states only → ripple, SAME as Conv MPCC)
                   ZERO parameters → immune to mismatch

The fair comparison is Conv MPCC vs MPF-MPCC (both FCS). 
PI is the SVPWM baseline for reference.
"""
from __future__ import annotations

import math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bldc.controller import FOCController, MPFMPCCController, ConventionalMPCCController
from bldc.inverter import ThreePhaseInverter
from bldc.motor import BLDCMotor
from bldc.transforms import clarke, inv_clarke, inv_park, park

RPM2RAD = 2*math.pi/60
M_NOM = dict(Rs=0.5, Ld=0.0015, Lq=0.0015, Ke=0.01, J=1e-4, B=1e-4, P=4)
M_2X  = dict(Rs=1.0, Ld=0.003,  Lq=0.003,  Ke=0.01, J=1e-4, B=1e-4, P=4)


def sim(dt, t_end, wref, tl, ctype, mkw=None):
    n = len(wref)
    m = BLDCMotor(**(mkw or M_NOM))
    inv = ThreePhaseInverter()
    
    if ctype == "PI":
        c = FOCController()
    elif ctype == "CMPC":
        kw = mkw or M_NOM
        c = ConventionalMPCCController(Ts=dt, Udc=inv.Vdc,
                Rs=kw["Rs"], Ls=kw["Ld"], psi_f=kw["Ke"])
    else:
        c = MPFMPCCController(Ts=dt, Udc=inv.Vdc)
    
    wo = np.zeros(n); ido = np.zeros(n); iqo = np.zeros(n)
    vd_p = 0.0; vq_p = 0.0
    
    for k in range(n):
        wr = wref[k]*RPM2RAD
        
        if ctype in ("PI","CMPC"):
            vd,vq = c.update(wr, m.omega_m, m.id, m.iq, m.theta_e, dt)
        else:
            vd,vq = c.update(wr, m.omega_m, m.id, m.iq, vd_p, vq_p, m.theta_e, dt)
        
        va,vb = inv_park(vd,vq,m.theta_e)
        da,db,dc = inv.svpwm(va,vb)
        vap,vbp,vcp = inv.apply(da,db,dc)
        vac,vbc = clarke(vap,vbp,vcp)
        vd_a,vq_a = park(vac,vbc,m.theta_e)
        
        m.TL = float(tl[k])
        s = m.step(vd_a,vq_a,dt)
        wo[k]=s["omega_m"]/RPM2RAD; ido[k]=s["id"]; iqo[k]=s["iq"]
        vd_p=vd_a; vq_p=vq_a
    
    return {"t":np.arange(n)*dt,"w":wo,"wr":wref,"id":ido,"iq":iqo}


def full_plot(data, title, fname):
    """Full 0~t_end view, all 3 controllers."""
    fig, ax = plt.subplots(3,3,figsize=(16,10))
    ttl = ["Speed","id current","iq current"]
    ylb = ["Speed [rpm]","id [A]","iq [A]"]
    keys = [("w","wr"),("id",None),("iq",None)]
    
    for row in range(3):
        for col, (label, d) in enumerate([
            ("PI-FOC (SVPWM)", data["PI"]),
            ("Conv FCS-MPCC", data["CMPC"]),
            ("MPF-MPCC", data["MPF"])
        ]):
            ax[row,col].plot(d["t"], d[keys[row][0]], lw=1.0)
            if keys[row][1]:
                ax[row,col].plot(d["t"], d[keys[row][1]], "k--", lw=0.7)
            ax[row,col].set_title(f"{label}", fontsize=9)
            ax[row,col].set_ylabel(ylb[row]); ax[row,col].grid(True)
            ax[row,col].set_xlabel("Time [s]")
    
    fig.suptitle(title, fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    print(f"  Saved: {fname}")
    plt.close(fig)


# ================================================================
#  Experiment 1: Nominal parameters — show PI is best baseline
# ================================================================

def exp1_nominal():
    print("\n=== Experiment 1: Nominal Parameters ===")
    dt,te=1e-5,1.0; n=int(te/dt); tt=np.arange(n)*dt
    wr = np.where(tt<0.2,0.0,np.where(tt<0.5,1000.0,2000.0))
    tl = np.zeros(n)
    
    d = {}
    print("  PI-FOC..."); d["PI"] = sim(dt,te,wr,tl,"PI",M_NOM)
    print("  Conv FCS-MPCC..."); d["CMPC"] = sim(dt,te,wr,tl,"CMPC",M_NOM)
    print("  MPF-MPCC..."); d["MPF"] = sim(dt,te,wr,tl,"MPC",M_NOM)
    
    full_plot(d, "Exp 1: Nominal Parameters — PI is smoothest baseline",
              "final_exp1_nominal.png")
    return d


# ================================================================
#  Experiment 2: Parameter mismatch (2R,2L) — MPF-MPCC advantage
# ================================================================

def exp2_mismatch():
    print("\n=== Experiment 2: Parameter Mismatch (2R, 2L) ===")
    dt,te=1e-5,1.0; n=int(te/dt); tt=np.arange(n)*dt
    wr = np.where(tt<0.2,0.0,np.where(tt<0.5,1000.0,2000.0))
    tl = np.zeros(n)
    
    d = {}
    print("  PI-FOC (2R,2L motor)..."); d["PI"] = sim(dt,te,wr,tl,"PI",M_2X)
    print("  Conv FCS-MPCC (2R,2L motor, WRONG pred)..."); d["CMPC"] = sim(dt,te,wr,tl,"CMPC",M_2X)
    print("  MPF-MPCC (2R,2L motor, NO params)..."); d["MPF"] = sim(dt,te,wr,tl,"MPC",M_2X)
    
    full_plot(d, "Exp 2: Parameter Mismatch (2R,2L) — MPF-MPCC unaffected",
              "final_exp2_mismatch.png")
    return d


# ================================================================
#  Experiment 3: Load disturbance under nominal params
# ================================================================

def exp3_load():
    print("\n=== Experiment 3: Load Disturbance ===")
    dt,te=1e-5,1.0; n=int(te/dt); tt=np.arange(n)*dt
    wr = np.full(n, 1500.0)
    tl = np.where(tt<0.5, 0.0, 0.05)
    
    d = {}
    print("  PI-FOC..."); d["PI"] = sim(dt,te,wr,tl,"PI",M_NOM)
    print("  Conv FCS-MPCC..."); d["CMPC"] = sim(dt,te,wr,tl,"CMPC",M_NOM)
    print("  MPF-MPCC..."); d["MPF"] = sim(dt,te,wr,tl,"MPC",M_NOM)
    
    fig, ax = plt.subplots(2,3,figsize=(14,8))
    ykeys = [("w","wr","Speed [rpm]"),("id",None,"id [A]"),("iq",None,"iq [A]")]
    
    for col, (label, dkey) in enumerate([
        ("PI-FOC","PI"),("Conv FCS-MPCC","CMPC"),("MPF-MPCC","MPF")
    ]):
        for row in range(2):
            yk,yrk,yl = ykeys[row]
            ax[row,col].plot(d[dkey]["t"], d[dkey][yk], lw=1.0)
            if yrk: ax[row,col].plot(d[dkey]["t"], d[dkey][yrk],"k--",lw=0.7)
            ax[row,col].set_title(f"{label}",fontsize=9)
            ax[row,col].set_ylabel(yl); ax[row,col].grid(True)
        ax[0,col].set_xlabel(""); ax[1,col].set_xlabel("Time [s]")
    
    fig.suptitle("Exp 3: Load Disturbance (0→0.05 Nm at t=0.5s)",fontweight="bold",fontsize=12)
    fig.tight_layout()
    fig.savefig("final_exp3_load.png",dpi=120)
    print("  Saved: final_exp3_load.png")
    plt.close(fig)
    return d


# ================================================================

def main():
    exp1_nominal()
    exp2_mismatch()
    exp3_load()
    print("\n=== Complete ===")

if __name__ == "__main__":
    main()
