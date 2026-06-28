"""
=============================================================================
 MPF-MPCC 定量分析 — THD, id误差, iq跟踪误差
 仿照原论文 Figs 12, 15 (THD柱状图对比)
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
M_NOM   = dict(Rs=0.5, Ld=0.0015, Lq=0.0015, Ke=0.01, J=1e-4, B=1e-4, P=4)
M_MIS   = dict(Rs=1.0, Ld=0.003,  Lq=0.003,  Ke=0.02, J=1e-4, B=1e-4, P=4)  # 2R,2L,2ψf


def run_sim(dt, te, wref, tl, ctype, mkw=None):
    n = len(wref); motor = BLDCMotor(**(mkw or M_NOM)); inv = ThreePhaseInverter()
    if ctype == "PI": ctrl = FOCController()
    elif ctype == "CMPC":
        ctrl = ConventionalMPCCController(Ts=dt, Udc=inv.Vdc,
                Rs=M_NOM["Rs"], Ls=M_NOM["Ld"], psi_f=M_NOM["Ke"])
    else: ctrl = MPFMPCCController(Ts=dt, Udc=inv.Vdc)
    wl,idl,iql,ial,ibl,icl = [np.zeros(n) for _ in range(6)]
    vp,vq=0.0,0.0
    for k in range(n):
        wr=wref[k]*RPM2RAD
        if ctype in ("PI","CMPC"):
            vd,vc=ctrl.update(wr,motor.omega_m,motor.id,motor.iq,motor.theta_e,dt)
        else: vd,vc=ctrl.update(wr,motor.omega_m,motor.id,motor.iq,vp,vq,motor.theta_e,dt)
        va,vb=inv_park(vd,vc,motor.theta_e); da,db,dc=inv.svpwm(va,vb)
        vap,vbp,vcp=inv.apply(da,db,dc); vac,vbc=clarke(vap,vbp,vcp)
        vda,vqa=park(vac,vbc,motor.theta_e); motor.TL=float(tl[k]); s=motor.step(vda,vqa,dt)
        wl[k]=s["omega_m"]/RPM2RAD; idl[k]=s["id"]; iql[k]=s["iq"]
        ia_a,ib_a=inv_park(s["id"],s["iq"],s["theta_e"])
        ia_p,ib_p,ic_p=inv_clarke(ia_a,ib_a)
        ial[k]=ia_p; ibl[k]=ib_p; icl[k]=ic_p
        vp,vq=vda,vqa
    return {"t":np.arange(n)*dt,"w":wl,"wr":wref,"id":idl,"iq":iql,"ia":ial,"ib":ibl,"ic":icl}


def compute_thd(signal, fs, f0=None):
    """Compute THD of a periodic signal using FFT."""
    N = len(signal)
    if f0 is None: f0 = fs / N  * 10  # guess fundamental ~10th bin
    # Remove DC, apply Hanning window
    x = signal - np.mean(signal)
    w = np.hanning(N)
    X = np.fft.rfft(x * w)
    freqs = np.fft.rfftfreq(N, 1/fs)
    # Find fundamental (max near expected frequency)
    f0_idx = np.argmax(np.abs(X[1:])) + 1  # skip DC
    fundamental = np.abs(X[f0_idx])
    # Sum harmonics (multiples of fundamental)
    harm_power = 0
    for h in range(2, 20):
        idx = f0_idx * h
        if idx < len(X):
            harm_power += np.abs(X[idx])**2
    if fundamental > 1e-10:
        return np.sqrt(harm_power) / fundamental * 100
    return 0.0


def compute_metrics(data, steady_start, steady_end):
    """Compute id RMS error, iq tracking error, THD over steady-state window."""
    mask = (data["t"] >= steady_start) & (data["t"] <= steady_end)
    # id should be 0
    id_rms = np.sqrt(np.mean(data["id"][mask]**2))
    # iq tracking (uses speed PI, estimate iq_ref from steady speed)
    iq_std = np.std(data["iq"][mask])
    # THD of phase a
    dt = data["t"][1] - data["t"][0]
    fs = 1.0 / dt
    ia_ss = data["ia"][mask]
    thd = compute_thd(ia_ss, fs)
    return id_rms, iq_std, thd


# ================================================================
#  Metrics comparison experiment
# ================================================================

def exp_metrics():
    """
    Run all 3 controllers at 1000 rpm with load under different param sets.
    Compute id RMS, iq std, THD and generate bar charts like the paper.
    """
    print("\n" + "="*60)
    print("  Metrics Analysis: id error | iq ripple | THD")
    print("="*60)
    
    dt,te=8e-5,5.0; n=int(te/dt); t=np.arange(n)*dt
    wr=np.where(t<0.5,0.0,np.where(t<2.0,1000.0,2000.0))
    tl=np.zeros(n)
    
    # --- Run all combos ---
    ctls = [("PI","PI-FOC"),("CMPC","Conv FCS-MPCC"),("MPF","MPF-MPCC")]
    mks  = [(M_NOM,"Nominal"),(M_MIS,"2R,2L,2psif")]
    
    id_rms_arr = np.zeros((3,2))
    iq_std_arr = np.zeros((3,2))
    thd_arr    = np.zeros((3,2))
    
    for ri, (ct, cl) in enumerate(ctls):
        for ci, (mk, ml) in enumerate(mks):
            print(f"  {cl} @ {ml}...")
            d = run_sim(dt, te, wr, tl, ct, mk)
            ir, iq_s, th = compute_metrics(d, 2.5, 5.0)
            id_rms_arr[ri,ci] = ir
            iq_std_arr[ri,ci] = iq_s
            thd_arr[ri,ci]   = th
    
    # --- Plot: 3 bar charts ---
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(2)
    wd = 0.25
    colors = ['#4472C4','#ED7D31','#A5A5A5']
    labels = [l for _,l in ctls]
    
    for col, (arr, ylabel, title) in enumerate([
        (id_rms_arr, "id RMS [A]", "id Current Error (lower is better)"),
        (iq_std_arr, "iq Std [A]", "iq Current Ripple (lower is better)"),
        (thd_arr,    "THD [%]",     "Phase Current THD (lower is better)"),
    ]):
        a = ax[col]
        for i in range(3):
            a.bar(x+i*wd, arr[i], wd, color=colors[i], label=labels[i])
        a.set_xticks(x + wd)
        a.set_xticklabels([l for _,l in mks])
        a.set_ylabel(ylabel)
        a.set_title(title, fontsize=10)
        if col == 0: a.legend(fontsize=7)
        a.grid(True, axis='y', alpha=0.3)
    
    fig.suptitle("Metrics Comparison: PI-FOC vs Conv FCS-MPCC vs MPF-MPCC",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig("fig_metrics.png", dpi=150)
    print("  [OK] fig_metrics.png")
    plt.close(fig)
    
    # --- Print table ---
    print("\n  id RMS Error [A] (lower=better):")
    print(f"  {'':20s} {'Nominal':>10s} {'2R,2L,2psif':>12s}")
    for i, (_,l) in enumerate(ctls):
        print(f"  {l:20s} {id_rms_arr[i,0]:10.4f} {id_rms_arr[i,1]:12.4f}")
    
    print("\n  iq Std [A] (lower=better):")
    for i, (_,l) in enumerate(ctls):
        print(f"  {l:20s} {iq_std_arr[i,0]:10.4f} {iq_std_arr[i,1]:12.4f}")
    
    print("\n  THD [%]:")
    for i, (_,l) in enumerate(ctls):
        print(f"  {l:20s} {thd_arr[i,0]:10.1f} {thd_arr[i,1]:12.1f}")


if __name__ == "__main__":
    exp_metrics()
