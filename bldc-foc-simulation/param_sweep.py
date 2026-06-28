"""Quick parameter sweep for MPF-MPCC alpha0."""
from __future__ import annotations
import math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bldc.controller import FOCController, ConventionalMPCCController, MPFMPCCController
from bldc.inverter import ThreePhaseInverter
from bldc.motor import BLDCMotor
from bldc.transforms import clarke, inv_clarke, inv_park, park

RPM2RAD = 2.0*math.pi/60
M_NOM = dict(Rs=0.5, Ld=0.0015, Lq=0.0015, Ke=0.01, J=1e-4, B=1e-4, P=4)
M_2X  = dict(Rs=1.0, Ld=0.003,  Lq=0.003,  Ke=0.02, J=1e-4, B=1e-4, P=4)

def run_one(dt, te, wref, tl, ctype, mkw, alpha0=None):
    n = len(wref); m = BLDCMotor(**(mkw or M_NOM)); inv = ThreePhaseInverter()
    if ctype == "PI": c = FOCController()
    elif ctype == "CMPC":
        c = ConventionalMPCCController(Ts=dt, Udc=inv.Vdc, Rs=M_NOM["Rs"], Ls=M_NOM["Ld"], psi_f=M_NOM["Ke"])
    else:
        c = MPFMPCCController(Ts=dt, Udc=inv.Vdc)
        if alpha0 is not None:
            c.alpha = alpha0  # override initial alpha
    wl,idl,iql = np.zeros(n),np.zeros(n),np.zeros(n)
    vp,vq=0.0,0.0
    for k in range(n):
        wr=wref[k]*RPM2RAD
        if ctype in ("PI","CMPC"): vd,vc=c.update(wr,m.omega_m,m.id,m.iq,m.theta_e,dt)
        else: vd,vc=c.update(wr,m.omega_m,m.id,m.iq,vp,vq,m.theta_e,dt)
        va,vb=inv_park(vd,vc,m.theta_e); da,db,dc=inv.svpwm(va,vb)
        vap,vbp,vcp=inv.apply(da,db,dc); vac,vbc=clarke(vap,vbp,vcp)
        vda,vqa=park(vac,vbc,m.theta_e); m.TL=float(tl[k]); s=m.step(vda,vqa,dt)
        wl[k]=s["omega_m"]/RPM2RAD; idl[k]=s["id"]; iql[k]=s["iq"]
        vp,vq=vda,vqa
    # Metrics over steady state (1.0-1.8s)
    t = np.arange(n)*dt; mask = (t>1.0)&(t<1.8)
    id_rms = np.sqrt(np.mean(idl[mask]**2))
    iq_std = np.std(iql[mask])
    return id_rms, iq_std

def sweep():
    dt,te=1e-4,2.0; n=int(te/dt); t=np.arange(n)*dt
    wr=np.where(t<0.3,0.0,1000.0); tl=np.zeros(n)
    alphas = [0.005,0.01,0.02,0.05,0.08,0.1,0.2,0.5]
    
    print("  Sweeping alpha0 for MPF-MPCC...")
    print(f"  {'alpha0':>8s}  {'idRMS_nom':>10s} {'iqStd_nom':>10s} {'idRMS_2X':>10s} {'iqStd_2X':>10s}")
    
    id_nom, iq_nom, id_2x, iq_2x = [], [], [], []
    for a in alphas:
        ir_n,iq_n = run_one(dt,te,wr,tl,"MPC",M_NOM,a)
        ir_2,iq_2 = run_one(dt,te,wr,tl,"MPC",M_2X,a)
        id_nom.append(ir_n); iq_nom.append(iq_n); id_2x.append(ir_2); iq_2x.append(iq_2)
        print(f"  {a:8.4f}  {ir_n:10.5f} {iq_n:10.5f} {ir_2:10.5f} {iq_2:10.5f}")
    
    # Baseline
    ir_pi_n,iq_pi_n = run_one(dt,te,wr,tl,"PI",M_NOM,None)
    ir_cm_n,iq_cm_n = run_one(dt,te,wr,tl,"CMPC",M_NOM,None)
    ir_cm_2,iq_cm_2 = run_one(dt,te,wr,tl,"CMPC",M_2X,None)
    
    print(f"\n  --- Baselines ---")
    print(f"  PI-FOC  nom: idRMS={ir_pi_n:.5f} iqStd={iq_pi_n:.5f}")
    print(f"  Conv FCS nom: idRMS={ir_cm_n:.5f} iqStd={iq_cm_n:.5f}")
    print(f"  Conv FCS 2X:  idRMS={ir_cm_2:.5f} iqStd={iq_cm_2:.5f}")
    
    # Find best alpha: min(idRMS_nom + idRMS_2X) with idRMS_nom close to Conv
    best_idx = np.argmin(np.array(id_nom) + np.array(id_2x))
    print(f"\n  >>> Best alpha0 = {alphas[best_idx]:.4f}")
    print(f"      MPF idRMS nom={id_nom[best_idx]:.5f} 2X={id_2x[best_idx]:.5f}")
    print(f"      Conv idRMS nom={ir_cm_n:.5f} 2X={ir_cm_2:.5f}")
    
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    ax[0].plot(alphas,id_nom,'bo-',label='MPF Nominal')
    ax[0].axhline(ir_cm_n,color='C1',ls='--',label='Conv FCS nominal')
    ax[0].set_xlabel('alpha0'); ax[0].set_ylabel('id RMS [A]'); ax[0].legend()
    ax[0].set_title('id RMS Error vs alpha0')
    ax[1].plot(alphas,iq_nom,'bo-',label='MPF Nominal')
    ax[1].axhline(iq_cm_n,color='C1',ls='--',label='Conv FCS nominal')
    ax[1].set_xlabel('alpha0'); ax[1].set_ylabel('iq Std [A]'); ax[1].legend()
    ax[1].set_title('iq Ripple vs alpha0')
    fig.tight_layout(); fig.savefig('fig_sweep.png',dpi=120)
    print("  [OK] fig_sweep.png")

if __name__=="__main__":
    sweep()
