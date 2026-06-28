"""原GitHub仿真参数: 3控制器对比 (1000->2000rpm, 0.05Nm负载阶跃, 0.5s)"""
import sys; sys.path.insert(0,'.')
import numpy as np, math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bldc.controller import FOCController, ConventionalMPCCController, MPFMPCCController
from bldc.inverter import ThreePhaseInverter
from bldc.motor import BLDCMotor
from bldc.transforms import clarke, inv_clarke, inv_park, park

RPM2RAD = 2*math.pi/60
M_NOM = dict(Rs=0.5, Ld=0.0015, Lq=0.0015, Ke=0.01, J=1e-4, B=1e-4, P=4)

def run(dt, te, wref, tl, ct):
    n=len(wref); m=BLDCMotor(**M_NOM); inv=ThreePhaseInverter()
    if ct=="PI": c=FOCController()
    elif ct=="CMPC": c=ConventionalMPCCController(Ts=dt,Udc=inv.Vdc,Rs=0.5,Ls=0.0015,psi_f=0.01)
    else: c=MPFMPCCController(Ts=dt,Udc=inv.Vdc)
    wl,idl,iql,ial,ibl,icl=np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n)
    vdl,vql,tel=np.zeros(n),np.zeros(n),np.zeros(n)
    vp,vq=0.0,0.0
    for k in range(n):
        wr=wref[k]*RPM2RAD
        if ct in ("PI","CMPC"): vd,vc2=c.update(wr,m.omega_m,m.id,m.iq,m.theta_e,dt)
        else: vd,vc2=c.update(wr,m.omega_m,m.id,m.iq,vp,vq,m.theta_e,dt)
        va,vb=inv_park(vd,vc2,m.theta_e); da,db,dc=inv.svpwm(va,vb)
        vap,vbp,vcp=inv.apply(da,db,dc); vac,vbc=clarke(vap,vbp,vcp)
        vda,vqa=park(vac,vbc,m.theta_e); m.TL=float(tl[k]); s=m.step(vda,vqa,dt)
        wl[k]=s["omega_m"]/RPM2RAD; idl[k]=s["id"]; iql[k]=s["iq"]
        ia_a,ib_a=inv_park(s["id"],s["iq"],s["theta_e"])
        ia_p,ib_p,ic_p=inv_clarke(ia_a,ib_a)
        ial[k]=ia_p; ibl[k]=ib_p; icl[k]=ic_p
        vdl[k]=vd; vql[k]=vc2; tel[k]=s["Te"]
        vp,vq=vda,vqa
    return dict(t=np.arange(n)*dt,w=wl,wr=wref,id=idl,iq=iql,ia=ial,ib=ibl,ic=icl,vd=vdl,vq=vql,Te=tel)

print("Running 3 controllers with original GitHub profile...")
dt,te=8e-5,0.5; n=int(te/dt); t=np.arange(n)*dt
wr=np.where(t<0.2,1000.0,2000.0); tl=np.where(t<0.35,0.02,0.05)  # 0.02Nm constant base load
r={}
for lb,ct in [("PI","PI"),("CMPC","CMPC"),("MPF","MPC")]:
    print(f"  {lb}..."); r[lb]=run(dt,te,wr,tl,ct)

# 5-panel per controller (like original)
fig,ax=plt.subplots(5,3,figsize=(16,18))
ctls=[("PI","PI-FOC"),("CMPC","Conv FCS-MPCC"),("MPF","MPF-MPCC")]
for ci,(ky,lb) in enumerate(ctls):
    d=r[ky]
    # Speed
    ax[0,ci].plot(d["t"],d["wr"],"k--",lw=0.8,label="Ref")
    ax[0,ci].plot(d["t"],d["w"],lw=1.2); ax[0,ci].set_title(f"{lb}",fontsize=10)
    ax[0,ci].set_ylabel("Speed [rpm]"); ax[0,ci].grid(True); ax[0,ci].legend(fontsize=7)
    # dq
    ax[1,ci].plot(d["t"],d["id"],lw=1.0,label="id")
    ax[1,ci].plot(d["t"],d["iq"],lw=1.0,label="iq")
    ax[1,ci].set_ylabel("Current [A]"); ax[1,ci].grid(True); ax[1,ci].legend(fontsize=7)
    # 3-phase zoom
    mz=(d["t"]>=0.3)&(d["t"]<=0.32)
    ax[2,ci].plot(d["t"][mz],d["ia"][mz],lw=1.0,label="ia")
    ax[2,ci].plot(d["t"][mz],d["ib"][mz],lw=1.0,label="ib")
    ax[2,ci].plot(d["t"][mz],d["ic"][mz],lw=1.0,label="ic")
    ax[2,ci].set_ylabel("Current [A]"); ax[2,ci].set_title("3-phase zoom 0.30-0.32s",fontsize=8)
    ax[2,ci].grid(True); ax[2,ci].legend(fontsize=7,ncol=3)
    # Torque
    ax[3,ci].plot(d["t"],d["Te"],lw=1.0); ax[3,ci].set_ylabel("Torque [Nm]")
    ax[3,ci].grid(True)
    # Vdq
    ax[4,ci].plot(d["t"],d["vd"],lw=1.0,label="vd")
    ax[4,ci].plot(d["t"],d["vq"],lw=1.0,label="vq")
    ax[4,ci].set_ylabel("Voltage [V]"); ax[4,ci].set_xlabel("Time [s]")
    ax[4,ci].grid(True); ax[4,ci].legend(fontsize=7)

fig.suptitle("GitHub Profile: PI-FOC vs Conv FCS-MPCC vs MPF-MPCC (48V bus)",fontweight="bold",fontsize=13)
fig.tight_layout(); fig.savefig("fig_github.png",dpi=120)
print("[OK] fig_github.png")
plt.close(fig)
