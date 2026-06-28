import sys; sys.path.insert(0,'.')
import numpy as np
from simulation_final import run_sim, M_NOM, M_2XALL

dt,te=8e-5,2.0; n=int(te/dt); t=np.arange(n)*dt
wr=np.where(t<0.3,0.0,1000.0); tl=np.zeros(n)

for mkw, ml in [(M_NOM,"Nominal"),(M_2XALL,"2XALL")]:
    for ct, cl in [("PI","PI"),("CMPC","CMPC"),("MPC","MPC")]:
        d=run_sim(dt,te,wr,tl,ct,mkw)
        wmin,wmax=d["w"].min(),d["w"].max()
        imin,imax=d["id"].min(),d["id"].max()
        iqmin,iqmax=d["iq"].min(),d["iq"].max()
        ok="OK" if wmax>500 else "FAIL"
        print(f"  {cl} @ {ml}: w={wmin:.0f}~{wmax:.0f} id={imin:.2f}~{imax:.2f} iq={iqmin:.2f}~{iqmax:.2f} [{ok}]")
print("Done")
