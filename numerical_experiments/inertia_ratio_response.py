"""Forced roll at the instantaneous fold charge as the inertia ratio kappa=|A-B|/C varies.

For each A (B=C=1) the charge is the fold charge of that inertia triple at the
reference phase f=60 deg. The roll is released from rest at the zero-charge
centre phi=pi and integrated along the reference Keplerian orbit. The output is
the unwrapped roll span after 40 and 400 orbits, at two tolerances.
"""
import sys,csv
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from model import roll_model as m
OUT=Path(__file__).resolve().parents[1]/'data'/'results'; OUT.mkdir(parents=True,exist_ok=True)
N_MEAN=np.sqrt(m.MU_E/m.A_ORB**3)
ORBITS=(40,400)

def span(A,q,rtol):
    alpha=q*m.fold_threshold(*m.roll_coefficients(m.F_REF,A=A,B=m.B_INERTIA))
    def rhs(s,y):
        aL,bL,c=m.roll_coefficients(y[0],A=A,B=m.B_INERTIA)
        fdot=m.orbit_state(y[0])[3]
        return [fdot/N_MEAN,y[2],m.h(y[1],alpha,aL,bL,c)/(m.C_INERTIA*N_MEAN**2)]
    s=np.linspace(0,2*np.pi*max(ORBITS),max(ORBITS)*200+1)
    sol=solve_ivp(rhs,[0,s[-1]],[m.F_REF,np.pi,0.],method='DOP853',rtol=rtol,atol=rtol*1e-2,t_eval=s)
    phi=sol.y[1]
    return [float(np.degrees(np.ptp(phi[:n*200+1]))) for n in ORBITS]

def main():
    rows=[]
    for A in [.9994,.99,.9,.4]:
        kappa=abs(A-m.B_INERTIA)/m.C_INERTIA
        for rtol in [1e-10,1e-12]:
            s40,s400=span(A,1.,rtol)
            rows.append(dict(A_m2=A,kappa=kappa,period_ratio=kappa**-.5,charge_ratio=1.,rtol=rtol,
                span_40_orbits_deg=s40,span_400_orbits_deg=s400,
                span_40_over_sqrt_kappa_rad=float(np.radians(s40)/np.sqrt(kappa))))
    with (OUT/'inertia_ratio_response.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
    for row in rows:print(row)

if __name__=='__main__':main()
