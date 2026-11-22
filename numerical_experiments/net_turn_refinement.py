"""Check 32-orbit net rotation, excursions and onset across solver tolerances."""
import sys,csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from numerical_experiments import orbital_poincare as o

def run(job):
 q,tol,step=job
 sol=o.integrate(q,[np.deg2rad(o.PHI0_DEG),0],rtol=tol,step=step)
 t=np.linspace(0,64*np.pi,32*720+1);phi=sol.sol(t)[1];span=np.maximum.accumulate(phi)-np.minimum.accumulate(phi)
 return dict(ratio=q,rtol=tol,max_step_orbits=step/(2*np.pi),net_turns=(phi[-1]-phi[0])/(2*np.pi),span_32_deg=np.rad2deg(span[-1]),span_8_deg=np.rad2deg(span[8*720]),first_full_turn_orbits=np.argmax(span>2*np.pi)/720)
if __name__=='__main__':
 jobs=[(q,tol,step) for step in [2*np.pi/240,np.inf] for q in [.5,.75,.85,.9,1.1] for tol in [1e-10,1e-12,2e-13,3e-14]]
 with ProcessPoolExecutor(max_workers=4) as pool:
  rows=list(pool.map(run,jobs))
 with (o.OUT/'orbital_net_turn_refinement.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 for row in rows:print(row)
