"""Recompute the existing one-orbit table and three-axis tolerance comparison."""
from pathlib import Path
import sys,json,csv
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import orbital_poincare as o
import controlled_three_axis_validation as c
m=o.m
rows=[]
for ratio in [.25,.5,.9]:
 al,bl,cc=m.roll_coefficients(m.F_REF);aa=ratio*o.AF
 roots,iscentre=m.equilibria(aa,al,bl,cc,n=10000);cs=roots[iscentre]
 phi=cs[np.argmin(abs(np.angle(np.exp(1j*(cs-np.deg2rad(314.4798663))))))]
 sol=o.integrate(ratio,[phi,0],orbits=1); ss=np.linspace(0,2*np.pi,1801);ff,pp,vv=sol.sol(ss)
 rate=vv*o.n; _,r,_,fd=m.orbit_state(ff)
 fdd=-2*fd*m.orbit_state(ff)[2]/r
 required1=m.A_INERTIA*fdd*np.sin(pp)+(m.A_INERTIA+m.C_INERTIA-m.B_INERTIA)*fd*rate*np.cos(pp)
 required2=m.B_INERTIA*fdd*np.cos(pp)+(m.A_INERTIA-m.B_INERTIA-m.C_INERTIA)*fd*rate*np.sin(pp)
 radial=aa*m.lorentz_projections(ff)[2]
 hold1=required1-m.Y0*radial;hold2=required2+m.X0*radial
 rows.append(dict(ratio=ratio,initial_phi_deg=float(np.rad2deg(phi)),span_deg=float(np.rad2deg(np.ptp(pp))),peak_holding_microNm_10kg=float(1e7*np.max(np.hypot(hold1,hold2)))))
o.write('one_orbit_existing_table.csv',rows)
tests=[]
for tol in [1e-8,1e-10,1e-12]:
 row,curve=c.run(.9,perturb_deg=2,orbits=3,rtol=tol,feedforward_enabled=True,feedback_enabled=True)
 tests.append(dict(rtol=tol,**row));print(tol,row,flush=True)
o.write('three_axis_tolerance.csv',tests)
for key in ['end_radial_misalignment_deg','max_abs_roll_tracking_error_deg','peak_two_axis_control_microNm_10kg']:
 assert abs(tests[1][key]-tests[2][key])<1e-5
print(rows,flush=True)

