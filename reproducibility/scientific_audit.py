"""Independent root/scaling, energy/period, and official coefficient checks."""
from pathlib import Path
import sys,json,hashlib,platform
import numpy as np
import scipy,matplotlib
from scipy.optimize import brentq
from scipy.integrate import solve_ivp,quad
from model import roll_model as m
from numerical_experiments.controlled_three_axis_validation import init_orbit,coeff,reference_dynamics_check
OUT=Path(__file__).resolve().parents[1]/'data'/'results';OUT.mkdir(parents=True,exist_ok=True)
v=m.reference_fold(); al,bl,c,af,phif=(v[k] for k in ['aL','bL','c','alphaF','phiF'])
# Three-axis vector formulation gives independent coefficients from orbital states.
R,V=init_orbit(.001); independent=coeff(R,V)
assert np.allclose(independent[:3],[al,bl,c],rtol=2e-12,atol=1e-18)
assert abs(m.h(phif,af,al,bl,c))<1e-20
assert abs(m.h_phi(phif,af,al,bl,c))<1e-20
reference_dynamics_check()
xs=np.logspace(-6,-3,21); sep=[]; barrier=[]; periods=[]
for x in xs:
 alpha=(1-x)*af
 # Bracket independently around the exact double root.
 lo=brentq(lambda z:m.h(z,alpha,al,bl,c),phif-.2,phif,xtol=1e-14)
 hi=brentq(lambda z:m.h(z,alpha,al,bl,c),phif,phif+.2,xtol=1e-14)
 pc,ps=(lo,hi) if m.h_phi(lo,alpha,al,bl,c)<0 else (hi,lo)
 sep.append(abs(pc-ps));barrier.append(abs(m.potential(pc,alpha,al,bl,c)-m.potential(ps,alpha,al,bl,c)))
 periods.append(2*np.pi*np.sqrt(m.C_INERTIA/-m.h_phi(pc,alpha,al,bl,c)))
slopes=[float(np.polyfit(np.log(xs),np.log(y),1)[0]) for y in [sep,barrier,periods]]
assert max(abs(np.array(slopes)-[.5,1.5,-.25]))<5e-4
# Integrate a frozen libration and compare a return event with energy quadrature.
alpha=.5*af; roots,centres=m.equilibria(alpha,al,bl,c); pc=roots[centres][-1]
phi0=pc+.05;energy=m.potential(phi0,alpha,al,bl,c)
turn=brentq(lambda x:m.potential(x,alpha,al,bl,c)-energy,pc-.2,pc)
mid=(turn+phi0)/2;half=(phi0-turn)/2
# sin substitution removes endpoint singularities.
def integrand(theta):
 phi=mid+half*np.sin(theta)
 return half*np.cos(theta)/np.sqrt(2*(energy-m.potential(phi,alpha,al,bl,c))/m.C_INERTIA)
period_quad=2*quad(integrand,-np.pi/2,np.pi/2,epsabs=1e-5,epsrel=1e-9)[0]
def event(t,y):return y[0]-pc
event.direction=-1
sol=solve_ivp(lambda t,y:[y[1],m.h(y[0],alpha,al,bl,c)/m.C_INERTIA],
 [0,3.1*period_quad],[phi0,0],events=event,rtol=1e-12,atol=1e-14,
 max_step=period_quad/200,dense_output=True,method='DOP853')
assert sol.success
period_event=float(sol.t_events[0][1]-sol.t_events[0][0])
energy_values=.5*m.C_INERTIA*sol.y[1]**2+m.potential(sol.y[0],alpha,al,bl,c)
drift=float(np.ptp(energy_values)/(energy-m.potential(pc,alpha,al,bl,c)))
assert drift<1e-7 and abs(period_event/period_quad-1)<1e-7
p=Path(__file__).resolve().parents[1]/'data'
official={};local={}
for line in (p/'igrf14coeffs_official.txt').read_text().splitlines():
 t=line.split()
 if t and t[0] in ['g','h']:official[tuple(t[:3])]=float(t[-2])
for line in (p/'igrf14_2025.txt').read_text().splitlines():
 t=line.split()
 if t and t[0] in ['g','h']:local[tuple(t[:3])]=float(t[-1])
assert len(official)==195 and local==official
results=dict(reference=v,scaling_fit_relative_delta=[float(xs.min()),float(xs.max())],scaling_slopes=slopes,
 frozen_period_quadrature_s=period_quad,frozen_period_event_s=period_event,
 frozen_relative_energy_drift=drift,official_IGRF_coefficients_equal=195,
 official_IGRF_sha256=hashlib.sha256((p/'igrf14coeffs_official.txt').read_bytes()).hexdigest(),
 versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__))
(OUT/'scientific_audit.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

# Independently differentiate one-orbit maps to verify the variational propagator.
from numerical_experiments import orbital_poincare as o
import csv
if (OUT/'orbital_periodic_branch.csv').exists():
    shooting=list(csv.DictReader((OUT/'orbital_periodic_branch.csv').open()))
    comparison=[]
    for q in [.5,.9,1.1]:
        row=min(shooting,key=lambda r:abs(float(r['ratio'])-q))
        z=np.array([float(row['phi0_rad']),float(row['p0'])]);step=1e-5
        sol=o.integrate(q,z,orbits=1,variational=True,rtol=2e-13,step=2*np.pi/240)
        mat=sol.y[3:,-1].reshape(2,2);fd=np.empty((2,2))
        for j in range(2):
            dz=np.eye(2)[j]*step
            fd[:,j]=(o.integrate(q,z+dz,orbits=1,rtol=2e-13).y[1:3,-1]-o.integrate(q,z-dz,orbits=1,rtol=2e-13).y[1:3,-1])/(2*step)
        err=float(abs(fd-mat).max());assert err<1e-7
        comparison.append(dict(ratio=q,finite_difference_step=step,max_matrix_difference=err))
    o.write('orbital_floquet_finite_difference.csv',comparison)

