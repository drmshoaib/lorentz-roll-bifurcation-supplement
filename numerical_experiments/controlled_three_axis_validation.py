#!/usr/bin/env python3
"""Three-axis consistency and transverse-control tests for Lorentz roll.

SI units, inertias mass-normalised I/m (m^2). Central Kepler orbit, centred
geomagnetic dipole, displaced charge, gravity-gradient. No intrinsic dipole.
State includes an independent constrained roll reference on the SAME orbit.

The two-axis controller is a reference-tracking diagnostic. Its third body-axis
component is identically zero, so it does not directly impose roll dynamics.
Validation: nominal exact recovery, perturbed transverse tracking, circular
uncharged gravity-gradient invariant solution and integration tolerance check.
"""
import csv
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from scipy.optimize import brentq

MU = 3.9860e14
W_E = 7.2921e-5
B_D = 7.943e15
A, B, C = 0.4, 1.0, 1.0  # I/m, m^2
I = np.diag([A,B,C]); I_INV=np.diag([1/A,1/B,1/C]);
RHO = np.array([1.,1.,0.])
AORB = 6900e3
INC = np.deg2rad(51.)
F0 = np.deg2rad(60.)
P_DIR = np.array([1.,0.,0.])
Q_DIR = np.array([0.,np.cos(INC),np.sin(INC)])
DIPOLE = np.array([0.,0.,1.])
OMEGA_E = np.array([0.,0.,W_E])


def init_orbit(e):
    p=AORB*(1-e*e)
    r=p/(1+e*np.cos(F0))
    n=np.sqrt(MU*p)/r**2
    rdot=e*np.sqrt(MU/p)*np.sin(F0)
    rh=np.cos(F0)*P_DIR+np.sin(F0)*Q_DIR
    th=-np.sin(F0)*P_DIR+np.cos(F0)*Q_DIR
    return r*rh, rdot*rh+r*n*th


def kepler(R, V):
    rm=np.linalg.norm(R)
    rh=R/rm
    normal=np.cross(R,V)
    nh=normal/np.linalg.norm(normal)
    th=np.cross(nh,rh)
    n=np.linalg.norm(normal)/rm**2
    rdot=np.dot(V,rh)
    ndot=-2*rdot*n/rm
    E=np.column_stack((th,nh,rh))
    return rm,rh,nh,th,n,ndot,E


def field_force(R,V):
    rm=np.linalg.norm(R);rh=R/rm
    b=B_D/rm**3 * (3*np.dot(DIPOLE,rh)*rh-DIPOLE)
    return np.cross(V-np.cross(OMEGA_E,R),b)


def coeff(R,V):
    rm,rh,nh,th,n,ndot,E=kepler(R,V)
    f=field_force(R,V)
    T,N,Rbar = th@f,nh@f,rh@f
    al=RHO[0]*N-RHO[1]*T
    bl=-(RHO[0]*T+RHO[1]*N)
    cc=.5*n*n*(A-B)
    return al,bl,cc,Rbar,n,ndot,E


def fold(aL,bL,c):
    return 2*abs(c)/(abs(aL)**(2/3)+abs(bL)**(2/3))**1.5


def centre(R,V,alpha):
    al,bl,c,*_=coeff(R,V)
    def h(phi):return alpha*(al*np.cos(phi)+bl*np.sin(phi))+c*np.sin(2*phi)
    def hp(phi):return alpha*(-al*np.sin(phi)+bl*np.cos(phi))+2*c*np.cos(2*phi)
    x=np.linspace(0,2*np.pi,12001);y=h(x)
    roots=[brentq(h,x[k],x[k+1]) for k in np.flatnonzero(y[:-1]*y[1:]<0)]
    centres=[z for z in roots if hp(z)<0]
    if not centres:raise ValueError("No stable roll centre")
    target=np.deg2rad(314.4798663)
    return min(centres,key=lambda v:abs(np.angle(np.exp(1j*(v-target)))))


def ref_orientation(E,phi):
    co,si=np.cos(phi),np.sin(phi)
    rot=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
    return E@rot


def nominal_rates(phi,rate,n,ndot):
    # Euler angular velocity in body frame (first two components are orbital rate)
    s,co=np.sin(phi),np.cos(phi)
    return np.array([n*s,n*co,rate]),np.array([
      ndot*s+n*rate*co,
      ndot*co-n*rate*s,
      0.  # computed later from Lorentz roll and inertial coupling
    ])


def feedforward(R,V,phi,rate,alpha):
    al,bl,c,rbar,n,ndot,E=coeff(R,V)
    t1=A*ndot*np.sin(phi)+(A+C-B)*n*rate*np.cos(phi)-RHO[1]*alpha*rbar
    t2=B*ndot*np.cos(phi)+(A-B-C)*n*rate*np.sin(phi)+RHO[0]*alpha*rbar
    return np.array([t1,t2,0.]),(al,bl,c,n,ndot,E)


def quaternion_dot(q,omega):
    x,y,z,w=q; a,b,c=omega
    return .5*np.array([w*a+y*c-z*b,
                         w*b+z*a-x*c,
                         w*c+x*b-y*a,
                        -x*a-y*b-z*c])


def control(R,V,q,omega,phi,rate,alpha,use_feedforward=True,use_feedback=False,
            control_period_fraction=.25,zeta=.9):
    # C maps body to inertial, E maps orbital RTN to inertial
    rm,rh,nh,th,n,ndot,E=kepler(R,V)
    Ccur=Rotation.from_quat(q/np.linalg.norm(q)).as_matrix()
    Cref=ref_orientation(E,phi)
    ff, _=feedforward(R,V,phi,rate,alpha)
    # Error: rotation from current body frame to reference body frame.
    rotation_error=Rotation.from_matrix(Ccur.T@Cref).as_rotvec()
    wr=np.array([n*np.sin(phi),n*np.cos(phi),rate])
    wr_current=Ccur.T@(Cref@wr)
    erate=omega-wr_current
    tau=np.array(ff if use_feedforward else np.zeros(3))
    if use_feedback:
        omega_ctrl=2*np.pi/(control_period_fraction*(2*np.pi*np.sqrt(AORB**3/MU)))
        kp=np.array([A,B])*omega_ctrl**2
        kd=np.array([A,B])*2*zeta*omega_ctrl
        tau[:2]+=kp*rotation_error[:2]-kd*erate[:2]
    tau[2]=0.0
    return tau,rotation_error,erate


def torque_environment(R,V,q,alpha):
    Ccur=Rotation.from_quat(q/np.linalg.norm(q)).as_matrix()
    rm=np.linalg.norm(R); rh=R/rm
    rb=Ccur.T@rh
    gg=3*MU/rm**3 * np.cross(rb,I@rb)
    lorentz=alpha*np.cross(RHO,Ccur.T@field_force(R,V))
    return gg+lorentz


def deriv(t,y,alpha,feedforward_enabled,feedback_enabled,ctrl_period_fraction,zeta):
    R=y[:3]; V=y[3:6]; q=y[6:10]; w=y[10:13]; phi=y[13]; rate=y[14]
    ff, (al,bl,c,n,ndot,E)=feedforward(R,V,phi,rate,alpha)
    ctrl, _, _=control(R,V,q,w,phi,rate,alpha,feedforward_enabled,feedback_enabled,control_period_fraction=ctrl_period_fraction,zeta=zeta)
    tau=torque_environment(R,V,q,alpha)+ctrl
    wdot=I_INV@(tau-np.cross(w,I@w))
    h=alpha*(al*np.cos(phi)+bl*np.sin(phi))+c*np.sin(2*phi)
    return np.r_[V,-MU*R/np.linalg.norm(R)**3,quaternion_dot(q,w),wdot,rate,h/C]


def run(frac,e=.001,perturb_deg=0.,orbits=1.,feedforward_enabled=True,
        feedback_enabled=True,rtol=1e-10,samples_per_orbit=720,
        ctrl_period_fraction=.25,zeta=.9):
    R0,V0=init_orbit(e)
    al,bl,c,_,n,ndot,E=coeff(R0,V0)
    alpha_fold=fold(al,bl,c)
    alpha=frac*alpha_fold
    phi0=centre(R0,V0,alpha)
    Cref=ref_orientation(E,phi0)
    p=np.deg2rad(perturb_deg)
    dC=Rotation.from_rotvec(np.array([p,-.5*p,0.])).as_matrix()
    Ccur=Cref@dC
    q0=Rotation.from_matrix(Ccur).as_quat()
    wref=np.array([n*np.sin(phi0),n*np.cos(phi0),0.])
    w0=Ccur.T@(Cref@wref)  # no initial relative angular-rate error
    y0=np.r_[R0,V0,q0,w0,phi0,0.]
    period=2*np.pi*np.sqrt(AORB**3/MU)
    t_eval=np.linspace(0.,orbits*period,int(samples_per_orbit*orbits)+1)
    atol=np.r_[[1e-6]*3,[1e-8]*3,[1e-12]*4,[1e-13]*3,[1e-12,1e-14]]
    sol=solve_ivp(lambda t,y:deriv(t,y,alpha,feedforward_enabled,feedback_enabled,ctrl_period_fraction,zeta),
                  (0.,orbits*period),y0,t_eval=t_eval,rtol=rtol,atol=atol,method='DOP853',max_step=period/100)
    if not sol.success:raise RuntimeError(sol.message)
    tilt=[]; roll_diff=[]; torques=[]; errors=[]; werrors=[]; control3=[]
    for y in sol.y.T:
        R,V,q,w,phi,rate=y[:3],y[3:6],y[6:10],y[10:13],y[13],y[14]
        _,rh,nh,th,n,ndot,E=kepler(R,V)
        Ccur=Rotation.from_quat(q/np.linalg.norm(q)).as_matrix()
        tilt.append(np.rad2deg(np.arccos(np.clip(Ccur[:,2]@rh,-1,1))))
        phi_actual=np.arctan2(Ccur[:,0]@nh,Ccur[:,0]@th)
        roll_diff.append(np.rad2deg(np.angle(np.exp(1j*(phi_actual-phi)))))
        ctrl,err,werr=control(R,V,q,w,phi,rate,alpha,feedforward_enabled,feedback_enabled,control_period_fraction=ctrl_period_fraction,zeta=zeta)
        torques.append(ctrl)
        errors.append(np.rad2deg(err))
        werrors.append(werr)
        control3.append(ctrl[2])
    torques=np.asarray(torques); errors=np.asarray(errors);
    tilt=np.array(tilt);roll_diff=np.array(roll_diff)
    mt=10.0*np.linalg.norm(torques[:,:2],axis=1)*1e6
    data={
        'charge_fraction':frac,'eccentricity':e,'perturbation_deg':perturb_deg,
        'feedforward':feedforward_enabled,'feedback':feedback_enabled,
        'feedback_period_fraction':ctrl_period_fraction,'zeta':zeta,
        'max_radial_misalignment_deg':float(np.max(tilt)),
        'end_radial_misalignment_deg':float(tilt[-1]),
        'max_abs_roll_tracking_error_deg':float(np.max(np.abs(roll_diff))),
        'end_roll_tracking_error_deg':float(roll_diff[-1]),
        'peak_two_axis_control_microNm_10kg':float(np.max(mt)),
        'control_abs_integral_mNms_10kg':float(np.trapezoid(mt,sol.t)/1000),
        'max_abs_control_roll_torque_Nm_10kg':float(np.max(np.abs(control3))*10),
        'nfev':sol.nfev,
    }
    return data, (sol.t,tilt,roll_diff,mt,errors)


def reference_dynamics_check(e=.001):
    R,V=init_orbit(e)
    al,bl,c,rr,n,ndot,E=coeff(R,V)
    af=fold(al,bl,c);phi=centre(R,V,.5*af)
    for rate in [0.,1e-4]:
        w,wdot=nominal_rates(phi,rate,n,ndot)
        rollacc=(.5*af*(al*np.cos(phi)+bl*np.sin(phi))+c*np.sin(2*phi))/C
        wdot[2]=rollacc
        Cref=ref_orientation(E,phi)
        q=Rotation.from_matrix(Cref).as_quat()
        t_env=torque_environment(R,V,q,.5*af)
        hold,_,_=control(R,V,q,w,phi,rate,.5*af,True,False)
        torque=t_env+hold
        euler=I@wdot+np.cross(w,I@w)
        residual=euler-torque
        print('EULER pointwise rate=',rate,'residual Nm/kg=',residual,'norm=',np.linalg.norm(residual))
        assert np.linalg.norm(residual)<1e-17, 'Pointwise Euler consistency failed'


def main():
    reference_dynamics_check()
    out=Path(__file__).resolve().parents[1]/'data'/'results';out.mkdir(parents=True,exist_ok=True)
    scenarios=[
     (0.,.0,False,False,'circular uncharged exact control',0.),
     (.0,.001,False,False,'elliptic uncharged free',0.),
     (.25,.001,False,False,'free 0.25 fold',0.),
     (.5,.001,False,False,'free 0.5 fold',0.),
     (.9,.001,False,False,'free 0.9 fold',0.),
     (.25,.001,True,False,'feedforward .25',0.),
     (.5,.001,True,False,'feedforward .5',0.),
     (.9,.001,True,False,'feedforward .9',0.),
     (.25,.001,True,True,'perturbed ctrl .25',2.),
     (.5,.001,True,True,'perturbed ctrl .5',2.),
     (.9,.001,True,True,'perturbed ctrl .9',2.),
    ]
    results=[]; curves={}
    for frac,e,ff,fb,label,pert in scenarios:
        row, curve=run(frac,e,pert,orbits=1,feedforward_enabled=ff,feedback_enabled=fb)
        row['case']=label
        results.append(row);curves[label]=curve
        print('%-29s charge %.2f tilt max %8.4f deg tilt end %8.4f deg roll max %8.4f deg peak %.4f uNm'%(label,frac,row['max_radial_misalignment_deg'],row['end_radial_misalignment_deg'],row['max_abs_roll_tracking_error_deg'],row['peak_two_axis_control_microNm_10kg']))
    # The three-orbit tests are essential: close radial alignment at one
    # orbital period need not imply longitudinal/roll phase stability.
    for frac in [.25,.5,.9]:
        for perturb in [0.,2.]:
            row,curve=run(frac,e=.001,perturb_deg=perturb,orbits=3.,
                          feedforward_enabled=True,feedback_enabled=True,
                          samples_per_orbit=720)
            label=f'three_orbit_control_{frac:g}_tilt_{perturb:g}'
            row['case']=label
            results.append(row)
            curves[label]=curve
            print('%-29s max tilt %8.4f end tilt %8.4f max roll err %8.4f peak torque %.4f microN m'
                  %(label,row['max_radial_misalignment_deg'],
                    row['end_radial_misalignment_deg'],
                    row['max_abs_roll_tracking_error_deg'],
                    row['peak_two_axis_control_microNm_10kg']))
    with (out/'three_axis_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]),lineterminator='\n');writer.writeheader();writer.writerows(results)
    np.savez_compressed(out/'three_axis_trajectories.npz',**{k+'__'+j:np.asarray(v) for k,seq in curves.items() for j,v in zip(['time','tilt','roll_error','torque','orientation_error'],seq)})
    return results,curves


if __name__=='__main__':main()

