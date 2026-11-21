"""Matched constrained-roll integrations and one-orbit Hamiltonian shooting.
Time s=n*t, momentum coordinate p=phi_dot/n. Samples s=2*pi*k
start at true anomaly 60 deg, so every map point has identical orbital phase.
"""
from pathlib import Path
import sys,csv,json
from multiprocessing import Pool
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root,brentq,least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import roll_model as m
OUT=Path(__file__).resolve().parents[1]/'data'/'results'; OUT.mkdir(parents=True,exist_ok=True)
n=np.sqrt(m.MU_E/m.A_ORB**3); AF=m.reference_values()['alphaF']
# Initial angle exactly as printed in the manuscript (frozen centre at 0.5 alpha_F,
# rounded to six decimals). Circulating cases are sensitive to this rounding, so the
# printed value is used to make the released data reproducible from the text.
PHI0_DEG=347.262167
RATIOS=[0.,.25,.5,.75,.85,.9,1.1]
ENSEMBLE_OFFSETS_DEG=np.linspace(-.01,.01,21)
PROBE_OFFSETS_DEG=[-1e-6,1e-6]
SAMPLES_PER_ORBIT=720

def rhs(s,y,ratio,variational=False,frozen=False):
    f,phi,p=y[:3]; al,bl,c=m.roll_coefficients(m.F_REF if frozen else f)
    fdot=m.orbit_state(f)[3]
    dy=[fdot/n,p,m.h(phi,ratio*AF,al,bl,c)/(m.C_INERTIA*n*n)]
    if variational:
        mat=y[3:].reshape(2,2)
        jac=np.array([[0.,1.],[m.h_phi(phi,ratio*AF,al,bl,c)/(m.C_INERTIA*n*n),0.]])
        dy.extend((jac@mat).ravel())
    return dy

def integrate(ratio,z0,orbits=32,rtol=1e-11,step=2*np.pi/120,variational=False,frozen=False):
    y0=[m.F_REF,*z0];
    if variational:y0.extend(np.eye(2).ravel())
    sol=solve_ivp(lambda s,y:rhs(s,y,ratio,variational,frozen),[0,2*np.pi*orbits],y0,
        method='DOP853',rtol=rtol,atol=rtol*.01,max_step=step,dense_output=True)
    if not sol.success:raise RuntimeError(sol.message)
    return sol

def excursion(job):
    """Running unwrapped excursion of one forced trajectory over 32 orbits."""
    ratio,offset=job
    sol=integrate(ratio,[np.deg2rad(PHI0_DEG+offset),0])
    phi=np.rad2deg(sol.sol(np.linspace(0,64*np.pi,32*SAMPLES_PER_ORBIT+1))[1])
    span=np.maximum.accumulate(phi)-np.minimum.accumulate(phi)
    onset=float(np.argmax(span>360.)/SAMPLES_PER_ORBIT) if span[-1]>360. else float('nan')
    return dict(ratio=ratio,initial_angle_offset_deg=float(offset),
        span_8_orbits_deg=float(span[8*SAMPLES_PER_ORBIT]),span_32_orbits_deg=float(span[-1]),
        circulation_onset_orbits=onset)

def ensemble():
    """Initial-angle ensemble behind the matched-data table.

    Onset is the first time the running excursion exceeds 360 degrees. The
    32-orbit spans of circulating cases are sample ranges, not converged values.
    """
    jobs=[(r,d) for r in RATIOS for d in [*ENSEMBLE_OFFSETS_DEG,*PROBE_OFFSETS_DEG]]
    with Pool() as pool:members=pool.map(excursion,jobs)
    write('orbital_ensemble_members.csv',members)
    summary=[]
    for r in RATIOS:
        mine=[v for v in members if v['ratio']==r]
        ens=[v for v in mine if abs(v['initial_angle_offset_deg'])!=1e-6]
        nominal=min(ens,key=lambda v:abs(v['initial_angle_offset_deg']))
        probe={v['initial_angle_offset_deg']:v for v in mine if abs(v['initial_angle_offset_deg'])==1e-6}
        onsets=[v['circulation_onset_orbits'] for v in ens]
        summary.append(dict(ratio=r,ensemble_size=len(ens),
            nominal_span_8_orbits_deg=nominal['span_8_orbits_deg'],
            nominal_span_32_orbits_deg=nominal['span_32_orbits_deg'],
            nominal_onset_orbits=nominal['circulation_onset_orbits'],
            ensemble_span_8_min_deg=min(v['span_8_orbits_deg'] for v in ens),
            ensemble_span_8_max_deg=max(v['span_8_orbits_deg'] for v in ens),
            ensemble_span_32_min_deg=min(v['span_32_orbits_deg'] for v in ens),
            ensemble_span_32_max_deg=max(v['span_32_orbits_deg'] for v in ens),
            ensemble_onset_min_orbits=float(np.nanmin(onsets)) if not np.all(np.isnan(onsets)) else float('nan'),
            ensemble_onset_max_orbits=float(np.nanmax(onsets)) if not np.all(np.isnan(onsets)) else float('nan'),
            fraction_circulating=float(np.mean([v['span_32_orbits_deg']>360. for v in ens])),
            span_32_offset_minus_1e6th_deg=probe[-1e-6]['span_32_orbits_deg'],
            span_32_offset_plus_1e6th_deg=probe[1e-6]['span_32_orbits_deg']))
    write('orbital_ensemble_summary.csv',summary)
    return summary

def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    al,bl,c=m.roll_coefficients(m.F_REF)
    angles,centres=m.equilibria(.5*AF,al,bl,c,n=12000)
    pcs=angles[centres]; target=np.deg2rad(314.4798663)
    centre=pcs[np.argmin(abs(np.angle(np.exp(1j*(pcs-target)))))]
    phi0=np.deg2rad(PHI0_DEG)
    assert abs(np.rad2deg(centre)-PHI0_DEG)<1e-6
    fs=np.linspace(0,2*np.pi,40001); threshold=m.fold_threshold(*m.roll_coefficients(fs))/AF
    rows=[]; points=[]; curves={}; sensitivity=[]
    for ratio in RATIOS:
        sol=integrate(ratio,[phi0,0]);st=2*np.pi*np.arange(33); po=sol.sol(st)
        dense=sol.sol(np.linspace(0,64*np.pi,32*720+1)); curves[str(ratio)]=dense
        check=integrate(ratio,[phi0,0],orbits=3,rtol=2e-13,step=2*np.pi/240)
        baseline=sol.sol(6*np.pi);delta=check.y[:,-1]-baseline
        frozen=integrate(ratio,[phi0,0],frozen=True)
        fdense=frozen.sol(np.linspace(0,64*np.pi,32*720+1))
        row=dict(ratio=ratio,initial_phi_deg=np.rad2deg(phi0),initial_p=0.,orbits=32,
            frozen_fold_crossed=bool(ratio>threshold.min() and ratio<threshold.max()),
            fraction_true_anomaly_above_fold=float(np.mean(ratio>threshold)),
            phi_span_deg=float(np.rad2deg(np.ptp(dense[1]))),
            max_abs_p=float(abs(dense[2]).max()),net_turns=float((dense[1,-1]-phi0)/(2*np.pi)),
            frozen_phi_span_deg=float(np.rad2deg(np.ptp(fdense[1]))),
            refinement_3orbit_phi_rad=float(abs(delta[1])),refinement_3orbit_p=float(abs(delta[2])),
            strobe_anomaly_error_rad=float(np.max(abs(po[0]-m.F_REF-st))))
        assert row['strobe_anomaly_error_rad']<1e-8
        assert max(row['refinement_3orbit_phi_rad'],row['refinement_3orbit_p'])<2e-8
        rows.append(row)
        for k in range(33):points.append(dict(ratio=ratio,orbit=k,phi_rad=po[1,k],p=po[2,k],f_rad=po[0,k]))
        for dphi in [-2.,-.01,.01,2.]:
            pert=integrate(ratio,[phi0+np.deg2rad(dphi),0])
            zz=pert.sol(st)
            dist=np.hypot(np.angle(np.exp(1j*(zz[1]-po[1]))),zz[2]-po[2])
            sensitivity.append(dict(ratio=ratio,initial_angle_offset_deg=dphi,
                final_cylindrical_separation=float(dist[-1]),max_strobe_separation=float(dist.max())))
    write('orbital_matched_summary.csv',rows);write('orbital_stroboscopic_points.csv',points)
    write('orbital_initial_condition_sensitivity.csv',sensitivity)
    np.savez_compressed(OUT/'orbital_matched_trajectories.npz',s=np.linspace(0,64*np.pi,32*720+1),**curves)
    print(*ensemble(),sep='\n')
    shooting,metadata=continue_periodic(phi0,threshold)
    plot_saved()
    print(json.dumps(metadata,indent=2));print(*rows,sep='\n')

def continue_periodic(phi0,threshold):
    # Continue the non-rotating 1-orbit response from the uncharged centre.
    shooting=[];z=np.array([0.,0.]); failed=[]
    for ratio in np.linspace(0,1.1,56):
        def fun(z):
            ss=integrate(ratio,z,orbits=1,variational=True)
            return ss.y[1:3,-1]-z
        def jac(z):
            ss=integrate(ratio,z,orbits=1,variational=True)
            return ss.y[3:,-1].reshape(2,2)-np.eye(2)
        rr=least_squares(fun,z,jac=jac,bounds=([-np.pi,-2.],[np.pi,2.]),
                         xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=40)
        residual=float(np.linalg.norm(fun(rr.x)))
        if residual>1e-8:
            failed.append(dict(ratio=float(ratio),residual=residual));break
        z=rr.x; ss=integrate(ratio,z,orbits=1,variational=True)
        M=ss.y[3:,-1].reshape(2,2); eig=np.linalg.eigvals(M)
        assert abs(np.linalg.det(M)-1)<1e-7
        refined=integrate(ratio,z,orbits=1,variational=True,rtol=2e-13,step=2*np.pi/240)
        mr=refined.y[3:,-1].reshape(2,2)
        print('periodic shooting ratio',ratio,'residual',residual,flush=True)
        shooting.append(dict(ratio=float(ratio),phi0_rad=z[0],p0=z[1],residual=residual,
            trace=float(np.trace(M)),det=float(np.linalg.det(M)),max_multiplier_modulus=float(abs(eig).max()),
            multiplier1_real=float(eig[0].real),multiplier1_imag=float(eig[0].imag),
            multiplier2_real=float(eig[1].real),multiplier2_imag=float(eig[1].imag),
            refined_matrix_max_difference=float(abs(M-mr).max())))
    write('orbital_periodic_branch.csv',shooting)
    metadata=dict(reference_fold_Ckg=AF,initial_phi_rad=phi0,mean_motion_s=n,
        orbital_period_s=2*np.pi/n,minimum_instantaneous_fold_ratio=float(threshold.min()),
        max_step_s=(2*np.pi/120)/n,rtol=1e-11,atol=1e-13,shooting_failures=failed,
        interpretation='Finite-time maps do not establish global boundedness or chaos. Floquet classification applies only to shooting solutions.')
    (OUT/'orbital_metadata.json').write_text(json.dumps(metadata,indent=2))
    return shooting,metadata

def plot_saved():
    rows=list(csv.DictReader((OUT/'orbital_matched_summary.csv').open()))
    phi0=np.deg2rad(float(rows[0]['initial_phi_deg']))
    curves=np.load(OUT/'orbital_matched_trajectories.npz')
    points=[{k:float(v) for k,v in row.items()} for row in csv.DictReader((OUT/'orbital_stroboscopic_points.csv').open())]
    shooting=[{k:float(v) for k,v in row.items()} for row in csv.DictReader((OUT/'orbital_periodic_branch.csv').open())]
    fig,ax=plt.subplots(1,3,figsize=(12,3.5))
    for ratio in [float(row['ratio']) for row in rows]:
        vals=curves[str(ratio)]
        ax[0].plot(np.linspace(0,32,vals.shape[1]),np.rad2deg(vals[1]-phi0),lw=.7,label=f'{ratio:g}')
        pp=[v for v in points if v['ratio']==ratio]
        ax[1].scatter([np.rad2deg(np.angle(np.exp(1j*v['phi_rad']))) for v in pp],
            [v['p'] for v in pp],s=9,label=f'{ratio:g}')
    ax[0].set_xlabel('Orbital periods');ax[0].set_ylabel('Unwrapped roll change (degrees)');ax[0].legend(fontsize=6)
    ax[1].set_xlabel('Roll modulo 360 degrees');ax[1].set_ylabel(r'$p=\dot\phi/n$');ax[1].legend(fontsize=6)
    ax[2].plot([v['ratio'] for v in shooting],[v['trace'] for v in shooting]);ax[2].axhline(2,c='grey',ls='--');ax[2].axhline(-2,c='grey',ls='--')
    ax[2].set_xlabel('Charge / reference fold');ax[2].set_ylabel('One-orbit monodromy trace')
    for a in ax:a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'orbital_matched_poincare.pdf');plt.close(fig)

def periodic_branch_only():
    rows=list(csv.DictReader((OUT/'orbital_matched_summary.csv').open()))
    phi0=np.deg2rad(float(rows[0]['initial_phi_deg']))
    fs=np.linspace(0,2*np.pi,40001);threshold=m.fold_threshold(*m.roll_coefficients(fs))/AF
    return continue_periodic(phi0,threshold)


def refine_long_horizon():
    rows=list(csv.DictReader((OUT/'orbital_matched_summary.csv').open()))
    result=[]
    for row in rows:
        ratio=float(row['ratio']);phi=np.deg2rad(float(row['initial_phi_deg']))
        sol=integrate(ratio,[phi,0],orbits=32,rtol=2e-13,step=2*np.pi/240)
        y=sol.sol(np.linspace(0,64*np.pi,32*720+1))
        result.append(dict(ratio=ratio,refined_phi_span_deg=float(np.rad2deg(np.ptp(y[1]))),
            refined_max_p=float(abs(y[2]).max()),refined_net_turns=float((y[1,-1]-phi)/(2*np.pi)),
            coarse_phi_span_deg=float(row['phi_span_deg'])))
    write('orbital_32orbit_refinement.csv',result)

if __name__=='__main__':
    if '--shooting-only' in sys.argv:periodic_branch_only()
    elif '--plot-only' in sys.argv:plot_saved()
    elif '--refine-only' in sys.argv:refine_long_horizon()
    elif '--ensemble-only' in sys.argv:print(*ensemble(),sep='\n')
    else:
        main()
        refine_long_horizon()

