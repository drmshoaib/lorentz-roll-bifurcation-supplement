"""Current-balance screening, not a sheath solution or hardware qualification.
Run from any directory. SI units except temperatures in eV (=volts here).
OML output is explicitly an extrapolation if R is not small versus lambda_D.
"""
from pathlib import Path
import sys, csv, json
import numpy as np
from scipy.constants import epsilon_0, elementary_charge as qe, m_e, atomic_mass, c as light_speed
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import roll_model as m
OUT=Path(__file__).resolve().parents[1]/'data'/'results'; OUT.mkdir(parents=True,exist_ok=True)
AF=m.reference_values()['alphaF']
# O+ mass from the standard atomic weight of oxygen (15.999 u), not 16 proton masses.
M_OXYGEN_ION=15.999*atomic_mass
# Magnitude of the manuscript dipole at the reference orbital location (3.7192e-5 T),
# rounded to the three figures quoted in the text; main() checks it against roll_model.
B_MAG_REF=3.72e-5

def ion_current_negative_surrogate(u, ram_current, thermal_current, ram_energy_V):
    """Legacy screening interpolation, restricted to -E_ram/e <= U <= 0.

    It is not a drifting-ion/sheath collection law. In particular, it must
    not supply a positive-bias current or be used to infer one.
    """
    if ram_energy_V <= 0 or not -ram_energy_V <= u <= 0:
        raise ValueError("Ion screening surrogate requires -E_ram/e <= U <= 0")
    return (ram_current + thermal_current) * (1 - u / ram_energy_V)


def negative_bias_root(balance, ram_energy_V):
    """Return a screening root only within the stated non-positive domain.

    None means no root is bracketed in this domain, not a prediction of a
    positive floating potential or of the absence of a physical equilibrium.
    """
    lo, hi = -ram_energy_V, 0.
    if balance(lo) * balance(hi) > 0:
        return None
    return float(brentq(balance, lo, hi, xtol=1e-13))


def scales(mass=10.,radius=1.,asym=.6,offset=1.,density=1e11,te=.2,ti=.1, speed=7600.,magnetic_field=B_MAG_REF):
    alpha=AF*asym/.6/offset
    charge=mass*alpha; cap=4*np.pi*epsilon_0*radius; U=charge/cap
    debye=np.sqrt(epsilon_0*te/(density*qe))
    ie0=4*np.pi*radius**2*qe*density*np.sqrt(qe*te/(2*np.pi*m_e))
    # Classical Parker--Murphy benchmark; ie0 uses the full sphere area.
    magnetic_voltage=qe*magnetic_field**2*radius**2/(8*m_e)
    pm_factor=.5*(1+np.sqrt(U/magnetic_voltage))
    pm_current=ie0*pm_factor
    ii_ram=np.pi*radius**2*qe*density*speed
    ii_th=4*np.pi*radius**2*qe*density*np.sqrt(qe*ti/(2*np.pi*M_OXYGEN_ION))
    # Positive-sphere OML. Deliberately retain validity flag in every row.
    ioml=ie0*(1+U/te)
    # The directed O+ energy sets the scale for an ion potential barrier.
    # No positive-bias ion collection law is evaluated here.
    eram=.5*M_OXYGEN_ION*speed**2/qe
    iph0=40e-6*np.pi*radius**2
    # Photoemission and escaping-yield sensitivities are evaluated separately.
    def balance(u,photo=True,yield_escape=0.):
        ii=ion_current_negative_surrogate(u,ii_ram,ii_th,eram)
        ie=ie0*np.exp(max(u/te,-700))
        iph=iph0 if photo else 0.
        return ii+iph-(1-yield_escape)*ie
    floating_dark=negative_bias_root(lambda u:balance(u,False),eram)
    floating_sun=negative_bias_root(balance,eram)
    return dict(mass_kg=mass,radius_m=radius,asymmetry_m2=asym,offset_component_m=offset,
        density_m3=density,Te_eV=te,Ti_eV=ti,alpha_Ckg=alpha,charge_C=charge,
        vacuum_cap_F=cap,vacuum_potential_V=U,debye_m=debye,R_over_debye=radius/debye,
        OML_small_body=bool(radius/debye<.1),thermal_electron_A=ie0,
        ion_ram_A=ii_ram,ion_thermal_A=ii_th,ion_ram_energy_eV=eram,
        photo_zero_bias_A=iph0,OML_extrapolated_A=ioml,
        geometric_power_W=ie0*U,OML_extrapolated_power_W=ioml*U,
        PM_magnetic_field_T=magnetic_field,PM_voltage_V=magnetic_voltage,
        PM_current_factor=pm_factor,PM_current_A=pm_current,PM_ideal_power_W=U*pm_current,
        PM_model_status='classical_spherical_collisionless_magnetic_collection_benchmark',
        PM_minimum_beam_recoil_N=pm_current*np.sqrt(U*(U+2*m_e*light_speed**2/qe))/light_speed,
        geometric_charge_loss_scale_s=charge/ie0,OML_extrapolated_loss_scale_s=charge/ioml,
        leakage_resistance_for_one_percent_per_orbit_ohm=(2*np.pi*np.sqrt(m.A_ORB**3/m.MU_E))/(cap*(-np.log(.99))),
        minimum_beam_recoil_at_geometric_current_N=ie0*np.sqrt(U*(U+2*m_e*light_speed**2/qe))/light_speed,
        vacuum_energy_J=.5*charge*U,vacuum_surface_field_Vm=U/radius,
        floating_dark_surrogate_V=floating_dark,floating_sun_surrogate_V=floating_sun,
        floating_dark_status=('negative_bias_screening_root' if floating_dark is not None else 'outside_screening_domain'),
        floating_sun_status=('negative_bias_screening_root' if floating_sun is not None else 'outside_screening_domain'))

def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    r_ref=m.A_ORB*(1-m.ECC**2)/(1+m.ECC*np.cos(m.F_REF))
    cos_colat=np.sin(m.INC)*np.sin(m.OMEGA_PERI+m.F_REF)
    assert np.isclose(B_MAG_REF,abs(m.B_D)/r_ref**3*np.sqrt(1+3*cos_colat**2),rtol=1e-3)
    rows=[scales(density=n,te=t) for n in [1e9,1e10,1e11,1e12] for t in [.1,.2,1.]]
    write('charging_environment.csv',rows)
    scan=[scales(mass=mass,radius=r,asym=d,offset=rho) for mass in [1.,10.,100.]
          for r in [.1,.5,1.,2.] for d in [.6,.06,.006,.0006] for rho in [.1,1.,5.]]
    write('charging_design_sensitivity.csv',scan)
    orbit=[]
    for a in [6771e3,6900e3,7371e3]:
        for inc in np.deg2rad([10.,51.,85.]):
            fs=np.linspace(0,2*np.pi,20001)
            al,bl,c=m.roll_coefficients(fs,a=a,inc=inc)
            thresholds=m.fold_threshold(al,bl,c)
            orbit.append(dict(a_m=a,inclination_deg=np.rad2deg(inc),
                min_alpha_Ckg=float(thresholds.min()),max_alpha_Ckg=float(thresholds.max()),
                reference_phase_alpha_Ckg=float(m.fold_threshold(*m.roll_coefficients(m.F_REF,a=a,inc=inc)))))
    write('charging_orbital_sensitivity.csv',orbit)
    emission=[]
    for n in [1e9,1e11,1e12]:
        for j in [0.,20e-6,40e-6,80e-6]:
            for tph in [1.,2.,3.]:
                for eta in [0.,.5,.9]:
                    q=scales(density=n)
                    U=q['vacuum_potential_V']; ie=q['thermal_electron_A']*(1+U/.2)
                    iph=j*np.pi*np.exp(-U/tph)
                    emission.append(dict(density_m3=n,Jphoto_Am2=j,Tphoto_eV=tph,
                        escaping_secondary_fraction=eta,net_OML_extrapolated_A=(1-eta)*ie-iph))
    write('charging_emission_sensitivity.csv',emission)
    ref=scales(); reduced=scales(asym=.0006)
    assert 97e3 < ref['vacuum_potential_V'] < 98e3
    assert np.isclose(reduced['vacuum_potential_V']/ref['vacuum_potential_V'],1e-3)
    assert np.isclose(scales(density=1e12)['thermal_electron_A']/ref['thermal_electron_A'],10)
    # Physical inertia triangle inequalities hold for both selected inertia triples.
    assert abs(.9994-1)<1 and abs(.4-1)<1
    actual_reduced_fold=m.fold_threshold(*m.roll_coefficients(m.F_REF,A=.9994,B=1.))
    assert np.isclose(actual_reduced_fold/AF,1e-3,rtol=3e-12)
    example=dict(A_m2=.9994,B_m2=1.,C_m2=1.,
        fold_ratio_from_governing_coefficients=float(actual_reduced_fold/AF),
        alpha_from_governing_coefficients_Ckg=float(actual_reduced_fold),
        zero_charge_circular_roll_period_ratio=float(np.sqrt(1./abs(.9994-1.))),
        charge_centre_offset_norm_m=float(np.hypot(m.X0,m.Y0)),
        geometry_status='mathematical_sensitivity_only_no_joint_mass_and_conductor_geometry_specified')
    summary=dict(reference=ref,reduced_asymmetry=reduced,sensitivity_example=example)
    (OUT/'charging_summary.json').write_text(json.dumps(summary,indent=2))
    ns=np.logspace(9,12,100)
    fig,ax=plt.subplots(1,2,figsize=(9.4,3.5))
    for d in [.6,.006,.0006]:
        vals=[scales(asym=d,density=n) for n in ns]
        label=rf'$|A-B|={d:g}$ m$^2$'
        ax[0].loglog(ns,[v['OML_extrapolated_A'] for v in vals],label=label)
        ax[1].loglog(ns,[v['OML_extrapolated_power_W'] for v in vals],label=label)
    for a in ax:
        a.set_xlabel(r'Electron density (m$^{-3}$)'); a.legend(fontsize=7);a.grid(alpha=.2)
    ax[0].set_ylabel('OML-extrapolated electron current (A)');ax[1].set_ylabel('OML-extrapolated beam power (W)')
    fig.suptitle('Spherical small-body OML extrapolation: outside the metre-scale LEO domain',fontsize=10)
    fig.tight_layout();fig.savefig(OUT/'charging_screening.pdf');plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
