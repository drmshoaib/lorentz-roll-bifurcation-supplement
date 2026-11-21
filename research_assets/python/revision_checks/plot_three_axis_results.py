"""Plot directly integrated three-axis control response; all curves numerical."""
from pathlib import Path
import matplotlib.pyplot as plt
from controlled_three_axis_validation import run

out=Path(__file__).resolve().parent
fig,axes=plt.subplots(3,1,figsize=(7.15,7.9),sharex=True,constrained_layout=True)
for ratio in [0.5,0.9]:
    row,(t,tilt,roll,torque,err)=run(ratio,perturb_deg=2.,orbits=3.,samples_per_orbit=900,
        feedforward_enabled=True,feedback_enabled=True)
    orbital_period=2*3.141592653589793*(6900e3)**1.5/(3.9860e14)**.5
    x=t/orbital_period
    axes[0].plot(x,tilt,label=rf'$\alpha^*/\alpha_F^*={ratio}$',lw=1.25)
    axes[1].plot(x,roll,label=rf'$\alpha^*/\alpha_F^*={ratio}$',lw=1.25)
    axes[2].plot(x,torque,label=rf'$\alpha^*/\alpha_F^*={ratio}$',lw=1.25)
axes[0].set_ylabel('Radial-axis misalignment (deg)')
axes[1].set_ylabel('Roll tracking error (deg)')
axes[2].set_ylabel(r'Two-axis control magnitude ($\mu$N m)')
axes[2].set_xlabel('Elapsed orbital periods')
axes[0].set_title(r'Controlled 3-axis response to $2^\circ$ / $-1^\circ$ transverse offset')
for a in axes:
    a.legend(fontsize=8,loc='upper right')
    a.grid(alpha=.2)
    a.set_xlim(0,3)
for ext in ('png','pdf'):
    fig.savefig(out/('three_axis_perturbation_three_orbits.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
print('Plotted controlled three-axis trajectories.')

