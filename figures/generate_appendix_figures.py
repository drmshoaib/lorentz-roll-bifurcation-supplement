from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import fitz

out = Path(__file__).resolve().parent / "generated"
out.mkdir(exist_ok=True)

def pdf_first_page_to_png(pdf_path: Path, png_path: Path, scale: float = 2.8):
    doc = fitz.open(pdf_path)
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    pix.save(png_path)
    doc.close()

def add_panel_label(img_path: Path, label: str):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    except:
        font = ImageFont.load_default()
    draw.text((18, 14), label, fill="black", font=font)
    img.save(img_path)

# B1: actual non-autonomous constrained-roll integration, not illustrative curves.
from scipy.integrate import solve_ivp
from model import roll_model as model

alphaF0 = model.reference_values()["alphaF"]
f0 = model.F_REF
Torb = 2 * np.pi * np.sqrt(model.A_ORB**3 / model.MU_E)
a0, b0, c0 = model.roll_coefficients(f0)

def initial_centre(alpha):
    angles, is_centre = model.equilibria(alpha, a0, b0, c0, n=10000)
    centres = angles[is_centre]
    reference_fold_angle = np.deg2rad(134.4798663)
    distance = np.abs(np.angle(np.exp(1j * (centres-reference_fold_angle))))
    return float(centres[np.argmin(distance)])

def nonautonomous_rhs(t, y, alpha):
    f, phi, rate = y
    _, _, _, fdot = model.orbit_state(f)
    aL, bL, c = model.roll_coefficients(f)
    return [fdot, rate, model.h(phi, alpha, aL, bL, c) / model.C_INERTIA]

fractions = [0.25, 0.50, 0.90]
sample_time = np.linspace(0.0, Torb, 1801)
trajectories = {}
for ratio in fractions:
    alpha = ratio * alphaF0
    phi0 = initial_centre(alpha)
    sol = solve_ivp(
        lambda t, y: nonautonomous_rhs(t, y, alpha),
        [0.0, Torb], [f0, phi0, 0.0],
        t_eval=sample_time, method="DOP853",
        rtol=1e-10, atol=1e-11,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    trajectories[ratio] = sol

angle_grid = np.linspace(0, 2*np.pi, 1200)
thresholds = []
for fval in angle_grid:
    al, bl, c = model.roll_coefficients(fval)
    thresholds.append(model.fold_threshold(al, bl, c) / alphaF0)
thresholds = np.asarray(thresholds)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.7))
ax1.plot(np.rad2deg(angle_grid), thresholds, label="instantaneous fold")
for ratio in fractions:
    ax1.axhline(ratio, linestyle="--", linewidth=0.8,
                label=f"charge ratio {ratio:g}")
ax1.set_yscale("log")
ax1.set_ylim(0.15, max(10., min(np.nanmax(thresholds), 10000.)))
ax1.set_xlim(0, 360)
ax1.set_xlabel("true anomaly f (degrees)")
ax1.set_ylabel("instantaneous fold charge / reference fold charge")
ax1.set_title("(a) Frozen-orbit threshold along the orbit")
ax1.legend(fontsize=6)

for ratio, sol in trajectories.items():
    relative_roll = np.rad2deg(sol.y[1] - sol.y[1, 0])
    ax2.plot(sol.t / Torb, relative_roll, label=f"charge ratio {ratio:g}")
    if abs(ratio - 0.90) < 1e-12:
        crossing = np.zeros_like(sol.t, dtype=bool)
        for j, fval in enumerate(sol.y[0]):
            al, bl, c = model.roll_coefficients(fval)
            crossing[j] = ratio * alphaF0 > model.fold_threshold(al, bl, c)
        is_on = np.diff(np.r_[False, crossing, False].astype(int))
        starts = np.where(is_on == 1)[0]
        stops = np.where(is_on == -1)[0]
        for first, last in zip(starts, stops):
            ax2.axvspan(sol.t[first] / Torb, sol.t[last-1] / Torb,
                        color="grey", alpha=0.15)
ax2.set_xlabel("time / orbital period")
ax2.set_ylabel("roll angle change from initial value (degrees)")
ax2.set_title("(b) Integrated constrained-roll response")
ax2.legend(fontsize=7)
fig.tight_layout()
fig.savefig(out / "figure_B1_time_dependent.pdf", bbox_inches="tight")
fig.savefig(out / "figure_B1_time_dependent.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# C1

alpha_ratio = np.array([0.25, 0.50, 0.90, 0.99])
transverse_norm = alpha_ratio * alphaF0 * np.hypot(model.X0, model.Y0) * abs(model.lorentz_projections(model.F_REF)[2]) / (2 * abs(c0))
aL, bL, c = a0, b0, c0
alphaF = alphaF0
lorentz_amp_norm = alpha_ratio * alphaF * np.sqrt(aL*aL + bL*bL) / (2*abs(c))
fig = plt.figure(figsize=(4.2, 3.0))
ax = fig.add_axes([0.14, 0.18, 0.82, 0.74])
x = np.arange(len(alpha_ratio))
width = 0.36
ax.bar(x - width/2, transverse_norm, width, label=r"omitted transverse $|\mathbf{T}_\perp|/(2|c|)$")
ax.bar(x + width/2, lorentz_amp_norm, width, label=r"retained roll $|T_{L,3}|_{\max}/(2|c|)$")
ax.axhline(1.0, linestyle="--", linewidth=1.0, label=r"inertial scale $2|c|$")
ax.set_xticks(x)
ax.set_xticklabels([f"{v:.2f}" for v in alpha_ratio], fontsize=7)
ax.set_xlabel(r"$\alpha^*/\alpha_F^*$", fontsize=8)
ax.set_ylabel(r"normalized torque magnitude", fontsize=8)
ax.set_title("Transverse Lorentz torque omitted by the constrained roll model", fontsize=9)
ax.legend(fontsize=5.9, loc="upper left")
ax.tick_params(labelsize=7)
fig.savefig(out / "figure_C1_transverse_torque.pdf", bbox_inches="tight")
plt.close(fig)

# D1 is generated exclusively by generate_igrf_spectrum.py from IGRF-14 data.

# E1
eps0 = 8.854187817e-12
alphaF = alphaF0
m = np.linspace(1.0, 200.0, 400)
radii = [0.5, 1.0, 2.0]
fig = plt.figure(figsize=(4.2, 3.0))
ax = fig.add_axes([0.14, 0.18, 0.82, 0.74])
for R in radii:
    V = alphaF*m/(4*np.pi*eps0*R)
    ax.plot(m, V/1000.0, linewidth=1.3, label=rf"$R_{{\rm eff}}={R:g}\,$m")
ax.axhline(1.0, linestyle="--", linewidth=1.0, label=r"$1\,$kV")
ax.axhline(10.0, linestyle=":", linewidth=1.0, label=r"$10\,$kV")
ax.set_xlabel(r"spacecraft mass $m$ (kg)", fontsize=8)
ax.set_ylabel(r"$V_{\rm sph}$ (kV)", fontsize=8)
ax.set_title("Electrostatic scale implied by the reference fold charge", fontsize=9)
ax.legend(fontsize=6.2, loc="upper left")
ax.tick_params(labelsize=7)
fig.savefig(out / "figure_E1_electrostatic_scale.pdf", bbox_inches="tight")
plt.close(fig)

print("Appendix figures generated.")

