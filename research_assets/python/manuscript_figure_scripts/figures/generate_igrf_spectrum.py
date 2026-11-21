from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "igrf14_2025.txt"
OUT = HERE.parent / "figures_out"
OUT.mkdir(exist_ok=True)

A_E = 6371.2e3
R = 6900e3

coeff = {}
with DATA.open() as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kind, n, m, value = line.split()
        coeff[(kind, int(n), int(m))] = float(value)

def degree_power(n):
    s = 0.0
    for m in range(n + 1):
        g = coeff.get(("g", n, m), 0.0)
        h = coeff.get(("h", n, m), 0.0)
        s += g*g + h*h
    return (n + 1) * (A_E / R)**(2*n + 4) * s

ns = np.arange(1, 14)
R_n = np.array([degree_power(int(n)) for n in ns])
rms = np.sqrt(R_n / R_n[0])
combined = np.sqrt(R_n[1:].sum() / R_n[0])

g10 = coeff[("g", 1, 0)]
g11 = coeff[("g", 1, 1)]
h11 = coeff[("h", 1, 1)]
tilt = np.degrees(np.arctan2(np.hypot(g11, h11), abs(g10)))

# degree-1 dipole constant from the same coefficients; nT -> T
R1_surface = 2.0 * sum(
    coeff.get(("g", 1, m), 0.0)**2 + coeff.get(("h", 1, m), 0.0)**2
    for m in range(2)
)
Bdip = np.sqrt(R1_surface / 2.0) * 1e-9 * A_E**3

print(f"tilt_deg={tilt:.4f}")
print(f"sqrt(R2/R1)={rms[1]:.6f}")
print(f"sqrt(R3/R1)={rms[2]:.6f}")
print(f"quadrature_deg2_13={combined:.6f}")
print(f"degree1_dipole_constant={Bdip:.6e} T m^3")

fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.bar(ns, rms)
ax.set_yscale("log")
ax.set_xlabel("spherical-harmonic degree $n$")
ax.set_ylabel(r"rms amplitude $\sqrt{R_n/R_1}$")
ax.set_xticks(ns)
ax.set_xlim(0.4, 13.6)
fig.tight_layout()
fig.savefig(OUT / "figure_D1_igrf_spectrum.pdf", bbox_inches="tight")
fig.savefig(OUT / "figure_D1_igrf_spectrum.png", dpi=200, bbox_inches="tight")

