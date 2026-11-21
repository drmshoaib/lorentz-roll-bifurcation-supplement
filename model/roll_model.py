"""Reduced constrained-roll model shared by the figure scripts.

Direct transcription of the orbital, Lorentz, roll-coefficient and fold
relations used in the manuscript. All forces and torques are mass-normalised.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

MU_E = 3.9860e14
OMEGA_E = 7.2921e-5
B_D = 7.943e15

A_INERTIA = 0.4
B_INERTIA = 1.0
C_INERTIA = 1.0
A_ORB = 6900e3
ECC = 0.001
INC = np.radians(51.0)
F_REF = np.radians(60.0)
OMEGA_PERI = 0.0
X0 = 1.0
Y0 = 1.0


def orbit_state(f, a=A_ORB, e=ECC):
    p = a * (1.0 - e**2)
    r = p / (1.0 + e * np.cos(f))
    rdot = e * np.sqrt(MU_E / p) * np.sin(f)
    fdot = np.sqrt(MU_E * p) / r**2
    return p, r, rdot, fdot


def lorentz_projections(f, omega_peri=OMEGA_PERI, inc=INC, a=A_ORB, e=ECC):
    p, r, rdot, fdot = orbit_state(f, a, e)
    u = omega_peri + f
    pref = B_D * fdot / np.sqrt(MU_E * p)
    Tbar = pref * ((rdot / r) * np.cos(inc) + 2.0 * OMEGA_E * np.sin(inc)**2 * np.sin(u) * np.cos(u))
    Nbar = pref * np.sin(inc) * (2.0 * (OMEGA_E * np.cos(inc) - fdot) * np.sin(u) - (rdot / r) * np.cos(u))
    Rbar = pref * (-(fdot - OMEGA_E * np.cos(inc)) * np.cos(inc) + OMEGA_E * np.sin(inc)**2 * np.cos(u)**2)
    return Tbar, Nbar, Rbar


def roll_coefficients(f, omega_peri=OMEGA_PERI, x0=X0, y0=Y0, A=A_INERTIA, B=B_INERTIA, **kw):
    Tbar, Nbar, _ = lorentz_projections(f, omega_peri, **kw)
    aL = x0 * Nbar - y0 * Tbar
    bL = -(x0 * Tbar + y0 * Nbar)
    _, _, _, fdot = orbit_state(f, kw.get("a", A_ORB), kw.get("e", ECC))
    c = 0.5 * fdot**2 * (A - B)
    return aL, bL, c


def fold_threshold(aL, bL, c):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * abs(c) / (abs(aL)**(2.0 / 3.0) + abs(bL)**(2.0 / 3.0))**1.5


def h(phi, alpha, aL, bL, c, aM=0.0, bM=0.0):
    a = aM + alpha * aL
    b = bM + alpha * bL
    return a * np.cos(phi) + b * np.sin(phi) + c * np.sin(2.0 * phi)


def h_phi(phi, alpha, aL, bL, c, aM=0.0, bM=0.0):
    a = aM + alpha * aL
    b = bM + alpha * bL
    return -a * np.sin(phi) + b * np.cos(phi) + 2.0 * c * np.cos(2.0 * phi)


def potential(phi, alpha, aL, bL, c, aM=0.0, bM=0.0):
    a = aM + alpha * aL
    b = bM + alpha * bL
    return -a * np.sin(phi) + b * np.cos(phi) + 0.5 * c * np.cos(2.0 * phi)


def equilibria(alpha, aL, bL, c, n=4000, **kw):
    grid = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    vals = h(grid, alpha, aL, bL, c, **kw)
    roots = []
    for k in range(n):
        k2 = (k + 1) % n
        lo, hi = grid[k], grid[k] + 2.0 * np.pi / n
        if vals[k] == 0.0:
            roots.append(lo)
        elif vals[k] * vals[k2] < 0.0:
            roots.append(brentq(lambda x: h(x, alpha, aL, bL, c, **kw), lo, hi, xtol=1e-14))
    roots = np.array(roots)
    centre = h_phi(roots, alpha, aL, bL, c, **kw) < 0.0
    return roots, centre


def reference_values():
    aL, bL, c = roll_coefficients(F_REF)
    return dict(aL=aL, bL=bL, c=c, alphaF=fold_threshold(aL, bL, c))


if __name__ == "__main__":
    v = reference_values()
    T, N, R = lorentz_projections(F_REF)
    print(f"Tbar={T:.5e}  Nbar={N:.5e}  Rbar={R:.5e}")
    print(f"aL={v['aL']:.5e}  bL={v['bL']:.5e}  c={v['c']:.5e}  alphaF={v['alphaF']:.5e}")


def reference_fold():
    """Derived reference coefficients and fold derivatives, without fitted constants."""
    values=reference_values()
    al,bl,c,af=(values[k] for k in ('aL','bL','c','alphaF'))
    phi=float(np.arctan2(np.cbrt(-af*al/(2*c)),np.cbrt(-af*bl/(2*c)))%(2*np.pi))
    return values | dict(phiF=phi,h_alpha_F=al*np.cos(phi)+bl*np.sin(phi),
                         h_phiphi_F=-3*c*np.sin(2*phi))

