"""
Reproduces figure22_robustness_checks.pdf (Section 8): intrinsic
geomagnetic-torque robustness of the fold charge (panel a) and barrier-
scaling robustness across eight orbital orientations (panel b), both
derived from first principles using the orbital/Lorentz/intrinsic-moment
model of Section 2.
"""
from pathlib import Path
import roll_model as reference_model
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

out = Path(__file__).resolve().parent.parent / "figures_out"
out.mkdir(exist_ok=True)

MU_E = reference_model.MU_E
OMEGA_E = reference_model.OMEGA_E
B_D = 7.943e15
A_INERTIA = 0.4
B_INERTIA = 1.0
C = 1.0
A0 = 6900e3
ECC = 0.001
INC = np.deg2rad(51.0)
F_REF = np.deg2rad(60.0)
X0 = 1.0
Y0 = 1.0
P = A0 * (1 - ECC ** 2)


def orbital_quantities(f):
    r = P / (1 + ECC * np.cos(f))
    rdot = ECC * np.sqrt(MU_E / P) * np.sin(f)
    fdot = np.sqrt(MU_E * P) / r ** 2
    return r, rdot, fdot


def tbar_nbar(f, omega_star):
    r, rdot, fdot = orbital_quantities(f)
    u = omega_star + f
    tbar = B_D * fdot / np.sqrt(MU_E * P) * (
        (rdot / r) * np.cos(INC) + 2 * OMEGA_E * np.sin(INC) ** 2 * np.sin(u) * np.cos(u)
    )
    nbar = B_D * fdot * np.sin(INC) / np.sqrt(MU_E * P) * (
        2 * (OMEGA_E * np.cos(INC) - fdot) * np.sin(u) - (rdot / r) * np.cos(u)
    )
    return tbar, nbar


def lorentz_coefficients(f, omega_star, x0=X0, y0=Y0):
    t, n = tbar_nbar(f, omega_star)
    return x0 * n - y0 * t, -(x0 * t + y0 * n), t, n


def c_coeff(f, A_=A_INERTIA, B_=B_INERTIA):
    return 0.5 * orbital_quantities(f)[2] ** 2 * (A_ - B_)


def alpha_fold(aL, bL, c):
    return 2 * abs(c) / (abs(aL) ** (2 / 3) + abs(bL) ** (2 / 3)) ** (3 / 2)


def cbrt(x):
    return np.sign(x) * np.abs(x) ** (1 / 3)


def fold_angle(aL, bL, c, alphaF):
    a = alphaF * aL
    b = alphaF * bL
    s = cbrt(-a / (2 * c))
    co = cbrt(-b / (2 * c))
    norm = np.hypot(s, co)
    return np.arctan2(s / norm, co / norm)


def h(phi, alpha, aM, bM, aL, bL, c):
    a = aM + alpha * aL
    b = bM + alpha * bL
    return a * np.cos(phi) + b * np.sin(phi) + c * np.sin(2 * phi)


def hphi(phi, alpha, aM, bM, aL, bL, c):
    a = aM + alpha * aL
    b = bM + alpha * bL
    return -a * np.sin(phi) + b * np.cos(phi) + 2 * c * np.cos(2 * phi)


def V(phi, alpha, aL, bL, c):
    return -alpha * aL * np.sin(phi) + alpha * bL * np.cos(phi) + 0.5 * c * np.cos(2 * phi)


def two_nearest_roots(alpha, phiF, r_nf, aL, bL, c, ngrid=4000, span=6.0):
    lo, hi = phiF - span * r_nf, phiF + span * r_nf
    xs = np.linspace(lo, hi, ngrid + 1)
    ys = h(xs, alpha, 0.0, 0.0, aL, bL, c)
    roots = []
    for i in range(ngrid):
        if ys[i] == 0:
            roots.append(xs[i])
        elif ys[i] * ys[i + 1] < 0:
            roots.append(
                brentq(lambda p: h(p, alpha, 0.0, 0.0, aL, bL, c), xs[i], xs[i + 1],
                       xtol=1e-15, rtol=1e-14)
            )
    if len(roots) < 2:
        return None
    roots = sorted(roots, key=lambda r: abs(r - phiF))
    r1, r2 = sorted(roots[:2])
    return r1, r2


def newton_fold(phi0, alpha0, aM, bM, aL, bL, c, iters=100):
    phi, alpha = phi0, alpha0
    for _ in range(iters):
        H = h(phi, alpha, aM, bM, aL, bL, c)
        Hp = hphi(phi, alpha, aM, bM, aL, bL, c)
        dHda = aL * np.cos(phi) + bL * np.sin(phi)
        dHpp = (-(aM + alpha * aL) * np.cos(phi) - (bM + alpha * bL) * np.sin(phi)
                - 4 * c * np.sin(2 * phi))
        dHpda = -aL * np.sin(phi) + bL * np.cos(phi)
        J = np.array([[Hp, dHda], [dHpp, dHpda]])
        F = np.array([H, Hp])
        step = np.linalg.solve(J, -F)
        phi += step[0]
        alpha += step[1]
        if np.hypot(*F) < 1e-20:
            break
    return phi, alpha


aL0, bL0, _, _ = lorentz_coefficients(F_REF, 0.0)
c0 = c_coeff(F_REF)
alphaF0 = alpha_fold(aL0, bL0, c0)
phiF0_pos = fold_angle(aL0, bL0, c0, alphaF0)
phiF0_neg = fold_angle(aL0, bL0, c0, -alphaF0)

r_ref = orbital_quantities(F_REF)[0]
u_ref = F_REF
BT = -(B_D / r_ref ** 3) * np.sin(INC) * np.cos(u_ref)
BN = -(B_D / r_ref ** 3) * np.cos(INC)
Bdip_r3 = B_D / r_ref ** 3
mhat = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)

eta_grid = np.array([0.0, 0.1, 0.25, 0.5])
eta_fine = np.linspace(0.0, 0.5, 201)

phi_p, a_p = phiF0_pos, alphaF0
phi_m, a_m = phiF0_neg, -alphaF0
track_p, track_m = [], []
for eta_val in eta_fine:
    M = eta_val * 2 * abs(c0) / Bdip_r3
    Mx, My = M * mhat[0], M * mhat[1]
    aM = Mx * BN - My * BT
    bM = -(Mx * BT + My * BN)
    phi_p, a_p = newton_fold(phi_p, a_p, aM, bM, aL0, bL0, c0)
    phi_m, a_m = newton_fold(phi_m, a_m, aM, bM, aL0, bL0, c0)
    track_p.append(a_p)
    track_m.append(a_m)

alpha_plus = np.interp(eta_grid, eta_fine, track_p)
alpha_minus = np.interp(eta_grid, eta_fine, track_m)

omega_list_deg = [0, 30, 60, 90, 150, 210, 240, 330]
ratio_grid = np.logspace(-4.0, -2.0, 9)

barrier_by_omega = {}
for omega_deg in omega_list_deg:
    omega = np.deg2rad(omega_deg)
    aL, bL, _, _ = lorentz_coefficients(F_REF, omega)
    c = c_coeff(F_REF)
    alphaF = alpha_fold(aL, bL, c)
    phiF = fold_angle(aL, bL, c, alphaF)

    h_alpha_F = aL * np.cos(phiF) + bL * np.sin(phiF)
    h_phiphi_F = -3 * c * np.sin(2 * phiF)
    L = np.sqrt(2 * abs(h_alpha_F) / abs(h_phiphi_F))

    deltas = ratio_grid * alphaF
    barrier = []
    for delta in deltas:
        alpha = alphaF - delta
        r_nf = L * np.sqrt(delta)
        p1, p2 = two_nearest_roots(alpha, phiF, r_nf, aL, bL, c)
        barrier.append(abs(V(p1, alpha, aL, bL, c) - V(p2, alpha, aL, bL, c)))
    barrier_by_omega[omega_deg] = (deltas / alphaF, np.asarray(barrier))

fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.1))

axs[0].plot(eta_grid, alpha_plus, marker="o", linewidth=1.15, label=r"$\alpha_F^{*+}$")
axs[0].plot(eta_grid, np.abs(alpha_minus), marker="o", linewidth=1.15, label=r"$|\alpha_F^{*-}|$")
axs[0].set_xlabel(r"Intrinsic magnetic strength $\eta$")
axs[0].set_ylabel("Fold charge magnitude")
axs[0].set_title("(a) Intrinsic geomagnetic-torque robustness")
axs[0].legend()

cmap = plt.get_cmap("viridis")
for i, omega_deg in enumerate(omega_list_deg):
    x, y = barrier_by_omega[omega_deg]
    axs[1].loglog(x, y, marker="o", markersize=3, linewidth=0.9,
                  color=cmap(i / (len(omega_list_deg) - 1)),
                  label=rf"$\omega^*={omega_deg}^\circ$")
x0_, y0_ = barrier_by_omega[0]
ref = y0_[0] * (x0_ / x0_[0]) ** 1.5
axs[1].loglog(x0_, ref, linestyle="--", color="black", linewidth=1.3, label="slope 3/2")
axs[1].set_xlabel(r"$\delta/\alpha_F^*$")
axs[1].set_ylabel(r"$\Delta V$")
axs[1].set_title("(b) Barrier-scaling robustness across orbital orientation")
axs[1].legend(fontsize=6.3, ncol=2)

fig.tight_layout()
fig.savefig(out / "figure22_robustness_checks.pdf", bbox_inches="tight")
plt.close(fig)

