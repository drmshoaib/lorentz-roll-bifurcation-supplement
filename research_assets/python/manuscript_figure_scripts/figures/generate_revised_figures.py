"""Figures for the revised (shortened) manuscript.

All nine figures are drawn at the manuscript text width (390 pt, elsarticle
preprint 12 pt) in Computer Modern, so that they are included at their natural
size and their lettering matches the text. Colours follow one scheme throughout:
roll centres are blue circles, saddles orange crosses, and charge levels a
single-hue ramp from light (far from the fold) to dark (close to it).

Outputs (figures_out/): figure_*_revised.pdf, and
orbital_matched_poincare_revised.pdf, which is drawn from the data written by
research_assets/python/revision_checks/orbital_poincare.py.
The original manuscript's figures and scripts are not affected.
"""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import quad
from scipy.optimize import brentq

import roll_model as m

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures_out"
OUT.mkdir(exist_ok=True)
ORBITAL = HERE.parents[1] / "revision_checks" / "results"

# ---------------------------------------------------------------- style
TEXTWIDTH = 390.0 / 72.27          # inches
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "lines.linewidth": 1.0, "legend.frameon": False, "legend.handlelength": 1.6,
    "pdf.fonttype": 42, "contour.negative_linestyle": "solid", "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
INK, GREY, LIGHT = "#0b0b0b", "#6b6a66", "#c9c8c2"
BLUE, ORANGE = "#2a78d6", "#eb6834"
WELL_FILL = "#dbe9fb"
# Single-hue ordinal ramp for the charge level (validated: monotone, one hue).
RAMP = {0.5: "#6da7ec", 0.9: "#2a78d6", 0.99: "#1c5cab", 0.999: "#0d366b"}
CENTRE = dict(marker="o", color=BLUE, ms=3.6, ls="none", zorder=6)
SADDLE = dict(marker="x", color=ORANGE, ms=4.2, mew=1.1, ls="none", zorder=6)


def tag(ax, text, x=0.0, y=1.02, **kw):
    """Panel label above the top-left corner, clear of the data."""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, **kw)


def panels(ncols, height, **kw):
    """One row of panels filling the text width (constrained layout)."""
    return plt.subplots(1, ncols, figsize=(TEXTWIDTH, height), layout="constrained", **kw)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- model
REF = m.reference_fold()
aL, bL, c, AF, PHIF = (REF[k] for k in ("aL", "bL", "c", "alphaF", "phiF"))
C = m.C_INERTIA
A_FOLD, B_FOLD = abs(REF["h_alpha_F"]), abs(REF["h_phiphi_F"])
L = np.sqrt(2 * A_FOLD / B_FOLD)
T0 = np.sqrt(C * L / A_FOLD)
SIGMA = np.sign(REF["h_alpha_F"])
T_ORB = 2 * np.pi * np.sqrt(m.A_ORB**3 / m.MU_E)


def h(phi, al):
    return m.h(phi, al, aL, bL, c)


def hp(phi, al):
    return m.h_phi(phi, al, aL, bL, c)


def V(phi, al):
    return m.potential(phi, al, aL, bL, c)


def roots(al, n=20000):
    r, centre = m.equilibria(al, aL, bL, c, n=n)
    order = np.argsort(r)
    return r[order] % (2 * np.pi), centre[order]


def fold_well(al):
    """Centre that collides at the positive fold, with its two neighbouring saddles."""
    r, iscentre = roots(al)
    cs = r[iscentre]
    phic = cs[np.argmin(abs(np.angle(np.exp(1j * (cs - PHIF)))))]
    phic = PHIF + np.angle(np.exp(1j * (phic - PHIF)))
    sad = r[~iscentre]
    ext = np.concatenate([sad - 2 * np.pi, sad, sad + 2 * np.pi])
    left, right = ext[ext < phic].max(), ext[ext > phic].min()
    esc, other = (left, right) if V(left, al) <= V(right, al) else (right, left)
    return dict(c=phic, left=left, right=right, esc=esc, other=other,
                Vc=V(phic, al), Vs=V(esc, al))


def turning(al, E, w):
    f = lambda x: V(x, al) - E
    return brentq(f, w["left"], w["c"], xtol=1e-15), brentq(f, w["c"], w["right"], xtol=1e-15)


def period(al, E, w):
    lo, hi = turning(al, E, w)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    g = lambda t: half * np.cos(t) / np.sqrt(max(E - V(mid + half * np.sin(t), al), 1e-300))
    return np.sqrt(2 * C) * quad(g, -np.pi / 2, np.pi / 2, limit=500, epsabs=0, epsrel=1e-11)[0]


def canonical_turning(H):
    r = np.roots([1.0, 0.0, -3.0, 3.0 * H])
    r = np.sort(np.real(r[abs(np.imag(r)) < 1e-9]))
    return r[0], r[1]


def canonical_period(eps):
    H = -2 / 3 + 4 * eps / 3
    lo, hi = canonical_turning(H)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    g = lambda t: half * np.cos(t) / np.sqrt(max(H - (mid + half * np.sin(t)) + (mid + half * np.sin(t))**3 / 3, 1e-300))
    return np.sqrt(2) * quad(g, -np.pi / 2, np.pi / 2, limit=500, epsabs=0, epsrel=1e-12)[0]


# ================================================================ Figure 1
def fig_multiplicity():
    fig, ax = panels(3, 2.05, gridspec_kw=dict(width_ratios=[1, 1.15, 1.15]))
    th = np.linspace(0, 2 * np.pi, 1200)
    a = ax[0]
    a.fill(np.sin(th)**3, np.cos(th)**3, color=WELL_FILL, lw=0)
    a.plot(np.sin(th)**3, np.cos(th)**3, color=INK, lw=1.0)
    a.plot([0], [0], "o", color=BLUE, ms=4)
    a.plot([1.3], [0], "s", color=ORANGE, ms=4)
    tf = np.pi / 4 + np.pi
    a.plot([np.sin(tf)**3], [np.cos(tf)**3], "^", color=INK, ms=4.5)
    a.plot([0], [-1], "D", color=INK, ms=3.8, mfc="white", mew=1.0)
    a.text(0.0, 0.2, "4 eq.", ha="center", fontsize=6.5)
    a.text(0.78, 0.78, "2 eq.", ha="center", fontsize=7)
    a.set_xlim(-1.5, 1.5); a.set_ylim(-1.5, 1.5); a.set_aspect("equal")
    a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
    a.set_xlabel(r"$A_c=a/2|c|$"); a.set_ylabel(r"$B_c=b/2|c|$", labelpad=1)
    tag(a, "(a)")

    phi = np.linspace(0, 2 * np.pi, 800)
    b = ax[1]
    b.axhline(0, color=LIGHT, lw=0.6)
    hin = np.sin(2 * phi)
    hout = 2.6 * np.cos(phi) + np.sin(2 * phi)
    b.plot(phi, hin, color=BLUE)
    b.plot(phi, hout, color=ORANGE)
    b.plot(np.arange(4) * np.pi / 2, np.zeros(4), "o", color=BLUE, ms=3.2, mfc="white", mew=0.9)
    b.plot([np.pi / 2, 3 * np.pi / 2], [0, 0], "s", color=ORANGE, ms=2.6)
    b.set_xticks(np.arange(5) * np.pi / 2, ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    b.set_xlim(0, 2 * np.pi); b.set_ylim(-3.6, 3.6)
    b.set_xlabel(r"$\phi$"); b.set_ylabel(r"$h/c$", labelpad=1)
    tag(b, "(b)")

    x = np.linspace(-0.8, 0.8, 400)
    tf = np.pi / 4
    af, bf = -2 * np.sin(tf)**3, -2 * np.cos(tf)**3
    hfold = af * np.cos(tf + x) + bf * np.sin(tf + x) + np.sin(2 * (tf + x))
    hcusp = -2 * np.sin(x) + np.sin(2 * x)
    d = ax[2]
    d.axhline(0, color=LIGHT, lw=0.6)
    d.plot(x, hfold, color=INK)
    d.plot(x, hcusp, color=INK, ls="--")
    d.plot([0], [0], "o", color=INK, ms=2.8)
    d.set_xlim(-0.8, 0.8); d.set_ylim(-1.5, 0.95)
    d.set_xlabel(r"$\phi-\phi_0$"); d.set_ylabel(r"$h/c$", labelpad=1)
    tag(d, "(c)")
    handles = [Line2D([], [], color=BLUE, marker="o", ms=3.5, label=r"interior, $A_c=B_c=0$"),
               Line2D([], [], color=ORANGE, marker="s", ms=3.2, label=r"exterior, $A_c=1.3,\ B_c=0$"),
               Line2D([], [], color=INK, marker="^", ms=3.8, label="regular fold (quadratic contact)"),
               Line2D([], [], color=INK, ls="--", marker="D", ms=3.2, mfc="white", label="cusp (cubic contact)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=6.5, columnspacing=1.6)
    save(fig, "figure_equilibrium_multiplicity_cusps_revised.pdf")


# ================================================================ Figure 2
def fig_manifold():
    s = 1e7
    fig = plt.figure(figsize=(TEXTWIDTH * 1.06, 2.6))
    a = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    cmax = 1.35 * abs(c)
    cc = np.linspace(-cmax, cmax, 41)
    th = np.linspace(0, 2 * np.pi, 161)
    CC, TT = np.meshgrid(cc, th)
    Aa, Bb = -2 * CC * np.sin(TT)**3, -2 * CC * np.cos(TT)**3
    a.plot_surface(Aa * s, Bb * s, CC * s, color=BLUE, alpha=0.16, linewidth=0, antialiased=True, shade=False)
    for cl in np.linspace(-cmax, cmax, 7):
        if abs(cl) < 1e-12:
            continue
        a.plot(-2 * cl * np.sin(th)**3 * s, -2 * cl * np.cos(th)**3 * s, cl * s, color=BLUE, lw=0.35, alpha=0.7)
    a.plot(-2 * c * np.sin(th)**3 * s, -2 * c * np.cos(th)**3 * s, c * s, color=INK, lw=1.1)
    q = np.linspace(-1.45, 1.45, 2)
    a.plot(q * AF * aL * s, q * AF * bL * s, [c * s] * 2, color=ORANGE, lw=1.1)
    a.plot([AF * aL * s, -AF * aL * s], [AF * bL * s, -AF * bL * s], [c * s] * 2, "o", color=ORANGE, ms=3)
    a.view_init(elev=17, azim=-58)
    for axis in (a.xaxis, a.yaxis, a.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0)); axis.pane.set_edgecolor(LIGHT)
        axis._axinfo["grid"].update(color=(0.85, 0.85, 0.83, 1), linewidth=0.4)
    a.set_xticks([-8, 0, 8]); a.set_yticks([-8, 0, 8]); a.set_zticks([-4, 0, 4])
    a.tick_params(labelsize=6, pad=0)
    a.set_xlabel(r"$a$", labelpad=-6); a.set_ylabel(r"$b$", labelpad=-6); a.set_zlabel(r"$c$", labelpad=-7)
    a.set_box_aspect((1, 1, 0.9))
    a.text2D(0.0, 1.0, "(a)", transform=a.transAxes, fontsize=8, va="bottom")

    b = fig.add_subplot(1, 2, 2)
    th = np.linspace(0, 2 * np.pi, 1200)
    b.fill(-2 * c * np.sin(th)**3 * s, -2 * c * np.cos(th)**3 * s, color=WELL_FILL, lw=0)
    b.plot(-2 * c * np.sin(th)**3 * s, -2 * c * np.cos(th)**3 * s, color=INK, lw=1.0, label="fold set (astroid)")
    q = np.linspace(-1.45, 1.45, 2)
    b.plot(q * AF * aL * s, q * AF * bL * s, color=ORANGE, lw=1.1, label=r"physical line, varying $\alpha^*$")
    b.annotate("", xy=(0.95 * AF * aL * s * 1.4, 0.95 * AF * bL * s * 1.4), xytext=(1.15 * AF * aL * s, 1.15 * AF * bL * s),
               arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.0, mutation_scale=7))
    for sign, lab, off in ((1, r"$F_+$", (-1.3, -1.0)), (-1, r"$F_-$", (1.25, 0.35))):
        x0, y0 = sign * AF * aL * s, sign * AF * bL * s
        b.plot(x0, y0, "o", color=ORANGE, ms=4, mec=INK, mew=0.5, zorder=5)
        b.text(x0 + off[0], y0 + off[1], lab, fontsize=7.5, ha="center")
    b.plot(0, 0, "o", color=INK, ms=2.2)
    b.text(0.4, 0.45, r"$\alpha^*=0$", fontsize=6.5)
    for x0, y0 in ((0, 2), (0, -2), (2, 0), (-2, 0)):
        b.plot(x0 * abs(c) * s, y0 * abs(c) * s, "D", color=INK, ms=3, mfc="white", mew=0.8)
    b.text(-2.0, -1.45, "4 equilibria", fontsize=6.5, ha="center")
    b.text(5.6, 5.4, "2 equilibria", fontsize=6.5, ha="center")
    b.set_xlim(-9, 9); b.set_ylim(-9, 9); b.set_aspect("equal")
    b.set_xlabel(r"$a\ (10^{-7}\,\mathrm{m^2\,s^{-2}})$"); b.set_ylabel(r"$b\ (10^{-7}\,\mathrm{m^2\,s^{-2}})$", labelpad=0)
    b.legend(loc="lower left", fontsize=6)
    tag(b, "(b)")
    fig.subplots_adjust(left=0.0, right=1.0, wspace=0.42)
    save(fig, "figure_fold_manifold_revised.pdf")


# ================================================================ Figure 3
def fig_branches():
    fig, ax = panels(2, 2.3)
    qs = np.linspace(-1.25, 1.25, 1601)
    pc, ps = [], []
    for q in qs:
        r, isc = roots(q * AF, n=6000)
        pc += [(q, v) for v in np.degrees(r[isc])]
        ps += [(q, v) for v in np.degrees(r[~isc])]
    pc, ps = np.array(pc), np.array(ps)
    a = ax[0]
    for xv in (-1, 1):
        a.axvline(xv, color=LIGHT, lw=0.6, zorder=0)
    a.plot(pc[:, 0], pc[:, 1], ".", color=BLUE, ms=1.1, label="centres")
    a.plot(ps[:, 0], ps[:, 1], ".", color=ORANGE, ms=1.1, label="saddles")
    pf = np.degrees(PHIF)
    a.plot([1, -1], [pf, pf - 180], "o", color=INK, ms=3.2, mfc="white", mew=0.9, zorder=7)
    a.text(1.05, pf + 6, r"$F_+$", fontsize=7.5); a.text(-1.2, pf - 180 - 22, r"$F_-$", fontsize=7.5)
    a.set_xlim(-1.25, 1.25); a.set_ylim(0, 360); a.set_yticks([0, 90, 180, 270, 360])
    a.set_xlabel(r"$\alpha^*/\alpha_F^*$"); a.set_ylabel(r"$\phi_e$ (deg)")
    a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.97), markerscale=5, fontsize=6.5, handletextpad=0.1)
    tag(a, "(a)")

    b = ax[1]
    qz = np.linspace(0.90, 1.0, 400)
    xc, xs = [], []
    for q in qz[:-1]:
        w = fold_well(q * AF)
        xc.append(np.degrees(w["c"])); xs.append(np.degrees(w["esc"]))
    lead = np.degrees(np.sqrt(2 * (1 - qz) / 3))
    b.plot(qz[:-1], np.array(xc) % 360, color=BLUE, lw=1.6, label="centre")
    b.plot(qz[:-1], np.array(xs) % 360, color=ORANGE, lw=1.6, label="saddle")
    b.plot(qz, pf + lead, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=5)
    b.plot(qz, pf - lead, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=5, label=r"$\phi_F\pm L\delta^{1/2}$ (leading order)")
    b.plot([1], [pf], "o", color=INK, ms=3.2, mfc="white", mew=0.9, zorder=7)
    b.text(0.9955, pf + 1.5, r"$F_+$", fontsize=7.5, ha="right")
    b.set_xlim(0.90, 1.004); b.set_xlabel(r"$\alpha^*/\alpha_F^*$"); b.set_ylabel(r"$\phi_e$ (deg)")
    b.legend(loc="center left", fontsize=6.5)
    tag(b, "(b)")
    save(fig, "figure_stability_branches_revised.pdf")


# ================================================================ Figure 4
def fig_portraits():
    ratios = [0.0, 0.5, 0.9, 0.95, 0.99, 1.05]
    fig, axs = plt.subplots(2, 3, figsize=(TEXTWIDTH, 3.25), sharex=True, sharey=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
    lo, hi = np.radians(-100), np.radians(290)
    ph = np.linspace(lo, hi, 900)
    rate = np.linspace(-1.55e-3, 1.55e-3, 500)
    PH, RT = np.meshgrid(ph, rate)
    for ax, q in zip(axs.ravel(), ratios):
        al = q * AF
        E = 0.5 * C * RT**2 + V(PH, al)
        r, isc = roots(al)
        rr = np.concatenate([r - 2 * np.pi, r, r + 2 * np.pi]); ii = np.concatenate([isc] * 3)
        keep = (rr >= lo) & (rr <= hi)
        Vsad = sorted(V(x, al) for x in rr[keep & ~ii])
        vmin = V(rr[keep & ii], al).min()
        grey = list(vmin + (max(Vsad) - vmin) * np.array([0.2, 0.45, 0.7]))
        grey += [max(Vsad) + (max(Vsad) - vmin) * f for f in (0.35, 0.9)]
        ax.contour(np.degrees(PH), RT * 1e3, E, levels=sorted(grey), colors=LIGHT, linewidths=0.55)
        ax.contour(np.degrees(PH), RT * 1e3, E, levels=sorted(set(np.round(Vsad, 20))), colors=GREY, linewidths=0.8)
        if q < 1:
            w = fold_well(al)
            sh = 0.0
            while w["c"] + sh > hi:
                sh -= 2 * np.pi
            while w["c"] + sh < lo:
                sh += 2 * np.pi
            l_, r_ = turning(al, w["Vs"] - 1e-15 * abs(w["Vs"]), w)
            inside = (PH >= l_ + sh) & (PH <= r_ + sh) & (E <= w["Vs"])
            ax.contourf(np.degrees(PH), RT * 1e3, np.where(inside, 0.0, 1.0), levels=[-0.5, 0.5],
                        colors=[WELL_FILL], zorder=1)
            lev = w["Vc"] + (w["Vs"] - w["Vc"]) * np.array([0.25, 0.5, 0.75])
            mask = (PH >= l_ + sh - 0.02) & (PH <= r_ + sh + 0.02)
            ax.contour(np.degrees(PH), RT * 1e3, np.where(mask, E, np.nan), levels=lev,
                       colors=BLUE, linewidths=0.7, zorder=3)
            ax.contour(np.degrees(PH), RT * 1e3, np.where(mask, E, np.nan), levels=[w["Vs"]],
                       colors=BLUE, linewidths=1.3, zorder=4)
        ax.plot(np.degrees(rr[keep & ii]), np.zeros((keep & ii).sum()), **CENTRE)
        ax.plot(np.degrees(rr[keep & ~ii]), np.zeros((keep & ~ii).sum()), **SADDLE)
        ax.set_xlim(np.degrees(lo), np.degrees(hi)); ax.set_ylim(-1.55, 1.55)
        ax.set_xticks([-90, 0, 90, 180, 270])
        ax.text(0.97, 0.95, rf"$\alpha^*/\alpha_F^*={q:g}$", transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85), zorder=8)
    for ax in axs[1]:
        ax.set_xlabel(r"$\phi$ (deg)")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\dot\phi\ (10^{-3}\,\mathrm{rad\,s^{-1}})$")
    save(fig, "figure_phase_portrait_sequence_revised.pdf")


# ================================================================ Figure 5
def fig_period():
    fig, ax = panels(2, 2.2)
    eps = np.linspace(0.0, 0.95, 60)[1:]
    a = ax[0]
    for q in (0.5, 0.9, 0.99, 0.999):
        al = q * AF
        w = fold_well(al)
        T = [period(al, w["Vc"] + e * (w["Vs"] - w["Vc"]), w) for e in eps]
        Tlin = 2 * np.pi * np.sqrt(C / -hp(w["c"], al))
        a.plot(eps, np.array(T) / 3600, color=RAMP[q])
        a.text(0.965, T[-1] / 3600, f"{q:g}", fontsize=6.5, va="center")
        a.plot([0], [Tlin / 3600], "o", color=RAMP[q], ms=3, clip_on=False, zorder=5)
    a.axhline(T_ORB / 3600, color=GREY, lw=0.6)
    a.text(0.6, T_ORB / 3600 - 0.95, r"orbital period $T_{\rm orb}$", fontsize=6.5, color=GREY, ha="center")
    a.text(0.965, 16.4, r"$\alpha^*/\alpha_F^*$", fontsize=6.5, va="bottom")
    a.set_xlim(0, 0.95); a.set_ylim(0, 16)
    a.spines["right"].set_visible(True)
    a.set_xlabel(r"normalised family energy $\varepsilon$"); a.set_ylabel(r"$T$ (h)")
    tag(a, "(a)")

    b = ax[1]
    al = 0.5 * AF
    w = fold_well(al)
    ks = -m.h_phi(w["esc"], al, aL, bL, c) * -1
    one = np.logspace(-1, -5, 30)
    T = np.array([period(al, w["Vc"] + (1 - o) * (w["Vs"] - w["Vc"]), w) for o in one])
    x = np.log(1 / one)
    slope = np.sqrt(C / ks)
    off = np.mean(T[-8:] - slope * x[-8:])
    b.plot(x, (slope * x + off) / 3600, color=GREY, lw=0.8, label=r"slope $\sqrt{C/\kappa_s}$")
    b.plot(x, T / 3600, "o", color=RAMP[0.5], ms=2.6, label=r"computed, $\alpha^*/\alpha_F^*=0.5$")
    b.set_xlabel(r"$\log[1/(1-\varepsilon)]$"); b.set_ylabel(r"$T$ (h)")
    b.legend(loc="upper left", fontsize=6.2)
    tag(b, "(b)")
    print(f"Fig 5: separatrix slope {slope:.1f} s, fitted slope {np.polyfit(x[-10:], T[-10:], 1)[0]:.1f} s")
    save(fig, "figure_periodic_family_period_revised.pdf")


# ================================================================ Figure 6
def fig_collapse():
    fig, ax = panels(2, 2.35)
    a, b = ax
    ins = a.inset_axes([0.50, 0.50, 0.46, 0.44])
    for q in (0.5, 0.9, 0.99):
        al = q * AF
        w = fold_well(al)
        far = brentq(lambda x: V(x, al) - w["Vs"], *sorted([w["c"], w["other"]]), xtol=1e-15)
        x = np.linspace(min(w["esc"], far), max(w["esc"], far), 500)
        dV = (V(x, al) - w["Vc"]) * 1e7
        dVs = (w["Vs"] - w["Vc"]) * 1e7
        for axx in (a, ins):
            axx.plot(np.degrees(x - PHIF), dV, color=RAMP[q])
            axx.plot(np.degrees(w["c"] - PHIF), 0, marker="o", color=RAMP[q], ms=3, ls="none")
            axx.plot(np.degrees(w["esc"] - PHIF), dVs, marker="x", color=RAMP[q], ms=4, mew=1.0, ls="none")
        a.plot([], [], color=RAMP[q], label=rf"$\alpha^*/\alpha_F^*={q:g}$")
        E = w["Vc"] + 0.55 * (w["Vs"] - w["Vc"])
        lo, hi = turning(al, E, w)
        x = lo + (hi - lo) * (1 - np.cos(np.linspace(0, np.pi, 400))) / 2
        r = np.sqrt(np.maximum(2 * (E - V(x, al)) / C, 0)) * 1e3
        b.plot(np.degrees(np.r_[x, x[::-1]] - PHIF), np.r_[r, -r[::-1]], color=RAMP[q])
        b.plot(np.degrees(w["c"] - PHIF), 0, "o", color=RAMP[q], ms=2.6)
    a.set_xlabel(r"$\phi-\phi_F$ (deg)"); a.set_ylabel(r"$V-V_c\ (10^{-7}\,\mathrm{m^2\,s^{-2}})$")
    a.set_ylim(-0.08, 1.75)
    fig.legend(loc="outside lower center", ncol=3, fontsize=6.5, columnspacing=1.8)
    ins.set_xlim(-25, 30); ins.set_ylim(-0.012, 0.16)
    ins.tick_params(labelsize=5.5, length=2)
    a.indicate_inset_zoom(ins, edgecolor=GREY, lw=0.5)
    tag(a, "(a)")
    b.axhline(0, color=LIGHT, lw=0.6, zorder=0)
    b.set_xlabel(r"$\phi-\phi_F$ (deg)"); b.set_ylabel(r"$\dot\phi\ (10^{-3}\,\mathrm{rad\,s^{-1}})$")
    tag(b, "(b)")
    save(fig, "figure_periodic_family_collapse_revised.pdf")


# ================================================================ Figure 7
def fig_scaling():
    S2, K2 = np.sin(2 * PHIF), np.cos(2 * PHIF)

    def hc(x, e):
        return (-2 * S2 * np.sin(1.5 * x) * np.sin(0.5 * x) - 4 * K2 * np.sin(x) * np.sin(0.5 * x)**2
                + e * (S2 * np.cos(x) + 2 * K2 * np.sin(x)))

    def hxc(x, e):
        return (S2 * (np.sin(x) - 2 * np.sin(2 * x)) - 4 * K2 * np.sin(1.5 * x) * np.sin(0.5 * x)
                + e * (-S2 * np.sin(x) + 2 * K2 * np.cos(x)))

    def prim(x, e):
        return (-2 * S2 * np.sin(x) * np.sin(0.5 * x)**2 - 4 * K2 * np.sin(0.5 * x)**4
                + e * (S2 * np.sin(x) + 4 * K2 * np.sin(0.5 * x)**2))

    fit_range = (3e-8, 1e-2)
    eps = np.logspace(np.log10(fit_range[0]), -1, 141)
    sep, bar, per = [], [], []
    for e in eps:
        r0 = np.sqrt(2 * e / 3)
        rr = [brentq(hc, s * 0.3 * r0, s * 2.5 * r0, args=(e,), xtol=1e-300, rtol=8.9e-16) for s in (-1, 1)]
        xc = next(x for x in rr if c * hxc(x, e) < 0); xs = next(x for x in rr if c * hxc(x, e) > 0)
        sep.append(abs(xc - xs)); bar.append(abs(c) * abs(prim(xs, e) - prim(xc, e)))
        per.append(2 * np.pi * np.sqrt(C / (-c * hxc(xc, e))))
    data = [np.array(sep), np.array(bar), np.array(per) / 3600]
    cs = abs(c * S2)
    laws = [2 * np.sqrt(2 / 3) * eps**0.5, (4 / 3) * np.sqrt(2 / 3) * cs * eps**1.5,
            2 * np.pi * 1.5**0.25 * np.sqrt(C / (3 * cs)) * eps**-0.25 / 3600]
    fit = (eps >= fit_range[0]) & (eps <= fit_range[1])
    slopes = [np.polyfit(np.log(eps[fit]), np.log(y[fit]), 1)[0] for y in data]
    dev = [abs(y / l - 1) for y, l in zip(data, laws)]
    fig, axs = plt.subplots(2, 3, figsize=(TEXTWIDTH, 3.35), sharex=True, layout="constrained",
                            gridspec_kw=dict(height_ratios=[1.25, 1.0]))
    ylab = [r"$|\phi_c-\phi_s|$ (rad)", r"$\Delta V\ (\mathrm{m^2\,s^{-2}})$", r"$T_{\rm lin}$ (h)"]
    tags = ["(a)", "(b)", "(c)"]
    for j in range(3):
        t, lo = axs[0, j], axs[1, j]
        for axx in (t, lo):
            axx.axvspan(fit_range[1], eps[-1] * 1.3, color="#efeeea", lw=0, zorder=0)
        t.loglog(eps, laws[j], color=INK, lw=0.8, label="leading order")
        t.loglog(eps[::10], data[j][::10], "o", color=BLUE, ms=2.6, label="computed")
        t.set_ylabel(ylab[j], labelpad=1)
        t.text(0.05 if j < 2 else 0.95, 0.95, tags[j], transform=t.transAxes, va="top", ha="left" if j < 2 else "right")
        t.text(0.95 if j < 2 else 0.05, 0.05, f"slope {slopes[j]:.5f}", transform=t.transAxes, fontsize=6.3,
               ha="right" if j < 2 else "left")
        lo.loglog(eps, dev[j], color=BLUE, lw=1.0)
        lo.set_xlabel(r"$\delta/\alpha_F^*$")
        lo.set_xlim(eps[0], eps[-1] * 1.3)
        lo.set_xticks([1e-7, 1e-5, 1e-3, 1e-1])
    axs[1, 0].set_ylabel("relative deviation", labelpad=1)
    axs[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), fontsize=6, handletextpad=0.3)
    k = np.searchsorted(eps, 1e-3)
    for j, (pw, lab) in enumerate([(1.0, r"$\propto\delta$"), (1.0, r"$\propto\delta$"), (0.5, r"$\propto\delta^{1/2}$")]):
        k0 = k if j < 2 else np.searchsorted(eps, 1e-6)
        axs[1, j].loglog(eps, 3 * dev[j][k0] * (eps / eps[k0])**pw, color=GREY, ls="--", lw=0.7, label=lab)
        axs[1, j].legend(loc="upper left", fontsize=6)
    print("Fig 7 slopes", [round(s, 6) for s in slopes])
    save(fig, "figure_fold_scaling_convergence_revised.pdf")


# ================================================================ Figure 8
def fig_similarity():
    ratios = (0.9, 0.99, 0.999)
    fig, ax = panels(3, 2.15)
    Xs = np.linspace(-2, 1, 600); Ps = np.sqrt(2 / 3) * (1 - Xs) * np.sqrt(Xs + 2)
    H = -2 / 3 + 4 * 0.5 / 3
    x0, x1 = canonical_turning(H)
    Xm = x0 + (x1 - x0) * (1 - np.cos(np.linspace(0, np.pi, 400))) / 2
    Pm = np.sqrt(np.maximum(2 * (H - Xm + Xm**3 / 3), 0))
    eps = np.linspace(0.05, 0.95, 37)
    Tc = np.array([canonical_period(e) for e in eps])
    for q in ratios:
        al = q * AF; d = AF - al
        w = fold_well(al)
        far = brentq(lambda x: V(x, al) - w["Vs"], *sorted([w["c"], w["other"]]), xtol=1e-15)
        x = np.linspace(*sorted([w["esc"], far]), 500)
        r = np.sqrt(np.maximum(2 * (w["Vs"] - V(x, al)) / C, 0))
        X = SIGMA * (x - PHIF) / (L * np.sqrt(d)); P = r * T0 / (L * d**0.75)
        ax[0].plot(np.r_[X, X[::-1]], np.r_[P, -P[::-1]], color=RAMP[q], lw=1.0)
        E = w["Vc"] + 0.5 * (w["Vs"] - w["Vc"])
        lo, hi = turning(al, E, w)
        x = lo + (hi - lo) * (1 - np.cos(np.linspace(0, np.pi, 400))) / 2
        r = np.sqrt(np.maximum(2 * (E - V(x, al)) / C, 0))
        X = SIGMA * (x - PHIF) / (L * np.sqrt(d)); P = r * T0 / (L * d**0.75)
        ax[1].plot(np.r_[X, X[::-1]], np.r_[P, -P[::-1]], color=RAMP[q], lw=1.0, label=rf"$\alpha^*/\alpha_F^*={q:g}$")
        Ts = np.array([period(al, w["Vc"] + e * (w["Vs"] - w["Vc"]), w) for e in eps]) * d**0.25 / T0
        devn = abs(Ts / Tc - 1)
        ax[2].semilogy(eps, devn, color=RAMP[q])
        print(f"Fig 8: q={q} max relative period deviation {devn.max():.2e}, at eps=0.5 {np.interp(0.5, eps, devn):.2e}")
    for j, (X, P) in enumerate(((Xs, Ps), (Xm, Pm))):
        ax[j].plot(np.r_[X, X[::-1]], np.r_[P, -P[::-1]], color=INK, ls="--", lw=0.8, label="canonical" if j else None)
        ax[j].plot([-1], [0], **CENTRE); ax[j].set_xlabel(r"$X$"); ax[j].set_ylabel(r"$P$", labelpad=1)
    ax[0].plot([1], [0], **SADDLE)
    ax[0].set_xlim(-2.25, 1.25); ax[0].set_ylim(-1.95, 1.95); ax[0].set_xticks([-2, -1, 0, 1])
    ax[1].set_xlim(-1.95, 0.25); ax[1].set_ylim(-1.45, 1.45)
    fig.legend(*ax[1].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=6.5, columnspacing=1.4)
    ax[2].set_xlabel(r"$\varepsilon$"); ax[2].set_ylabel(r"$|T\delta^{1/4}/(t_0\mathcal{T})-1|$", labelpad=1)
    ax[2].set_xlim(0, 1)
    for a, t in zip(ax, ("(a)", "(b)", "(c)")):
        tag(a, t)
    save(fig, "figure_phase_similarity_canonical_revised.pdf")


# ================================================================ Figure 9
def fig_orbital():
    """Layout of the earlier figure (all seven charge levels, one colour each), restyled."""
    rows = list(csv.DictReader((ORBITAL / "orbital_matched_summary.csv").open()))
    phi0 = np.radians(float(rows[0]["initial_phi_deg"]))
    curves = np.load(ORBITAL / "orbital_matched_trajectories.npz")
    points = [{k: float(v) for k, v in r.items()} for r in csv.DictReader((ORBITAL / "orbital_stroboscopic_points.csv").open())]
    shoot = [{k: float(v) for k, v in r.items()} for r in csv.DictReader((ORBITAL / "orbital_periodic_branch.csv").open())]
    qmin = json.loads((ORBITAL / "orbital_metadata.json").read_text())["minimum_instantaneous_fold_ratio"]
    # Validated categorical palette in its fixed order (adjacent CVD and normal-vision checks pass).
    colours = dict(zip([0.0, 0.25, 0.5, 0.75, 0.85, 0.9, 1.1],
                       ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
    fig, ax = panels(3, 2.3, gridspec_kw=dict(width_ratios=[1.12, 1, 1]))
    for q, col in colours.items():
        v = curves[str(q)]
        ax[0].plot(np.linspace(0, 32, v.shape[1]), np.degrees(v[1] - phi0) / 1e3, color=col, lw=0.8, label=f"{q:g}")
        pp = [p for p in points if p["ratio"] == q]
        ax[1].plot([np.degrees(np.angle(np.exp(1j * p["phi_rad"]))) for p in pp], [p["p"] for p in pp],
                   "o", color=col, ms=1.9, mew=0, ls="none")
    ax[0].set_xlim(0, 32); ax[0].set_xlabel("orbital periods")
    ax[0].set_ylabel(r"unwrapped roll change ($10^3$ deg)", labelpad=1)
    ax[0].set_yticks([-5, 0, 5, 10])
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncol=7, fontsize=6.5,
               title=r"$\alpha^*/\alpha_{F,0}^*$", title_fontsize=6.5, handlelength=1.6, columnspacing=1.3)
    ax[1].set_xlim(-180, 180); ax[1].set_xticks([-180, -90, 0, 90, 180])
    ax[1].set_xlabel(r"$\phi$ mod $360^\circ$ (deg)"); ax[1].set_ylabel(r"$p=\dot\phi/n$", labelpad=1)
    b = ax[2]
    b.axhline(2, color=GREY, lw=0.6, ls="--"); b.axhline(-2, color=GREY, lw=0.6, ls="--")
    b.axvline(qmin, color=LIGHT, lw=0.7)
    b.plot([s["ratio"] for s in shoot], [s["trace"] for s in shoot], color=BLUE, lw=1.0)
    b.text(qmin - 0.03, 1.45, f"{qmin:.3f}", fontsize=6, color=GREY, ha="right")
    b.set_xlim(0, 1.12); b.set_ylim(-2.3, 2.3)
    b.set_xlabel(r"$\alpha^*/\alpha_{F,0}^*$"); b.set_ylabel(r"one-orbit monodromy trace", labelpad=1)
    for a, t in zip(ax, ("(a)", "(b)", "(c)")):
        tag(a, t)
    save(fig, "orbital_matched_poincare_revised.pdf")

if __name__ == "__main__":
    print(f"L={L:.6g}, t0={T0:.6g}, phiF={np.degrees(PHIF):.4f} deg, T_orb={T_ORB/60:.2f} min")
    fig_multiplicity()
    fig_manifold()
    fig_branches()
    fig_portraits()
    fig_period()
    fig_collapse()
    fig_scaling()
    fig_similarity()
    fig_orbital()

