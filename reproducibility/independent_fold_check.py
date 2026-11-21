#!/usr/bin/env python3
"""Independent numerical checks for the frozen-orbit Lorentz roll manuscript.

Depends on numpy and scipy. This script does not import the manuscript's code.
All inertias are mass-normalised (m^2); specific charge is C/kg.
"""
import numpy as np
from scipy.optimize import brentq

MU = 3.9860e14
W_E = 7.2921e-5
B_D = 7.943e15
A, B, C = 0.4, 1.0, 1.0
AORB = 6900e3
ECC = 0.001
INC = np.deg2rad(51.0)
F = np.deg2rad(60.0)
X0, Y0 = 1.0, 1.0
P = AORB * (1.0 - ECC**2)
R = P / (1.0 + ECC * np.cos(F))
RDOT = ECC * np.sqrt(MU / P) * np.sin(F)
FDOT = np.sqrt(MU * P) / R**2
U = F

# Independent orbital-frame vector cross product: (T,N,R) is right-handed.
b_t = -(B_D / R**3) * np.sin(INC) * np.cos(U)
b_n = -(B_D / R**3) * np.cos(INC)
b_r = 2 * (B_D / R**3) * np.sin(INC) * np.sin(U)
v_t = R * (FDOT - W_E * np.cos(INC))
v_n = R * W_E * np.sin(INC) * np.cos(U)
v_r = RDOT
force_per_specific_charge = np.cross([v_t, v_n, v_r], [b_t, b_n, b_r])
tbar, nbar, rbar = force_per_specific_charge
al = X0 * nbar - Y0 * tbar
bl = -(X0 * tbar + Y0 * nbar)
c = 0.5 * FDOT**2 * (A - B)
fold = 2 * abs(c) / (abs(al)**(2/3) + abs(bl)**(2/3))**(3/2)

def h(phi, alpha):
    return alpha * (al * np.cos(phi) + bl * np.sin(phi)) + c * np.sin(2*phi)

def hp(phi, alpha):
    return alpha * (-al * np.sin(phi) + bl * np.cos(phi)) + 2*c * np.cos(2*phi)

def potential(phi, alpha):
    return -alpha * al * np.sin(phi) + alpha * bl * np.cos(phi) + 0.5 * c * np.cos(2*phi)

def fixed_points(alpha):
    grid = np.linspace(0.0, 2*np.pi, 12001)
    vals = h(grid, alpha)
    roots = []
    if abs(vals[0]) < 1e-14:
        roots.append(0.0)
    for j in range(len(grid)-1):
        if vals[j]*vals[j+1] < 0:
            roots.append(brentq(lambda z: h(z, alpha), grid[j], grid[j+1]))
    roots = sorted(set(round(float(z), 12) for z in roots))
    return [(z, 'centre' if hp(z, alpha) < 0 else 'saddle') for z in roots]

# Find the positive fold root using astroid geometry; select the angle near 314 deg.
fold_candidates = np.linspace(0, 2*np.pi, 50000)
fold_angle = fold_candidates[np.argmin(np.abs(h(fold_candidates, fold)) + np.abs(hp(fold_candidates, fold)))]
from scipy.optimize import root
solution = root(lambda z: [h(z[0], z[1]), hp(z[0], z[1])], [fold_angle, fold])
assert solution.success
fold_angle, fold_direct = solution.x

fractions = np.geomspace(1e-7, 1e-2, 50)
separations, barriers, periods = [], [], []
for eps in fractions:
    charge = fold * (1.0 - eps)
    roots = fixed_points(charge)
    centres = [r for r, t in roots if t == 'centre']
    saddles = [r for r, t in roots if t == 'saddle']
    distance = lambda z: abs((z - fold_angle + np.pi) % (2*np.pi) - np.pi)
    centre = min(centres, key=distance)
    saddle = min(saddles, key=distance)
    separations.append(distance(centre - saddle + fold_angle))
    barriers.append(potential(saddle, charge) - potential(centre, charge))
    periods.append(2*np.pi * np.sqrt(C / -hp(centre, charge)))

slopes = {
    'angular separation': np.polyfit(np.log(fractions), np.log(separations), 1)[0],
    'potential barrier': np.polyfit(np.log(fractions), np.log(barriers), 1)[0],
    'linear period': np.polyfit(np.log(fractions), np.log(periods), 1)[0],
}

print(f'Tbar = {tbar:.12g}; Nbar = {nbar:.12g}; Rbar = {rbar:.12g}')
print(f'a_L = {al:.12g}; b_L = {bl:.12g}; c = {c:.12g}')
print(f'astroid fold = {fold:.12g} C/kg')
print(f'direct fold = {fold_direct:.12g} C/kg, phi_F = {np.rad2deg(fold_angle):.7f} degrees')
print(f'orbital period = {2*np.pi*np.sqrt(AORB**3/MU)/60:.6f} minutes')
print(f'zero-charge small-amplitude period = {2*np.pi*np.sqrt(C/(-2*c))/60:.6f} minutes')
print('fixed-point counts:', {frac: len(fixed_points(frac*fold)) for frac in [0,0.5,0.99,1.01]})
print('independent least-squares slopes:')
for key,val in slopes.items(): print(f'  {key}: {val:.8f}')
print(f'ratio of transverse Lorentz torque to retained roll torque at fold = '
      f'{np.sqrt(X0**2+Y0**2)*abs(rbar)/np.hypot(al,bl):.6f}')

assert np.isclose(fold,1.0855600203727837e-6,rtol=1e-10)
assert abs(np.rad2deg(fold_angle)-314.4798663) < .001
assert abs(slopes['angular separation']-0.5) < 0.001
assert abs(slopes['potential barrier']-1.5) < 0.001
assert abs(slopes['linear period']+0.25) < 0.001
