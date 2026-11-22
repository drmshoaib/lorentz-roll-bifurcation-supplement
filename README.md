# Lorentz-driven spacecraft roll dynamics — numerical supplement

Source code, numerical data and reproducibility checks accompanying:

**Muhammad Shoaib**, *Fold Bifurcations in Constrained Lorentz-Driven Spacecraft Roll: Frozen Periodic Families and Orbitally Forced Responses* (manuscript).

The reference dipole uses the terrestrial polarity, with a negative axial Gauss coefficient (`B_D < 0` in `model/roll_model.py`). Earlier versions of this repository used the opposite sign; the fold-charge magnitude is unchanged, and signed roll angles move by 180 degrees (reference fold at 134.48 degrees, matched initial angle 167.262167 degrees).

The paper contains the mathematical derivations and **retains the three-axis validation**. This public repository makes the underlying numerical calculations available for independent reproduction.

The frozen, constrained-roll equation is
```math
C\ddot\phi=a\cos\phi+b\sin\phi+c\sin(2\phi).
```
Its equilibrium astroid has precedents in satellite attitude dynamics (Gutnik and Sarychev, 2016). The present investigation addresses its Lorentz-force parameterisation, the disappearing trapped periodic family, near-fold scaling, and the distinct response of the orbitally forced system.

## Repository structure

```text
model/
    roll_model.py                        Reduced roll equation and reference parameters
numerical_experiments/
    orbital_poincare.py                  Matched trajectories, sensitivity ensemble and Floquet shooting
    controlled_three_axis_validation.py Full three-axis attitude consistency calculations
    charging_feasibility.py             Conditional plasma-current and power screening
reproducibility/
    independent_fold_check.py           Independent fold/scaling checks
    scientific_audit.py                 Coefficient, period, IGRF and Floquet checks
    validation_tables.py                Numerical table and tolerance regeneration
    plot_three_axis_results.py          Supporting three-axis figure
data/
    igrf14_2025.txt                     IGRF-14 epoch-2025 coefficient extract
    igrf14coeffs_official.txt           Original coefficient table for cross-checking
    results/                             Selected CSV/JSON numerical results
figures/
    generate_revised_figures.py         All nine figures of the revised manuscript
    generate_section8_figures.py        Additional fold-robustness figure
    generate_appendix_figures.py        Additional orbital/torque/charge figures
    generate_igrf_spectrum.py           IGRF harmonic spectrum
    generated/                          Outputs created by the scripts (not versioned)
docs/
    reproducibility.md                 Data-to-claim map and interpretation notes
run_reproduction.py                     Public reproduction entry point
requirements.txt                        Pinned Python dependencies
```

## Installation

Use Python 3.12 or 3.13 and a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
On Windows, activate with `.venv\Scripts\activate`.

## Reproduction

From the **repository root**, run:

```bash
python run_reproduction.py --smoke
```

This verifies the stored data, fold calculations, charging screen, IGRF-14 coefficients and numerical audit. The automated [GitHub Actions workflow](.github/workflows/verify.yml) runs the same check on new commits.

For full numerical reproduction:

```bash
python run_reproduction.py --full
```

This re-integrates the initial-angle ensemble and 32-orbit trajectories, continues the one-orbit periodic-response branch to 1.16 times the reference fold charge and the two-orbit branch created at its period-doubling point, runs the controlled three-axis experiments, recalculates the charging screen and regenerates all nine manuscript figures. The full calculation is substantially more expensive than the smoke test. Outputs are written under `data/results/` and `figures/generated/`.

Individual modules can also be run from the root, for example:

```bash
python -m reproducibility.independent_fold_check
python -m numerical_experiments.orbital_poincare --ensemble-only
python -m numerical_experiments.orbital_poincare --shooting-only
python -m numerical_experiments.orbital_poincare --period-two-only
python -m numerical_experiments.controlled_three_axis_validation
python -m figures.generate_igrf_spectrum
```

## Important distinctions

The trapped periodic family belongs to the **frozen autonomous** roll model. The coefficients of the actual orbital problem vary significantly within one orbital period, so the frozen family should not be treated as an adiabatically realised spacecraft orbit.

The orbitally forced 32-orbit trajectories are sensitive to the initial roll angle, including offsets of `1e-6` degrees. The 21-angle ensemble spans are finite **sample ranges**, not confidence intervals or converged long-term predictions. The times of the first full-turn excursion and short-horizon excursions are more repeatable. The separate one-orbit shooting solutions have their own initial conditions; Floquet multipliers give a linear classification. The one-orbit trace reaches -2 at 1.11406 times the reference fold charge, where a linearly elliptic two-orbit response bifurcates; that response ends in a fold at 1.11526.

The three-axis tests use ideal, unsaturated transverse feedback and are numerical consistency checks rather than a flight-qualified controller. The charging study uses screening currents and conditional power scales, not a self-consistent sheath or hardware model; the attracted-electron OML law is outside its small-body domain for the reference conductor.

The supplementary repository excludes private drafts, internal correspondence, revision history and unrelated exploratory material.

## References, citation and licence

Gutnik, S. A. and Sarychev, V. A. (2016), *A Symbolic Investigation of the Influence of Aerodynamic Forces on Satellite Equilibria*, CASC 2016. DOI: [10.1007/978-3-319-45641-6_16](https://doi.org/10.1007/978-3-319-45641-6_16).

IGRF-14 numerical coefficients are credited to the IAGA V-MOD working group / [NOAA](https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field).

Cite the associated manuscript for the science and this repository for the numerical implementation. Source code and author-created material use the [MIT licence](LICENSE); the IGRF reference coefficients retain their external source attribution. The manuscript is not represented here as a published article.
