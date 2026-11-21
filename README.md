# Lorentz-driven spacecraft roll: supplementary calculations

Supplementary numerical data and reproducibility code for:

**Muhammad Shoaib**, *Fold Bifurcations and Periodic-Orbit Collapse in Lorentz-Driven Spacecraft Roll Dynamics* (manuscript).

This repository accompanies the paper; it does **not** replace its mathematical derivations or the three-axis validation reported in the main text. The governing reduced equation is
```math
C\ddot{\phi}=a\cos\phi+b\sin\phi+c\sin(2\phi).
```

The astroid equilibrium boundary has established precedents in satellite attitude dynamics (Gutnik & Sarychev, 2016). The paper studies its **Lorentz-specific parameterisation**, disappearing frozen periodic family, regular-fold scaling, and the distinct orbitally forced responses.

## What's included

| Directory or file | Contents |
| --- | --- |
| `research_assets/python/manuscript_figure_scripts/figures/roll_model.py` | Prescribed Kepler orbit, Lorentz projections, frozen roll model and fold threshold |
| `.../figures/generate_revised_figures.py` | Source code for the nine revised manuscript figures (requires orbital data for the final panel) |
| `.../figures/generate_section8_figures.py` | Additional fold-robustness figure and numerical sensitivities |
| `.../figures/generate_igrf_spectrum.py` | Supplementary geomagnetic harmonic spectrum |
| `.../figures/generate_appendix_figures.py` | Additional orbital, transverse-torque and charging-scale figures |
| `.../data/` | IGRF-14 epoch-2025 coefficients and source coefficient table |
| `research_assets/python/revision_checks/orbital_poincare.py` | Matched orbitally forced integration, initial-angle ensemble and 56-step one-orbit shooting/monodromy continuation |
| `.../revision_checks/controlled_three_axis_validation.py` | Independent three-axis Newton–Euler/quaternion consistency and controlled-response experiments |
| `.../revision_checks/scientific_audit.py` | Independent coefficient, fold-scaling, quadrature, IGRF and Floquet checks |
| `.../revision_checks/charging_feasibility.py` | Explicitly conditional spacecraft current and power screening |
| `.../revision_checks/results/` | Selected CSV/JSON results. Dense trajectories and generated plots are recreated by the scripts. |
| `run_reproduction.py` | Quick checks and a full numerical/figure reproduction entry point |

Detailed three-axis results remain reported in the manuscript; the public code permits their independent repetition.

## Install

Use Python 3.12 or 3.13, with a virtual environment. Tested requirements are pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

Quick mathematical and charging-screening checks (no long orbital ensemble):

```bash
python run_reproduction.py --smoke
```

Full reproduction of the manuscript figures, 32-orbit ensemble, one-orbit shooting, three-axis calculations, charging screen and geomagnetic checks:

```bash
python run_reproduction.py --full
```

The full run carries out repeated high-accuracy orbital integrations and can take appreciable CPU time. Outputs are regenerated under `research_assets/python/revision_checks/results/`, `research_assets/python/revision_checks/` and `research_assets/python/manuscript_figure_scripts/figures_out/`. It does not alter a manuscript file or access the private source repository.

For individual checks:

```bash
python research_assets/python/revision_checks/independent_fold_check.py
python research_assets/python/revision_checks/orbital_poincare.py --ensemble-only
python research_assets/python/revision_checks/orbital_poincare.py --shooting-only
python research_assets/python/revision_checks/controlled_three_axis_validation.py
```

## Interpretation

The frozen-orbit Hamiltonian admits a family of periodic librations that disappears at a **regular saddle–centre fold**. Its local exponents are `1/2`, `3/2` and `-1/4`; cusp points require separate scaling. The frozen family is an instantaneous phase-space structure and should not be interpreted as adiabatically followed on the prescribed orbit.

The orbitally forced system is integrated directly. For circulating cases, 32-orbit unwrapped spans depend sensitively on the initial roll angle (even changes of `1e-6` degrees matter). The 21-angle sample spans are **sample ranges**, not converged predictions or evidence of chaos. Circulation-onset times and early-time excursions are more repeatable. The separately shot one-orbit periodic branch has its own initial data; Floquet multipliers describe *linear* classification only.

Three-axis consistency experiments use **ideal transverse control**, without saturation or hardware qualification. The charging calculations are screening estimates, not a self-consistent plasma sheath model or evidence of passive attainment of the reference specific charge. The attracted-electron OML formula is inappropriate for the stated radius-to-Debye-length ratios. Current–voltage scales are conditional.

## Sources and licence

- Gutnik, S. A. and Sarychev, V. A. (2016), *A Symbolic Investigation of the Influence of Aerodynamic Forces on Satellite Equilibria*, CASC 2016, DOI: [10.1007/978-3-319-45641-6_16](https://doi.org/10.1007/978-3-319-45641-6_16).
- IGRF-14 Gauss coefficients: IAGA V-MOD working group / [NOAA Geomagnetism](https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field). The supplied data are externally sourced reference coefficients.
- Original scripts and author-created material: MIT licence; see [LICENSE](LICENSE). Third-party scientific coefficients retain their source attribution.

Cite the associated paper for the scientific results and this repository for the numerical implementation. The manuscript has not been represented here as a published journal article.
