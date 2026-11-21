# Reproducibility and scope

The manuscript retains its analytical derivations and three-axis validation. This public repository makes the supporting calculations inspectable and reproducible.

## Where to find results

| Manuscript result | Source code | Stored output |
| --- | --- | --- |
| Frozen equilibrium and fold, exponents and period checks | `model/roll_model.py`, `reproducibility/independent_fold_check.py`, `reproducibility/scientific_audit.py` | `data/results/scientific_audit.json` |
| 21-angle orbital sensitivity ensemble | `numerical_experiments/orbital_poincare.py` | `data/results/orbital_ensemble_members.csv`, `data/results/orbital_ensemble_summary.csv` |
| Matched orbitally forced trajectories and stroboscopic points | `numerical_experiments/orbital_poincare.py` | `data/results/orbital_matched_summary.csv`, `data/results/orbital_stroboscopic_points.csv` |
| Separate one-orbit periodic-response continuation, monodromy trace and finite differences | `numerical_experiments/orbital_poincare.py`, `reproducibility/scientific_audit.py` | `data/results/orbital_periodic_branch.csv`, `data/results/orbital_floquet_finite_difference.csv` |
| Three-axis tracking and tolerance tests | `numerical_experiments/controlled_three_axis_validation.py`, `reproducibility/validation_tables.py` | `data/results/three_axis_results.csv`, `data/results/three_axis_tolerance.csv` |
| Positive/negative charging and sensitivity estimates | `numerical_experiments/charging_feasibility.py` | `data/results/charging_summary.json` and `data/results/charging_*.csv` |
| IGRF-14 spectrum | `figures/generate_igrf_spectrum.py` | `data/igrf14_2025.txt` and `data/igrf14coeffs_official.txt`; regenerated figure under `figures/generated/` |
| Nine manuscript figures and supplementary diagnostics | `figures/generate_revised_figures.py`, `figures/generate_section8_figures.py`, `figures/generate_appendix_figures.py` | Regenerated under `figures/generated/` |

All numerical outputs are calculations for the parameters specified in the paper. No observed spacecraft attitude data are asserted.

## Numerical definitions

The matched initial roll angle is printed to six decimals: `347.262167` degrees. The sensitivity ensemble uses 21 uniformly spaced offsets in `[-0.01, 0.01]` degrees and two additional offsets of `±1e-6` degrees. The circulation-onset time is the first time the running unwrapped excursion exceeds 360 degrees.

Thirty-two-orbit circulating angular spans are sensitive to initial data and numerical conditions. The ensemble ranges are finite sample ranges, not asymptotic statements or confidence intervals. The separate periodic-response shooting solutions are found by an orbital stroboscopic return condition and have their own initial states; their multipliers give a linear classification only.

The regular-fold laws apply away from cusp degeneracies. The frozen periodic family is an instantaneous Hamiltonian structure and does not imply an adiabatic orbitally forced periodic response.

The three-axis experiments use prescribed reference roll with ideal transverse feedforward and diagnostic proportional-derivative feedback. The control is not flight qualified and need not stabilise the roll phase. This numerical evidence remains in the main paper.

The charging calculations are conditional current-balance screens, not sheath solutions. The electron OML small-body approximation fails at the reference conductor-to-Debye-length ratio.

## How to run

Start from the repository root and install `requirements.txt`.

- Quick verification: `python run_reproduction.py --smoke`.
- All numerical integrations and figures: `python run_reproduction.py --full`.
- Individual modules: `python -m reproducibility.independent_fold_check`, `python -m numerical_experiments.orbital_poincare --ensemble-only`.

Dense trajectory arrays (`*.npz`) and all generated figures are excluded from the versioned repository. The scripts regenerate them. The stored CSV/JSON files are the selected numerical outputs used for inspection.

The IGRF-14 coefficients are externally sourced IAGA V-MOD reference data. Their provenance is documented in the repository README.
