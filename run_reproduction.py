#!/usr/bin/env python3
"""Run public numerical checks or full figure/data reproduction.

Usage (from repository root):
  python run_reproduction.py --smoke
  python run_reproduction.py --full
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data" / "results"
FIGURE_OUT = ROOT / "figures" / "generated"


def run(module: str) -> None:
    print(f"RUN python -m {module}", flush=True)
    subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)


def saved_results_integrity() -> None:
    with (RESULTS / "orbital_ensemble_members.csv").open(newline="") as f:
        members = list(csv.DictReader(f))
    with (RESULTS / "orbital_ensemble_summary.csv").open(newline="") as f:
        ensemble = list(csv.DictReader(f))
    with (RESULTS / "orbital_periodic_branch.csv").open(newline="") as f:
        branch = list(csv.DictReader(f))
    with (RESULTS / "three_axis_results.csv").open(newline="") as f:
        tracking = list(csv.DictReader(f))
    assert len(members) == 7 * 23, len(members)
    assert len(ensemble) == 7 and len(branch) == 56, (len(ensemble),len(branch))
    assert len(tracking) >= 15, len(tracking)
    assert all(int(v["ensemble_size"]) == 21 for v in ensemble)
    assert abs(float(ensemble[2]["nominal_span_32_orbits_deg"]) - 672.2356) < 2
    assert abs(float(branch[-1]["trace"]) + 1.94724) < 0.02
    with (RESULTS / "charging_summary.json").open() as f:
        c = json.load(f)
    assert abs(c["reference"]["alpha_Ckg"] / 1.0855600203727843e-6 - 1) < 1e-8
    assert c["reference"]["OML_small_body"] is False
    assert 4.78 < c["reference"]["ion_ram_energy_eV"] < 4.80
    print("Selected ensemble, shooting, three-axis and charging results: OK", flush=True)


def verify_igrf() -> None:
    data = ROOT / "data"
    official, reduced = {}, {}
    for row in (data / "igrf14coeffs_official.txt").read_text().splitlines():
        p = row.split()
        if p and p[0] in ("g", "h"):
            official[tuple(p[:3])] = float(p[-2])
    for row in (data / "igrf14_2025.txt").read_text().splitlines():
        p = row.split()
        if p and p[0] in ("g", "h"):
            reduced[tuple(p[:3])] = float(p[-1])
    assert len(official) == 195 and official == reduced
    print("IGRF-14 epoch-2025: 195 coefficients match", flush=True)


def verify_modules_import() -> None:
    # These three modules have no expensive computation on import and cover
    # the public package layout used by the full reproduction entry point.
    from model import roll_model
    from numerical_experiments import orbital_poincare, controlled_three_axis_validation
    from figures import generate_revised_figures
    assert roll_model.reference_values()["alphaF"] > 0
    assert orbital_poincare.AF > 0
    assert hasattr(controlled_three_axis_validation, "run")
    assert hasattr(generate_revised_figures, "fig_orbital")
    print("Package imports: OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()

    verify_modules_import()
    saved_results_integrity()
    verify_igrf()
    run("reproducibility.independent_fold_check")
    run("numerical_experiments.charging_feasibility")
    run("reproducibility.scientific_audit")

    if args.smoke:
        print("SMOKE CHECK PASSED", flush=True)
        return

    for module in (
        "numerical_experiments.orbital_poincare",
        "numerical_experiments.controlled_three_axis_validation",
        "reproducibility.validation_tables",
        "reproducibility.scientific_audit",
        "reproducibility.plot_three_axis_results",
        "figures.generate_revised_figures",
        "figures.generate_section8_figures",
        "figures.generate_appendix_figures",
        "figures.generate_igrf_spectrum",
    ):
        run(module)

    expected = (
        "figure_equilibrium_multiplicity_cusps_revised.pdf",
        "figure_fold_manifold_revised.pdf",
        "figure_stability_branches_revised.pdf",
        "figure_phase_portrait_sequence_revised.pdf",
        "figure_periodic_family_period_revised.pdf",
        "figure_periodic_family_collapse_revised.pdf",
        "figure_fold_scaling_convergence_revised.pdf",
        "figure_phase_similarity_canonical_revised.pdf",
        "orbital_matched_poincare_revised.pdf",
    )
    missing = [name for name in expected if not (FIGURE_OUT / name).is_file()]
    assert not missing, f"Missing manuscript figures: {missing}"
    print(f"FULL REPRODUCTION PASSED: {len(expected)} figures regenerated", flush=True)


if __name__ == "__main__":
    main()
