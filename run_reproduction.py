#!/usr/bin/env python3
"""Standalone public reproduction entry point. No private repository required.

--smoke: numerical sanity check and package/results integrity
--full: all expensive orbit integrations, three-axis checks, and figure generation
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "research_assets/python"
CHECKS = BASE / "revision_checks"
FIGS = BASE / "manuscript_figure_scripts/figures"
RESULTS = CHECKS / "results"
FIGURE_OUT = BASE / "manuscript_figure_scripts/figures_out"


def run(path: Path) -> None:
    print(f"RUN {path.relative_to(ROOT)}", flush=True)
    subprocess.run([sys.executable, "-u", str(path)], cwd=ROOT, check=True)


def saved_results_integrity() -> None:
    with (RESULTS / "orbital_ensemble_members.csv").open(newline="") as f:
        members = list(csv.DictReader(f))
    with (RESULTS / "orbital_ensemble_summary.csv").open(newline="") as f:
        ensemble = list(csv.DictReader(f))
    with (RESULTS / "orbital_periodic_branch.csv").open(newline="") as f:
        branch = list(csv.DictReader(f))
    assert len(members) == 7 * 23, len(members)
    assert len(ensemble) == 7, len(ensemble)
    assert len(branch) == 56, len(branch)
    assert all(int(v["ensemble_size"]) == 21 for v in ensemble)
    assert abs(float(ensemble[2]["nominal_span_32_orbits_deg"]) - 672.2356) < 2
    assert abs(float(branch[-1]["trace"]) + 1.94724) < 0.02
    with (RESULTS / "charging_summary.json").open() as f:
        c = json.load(f)
    assert abs(c["reference"]["alpha_Ckg"] / 1.0855600203727843e-6 - 1) < 1e-8
    assert c["reference"]["OML_small_body"] is False
    assert 4.78 < c["reference"]["ion_ram_energy_eV"] < 4.80
    print("Selected ensemble, 56-step shooting branch and charging data: OK", flush=True)


def verify_igrf() -> None:
    data = BASE / "manuscript_figure_scripts/data"
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


def main() -> None:
    ap = argparse.ArgumentParser()
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--full", action="store_true")
    args = ap.parse_args()
    run(CHECKS / "independent_fold_check.py")
    verify_igrf()
    saved_results_integrity()
    run(CHECKS / "charging_feasibility.py")
    if args.smoke:
        print("SMOKE CHECK PASSED", flush=True)
        return
    for path in (
        CHECKS / "orbital_poincare.py",
        CHECKS / "controlled_three_axis_validation.py",
        CHECKS / "validation_tables.py",
        CHECKS / "scientific_audit.py",
        CHECKS / "plot_three_axis_results.py",
        FIGS / "generate_revised_figures.py",
        FIGS / "generate_section8_figures.py",
        FIGS / "generate_appendix_figures.py",
        FIGS / "generate_igrf_spectrum.py",
    ):
        run(path)
    expected = [
        "figure_equilibrium_multiplicity_cusps_revised.pdf",
        "figure_fold_manifold_revised.pdf",
        "figure_stability_branches_revised.pdf",
        "figure_phase_portrait_sequence_revised.pdf",
        "figure_periodic_family_period_revised.pdf",
        "figure_periodic_family_collapse_revised.pdf",
        "figure_fold_scaling_convergence_revised.pdf",
        "figure_phase_similarity_canonical_revised.pdf",
        "orbital_matched_poincare_revised.pdf",
    ]
    missing = [name for name in expected if not (FIGURE_OUT/name).exists()]
    assert not missing, f"Missing manuscript figures: {missing}"
    print(f"FULL REPRODUCTION PASSED; generated {len(expected)} revised manuscript figures", flush=True)


if __name__ == "__main__":
    main()
