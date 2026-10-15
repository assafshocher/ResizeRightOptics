"""Named entry points for the preserved paper experiments.

Use --list to inspect recipes. Arguments after -- go directly to the original
script; the launcher only supplies working directories and import paths.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
RECIPES = {
    "prepare-set5": ("work/zssr_scale_transfer/prepare.py", "Set5 inputs and dataset manifest"),
    "zssr": ("work/zssr_scale_transfer/experiment.py", "One self-supervised SR experiment"),
    "set5-matched": ("work/zssr_scale_transfer/run_matched_suite.py", "Matched band-limited or literal Set5 suite"),
    "set5-gaussian": ("work/zssr_scale_transfer/run_gaussian_suite.py", "Gaussian-camera Set5 suite"),
    "bsd100-set14": ("work/benchmark_expansion_20260930/run_benchmarks.py", "BSD100/Set14 one-shot and SR experiments"),
    "library-audit": ("work/library_audit/library_model_audit.py", "Software resizing paths and classifier stability"),
    "classification-controls": ("figures/check_classifier_perturbations.py", "Classifier noise/crop controls"),
    "hpatches": ("outputs/hpatches_unit_kernel_experiment.py", "HPatches correspondence experiment"),
    "optics-simulation": ("outputs/paper_revision_2026-09-17/figures/gen_fig_physsim_col.py", "Pupil, optics, noise and composition simulation"),
    "geometric-transforms": ("outputs/paper_revision_2026-09-17/figures/gen_fig_rigid.py", "Subpixel transformation/round-trip figure"),
    "real-zoom": ("work/zoom_experiment/multiscene_physical_ls.py", "Fitted-PSF real RAW optical zoom experiment"),
    "kernel-surfaces": ("work/revised_manuscript_v2/scripts/make_fig_surfaces10.py", "Continuous kernel surfaces and slice figures"),
    "projected-grid": ("work/final_revision/scripts/make_projected_grid_figure.py", "Centered projected-grid derivation figure"),
    "solution-set": ("work/final_revision/figures/gen_fig_set.py", "Consistent solution-set schematic"),
    "solution-members": ("work/final_revision/figures/gen_fig_sets1d.py", "Equivalent continuous-kernel members"),
    "huygens": ("outputs/huygens_bidirectional_v3_equal_angles_2026-09-29/draw_figure.py", "Vector Huygens-style consistency diagram"),
    "one-shot": ("figures/gen_clean_upscaling_validation.py", "One-shot canonical refinement experiment"),
    "aliasing": ("figures/gen_aliasing_validation.py", "Aliasing stress test"),
    "plot-software": ("figures/gen_compact_library_audit.py", "Plot archived software/classification measurements"),
    "plot-sr": ("work/zssr_scale_transfer/figures/gen_fig_matched_modes.py", "Plot complete matched Set5 results"),
    "plot-bsd100": ("work/benchmark_expansion_20260930/figures/gen_fig_zssr_bsd100.py", "Plot the archived 100-image BSD100 SR results"),
    "plot-set14": ("work/benchmark_expansion_20260930/figures/gen_fig_zssr_set14.py", "Plot the archived 14-image Set14 SR results"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--captures", type=Path, help="Path to the separately supplied capture dataset")
    parser.add_argument("recipe", nargs="?", choices=RECIPES)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.list or args.recipe is None:
        for name, (_, explanation) in RECIPES.items():
            print(f"{name:24} {explanation}")
        return
    script = WORKSPACE / RECIPES[args.recipe][0]
    if not script.is_file():
        parser.error(f"Missing packaged source: {script}")
    env = os.environ.copy()
    directories = [script.parent, WORKSPACE, WORKSPACE / "figures", WORKSPACE / "outputs",
                   WORKSPACE / "work/zssr_scale_transfer", WORKSPACE / "work/library_audit",
                   WORKSPACE / "work/zoom_experiment"]
    env["PYTHONPATH"] = os.pathsep.join(map(str, directories)) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MPLBACKEND", "Agg")
    if args.captures:
        env["SCALING_CAPTURE_DATA"] = str(args.captures.resolve())
    for relative in ["paper_output/figs", "work/final_revision/figs",
                     "outputs/paper_revision_2026-09-17/figs",
                     "outputs/paper_revision_red_2026-09-20/figs"]:
        (WORKSPACE / relative).mkdir(parents=True, exist_ok=True)
    cwd = script.parent if args.recipe in {"prepare-set5", "zssr", "set5-matched", "set5-gaussian"} else WORKSPACE
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    raise SystemExit(subprocess.run([sys.executable, str(script), *arguments], cwd=cwd, env=env).returncode)


if __name__ == "__main__":
    main()
