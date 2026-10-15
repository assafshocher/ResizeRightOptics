# Paper experiments

Code and recorded measurements for **A Unified Theory of Image Scaling**.
RRO is the installable library at the repository root. This folder preserves
the paper experiments, including the numerical operators used in those runs.

```bash
pip install -r experiments/requirements.txt
python experiments/run.py --list
```

Add `jax` for the JAX software comparisons, and `rawpy` for RAW optical captures.
The full SR suites were run on CUDA. Dataset downloads and model-weight
downloads happen only when the relevant preparation or experiment is run.

## What is where

| Experiment | Entry point | Inputs |
| --- | --- | --- |
| Set5 self-supervised SR | `prepare-set5`, `zssr`, `set5-matched`, `set5-gaussian` | Pinned Set5 download; protocols beside the sources |
| BSD100 / Set14 SR and one-shot refinement | `bsd100-set14` | Pinned dataset downloads |
| Software resizers and classification stability | `library-audit` | Original scene collection; TorchVision classifier weights |
| Classification controls | `classification-controls` | Same scenes and archived audit protocol |
| HPatches geometric matching | `hpatches` | HPatches dataset supplied with `--dataset` |
| Simulated optics | `optics-simulation` | Synthetic signals, pupil/PSF and sensor/noise model |
| Geometric transformations | `geometric-transforms` | Kodak `kodim19.png` figure input |
| Real optical zoom | `real-zoom` | Original RAW zoom capture pairs, supplied separately |
| Continuous kernel families | `kernel-surfaces` | Analytic cameras and interpolation kernels |
| Grid, solution sets, Huygens diagram | `projected-grid`, `solution-set`, `solution-members`, `huygens` | Self-contained figure sources |
| One-shot and aliasing checks | `one-shot`, `aliasing` | Set5 or the declared synthetic/photographic inputs |

Pass script-specific options after `--`. For example:

```bash
python experiments/run.py zssr -- --help
python experiments/run.py prepare-set5 -- --scale 2 --radius 64
python experiments/run.py zssr -- --image baby --scale 2 --m 1 2 3 4 5 \
  --steps 10000 --lr 0.0001 --residual --crop-margin 16 --run-name my_run

python experiments/run.py library-audit -- --dataset /path/to/scenes \
  --scene-count 50 --source-size 2047 --target-size 255 --output my_audit

python experiments/run.py hpatches -- --help
python experiments/run.py bsd100-set14 -- prepare
python experiments/run.py bsd100-set14 -- summary
```

For capture scripts without a dataset argument, set `SCALING_CAPTURE_DATA` or use
`python experiments/run.py --captures /path/to/captures real-zoom -- ...`.
The SR-RAW test collection is not bundled; its download is linked in the
vendored `zoom-learn-zoom` README and [THIRD_PARTY.md](../THIRD_PARTY.md).
A figure requiring Kodak's
image is included at
`workspace/outputs/paper_revision_2026-09-17/figures/kodim19.png`;
the original source and credit are in [THIRD_PARTY.md](../THIRD_PARTY.md).

## Figures and recorded results

The folder `workspace/` keeps the source directory relationships used by the
experiments. This preserves imports and lets the original plot scripts find
their recorded JSON/CSV measurements. In particular:

- `workspace/work/zssr_scale_transfer/`: kernels, training, inference, tests,
  protocols, result summaries and per-image measurements.
- `workspace/work/benchmark_expansion_20260930/`: BSD100/Set14 runner,
  figure generators and archived score records.
- `workspace/work/library_audit/`: software and classifier audit, camera
  models and saved measurements.
- `workspace/outputs/paper_revision_2026-09-17/`: physical simulation,
  transformations, software-audit sources and data.
- `workspace/figures/`: one-shot, aliasing, classifier and numerical checks.
- `workspace/work/revised_manuscript_v2/scripts/`: theoretical figure sources,
  including the continuous kernel surfaces.
- `workspace/work/final_revision/`: grid, solution-set and limit figures.
- `workspace/work/zoom_experiment/`: real optical zoom fitting and evaluation.

The final figure assets are in [`paper/figures/`](../paper/figures/). The
[figure index](../paper/FIGURES.md) connects them to candidate generators.
Recorded measurements are included so results can be inspected and some plots
regenerated without repeating training. Checkpoints, raw datasets and training
loss histories are excluded. A script may require additional inputs listed in
its original protocol; inclusion is not a claim that every historical variant
is the final paper configuration.

The BSD100 and Set14 plot recipes include their complete frozen per-image
score snapshots:

```bash
python experiments/run.py plot-bsd100
python experiments/run.py plot-set14
```

The [source manifest](SOURCE_MANIFEST.json) records original and packaged hashes
and any mechanical path changes. The package replaces workstation-specific
output paths, exposes a configurable capture-data path, and resolves the
benchmark runner's shared source dependencies. These changes do not alter
the experiment mathematics or the archived measurement values.

## Reproduction scope

The release checks exercise the library, installation, entry points and selected
self-contained figures. Full neural training, RAW capture experiments and all
software-version comparisons are not re-run during packaging. Their included
records describe the original runs. See [release checks](../docs/RELEASE_CHECKS.md)
and each experiment's protocol for the exact scope.

Some original suite runners use Linux locking/process metadata. Use their
individual experiment entry points for smaller runs on other platforms.
Vendored sources and datasets retain their own licenses.
The original TensorFlow ZSSR files are preserved as Python 2 reference code;
the active SR experiment implementation is the separate Python 3/PyTorch code.

The original expansion runner's `preflight` and `suite` modes also expect the
separate staging artifacts `reference_clean_summary.json` and `reference_inputs/`.
These remote staging files are not included in this source release. The
individual `prepare`, `clean`, `zssr`, and `summary` modes and the archived-plot
recipes are separate entry points; the release does not bypass the missing
preflight comparisons.
