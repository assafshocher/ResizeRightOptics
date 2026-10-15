# Third-party code and data

The MIT license at the repository root covers RRO. It does not replace the
licenses of the following preserved experiment sources or benchmark datasets.

| Source | Location | License / credit |
| --- | --- | --- |
| [ResizeRight](https://github.com/assafshocher/ResizeRight), commit `510d4d5b67dccf4efdee9f311ed42609a71f17c5` | `experiments/workspace/work/zssr_scale_transfer/vendor/ResizeRight/` | MIT; original license retained. |
| [ZSSR](https://github.com/assafshocher/ZSSR), commit `d2948526fde799a0c111469b78cd9e1b8659cb0f` | `experiments/workspace/work/zssr_scale_transfer/vendor/ZSSR/` | WIS/Yeda Academic Non Commercial Software Code License; retained alongside the original source. |
| [Zoom to Learn, Learn to Zoom](https://github.com/ceciliavision/zoom-learn-zoom) | `experiments/workspace/work/zoom_experiment/vendor/zoom-learn-zoom/` | Upstream CC BY-NC 4.0 license retained. See the upstream README for authorship and SR-RAW dataset details. |
| Set5, Set14, BSD100 | Downloaded by experiment preparation scripts | Original datasets and their terms apply. Raw benchmark images and training checkpoints are not bundled. URLs, revisions and recorded hashes are preserved in the experiment metadata. |
| [HPatches](https://github.com/hpatches/hpatches-dataset) | Supplied separately to the geometry experiments | See the dataset's license and attribution. |
| Kodak image `kodim19.png` | `experiments/workspace/outputs/paper_revision_2026-09-17/figures/` | Kodak Lossless True Color Image Suite; original source `https://r0k.us/graphics/kodak/kodak/kodim19.png`. Included as the original figure's benchmark input. |

The paper's self-supervised SR code is a PyTorch experimental implementation
following the ZSSR setup described in its protocols, with disclosed differences
from the original TensorFlow implementation. Its inclusion should not be read
as a change to the original ZSSR license.

The real-capture experiments use the SR-RAW test scenes and require the dataset
to be downloaded separately. The upstream README links to the test archive.
Use `SCALING_CAPTURE_DATA` to point to its scene folders. Figure assets and
measurements are included; the RAW dataset itself is not bundled.
