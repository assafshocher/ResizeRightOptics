# Release checks

Checked on 15 October 2026, on macOS with Python 3.12.14, NumPy 2.3.5 and
PyTorch 2.14.0. These are packaging and implementation checks, not new paper
experiments. Archived research measurements in `VALIDATION.md` and
`validation.json` are preserved separately.

| Check | Result |
| --- | --- |
| Complete RRO test suite | **605 passed** |
| Quickstart | NumPy, Airy, PyTorch gradients and polyphase comparison passed |
| Wheel build and isolated installation | Passed; version 0.1.0 |
| Installed-wheel smoke test | NumPy and PyTorch resizing passed; finite image gradients |
| Core library source | All five files identical to the original tested implementation |
| Python source syntax | 169 Python 3 files passed; three upstream Python 2 reference files retained separately |
| Experiment launcher | All 22 named entry points exist; listing passed |
| Documentation links | Local links in the main README, API guide and figure index checked |

The quickstart's ordinary/polyphase maximum absolute difference was
`4.218847493575595e-15`. The isolated installation was imported from its new
installation directory, not the development checkout.

## Figure regeneration

The following recipes completed using the packaged sources and recorded inputs:

- `kernel-surfaces`
- `projected-grid`
- `solution-set`
- `huygens` (including its layout checks)
- `plot-bsd100` (100 images per panel, six panels; archived-data validation passed)
- `plot-set14` (14 images per panel, six panels; archived-data validation passed)

These checks regenerate figures; they do not repeat neural training. Full
training, RAW capture evaluation, GPU performance and all software-version
comparisons were not rerun for this release. See the experiment README for
dataset requirements and the original expansion runner's unavailable remote
preflight inputs.

## Distribution

Wheel: `resize_right_optics-0.1.0-py3-none-any.whl`

SHA-256:
`447d32da67595253d762ab6f94816b22503938a764db5e8463bc371985f14b3e`

The source manifest records provenance and mechanical path adaptations.
Upstream TensorFlow ZSSR Python 2 files are archival references; the active
paper SR experiments use the separate Python 3/PyTorch implementation.
