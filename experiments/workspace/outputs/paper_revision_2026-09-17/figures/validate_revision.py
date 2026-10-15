"""Checks for the exact revised source, saved empirical results, and algebra."""
from pathlib import Path
import re
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parents[1]
source = (PACKAGE / "main.tex").read_text()
def uncomment(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text)
combined = uncomment(source)
main, app = combined.split(r"\appendix", 1)
labels = re.findall(r"\\label\{([^}]+)\}", combined)
refs = re.findall(r"\\(?:eqref|ref|autoref)\{([^}]+)\}", combined)
assert not (set(refs)-set(labels)), set(refs)-set(labels)
assert len(labels) == len(set(labels)), "duplicate labels"
bibs = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", combined))
cites = set(k.strip() for group in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", combined) for k in group.split(","))
assert not (cites-bibs), cites-bibs
main_labels = set(re.findall(r"\\label\{([^}]+)\}", main))
for heading in app.splitlines():
    if heading.startswith(r"\section{"):
        targets = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", heading)
        assert targets and set(targets) <= main_labels, heading
assert not re.search(r"\\mathbf\s+[QP](?:_|\b)", app)
stack = []
for op, env in re.findall(r"\\(begin|end)\{([^}]+)\}", combined):
    if op == "begin": stack.append(env)
    else: assert stack and stack.pop() == env, (env,stack)
assert not stack
for asset in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", combined):
    assert (PACKAGE/asset).exists(), asset
assert source.count(r"\label{eq:requirefull}") == 1
assert source.count(r"\section{Wiener realization") == 1
assert "For $N\\ge\\lceil sN_0\\rceil$" not in source
assert "same-model empirical ceiling" not in source
assert source.count(r"Equation~\eqref{eq:master} is then") == 1
print("PASS: revised-source references, citations, environments, assets, appendix titles, restored equation/Wiener proof, and repaired claims")

optical = json.loads((PACKAGE/"optical_results.json").read_text())
assert optical["anchor_multiplier_max_abs_diff"] == 0
assert optical["anchor_output_max_abs_diff"] == 0
assert optical["cons_fam_x1/2"] == optical["cons_matched_x1/2"]
assert optical["cons_fam_x1/3"] < optical["cons_matched_x1/3"]
assert abs(optical["cons_fam_x1/3"]-0.200) < 0.001
assert abs(optical["cons_matched_x1/3"]-0.235) < 0.001
print("PASS: exact shared optical anchor; off-anchor result and rounded manuscript values")
geometry = json.loads((PACKAGE/"geometry_results.json").read_text())
assert geometry["translation_xy"] == [0.37,0.21]
assert geometry["consistent_rmse"] < 1e-12
assert abs(geometry["classical_rmse"]-0.00424) < 0.00001
assert abs(geometry["classical_relative_rmse"]-0.0315) < 0.0001
print("PASS: corrected geometric map, absolute-vs-relative RMSE, and round trip")

rng = np.random.default_rng(20260917)
omega = rng.uniform(-8, 8, 30000)

def h(w, holes=False):
    visible = np.abs(w) < 2.4
    if holes:
        visible &= ~((np.abs(w) > 0.6) & (np.abs(w) < 0.85))
    return visible * (0.4 + np.exp(-0.18*w*w)) * np.exp(-0.31j*w)

def reciprocal(v):
    out = np.zeros_like(v)
    np.divide(1, v, out=out, where=v != 0)
    return out

def kernel(w, s, holes=False):
    return h(w/s, holes) * reciprocal(h(w, holes))

scales = (0.25, 0.5, 0.73, 1.0, 1.4, 2.0, 3.0)
error = 0.0
for holes in (False, True):
    source = h(omega, holes) != 0
    for s in scales:
        retained = source & (h(omega/s, holes) != 0)
        roundtrip = kernel(omega/s, 1/s, holes)*kernel(omega, s, holes)
        np.testing.assert_allclose(roundtrip, retained, atol=2e-14, rtol=2e-14)
        for t in scales:
            lhs = kernel(omega/s, t, holes)*kernel(omega, s, holes)
            rhs = kernel(omega, s*t, holes)*retained
            error = max(error, float(np.max(np.abs(lhs-rhs))))
            np.testing.assert_allclose(lhs, rhs, atol=3e-14, rtol=3e-14)
            if not holes and (s >= 1 or t <= 1):
                np.testing.assert_allclose(lhs, kernel(omega, s*t), atol=3e-14, rtol=3e-14)
print(f"PASS: projector/composition symbols for nested and holed bands; max error {error:.3g}")

# Explicit finite periodic matrices, in both coordinates, for all four Penrose equations.
nh, nl = 17, 9
s = nl/nh
kh = np.arange(-(nh//2), nh//2+1)
kl = np.arange(-(nl//2), nl//2+1)
xh, xl = np.arange(nh)-(nh-1)/2, np.arange(nl)-(nl-1)/2
fh = np.exp(-2j*np.pi*np.outer(kh,xh)/nh)/np.sqrt(nh)
fl = np.exp(-2j*np.pi*np.outer(kl,xl)/nl)/np.sqrt(nl)
select = np.eye(nh)[nh//2-nl//2:nh//2+nl//2+1]
transfer = h(2*np.pi*kl/nl)*reciprocal(h(2*np.pi*kl/nh))
def mm(*matrices):
    result = matrices[0]
    for following in matrices[1:]:
        result = np.einsum("ij,jk->ik",result,following,optimize=False)
    return result
a = np.sqrt(s)*mm(fl.conj().T,np.diag(transfer),select,fh)
d = mm(fh.conj().T,select.T,np.diag(reciprocal(transfer)),fl)/np.sqrt(s)
assert np.isfinite(a).all() and np.isfinite(d).all()
for left, right in ((mm(a,d,a),a),(mm(d,a,d),d),
                    (mm(a,d).conj().T,mm(a,d)),(mm(d,a).conj().T,mm(d,a))):
    np.testing.assert_allclose(left,right,atol=1e-13,rtol=1e-13)
np.testing.assert_allclose(d,np.linalg.pinv(a),atol=1e-13,rtol=1e-13)
print("PASS: finite periodic paired matrices satisfy all four Penrose identities")

# The same global pair after centered finite crops need not remain a Penrose pair.
hi = np.arange(4,13)  # nine input samples
lo = np.arange(2,7)   # five output samples
acrop = a[np.ix_(lo,hi)]
dcrop = d[np.ix_(hi,lo)]
assert len(lo) >= np.ceil(s*len(hi))
crop_error = np.linalg.norm(mm(acrop,dcrop,acrop)-acrop)/np.linalg.norm(acrop)
assert crop_error > 1e-3
print(f"Finite-crop counterexample: N={len(lo)} >= ceil(s*N0)={int(np.ceil(s*len(hi)))}, "
      f"but ||ADA-A||/||A||={crop_error:.6f}")

# General integer strides: equality of all nonzero samples, allowing zero tails.
for stride_a,stride_b in ((2,2),(2,3),(3,2),(3,4)):
    signal = rng.normal(size=91)
    first_taps, second_taps = rng.normal(size=7), rng.normal(size=5)
    first = np.convolve(signal,first_taps)[::stride_a]
    chain = np.convolve(first,second_taps)[::stride_b]
    expanded = np.zeros(stride_a*(len(second_taps)-1)+1)
    expanded[::stride_a] = second_taps
    direct = np.convolve(signal,np.convolve(first_taps,expanded))[::(stride_a*stride_b)]
    length = max(len(chain),len(direct))
    np.testing.assert_allclose(np.pad(chain,(0,length-len(chain))),
                               np.pad(direct,(0,length-len(direct))),
                               atol=1e-13,rtol=1e-13)
print("PASS: zero-stuffed composition for four equal/unequal integer-stride pairs")

# Wiener/Tikhonov quotient against an independently solved circular normal system.
n = 37
w = 2*np.pi*np.fft.fftfreq(n)
source_symbol, target_symbol = h(w), h(w/1.7)
source, target = np.fft.ifft(source_symbol), np.fft.ifft(target_symbol)
projected_target = np.fft.ifft(target_symbol*(source_symbol != 0))
convolution = np.column_stack([np.roll(source,j) for j in range(n)])
adjoint = convolution.conj().T
for eta in (0.03,0.2):
    normal = mm(adjoint,convolution)+eta**2*np.eye(n)
    rhs = mm(adjoint,target[:,None])[:,0]
    solution = np.linalg.solve(normal,rhs)
    multiplier = target_symbol*np.conj(source_symbol)/(np.abs(source_symbol)**2+eta**2)
    np.testing.assert_allclose(solution,np.fft.ifft(multiplier),atol=1e-12,rtol=1e-12)
    projected_rhs = mm(adjoint,projected_target[:,None])[:,0]
    np.testing.assert_allclose(rhs,projected_rhs,atol=1e-13,rtol=1e-13)
    np.testing.assert_allclose(multiplier[source_symbol == 0],0,atol=1e-15)
print("PASS: Wiener quotient equals ridge normal solve; source projection leaves it unchanged")

# General centered grid offsets, including rounded intermediate sizes.
for n0,n1,n2,s,t in ((9,5,9,0.5,2.0),(17,12,5,0.7,0.4),(31,17,37,0.53,2.13)):
    q01 = (n0-1)/2-(n1-1)/(2*s)
    q12 = (n1-1)/2-(n2-1)/(2*t)
    q02 = (n0-1)/2-(n2-1)/(2*s*t)
    assert np.isclose(q01+q12/s,q02)
print("PASS: centered-grid composition independent of intermediate size")

# Check the published control statement against the saved result, not a new experiment.
summary = json.loads((PACKAGE/"data/hpatches_unit_kernel_full_summary.json").read_text())
for base, data in summary["bases"].items():
    ci = data["factors"]["32"]["paths"]["direct low"]["difference"]["success_2_5_output_px"]
    assert ci["low"] <= 0 <= ci["high"], (base,ci)
    print(f"HPatches direct-low x32 {base}: paired gain {100*ci['mean']:.3f} pp, "
          f"95% CI [{100*ci['low']:.3f}, {100*ci['high']:.3f}]")
