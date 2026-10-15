"""Numerical kernel diagnostics only; no image labels or PSNR are used."""
import json
from pathlib import Path
import numpy as np
from literal_spectrum import full_cubic_spectrum, full_camera


def main():
    # Candidate ridge is FIXED before inspecting any reconstruction score.
    eta = .001
    density, length = 512, 1 << 20
    omega_all = 2*np.pi*np.fft.rfftfreq(length, d=1/density)
    keep = omega_all <= 32*np.pi
    w = omega_all[keep]
    report = {"eta": eta, "definition": "Tikhonov toward literal AA, exact anchor",
              "uses_images": False, "rows": []}
    for anchor in (2, 3, 4):
        h = full_camera(w, anchor)
        for m in range(1, 11):
            r = anchor**(1/m)
            u = full_cubic_spectrum(r*w)
            target = full_camera(r*w, anchor)
            # Correction to AA; minimizer of ||H*K-H_r||²+eta²||K-U_r||².
            correction = (target-u*h)*h/(h*h+eta*eta)
            if m == 1:
                assert np.max(np.abs(correction)) < 1e-10
                correction[:] = 0.
            spec = np.zeros_like(omega_all)
            spec[keep] = correction
            spatial = np.fft.irfft(spec, n=length)*density
            radius = 32*anchor
            tail = 2*np.sum(np.abs(spatial[radius*density:length//2]))/density
            band = w <= np.pi
            exact = target[band]/h[band]
            error = np.max(np.abs((u+correction)[band]-exact))
            row = dict(anchor=anchor, m=m, max_spectrum=float(np.max(np.abs(u+correction))),
                       spatial_l1_tail_at_radius=float(tail),
                       correction_near_frequency_cutoff=float(np.max(np.abs(correction[-100:]))),
                       max_inband_ridge_bias=float(error))
            report["rows"].append(row)
    path = Path(__file__).parent / "literal_spectrum_diagnostic.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
