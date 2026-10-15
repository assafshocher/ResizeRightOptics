"""Full (unprojected) Keys a=-1/2 spectrum and its formal telescoped camera."""
import numpy as np


def full_cubic_spectrum(omega):
    w = np.asarray(omega, dtype=np.float64)
    z = w*w
    small = np.abs(w) < .08
    safe = np.where(small, 1., w)
    raw = 2*(-2*safe*np.sin(safe) + safe*np.sin(2*safe)
              -12*np.cos(safe)+3*np.cos(2*safe)+9)/safe**4
    series = 1-z*z/80+17*z**3/15120-31*z**4/604800
    return np.where(small, series, raw)


def full_camera(omega, anchor):
    result = np.ones_like(omega, dtype=np.float64)
    for j in range(18):
        result *= full_cubic_spectrum(omega / anchor**j)
    return result


def anchored_ridge_spectrum(omega, factor, anchor, eta=.001):
    """AA + H*(H_r-AA*H)/(H²+eta²), not a band-limited projection.

This uniquely minimizes |H*K-H_r|²+eta²|K-AA|² frequency by frequency.
The AA-centered ridge preserves the literal anchor: H_S=U_S*H implies
zero correction at factor S for ANY eta. No exact sampled composition is
claimed under aliasing/regularization. eta is fixed before SR scores.
"""
    h = full_camera(omega, anchor)
    aa = full_cubic_spectrum(factor*omega)
    target = full_camera(factor*omega, anchor)
    correction = (target-aa*h)*h/(h*h+eta*eta)
    if abs(factor-anchor) < 1e-12:
        if np.max(np.abs(correction)) > 1e-8:
            raise ArithmeticError("The literal anchor did not telescope")
        correction = np.zeros_like(correction)
    return aa, correction
