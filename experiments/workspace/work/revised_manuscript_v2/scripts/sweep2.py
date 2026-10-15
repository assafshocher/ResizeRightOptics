import numpy as np, glob, os, sys, json
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
def load(f): return np.asarray(Image.open(f), dtype=np.float64)/255.0
def design(HR, LR, a, R, ys, xs):
    cols = [HR[np.ix_(a*ys - dy, a*xs - dx)].reshape(-1, HR.shape[2]).T.ravel() for dy in range(-R, R+1) for dx in range(-R, R+1)]
    return np.stack(cols, 1), LR[np.ix_(ys, xs)].reshape(-1, LR.shape[2]).T.ravel()
def fit_ridge(HR, LR, a, R, region, lam):
    A, b = design(HR, LR, a, R, *region); k = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@b); return k.reshape(2*R+1, 2*R+1)
def score(HR, LR, a, k, region):
    A, b = design(HR, LR, a, k.shape[0]//2, *region); return np.sqrt(np.mean((A@k.ravel()-b)**2))
def centroid(k):
    R = k.shape[0]//2; g = np.arange(-R, R+1); return (k.sum(1)@g)/k.sum(), (k.sum(0)@g)/k.sum()
def dtft2(k, cy, cx, w):
    R = k.shape[0]//2; g = np.arange(-R, R+1); return np.exp(-1j*np.outer(w, g-cy)) @ k @ np.exp(-1j*np.outer(w, g-cx)).T
def taps(F, w, dw, R, shift):
    g = np.arange(-R, R+1); return np.real(np.exp(1j*np.outer(g-shift[0], w)) @ F @ np.exp(1j*np.outer(g-shift[1], w)).T)*dw*dw/(2*np.pi)**2
def regions(LR, R, n=120, m2=180):
    H, W = LR.shape[:2]; m = R+2
    return (np.arange(m, m+n), np.arange(m, m+m2)), (np.arange(H-m-n, H-m), np.arange(W-m-m2, W-m))
n = 257; w = np.linspace(-np.pi, np.pi, n); dw = w[1]-w[0]; WX, WY = np.meshgrid(w, w)
R = {2: 7, 3: 9, 4: 11}
def interp(F):
    fr = RegularGridInterpolator((w, w), F.real, bounds_error=False, fill_value=0.0); fi = RegularGridInterpolator((w, w), F.imag, bounds_error=False, fill_value=0.0)
    return lambda wx, wy: (fr(np.stack([wy.ravel(), wx.ravel()], 1)) + 1j*fi(np.stack([wy.ravel(), wx.ravel()], 1))).reshape(wx.shape)
results = []
cam = sys.argv[3] if len(sys.argv) > 3 else "Canon"
scenes = sorted(set(os.path.basename(f)[:9] for f in glob.glob("RealSR(V3)/%s/Test/2/*_HR.png" % cam)))
def predict(HR, a, k, region):
    ys, xs = region; R_ = k.shape[0]//2; out = np.zeros((len(ys), len(xs), HR.shape[2]))
    for i, dy in enumerate(range(-R_, R_+1)):
        for j, dx in enumerate(range(-R_, R_+1)): out += k[i, j]*HR[np.ix_(a*ys - dy, a*xs - dx)]
    return out
def radial_logspec(img):
    g = img.mean(2); g = g - g.mean(); F = np.abs(np.fft.fftshift(np.fft.fft2(g*np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])))))**2
    cy, cx = np.array(F.shape)//2; yy, xx = np.indices(F.shape); r = np.hypot(yy-cy, xx-cx)
    nb = min(cy, cx); prof = np.array([F[(r >= i) & (r < i+1)].mean() for i in range(1, nb)]); return np.log10(prof + 1e-12)
def spec_dist(HR, LR, a, k, region):
    ys, xs = region; return np.sqrt(np.mean((radial_logspec(predict(HR, a, k, region)) - radial_logspec(LR[np.ix_(ys, xs)]))**2))
for sc in scenes[int(sys.argv[1]):int(sys.argv[2])]:
    try:
        d = {}
        for a in (2, 3, 4):
            HR = load("RealSR(V3)/%s/Test/%d/%s_HR.png" % (cam, a, sc)); LR = load("RealSR(V3)/%s/Test/%d/%s_LR%d.png" % (cam, a, sc, a))
            fit, val = regions(LR, R[a]); best = None
            for lam in [1e-1, 1]:
                k = fit_ridge(HR, LR, a, R[a], fit, lam); sc_ = score(HR, LR, a, k, val)
                if best is None or sc_ < best[0]: best = (sc_, k)
            d[a] = dict(HR=HR, LR=LR, k=best[1], val=val, oracle=best[0])
        k2 = d[2]["k"]; cy2, cx2 = centroid(k2); K12 = interp(dtft2(k2/k2.sum(), cy2, cx2, w))
        K0 = np.ones_like(WX, dtype=complex)
        for j in range(1, 12): K0 = K0*K12(WX/2**j, WY/2**j)
        row = dict(scene=sc, oracle2=d[2]["oracle"])
        for a in (3, 4):
            K0a = np.ones_like(WX, dtype=complex)
            for j in range(1, 12): K0a = K0a*K12(a*WX/2**j, a*WY/2**j)
            ok = (np.abs(a*WX) <= np.pi) & (np.abs(a*WY) <= np.pi) & (np.abs(K0) > 1e-2)
            Kfam = np.where(ok, K0a/np.where(ok, K0, 1), 0.0); Kstr = K12(a*WX/2, a*WY/2)
            k = d[a]["k"]; c = centroid(k)
            kf = taps(Kfam, w, dw, R[a], c); kf *= k.sum()/kf.sum(); ks = taps(Kstr, w, dw, R[a], c); ks *= k.sum()/ks.sum()
            row["oracle%d" % a] = d[a]["oracle"]; row["fam%d" % a] = score(d[a]["HR"], d[a]["LR"], a, kf, d[a]["val"]); row["str%d" % a] = score(d[a]["HR"], d[a]["LR"], a, ks, d[a]["val"])
            row["kd_fam%d" % a] = np.linalg.norm(kf-k)/np.linalg.norm(k); row["kd_str%d" % a] = np.linalg.norm(ks-k)/np.linalg.norm(k)
            row["sp_or%d" % a] = spec_dist(d[a]["HR"], d[a]["LR"], a, k, d[a]["val"]); row["sp_fam%d" % a] = spec_dist(d[a]["HR"], d[a]["LR"], a, kf, d[a]["val"]); row["sp_str%d" % a] = spec_dist(d[a]["HR"], d[a]["LR"], a, ks, d[a]["val"])
        open("sweep_%s.jsonl" % cam.lower(), "a").write(json.dumps(row) + "\n"); print("%s  x3: oracle %.4f fam %.4f str %.4f | x4: oracle %.4f fam %.4f str %.4f" % (sc, row["oracle3"], row["fam3"], row["str3"], row["oracle4"], row["fam4"], row["str4"]), flush=True)
    except Exception as e: print(sc, "failed:", e, flush=True)

