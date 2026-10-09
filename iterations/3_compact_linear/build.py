"""Build C3 (COMPACT-LINEAR): C2's exact moment deskew, then 2x2 pooling and a linear ridge model, with an optional gated row/column profile extension."""
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np

FIT_IMAGES = "/fit/fit_images.npy"
FIT_LABELS = "/fit/fit_labels.npy"
OUT_DIR = "/output"
OUT_SRC = os.path.join(OUT_DIR, "classifier.py")
OUT_PROV = os.path.join(OUT_DIR, "provenance.json")
SEED = 0
HOLD_N = 5000
CHUNK = 5000
LAM_MULTS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
QUANT_TOL = 0.003
PROFILE_GAIN = 0.005
SIZE_TARGET = 6500
SOURCE_CAP = 65536
CHECK_N = 200
FEAT_CHECK_N = 20
TIME_N = 500
WARM_N = 20

DESKEW_SRC = '''_R, _C = np.indices((28, 28), dtype=np.float32)


def _deskew(a):
    m = a.sum()
    if m <= 1e-6:
        return a
    cy = (_R * a).sum() / m
    cx = (_C * a).sum() / m
    dr = _R - cy
    vy = (dr * dr * a).sum() / m
    cxy = (dr * (_C - cx) * a).sum() / m
    al = np.float32(np.clip(cxy / vy, -1, 1)) if vy > 1e-6 else np.float32(0)
    sr = _R + (cy - np.float32(12))
    sc = _C + al * (_R - np.float32(14)) + (cx - np.float32(12))
    r0 = np.floor(sr)
    c0 = np.floor(sc)
    fr = sr - r0
    fc = sc - c0
    i = np.clip(r0, 0, 30).astype(np.intp)
    j = np.clip(c0, 0, 30).astype(np.intp)
    p = np.zeros((32, 32), dtype=np.float32)
    p[2:30, 2:30] = a
    return (((p[i, j] * (1 - fr) + p[i + 1, j] * fr) * (1 - fc)
             + (p[i, j + 1] * (1 - fr) + p[i + 1, j + 1] * fr) * fc)
            .astype(np.float32))


def _x(image):
    a = _deskew(np.asarray(image).reshape(28, 28).astype(np.float32) / np.float32(255.0))
    return a.reshape(14, 2, 14, 2).mean(axis=(1, 3), dtype=np.float32).reshape(196)
'''

PROF_SRC = '''def _xp(image):
    a = _deskew(np.asarray(image).reshape(28, 28).astype(np.float32) / np.float32(255.0))
    return np.concatenate((a.reshape(14, 2, 14, 2).mean(axis=(1, 3), dtype=np.float32).reshape(196),
                           a.sum(axis=1, dtype=np.float32) / np.float32(28.0),
                           a.sum(axis=0, dtype=np.float32) / np.float32(28.0)))
'''

TEMPLATE = '''"""Moment-deskewed linear ridge on 2x2 mean-pooled MNIST%%DOC%% (NumPy only)."""
import base64

import numpy as np


%%BODY%%

def _dec(s, dtype):
    return np.frombuffer(base64.b64decode(s), dtype=dtype)


_W = (_dec((
%%WQ%%
), "%%WDT%%").reshape(-1, 10).astype(np.float32)
      * _dec((
%%WS%%
), "<f4").astype(np.float32))
_B = _dec((
%%B%%
), "<f4").astype(np.float32)
for _a in (_R, _C, _W, _B):
    _a.setflags(write=False)
del _dec, _a


def predict(image):
    return int(np.argmax(%%FN%%(image) @ _W + _B))
'''


def pool(imgs):
    a = imgs.astype(np.float32) / np.float32(255.0)
    n = a.shape[0]
    return a.reshape(n, 14, 2, 14, 2).mean(axis=(2, 4), dtype=np.float32).reshape(n, 196)


def quantize(m, bits):
    qmax = 127 if bits == 8 else 32767
    amax = np.abs(m).max(axis=0)
    scale = np.where(amax > 0, amax / qmax, 1.0).astype(np.float32)
    q = np.clip(np.rint(m / scale.astype(np.float64)), -qmax, qmax)
    return q.astype(np.int8 if bits == 8 else np.int16), scale


def b64block(arr, dtype):
    s = base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode("ascii")
    return "\n".join('    "' + s[i:i + 76] + '"' for i in range(0, len(s), 76))


ns = {"np": np}
exec(DESKEW_SRC, ns)
exec(PROF_SRC, ns)
feat = ns["_x"]
featp = ns["_xp"]

imgs = np.load(FIT_IMAGES, mmap_mode="r")
labels = np.asarray(np.load(FIT_LABELS)).reshape(-1).astype(np.int64)
n_all = int(imgs.shape[0])
assert labels.shape[0] == n_all and labels.min() >= 0 and labels.max() <= 9
assert n_all > HOLD_N

t_feat0 = time.perf_counter()
X = np.empty((n_all, 196), dtype=np.float32)
PF = np.empty((n_all, 56), dtype=np.float32)
XR = np.empty((n_all, 196), dtype=np.float32)
for s in range(0, n_all, CHUNK):
    blk = np.asarray(imgs[s:s + CHUNK]).reshape(-1, 28, 28)
    XR[s:s + blk.shape[0]] = pool(blk)
    for t in range(blk.shape[0]):
        v = featp(blk[t])
        X[s + t] = v[:196]
        PF[s + t] = v[196:]
    del blk
feat_seconds = time.perf_counter() - t_feat0

perm = np.random.default_rng(SEED).permutation(n_all)
hold_idx = np.sort(perm[:HOLD_N])
tr_idx = np.sort(perm[HOLD_N:])
n_tr = int(tr_idx.size)
yh = labels[hold_idx]

for i in hold_idx[:FEAT_CHECK_N]:
    assert np.array_equal(feat(np.asarray(imgs[i]).reshape(28, 28)), X[i]), "build _x/_xp mismatch"


def gram(F, idx):
    d = F.shape[1] + 1
    g = np.zeros((d, d))
    b = np.zeros((d, 10))
    for s in range(0, idx.size, CHUNK):
        sel = idx[s:s + CHUNK]
        f = np.empty((sel.size, d))
        f[:, 0] = 1.0
        f[:, 1:] = F[sel]
        g += f.T @ f
        y = np.zeros((sel.size, 10))
        y[np.arange(sel.size), labels[sel]] = 1.0
        b += f.T @ y
    return g, b


def ridge(g, b, lam):
    a = g.copy()
    r = np.arange(1, g.shape[0])
    a[r, r] += lam
    return np.linalg.solve(a, b)


def fit_variant(F, mults):
    d = F.shape[1] + 1
    fh = np.ascontiguousarray(F[hold_idx])

    def acc_float(w):
        out = fh @ w[1:].astype(np.float32) + w[0].astype(np.float32)
        return float(np.mean(np.argmax(out, axis=1) == yh))

    def quant_head(w, bits):
        wq, wsc = quantize(w[1:], bits)
        return wq, wsc, wq.astype(np.float32) * wsc, w[0].astype(np.float32)

    def acc_quant(w, bits):
        _, _, wd, bias = quant_head(w, bits)
        return float(np.mean(np.argmax(fh @ wd + bias, axis=1) == yh))

    g_tr, b_tr = gram(F, tr_idx)
    base_tr = float(np.trace(g_tr)) / d / 1000.0
    table = []
    best = None
    for m in mults:
        w = ridge(g_tr, b_tr, m * base_tr)
        a = acc_float(w)
        table.append({"mult": m, "lam": m * base_tr, "holdout_acc_float": a})
        if best is None or a > best[1]:
            best = (m, a, w)
    mult, sel_float, w_sel = best
    sel_int8 = acc_quant(w_sel, 8)
    w_bits = 8 if sel_float - sel_int8 <= QUANT_TOL else 16
    sel_quant = sel_int8 if w_bits == 8 else acc_quant(w_sel, 16)

    g_h, b_h = gram(F, hold_idx)
    g_all = g_tr + g_h
    b_all = b_tr + b_h
    base_all = float(np.trace(g_all)) / d / 1000.0
    w_fin = ridge(g_all, b_all, mult * base_all)
    wq, wsc, wd, bias = quant_head(w_fin, w_bits)
    pred_q = np.argmax(fh @ wd + bias, axis=1)
    return {
        "d": d, "table": table, "mult": mult, "lam_sel_45k": mult * base_tr,
        "lam_final_50k": mult * base_all, "sel_float": sel_float, "sel_int8": sel_int8,
        "sel_quant": sel_quant, "w_bits": w_bits, "wq": wq, "wsc": wsc, "bias": bias,
        "fin_float": acc_float(w_fin), "fin_quant": float(np.mean(pred_q == yh)),
        "hold_pred_quant": pred_q,
    }


def make_source(fit, prof):
    wdt = "<i1" if fit["w_bits"] == 8 else "<i2"
    src = TEMPLATE
    for key, val in (
        ("%%DOC%%", " plus deskewed row/column profiles" if prof else ""),
        ("%%WQ%%", b64block(fit["wq"], wdt)),
        ("%%WDT%%", wdt),
        ("%%WS%%", b64block(fit["wsc"], "<f4")),
        ("%%B%%", b64block(fit["bias"], "<f4")),
        ("%%FN%%", "_xp" if prof else "_x"),
        ("%%BODY%%", DESKEW_SRC + ("\n\n" + PROF_SRC if prof else "")),
    ):
        src = src.replace(key, val)
    assert "%%" not in src
    assert DESKEW_SRC in src
    if prof:
        assert PROF_SRC in src
    return src


base = fit_variant(X, LAM_MULTS)
base_src = make_source(base, False)
base_bytes = len(base_src.encode("utf-8"))

XP = np.concatenate((X, PF), axis=1)
prof = fit_variant(XP, LAM_MULTS)
prof_src = make_source(prof, True)
prof_bytes = len(prof_src.encode("utf-8"))

gain = prof["sel_float"] - base["sel_float"]
gate_gain = gain >= PROFILE_GAIN - 1e-9
gate_size = prof_bytes <= SIZE_TARGET
use_prof = bool(gate_gain and gate_size)
model = prof if use_prof else base
src = prof_src if use_prof else base_src
FE = XP if use_prof else X
src_bytes = len(src.encode("utf-8"))
assert src_bytes <= SOURCE_CAP

g_r, b_r = gram(XR, tr_idx)
w_r = ridge(g_r, b_r, base["mult"] * float(np.trace(g_r)) / 197 / 1000.0)
out_r = XR[hold_idx] @ w_r[1:].astype(np.float32) + w_r[0].astype(np.float32)
abl_float = float(np.mean(np.argmax(out_r, axis=1) == yh))
del XR, g_r

with open(OUT_SRC, "w", encoding="utf-8", newline="\n") as fh_out:
    fh_out.write(src)

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("c3_classifier_selfcheck", OUT_SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def ro_image(i):
    im = np.array(imgs[i], dtype=np.uint8).reshape(28, 28)
    im.setflags(write=False)
    return im


for i in hold_idx[:FEAT_CHECK_N]:
    assert np.array_equal(mod._x(ro_image(i)), X[i]), "deskew feature mismatch"
    if use_prof:
        assert np.array_equal(mod._xp(ro_image(i)), XP[i]), "profile feature mismatch"

got = []
for i in hold_idx[:CHECK_N]:
    r = mod.predict(ro_image(i))
    assert type(r) is int and 0 <= r <= 9
    got.append(r)
expected = model["hold_pred_quant"][:CHECK_N].astype(np.int64)
got_arr = np.array(got, dtype=np.int64)
assert np.array_equal(got_arr, expected), "self-check mismatch"

time_imgs = [ro_image(i) for i in hold_idx[:TIME_N]]
for im in time_imgs[:WARM_N]:
    mod.predict(im)
pt = []
for im in time_imgs:
    t0 = time.perf_counter()
    mod.predict(im)
    pt.append(time.perf_counter() - t0)
dt = []
for im in time_imgs:
    a = im.astype(np.float32) / np.float32(255.0)
    t0 = time.perf_counter()
    mod._deskew(a)
    dt.append(time.perf_counter() - t0)
pyc = os.path.join(OUT_DIR, "__pycache__")
if os.path.isdir(pyc):
    shutil.rmtree(pyc)


def table_out(t):
    return [{kk: (round(v, 8) if isinstance(v, float) else v) for kk, v in r.items()} for r in t]


deskew_bytes = len(DESKEW_SRC.encode("utf-8"))
prov = {
    "candidate_id": "C3",
    "decision_id": "D3",
    "batch_id": "B3",
    "direction_id": "COMPACT-LINEAR",
    "parents": [],
    "deskew_text_origin": "byte-for-byte copy of C2 DESKEW_SRC (historical reference, not a parent)",
    "deskew": {"centre": 14, "alpha": "cov(r,c)/var(r), 0 if var<=1e-6, clipped to [-1,1]", "empty_image": "m<=1e-6 returned unchanged",
               "sampling": "bilinear, 2-pixel zero pad, clipped floor indices [0,30], zero fill"},
    "variant_emitted": "profile_252" if use_prof else "base_196",
    "profile_gate": {"gain_float": round(gain, 6), "required_gain": PROFILE_GAIN, "gain_ok": bool(gate_gain),
                     "size_ok": bool(gate_size), "adopted": use_prof},
    "variant_source_bytes": {"base_196": base_bytes, "profile_252": prof_bytes},
    "size_target_bytes": SIZE_TARGET,
    "size_target_met": src_bytes <= SIZE_TARGET,
    "lam_grid_mults": LAM_MULTS,
    "lam_grid_rule": "mult * trace(G)/d/1000, bias row unpenalised, first max wins",
    "lam_table_base_45k": table_out(base["table"]),
    "lam_table_profile_45k": table_out(prof["table"]),
    "selected_mult_base": base["mult"],
    "selected_mult_profile": prof["mult"],
    "selected_mult": model["mult"],
    "selected_lam_45k": model["lam_sel_45k"],
    "final_lam_50k": model["lam_final_50k"],
    "holdout_acc_float_selection": round(model["sel_float"], 6),
    "holdout_acc_int8_selection": round(model["sel_int8"], 6),
    "holdout_acc_quant_selection": round(model["sel_quant"], 6),
    "base_holdout_acc_float_selection": round(base["sel_float"], 6),
    "profile_holdout_acc_float_selection": round(prof["sel_float"], 6),
    "ablation_no_deskew_holdout_acc_float": round(abl_float, 6),
    "ablation_rule": "identical base pipeline on non-deskewed pooled pixels, same 45000 rows, base selected mult; not emitted",
    "final_holdout_acc_float_in_sample": round(model["fin_float"], 6),
    "final_holdout_acc_quant_in_sample": round(model["fin_quant"], 6),
    "quant_tolerance_pp": QUANT_TOL * 100,
    "dtypes": {"W": ("int8" if model["w_bits"] == 8 else "int16") + " per-class float32 scale",
               "bias": "float32 (W bias row kept unquantized)"},
    "feature_dims_including_bias": model["d"],
    "source_bytes": src_bytes,
    "source_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
    "deskew_bytes": deskew_bytes,
    "deskew_share_of_source": round(deskew_bytes / src_bytes, 6),
    "fit_rows": {"total": n_all, "selection_train": n_tr, "internal_holdout": HOLD_N, "final_ridge": n_all},
    "holdout_seed": SEED,
    "in_build_timing_descriptive": {"predict_median_ms": round(float(np.median(pt)) * 1000.0, 5),
                                    "deskew_median_ms": round(float(np.median(dt)) * 1000.0, 5),
                                    "n": len(time_imgs), "warmup": WARM_N, "feature_build_seconds": round(feat_seconds, 3),
                                    "note": "Runner in-process timing; excludes IPC; no fallback depends on it"},
    "self_check": {"n": CHECK_N, "matches": int(np.sum(got_arr == expected)), "exact": True,
                   "feature_bit_exact_n": FEAT_CHECK_N},
    "template_support": "not applicable; no exemplars, templates or means stored",
    "numpy_version": np.__version__,
}
with open(OUT_PROV, "w", encoding="utf-8") as fp:
    json.dump(prov, fp, indent=1, sort_keys=True)
