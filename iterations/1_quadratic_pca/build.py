"""Build C1 (QUAD-PCA): degree-2 ridge on a whitened PCA projection of 2x2-pooled MNIST."""
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import sys

import numpy as np

FIT_IMAGES = "/fit/fit_images.npy"
FIT_LABELS = "/fit/fit_labels.npy"
OUT_DIR = "/output"
OUT_SRC = os.path.join(OUT_DIR, "classifier.py")
OUT_PROV = os.path.join(OUT_DIR, "provenance.json")
SEED = 0
HOLD_N = 5000
STAT_CHUNK = 5000
FEAT_CHUNK = 2500
K_PRIMARY = 40
K_FALLBACK = 30
LAM_MULTS = [0.1, 1.0, 10.0, 100.0, 1000.0]
QUANT_TOL = 0.003
SIZE_GUARD = 40000
SOURCE_CAP = 65536
CHECK_N = 200

TEMPLATE = '''"""Degree-2 polynomial ridge on a whitened %%K%%-dim PCA projection of 2x2 mean-pooled MNIST (NumPy only)."""
import base64

import numpy as np

_K = %%K%%


def _dec(s, dtype):
    return np.frombuffer(base64.b64decode(s), dtype=dtype)


_MU = _dec((
%%MU%%
), "<f4").astype(np.float32)
_S = _dec((
%%S%%
), "<f4").astype(np.float32)
_P = (_dec((
%%PQ%%
), "<i1").reshape(196, _K).astype(np.float32)
      * _dec((
%%PS%%
), "<f4").astype(np.float32))
_W = (_dec((
%%WQ%%
), "%%WDT%%").reshape(-1, 10).astype(np.float32)
      * _dec((
%%WS%%
), "<f4").astype(np.float32))
_B = _dec((
%%B%%
), "<f4").astype(np.float32)
_I, _J = np.triu_indices(_K)
_F = np.empty(_K + _I.size, dtype=np.float32)
_Z = _F[:_K]
_Q = _F[_K:]
for _a in (_MU, _S, _P, _W, _B):
    _a.setflags(write=False)
del _dec, _a


def predict(image):
    a = np.asarray(image).reshape(28, 28).astype(np.float32) / np.float32(255.0)
    p = a.reshape(14, 2, 14, 2).mean(axis=(1, 3), dtype=np.float32).reshape(196)
    np.multiply((p - _MU) @ _P, _S, out=_Z)
    np.multiply(_Z[_I], _Z[_J], out=_Q)
    return int(np.argmax(_F @ _W + _B))
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


imgs = np.load(FIT_IMAGES, mmap_mode="r")
labels = np.asarray(np.load(FIT_LABELS)).reshape(-1).astype(np.int64)
n_all = int(imgs.shape[0])
assert labels.shape[0] == n_all and labels.min() >= 0 and labels.max() <= 9
assert n_all > HOLD_N

X = np.empty((n_all, 196), dtype=np.float32)
for s in range(0, n_all, STAT_CHUNK):
    X[s:s + STAT_CHUNK] = pool(np.asarray(imgs[s:s + STAT_CHUNK]).reshape(-1, 28, 28))

perm = np.random.default_rng(SEED).permutation(n_all)
hold_idx = np.sort(perm[:HOLD_N])
tr_idx = np.sort(perm[HOLD_N:])
n_tr = int(tr_idx.size)
yh = labels[hold_idx]

s1 = np.zeros(196)
for s in range(0, n_tr, STAT_CHUNK):
    s1 += X[tr_idx[s:s + STAT_CHUNK]].astype(np.float64).sum(axis=0)
mean64 = s1 / n_tr
cov = np.zeros((196, 196))
for s in range(0, n_tr, STAT_CHUNK):
    xc = X[tr_idx[s:s + STAT_CHUNK]].astype(np.float64) - mean64
    cov += xc.T @ xc
cov /= n_tr
evals, evecs = np.linalg.eigh(cov)
order = np.argsort(evals)[::-1]
evals = evals[order]
evecs = evecs[:, order]
flip = np.sign(evecs[np.abs(evecs).argmax(axis=0), np.arange(196)])
flip[flip == 0] = 1.0
evecs = evecs * flip


def fit_model(k):
    assert evals[k - 1] > 0
    Pq, Ps = quantize(evecs[:, :k], 8)
    Pd = Pq.astype(np.float32) * Ps
    mu = mean64.astype(np.float32)
    ws = (1.0 / np.sqrt(evals[:k])).astype(np.float32)
    iu, ju = np.triu_indices(k)
    d = 1 + k + int(iu.size)

    def feats(xc):
        z = ((xc - mu) @ Pd) * ws
        f = np.empty((xc.shape[0], d), dtype=np.float32)
        f[:, 0] = 1.0
        f[:, 1:k + 1] = z
        np.multiply(z[:, iu], z[:, ju], out=f[:, k + 1:])
        return f

    def gram(idx):
        g = np.zeros((d, d))
        b = np.zeros((d, 10))
        for s in range(0, idx.size, FEAT_CHUNK):
            sel = idx[s:s + FEAT_CHUNK]
            f = feats(X[sel]).astype(np.float64)
            g += f.T @ f
            y = np.zeros((sel.size, 10))
            y[np.arange(sel.size), labels[sel]] = 1.0
            b += f.T @ y
        return g, b

    def ridge(g, b, lam):
        a = g.copy()
        r = np.arange(1, d)
        a[r, r] += lam
        return np.linalg.solve(a, b)

    fh = np.ascontiguousarray(feats(X[hold_idx])[:, 1:])

    def acc_float(w):
        out = fh @ w[1:].astype(np.float32) + w[0].astype(np.float32)
        return float(np.mean(np.argmax(out, axis=1) == yh))

    def quant_head(w, bits):
        wq, wsc = quantize(w[1:], bits)
        wd = wq.astype(np.float32) * wsc
        return wq, wsc, wd, w[0].astype(np.float32)

    def acc_quant(w, bits):
        _, _, wd, bias = quant_head(w, bits)
        return float(np.mean(np.argmax(fh @ wd + bias, axis=1) == yh))

    g_tr, b_tr = gram(tr_idx)
    base_tr = float(np.trace(g_tr)) / d / 1000.0
    lam_table = []
    best = None
    for m in LAM_MULTS:
        w = ridge(g_tr, b_tr, m * base_tr)
        a = acc_float(w)
        lam_table.append({"mult": m, "lam": m * base_tr, "holdout_acc_float": a})
        if best is None or a > best[1]:
            best = (m, a, w)
    mult, sel_float, w_sel = best
    sel_int8 = acc_quant(w_sel, 8)
    w_bits = 8 if sel_float - sel_int8 <= QUANT_TOL else 16
    sel_quant = sel_int8 if w_bits == 8 else acc_quant(w_sel, 16)

    g_h, b_h = gram(hold_idx)
    g_all = g_tr + g_h
    b_all = b_tr + b_h
    del g_tr, g_h
    base_all = float(np.trace(g_all)) / d / 1000.0
    w_fin = ridge(g_all, b_all, mult * base_all)
    wq, wsc, wd, bias = quant_head(w_fin, w_bits)
    fin_float = acc_float(w_fin)
    pred_q = np.argmax(fh @ wd + bias, axis=1)
    fin_quant = float(np.mean(pred_q == yh))

    src = TEMPLATE
    for key, val in (
        ("%%MU%%", b64block(mu, "<f4")),
        ("%%S%%", b64block(ws, "<f4")),
        ("%%PQ%%", b64block(Pq, "<i1")),
        ("%%PS%%", b64block(Ps, "<f4")),
        ("%%WQ%%", b64block(wq, "<i1" if w_bits == 8 else "<i2")),
        ("%%WDT%%", "<i1" if w_bits == 8 else "<i2"),
        ("%%WS%%", b64block(wsc, "<f4")),
        ("%%B%%", b64block(bias, "<f4")),
        ("%%K%%", str(k)),
    ):
        src = src.replace(key, val)
    assert "%%" not in src
    src_bytes = len(src.encode("utf-8"))

    return {
        "k": k, "d": d, "src": src, "source_bytes": src_bytes, "w_bits": w_bits,
        "lam_table": lam_table, "mult": mult, "lam_sel_45k": mult * base_tr,
        "lam_final_50k": mult * base_all, "sel_float": sel_float, "sel_int8": sel_int8,
        "sel_quant": sel_quant, "fin_float": fin_float, "fin_quant": fin_quant,
        "hold_pred_quant": pred_q,
    }


guard = False
model = fit_model(K_PRIMARY)
k40_bytes = model["source_bytes"]
if model["source_bytes"] > SIZE_GUARD:
    guard = True
    model = fit_model(K_FALLBACK)
assert model["source_bytes"] <= SOURCE_CAP

src = model["src"]
with open(OUT_SRC, "w", encoding="utf-8", newline="\n") as fh_out:
    fh_out.write(src)

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("c1_classifier_selfcheck", OUT_SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
got = []
for i in hold_idx[:CHECK_N]:
    im = np.array(imgs[i], dtype=np.uint8).reshape(28, 28)
    im.setflags(write=False)
    r = mod.predict(im)
    assert type(r) is int and 0 <= r <= 9
    got.append(r)
expected = model["hold_pred_quant"][:CHECK_N]
assert np.array_equal(np.array(got, dtype=np.int64), expected.astype(np.int64)), "self-check mismatch"
pyc = os.path.join(OUT_DIR, "__pycache__")
if os.path.isdir(pyc):
    shutil.rmtree(pyc)

prov = {
    "candidate_id": "C1",
    "decision_id": "D1",
    "batch_id": "B1",
    "direction_id": "QUAD-PCA",
    "features": "2x2 mean-pooled float32/255 (196) -> centred, int8 PCA projection, whitened -> [1, z, z_i*z_j i<=j]",
    "k": model["k"],
    "k_requested": K_PRIMARY,
    "size_guard_triggered": guard,
    "k40_source_bytes": k40_bytes,
    "feature_dims_including_bias": model["d"],
    "whitened": True,
    "projection_quantized_before_fit": True,
    "mean_and_P_source": "fit on 45000 train rows; not refit on 50000",
    "lam_grid_mults": LAM_MULTS,
    "lam_grid_rule": "mult * trace(G)/d/1000, bias row unpenalised, first max wins",
    "lam_table_45k": [{kk: (round(v, 8) if isinstance(v, float) else v) for kk, v in r.items()} for r in model["lam_table"]],
    "selected_mult": model["mult"],
    "selected_lam_45k": model["lam_sel_45k"],
    "final_lam_50k": model["lam_final_50k"],
    "holdout_acc_float_selection": round(model["sel_float"], 6),
    "holdout_acc_int8_selection": round(model["sel_int8"], 6),
    "holdout_acc_quant_selection": round(model["sel_quant"], 6),
    "final_holdout_acc_float_in_sample": round(model["fin_float"], 6),
    "final_holdout_acc_quant_in_sample": round(model["fin_quant"], 6),
    "quant_tolerance_pp": QUANT_TOL * 100,
    "dtypes": {"P": "int8 per-column float32 scale", "W": ("int8" if model["w_bits"] == 8 else "int16") + " per-class float32 scale",
               "mean": "float32", "whiten": "float32", "bias": "float32 (W bias row kept unquantized)"},
    "source_bytes": len(src.encode("utf-8")),
    "source_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
    "fit_rows": {"total": n_all, "projection_and_selection": n_tr, "internal_holdout": HOLD_N, "final_ridge": n_all},
    "holdout_seed": SEED,
    "self_check": {"n": CHECK_N, "matches": int(np.sum(np.array(got) == expected)), "exact": True},
    "template_support": "not applicable; no exemplars or templates stored",
    "numpy_version": np.__version__,
}
with open(OUT_PROV, "w", encoding="utf-8") as fp:
    json.dump(prov, fp, indent=1, sort_keys=True)
