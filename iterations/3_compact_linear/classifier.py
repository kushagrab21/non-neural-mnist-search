"""Moment-deskewed linear ridge on 2x2 mean-pooled MNIST (NumPy only)."""
import base64

import numpy as np


_R, _C = np.indices((28, 28), dtype=np.float32)


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


def _dec(s, dtype):
    return np.frombuffer(base64.b64decode(s), dtype=dtype)


_W = (_dec((
    "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"
    "70MM6ukqIQDlBfL5/wb7AQgL++sFDA=="
), "<i1").reshape(-1, 10).astype(np.float32)
      * _dec((
    "WLIJPKWBJjzx3Vs8jDUdPOQTbTyT/AE8GGJDPL2pejzsSCA8bjI0PA=="
), "<f4").astype(np.float32))
_B = _dec((
    "BKvXPfJJ1D37qg8+myf9PIS7QT7oHYE+Q36PPbZxDD6Fh4+8MRo4vA=="
), "<f4").astype(np.float32)
for _a in (_R, _C, _W, _B):
    _a.setflags(write=False)
del _dec, _a


def predict(image):
    return int(np.argmax(_x(image) @ _W + _B))
