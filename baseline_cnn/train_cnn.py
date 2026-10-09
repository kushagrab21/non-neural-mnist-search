"""Small CNN baseline (the neural comparison slide 67 asks for). Outside the search budget; its human design effort is disclosed, not counted.

Usage: baselines/.venv/bin/python baselines/cnn/train_cnn.py <development_dir> <out_dir>
  1. trains on folds 1-5 and scores fold 0 (the same development comparison the candidates get);
  2. refits on all 60k with the same recipe and saves the weights for the single sealed final assessment.
The test set is never touched here.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

SEED, EPOCHS, BATCH, LR = 20261007, 6, 128, 1e-3

class SmallCNN(nn.Module):
    """LeNet-style: two 5x5 conv blocks and two dense layers (about 65k parameters)."""
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(16, 32, 5), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                               nn.Linear(32 * 5 * 5, 64), nn.ReLU(), nn.Linear(64, 10))
    def forward(self, x):
        return self.f(x)

def to_t(X):
    return torch.from_numpy(X.astype(np.float32) / 255.0).unsqueeze(1)

def fit(X, y, dev):
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = SmallCNN().to(dev); opt = torch.optim.Adam(m.parameters(), lr=LR)
    Xt, yt = to_t(X), torch.from_numpy(y.astype(np.int64))
    g = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        m.train()
        for idx in torch.randperm(len(Xt), generator=g).split(BATCH):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(m(Xt[idx].to(dev)), yt[idx].to(dev))
            loss.backward(); opt.step()
    return m.cpu().eval()

def per_image_ms(m, X, n=1000):
    torch.set_num_threads(1)
    xs = to_t(X[:n]); times = []
    with torch.no_grad():
        for i in range(50):
            m(xs[i:i + 1])
        for i in range(n):
            t = time.perf_counter(); m(xs[i:i + 1]); times.append(time.perf_counter() - t)
    return float(np.median(times) * 1e3), float(np.percentile(times, 90) * 1e3)

def main(devdir, out):
    devdir, out = Path(devdir), Path(out); out.mkdir(parents=True, exist_ok=True)
    X = np.load(devdir / "train_images.npy"); y = np.load(devdir / "train_labels.npy"); folds = np.load(devdir / "folds.npy")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tr, va = folds != 0, folds == 0
    t = time.time(); m = fit(X[tr], y[tr], dev); train_s = time.time() - t
    with torch.no_grad():
        pred = m(to_t(X[va])).argmax(1).numpy()
    acc = float((pred == y[va]).mean())
    np.save(out / "cnn_fold0_preds.npy", pred)
    torch.save(m.state_dict(), out / "cnn_fold0.pt")
    med, p90 = per_image_ms(m, X[va])
    t = time.time(); mall = fit(X, y, dev); all_s = time.time() - t
    torch.save(mall.state_dict(), out / "cnn_all60k.pt")
    n_params = sum(p.numel() for p in m.parameters())
    res = {"model": "SmallCNN (LeNet-style)", "params": n_params,
           "size_bytes_state_dict": (out / "cnn_fold0.pt").stat().st_size,
           "size_bytes_float32_weights": n_params * 4,
           "fold0_accuracy": round(acc, 5), "fold0_errors": int((pred != y[va]).sum()),
           "per_image_median_ms_cpu_1thread": round(med, 4), "per_image_p90_ms_cpu_1thread": round(p90, 4),
           "train_seconds_folds1to5": round(train_s, 1), "train_seconds_all60k": round(all_s, 1), "train_device": dev,
           "recipe": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "optimizer": "Adam", "seed": SEED, "augmentation": "none"},
           "torch": torch.__version__,
           "uncounted_design_effort": "Architecture and recipe are standard human designs (LeNet lineage, Adam defaults). Decades of human design effort are not in any budget; disclose this next to the comparison.",
           "final_gate": "cnn_all60k.pt is evaluated once on the sealed test and QMNIST50K, at the same gate and with the same rules as the frozen candidates."}
    (out / "CNN_BASELINE.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
