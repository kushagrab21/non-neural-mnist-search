"""Single-image streaming worker. Never receives labels, data paths or a batch.

Child timing is diagnostic; authoritative elapsed time belongs to the parent.
No CLI here trains or loads data. CNN mode requires an explicitly supplied weight file.
"""
import base64
import contextlib
import json
import resource
import runpy
import sys
import time

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
source, mode, package_root, weights = sys.argv[1:5]
if package_root != "-":
    sys.path.insert(0, package_root)
import numpy as np


def load_predictor():
    if mode == "candidate":
        return runpy.run_path(source)["predict"]
    if mode != "cnn_baseline":
        raise ValueError("unknown inference mode")
    # The architecture is source, not a training entry point. No build/fit is run.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    namespace = runpy.run_path(source)
    model = namespace["SmallCNN"]().cpu().eval()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))

    def predict(image):
        with torch.inference_mode():
            # Conversion, normalization, forward and digit extraction share the
            # candidate raw uint8-image -> integer boundary.
            tensor = torch.from_numpy(image.astype(np.float32) / 255.0).reshape(1, 1, 28, 28)
            return int(model(tensor).argmax(dim=1).item())
    return predict


try:
    started = time.perf_counter_ns()
    with contextlib.redirect_stdout(sys.stderr):
        predict = load_predictor()
    print(json.dumps({"status": "ready", "numpy_version": np.__version__,
                      "child_load_ms": (time.perf_counter_ns()-started)/1e6}), flush=True)
except BaseException as exc:
    print(json.dumps({"status": "load_error", "error_type": type(exc).__name__, "error": str(exc)[:1000]}), flush=True)
    raise SystemExit(1)

while True:
    line = sys.stdin.buffer.readline(4097)
    if not line:
        break
    request = {}
    try:
        if len(line) > 4096 or not line.endswith(b"\n"):
            raise ValueError("invalid request framing")
        request = json.loads(line)
        raw = base64.b64decode(request["image_b64"], validate=True)
        if len(raw) != 784:
            raise ValueError("exactly one uint8 28x28 image required")
        # Immutable bytes backing prevents writeable=True from restoring writes.
        image = np.frombuffer(raw, dtype=np.uint8).reshape(28, 28)
        started = time.perf_counter_ns()
        with contextlib.redirect_stdout(sys.stderr):
            digit = predict(image)
        elapsed = (time.perf_counter_ns()-started)/1e6
        if type(digit) is not int or not 0 <= digit <= 9:
            response = {"nonce":request["nonce"], "status":"invalid_output"}
        else:
            response = {"nonce": request["nonce"], "status": "ok", "digit": digit, "child_predict_ms": elapsed}
    except BaseException as exc:
        response = {"nonce": request.get("nonce"), "status": "candidate_error",
                    "error_type": type(exc).__name__, "error": str(exc)[:1000]}
    print(json.dumps(response, allow_nan=False), flush=True)
