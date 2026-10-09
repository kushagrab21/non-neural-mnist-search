"""One image per fresh restricted process; labels and sibling images never arrive."""
import contextlib
import io
import json
import resource
import runpy
import sys
import time

resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
source = sys.argv[1]
request = json.load(sys.stdin)
image = tuple(request["image"])
del request
lines = set()


def trace(frame, event, arg):
    if event == "line" and frame.f_code.co_filename == source:
        lines.add(frame.f_lineno)
    return trace


try:
    clock = time.perf_counter_ns()
    # Candidate stdout is not allowed to masquerade as protocol output.
    # Python-level redirection is diagnostic, not a defense against hostile fd writes.
    with contextlib.redirect_stdout(sys.stderr):
        namespace = runpy.run_path(source)
        imported = time.perf_counter_ns()
        sys.settrace(trace)
        prediction = namespace["predict"](image)
        sys.settrace(None)
        finished = time.perf_counter_ns()
    if type(prediction) is not int or not 0 <= prediction <= 9:
        raise ValueError("predict must return an integer in 0..9")
    print(json.dumps({"status": "ok", "prediction": prediction,
                      "import_ms": (imported - clock) / 1e6,
                      "prediction_ms": (finished - imported) / 1e6,
                      "executed_lines": sorted(lines)}))
except BaseException as error:
    sys.settrace(None)
    print(json.dumps({"status": "candidate_error", "error_type": type(error).__name__,
                      "error": str(error)[:2000], "executed_lines": sorted(lines)}))
