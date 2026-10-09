"""Canonical-interface preparation evaluator; no dataset loader or real-run release.

Use synthetic_qualification with explicit in-memory synthetic arrays. Streaming
keeps future images/labels in the parent. Finite history checks do not prove
statelessness. The parent service time includes IPC; child predict time is a
separate diagnostic and is not trusted against hostile Python.
"""
from __future__ import annotations

import base64
import copy
import json
import math
import os
import random
import secrets
import selectors
import shutil
import signal
import statistics
import subprocess
import sys
import time
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .common import Blocked, canonical, digest, atomic_write, now, strict_json
from .evaluator import restricted_profile, python_binary

CHILD = Path(__file__).with_name("canonical_child.py")
TIMING_BOUNDARY = "parent request encoding/send -> nonce-validated integer reply; includes IPC and conversion; excludes worker startup"


@dataclass(frozen=True)
class Limits:
    startup_seconds: float = 5
    per_call_seconds: float = 1
    total_seconds: float = 60
    observed_rss_bytes: int = 256 * 1024 * 1024
    rss_poll_seconds: float = 0.02
    stream_bytes: int = 65536
    max_calls: int = 4096

    def validate(self):
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise Blocked("limits must be explicit positive finite numbers")
        if any(type(getattr(self, key)) is not int for key in ("observed_rss_bytes", "stream_bytes", "max_calls")):
            raise Blocked("byte and call caps must be integers")


class DeadlineFailure(Blocked):
    def __init__(self, code, phase, deadlines, observed):
        self.code, self.phase = code, phase
        self.evidence = {"phase": phase, "deadlines_monotonic": deadlines, "observed_monotonic": observed,
                         "governing_deadline": deadlines[code]}
        super().__init__(code + " deadline exceeded during " + phase)


def check_deadlines(deadlines, phase):
    # Earliest timestamp governs; code name resolves exact ties deterministically.
    code, deadline = min(deadlines.items(), key=lambda item: (item[1], item[0]))
    observed = time.monotonic()
    if observed >= deadline:
        raise DeadlineFailure(code, phase, dict(deadlines), observed)
    return deadline


class Worker:
    def __init__(self, source, evidence_dir, *, limits=Limits(), mode="candidate", package_root=None, weights=None, evaluation_deadline=None):
        limits.validate()
        if mode != "candidate":
            # Baseline loading is implemented in the worker but cannot be enabled
            # through this preparation release, even by a mode flag.
            raise Blocked("CNN mode awaits its independently approved real measurement release")
        if weights is not None or package_root is not None:
            raise Blocked("candidate mode cannot acquire a baseline weight/runtime exception")
        self.limits, self.directory = limits, Path(evidence_dir)
        self.directory.mkdir(parents=True, exist_ok=False)
        artifact=Path(source).read_bytes()
        (self.directory / "classifier.py").write_bytes(artifact)
        # Hide semantic evidence-directory names (order, anchor, history length)
        # from the child's argv/source path. Only an opaque random path is exposed.
        self.scratch=tempfile.TemporaryDirectory(prefix="mnist-inference-")
        self.source = Path(self.scratch.name).resolve() / "classifier.py"
        self.source.write_bytes(artifact)
        self.source.chmod(0o444)
        self.clock = time.monotonic()
        self.evaluation_deadline = evaluation_deadline
        self.last_rss_check, self.max_rss, self.calls = 0., 0, 0
        self.stdout, self.stderr = bytearray(), bytearray()
        self.receipts, self.termination, self.ready = [], None, None
        self.raw_frames, self.rss_samples, self.closed = [], [], False
        self.proc = None
        self.selector = selectors.DefaultSelector()
        self.profile = restricted_profile([CHILD, self.source]).replace("(allow mach-lookup)\n", "")
        package_root=Path(np.__file__).resolve().parent
        site_root=package_root.parent
        original='(subpath '+json.dumps(str(Path(sys.base_prefix).resolve()))+')'
        restricted='(require-all '+original+' (require-not (subpath '+json.dumps(str(site_root))+')))' 
        self.profile=self.profile.replace(original,restricted+' (literal '+json.dumps(str(site_root))+') (subpath '+json.dumps(str(package_root))+')')
        (self.directory / "worker.sb").write_text(self.profile)
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
        args = [shutil.which("sandbox-exec"), "-p", self.profile, str(python_binary()), "-I", "-B",
                str(CHILD), str(self.source), mode, "-", "-"]
        self.metadata = {"started_at": now(), "source_sha256": digest(self.source.read_bytes()),
            "worker_sha256": digest(CHILD.read_bytes()), "profile_sha256": digest(self.profile.encode()),
            "limits": asdict(limits), "mode": mode, "environment": env, "argv": args,
            "rss_semantics": "parent-observed RSS with polling; not a hard instantaneous peak guarantee"}
        try:
            self.launch_clock = time.monotonic()
            self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         bufsize=0, cwd="/", env=env, start_new_session=True)
            os.set_blocking(self.proc.stdin.fileno(), False)
            for stream, label in ((self.proc.stdout, "stdout"), (self.proc.stderr, "stderr")):
                os.set_blocking(stream.fileno(), False)
                self.selector.register(stream, selectors.EVENT_READ, label)
            self.ready = self._response(self.launch_clock + limits.startup_seconds, phase="startup", timeout_code="STARTUP_TIMEOUT")
            if self.ready.get("status") != "ready" or self.ready.get("numpy_version") != np.__version__:
                raise Blocked("worker did not become ready with the pinned NumPy version")
            self.metadata["parent_startup_ms"] = (time.monotonic()-self.launch_clock)*1000
            self.metadata["launch_monotonic"] = self.launch_clock
        except BaseException as exc:
            self.termination = str(exc)
            self.metadata["failure_evidence"] = getattr(exc, "evidence", None)
            self.close()
            raise

    def _rss(self, deadline):
        current = time.monotonic()
        if current-self.last_rss_check < self.limits.rss_poll_seconds:
            return
        self.last_rss_check = current
        observation_start=time.monotonic()
        try:
            result = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(self.proc.pid)], capture_output=True,
                                    timeout=max(.001,min(.25,deadline-observation_start)), text=True)
        except subprocess.TimeoutExpired as exc:
            raise Blocked("RSS observation timeout") from exc
        if result.returncode == 0 and result.stdout.strip().isdigit():
            rss=int(result.stdout.strip()) * 1024
            self.rss_samples.append({"elapsed_seconds":time.monotonic()-self.clock,"rss_bytes":rss,
                                     "monitor_seconds":time.monotonic()-observation_start})
            self.max_rss = max(self.max_rss, rss)
            if self.max_rss > self.limits.observed_rss_bytes:
                raise Blocked("parent observed RSS cap exceeded")
        elif self.proc.poll() is None:
            raise Blocked("RSS observation unavailable; memory enforcement incomplete")

    def _deadlines(self, local_deadline=None, local_code=None):
        deadlines = {"SESSION_TOTAL_TIMEOUT": self.clock + self.limits.total_seconds}
        if self.evaluation_deadline is not None:
            deadlines["WHOLE_EVALUATION_TIMEOUT"] = self.evaluation_deadline
        if local_deadline is not None:
            deadlines[local_code] = local_deadline
        return deadlines

    def _response(self, deadline, *, phase="response", timeout_code="RESPONSE_TIMEOUT"):
        while True:
            deadlines = self._deadlines(deadline, timeout_code)
            governing = check_deadlines(deadlines, phase)
            try:
                self._rss(governing)
            except Blocked:
                check_deadlines(deadlines, phase)
                raise
            check_deadlines(deadlines, phase)
            if b"\n" in self.stdout:
                line, rest = self.stdout.split(b"\n", 1)
                self.stdout[:] = rest
                self.raw_frames.append({"elapsed_seconds":time.monotonic()-self.clock,"line":line.decode(errors="replace")})
                if rest:
                    raise Blocked("unsolicited extra response bytes")
                try:
                    value = strict_json(line)
                except (ValueError, UnicodeError) as exc:
                    raise Blocked("malformed worker response") from exc
                if not isinstance(value, dict):
                    raise Blocked("worker response is not an object")
                return value
            for key, _ in self.selector.select(0.01):
                data = os.read(key.fileobj.fileno(), 8192)
                if not data:
                    self.selector.unregister(key.fileobj)
                    continue
                target = self.stdout if key.data == "stdout" else self.stderr
                if len(target)+len(data) > self.limits.stream_bytes:
                    target.extend(data[:max(0, self.limits.stream_bytes-len(target))])
                    raise Blocked("worker output cap exceeded")
                target.extend(data)
            if self.proc.poll() is not None and not self.selector.get_map() and b"\n" not in self.stdout:
                raise Blocked("worker exited without complete response")

    def predict(self, image):
        if self.termination is not None or self.proc.poll() is not None:
            raise Blocked("worker is closed or failed")
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (28,28):
            raise Blocked("image must be a uint8 (28,28) NumPy array")
        if self.calls >= self.limits.max_calls:
            raise Blocked("worker call cap exceeded")
        check_deadlines(self._deadlines(), "before_request")
        self.calls += 1
        nonce = secrets.token_hex(16)
        started = time.monotonic()
        try:
            request = canonical({"nonce": nonce, "image_b64": base64.b64encode(image.tobytes()).decode()}) + b"\n"
            remaining=memoryview(request)
            while remaining:
                check_deadlines(self._deadlines(started+self.limits.per_call_seconds, "REQUEST_WRITE_TIMEOUT"), "request_write")
                try:
                    sent=os.write(self.proc.stdin.fileno(),remaining)
                    remaining=remaining[sent:]
                except BlockingIOError:
                    time.sleep(.001)
            reply = self._response(started+self.limits.per_call_seconds)
            if reply.get("nonce") != nonce:
                raise Blocked("stale, unsolicited or mismatched response nonce")
            if reply.get("status") == "invalid_output":
                raise Blocked("invalid digit in worker response: candidate output contract violation")
            if reply.get("status") != "ok":
                raise Blocked("candidate failed: " + str(reply))
            if set(reply) != {"nonce", "status", "digit", "child_predict_ms"}:
                raise Blocked("unexpected worker response fields")
            if type(reply["digit"]) is not int or not 0 <= reply["digit"] <= 9:
                raise Blocked("invalid digit in worker response")
            if type(reply["child_predict_ms"]) not in (int,float) or not math.isfinite(reply["child_predict_ms"]) or reply["child_predict_ms"] < 0:
                raise Blocked("invalid diagnostic timing")
            validated_at = time.monotonic()
            receipt = {"call": self.calls, "image_sha256": digest(image.tobytes()), "response": reply,
                       "service_ms": (validated_at-started)*1000}
            self.receipts.append(receipt)
            return copy.deepcopy(receipt)
        except BaseException as exc:
            self.receipts.append({"call": self.calls, "image_sha256": digest(image.tobytes()), "error": str(exc), "failure_evidence": getattr(exc, "evidence", None)})
            self.termination = str(exc)
            self.metadata["failure_evidence"] = getattr(exc, "evidence", None)
            self.close()
            if isinstance(exc, Blocked):
                raise
            raise Blocked(str(exc)) from exc

    def close(self):
        if self.closed:
            return
        self.closed=True
        if self.proc is not None:
            if self.proc.poll() is None:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.proc.wait(timeout=5)
            for stream in (self.proc.stdin,self.proc.stdout,self.proc.stderr):
                stream.close()
        self.selector.close()
        result = dict(self.metadata, ended_at=now(), calls=self.calls, max_observed_rss_bytes=self.max_rss,
                      termination=self.termination, ready=self.ready, receipts=self.receipts,
                      raw_frames=self.raw_frames, rss_samples=self.rss_samples,
                      memory_enforcement="sampled_rss_kill", cleanup="worker_reaped" if self.proc else "not_spawned",
                      stderr=self.stderr.decode(errors="replace"), leftover_stdout=self.stdout.decode(errors="replace"),
                      elapsed_seconds=time.monotonic()-self.clock)
        atomic_write(self.directory / "RECEIPT.json", canonical(result)+b"\n")
        self.scratch.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def summary(values):
    if not values:
        return None
    ordered=sorted(values)
    return {"n":len(values), "median":statistics.median(values), "p90":float(np.percentile(ordered,90)),
            "minimum":ordered[0], "maximum":ordered[-1]}


def synthetic_qualification(source, images, labels, output, *, warmups=50, blocks=3,
                            history_lengths=(1,16,128), shuffle_seed=20261007, limits=Limits()):
    """Bounded in-memory fixture entry point. No loader/CLI accepts real datasets."""
    if not isinstance(images,np.ndarray) or images.dtype!=np.uint8 or images.ndim!=3 or images.shape[1:]!=(28,28) or len(images)==0:
        raise Blocked("nonempty uint8 (n,28,28) images required")
    if not isinstance(labels,np.ndarray) or labels.dtype.kind not in "iu" or labels.shape!=(len(images),) or np.any(labels>9) or np.any(labels<0):
        raise Blocked("matching integer labels in 0..9 required")
    if type(warmups) is not int or warmups<0 or type(blocks) is not int or blocks<1:
        raise Blocked("invalid warmup/block recipe")
    if not history_lengths or any(type(n) is not int or n<1 for n in history_lengths):
        raise Blocked("nonempty positive history lengths required")
    limits.validate()
    required_calls = warmups+len(images)*blocks
    if required_calls>limits.max_calls or max(history_lengths)+2>limits.max_calls or len(images)>limits.max_calls:
        raise Blocked("recipe exceeds declared per-worker call allocation")
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    order=list(range(len(images))); shuffled=order.copy(); random.Random(shuffle_seed).shuffle(shuffled)
    recipe={"warmups":warmups,"blocks":blocks,"history_lengths":list(history_lengths),"shuffle_seed":shuffle_seed,
            "forward":order,"reverse":list(reversed(order)),"shuffle":shuffled,"limits":asdict(limits)}
    atomic_write(output/'RECIPE.json',canonical(recipe)+b'\n')
    predictions, failure, differences, measurements = {}, None, [], []
    started=time.monotonic()
    try:
        for name,sequence in [('forward',order),('reverse',order[::-1]),('shuffle',shuffled)]:
            preds=[None]*len(images)
            with Worker(source,output/name,limits=limits) as worker:
                for index in sequence:
                    preds[index]=worker.predict(images[index])["response"]["digit"]
            predictions[name]=preds
        forward=predictions['forward']
        for name in ('reverse','shuffle'):
            differences += [{"control":name,"index":i} for i,(a,b) in enumerate(zip(forward,predictions[name])) if a!=b]
        for length in history_lengths:
            with Worker(source,output/('history-'+str(length)),limits=limits) as worker:
                first=worker.predict(images[0])["response"]["digit"]
                for i in range(length): worker.predict(images[(i+1)%len(images)])
                repeated=worker.predict(images[0])["response"]["digit"]
            if first!=repeated or first!=forward[0]:
                differences.append({"control":"repeated_history","length":length,"first":first,"repeated":repeated,"reference":forward[0]})
        with Worker(source,output/'timing',limits=limits) as worker:
            for i in range(warmups): worker.predict(images[i%len(images)])
            for block in range(blocks):
                receipts=[worker.predict(im) for im in images]
                values=[r['response']['digit'] for r in receipts]
                if values!=forward: differences.append({"control":"warmed_block","block":block})
                measurements.append({"block":block,"parent_service_ms":summary([r['service_ms'] for r in receipts]),
                                     "child_predict_ms_diagnostic":summary([r['response']['child_predict_ms'] for r in receipts])})
    except (Blocked,OSError,ValueError) as exc:
        failure=str(exc)
    # This is a finite synthetic qualification, never campaign admissibility.
    status='INCOMPLETE' if failure else ('ORDER_DEPENDENT' if differences else 'SYNTHETIC_CHECKS_PASS')
    result={"scope":"synthetic_preparation_only", "status":status,"campaign_qualified":False,
        "source_sha256":digest(Path(source).read_bytes()),"source_bytes":Path(source).stat().st_size,
        "numpy_version":np.__version__,"worker_sha256":digest(CHILD.read_bytes()),
        "recipe_sha256":digest(recipe),"input_sha256":digest(images.tobytes()),"labels_sha256":digest(labels.tobytes()),
        "n":len(images),"predictions":predictions,"accuracy":float(np.mean(np.asarray(predictions['forward'])==labels)) if 'forward' in predictions else None,
        "failure":failure,"order_history_differences":differences,"measurements":measurements,
        "parent_service_boundary":TIMING_BOUNDARY,"predict_timing_trust":"child diagnostic; not accepted scientific performance",
        "block_median_service_variability_ms":summary([b['parent_service_ms']['median'] for b in measurements]),
        "coverage":"all supplied synthetic images; finite repeated histories on image 0; not a whole MNIST fold",
        "runtime_dependency_size":"not measured; dependency identity NumPy recorded, CNN exception disabled",
        "elapsed_seconds":time.monotonic()-started}
    atomic_write(output/'RESULT.json',canonical(result)+b'\n')
    return result
