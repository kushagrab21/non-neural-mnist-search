from __future__ import annotations

import os
import selectors
import signal
import subprocess
import tempfile
import time

from .common import now


def run_process(argv, input_bytes=b"", *, timeout=5, cwd="/", env=None, output_limit=1024 * 1024, guard=None):
    """Bounded local process transport. A timeout never proves remote cancellation."""
    started, clock = now(), time.monotonic()
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    status = "exited"
    proc = None
    with tempfile.TemporaryFile() as source:
        source.write(input_bytes)
        source.seek(0)
        try:
            proc = subprocess.Popen(argv, stdin=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    cwd=cwd, env=env, start_new_session=True)
        except OSError as exc:
            return {"argv": argv, "pid": None, "started_at": started, "ended_at": now(),
                    "elapsed_seconds": time.monotonic() - clock, "status": "spawn_error",
                    "returncode": None, "stdout": "", "stderr": str(exc), "remote_completion": "unknown"}
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
            selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                if guard is not None:
                    try: guard()
                    except Exception as exc:
                        status = "guard_stop"
                        streams["stderr"].extend(str(exc).encode()[:4096])
                        break
                remaining = timeout - (time.monotonic() - clock)
                if remaining <= 0:
                    status = "timeout"
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    target = streams[key.data]
                    room = output_limit - len(target)
                    target.extend(data[:room])
                    if len(data) > room:
                        status = "output_limit"
                        break
                if status != "exited":
                    break
        # The parent can finish after closing output; it still needs a deadline.
        if status == "exited":
            try:
                proc.wait(timeout=max(0.01, timeout - (time.monotonic() - clock)))
            except subprocess.TimeoutExpired:
                status = "timeout"
        if status != "exited":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)
        proc.stdout.close()
        proc.stderr.close()
    return {"argv": argv, "pid": proc.pid, "started_at": started, "ended_at": now(),
            "elapsed_seconds": time.monotonic() - clock, "status": status, "returncode": proc.returncode,
            "stdout": bytes(streams["stdout"]).decode("utf-8", errors="replace"),
            "stderr": bytes(streams["stderr"]).decode("utf-8", errors="replace"),
            "remote_completion": "unknown", "output_limit_bytes_per_stream": output_limit}
