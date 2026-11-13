#!/usr/bin/env python3
"""Run each integration case in a process with a hard five-minute deadline."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time


def windows_child(command):
    """Keep a kill-on-close job handle in the supervisor, before starting the test."""
    import ctypes as c

    class BasicLimits(c.Structure):
        _fields_ = [("process_time", c.c_int64), ("job_time", c.c_int64),
                    ("flags", c.c_uint32), ("minimum_working_set", c.c_size_t),
                    ("maximum_working_set", c.c_size_t), ("active_processes", c.c_uint32),
                    ("affinity", c.c_size_t), ("priority", c.c_uint32), ("scheduling", c.c_uint32)]

    class ExtendedLimits(c.Structure):
        _fields_ = [("basic", BasicLimits), ("io_counters", c.c_uint64 * 6),
                    ("process_memory", c.c_size_t), ("job_memory", c.c_size_t),
                    ("peak_process_memory", c.c_size_t), ("peak_job_memory", c.c_size_t)]

    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [c.c_void_p, c.c_wchar_p]
    kernel.CreateJobObjectW.restype = c.c_void_p
    kernel.SetInformationJobObject.argtypes = [c.c_void_p, c.c_int, c.c_void_p, c.c_uint32]
    kernel.AssignProcessToJobObject.argtypes = [c.c_void_p, c.c_void_p]
    kernel.GetCurrentProcess.restype = c.c_void_p
    job = kernel.CreateJobObjectW(None, None)
    limits = ExtendedLimits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not job or not kernel.SetInformationJobObject(job, 9, c.byref(limits), c.sizeof(limits)):
        raise c.WinError(c.get_last_error())
    if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        raise c.WinError(c.get_last_error())
    # Deliberately keep this non-inherited handle until supervisor exit. Killing
    # it closes the handle and kills every descendant, without a spawn/attach race.
    return subprocess.call(command)


class Runner:
    def __init__(self):
        self.lock = threading.Lock()
        self.stopped = False
        self.active = set()

    @staticmethod
    def kill(process):
        if os.name == "nt":
            if process.poll() is None:
                process.kill()  # The supervisor's job owns the complete process tree.
            return
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()

    def cancel(self):
        with self.lock:
            self.stopped = True
            for process in self.active:
                self.kill(process)

    def run(self, command, env, timeout=300):
        started = time.monotonic()
        if os.name == "nt":
            command = [sys.executable, str(Path(__file__).resolve()), "--child", *command]
        with self.lock:
            if self.stopped:
                return False, 0, "cancelled"
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       encoding="utf-8", errors="replace",
                                       start_new_session=os.name != "nt")
            self.active.add(process)
        try:
            try:
                output, _ = process.communicate(timeout=timeout)
                return process.returncode == 0, time.monotonic() - started, output
            except subprocess.TimeoutExpired:
                self.kill(process)
                output, _ = process.communicate()
                return False, time.monotonic() - started, f"TIMEOUT after {timeout}s\n{output}"
        finally:
            with self.lock:
                self.active.remove(process)


def cases(artifacts, root, selected):
    for artifact in artifacts:
        executable = artifact["executable"]
        target = artifact["target"]["name"]
        listing = subprocess.check_output([executable, "--list", "--format", "terse"], text=True)
        for line in listing.splitlines():
            if not line.endswith(": test"):
                continue
            test = line.removesuffix(": test")
            if target == "compiler_fixtures" and test == "compiler_fixtures":
                for directory in ["tests/compiler/cases", "pkg/sort/test/compiler"]:
                    for path in sorted((root / directory).glob("*.wyn")):
                        label = f"{target}::{path.relative_to(root).as_posix()}"
                        if selected in label:
                            yield label, [executable, test, "--exact", "--nocapture"], {
                                "WYN_TEST_FILE": str(path.resolve()), "WYN_TEST_FILTER": ""}
            else:
                label = f"{target}::{test}"
                if selected in label:
                    yield label, [executable, test, "--exact", "--nocapture"], {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", default="", help="Run labels containing this text")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    env = dict(os.environ, RUST_MIN_STACK="16777216")
    build = subprocess.run(["cargo", "test", "--workspace", "--tests", "--no-run",
                            "--message-format=json"], env=env, stdout=subprocess.PIPE,
                           text=True, encoding="utf-8", errors="replace")
    messages = [json.loads(line) for line in build.stdout.splitlines() if line.strip()]
    if build.returncode:
        for message in messages:
            if message["reason"] == "compiler-message":
                print(message["message"].get("rendered", ""))
        return 1
    artifacts = {m["executable"]: m for m in messages if m["reason"] == "compiler-artifact"
                 and m.get("executable") and "test" in m["target"]["kind"]}
    pending = iter(cases(artifacts.values(), root, args.filter))
    runner = Runner()
    started = time.monotonic()
    count = 0
    # All fixture workers share one stable compiler snapshot, not a copy per case.
    with tempfile.TemporaryDirectory(prefix="wyn_integration_compiler_") as directory:
        binaries = [m["executable"] for m in messages if m["reason"] == "compiler-artifact"
                    and m.get("executable") and m["target"]["name"] == "wyn"
                    and "bin" in m["target"]["kind"] and not m["profile"]["test"]]
        if binaries:
            compiler = Path(directory) / Path(binaries[-1]).name
            shutil.copy2(binaries[-1], compiler)
            env["WYN_TEST_COMPILER"] = str(compiler)
        with ThreadPoolExecutor(max_workers=4) as pool:
            active = {}
            def submit():
                case = next(pending, None)
                if case is not None:
                    label, command, extra = case
                    active[pool.submit(runner.run, command, env | extra)] = label
            try:
                for _ in range(4):
                    submit()
                while active:
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        label = active.pop(future)
                        ok, elapsed, output = future.result()
                        print(f"{elapsed:.3f}s {'ok' if ok else 'FAILED'}: {label}", flush=True)
                        if not ok:
                            runner.cancel()
                            print(output, flush=True)
                            return 1
                        count += 1
                    for _ in done:
                        submit()
            finally:
                runner.cancel()
    print(f"{count} integration cases completed in {time.monotonic() - started:.3f}s (build excluded)")
    return int(count == 0)


if __name__ == "__main__":
    raise SystemExit(windows_child(sys.argv[2:]) if sys.argv[1:2] == ["--child"] else main())
