#!/usr/bin/env python3
"""Enforce workspace unit-test timing without updating the exception list."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def timing_violations(tests, exceptions, limit=3):
    return [test for test in tests if test["event"] != "ignored" and (
        not isinstance(test.get("exec_time"), (int, float))
        or not math.isfinite(test["exec_time"])
        or (test["exec_time"] > limit and test["id"] not in exceptions))]


def read_exceptions(text):
    names = [line.strip() for line in text.splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate unit timing exception")
    return set(names)


def run_harness(artifact, exceptions, exact=None):
    target = artifact["target"]["name"]
    args = [artifact["executable"], "-Z", "unstable-options", "--format", "json",
            "--report-time", f"--test-threads={1 if exact else 8}"]
    if exact:
        args += [exact, "--exact"]
    env = dict(os.environ, RUST_MIN_STACK="16777216", RUSTC_BOOTSTRAP="1")
    tests, summary, invalid = [], None, None
    with subprocess.Popen(args, env=env, stdout=subprocess.PIPE, text=True,
                          encoding="utf-8", errors="replace") as child:
        for line in child.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                invalid = f"{error}: {line[:200]}"
                continue
            if event.get("type") == "test" and event["event"] in ("ok", "failed", "ignored"):
                test = dict(event, id=f"{target}::{event['name']}")
                tests.append(test)
                if event["event"] == "failed":
                    print(f"{test['id']}: FAILED\n{event.get('stdout', '')}", file=sys.stderr)
                if event.get("exec_time", 0) > 3:
                    print(f"{event['exec_time']:.3f}s {test['id']}", flush=True)
            if event.get("type") == "suite" and event["event"] in ("ok", "failed"):
                summary = event
        code = child.wait()
    if invalid or summary is None or len(tests) != sum(summary[key] for key in ("passed", "failed", "ignored")):
        raise RuntimeError(f"Incomplete test timing for {target}: {invalid or 'missing results'}")
    candidates = timing_violations(tests, exceptions)
    print(f"{target}: {summary['passed']} passed, {summary['failed']} failed, "
          f"{summary['ignored']} ignored, {len(candidates)} unlisted timing candidates", flush=True)
    return code != 0, tests, candidates


def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    exceptions = read_exceptions(Path("tests/unit-time-exceptions.txt").read_text())
    # Bootstrap applies only to libtest's timing output, never to compilation.
    env = dict(os.environ, RUST_MIN_STACK="16777216")
    env.pop("RUSTC_BOOTSTRAP", None)
    build = subprocess.run(["cargo", "test", "--workspace", "--lib", "--bins", "--no-run",
                            "--message-format=json"], env=env, stdout=subprocess.PIPE,
                           text=True, encoding="utf-8", errors="replace")
    messages = [json.loads(line) for line in build.stdout.splitlines() if line.strip()]
    if build.returncode:
        for message in messages:
            if message["reason"] == "compiler-message":
                print(message["message"].get("rendered", ""), file=sys.stderr)
        raise RuntimeError(f"Unit test build failed: {build.returncode}")
    binaries = {a["executable"]: a for a in messages if a["reason"] == "compiler-artifact"
                and a["profile"]["test"] and a.get("executable")}
    if not binaries:
        raise RuntimeError("No workspace unit test harnesses discovered")
    started = time.perf_counter()
    failed, tests, candidates = False, [], []
    for artifact in binaries.values():
        bad, results, slow = run_harness(artifact, exceptions)
        failed |= bad
        tests.extend(results)
        candidates.extend((artifact, test) for test in slow)
    violations = 0
    # Confirm without parallel contention; reruns cannot erase functional failures.
    for artifact, test in candidates:
        print(f"Confirming unit time limit in isolation: {test['id']}", flush=True)
        bad, results, slow = run_harness(artifact, exceptions, test["name"])
        failed |= bad
        if len(results) != 1 or slow or results[0]["event"] == "ignored":
            violations += 1
            elapsed = results[0].get("exec_time", "missing") if results else "missing"
            print(f"UNIT TIME LIMIT: {test['id']}: {elapsed}s (limit 3s)", file=sys.stderr)
    print("Top 20 unit tests by elapsed time:")
    timed = [test for test in tests if isinstance(test.get("exec_time"), (int, float))]
    for test in sorted(timed, key=lambda test: test["exec_time"], reverse=True)[:20]:
        print(f"{test['exec_time']:.3f}s {test['event']}: {test['id']}")
    print(f"Unit test wall time: {time.perf_counter() - started:.3f}s (build excluded)")
    print(f"Confirmed unlisted timing violations: {violations}")
    return int(failed or violations > 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
