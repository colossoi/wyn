"""Exercise deadline, output draining, process-tree cleanup, and fail-fast cancellation."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from test_integrations import Runner, cases


class IntegrationRunnerTests(unittest.TestCase):
    def test_output_and_exit_status(self):
        runner = Runner()
        ok, _, output = runner.run([sys.executable, "-c", "print('x'*262144)"], os.environ, 3)
        self.assertTrue(ok)
        self.assertEqual(len(output.strip()), 262144)
        ok, _, output = runner.run([sys.executable, "-c", "print('failure');exit(7)"], os.environ, 3)
        self.assertFalse(ok)
        self.assertIn("failure", output)

    def test_timeout_kills_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "orphan.txt"
            child = f"import time;from pathlib import Path;time.sleep(2);Path({str(marker)!r}).touch()"
            parent = f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',{child!r}]);time.sleep(10)"
            ok, elapsed, output = Runner().run([sys.executable, "-c", parent], os.environ, 0.4)
            self.assertFalse(ok)
            self.assertIn("TIMEOUT", output)
            self.assertLess(elapsed, 3)
            time.sleep(2)
            self.assertFalse(marker.exists(), "compiler descendant survived the timeout")

    def test_cancel_stops_other_workers_and_prevents_new_processes(self):
        runner = Runner()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(runner.run, [sys.executable, "-c", "import time;time.sleep(10)"],
                                   os.environ, 3) for _ in range(2)]
            try:
                until = time.monotonic() + 2
                while time.monotonic() < until:
                    with runner.lock:
                        if len(runner.active) == 2:
                            break
                    time.sleep(0.01)
            finally:
                runner.cancel()
            self.assertTrue(all(not future.result(timeout=3)[0] for future in futures))
        self.assertEqual(runner.run(["must-not-be-started"], os.environ)[2], "cancelled")

    def test_source_fixtures_are_individual_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "tests/compiler/cases/rows.wyn"
            fixture.parent.mkdir(parents=True)
            fixture.touch()
            fixture.with_name("rows_optimized.wyn").touch()
            artifacts = [{"executable": "harness", "target": {"name": "compiler_fixtures"}}]
            with patch("test_integrations.subprocess.check_output", return_value="compiler_fixtures: test\n"):
                found = list(cases(artifacts, root, "rows.wyn"))
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0][2]["WYN_TEST_FILE"], str(fixture.resolve()))


if __name__ == "__main__":
    unittest.main()
