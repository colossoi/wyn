import json
import os
import tempfile
import unittest
from pathlib import Path

from report_full_gates import render, render_run


class FullGateReportTest(unittest.TestCase):
    def test_interrupted_journal_keeps_completed_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run = base / "interrupted"
            run.mkdir()
            (run / "workspace.jsonl").write_text(
                '{"name":"format","seconds":1,"exit_code":0}\n{"name":"workspace-build"'
            )
            (run / "integrations.jsonl").write_text(
                '{"name":"integration-tests","seconds":1,"exit_code":0}\n{"name":"retry"'
            )
            result = render_run(run, base)
            self.assertEqual(result.status, "Incomplete")
            checks = {check.name: check.status for check in result.checks}
            self.assertEqual(checks["Formatting"], "Passed")
            self.assertEqual(checks["Workspace build"], "Incomplete")
            self.assertEqual(checks["Integration tests"], "Incomplete")
            self.assertIn("Incomplete journal record", render(base))

    def test_latest_retry_and_five_run_history(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for index in range(6):
                run = base / f"run-{index}"
                run.mkdir()
                (run / "run.json").write_text(json.dumps({"title": f"Round <{index}>"}))
                path = run / "integrations.jsonl"
                data = [
                    {"name": "first", "seconds": 60.0, "exit_code": 1},
                    {"name": "retry", "seconds": 3.0, "exit_code": 0},
                ]
                path.write_text("\n".join(map(json.dumps, data)))
                stamp = 1_700_000_000 + index
                os.utime(path, (stamp, stamp))
                log = "2.000s ok: fixture\n"
                if index == 5:
                    log += "1 integration cases completed in 2.300s (build excluded)\n"
                (run / "retry.log").write_text(log)
            html = render(base)
            self.assertIn('<details open class="history"><summary>Last 5 runs</summary>', html)
            matrix = html.split('<table class="matrix">')[1]
            self.assertEqual(matrix.count('<th scope="col"'), 6)
            self.assertIn('title="Round &lt;5&gt;">Latest', matrix)
            self.assertIn('title="Round &lt;1&gt;">Run 5', matrix)
            self.assertEqual(matrix.count('<td title="'), 50)
            self.assertIn(
                'title="Passed. 2.3s. 1 integration test passed. Build excluded. Gate total including build 3.0s."',
                matrix,
            )
            self.assertIn('<span class="cell-time">2.3s</span>', matrix)
            self.assertIn(
                'title="Passed. 3.0s. 1 integration test passed. Gate time includes build."',
                matrix,
            )
            self.assertIn('<span class="cell-time">3.0s</span>', matrix)
            self.assertIn("Round &lt;5&gt;", html.split("Last 5 runs")[0])
            self.assertNotIn("Round &lt;0&gt;", html)
            self.assertIn("1 integration test passed.", html)
            self.assertIn("run-5/retry.log", html)
            self.assertNotIn("run-5/first.log", html)
            self.assertEqual(render_run(run, base).status, "Incomplete")


if __name__ == "__main__":
    unittest.main()
