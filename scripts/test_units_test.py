import unittest
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from unittest.mock import patch
from test_units import read_exceptions, run_harness, timing_violations


class UnitTimingGateTests(unittest.TestCase):
    def test_new_slow_tests_fail_including_failing_tests(self):
        tests = [dict(id="old", event="ok", exec_time=9),
                 dict(id="new", event="failed", exec_time=3.01)]
        self.assertEqual(timing_violations(tests, {"old"}), [tests[1]])

    def test_fast_and_ignored_pass_but_missing_or_invalid_times_fail_closed(self):
        tests = [dict(id="fast", event="ok", exec_time=3), dict(id="skip", event="ignored"),
                 dict(id="lost", event="ok"), dict(id="nan", event="ok", exec_time=float("nan"))]
        self.assertEqual(timing_violations(tests, set()), tests[2:])

    def test_exception_names_match_exactly_and_comments_are_ignored(self):
        exceptions = read_exceptions("# approval required\ncrate::test\n\n")
        test = dict(id="crate::test_extra", event="ok", exec_time=4)
        self.assertEqual(timing_violations([test], exceptions), [test])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_exceptions("same\nsame\n")

    def test_functional_failures_survive_an_isolated_timing_recheck(self):
        events = [dict(type="test", name="regression", event="failed", exec_time=0.1),
                  dict(type="suite", event="failed", passed=0, failed=1, ignored=0)]
        with patch("test_units.subprocess.Popen") as spawn:
            child = spawn.return_value.__enter__.return_value
            child.stdout = iter(json.dumps(event) for event in events)
            child.wait.return_value = 101
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                failed, tests, slow = run_harness(
                    dict(target=dict(name="crate"), executable="test.exe"), set(), "regression")
            self.assertTrue(failed)
            self.assertEqual(slow, [])
            self.assertEqual(tests[0]["event"], "failed")
            self.assertIn("--test-threads=1", spawn.call_args.args[0])
            self.assertIn("--exact", spawn.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
