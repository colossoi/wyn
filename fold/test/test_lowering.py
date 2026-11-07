"""CPU-only checks. No graphics APIs or GPU drivers are initialized."""
from pathlib import Path
import signal
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deadline import run
from lower_serial import audit_hash_construction
from build import audit_native_hash_construction


def hash_writes():
    return '\n'.join(f'_scratch.{name}[i] = 0i;'
                     for name in ['heads', 'next', 'cells'] * 2)


class WorkloadAudit(unittest.TestCase):
    def test_hash_constructed_once_per_contact_pass(self):
        audit_hash_construction('fn main() { loop { loop { '
                                + hash_writes() + ' } } }')

    def test_reject_constructor_inside_particle_or_neighbor_loop(self):
        for extra in range(1, 5):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'Unsafe'):
                audit_hash_construction('fn main() {' + 'loop {' * (2 + extra)
                                        + hash_writes() + '}' * (2 + extra) + '}')

    def test_reject_missing_hash_construction(self):
        with self.assertRaisesRegex(ValueError, 'Unrecognized'):
            audit_hash_construction('fn main() {}')

    def test_comments_cannot_hide_loop_nesting(self):
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            audit_hash_construction('fn main() { loop /* } */ { loop { loop { '
                                    + hash_writes() + ' } } } }')

    def test_native_compiler_output_is_audited_without_rewriting_it(self):
        native = '''fn main() {
          var h: array<i32, 8192>;
          var n: array<i32, 2483>;
          var c: array<vec3<i32>, 2483>;
          loop { loop {
            h[i] = -1i; n[i] = -1i; c[i] = vec3<i32>(0i);
            h[i] = i; n[i] = i; c[i] = vec3<i32>(i);
          } }
        }'''
        audit_native_hash_construction(native)
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            audit_native_hash_construction(native.replace('loop { loop {', 'loop { loop { loop {')
                                           .replace('} }\n', '} } }\n'))


class Deadline(unittest.TestCase):
    def test_sigkill_even_if_child_ignores_sigterm(self):
        started = time.monotonic()
        with patch('deadline.os.killpg', wraps=__import__('os').killpg) as kill:
            with self.assertRaises(TimeoutError):
                run([sys.executable, '-c',
                     'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'],
                    seconds=0.1)
            self.assertEqual(kill.call_args.args[1], signal.SIGKILL)
        self.assertLess(time.monotonic() - started, 1.5)

    def test_reject_long_deadline(self):
        with self.assertRaises(ValueError):
            run(['unused'], seconds=6)


if __name__ == '__main__':
    unittest.main()
