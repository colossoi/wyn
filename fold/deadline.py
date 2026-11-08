"""Run a child process with a short process-group SIGKILL deadline."""
import argparse
import os
import signal
import subprocess
import sys


def run(command, seconds=3, **kwargs):
    if not 0 < seconds <= 30:
        raise ValueError('Deadline must be positive and at most thirty seconds')
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        status = process.wait(timeout=seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise TimeoutError(f'Killed process group after {seconds:g}s: {command[0]}')
    if status:
        raise subprocess.CalledProcessError(status, command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=3)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if not options.command:
        parser.error('A command is required')
    try:
        run(options.command, options.seconds)
    except TimeoutError as error:
        print(error, file=sys.stderr)
        sys.exit(124)
