"""Launch the native driver; SIGKILL its process group five seconds after READY.

Shader compilation is allowed to finish before the simulation deadline starts.
The driver never dispatches the simulation before announcing READY.
"""
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parent


def run(arguments):
    binary=ROOT.parent/'extra/viz/target/release/wyn-fold'
    process=subprocess.Popen([str(binary),*arguments],start_new_session=True,
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    timer=None
    expired=threading.Event()
    def stop():
        expired.set()
        try: os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError: pass
    try:
        for line in process.stdout:
            print(line,end='',flush=True)
            if line.startswith('FOLD_READY') and timer is None:
                timer=threading.Timer(5,stop)
                timer.start()
        status=process.wait()
    except KeyboardInterrupt:
        stop()
        process.wait()
        raise
    finally:
        if timer is not None: timer.cancel()
    if expired.is_set():
        print('Native Fold stopped at the five-second simulation deadline.',flush=True)
        return 124
    return status


if __name__=='__main__':
    sys.exit(run(sys.argv[1:]))
