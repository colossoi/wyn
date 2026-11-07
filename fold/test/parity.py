"""Run the compiled Wyn solver once per upstream fixture in the native driver.

Every native simulation invocation has a five-second process-group SIGKILL deadline.
No browser is opened. Build the native driver and shader separately before running.
"""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    with tempfile.TemporaryDirectory(prefix='fold-parity-') as directory:
        directory=Path(directory)
        subprocess.run(['node',str(ROOT/'test/fixtures.mjs'),str(directory)],check=True)
        for name in ['hanging','contact-grab-walls','retirement','weighted-contact','full-hanging']:
            case=directory/name
            subprocess.run([sys.executable,str(ROOT/'run.py'),'--headless','--fixture',str(case)],check=True)
            expected=json.loads((case/'expected.json').read_text())
            data=(case/'actual.bin').read_bytes()
            actual=struct.unpack('<'+str(len(data)//4)+'f',data)
            if actual[3]<0: raise AssertionError(f'{name}: malformed contact chain')
            errors=[abs(actual[i*4+a]-expected['positions'][i*3+a])
                    for i in range(expected['count']) for a in range(3)]
            worst=max(errors)
            # Four ordered XPBD iterations, starting from identical f32 input.
            if not all(error<=0.0002 for error in errors):
                raise AssertionError(f'{name}: max coordinate error {worst:g} exceeds 0.0002')
            print(f'{name}: {expected["count"]} particles, max coordinate error {worst:.8g}',flush=True)


if __name__=='__main__':
    main()
