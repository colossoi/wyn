"""Numerical parity for DFSPH consuming upstream's tiled neighbor layout."""
import json
from pathlib import Path
import subprocess
import tempfile
import math
from check import ROOT, WYN, VIZ, reference

with tempfile.TemporaryDirectory(prefix='fluid-neighbor-step-') as tmp:
    for target,ext in [('spirv','spv'),('wgsl','wgsl')]:
        path = Path(tmp)/f'step.{ext}'
        subprocess.run([str(WYN),'build',str(ROOT/'fluid-simulation/test/step_neighbors.wyn'),'-t',target,'-o',str(path)],check=True)
        desc = json.loads(path.with_suffix('.json').read_text())
        args = [str(VIZ),'pipeline',str(path)]
        files = []
        for r in desc['source_results']:
            b = next(b for b in desc['pipelines'][r['pipeline_index']]['bindings'] if b.get('set')==r['set'] and b.get('binding')==r['binding'])
            f = Path(tmp)/f"result-{r['result']}.json"
            files.append(f)
            args += ['--output',f"{b['name']}:{f}"]
        subprocess.run(args,check=True,stdout=subprocess.DEVNULL,timeout=120)
        for f,expected in zip(files,reference()):
            actual = json.loads(f.read_text())
            assert len(actual)==len(expected)
            assert all(math.isclose(a,b,rel_tol=2e-4,abs_tol=2e-4) for a,b in zip(actual,expected)),(actual,expected)
        print(target, 'tiled neighbor DFSPH matches scalar reference')
