"""Numerical parity for DFSPH consuming upstream's tiled neighbor layout."""
from pathlib import Path
import tempfile
import math
from check import reference
from gpu import ROOT, run

with tempfile.TemporaryDirectory(prefix='fluid-neighbor-step-') as tmp:
    for target in ['spirv', 'wgsl']:
        outputs = run(ROOT/'fluid-simulation/test/step_neighbors.wyn', Path(tmp), target)
        assert len(outputs) == 2
        for actual, expected in zip(outputs, reference()):
            assert len(actual) == len(expected)
            assert all(math.isfinite(a) and math.isclose(a,b,rel_tol=2e-4,abs_tol=2e-4)
                       for a,b in zip(actual,expected)), (actual,expected)
        print(target, 'tiled neighbor DFSPH matches scalar reference')
