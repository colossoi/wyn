"""GPU readbacks for the Wyn spatial primitives (uses the checkout compiler)."""
from pathlib import Path
import tempfile
from gpu import ROOT, run


def encode(p):
    # Scalar transcription of upstream HilbertHelpers.slang, no Wyn helpers.
    x,y,z = p
    key = 0
    table = [0,1,3,2,7,6,4,5]
    for level in range(9,-1,-1):
        a,b,c = (x>>level)&1,(y>>level)&1,(z>>level)&1
        key = (key<<3)+table[(a<<2)|(b<<1)|c]
        if a and (not b or c): x ^= 0xffffffff
        if (a and (b or c)) or (b and not c): y ^= 0xffffffff
        if (a and not b and not c) or (b and not c): z ^= 0xffffffff
        if c: x,y,z = y,z,x
        elif not b: x,z = z,x
    return key


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='fluid-spatial-') as tmp:
        directory = Path(tmp)
        for target in ['spirv','wgsl']:
            hilbert, = run(ROOT/'fluid-simulation/test/hilbert.wyn',directory,target)
            for i in range(128):
                p = [(i*137)%1024,(i*293)%1024,(i*617)%1024]
                assert hilbert[i*4:i*4+4] == [encode(p),*p], (target,i)
            sort, = run(ROOT/'fluid-simulation/test/spatial.wyn',directory,target)
            assert sort == [0,3,1,2,2,1,3,0], (target,sort)
            radix, = run(ROOT/'fluid-simulation/test/radix.wyn',directory,target)
            pairs = [((((i*73)%31) << 25) + i%3, i) for i in range(257)]
            expected = [v for pair in sorted(pairs, key=lambda p: p[0]) for v in pair]
            assert radix == expected, (target, 'radix ordering/stability', radix)
            tree,overflow = run(ROOT/'fluid-simulation/test/octree.wyn',directory,target)
            assert tree == [i*(1<<27) for i in range(8)]+[1<<30]*8,(target,tree)
            assert overflow == [0]
            print(target, 'Hilbert oracle, 30-bit radix sort and stability across scan blocks, cornerstone split: passed')
