"""Compile the ordered Wyn solver for the native Viz driver."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
from lower_serial import audit_hash_construction
from spill import spill
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'generated'
def build():
    OUT.mkdir(exist_ok=True)
    compiler=os.environ.get('WYN',str(ROOT.parent/'target/release/wyn'))
    # Never leave a previous, unsafe artifact available after a failed build.
    for name in ['constraints.wgsl', 'constraints.wynhost', 'bindings.json']:
        (OUT/name).unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix='wyn-fold-') as directory:
        temporary=Path(directory)/'constraints.wgsl'
        subprocess.run([compiler,'build',str(ROOT/'src/constraints.wyn'),'-t','wgsl','--direct','-o',str(temporary)],check=True)
        publish(temporary)

def publish(temporary):
    path=OUT/'constraints.wgsl'
    code=temporary.read_text()
    audit_native_hash_construction(code)
    entries=re.findall(r'@compute\s+@workgroup_size\([^)]*\)\s+fn\s+(\w+)',code)
    if entries!=['constraints_compute']:
        raise RuntimeError(f'Expected one serial kernel, got {entries}')
    bindings=re.findall(r'@group\((\d+)\)\s+@binding\((\d+)\)\s+var<storage,\s*\w+>\s+(\w+)',code)
    # Small fixed arrays share the compiler's parameter block. Check its layout
    # so compiler changes cannot silently corrupt the native transport.
    fields=re.search(r'struct _PcBlock0 \{(.*?)\}',code,re.S).group(1)
    expected=['w_planes: array<vec4<f32>, 3>','w_config: array<vec4<f32>, 3>','w_grab_Uinfo: array<vec2<f32>, 9>']
    if not all(field in fields for field in expected):
        raise RuntimeError(f'Unexpected parameter layout: {fields}')
    names={(1,0):'parameters',(0,0):'input_p',(0,1):'old',(0,2):'edges',(0,3):'grabs',(0,4):'output'}
    if {(int(g),int(b)) for g,b,_ in bindings}!=names.keys():
        raise RuntimeError(f'Unexpected bindings: {bindings}')
    # Driver-private arrays exceed what the AMD Metal compiler can compile.
    # Use one explicit scratch allocation; no aliases or equation rewrites.
    path.write_text(spill(code))
    print(f'Built {path}')
def audit_native_hash_construction(code):
    # Derive local-array aliases from types before auditing loop placement.
    # This only annotates a string for inspection; it does not change emitted
    # WGSL, storage, instruction order or expressions.
    from lower_serial import ARRAY, FAMILIES
    declarations=re.finditer(r'\b(?:let|var)\s+(\w+)\s*:\s*'+ARRAY,code)
    aliases={m[1]:FAMILIES[m[2],m[3]] for m in declarations if (m[2],m[3]) in FAMILIES}
    inspected=code
    for name,family in aliases.items():
        inspected=re.sub(r'\b'+re.escape(name)+r'\s*\[',f'_scratch.{family}[',inspected)
    audit_hash_construction(inspected)

if __name__=='__main__':
    build()
