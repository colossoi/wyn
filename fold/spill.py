"""Allocate serial solver arrays in storage instead of driver-private memory.

No arrays alias. Every declaration retains its zero initialization at the same
point in control flow. This lowering only accepts the audited one-invocation
solver and rejects whole-array assignments or unexpected allocation shapes.
"""
import re

ARRAYS=[('f32',13880),('vec3<f32>',9),('vec4<f32>',2483),
        ('i32',2483),('i32',8192),('vec3<i32>',2483)]
SCRATCH_BYTES=177824


def spill(code):
    if re.findall(r'@compute\s+@workgroup_size\(([^)]*)\)',code)!=['1, 1, 1']:
        raise ValueError('Scratch lowering requires exactly one serial compute invocation')
    pattern=re.compile(r'\bvar\s+(\w+)\s*:\s*array<\s*(vec[234]<\w+>|\w+)\s*,\s*(\d+)\s*>\s*;')
    arrays=list(pattern.finditer(code))
    if [(m[2],int(m[3])) for m in arrays]!=ARRAYS:
        raise ValueError('Unexpected solver arrays; review the scratch layout')
    fields='\n'.join(f'    {m[1]}: array<{m[2]}, {m[3]}>, ' for m in arrays)
    declarations=f'\nstruct FoldScratch {{\n{fields}\n}}\n@group(2) @binding(0) var<storage, read_write> fold_scratch: FoldScratch;\n'
    for m in reversed(arrays):
        # WGSL local var declarations initialize every element to zero, even
        # inside branches/loops. Preserve that behavior explicitly in storage.
        zero=f'{m[2]}(0)' if m[2].startswith('vec') else ('0.0f' if m[2]=='f32' else '0i')
        init=f'for (var fold_zero_{m[1]} = 0i; fold_zero_{m[1]} < {m[3]}i; fold_zero_{m[1]}++) {{ {m[1]}[fold_zero_{m[1]}] = {zero}; }}'
        code=code[:m.start()]+init+code[m.end():]
    for m in arrays:
        # A scratch variable must only be accessed by indexing; never copied,
        # passed by value, or used as a differently scoped binding.
        if re.search(r'\b'+m[1]+r'\b(?!\s*\[)',code):
            raise ValueError(f'Whole-array access to {m[1]}')
        code=re.sub(r'\b'+m[1]+r'\b','fold_scratch.'+m[1],code)
    return declarations+code
