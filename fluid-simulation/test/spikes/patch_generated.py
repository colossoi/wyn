"""One-off lowering experiments on /tmp/spike-{neighbor,subgroup}.wgsl.

These patches deliberately assert compiler spelling: they are evidence for a
compiler optimization, not a maintained shader implementation or workaround.
"""
from pathlib import Path

root = Path('/tmp')
s = (root / 'spike-subgroup.wgsl').read_text()
assert 'fn main_segments(' in s
s = '''// Naga 27 accepts subgroup operations without an enable directive.
@group(0) @binding(0) var<storage, read> _buf_0_0: array<array<u32, 32>>;
@group(0) @binding(1) var<storage, read_write> _buf_0_1: array<array<u32, 32>>;
// shortcut: requires a 32-lane subgroup; general lowering needs width selection.
@compute @workgroup_size(32)
fn main_segments(@builtin(workgroup_id) group: vec3<u32>,
                 @builtin(local_invocation_id) lane: vec3<u32>) {
    _buf_0_1[group.x][lane.x] = subgroupInclusiveAdd(_buf_0_0[group.x][lane.x]);
}
'''
(root / 'spike-subgroup-native.wgsl').write_text(s)
h = (root / 'spike-subgroup.wynhost').read_text()
(root / 'spike-subgroup-native.wynhost').write_text(h.replace('spike-subgroup.wgsl', 'spike-subgroup-native.wgsl').replace("'(256 1 1)", "'(32 1 1)"))

s = (root / 'spike-neighbor.wgsl').read_text()
# Control: share the query result but retain the row buffer and tiling kernel.
a = s.index('        var p17_1: array<i32, 258>;')
b = s.index('        v14_1 = (v14_1 + v12_1);', a)
(root / 'spike-neighbor-reuse.wgsl').write_text(s[:a] + '        _buf_0_4[v14_1] = p3_1[96];\n' + s[b:])
(root / 'spike-neighbor-reuse.wynhost').write_text((root / 'spike-neighbor.wynhost').read_text().replace('spike-neighbor.wgsl', 'spike-neighbor-reuse.wgsl'))
start = s.index('        v223_1 = 0i;')
end = s.index('        v14_1 = (v14_1 + v12_1);', start)
assert 'var p17_1: array<i32, 258>;' in s[start:end]
s = s[:start] + '''        for (var ordinal = 0u; ordinal < 96u; ordinal++) {
            _buf_0_2[(v14_1 / 32u) * 3072u + ordinal * 32u + v14_1 % 32u] = p3_1[ordinal];
        }
        _buf_0_4[v14_1] = p3_1[96];
''' + s[end:]
(root / 'spike-neighbor-fused.wgsl').write_text(s)
h = (root / 'spike-neighbor.wynhost').read_text().replace('spike-neighbor.wgsl','spike-neighbor-fused.wgsl')
a, b = h.split("(define-gpu-kernel 'kernel-0-1", 1)
a = a.replace('(argument-1 :buffer :read-write :element :u8 :stride 1 :min-bytes 1605632)', '(argument-3 :buffer :write :element :u8 :stride 1 :min-bytes 1572864)').replace('(argument-1 :storage 0 3)', '(argument-3 :storage 0 2)')
h = a + "(define-gpu-kernel 'kernel-0-1" + b
h = h.replace('    (resource-1 (gpu-alloc (i64 1605632)))\n','').replace(':args (list resource-0 resource-1 resource-2)', ':args (list resource-0 resource-3 resource-2)').replace('    (gpu-free resource-1)\n','')
start = h.index("    (gpu-dispatch 'kernel-0-1")
end = h.index('    (let* (', start)
h = h[:start] + h[end:]
(root / 'spike-neighbor-fused.wynhost').write_text(h)
