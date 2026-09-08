# Host-evaluable logical lengths

`BufferLen::HostExpression` describes a resource's logical element count independently
of its producing stage's physical dispatch. Hosts allocate `count * elem_bytes` bytes.
A fixed `1 × 1 × 1` dispatch may execute a serial loop over this entire capacity.
Do not substitute the workgroup count or padded invocation count for logical length.

The expression vocabulary is `constant`, `uniform`, `convert`, and `binary`.
Scalars are `i32`, `u32`, and `f32`; literals preserve their 32-bit representation in
`bits`. Uniform leaves carry descriptor `set`, `binding`, byte `offset`, and scalar
type. Offsets use the published uniform ABI layout, including vector alignment.
Binary operations are add, subtract, multiply, divide, and remainder. Both operands
must have the same scalar type. Conversions occur exactly where the shader performs
them; `i32(width_f32) * i32(height_f32)` differs from converting a float product.

Call `BufferLen::resolve_host_bytes` with a lookup of uniform words from the same
host snapshot that will be uploaded for this work. Float-to-integer conversions
truncate toward zero and reject non-finite or out-of-range results. Integer
arithmetic is checked at the shader's 32-bit width; overflow and division by zero
are errors, rather than silently computing a wider, different length. Final counts
must be nonnegative integers. A zero count remains zero logically; APIs requiring
nonempty bound buffers may reserve their minimum binding size separately.

The compiler preserves these expressions on semantic iteration spaces and output
allocation policies before physical scheduling. The initial recovery supports
scalar uniform projections, numeric casts, and the arithmetic above in map/range
domains. It does not evaluate arbitrary shader calls or GPU-produced scalar lengths
on the CPU. An unsupported scalar output length produces an allocation diagnostic.
Existing fixed, buffer-derived, and filter-capacity policies remain supported.

Hosts must re-evaluate capacity when referenced uniform values change. Tiny Porto
uses the same packed bytes for allocation and upload, grows buffers before work,
preserves their existing contents, and rebuilds affected resource bindings. Shrinking
a logical length does not require shrinking its backing allocation.
