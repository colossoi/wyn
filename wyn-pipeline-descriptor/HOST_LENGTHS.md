# Host-provided logical capacities

`BufferLen::HostProvided` marks a storage buffer whose allocation capacity cannot be
derived generically from the pipeline descriptor. The host application must allocate
the buffer explicitly.

The descriptor includes an `inputs` list containing host-visible scalar fields that may
influence the program's logical output length. Each item identifies a field by name,
uniform binding, byte offset, and scalar type. This list is diagnostic dependency
metadata only; it intentionally does not encode or evaluate the program's calculation.
An empty list means the compiler found no host-visible scalar dependencies, not that the
logical length is constant.

`elem_bytes` records the storage stride of one logical output element. The capacity the
host supplies is still measured in bytes and may conservatively exceed the exact logical
length. The shader retains the original calculation and computes the exact logical
length during execution when its serial lowering needs it.

`SameAsDispatch` remains a distinct policy for resources whose logical element count is
exactly the dispatch domain. Do not infer either policy from padded workgroup counts.
