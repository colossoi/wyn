# Host expressions

A host expression computes a logical element count using a snapshot of the
uniform bytes uploaded for the execution. Allocation and dispatch use the same
JSON shape:

```json
{
  "kind": "host_expression",
  "inputs": {
    "width":  { "set": 0, "binding": 0, "offset": 0, "type": "f32" },
    "height": { "set": 0, "binding": 0, "offset": 4, "type": "f32" }
  },
  "count": "i32(width) * i32(height)"
}
```

For a buffer length, add `"elem_bytes": 4`: allocation requires `count * 4`
bytes. For a dispatch, place this object in `len` of a `derived_from` dispatch;
`workgroup_size` determines `[ceil(count / workgroup_size), 1, 1]` workgroups.
A zero count is valid and produces zero workgroups.

## Inputs

Each input name is local to the expression. It identifies a four-byte scalar
at byte `offset` in uniform buffer `(set, binding)`. Its `type` is `i32`, `u32`,
or `f32`. The bundled runtime reads words in little-endian order. Names are ASCII
identifiers: a letter or underscore, followed by letters, digits or underscores.
The compiler derives aliases from uniform field paths, such as
`frame_resolution_x`; consumers resolve them through `inputs`.

## Grammar and arithmetic

- Input names, parentheses, and casts `i32(expr)`, `u32(expr)`, `f32(expr)`.
- Binary `+`, `-`, `*`, `/`, `%`, plus unary `-`.
- Multiplication, division and remainder bind tighter than addition and
  subtraction. Binary operators associate left to right. Parentheses preserve
  explicit evaluation order.
- Decimal integer literals default to `i32`. Suffixes make the type explicit:
  `7i32`, `7u32`, `7.0f32`, `1.25e2f32`. Float literals require `f32` and must be
  finite.
- Binary operands must have identical types. Use casts or typed literals when
  combining inputs with different types.
- Integer arithmetic is checked at 32 bits. Overflow and division by zero are
  errors. Signed division truncates toward zero; remainder follows the dividend's
  sign.
- Float arithmetic uses `f32` and rejects non-finite results. Float-to-integer
  casts truncate toward zero and reject non-finite or out-of-range values.
  Integer-to-float casts round to `f32`. Casts between `i32` and `u32` preserve bits.
- The final element count must be an integer and nonnegative. Intermediate
  values may be negative.

For example, an eight-pixel tile grid is:

```text
((i32(width) + 7) / 8) * ((i32(height) + 7) / 8)
```

Expressions have no assignments, property access, indexing, or general function
calls. Missing inputs, missing uniform bytes, invalid syntax, and invalid
arithmetic are errors. Parsers bound expression size and nesting.

The Rust descriptor deserializer parses once into the checked evaluator's
internal representation. Hosts must evaluate against current uniforms and ensure
sufficient output capacity before dispatching.
