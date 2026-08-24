# EGIR Display Syntax

Status: draft display contract, version 0.3. Source audit: 2026-08-17.

This document defines the textual syntax used by the EGIR pass inspector. It is
a deterministic, typed serialization of compiler-owned EGIR, intended for
reading and before/after comparison. It is not Wyn source, executable input to
the compiler, or a rendering of Rust `Debug` output.

Version 0.2 is based on an inventory of the whole current EGIR surface, not
only `SegMap`. It covers program declarations, interfaces, resources, value and
place graphs, calls, results and destinations, all effect operations, all three
SOAC families, segmented state, basic blocks, structured-control headers, and
terminators.

## The shape of EGIR

EGIR is not conventional block-only SSA. A graph-bearing body has two layers:

1. an acyclic, unordered sea of pure values and addressable places; and
2. a CFG skeleton whose basic blocks have block parameters, ordered side
   effects, and one terminator.

Calls are complete records in a call-site arena and are anchored in the CFG as
effects. A call is not a pure value node. SOACs are likewise ordered skeleton
effects until expansion. Pure definitions may be shown near the block that uses
them, but indentation must not claim they are block-owned.

The listing therefore has function-shaped declarations and basic blocks, while
retaining an explicit `VALUES` section for the floating graph. This is the
intended middle ground between a graph dump and ordinary code.

## Design requirements

The display syntax must:

- use one operation grammar and one recursive data grammar;
- preserve every compiler-owned variant and field that affects semantics;
- distinguish values, views, places, resources, symbols, blocks, and effect
  tokens lexically;
- retain operand roles, result destinations, lambda signatures, captures,
  resource access, segmented state, and successor arguments;
- retain nested and repeated structure rather than flattening it into a
  variable-length argument list;
- use stable field names and variant names, defined in this document;
- retain complete types, names, literals, and identifiers without truncation;
- order unordered compiler collections deterministically;
- render the same compiler structure identically in both panes; and
- be generated from structured snapshot data, never by parsing diagnostic or
  `Debug` strings.

Pretty-printing may change whitespace and line breaks, but not the grammar or
field ordering.

## Lexical namespaces

| Form | Meaning | Example |
| --- | --- | --- |
| `%name` | ordinary SSA value | `%3`, `%length` |
| `~name` | value-channel storage view | `~input`, `~7` |
| `&name` | addressable place | `&out`, `&12` |
| `$name` | logical or physical resource | `$storage0`, `$binding_0_2` |
| `@name` | function, entry, constant, or external symbol | `@_w_lambda` |
| `^name` | basic block | `^bb0` |
| `!name` | effect-chain token | `!e17` |
| `dialect.op` | operation mnemonic | `arith.add`, `soac.screma` |
| `name` | field, enum variant, or static identifier | `inputs`, `serial` |

The sigil is semantic, not cosmetic. In particular, `~view` and `&place` must
not be printed as ordinary `%values`, even when their internal implementation
uses a `ValueId` or a small integer key.

Names must be unique within their namespace and stable for one snapshot. When
compiler identity survives a pass, both panes must use the same display name.
Source names are preferred when unambiguous; otherwise a stable numeric suffix
is added. Rust wrapper names such as `SemanticResourceRef(...)` are never part
of a display name.

## One recursive data grammar

All operations use named fields whose values come from one recursive data
grammar. Compiler structs become records and compiler enum variants become
variant terms. This is the only nested subsyntax; individual operations do not
invent private mini-languages.

```ebnf
program          ::= program-header declaration* ;
program-header   ::= "PROGRAM" properties? ;

operation        ::= [ definition-tree "=" ] opcode "(" [ fields ] ")" ;
fields           ::= field ("," field)* ;
field            ::= role ":" datum ;

datum            ::= atom
                   | list
                   | record
                   | variant
                   | definition-tree ;
list             ::= "[" [ datum ("," datum)* ] "]" ;
record           ::= "{" [ fields ] "}" ;
variant          ::= identifier [ "(" [ fields ] ")" ] ;

atom             ::= runtime-id
                   | resource-id
                   | symbol
                   | block-target
                   | effect-token
                   | type
                   | literal
                   | "none" ;

runtime-id       ::= value-id | view-id | place-id ;
flow-id          ::= value-id | view-id ;
value-id         ::= "%" identifier ;
view-id          ::= "~" identifier ;
place-id         ::= "&" identifier ;
resource-id      ::= "$" identifier ;
symbol           ::= "@" identifier ;
block-id         ::= "^" identifier ;
effect-token     ::= "!" identifier ;
opcode           ::= identifier "." identifier ;
role             ::= identifier ;

block-target     ::= block-id "(" [ flow-id ("," flow-id)* ] ")" ;
properties       ::= "WITH" record ;

identifier       ::= /* canonical escaped identifier */ ;
integer          ::= /* base-10 integer */ ;
literal          ::= /* canonical bool, number, string, or constant literal */ ;
```

Every operation schema below fixes its field names, their order, and their
allowed datum shape. Optional fields are omitted rather than printed as
`none`, except where absence is itself a meaningful stored variant.

References cannot be disguised as strings. A logical resource is `$r0`, a
callable is `@f`, and a place is `&p`; none may occur inside an opaque attribute
or preformatted detail string.

## Types and representation types

EGIR stores structural types inline. There is no separate EGIR type-definition
arena, so the listing does not invent `TYPE` declarations. Named type
constructors remain names inside the structural type that uses them.

One canonical type printer must cover every `TypeName` currently admitted by
EGIR:

- scalars: `bool`, `iN`, `uN`, `fN`, unit `()`, and no-result `!()`;
- arrays, including every dimension, representation variant, and resource or
  buffer region;
- vectors, matrices, tuples, records, sums, and existentials;
- named types, user variables, inference variables, size variables, and
  skolems when they are present at the inspected checkpoint;
- pointer/address-space types that survive into the checkpoint; and
- opaque GPU types such as textures, samplers, storage images, raster values,
  invocation values, draws, render targets, and fragment outputs.

Array types use this normalized form rather than Rust constructor syntax:

```text
array<element=i32, dimensions=[4], variant=view, region=$r0>
array<element=i32, dimensions=[n], variant=composite, region=none>
```

The `variant` value is one of `view`, `composite`, `bounded`, `virtual`,
`abstract`, or `address_placeholder`. A physical buffer region is written as
`buffer(set: 0, binding: 2)`; a semantic region is its `$resource` reference.

The complete physical representation at a callable boundary is explicit:

```text
value<i32>
view<array=array<...>, region=$r0, access=read_only>
place<pointee=i32, region=function, access=read_write>
```

Place regions are `function`, `workgroup`, `parametric`, `resource($r)`, or
`output`. Place/view access is `read_only`, `write_only`, or `read_write`.
These wrappers are mandatory in function and entry parameter declarations;
inside value definitions the short logical type may be used because the sigil
already states the representation class.

## Result trees and destinations

Results are trees, not flat SSA lists. Product nodes carry logical structure;
leaf routes are either by-value returns or addressable destinations. The
display syntax mirrors that structure exactly.

```ebnf
definition-tree  ::= definition | product-definition ;
definition       ::= runtime-id ":" type
                   | "into" place-id ":" type
                   | "into" bounded-destination ":" type ;
product-definition
                  ::= "product" "(" "type" ":" type ","
                      "fields" ":" "[" [ definition-tree
                      ("," definition-tree)* ] "]" ")" ;
bounded-destination
                  ::= "bounded" "(" "storage" ":" place-id ","
                      "length" ":" place-id ")" ;

function-result-tree
                  ::= function-result | product-function-result ;
function-result   ::= "return" "[" integer "]" ":" type
                    | "into" place-id ":" type
                    | "into" bounded-destination ":" type ;
product-function-result
                  ::= "product" "(" "type" ":" type ","
                      "fields" ":" "[" [ function-result-tree
                      ("," function-result-tree)* ] "]" ")" ;
```

A `FunctionResult` uses `return[n]` for a physical return slot and the named
`&parameter` for a destination parameter. A concrete `ResultBinding` uses a
`%value`, `into &place`, or `into bounded(...)` leaf. The type on every leaf and
the `type` on every product node come from the stored `ResultTree`; they are not
reconstructed from display spelling.

The same concrete definition tree is used on the left side of calls, effects,
and SOACs. `cf.return` carries the concrete tree in its `result` field. This
keeps destination writes visible and prevents a result such as a bounded array
from being mislabeled as a single ordinary SSA value.

## Program declarations

```ebnf
declaration      ::= resource-decl
                   | extern-decl
                   | function-decl
                   | entry-decl
                   | constant-decl ;

resource-decl    ::= "RESOURCE" resource-id ":" type properties ;
extern-decl      ::= "EXTERN" symbol parameter-list "->" type properties ;
function-decl    ::= "FUNCTION" symbol parameter-list "->" type
                    "RESULTS" function-result-tree properties? body ;
entry-decl       ::= "ENTRY" symbol parameter-list "->" type
                    "RESULTS" function-result-tree properties? entry-body ;
constant-decl    ::= "CONSTANT" symbol ":" type properties? body ;

parameter-list   ::= "(" [ parameter ("," parameter)* ] ")" ;
parameter        ::= runtime-id ":" representation-type ;
body             ::= "{" values-section block* "}" ;
entry-body       ::= "{" interface? values-section block* "}" ;
interface        ::= "INTERFACE" record ;
values-section   ::= "VALUES" "{" operation* "}" ;
block            ::= block-id flow-parameter-list control-header? ":" operation* ;
flow-parameter-list
                  ::= "(" [ flow-parameter ("," flow-parameter)* ] ")" ;
flow-parameter   ::= flow-id ":" type ;
control-header   ::= "CONTROL" ("loop" | "selection") "(" fields ")" ;
representation-type
                  ::= "value" "<" type ">"
                    | "view" "<" fields ">"
                    | "place" "<" fields ">" ;
```

Function properties include the stable region identity, optional linkage name,
and `effects: pure | destination_write | general`. External properties include
the linker ABI name. Entry properties include its stable entry identity and
execution model: `vertex`, `fragment`, or
`compute(local_size: [x, y, z])`.

Function parameters appear only in the function header. Block parameters
appear only in their block label. A graph node whose `FuncParam` index is no
longer in the physical ABI is an internal tombstone and is rendered in
`VALUES` as `egir.param_tombstone(index: n)` rather than masquerading as a live
parameter.

### Program-owned checkpoint data

`ProgramData` and `GlobalContext` describe the checkpoint around the code; they
are not instructions and do not belong in a function body. Stable fields are
serialized in the `PROGRAM WITH { ... }` record:

- the compiler phase/checkpoint name;
- pipeline descriptor and `stage_entries` associations;
- the logical resource arena, whose members become `RESOURCE` declarations;
- staged nodes, resident flows, and external inputs when present; and
- selected lowering profile or kernel-plan metadata when present.

A staged node names its owned semantic EGIR body and lists incoming and outgoing
resident flows. Each flow records one producer, all consumers, typed logical
storage, and whether it is published. Physical snapshots replace that topology
with `kernels`; each kernel names one physical body and records stable kernel
identity, dependencies, dispatch domain, and logical-resource accesses.

Identity arenas are used to resolve function, global, and entry IDs to the
symbols printed elsewhere. ID-source counters and phase proof tags are
provenance, not semantic declarations. A checkpoint-specific global context
may appear in a typed metadata inspector, but there is no generic `Debug`
fallback for an unknown context type.

### Entry interfaces

An `ENTRY` has an `INTERFACE` section before `VALUES`. It uses the recursive
data grammar and preserves the compiler's slot structure:

```text
INTERFACE {
  inputs: [input(...)],
  parameter_inputs: [parameter(index: 0, slots: [0, 1])],
  outputs: [output(...)],
  resource_declarations: [resource_decl(...)]
}
```

Each `input` retains `name`, `type`, optional `size_hint`, optional
`$resource`, and exactly one kind:

- `value(decoration: none | builtin(...) | location(...))`;
- `storage(exposure: host(binding(...)) | internal, access: ...,
  length: ...)`;
- `uniform(binding: ...)`;
- `push_constant(offset: ..., size: ...)`;
- `texture(binding: ..., source: external | backing(...) |
  resource(name: ..., backing: ...))`;
- `sampler(binding: ...)`; or
- `storage_image(binding: ..., format: ..., access: ..., size: ...,
  resource_name: ...)`.

Each `output` retains its type, optional `$resource`, routes, and either
`value(destination: plain | builtin(...) | location(...) | target(...))` or
`storage(exposure: ..., length: ...)`.

An output route is
`route(source: source(block: ^bb, value: %v), writers: [...])`. A writer is
`value(%v)` or `effect(!e)`. Interface order is source/ABI order and is never
sorted.

### Logical resources

A resource declaration has these fixed properties:

```text
RESOURCE $r0: i32 WITH {
  origin: host(binding: binding(set: 0, binding: 2), name: "xs"),
  size: fixed_bytes(value: 4096)
}
```

`origin` is `host(...)` or
`compiler(kind: ..., owner: ..., slot: ...)`. Compiler kinds are `staging`,
`gather_handoff`, `reduce_partial`, `scan_block_sums`, `scan_block_offsets`,
`scan_prefixes`, `filter_scratch`, `filter_len_cell`, `filter_flags`,
`filter_offsets`, `filter_scan_block_sums`, `filter_scan_block_offsets`,
`bucket_counts`, `bucket_overflow`, `scalar_handoff`, and
`multi_consumer_array`.

Logical size is one of:

- `fixed_bytes(value: n)`;
- `like_resource(resource: $r, elem_bytes: n, src_elem_bytes: m)`;
- `same_as_dispatch(elem_bytes: n)`; or
- `unspecified`.

Entry-local semantic resource declarations additionally retain `role`, element
type, and logical size. Resource arena order is resource identity order.

### `reify_soacs` checkpoint boundary

The left pane is converted raw EGIR: entry interfaces already contain every
declared output route and ABI size policy, while route writer lists are empty.
The right pane is segmented semantic EGIR. Reification privately links each
route to the semantic SOAC/effect producers reachable from its source, then
records publication uniformly as output slots and resource accesses.

At these two checkpoints the binding on an entry output and the bindings in a
SOAC's resource-access list are host interface identities, not allocated
logical resources. The inspector therefore prints `binding(set: ..., binding:
...)` directly and does not synthesize `RESOURCE` declarations or `$resource`
names for them.

A raw runtime Filter output carries only its capacity rule. Reification adds
`backing: deferred` and `length: implicit`; these are explicit promises that no
backing buffer or length cell has been selected yet. A later
`plan_logical_resources` pass may replace them with `bound(resource: $r)` and
`stored(resource: $r)` when publication and scheduling actually require those
representations.

### `plan_logical_resources` checkpoint boundary

The left pane is optimized semantic EGIR. Storage identities in graphs and
SOAC state are still authored host `binding(...)` values, the program has no
logical-resource arena, and runtime Filter storage may remain `deferred` with
an `implicit` length.

The right pane is `ResourcesAllocated` EGIR. Every executable storage identity
has become a target-independent `$resource`; host bindings survive only in
interface exposure and `RESOURCE ... origin: host(...)` constraints. The pane
therefore includes program-owned `RESOURCE` declarations and entry-local
`resource_decl(...)` sidecars. A logical size may be `fixed_bytes(...)`,
`like_resource(...)`, `same_as_dispatch(...)`, or `unspecified`.

Residency planning may also extract a compiler-owned producer stage. Such
stages use the same body and interface grammar as authored stages and connect
to their consumers through `PROGRAM WITH { stages: [...], flows: [...] }`.
Runtime-array residency binds a Filter's backing and stored length to
`filter_scratch` and `filter_len_cell` resources. This pass does not choose
descriptor bindings for compiler resources, a target recipe, dispatch
geometry, or a physical schedule; those decisions belong to `egir::plan`.

### `plan` checkpoint boundary

The left pane is staged IR: semantic EGIR bodies are connected by typed
resident flows. The right pane is Physical EGIR: `PROGRAM WITH { kernels:
[...] }` is the physical kernel DAG, and each `kernel` body uses physical
bindings and scheduled SOAC state. Kernel dependency entries refer to stable
kernel identities; body names and resource accesses can be matched directly
against the corresponding kernel record.

## Floating values and places

The `VALUES` section serializes the value and place arenas in dependency order,
with stable identity as the tie-breaker. It does not imply execution order.

### `ValueKind` mapping

| Compiler construct | Canonical display |
| --- | --- |
| `Pure { op, operands }` | the corresponding pure opcode below |
| `Union { left, right }` | `%u: T = egir.union(left: %a, right: %b)` |
| `FuncParam` | declaration parameter; tombstone form if outside the ABI |
| `BlockParam` | block-label parameter |
| `CallResult` | leaf on the producing `func.call` definition tree |
| `PlaceLength { place }` | `%n: T = place.length(place: &p)` |
| `PlaceView { place }` | `~v: T = place.view(place: &p)` |
| `Constant(c)` | `%v: T = const.value(value: c)` |
| `SideEffectResult` | leaf on the producing effect definition tree |

`Union` is e-graph equivalence, not a runtime instruction. Its `egir` dialect
makes that distinction visible while preserving the universal operation form.

### `PlaceOp` mapping

| Compiler construct | Canonical display |
| --- | --- |
| `Parameter` | declaration parameter with `&` sigil |
| `View { view }` | `&p: T = place.view(view: ~v)` |
| `AllocaResult` | `&p: T = mem.alloca()` at the owning effect |
| `Index { base, index }` | `&p: T = place.index(base: &base, index: %i)` |
| `Slice { base, start, length }` | `&p: T = place.slice(base: &base, start: %s, length: %n)` |
| `ViewIndex { view, index }` | `&p: T = place.view_index(view: ~v, index: %i)` |
| `OutputSlot { index }` | `&p: T = place.output_slot(index: n)` |

The alloca place is printed once, on `mem.alloca`; it is not duplicated as a
second arena definition.

### Pure operation mapping

Every `PureOp` variant has a fixed opcode and role schema:

| `PureOp` | Opcode and fields |
| --- | --- |
| `Int`, `Uint`, `Float`, `Bool`, `Unit` | `const.int(value: ...)`, `const.uint(value: ...)`, `const.float(value: ...)`, `const.bool(value: ...)`, `const.unit()` |
| `Global` | `symbol.global(symbol: @constant)` |
| `BinOp` | opcode from the binary table; `left`, `right` |
| `UnaryOp` | opcode from the unary table; `operand` |
| `Tuple(n)` | `aggregate.tuple(fields: [...], arity: n)` |
| `Vector(n)` | `aggregate.vector(elements: [...], length: n)` |
| `Matrix { rows, cols }` | `aggregate.matrix(elements: [...], rows: r, cols: c, order: row_major)` |
| `ArrayLit(n)` | `array.literal(elements: [...], length: n)` |
| `ArrayRange { has_step }` | `array.range(start: %s, length: %n[, step: %p])` |
| `Project { index }` | `aggregate.project(base: %v, index: n)` |
| `Index` | `array.index(base: %v, index: %i)` |
| `Materialize` | `value.materialize(value: %v)` |
| `DynamicExtract` | `aggregate.dynamic_extract(base: %v, index: %i)` |
| `Intrinsic { id, overload_idx }` | `builtin.call(id: ..., overload: n, arguments: [...])` |
| `StorageImageLoad(resource)` | `image.load(resource: $r, coordinate: %coord)` |
| `StorageImageStore(resource)` | `image.store(resource: $r, coordinate: %coord, texel: %texel)` |
| `StorageView(Storage(resource))` | `view.storage(resource: $r, offset: %o, length: %n)` |
| `StorageView(Inherited)` | `view.inherited(offset: %o, length: %n, parent: ~v)` |
| `StorageView(Workgroup { id, count })` | `view.workgroup(id: n, count: n)` |
| `ResourceLen(resource)` | `resource.length(resource: $r)` |
| `StorageViewLen` | `view.length(view: ~v)` |

`PureOp`'s call target is uninhabited in EGIR. `OpTag::Call` is therefore
illegal in the pure table; a real call is always `func.call` as described
below.

Binary opcode mapping is exhaustive:

| Variants | Opcodes |
| --- | --- |
| `Add`, `Subtract`, `Multiply`, `Divide`, `Remainder`, `FloorDivide`, `FloorRemainder`, `Power` | `arith.add`, `arith.sub`, `arith.mul`, `arith.div`, `arith.rem`, `arith.floor_div`, `arith.floor_rem`, `arith.pow` |
| `Equal`, `NotEqual`, `Less`, `LessEqual`, `Greater`, `GreaterEqual` | `cmp.eq`, `cmp.ne`, `cmp.lt`, `cmp.le`, `cmp.gt`, `cmp.ge` |
| `LogicalAnd`, `LogicalOr` | `logic.and`, `logic.or` |
| `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `ShiftLeft`, `ShiftRight`, `ShiftRightLogical` | `bit.and`, `bit.or`, `bit.xor`, `bit.shl`, `bit.shr`, `bit.shr_logical` |

Unary `Negate` is `arith.neg`; unary `LogicalNot` is `logic.not`.

## Calls and side effects

Skeleton operations appear in their stored block order. If an effect-chain
edge is present, the last field is
`effect: chain(input: !before, output: !after)`. The explicit token is retained
even though block order normally makes the dependency visually obvious.

### Calls

A call is rendered from its complete `CallSite`, not from the thin skeleton
reference and not from `CallResult` nodes independently:

```text
product(type: ..., fields: [...]) = func.call(
  callee: @f,
  arguments: [%x, ~view, &destination],
  effects: destination_write,
  effect: chain(input: !e0, output: !e1)
)
```

Arguments are in the callee's physical parameter order and retain their
value/view/place sigils. The left-hand definition tree is the call site's
stored `ResultBinding`. `effects` is `pure`, `destination_write`, or `general`.

### Effect operation mapping

| `EffectOp` | Canonical display |
| --- | --- |
| `Call { site }` | `func.call` using the complete call-site schema above |
| `Op { tag }` | the corresponding pure-op opcode and field schema, anchored in the block and with its result binding/effect edge |
| `Alloca { result }` | `&p: T = mem.alloca(effect: ...)` |
| `Load { place }` | `%v: T = mem.load(place: &p, effect: ...)` |
| `Store { place }` | `mem.store(place: &p, value: <operand>, effect: ...)` |
| `Atomic { place, op }` | the atomic opcode below with `place`, value operands, and `effect` |
| `ControlBarrier` | `sync.control_barrier(effect: ...)` |

Atomic mapping is exhaustive:

| `AtomicOp` | Opcode | Value fields |
| --- | --- | --- |
| `Load` | `atomic.load` | none |
| `Add` | `atomic.add` | `value` |
| `SignedMin` | `atomic.smin` | `value` |
| `UnsignedMin` | `atomic.umin` | `value` |
| `SignedMax` | `atomic.smax` | `value` |
| `UnsignedMax` | `atomic.umax` | `value` |
| `And`, `Or`, `Xor` | `atomic.and`, `atomic.or`, `atomic.xor` | `value` |
| `Exchange` | `atomic.exchange` | `value` |
| `CompareExchange` | `atomic.compare_exchange` | `expected`, `replacement` |

The enclosing side effect's structured result, if any, remains on the left
side. A result is never summarized as the text `effect result`.

## SOACs

The canonical semantic EGIR SOAC opcodes are exactly:

- `soac.screma`;
- `soac.filter`; and
- `soac.hist`.

`SegMap`, `SegScan`, and `SegRed` are not distinct semantic EGIR variants.
`SegMap` is a derived classification of a `Screma` with empty `scans` and
`reductions`; a `Screma` may also contain scans, reductions, mapped values, or
a mixture. The UI may show a deterministic badge such as “map” or “reduce”,
but the canonical listing remains `soac.screma` and never erases the complete
form.

Every semantic SOAC retains its stable semantic operation identity in the `id`
field. The viewer does not generate prose summaries of fusion or other pass
effects.

### Shared SOAC terms

A typed co-iterated input is:

```text
soac_input(
  operand: %value | ~view,
  array: <structural array type>,
  dimensions: [0, ...],
  layout: composite | storage_aos | structure_of_arrays | generated |
          strided_fields(element_stride_bytes: n, field_offsets_bytes: [...])
)
```

The operand and `SoacInputType` at the same slot are serialized together. This
eliminates the current display ambiguity between a variable number of raw
operands and their meanings.

A lambda is exactly one of:

```text
identity(parameters: [T, ...], results: [U, ...])

region(
  symbol: @body,
  captures: [%v, ~view, &place, ...],
  parameters: [T, ...],
  results: [U, ...]
)
```

The callable symbol is a direct region identity, not an SSA function pointer or
closure operand. Captures use the complete operand vocabulary and remain inside
the lambda term. Parameter and result types are the lambda's stored types, even
when the target function declaration could be consulted separately.

A segmented space is a list of dimensions. Each dimension is exactly one of:

- `fixed(value: n)`;
- `push_constant(value: %n, offset: bytes)`;
- `resource_length(view: ~v, binding: binding(...), elem_bytes: n)`; or
- `value(value: %n)`.

A segmented resource entry is
`resource_access(binding: binding(...), access: read | write | read_write)`.
These terms name semantic reads and writes of host-visible bindings; they do
not imply that compiler-owned storage has been allocated.

Every side-effect result uses the canonical result-binding tree. The display
flattens its destination leaves while retaining each logical product path:

```text
result(
  path: [0, ...],
  type: T,
  destination: return_value(value: %v)
             | place(storage: &p)
             | bounded_place(storage: &data, length: &length)
)
```

`return_value` is the value channel. `place` and `bounded_place` are explicit
destination-passing routes; a bounded result has separate data and length
places. This binding is authoritative. A SOAC does not carry a parallel list
of output-view operands.

### `soac.screma`

The fixed field schema is:

```text
<result-binding> = soac.screma(
  id: <semantic-op-id>,
  inputs: [soac_input(...)],
  results: [result(...)],
  form: screma(
    pre: <lambda>,
    scans: [scan(operator: <lambda>, neutral: [%v, ...])],
    reductions: [
      reduce(operator: <lambda>, neutral: [%v, ...], commutative: true | false)
    ],
    post: <lambda>
  ),
  result_state: [result(field: n, ownership: fresh | unique_input)],
  state: <screma-state>,
  effect: chain(...)
)
```

`result_state` retains source ownership capability only. Concrete return-value
or place routing lives in `results`. Both follow the canonical Screma result
order: reduction components, then post-lambda results. Compact Screma operands
contain only the typed co-iterated inputs.

At the semantic checkpoint, `screma-state` is either `serial` or:

```text
segmented(
  space: [<seg-extent>, ...],
  output_slots: [n, ...],
  resources: [resource_access(...), ...]
)
```

Scans and reductions are indexed by list position. Multi-component neutral
values stay grouped with their operator. No scan/reduction/capture value is
folded into the `inputs` list.

### `soac.filter`

The fixed field schema is:

```text
<result-binding> = soac.filter(
  id: <semantic-op-id>,
  inputs: [soac_input(...)],
  results: [result(...)],
  body: filter_body(map: <lambda>, predicate: <lambda>),
  state: segmented(
    space: [<seg-extent>, ...],
    output_slots: [n, ...],
    resources: [resource_access(...), ...],
    output: <filter-output>
  ),
  effect: chain(...)
)
```

`filter-output` is one of:

- `local(capacity: type(value: T), ownership: fresh | unique_input)`; or
- `runtime(capacity: like_input(input: n), backing: deferred |
  bound(binding: binding(...)), length: implicit |
  stored(binding: binding(...)))`.

In converted raw EGIR the same operation has
`state: raw(output: <filter-output>)`. A raw runtime output contains the
`capacity` field but omits `backing` and `length`; those two deferral markers
are established by reification. A local output already owns its complete
capacity and ownership policy, so reification preserves those fields.

Map and predicate are separate lambda roles. The predicate is not encoded as a
magic trailing capture or argument.

### `soac.hist`

The fixed field schema is:

```text
<result-binding> = soac.hist(
  id: <semantic-op-id>,
  inputs: [soac_input(...)],
  results: [result(...)],
  form: histogram(
    bucket: <lambda>,
    operations: [
      hist_op(
        emission: always | guarded,
        shape: [%extent, ...],
        race_factor: %factor,
        destinations: [~view, ...],
        update: <hist-update>
      )
    ]
  ),
  state: serial | segmented(space: [<seg-extent>, ...]),
  effect: chain(...)
)
```

`hist-update` is exactly one of:

- `ordered_overwrite(value_types: [T, ...])`;
- `reduce(operator: <lambda>, neutral: [%v, ...])`; or
- `bucket_insert(value_types: [T, ...], counts: ~counts,
  overflow: ~overflow, capacity: %capacity)`.

Histogram operation order and component order are preserved. Shape, race
factor, destination views, reducer captures/neutrals, and bucket-insert
bookkeeping stay in their owning `hist_op`; none are emitted as one unlabeled
referenced-value list.

### Other phase states

The same opcodes and recursive grammar apply when a later pass exposes
scheduled or physical EGIR. Only the stored `state` variant changes:

- Screma: `serial` or
  `segmented(space: ..., output_slots: ..., resources: ...)`;
- Filter: `loop(space: ..., storage: ...)` or
  `pipeline(space: ..., storage: runtime(...), plan:
  parallel(stage: flags | scan | scatter, buffers: ..., scan_workgroup_width: n))`;
- Hist: `serial`,
  `atomic(space: ..., operations: [direct(... ) | compare_exchange])`, or
  `bucket(space: ..., stage: init | insert | finish, topology: ...)`.

Raw phase state is rendered explicitly when it owns data (for example a raw
Filter output); zero-field raw state is the enum atom `raw`. Physical resources
still use the `$` namespace in checkpoints that actually contain them. A
descriptor binding at the reification boundary remains a structured
`binding(...)` term and is never disguised as a physical resource.

## Basic blocks and control flow

Block labels define their parameters:

```text
^bb1(%index: i32, ~input: array<...>):
```

`FlowValueId` admits only CFG-carried ordinary/view values; places are not valid
branch arguments. The sigil is selected from the underlying representation.

A structured-control header is printed on the block label because it belongs
to the block, not to a side effect:

```text
^loop(%i: i32) CONTROL loop(merge: ^done(), continue: ^continue()):
^if() CONTROL selection(merge: ^merge()):
```

The only header variants are `loop` and `selection`. The label's control target
uses the same block-target grammar as terminators.

Terminator mapping is exhaustive:

| Terminator | Canonical operation |
| --- | --- |
| `Return(None)` | `cf.return()` |
| `Return(Some(result))` | `cf.return(result: <definition-tree>)` |
| `Branch` | `cf.br(target: ^bb(arguments...))` |
| `CondBranch` | `cf.cond_br(condition: %c, then: ^a(args...), else: ^b(args...))` |
| `Unreachable` | `cf.unreachable()` |

Terminators are the final operation in a block. Successor arguments remain
inside their target term and align positionally with the target's block
parameters.

## E-graph and provenance metadata

The canonical snapshot retains graph metadata without pretending it executes:

- `egir.union` is the `ValueKind::Union` definition described above;
- `egir.alias(value: %old, canonical: %new)` records a non-empty canonical
  replacement;
- `egir.result_origin(value: %owner, result: <definition-tree>)` records each
  stored result origin; and
- source spans, stable arena IDs, and before/after relation IDs live in typed
  provenance fields on the display DTO.

Span and identity fields may be hidden in the normal listing to reduce noise,
but they must remain available to selection, the inspector, and copied
canonical metadata. Hiding metadata is a presentation option, not permission to
derive or discard semantic fields.

## Canonical ordering

Ordering rules are:

1. top-level resources by resource identity;
2. externs, functions, entries, and constants in stored program order;
3. `VALUES` definitions in dependency order, stable identity as tie-breaker;
4. CFG blocks in reverse postorder from the entry block, then unreachable
   blocks by stable identity;
5. side effects in stored block order;
6. lists that encode ABI, result, dimension, component, or source order remain
   in that order; and
7. true maps and sets are sorted by their canonical rendered key.

No operation-specific renderer may reorder a list merely to produce a smaller
diff.

## Formatting and elision

Canonical text is lossless. The renderer must never shorten a token or datum
with `...` or `…`. Long lines may scroll horizontally. The pretty-printer may
break after `(`, between fields, inside lists, and between product fields while
preserving the same parse tree.

Optional folding is presentation state. A folded value must be visibly marked,
must expand to the canonical text, and must not change copied output. Selection
may add a background or outline, but may not make unrelated code unreadable.

## Structured WASM snapshot

The WASM boundary must expose the syntax tree, not the finished text:

```text
DisplayProgram {
  phase,
  properties,
  resources,
  externs,
  functions,
  entries,
  constants,
  relations
}

DisplayBody {
  parameters,
  result_type,
  result_routes,
  values,
  blocks
}

DisplayOperation {
  definitions,
  opcode,
  fields,
  provenance
}

DisplayDatum =
  Reference(Value | View | Place | Resource | Symbol | Block | Effect)
  | Type(...)
  | Literal(...)
  | List([...])
  | Record([...])
  | Variant { name, fields }
```

SOAC DTOs may remain strongly typed in Rust and convert into `DisplayDatum` at
the final boundary. JavaScript is responsible only for stable naming,
escaping, layout, folding, hit testing, and synchronized selection. It must not
parse `Debug`, infer operand roles from positions, recover types from labels, or
synthesize natural-language pass summaries.

## Exhaustiveness rule

The serializer must match compiler enums exhaustively and must not use a
wildcard arm that falls back to `Debug`. Adding a new `ValueKind`, `PlaceOp`,
`PureOp`, `EffectOp`, `AtomicOp`, `Soac`, segmented extent, result destination,
control header, terminator, interface kind, or phase-state variant must cause a
compile failure until this contract and its structured serializer are updated.

That rule is the mechanism that keeps the display syntax disciplined as EGIR
evolves.
