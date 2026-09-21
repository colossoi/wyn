# Wyn Host Lisp

## Purpose and status

Wyn Host Lisp (WHL) describes the sequential host program that orchestrates a
compiled Wyn program. It records resource allocation and lifetime, scalar
calculations, kernel arguments and dispatches, graphics operations, copies,
branches, loops, device queries, and entry-point inputs and results.

This document specifies the language and its observable semantics. WHL replaces
the JSON pipeline descriptor; preserving JSON schemas, readers, or backward
compatibility is not a requirement. This is a language contract, not a statement
that the compiler already implements it.

WHL makes compiler output readable for debugging and provides a portable
description to guide future ports. Emitting it does not require an interpreter,
a Common Lisp installation, or a runtime that executes WHL. References below to
execution define the program's meaning and the obligations of any future
consumer that executes or translates it.

The compiler's internal representation is an implementation choice. WHL need not
be an intermediate step through which other host-language outputs pass. The
Rust/WGPU backend lowers directly from shared compiler structures without
generating or parsing WHL.

## Compiler and consumer responsibilities

The compiler determines the algorithm: kernels and graphics stages, arguments
and ordering, launch dimensions, intermediate resources, algorithm selection,
and iteration structure. WHL makes those decisions explicit. A consumer realizes
the described operations using its target API.

Host orchestration and device computation are separate contracts. WHL refers to
compiled shader artifacts; it does not contain or interpret their device code.
Porting host operations does not by itself translate WGSL, SPIR-V, or another
device language into a new one.

A consumer may batch commands, pool allocations, or translate the host program
while preserving its semantics. Bind groups, descriptor-set objects, command
buffers, queues, streams, and explicit API barriers are not WHL operations.
Physical shader-interface metadata may identify binding locations; those
locations do not prescribe runtime API objects.

## Language foundation

WHL uses a strict subset of Common Lisp syntax and semantics, extended with the
Wyn declarations and GPU primitives defined here. The subset is positive: only
explicitly permitted forms, functions, reader features, and declaration options
are supported. The rest of Common Lisp is not implicitly available.

The [Common Lisp HyperSpec](https://www.lispworks.com/documentation/HyperSpec/Front/index.htm)
is the reference for included Common Lisp constructs. WHL restricts permitted
syntax and operands rather than assigning familiar forms different meanings.
Standard constructs such as `dotimes` and `and` are built into WHL; no
macroexpander is required.

### Reader and values

A `.wynhost` file is UTF-8 text containing a sequence of S-expressions. The
reader accepts:

- Proper parenthesized lists, including the empty list `()`.
- Symbols, keyword symbols such as `:read`, and the constants `t` and `nil`.
- Signed decimal integers of arbitrary size.
- Decimal floating-point literals such as `3.5`, `3.5f0`, and `3.5d0`.
- Double-quoted strings. Backslash quotes the following character; it does not
  introduce C-style escapes such as a newline escape.
- Quote, written `(quote datum)` or `'datum`.
- Semicolon comments extending to the end of the line.

Ordinary identifiers start with an ASCII letter and continue with ASCII letters,
digits, or hyphens. The explicitly listed operator names, including `let*`,
are also permitted symbols. Keywords use a leading colon followed by an
identifier. Symbols are case-insensitive; generated output uses lowercase.
Strings, including paths and shader entry names, retain case. Keywords and `t`
evaluate to themselves. `nil` denotes both false and the empty list; every other
value, including zero, is true.

Integers are read in base ten. An omitted float exponent marker, or `e` or `f`,
denotes single precision; `d` denotes double precision. WHL uses finite IEEE
binary32 and binary64 host floats. A Common Lisp reference implementation must
provide those formats and use the normal single-precision reader default. Float
overflow and non-finite host results are errors. Shader arithmetic and raw buffer
contents are governed by the device contract instead.

Quoted data may contain permitted literals, symbols, and proper lists. Runtime
values also include exact rationals produced by arithmetic and opaque resource
handles. Ratio literals, dotted lists, character and vector literals, backquote,
comma, escaped symbol names, package qualification other than keywords, custom
readtables, and all `#` reader syntax are excluded. In particular, reading an
artifact cannot execute code through `#.`.

### Evaluation and binding

Ordinary function arguments evaluate from left to right before the call. Bodies
evaluate their forms in order. Variables have lexical scope. Function names and
variable names occupy separate namespaces, as in Common Lisp.

| Form | Meaning and restrictions |
| --- | --- |
| `(quote datum)` | Return literal data without evaluation. |
| `(let ((name init) ...) body...)` | Evaluate initializers left to right in the outer scope, then bind all names for the body. |
| `(let* ((name init) ...) body...)` | Evaluate and bind sequentially; later initializers can use earlier bindings. |
| `(setq name value ...)` | Assign existing lexical variables in order; return the final assigned value, or `nil` for no pairs. |
| `(progn form...)` | Return the final form's result, or `nil` if empty. |
| `(if test then [else])` | Evaluate exactly one branch; the omitted else branch is `nil`. |
| `(cond (test form...) ...)` | Select the first true test; evaluate its body, or return the test value for an empty body. Return `nil` if none matches. |
| `(and form...)` | Stop at the first false value; otherwise return the final value. With no operands, return `t`. |
| `(or form...)` | Stop at the first true value; otherwise return `nil`. |
| `(defun name (parameter...) body...)` | Define a top-level function with required positional parameters only. |

Bindings include initializers. Parameter and binding names are distinct within
a parallel binding list. Constants, keywords, and built-in names cannot be
redefined. `setq` cannot create globals or assign a `dotimes` control variable.
A body returns its final form, or `nil` if empty.

Calls name a built-in or declared function directly. The function call graph
must be acyclic. Recursion, function values, `lambda`, `function`, `funcall`,
`apply`, local function definitions, optional or keyword function parameters,
dynamic binding, declarations, and arbitrary evaluation are excluded. Generated
names avoid Common Lisp's standard function and constant names, so programs can
also be read in a Common Lisp package using the standard vocabulary.

### Arithmetic, comparisons, and lists

The ordinary functions are:

```text
+ - * /
floor ceiling mod min max
= /= < <= > >=
not list
```

They use Common Lisp argument counts and numeric semantics, restricted to real
numbers. Integers and rational arithmetic are exact and do not wrap. In
particular, `(/ 3 2)` produces the exact rational number three halves; `/` is not
truncating integer division. `floor` and `ceiling` accept an optional nonzero
divisor and round the quotient toward negative and positive infinity,
respectively. `mod` uses the remainder associated with `floor`. Division by zero
is an error.

Common Lisp multiple-value behavior is retained, including the remainder returned
by `floor` and `ceiling`. WHL has no forms for capturing secondary values;
argument and initializer positions use the primary value. The external entry
interface likewise observes only the primary return value. See the definitions
of [`/`](https://www.lispworks.com/documentation/HyperSpec/Body/f_sl.htm) and
[`floor` and `ceiling`](https://www.lispworks.com/documentation/HyperSpec/Body/f_floorc.htm).

Mixed rational and float calculations use Common Lisp numeric coercion, with
double precision taking precedence over single precision. Allocation sizes,
offsets, logical lengths, loop counts, and dispatch dimensions must be integers
when consumed. A rational or float is not silently truncated at a GPU boundary.
Typed scalar arguments are checked against the declared device type.

`not` returns `t` exactly when its operand is `nil`. `list` constructs a proper
list from its evaluated arguments. Lists carry argument collections, metadata,
and multiple entry results. WHL provides no list mutation operations.

### Iteration

`dotimes` has the form `(dotimes (name count [result]) body...)`. It evaluates
the integer count once and visits indices from zero up to, but excluding, that
count. Zero or negative counts execute no iterations. The optional result form
runs afterward with the variable bound to the number of iterations; without it
the result is `nil`.

`do` has the form `(do ((name init [step]) ...) (test result...) body...)`.
Initializers evaluate in the outer scope and bind in parallel. The test runs
before each iteration; when true, the result forms evaluate in order. After each
body, step expressions evaluate left to right using the current bindings, then
all their results are assigned in parallel. A variable without a step retains
its value.

```lisp
(do ((remaining n (ceiling remaining 64)))
    ((<= remaining 1) remaining)
  (host-reduce-level remaining))
```

Loop bodies contain compound forms only. Tags, `go`, `return`, `return-from`,
`do*`, and the general `loop` language are excluded. There is no general macro
facility, including `defmacro`, `macrolet`, or `symbol-macrolet`. These rules
restrict the ordinary semantics of
[`dotimes`](https://www.lispworks.com/documentation/HyperSpec/Body/m_dotime.htm) and
[`do`](https://www.lispworks.com/documentation/HyperSpec/Body/m_do_do.htm).

## Programs, entries, and shader artifacts

A program begins with `(define-host-program :version 1)`. Other top-level forms
are `defun` and the declarations below. Declarations associate names with
metadata; they do not allocate resources, dispatch work, or run entry functions.
They can be implemented as registration functions in a Common Lisp reference
environment. Their return values are ignored.

Names are unique within each declaration category. Forward references are
permitted but must resolve within the artifact. Declaration arguments are
literal data, optionally quoted; computations belong in function bodies.
Unknown or duplicate keyword options are errors, including on GPU calls.
Options shown in signatures and declaration examples are required unless stated
otherwise. Notation containing `...` or metavariables describes a schema; it is
not literal generated code.

The host artifact has extension `.wynhost`. A conventional layout is
`program.wynhost` alongside `program.wgsl`. Multiple shader modules and SPIR-V
modules are permitted. Module paths are relative to the host artifact. An
archive may bundle the same files without changing their logical identities.
No JSON companion is part of the contract.

### Modules and compute kernels

```lisp
(define-gpu-module 'kernels :format :wgsl :path "program.wgsl")

(define-gpu-kernel 'reduce-phase1
  :module 'kernels
  :entry "reduce_phase1"
  :workgroup-size '(64 1 1)
  :parameters '((input :buffer :read :element :f32 :stride 4)
                (partial :buffer :write :element :f32 :stride 4)
                (n :u32))
  :abi '((input :storage 0 0)
         (partial :storage 0 1)
         (:scalar-block :storage 0 2 4 ((n 0)))))
```

`define-gpu-module` identifies an artifact with format `:wgsl` or `:spirv`.
`define-gpu-kernel` associates a static WHL name with a compute entry in that
module. Workgroup size is exactly three positive integers matching the device
artifact. A different size requires a separately declared kernel variant;
dispatch calls cannot silently specialize or override it.

### Parameter types and physical interfaces

Parameter lists are ordered, matching values supplied in `:args`. Device scalar
types are `:bool`, `:i8`, `:i16`, `:i32`, `:i64`, `:u8`, `:u16`, `:u32`,
`:u64`, `:f16`, `:f32`, and `:f64`. Integer conversion is range-checked. Floats
use the declared precision with rounding to nearest, ties to even. A boolean
accepts only `t` or `nil`. Target support is checked separately from language
support for a type.

Resource parameter descriptions have these shapes:

```text
(name :buffer access :element type :stride bytes)
(name :buffer access :layout layout)
(name :texture access :dimension dimension :format format :samples count)
(name :sampler :kind kind)
```

Access is `:read`, `:write`, or `:read-write`. Buffer handles are untyped byte
allocations; element type and positive stride describe the shader's
interpretation. An element type is a device scalar, an opaque byte sequence `(:bytes count)`,
`(:vector scalar count)`
with count 2, 3, or 4, or a record layout. A record layout is
`(:size bytes :alignment bytes :fields ((name type offset) ...))`. Field names
are unique. Sizes, strides, offsets, and alignment include device padding and
must be consistent, in bounds, and match the shader artifact. An optional
`:min-bytes` nonnegative integer on a buffer parameter states a fixed minimum
binding size.
Dynamic bounds remain host calculations and shader preconditions.

A texture parameter accepts a view with the declared dimension, format, and
sample count. The ABI specifies sampled or storage usage. Sampled parameters
are read-only and may specify `:sample-type` as `:filterable-float`, `:float`,
`:sint`, `:uint`, or `:depth`. `:format :caller` constrains the caller-supplied
format by that sample type instead of selecting an allocation format.
`:samples :multisampled` requires a caller-selected sample count greater than one.
Color outputs may likewise use `:caller` when the target supplies their format.
Allocated textures always have concrete formats and sample counts. Sampler kinds
are `:filtering`, `:non-filtering`, or `:comparison`.

The required `:abi` list supplies the selected artifact's physical interface:

```text
(parameter kind group binding)
(:scalar-block kind group binding bytes ((parameter offset) ...))
(:scalar-block :push-constant base-offset bytes ((parameter offset) ...))
(parameter :push-constant base-offset bytes)
```

An opaque byte sequence preserves a packed field whose interpretation belongs
to the shader interface. Raw push-constant mappings take a `:host-buffer :read`
parameter with an optional `:min-bytes` bound and pass its bytes without repacking.
Scalar reads may also address host byte spans.

For resources, kind is `:storage`, `:uniform`, `:sampled-texture`,
`:storage-texture`, or `:sampler`. For a scalar block it is `:storage` or
`:uniform`. Each parameter has exactly one resource mapping or scalar field
mapping. Scalar offsets are relative to their block. A consumer packs arguments
into the specified blocks, including padding. Uniform and push-constant layouts
are never inferred from field names. Group/set and binding numbers identify
shader locations, not runtime objects. WGSL uses buffer mappings instead of
push constants. Mappings must match the shader and target.

Resource kinds must agree with parameter types and access. Uniform buffers are
read-only. Physical binding locations are unique within an interface, and scalar
fields must fit their block without overlapping and satisfy the shader's
alignment requirements. Shader parameters cannot name host-only scalar types.

Signed integers use two's complement. Floats use IEEE binary16, binary32, or
binary64. Boolean fields occupy a 32-bit word containing zero or one. Portable
buffer bytes use little-endian encoding; another host architecture must preserve
that representation. Layout offsets and strides refer to these encodings.

### Host entry points

```lisp
(define-host-entry 'sum-array
  :function 'host-sum-array
  :parameters '((input :buffer :read :element :f32 :stride 4)
                (n :integer))
  :results '((result :buffer :read :element :f32 :stride 4 :ownership :owned)))
```

An entry associates a public name with a WHL function. Input names and order
match its parameters. Host scalar types are `:integer` and `:real`, as well as
the device scalar types. Resources use the descriptions above. Entries also
allow `(name :host-buffer access)` for a caller-supplied byte span and
`(name :texture-view usage :dimension dimension :format format :samples count)`
for a view, including a render target. A host `:texture` description can name an
allocation or a view; shader calls always take a view of the required usage.
For `:texture-view`, sampled usage permits reads, storage usage permits reads
and writes, and render-target usage permits attachment reads and writes.

Host entry declarations and resource parameters may include a `:source-name`
string preserving the authored name independently of WHL symbol encoding.

Inputs are borrowed; access annotations state whether the program may modify
them. Resource results require `:ownership :owned` for a newly allocated resource
or `:ownership :borrowed :alias input-name` for an input alias. A result that may
alias several inputs lists their names, for example `:alias (input scratch)`.
Returning an owned view retains its backing allocation. Ownership must be
unambiguous, and the same allocation cannot be returned as two independent owned
results.

With no results the function returns `nil`; with one it returns the value;
with several it returns a list in declaration order. This preserves source-level
result identity independently of generated binding names. Logical lengths and
dimensions are explicit scalar results where needed, not inferred from capacity.

The caller supplies external state, including target dimensions, frame
parameters, host-populated records, input lengths, and capacities that cannot be
computed by the host program. There are no implicit bindings based on familiar
names such as time or resolution. A device-computed length used in host control
flow requires explicit readback.

## Buffers and transfers

Sizes and offsets are nonnegative integer byte counts. Operations must stay
within resources. Capacity and logical array length are distinct: dispatch
padding and spare capacity do not change the logical length.

| Operation | Meaning |
| --- | --- |
| `(gpu-alloc bytes)` | Create an owned device buffer of the requested logical byte capacity, with undefined contents. |
| `(gpu-free resource)` | End ownership of an allocation or sampler; return `nil`. |
| `(gpu-buffer-size buffer)` | Return byte capacity. |
| `(gpu-copy destination destination-offset source source-offset bytes)` | Copy between device buffers, host spans, or one of each; return `nil`. |
| `(gpu-read-scalar buffer offset 'type)` | Read a device scalar and return its host value. |
| `(gpu-write-scalar buffer offset 'type value)` | Encode and write a scalar; return `nil`. |

Zero-byte allocation is valid even if the physical allocation is larger.
Zero-byte copies are no-ops. Other copies require nonoverlapping ranges when
source and destination share an allocation. Device alignment, binding-size, and
usage limits must be satisfied; internal padding is not additional logical
capacity.

`gpu-free` invalidates the allocation and all aliases and texture views. Freeing
a borrowed resource, double-free, and use-after-free are errors. Rebinding with
`setq` does not copy an allocation. Owned temporaries must be freed on every
normal return path unless returned. A consumer may defer destruction or pool
storage but must keep it alive until previously issued work finishes.

Scalar transfers use the types and byte encodings specified for shader
interfaces. Their type argument is a quoted ordinary symbol, such as `'u32` or
`'f32`, corresponding to the declaration keyword `:u32` or `:f32`. The offset
must satisfy the scalar's alignment and leave room for the complete value.
Reads wait for relevant prior GPU writes and make their result
available to subsequent host calculations. Writes capture their host value at
the call and become visible to later GPU operations. Copies to host destinations
complete before subsequent host use; copies from host sources capture their
bytes before later host mutation can change them. Device-to-device copies need
no host-visible wait.

## Compute dispatch

```lisp
(gpu-dispatch 'reduce-phase1
              :groups (list groups 1 1)
              :args (list input partials n))
```

The kernel name is quoted and statically declared. Both keywords are required.
Groups are exactly three nonnegative integers giving workgroup counts, not
invocation counts. Arguments match the declaration in number, order, type,
access, and layout. The operation returns `nil`.

A grid with a zero dimension executes no work; its arguments must still be
valid. Nonempty grids and fixed workgroup sizes must satisfy target limits.
Rounding a grid up requires guards in the shader for invocations outside the
logical domain. Clamping a grid requires corresponding traversal in the kernel,
such as a grid-stride loop. A consumer cannot invent extra reduction phases or
alter iteration counts.

## Textures, views, and samplers

Sampled, storage, and render-target views can share a texture allocation. They
are aliases, not independent resources. Dependencies follow the allocation and
subresources even when views have different parameter names or bindings.

```lisp
(gpu-alloc-texture :dimension :d2
                   :size (list width height 1)
                   :format :rgba16float
                   :mip-levels 1
                   :samples 1
                   :usage '(:storage :sampled :render-target :copy))

(gpu-texture-view texture :usage :sampled
                          :dimension :d2
                          :mip 0 :mip-count 1
                          :layer 0 :layer-count 1)

(gpu-texture-size texture 0)
(gpu-texture-dimension texture 0 'width)
```

Allocation arguments are required. Dimensions are `:d1`, `:d2`, or `:d3`; size
is `(width height depth-or-layers)`. Extents, mip levels, and samples are positive
integers. Unused dimensions are one. A 2D texture can have array layers; a 3D
texture has depth slices. Multisampling requires 2D, one layer, and one mip.
Contents start undefined.

View arguments are required. Dimensions are `:d1`, `:d2`, `:d2-array`, `:cube`,
`:cube-array`, or `:d3`, compatible with the allocation. Mip and layer indices
are zero-based; ranges are in bounds. Nonarray 1D and 2D views select one layer.
Cube views select six square 2D layers; cube arrays select a multiple of six.
A 3D view uses layer zero and layer-count one and includes the mip's depth.
Mip extents halve per level, rounding down but never below one; array layer
counts do not shrink. The number of mips cannot exceed the complete mip chain.
Views retain the allocation's format.

View usage is `:sampled`, `:storage`, or `:render-target`, as permitted by the
allocation's usage list. Render targets select one mip and one 2D layer. Views
have no independent allocation to free. `gpu-free` releases an owned texture
and invalidates its views. Borrowed views cannot free their backing allocation;
a caller receiving an owned view owns that backing allocation.

`gpu-texture-size` returns the selected mip's three extents.
`gpu-texture-dimension` returns the extent selected by the quoted symbol `width`,
`height`, or `depth-or-layers`, allowing scalar calculations without list access.
Both accept an allocation or view, with mip indices relative to the view.
`(gpu-texture-mip-levels texture)` returns the number of mip levels available
through an allocation or view.
For array views, the layer extent is the selected layer count. Target-size
allocations use explicit entry arguments or queries on a passed-in target,
not an implicit window-size policy.

Portable formats are `:rgba8unorm`, `:rgba16float`, `:rgba32float`, `:r32float`,
and `:depth32float`. The first four contain RGBA or R data of the named precision;
`rgba8unorm` maps unsigned bytes to zero through one. `depth32float` is a depth
format and cannot be a storage view or color target. Color sampling returns
floating-point components; missing RGB components are zero and missing alpha
is one. Format, usage, filtering, dimensions, and sample combinations must be
supported by the target. WHL does not require emulating unsupported combinations.

Samplers can be borrowed inputs or explicitly allocated:

```lisp
(gpu-alloc-sampler :kind :filtering
                   :min-filter :linear :mag-filter :linear :mip-filter :nearest
                   :address '(:clamp-to-edge :clamp-to-edge :clamp-to-edge)
                   :lod '(0.0 16.0)
                   :compare nil)
```

All options are required. Filters are `:nearest` or `:linear`. Address modes,
in width/height/depth order, are `:clamp-to-edge`, `:repeat`, or `:mirror-repeat`.
LOD bounds are finite with minimum no greater than maximum. Non-filtering
samplers use nearest filters. Comparison samplers use a depth comparison from
the graphics vocabulary below, excluding `:disabled`. Other samplers require
`:compare nil`. Owned samplers are released with `gpu-free`.

### Texture copies

```lisp
(gpu-copy-texture destination destination-mip destination-origin
                  source source-mip source-origin extent)

(gpu-copy-buffer-to-texture texture mip origin extent
                            buffer offset row-bytes rows-per-image)

(gpu-copy-texture-to-buffer buffer offset row-bytes rows-per-image
                            texture mip origin extent)
```

These operations return `nil`. Texture operands are allocations or views;
coordinates on a view are relative to its selected subresources. Origins and
extents are three-integer lists with nonnegative components. Regions are in
bounds, single-sampled, and copy-capable. Texture-to-texture copies require
identical formats and nonoverlapping regions when they share storage. There is
no format conversion or multisample resolve in this vocabulary.

Buffer operands can be device buffers or host spans. Rows contain texels in
format component order. `row-bytes` is the distance between rows and covers a
full copied row; `rows-per-image` is the distance between slices in rows and
covers the copied height. The addressed bytes must fit the buffer. Host
visibility follows `gpu-copy`; zero extents are no-ops. Target alignment limits
apply, and a consumer may repack through staging memory while preserving the
logical layout.

## Graphics

Graphics declarations associate a vertex and optional fragment entry with a
shared interface and fixed rasterization state. A draw is one ordered host
operation, including attachment reads and writes. Compute operations can produce
textures, vertex and index arrays, or indirect commands used by later draws.
Draws can produce textures used by later compute or graphics work.

### Graphics declarations

```lisp
(define-gpu-graphics 'present
  :vertex '(kernels "fullscreen_vertex")
  :fragment '(kernels "present_fragment")
  :parameters '((image :texture :read :dimension :d2
                        :format :rgba16float :samples 1)
                (filter :sampler :kind :filtering))
  :abi '((image :sampled-texture 0 0)
         (filter :sampler 0 1))
  :vertex-inputs '()
  :color-outputs '((0 :rgba8unorm))
  :depth-format nil
  :samples 1
  :topology :triangle-list
  :front-face :counter-clockwise
  :cull :none
  :fill :fill
  :depth-test :disabled
  :depth-write nil
  :blend :replace
  :color-write t)
```

All options are required. Shader references are `(module-name entry-string)`;
`:fragment nil` denotes no fragment stage. Parameters and ABI mappings cover the
union of both stages' resources and scalars; shared locations must have
compatible meanings. Varyings and built-ins belong to the shader interface and
must agree between stages.

A vertex-input entry is `(location name type stride offset step)`. Type is
`:f32`, `:i32`, `:u32`, or a vector of two to four of those components. Step is
`:vertex` or `:instance`. Each entry consumes one buffer in `:vertices` order.
Stride and byte offset describe its attribute layout; supplying a buffer more
than once permits interleaving. Locations are unique. Color outputs are
`(location format)` pairs matching fragment output locations. Depth format is
`nil` or `:depth32float`. Sample count is a positive integer matching attachments.

| State | Permitted values and semantics |
| --- | --- |
| Topology | `:triangle-list`, `:triangle-strip`, `:line-list`, `:line-strip`, `:point-list`; no primitive restart. |
| Front face | `:clockwise` or `:counter-clockwise`, according to vertex winding in clip-space x/y after division by w, before viewport mapping. |
| Cull | `:none`, `:front`, or `:back`. |
| Fill | `:fill`, `:line`, or `:point`; unsupported target modes are errors. |
| Depth test | `:disabled`, `:never`, `:less`, `:less-equal`, `:equal`, `:greater-equal`, `:greater`, or `:always`; compare incoming depth with stored depth. |
| Depth write | `t` or `nil`; write surviving fragments only. With depth testing disabled this must be `nil`, and no depth read or write occurs. |
| Blend | `:replace`, `:source-over`, or `:add`, applied to each color output. |
| Color write | `t` or `nil`, enabling or disabling all color components. |

Enabled depth testing requires a depth attachment. Replace uses source color and
alpha. Source-over RGB is `source.rgb * source.a + destination.rgb * (1 - source.a)`;
alpha is `source.a + destination.a * (1 - source.a)`. Input RGB is not
premultiplied. Add sums source and destination componentwise, including alpha.
Attachment-format conversion applies after blending.

Clip depth ranges from zero to w, normalized depth from zero to one, and the
framebuffer origin is upper left. A consumer or device lowering must account for
API coordinate differences. Pixel coverage, sampling, and floating-point
rasterization precision follow the device artifact and target; WHL does not
promise bit-identical rendering across APIs. Swapchain acquisition,
presentation, and window management belong to the caller.

### Draw operations and attachments

```lisp
(gpu-draw 'present
          :args (list image-view filter)
          :vertices '()
          :colors (list (list 0 screen :clear :store '(0.0 0.0 0.0 1.0)))
          :depth nil
          :viewport :target
          :scissor :target
          :draw '(:direct 3 1 0 0))
```

The graphics name is quoted and static. All keywords are required. Args follow
parameter order; vertices supply the declared vertex buffers. Colors contain
`(location view load store clear-value)` entries matching all color outputs
exactly once. Depth is `nil` or `(view load store clear-value)`. A depth clear
value is a number from zero to one; a color clear value is a four-number list.

Load is `:load`, `:clear`, or `:discard`: preserve contents, initialize to the
clear value, or make prior contents undefined. Store is `:store` or `:discard`;
the latter makes contents undefined after the draw. Clear value is `nil` unless
load is `:clear`. Blend and depth inputs must be defined where read. Discarding
is not clearing to zero.

Attachment views are render targets with matching extents, declared formats,
and sample count. At least one color or depth attachment is required. Overlapping
attachments, and simultaneous sampling or storage access to a written attachment,
are invalid. Sampling that allocation in a later operation is permitted.

Viewport `:target` covers the target with depth zero to one; an explicit viewport
is `(x y width height min-depth max-depth)`, with finite components, positive
width and height, and `0 <= min-depth <= max-depth <= 1`. Coordinates are
framebuffer pixels. Scissor `:target` covers the target; an explicit scissor is
`(x y width height)` with integer origin and nonnegative integer extents, clipped
to target bounds. The draw returns `nil`.

### Direct, indexed, and indirect draws

The draw value has one of these shapes; use `list` for computed fields:

```text
(:direct vertex-count instance-count first-vertex first-instance)
(:indexed indices index-type index-count instance-count first-index vertex-offset first-instance)
(:indirect commands byte-offset draw-count stride)
(:indexed-indirect indices index-type commands byte-offset draw-count stride)
```

Counts and first indices are nonnegative integers fitting `u32`; vertex offset
fits `i32`. Index type is `:u16` or `:u32`. Index buffers begin at byte zero;
first-index is measured in elements. A vertex index is the loaded unsigned index
plus signed vertex offset; resulting accesses must be in bounds. Direct draws
use first-vertex instead. Vertex and instance stepping include first indices.

Indirect commands are read from a device buffer in command order. Draw count is
a host integer, never inferred from capacity; a device-produced count requires
explicit scalar readback. Command fields are tightly packed:

| Command | Fields in order | Byte size |
| --- | --- | --- |
| Direct indirect | `u32 vertex-count`, `u32 instance-count`, `u32 first-vertex`, `u32 first-instance` | 16 |
| Indexed indirect | `u32 index-count`, `u32 instance-count`, `u32 first-index`, `i32 vertex-offset`, `u32 first-instance` | 20 |

Byte offset and stride are multiples of four; stride is at least the command
size. Addressed commands fit the buffer. Prior writes to commands and indices
are dependencies of the draw. Shader-visible draw index is zero for a direct or
indexed draw, and the zero-based command ordinal within an indirect operation.
A consumer issuing separate API draws must preserve that value.

Zero draw, vertex, index, or instance count produces no primitives. Attachment
load/store semantics still apply once to the whole operation, even with zero
indirect commands; they are not repeated for each command. Out-of-bounds accesses
and unsupported features are errors, not requests to change a draw.

## Device properties and tuning

```lisp
(gpu-device-property 'max-workgroup-size)
(gpu-device-property 'max-workgroups-x)
(gpu-device-property 'shared-memory-per-workgroup)
(tuning-parameter 'reduce-block-size 256)
```

Names are quoted static symbols. Device properties are capabilities exposed by
the selected device and execution environment, not raw API struct fields.
Numeric properties return nonnegative integers:

| Property | Unit or meaning |
| --- | --- |
| `max-workgroup-size` | Maximum total invocations in one workgroup. |
| `max-workgroup-size-x`, `max-workgroup-size-y`, `max-workgroup-size-z` | Per-axis workgroup dimensions. |
| `max-workgroups-x`, `max-workgroups-y`, `max-workgroups-z` | Per-axis workgroup counts in a dispatch. |
| `shared-memory-per-workgroup` | Bytes of workgroup-local storage. |
| `max-buffer-bytes` | Maximum buffer allocation in bytes. |
| `max-texture-dimension-1d`, `max-texture-dimension-2d`, `max-texture-dimension-3d` | Maximum per-axis texture extent for that dimension. |
| `max-texture-array-layers` | Maximum array layer count. |

Unknown properties are errors. Values are stable within an entry invocation.
A queried limit does not imply support for every format or draw feature;
artifacts and operations must also be valid for the target.

`tuning-parameter` returns an environment override or the compiler-provided
default, a host scalar value. A name has one consistent type and default within
a program, and its value is stable within an invocation. The host program must
ensure selected values meet algorithm and device constraints. Tuning does not
override declared workgroup sizes: varying those requires explicit selection
among kernel variants.

## Ordering, visibility, and validation

Host evaluation order defines the logical order of GPU operations along selected
branches and loop iterations. Later reads observe earlier writes to the same
resource, including aliases, texture views, vertex/index buffers, and indirect
commands. Later writes respect preceding reads and writes as well. A consumer
may overlap independent work only if the result is indistinguishable from this
logical order.

Ordinary host arithmetic does not force queued GPU work to complete. Scalar
readback and copies to host memory explicitly make the needed GPU results
available to the host. Returning a device resource does not imply a full-device
wait; any future invocation API must preserve dependencies on pending writes
before subsequent use. Reuse after `gpu-free` must also respect pending work.

These are visibility requirements, not a barrier algorithm. Ordered streams,
queue synchronization, memory barriers, command dependencies, and other target
mechanisms may realize them. There is no explicit barrier or synchronization
primitive in WHL.

A well-formed artifact uses only permitted forms, resolves static references,
and provides complete entry and shader interfaces that match its shader
artifacts. A valid invocation satisfies numeric ranges, capacities, formats,
lifetimes, target capabilities, and access restrictions. Reads require defined
contents; read/write metadata does not initialize memory. Aliasing that conflicts
with an interface or operation is invalid.

Errors must be reported rather than silently truncating integers, skipping
unsupported operations, or guessing missing interface information. A compiler
should reject statically detectable violations. A future executing consumer must
check remaining invocation-dependent constraints. Error recovery and rollback
are not specified. A tool that only emits or displays WHL need not execute it
to check device-dependent conditions, but must preserve the information needed
to understand and validate them.

## Examples

These functions assume matching module, kernel, graphics, and entry declarations.
They illustrate orchestration; device algorithms remain in referenced shaders.

### Two-stage reduction

```lisp
(defun host-sum-array (input n)
  (let* ((groups (ceiling n 64))
         (partials (gpu-alloc (* groups 4)))
         (result (gpu-alloc 4)))
    (if (= n 0)
        (gpu-write-scalar result 0 'f32 0.0)
        (progn
          (gpu-dispatch 'reduce-phase1
                        :groups (list groups 1 1)
                        :args (list input partials n))
          (gpu-dispatch 'reduce-phase2
                        :groups '(1 1 1)
                        :args (list partials result groups))))
    (gpu-free partials)
    result))
```

Here `n` is a nonnegative element count, input contains at least `n` floats, and
the grid fits target limits. Phase one writes one partial per group; phase two
consumes the explicit partial count. Empty input produces the additive identity
without binding the empty partial buffer. The result belongs to the caller.
Logical release of partials is safe even while phase two is queued.

### Host-controlled radix passes

```lisp
(defun host-radix-sort (input scratch n passes)
  (let* ((groups (ceiling n 64))
         (src input)
         (dst scratch)
         (flags (gpu-alloc (* n 4)))
         (offsets (gpu-alloc (* n 4))))
    (if (> n 0)
        (dotimes (pass passes)
          (gpu-dispatch 'radix-classify
                        :groups (list groups 1 1)
                        :args (list src flags n pass))
          (gpu-dispatch 'radix-scan
                        :groups '(1 1 1)
                        :args (list flags offsets n))
          (gpu-dispatch 'radix-scatter
                        :groups (list groups 1 1)
                        :args (list src dst offsets n pass))
          (let ((tmp src))
            (setq src dst)
            (setq dst tmp))))
    (gpu-free flags)
    (gpu-free offsets)
    src))
```

This assumes disjoint writable input and scratch buffers, nonnegative `n` and
`passes`, and kernels with the shown interfaces. The scan kernel computes the
whole scan in one workgroup; a parallel scan can instead use a helper containing
several dispatches. Pass numbers fit the declared scalar type; grids fit target
limits. The result aliases input or scratch according to the passes executed.
Neither borrowed buffer is freed.

### Compute into a texture, then draw

```lisp
(defun host-frame (screen filter width height)
  (let* ((image (gpu-alloc-texture
                 :dimension :d2 :size (list width height 1)
                 :format :rgba16float :mip-levels 1 :samples 1
                 :usage '(:storage :sampled)))
         (storage (gpu-texture-view image :usage :storage :dimension :d2
                                    :mip 0 :mip-count 1
                                    :layer 0 :layer-count 1))
         (sampled (gpu-texture-view image :usage :sampled :dimension :d2
                                    :mip 0 :mip-count 1
                                    :layer 0 :layer-count 1)))
    (gpu-dispatch 'make-image
                  :groups (list (ceiling width 8) (ceiling height 8) 1)
                  :args (list storage width height))
    (gpu-draw 'present
              :args (list sampled filter)
              :vertices '()
              :colors (list (list 0 screen :clear :store '(0.0 0.0 0.0 1.0)))
              :depth nil :viewport :target :scissor :target
              :draw '(:direct 3 1 0 0))
    (gpu-free image)
    screen))
```

Screen is a borrowed `rgba8unorm` render-target view; filter is a borrowed
filtering sampler. Width and height are positive integers. The compute kernel
has an 8 by 8 by 1 workgroup, guards excess invocations, and initializes the whole
image. The sampled view aliases its storage output, so the draw must observe
the compute writes. The temporary texture is logically freed after the draw;
the caller retains the screen for presentation or further work.

## Compiler output selection

`wyn build` selects the host output with `--target-double whl-unknown` (the
default) or `--target-double rust-wgpu`. WHL uses `.wynhost`; Rust/WGPU uses
`.rs`. The shader remains a separate sibling artifact. `--target` selects
SPIR-V or WGSL; its default is SPIR-V for WHL and WGSL for Rust/WGPU. Rust/WGPU
requires WGSL. There is no JSON pipeline output.

The Rust emitter constructs syntax with `quote` and `syn` and formats it with
`prettyplease`. Its generated module uses WGPU 27 and `num-bigint`, `num-integer`,
and `num-traits` for mathematical size arithmetic. Host inputs are explicit
function parameters; resource-name and packed-field tables expose their source
identities and layouts. Scalar readback uses native WGPU polling. No WHL parser
or interpreter is involved.

The compiler publishes host-computable integer allocation expressions, including
scalar interface reads, arithmetic, and logical input lengths. Expressions lifted
from typed 32-bit Wyn arithmetic preserve wrapping using `mod`; size arithmetic
uses mathematical integers. Capacities requiring unsupported scalar conversions
or device-only values remain explicit caller-supplied resources. The source
program still determines logical lengths independently of allocation capacity.
