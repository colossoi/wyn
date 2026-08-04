# Wyn Language Specification

## Introduction

Wyn is a functional, array-centric language for GPU programming.
Function values are erased before execution, so programs run on GPU
targets that have no first-class function pointers. The array surface
provides one array kind for external inputs, intermediate values, and results.

### Design Goals

- **GPU-targeted**: language constraints (regular arrays, no
  recursion, no first-class function values) keep programs compatible
  with massively parallel hardware.
- **Array-oriented**: first-class support for multi-dimensional arrays
  and array operations.
- **Type-safe**: static type checking with type inference.
- **Functional**: immutable data structures and expression-based
  computation.
- **Modular**: encapsulation and reuse through first-class modules —
  signatures hide implementation details, and parameterized modules
  generalize components across types.

### Key Features

- **Second-order array operators**: `map`, `reduce`, `scan`, `filter`,
  and related combinators are the primary way to express bulk
  operations on arrays.
- **Uniqueness types**: `*T` marks a value as consuming-only, letting
  `arr with [i] = x` mutate in place when the source is unique and
  copy when it is not.
- **Size-typed arrays**: array lengths participate in the type system;
  `def f(xs: [n]i32) [n]i32` declares a function whose output has the
  same length as its input.
- **Unified pipeline entries**: a host invokes one `entry`; ordinary
  values and array operators express computation within it, while special
  invocation forms introduce rasterization and fragment processing.

### Program Structure

Where other shader languages iterate, Wyn transforms.

Arrays are the primary data structure, aligning with the regular,
data-parallel organization of modern GPU hardware. Operators such as
`map`, `reduce`, `scan`, and `filter` consume and produce arrays,
while function values specialize each operator invocation by
capturing values from the surrounding scope.

Most non-trivial programs are compositions of these operators. A program
describes transformations of values rather than a sequence of storage
operations. Parallel evaluation is permitted only where the relevant
language construct permits it.

A Wyn program is a sequence of declarations. The smallest interesting
program is a single host-visible entry:

```wyn
entry double(arr: []f32) []f32 = map(|x| x * 2.0, arr)
```

`entry` marks the root operation visible to the host. It does not name
a GPU pipeline stage. Anything that is not a root entry is an ordinary
function or value, defined with `def`:

```wyn
def gravity: f32 = 9.81

def step(dt: f32, v: f32) f32 = v + gravity * dt
```

Functions are first-class within the program — they can be passed to
higher-order operators like `map` and `reduce` — but function values
are erased before execution and do not exist at runtime.

Arrays are the primary aggregate. Sizes participate in the type
system, so a function that takes an `[n]f32` returns an array whose
length is bound to that same `n`:

```wyn
def normalize(xs: [n]f32) [n]f32 =
  let total = reduce(|a, b| a + b, 0.0, xs) in
  map(|x| x / total, xs)
```

A graphics entry contains the relationship among its array computation,
vertex processing, rasterization, and fragment processing. Stage callbacks
are ordinary function expressions; their role follows from the special
invocation form to which they are passed:

```wyn
entry triangle(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      triangle_position(vertex.vertex_index),
      @[1.0, 0.0, 0.0])) in
  shade(target, covered,
    |fragment| @[fragment.value.x,
                  fragment.value.y,
                  fragment.value.z,
                  1.0])
```

The observable operation is the source `entry` as a whole. See Unified
Pipeline Entries and Stage Invocation.

A Wyn program can span multiple files. Each file is implicitly a
module: declarations at file scope are members of that module. Files
reference each other with `import`, which loads a sibling file and
binds its declarations under the imported name; `open` brings a
module's members into the current scope unqualified. Modules can also
be defined inline with `module m = { ... }`. Module types describe a
module's interface; parameterized modules take other modules as
arguments.

A small standard library is automatically loaded. Top-level
declarations from its files — including the second-order array
operators (`map`, `reduce`, `scan`, `filter`, …) — are available
unqualified throughout the program. The standard library also defines
per-type modules (`i32`, `f32`, `bool`, …) that group operations on
their type; programs can `open` such a module to bring its members
into scope or address them by qualified name (`f32.sqrt`, `i32.abs`).

## Grammar Notation

This specification uses Extended Backus-Naur Form (EBNF) notation:

- `::=` means "is defined as"
- `|` separates alternatives
- `*` means zero or more repetitions
- `+` means one or more repetitions
- `?` means optional (zero or one)
- `()` groups elements
- `[]` denotes literal square brackets in the language
- `""` denotes literal strings

---

## Identifiers and Keywords

### Grammar

```ebnf
name         ::= letter constituent* | "_" constituent+
constituent  ::= letter | digit | "_" | "'"
quals        ::= (name ".")+
qualname     ::= name | quals name
symbol       ::= symstartchar symchar*
qualsymbol   ::= symbol | quals symbol | "`" qualname "`"
fieldid      ::= decimal | name
symstartchar ::= "+" | "-" | "*" | "/" | "%" | "=" | "!" | ">" | "<" | "|" | "&" | "^"
symchar      ::= symstartchar | "."
constructor  ::= "#" name
```

### Description

A `name` is an unqualified identifier used at definition sites. A
`qualname` is the dotted form used to reference something inside a
module. A `symbol` (or `qualsymbol`) names an operator.

A bare `_` is not a `name`. It is reserved for wildcard patterns and
call-section placeholders. Identifiers beginning with `_`, such as
`_value`, remain ordinary names.

Constructor names of sum types are identifiers prefixed with `#`,
with no whitespace between the `#` and the name. Record fields use
`fieldid`, which is either a name or a decimal.

Wyn has three distinct namespaces:

- **Terms**: variables, functions, and modules.
- **Module types**: module type definitions.
- **Types**: type names and type constructors.

Modules (including parameterized modules) and values share the term
namespace.

### Reserved Names

Reserved names and symbols may appear only where the grammar
explicitly admits them; they cannot be bound in definitions.

**Reserved identifiers:**
```
case, def, do, else, entry, extern, false, for, functor, if,
import, in, include, let, loop, match, module, open, sig, then,
true, type, while, with
```

**Reserved symbols:**
```
=    ->    |    |>
```

---

## Primitive Types and Values

### Grammar

```ebnf
literal ::= intnumber | floatnumber | "true" | "false"

int_type   ::= "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64"
float_type ::= "f16" | "f32" | "f64"

intnumber   ::= (decimal | hexadecimal | binary) [int_type]
decimal     ::= decdigit (decdigit |"_")*
hexadecimal ::= 0 ("x" | "X") hexdigit (hexdigit |"_")*
binary      ::= 0 ("b" | "B") bindigit (bindigit | "_")*

floatnumber      ::= (pointfloat | exponentfloat) [float_type]
pointfloat       ::= [intpart] fraction
exponentfloat    ::= (intpart | pointfloat) exponent
intpart          ::= decdigit (decdigit |"_")*
fraction         ::= "." decdigit (decdigit |"_")*
exponent         ::= ("e" | "E") ["+" | "-"] decdigit+

decdigit ::= "0"..."9"
hexdigit ::= decdigit | "a"..."f" | "A"..."F"
bindigit ::= "0" | "1"
```

### Description

Boolean literals are written `true` and `false`. The primitive types in Wyn are:
- **Signed integer types**: `i8`, `i16`, `i32`, `i64`
- **Unsigned integer types**: `u8`, `u16`, `u32`, `u64`
- **Floating-point types**: `f16`, `f32`, `f64`
- **Boolean type**: `bool`

Each primitive type name doubles as a **polymorphic conversion
builtin**: applied as `T(value)` it dispatches on `value`'s inferred
type to the corresponding per-type catalog entry. The vec shorthand
names (`vec2i32`, `vec3f32`, `vec4u32`, …) extend the same scheme to
vectors — `vecNT(v)` is componentwise. See *Type Conversions* below
for the full dispatch table and the dot-form `T.source(value)` compatibility syntax.

### Numeric Literals

Numeric literals can be suffixed with their intended type. For example
`42i8` is of type `i8`, and `1337e2f64` is of type `f64`. If no suffix
is given, the type of the literal will be inferred based on its use.
If the use is not constrained, integral literals will be assigned type
`i32`, and decimal literals type `f32`.

**Integer formats:**
- **Decimal**: `42`, `1000`, `42i8`
- **Hexadecimal**: `0xFF`, `0x1A2B` (prefixed with `0x`)
- **Binary**: `0b1010`, `0b11110000` (prefixed with `0b`)

**Float formats:**
- **Decimal**: `3.14`, `1.5e-10`, `2.0f32`

Underscores may be used as digit separators in numeric literals for
readability (e.g., `1_000_000`, `0xFF_FF_FF`).

---

## Type Conversions

Wyn performs no implicit numeric coercion — conversions between
distinct numeric types must be written explicitly. The recommended
form is **constructor-style**: `T(value)` invokes `T`'s conversion
of `value`, with the source type read from `value`'s inferred type.

```wyn
let xi: i32 = i32(2.5f32)        -- f32 -> i32
let xf: f32 = f32(7i32)          -- i32 -> f32
let xu: u32 = u32(-1i32)         -- i32 -> u32 (bitcast)
let v:  vec2i32 = vec2i32(uv)    -- vec2f32 -> vec2i32, componentwise
```

For each `(target, source)` pair the compiler resolves an overload
from the per-type conversion catalog (`T.source`) — e.g.
`i32(x: f32)` dispatches to the same `_w_intrinsic_i32_from_f32`
that `i32.f32(x)` does. The dispatch table:

| Source        | Available targets                  | Lowering          |
|---------------|------------------------------------|-------------------|
| `fN`          | `i8`, `i16`, `i32`, `i64`          | float→signed-int  |
| `fN`          | `u8`, `u16`, `u32`, `u64`          | float→unsigned-int|
| `iN`          | `f16`, `f32`, `f64`                | signed-int→float  |
| `uN`          | `f16`, `f32`, `f64`                | unsigned-int→float|
| `iN` ↔ `iM`   | same-signedness, different widths  | sign-extending convert |
| `uN` ↔ `uM`   | same-signedness, different widths  | zero-extending convert |
| `iN` ↔ `uN`   | same width                         | bitcast           |
| `iN` ↔ `uM`   | different widths                   | sign/zero convert |

Vec shorthand: `vecNT(v)` desugars to a `@[T(v.x), T(v.y), ...]`
literal, applying the scalar `T(...)` conversion to each component.
Arity is read from the constructor name (`vec2`/`vec3`/`vec4`); the
arg must be a vec of matching arity, or it's a type error.

### Dot-form compatibility syntax (`T.source(value)`)

The original surface — `i32.f32(x)` — remains supported and resolves
to the same catalog entry as `i32(x)`. The constructor form is
preferred because it doesn't require restating the source type. Vec
conversions have no dot-form equivalent; use `vecNT(v)`.

Broadcasting between vector and scalar in arithmetic (`v + 1.0`)
performs no implicit numeric conversion — both sides must already
share an element type (see the operators section for the broadcast
rules).

---

## Compound Types and Values

### Grammar

```ebnf
type ::= qualname
       | array_type
       | tuple_type
       | record_type
       | sum_type
       | function_type
       | type_application
       | unique_type
       | existential_size

tuple_type ::= "(" ")" | "(" type ("," type)+ [","] ")"

array_type ::= "[" [exp] "]" type

sum_type ::= sum_variant ("|" sum_variant)*
sum_variant ::= constructor [ "(" type ("," type)* [","] ")" ]

record_type ::= "{" "}" | "{" fieldid ":" type ("," fieldid ":" type)* [","] "}"

type_application ::= qualname "<" type_argument ("," type_argument)* ">"
type_argument    ::= "[" [dim] "]" | type
unique_type      ::= "*" type

function_type ::= param_type "->" type
param_type    ::= type | "(" name ":" type ")"

stringlit  ::= '"' stringchar* '"'
stringchar ::= <any source character except "\" or newline or double quotes>

existential_size ::= "?" ("[" name "]")+ "." type
```

### Description

Compound types can be constructed based on the primitive types. The
Wyn type system is entirely structural, and type abbreviations are
merely shorthands. The only exception is abstract types whose
definition has been hidden via the module system.

#### Tuple Types

A tuple value or type is written as a sequence of comma-separated
values or types enclosed in parentheses. For example, `(0, 1)` is a
tuple value of type `(i32, i32)`. The elements of a tuple need not
have the same type – the value `(false, 1, 2.0)` is of type
`(bool, i32, f32)`. A tuple element can also be another tuple, as in
`((1,2),(3,4))`, which is of type `((i32, i32), (i32, i32))`. A
tuple cannot have just one element, but empty tuples are permitted,
although they are not very useful. Empty tuples are written `()` and
are of type `()`.

#### Array Types

An array value is written as a sequence of zero or more
comma-separated values enclosed in square brackets: `[1, 2, 3]`. An
array type is written as `[d]t`, where `t` is the element type of the
array, and `d` is an expression of type `i64` indicating the number of
elements in the array. We can elide `d` and write just `[]` (an
anonymous size), in which case the size will be inferred.

As an example, an array of three integers could be written as
`[1, 2, 3]`, and has type `[3]i32`. An empty array is written as `[]`,
and its type is inferred from its use. When writing Wyn values for
testing purposes, empty arrays are written `empty([0]t)` for an empty
array of type `[0]t`.

**Multi-dimensional arrays** are supported in Wyn, but they must be
regular, meaning that all inner arrays must have the same shape. For
example, `[[1,2], [3,4], [5,6]]` is a valid array of type `[3][2]i32`,
but `[[1,2], [3,4,5], [6,7]]` is not, because we cannot come up with
integers `m` and `n` such that `[m][n]i32` describes the array. The
restriction to regular arrays is rooted in low-level concerns about
efficient compilation.

#### Vector Types

A vector is a fixed-width aggregate of scalar components, written
`vecNT` where `N` is the component count (2, 3, or 4) and `T` is the
scalar element type — e.g. `vec3f32`, `vec4i32`. Vector literals use
the `@[...]` syntax: `let v: vec3f32 = @[1.0, 2.0, 3.0]`. See Vector
Types below for the full naming table, constructors, and swizzles.

#### Matrix Types

A matrix is a fixed-shape aggregate of scalar components, written
`matRxCT` (rectangular) or `matNT` (square, equivalent to `matNxNT`).
Supported dimensions are R, C ∈ {2, 3, 4}; element types are the
primitive scalar types. Matrix literals use the
`@[[...], [...], ...]` syntax. See Matrix Types below for naming and
construction details.

#### Sum Types

Sum types are anonymous in Wyn, and are written as the constructors
separated by vertical bars. Each constructor consists of a
`#`-prefixed name and an optional parenthesised payload — a
comma-separated list of payload types (or sub-patterns /
sub-expressions, depending on the syntactic position). A constructor
with no payload is written bare, with no parentheses.

Because sum types are structural, constructor names are not globally
unique — they are tags inside a sum type, not declarations. The same
name `#left` belongs to infinitely many possible sum types
(`#left(i32) | #right(f32)`, `#left(bool) | #middle | #right`, …), so
a bare constructor expression like `#left(3)` is ambiguous in
isolation. The type checker resolves it from context — the expected
type at the use site, the type of an argument it's being passed as,
or an explicit annotation. When context doesn't pin a single sum
type down, an annotation is required:

```wyn
let x: #left(i32) | #right(f32) = #left(3)
```

**Note:** Implementations are not required to optimize the
representation of sum-typed values. A sum-typed value may carry
storage for every constructor's payload; sum types with multiple
large-payload constructors can be costly.

#### Record Types

Records are mappings from field names to values, with the field names
known statically. A tuple behaves in all respects like a record with
numeric field names starting from zero, and vice versa. It is an
error for a record type to name the same field twice. A trailing
comma is permitted.

Records are structural: a field name `x` does not identify a single
record type, so a bare projection `r.x` does not fully determine
`r`'s type by itself. As with sum constructors, the type checker
uses context — the inferred type of `r` from the surrounding
expression, or an explicit annotation on `r` — to pick a single
record type. When context doesn't suffice, an annotation is
required:

```wyn
let r: { x: f32, y: f32 } = make_point() in r.x
```

#### Function Types

Functions are classified via function types, but they are not fully
first class. See Higher-order functions for the details.

#### Generic Type Applications

A generic type application names a type constructor followed by a fully
saturated, comma-separated argument list in angle brackets. Ordinary type
arguments are types; size arguments are enclosed in brackets.

```wyn
pair<i32, bool>
two_vecs<[4], f32>
raster<vec2f32>
```

Type constructors are first-order. A type constructor cannot be used without
its arguments, partially applied, passed as a value, or received through a
type parameter. Each constructor fixes the number and kind of its arguments.

#### String Literals

Wyn has no first-class string type. A string literal (`"..."`) is a
lexical token accepted only in two non-expression positions: the path
in `import "..."` and the linkage name in `#[linked("...")]`. Writing
a string where a value is expected is a syntax error.

#### Existential Size Quantifiers

An existential size quantifier brings an unknown size into scope
within a type. It is used to describe results whose size is not
statically known — most commonly, the output of `filter`:

```wyn
def is_even(x: i32) bool = x % 2 == 0

def evens(arr: [8]i32) ?k. [k]i32 =
    filter(is_even, arr)
```

The return type `?k. [k]i32` says "for some size `k`, an array of
length `k`". A caller of `evens` receives an array of
unknown-but-fixed length; `k` is in scope within the type but cannot
be statically determined by the caller.

---

## Declarations

### Grammar

```ebnf
dec ::= def_bind | type_bind | mod_bind | mod_type_bind | entry_bind
      | "open" mod_exp
      | "import" stringlit
      | "local" dec
      | "#[" attr "]" dec
```

### Description

A Wyn module consists of a sequence of declarations. Declarations are
processed in order, and a declaration may refer only to names bound
by preceding declarations — forward references are not permitted.

The five binding forms — `def_bind`, `type_bind`, `mod_bind`,
`mod_type_bind`, and `entry_bind` — bind values (including functions),
types, modules, module types, and host-visible pipeline roots respectively.
Their syntax is detailed in the sections that follow.

Names bound by a declaration inside a module are visible to users of
the module by default (see Modules); the `local` modifier suppresses
this.

#### Declaration Modifiers

- **`open mod_exp`** brings the names bound in `mod_exp` into the
  current scope. They are also re-exported through the enclosing
  module.

- **`local dec`** binds the names defined by `dec` in the current
  scope but hides them from users of the enclosing module.

- **`import "foo"`** is shorthand for `local open` of the module
  expression `import "foo"` (see Modules) — it pulls in another
  file's exports without re-exporting them.

> **DISCREPANCY:** The current compiler does not implement the
> `local` modifier (no `local` keyword in the lexer) and does not
> apply local-open semantics to plain `import "foo"` —
> `resolve_imports::run` literally inlines the imported file's
> top-level decls into the importer, which re-exports them. See the
> ignored tests `local_open_parses_per_spec` and
> `bare_import_does_not_reexport_per_spec` in
> `wyn-core/src/integration_tests.rs` for the intended behavior and
> implementation options. Remove this callout when both tests pass
> without `#[ignore]`.

- **`#[attr] dec`** attaches an attribute to the declaration it
  precedes (see Attributes).

---

## Declaring Functions and Values

### Grammar

```ebnf
def_bind      ::= "def" def_name [generics] [def_signature] "=" exp
def_name      ::= name | "(" symbol ")"
def_signature ::= "(" param ("," param)* ")" type   -- function form
                | ":" type                          -- typed constant
param         ::= name ":" type
generics      ::= "<" generic_param ("," generic_param)* ">"
generic_param ::= "[" name "]" | UpperName

entry_bind    ::= "entry" name "(" [entry_param ("," entry_param)*] ")"
                  type "=" exp
entry_param   ::= name ":" type
```

`UpperName` is an identifier whose first character is uppercase.

### Description

A `def` declaration binds a value or function to a name at module
scope. The body is an arbitrary expression that may use only names
already in scope at the point of binding; forward references are
not permitted, and functions may not be recursive.

A `def` may take parameters and a return type, in which case it
defines a function; without parameters it is a constant.

```wyn
def gravity: f32 = 9.81
def step(dt: f32, v: f32) f32 = v + gravity * dt
```

#### Type Inference

Hindley-Milner-style type inference fills in argument and return
types when context allows. Explicit annotations are required only
when inference cannot determine a unique type. Sizes participate
in the type system; see Size Types for the rules.

```wyn
def step(dt, v) = v + 9.81 * dt
```

Here `dt`, `v`, and the return type are all inferred as `f32`:
`9.81` defaults to `f32`, the multiplication constrains `dt`, and
the addition constrains `v`.

#### Polymorphic Functions

A function may be polymorphic over types and sizes through the
`<...>` generics list. Size parameters are written `[n]`; type
parameters are uppercase identifiers:

```wyn
def reverse<[n], A>(xs: [n]A) [n]A = ???
```

Generics need not cover the type of every parameter. Any argument
whose type isn't tied to a declared generic is given a fresh type
variable by inference:

```wyn
def pair<A>(x: A, y) = (x, y)
```

A fresh type variable is invented for `y`.

#### Type Parameter Resolution

Type and size parameters are inferred from arguments at call sites
— they are not passed explicitly. If the same type parameter `A`
appears in multiple parameter positions, all arguments bound to it
must agree in both shape and element type. For example:

```wyn
def pair<A>(x: A, y: A) = (x, y)
```

`pair([1], [2, 3])` is ill-typed because the two arguments bind
`A` to arrays of different sizes.

#### Aliasing Restrictions

To simplify the handling of in-place updates (see In-place
Updates), the value returned by a function may not alias any
global variables.

#### Pipeline Entries

An `entry` declaration defines a complete operation exposed to the
host. It is not a function value and cannot be called by another Wyn
declaration. Array operators express ordinary data-parallel computation;
vertex and fragment processing are introduced inside the entry expression
by the special forms specified under Unified Pipeline Entries and Stage
Invocation.

Entry declarations differ from `def` declarations in three ways:

- Parameters require the `name: type` form; pattern destructuring
  is not allowed.
- Every parameter and result is part of the external interface of the
  operation. Shader built-ins and intermediate values are not entry
  parameters or results.
- An empty parameter list still requires the parentheses:
  `entry foo() ret = ...`.

The name of an entry must not contain an apostrophe (`'`), even though
apostrophes are otherwise permitted in identifiers.

---

## Type Abbreviations

### Grammar

```ebnf
type_bind     ::= type_keyword name [generics] "=" type
type_keyword  ::= "type" | "type~" | "type^"
generics      ::= "<" generic_param ("," generic_param)* ">"
generic_param ::= "[" name "]" | UpperName
```

### Description

A type abbreviation is a shorthand: after `type t1 = t2`, the name
`t1` is interchangeable with the type `t2`. Type abbreviations do
not introduce distinct types; the abbreviation and its definition
denote the same type.

The bound name must start with a lowercase letter or underscore.
Uppercase identifiers in type position are reserved for type
parameters (see `UpperName` above).

### Lifted Types

A type abbreviation must be marked **size-lifted** (`type~`) if its
right-hand side contains existential sizes, and **fully lifted**
(`type^`) if it (potentially) contains a function. A fully-lifted
type may also contain existential sizes.

```wyn
type~ bag = ?n. [n]i32

def empty_bag: bag = []
```

```wyn
type^ cmp = i32 -> i32 -> i32

def ascending: cmp = |x: i32, y: i32| x - y
def descending: cmp = |x: i32, y: i32| y - x
```

**Restrictions:**
- Lifted types cannot appear as array elements.
- Fully-lifted types cannot be returned from a conditional or loop
  expression.

### Type Parameters

A type abbreviation may take size and/or type parameters via the
`<...>` generics list. Size parameters are written `[n]` and stand
in for array sizes; type parameters are uppercase identifiers and
stand in for arbitrary types:

```wyn
type two_intvecs<[n]>  = ([n]i32, [n]i32)
type two_vecs<[n], T>  = ([n]T,   [n]T)
```

When applying a parameterised abbreviation, size arguments go in
brackets (`<[2]>`) and type arguments are written bare (`<i32>`):

```wyn
def x: two_intvecs<[2]> = ([1, 2], [3, 4])
def y: two_vecs<[2], i32> = ([1, 2], [3, 4])
```

All declared size parameters must appear in the definition.

Type constructors are first-order. An application must supply exactly one
argument for every declared parameter; type abbreviations cannot be partially
applied, passed as values, or abstracted over as higher-kinded parameters.
A size parameter accepts only a bracketed size argument such as `[4]`, `[n]`,
or `[]`; an ordinary type parameter accepts a type. Arity and parameter-kind
mismatches are type errors.

Applications nest in the usual way. For example, `outer<inner<i32>>` is
accepted without whitespace between the two closing `>` characters.

---

## User-Defined Operators

### Description

An infix operator is defined like an ordinary function, with the
operator name enclosed in parentheses at the name position and the
operands listed as a single parameter list:

```wyn
def (+^)((a: i32, b: i32), (c: i32, d: i32)) = (a + c, b + d)
```

Call sites use the operator in infix position: `(1, 2) +^ (3, 4)`.
The parenthesised-name form is the only way to declare an operator.

### Operator Names and Fixity

A valid operator name is a non-empty sequence of characters chosen
from the string `"+-*/%=!><&^|"`. The fixity of an operator is
determined by its leading characters, which must correspond to a
built-in operator. Thus `+^` binds like `+`, while `*^` binds like
`*`. The longest such prefix wins, so `>>=` binds like `>>`, not
like `>`.

### Restrictions

It is not permitted to define operators with the names `&&` or `||`
(although these as prefixes are accepted). A user-defined version of
either would not be short-circuiting. User-defined operators behave
exactly like ordinary functions, except for being infix.

### Shadowing Built-in Operators

A built-in operator may be shadowed (e.g. a new `+` can be defined).
The built-in polymorphic operator then becomes inaccessible except
through the intrinsics module.

---

## Expressions

Expressions are the basic construct of any Wyn program. An expression
has a statically determined type and produces a value at runtime.
Wyn is an eager/strict language ("call by value"). The basic elements
of expressions are called atoms — for example literals and variables,
plus the more complicated forms with their own productions in the
grammar below.

Some expression forms — notably the second-order array operators
(`map`, `reduce`, `scan`, `filter`) — have parallel semantics. The
compiler may but need not execute them in parallel; programs must not
depend on a specific evaluation order beyond what the operator itself
promises.

### Grammar

```ebnf
atom        ::= literal
                | qualname ("." fieldid)*
                | qualname "(" [callarg ("," callarg)*] ")"
                | qualname slice
                | "(" ")"
                | "(" exp ")" ("." fieldid)*
                | "(" exp ")" "(" [callarg ("," callarg)*] ")"
                | "(" exp ")" slice
                | "(" exp ("," exp)+ [","] ")"
                | "{" "}"
                | "{" field ("," field)* [","] "}"
                | quals "." "(" exp ")"
                | "[" exp ("," exp)* [","] "]"
                | "@[" exp ("," exp)* [","] "]"
                | "(" qualsymbol ")"
                | "(" exp qualsymbol ")"
                | "(" qualsymbol exp ")"
                | "(" ( "." field )+ ")"
                | "(" "." slice ")"
                | "???"

exp         ::= atom
                | exp qualsymbol exp
                | "!" exp
                | "-" exp
                | constructor [ "(" exp ("," exp)* [","] ")" ]
                | exp ":" type
                | exp ":>" type
                | exp [ ".." exp ] "..." exp
                | exp [ ".." exp ] "..<" exp
                | exp [ ".." exp ] "..>" exp
                | "if" exp "then" exp "else" exp
                | "let" pat [":" type] "=" exp ["in"] exp
                | "let" size+ pat "=" exp ["in"] exp
                | "let" name [generics] "(" [param ("," param)*] ")" "=" exp ["in"] exp
                | "|" pat ("," pat)* "|" exp
                | "loop" pat ["=" exp] loopform "do" exp
                | "#[" attr "]" exp
                | exp "with" slice "=" exp
                | exp "with" fieldid ("." fieldid)* "=" exp
                | exp "with" "." swizzle assign_op exp
                | "match" exp ("case" pat "->" exp)+

callarg     ::= exp | "_"

assign_op   ::= "=" | "*=" | "+=" | "-=" | "/="
swizzle     ::= [xyzw]+    -- or [rgba]+; one set, distinct chars, len 1..4

slice       ::= "[" index ("," index)* [","] "]"
field       ::= fieldid "=" exp
                | name
size        ::= "[" name "]"

pat         ::= name
                | pat_literal
                | "_"
                | "(" ")"
                | "(" pat ")"
                | "(" pat ("," pat)+ [","] ")"
                | "{" "}"
                | "{" fieldid ["=" pat] ("," fieldid ["=" pat])* [","] "}"
                | constructor [ "(" pat ("," pat)* [","] ")" ]
                | pat ":" type
                | "#[" attr "]" pat

pat_literal ::= [ "-" ] intnumber
                | [ "-" ] floatnumber
                | "true"
                | "false"

loopform    ::= "for" name "<" exp
                | "for" pat "in" exp
                | "while" exp

index       ::= exp
                | [exp] ".." [exp]
```

### Resolving Ambiguities

The grammar above contains ambiguities; they are resolved by the
rules below.

An expression `x.y` is either a reference to the name `y` in the
module `x`, or the field `y` in the record `x`. Modules and values
occupy the same namespace, so this is disambiguated by whether `x` is
a value or a module.

A type ascription (`exp : type`) cannot appear as an array index, as
it conflicts with the syntax for slicing.

An expression `(-x)` is parsed as the variable `x` negated and
enclosed in parentheses, rather than an operator section partially
applying the infix operator `-`.

Prefix operators bind more tightly than infix operators. The only
prefix operators are the built-in `!` and `-`; user-defined prefix
operators are not supported. A user-defined operator beginning with
`!` binds as the infix operator (e.g. `!=` row in the table below),
not as the prefix `!`.

Attributes bind less tightly than any other syntactic construct.

The bodies of `let`, `if`, and `loop` extend as far to the right as
possible.

The following table describes the precedence and associativity of
infix operators in both expressions and type expressions. All
operators in the same row have the same precedence. Rows are listed
in increasing order of precedence. Not every operator listed is used
in expressions; they remain in the table for ambiguity resolution.

| Associativity | Operators |
|---------------|-----------|
| left  | `,` |
| left  | `:`, `:>` |
| left  | `` `symbol` `` |
| left  | `\|>` |
| left  | `\|\|` |
| left  | `&&` |
| left  | `<= >= > < == != ! =` |
| left  | `& ^ \|` |
| left  | `<< >> >>>` |
| left  | `+ -` |
| left  | `* / % // %%` |
| right | `->` |
| left  | `**` |

### Semantics of Simple Expressions

#### literal
Evaluates to itself.

#### qualname
A variable name; evaluates to its value in the current environment.

#### ()
Evaluates to an empty tuple.

#### ( e )
Evaluates to the result of `e`.

#### ???
A typed hole, usable as a placeholder expression. The type checker
will infer any necessary type for this expression. This can sometimes
result in an ambiguous type, which can be resolved using a type
ascription. Evaluating a typed hole results in a run-time error.

#### (e1, e2, ..., eN)
Evaluates to a tuple containing N values. Equivalent to the record
literal `{0=e1, 1=e2, ..., N-1=eN}`.

#### {f1, f2, ..., fN}
A record expression consists of a comma-separated sequence of field
expressions. Each field expression defines the value of a field in
the record. A field expression takes one of two forms:

- `f = e`: defines a field with the name `f` and the value resulting
  from evaluating `e`.
- `f`: defines a field with the name `f` and the value of the
  variable `f` in scope.

Each field may only be defined once.

#### a[i]
Return the element at the given position in the array. The index may
be of any unsigned integer type. Multi-dimensional arrays are indexed
by chaining: `a[i][j]` selects an element from a rank-2 array; `a[i]`
alone returns the inner sub-array.

#### a[i..j]
Return a slice of the array `a` from index `i` (inclusive) to `j`
(exclusive). Both bounds are optional: `a[..j]` slices from the start,
`a[i..]` slices to the end, and `a[..]` is the whole-array identity
slice. Slicing indices have type `i64`.

Slicing of multiple dimensions is done by chaining: `a[i..j][k..l]`
slices the outer dimension first, then the inner one.

In the general case the size of the slice is unknown (see Size
Types). In a few cases the size is known statically:

- `a[0..n]` has size `n`
- `a[..n]` has size `n`

This holds only if `n` is a variable or constant.

#### [x, y, z]
Create an array containing the indicated elements. Each element must
have the same type and shape.

#### x..y...z
Construct a signed integer array whose first element is `x`, which
proceeds with a stride of `y-x` until reaching `z` (inclusive). The
`..y` part may be elided, in which case a stride of 1 is used. All
components must be of the same signed integer type.

A run-time error occurs if `z` is less than `x` or `y`, or if `x` and
`y` are the same value.

In the general case, the size of the array produced by a range is
unknown (see Size Types). In a few cases the size is known
statically:

- `0..<n` has size `n`
- `0..1..<n` has size `n`
- `1..2...n` has size `n`

#### x..y..<z
Construct a signed integer array whose first element is `x`, which
proceeds upwards with a stride of `y-x` until reaching `z`
(exclusive). The `..y` part may be elided, in which case a stride of
1 is used. A run-time error occurs if `z` is less than `x` or `y`,
or if `x` and `y` are the same value.

- `0..1..<n` has size `n`
- `0..<n` has size `n`

This holds only if `n` is a variable or constant.

#### x..y..>z
Construct a signed integer array whose first element is `x`, which
proceeds downwards with a stride of `y-x` until reaching `z`
(exclusive). The `..y` part may be elided, in which case a stride
of -1 is used. A run-time error occurs if `z` is greater than `x` or
`y`, or if `x` and `y` are the same value.

#### e.f
Access field `f` of the expression `e`, which must be a record or
tuple.

#### m.(e)
Evaluate the expression `e` with the module `m` locally opened, as if
by `open`. This can make some expressions easier to read and write
without polluting the surrounding scope with a declaration-level
`open`.

#### x binop y
Apply an operator to `x` and `y`. Operators are functions like any
other and can be user-defined. Wyn pre-defines a set of overloaded
operators that work across multiple numeric types; these overloaded
operators cannot be redefined by the user (but they may be shadowed —
see User-Defined Operators). Both operands must have the same type,
except where noted below for `**`. The predefined operators are:

- **`**`**: Power operator, defined for all numeric types. The base
  and exponent must have the same type, **with one exception**: if
  the base is a floating-point scalar (`f16` / `f32` / `f64`), the
  exponent may be any signed or unsigned integer type
  (`i8` … `i64`, `u8` … `u64`); the result type is the base's float
  type, computed as if the integer exponent were first converted to
  the base's float type.
- **`//`, `%%`**: Integer division and remainder, rounding towards
  zero.
- **`*`, `/`, `%`, `+`, `-`**: The usual arithmetic operators,
  defined for all numeric types. `/` and `%` round towards negative
  infinity when used on integers — different from C.
- **`^`, `&`, `|`, `>>`, `<<`, `>>>`**: Bitwise operators —
  respectively bitwise xor, and, or, arithmetic shift right, left
  shift, and logical (unsigned) shift right. Shifting is undefined if
  the right operand is negative, or greater than or equal to the bit
  width of the left operand.

Unlike in C, bitwise operators have higher priority than arithmetic
operators. This means that `x & y == z` is understood as `(x & y) ==
z`, rather than `x & (y == z)` as it would in C. (The latter is a
type error in Wyn anyway.)

- **`==`, `!=`**: Compare any two values of built-in or compound type
  for equality.
- **`<`, `<=`, `>`, `>=`**: Compare any two values of numeric type
  for ordering.
- **`` `qualname` ``**: Use `qualname`, which may be any non-operator
  function name, as an infix operator.

#### x && y
Short-circuiting logical conjunction; both operands must be of type
`bool`.

#### x || y
Short-circuiting logical disjunction; both operands must be of type
`bool`.

#### f(x, y, z)
Apply the function `f` to the arguments `x`, `y`, and `z`. Function
application is always fully saturated: when a call contains no `_`
placeholders, every parameter of `f` must be given a value at the call
site. Supplying fewer arguments is an error; it does not produce a
partially applied function.

#### f(a, _, c, _)
A call containing one or more `_` placeholders is a **call section**, not a
function application. It is syntactic sugar for a lambda with one distinct
parameter per placeholder, ordered from left to right. For example,

```wyn
f(a, _, c, _)
```

desugars to `|b, d| f(a, b, c, d)`.

The call inside the generated lambda is fully saturated. Non-placeholder
argument expressions remain in the lambda body and are evaluated when that
lambda is invoked.

#### x |> f(a, b)
The pipe operator threads its left operand into the call on its
right as the **last** argument: `x |> f(a, b)` is exactly
`f(a, b, x)`, and `x |> f` (a bare callee) is `f(x)`. It is purely
syntactic sugar for a fully-saturated application — because the
piped value becomes an ordinary trailing argument, no partial
application or currying is involved. `|>` is left-associative and
has the lowest precedence of any operator, so everything to its left
is grouped before being piped: `a + b |> f` is `f(a + b)`, and
`xs |> map(g) |> filter(p)` is `filter(p, map(g, xs))`. Appending as
the last argument matches the array-last convention of the built-in
array operators (`map(f, xs)`, `filter(p, xs)`, `reduce(op, z, xs)`,
`scan(op, z, xs)`), so those read naturally in a pipeline. The
explicit call-section form `x |> f(a, b, _)` is equivalent: the right
operand is `|value| f(a, b, value)`, to which the pipe applies `x`.

#### #c(x, y, z)
Apply the sum type constructor `#c` to the payload `x`, `y`, and
`z`. A constructor application is always assumed to be saturated, so
constructors may not be partially applied. A nullary constructor is
written bare, with no parentheses (`#c`).

#### e : t
Annotate that `e` is expected to be of type `t`, failing with a type
error if it is not.

Due to ambiguities, this syntactic form cannot appear as an array
index expression unless it is first enclosed in parentheses. However,
as an array index must always be of type `i64`, there is never a
reason to put an type ascription there.

#### e :> t
Coerce the size of `e` to `t`. The type of `t` must match the type
of `e`, except that the sizes may be statically different. At
run-time it will be verified that the sizes are the same.

#### ! x
Logical negation if `x` is of type `bool`. Bitwise negation if `x` is
of integral type.

#### - x
Numerical negation of `x`, which must be of numeric type.

#### #[attr] e
Apply an attribute defined for expression position. This specification defines
no expression attributes, so every such expression is currently a static error.
See Attributes.

#### a with [i] = e
Return `a`, but with the element at position `i` changed to contain
the result of evaluating `e`. Consumes `a`.

#### r with f = e
Return the record `r`, but with field `f` changed to have value `e`.
The type of the field must remain unchanged. Type inference here is
limited: `r` must have a completely known type up to `f`. This
sometimes requires extra type annotations to make the type of `r`
known.

#### if c then a else b
If `c` evaluates to true, evaluate `a`; otherwise evaluate `b`.

### Binding Expressions

#### let pat = e in body
Evaluate `e` and bind the result to the irrefutable pattern `pat`
(see Patterns) while evaluating `body`. The `in` keyword may be
omitted when `body` is itself a `let` expression, so chained
bindings need only one closing `in`:

```wyn
let x = 1
let y = 2 in
x + y
```

The binding is not let-generalised, meaning it has a monomorphic
type. This can be significant if `e` is of functional type.

If `e` is of type `i64` and `pat` binds only a single name `v`, then
the type of the overall expression is the type of `body`, but with
any occurrence of `v` replaced by `e`.

#### let [n] pat = e in body
As above, but additionally bind sizes (here `n`) used in the
pattern (here to the size of the array being bound). All declared
sizes must be used in the pattern.

#### let f(x, y) = e in body
Bind `f` to a local function with the given parameters and
definition (`e`) and evaluate `body`. The function aliases any free
variables in `e`.

#### loop pat = initial for x in a do loopbody
Bind `pat` to the initial values given in `initial`. For each
element `x` in `a`, evaluate `loopbody` and rebind `pat` to the
result of the evaluation. Return the final value of `pat`.

The `= initial` may be omitted, in which case initial values for the
pattern are taken from equivalently named variables in the
environment — `loop (x) = ...` is equivalent to `loop (x = x) = ...`.

#### loop pat = initial for x < n do loopbody
Equivalent to `loop (pat = initial) for x in (0..1..<n) do loopbody`.

#### loop pat = initial while cond do loopbody
Bind `pat` to the initial values given in `initial`. If `cond`
evaluates to true, bind `pat` to the result of evaluating
`loopbody`, and repeat. Return the final value of `pat` when `cond`
is false.

#### match x case p1 -> e1 case p2 -> e2
Match the value produced by `x` against each pattern in turn,
picking the first one that succeeds. The result of the corresponding
expression is the value of the entire `match` expression. All the
expressions on the right of a `case` must have the same type (which
need not be the type of `x`). It is a type error if the cases do not
cover every possible value of `x` — non-exhaustive pattern matching
is not allowed.

### Function Expressions

#### |x, y, z| e
Produce an anonymous function taking parameters `x`, `y`, and `z`,
whose body is `e`. See Lambdas for the semantics of environment
capture and the restrictions that apply.

#### (binop)
An operator section that is equivalent to `|x, y| x binop y`.

#### (x binop)
An operator section that is equivalent to `|y| x binop y`.

#### (binop y)
An operator section that is equivalent to `|x| x binop y`.

#### (.a.b.c)
An operator section that is equivalent to `|x| x.a.b.c`.

#### (.[i])
An operator section that is equivalent to `|x| x[i]`. For
multi-dimensional indexing, chain: `(.[i][j])` is `|x| x[i][j]`.

---

## Higher-order Functions

Within a Wyn program, functions can be named, passed as arguments,
and returned from other functions. Function values do not exist at
runtime, however (see Program Structure), so the following
restrictions apply to functions and to any record or tuple
containing a function (a functional type):

- Arrays of functions are not permitted.
- A function cannot be returned from an `if` expression.
- A loop parameter cannot be a function.

See also In-place Updates for details on how consumption interacts
with higher-order functions.

### Function Arity and Partial Application

Wyn functions are **not curried** by default. Every function has a
fixed arity (number of arguments), and every function application must
supply exactly that many arguments. Partial application is not allowed.
Call sections do not relax this rule: they construct explicit lambdas whose
bodies contain fully saturated applications.

```wyn
def add(x: i32, y: i32) i32 = x + y

-- Valid: fully applied
def result = add(1, 2)

-- INVALID: partial application
def add_one = add(1)  -- Error: function requires 2 arguments
```

This restriction applies uniformly to:
- Top-level function definitions
- Anonymous functions
- Built-in functions
- Functions passed as higher-order arguments

#### Call Sections

When a function value with some argument positions left open is needed, use
placeholder arguments:

```wyn
def add(x: i32, y: i32, z: i32) i32 = x + y + z

-- Create a 2-arity function that calls `add` with the middle arg
-- fixed.
def add_with_5 = add(_, 5, _)        -- Produces (i32, i32) -> i32

-- Create a 1-arity function with two of the three args fixed.
def add_one = add(_, 1, 0)           -- Produces i32 -> i32

-- Usage
def result = add_one(5)               -- Returns 6
```

A call containing `_` placeholders is a **call section**:

```wyn
add(_, 5, _)
```

This desugars to `|x, z| add(x, 5, z)`. No partial application is involved.

- Each `_` marks a placeholder position that becomes a distinct
  parameter of the new function.
- Non-placeholder arguments remain ordinary expressions in the generated
  lambda body.
- The resulting function has arity equal to the number of `_`
  placeholders.
- The resulting function is itself non-curried (it requires all
  placeholders to be filled at once).
- A placeholder belongs to its nearest containing call. Thus
  `outer(inner(_))` is `outer(|x| inner(x))`, not
  `|x| outer(inner(x))`.

A section in which every argument is a placeholder is eta-equivalent to the
bare function value. Thus `f(_)` and `f` are interchangeable when `f` has
arity one, and `f(_, _)` and `f` are interchangeable when `f` has arity two.
The bare form is normally clearer.

---

## Lambdas

A lambda — an anonymous function written `|x, y, z| body` — is the
primary way to specialise a higher-order operator per call site.
Parameter types may be inferred from context or given explicitly:

```wyn
map(|x| x * 2.0, arr)
map(|x: f32| x * 2.0, arr)
```

### Environment Capture

A lambda captures every free variable in its body — every identifier
that is bound outside the lambda but referenced inside it. The
captured values become part of the lambda's behaviour and travel
with it wherever it is passed.

```wyn
def above(threshold: i32, arr: [n]i32) ?k. [k]i32 =
    filter(|x| x > threshold, arr)
```

The lambda captures `threshold` from the enclosing function's
parameters; `filter` invokes the lambda once per element of `arr`,
and each invocation sees the captured `threshold`.

Capture is by value: each captured value is snapshotted at the
lambda's construction site. Because Wyn is purely functional there
is no observable difference between by-value and by-reference
capture in any case.

A lambda may capture any value visible at its definition site —
scalars, arrays, vectors, records, tuples, and references to named
functions. References to named functions are resolved statically;
they do not become runtime function values.

### Restrictions

Lambdas do not permit type parameters — they are inherently
monomorphic at their definition site. To express a polymorphic
higher-order computation, use a named function and pass it to the
operator instead.

The restrictions on functional types described under Higher-order
Functions apply to lambda-valued expressions just as to any other
functional-typed value.

---

## Type Inference

Wyn supports Hindley-Milner-style type inference; in many cases
explicit type annotations can be omitted. Annotations are still
required in the following situations:

- **Record field projection** is not in general unambiguous from a
  bare projection like `r.x`, so `r` must have a type known from
  context or by annotation.
- **Sum-type constructors** do not by themselves determine their sum
  type — `#foo(1)` is ambiguous in isolation — so the expected type
  must be available from context or an annotation.
- **Consumed parameters** (see In-place Updates) must be annotated
  explicitly.

Top-level declarations are processed in order, and each top-level
function's type must be completely resolved at its definition site.
If a top-level function uses overloaded arithmetic operators, the
choice of overload cannot be influenced by later use sites — either
the operand types are determined locally (e.g. by annotation) or
the default arithmetic resolution applies.

Local `let` bindings are monomorphic; their types are not
let-generalised.

---

## Size Types

Wyn supports a system of size-dependent types that statically checks
that the sizes of arrays passed to a function are compatible.

Whenever a pattern occurs (in `let`, `loop`, and function
parameters), as well as in return types, the types of the bindings
express invariants about the shapes of arrays accepted or produced
by the function. For example:

```wyn
def double<[n]>(a: [n]i32) [n]i32 = map(|x| x * 2, a)
```

A size parameter, `[n]`, explicitly quantifies a size. The `[n]`
parameter is not passed explicitly when calling the function;
instead its value is implicitly deduced from the arguments. An array
type can contain an anonymous size, e.g. `[]i32`, for which the type
checker invents a fresh size parameter — every array has a size in
the type system. In return-type position this can produce an
existential size that is not known until the function is fully
applied. For example, `filter` has a return type along these lines:

```
filter : <[n], A>(A -> bool, [n]A) -> ?k. [k]A
```

Sizes may be any expression of type `i64` that does not consume any
free variables. Size parameters can be used as ordinary `i64`
variables within their scope. The type checker verifies that the
program obeys any constraints imposed by size annotations.

Size-dependent types are supported, as the names of value parameters
can be used in the return type of a function:

```wyn
def replicate<T>(n: i64, x: T) [n]T = ???
```

An application `replicate(10, 0)` produces a value of type `[10]i32`.

Whenever a type `[e]t` is written, `e` must be a well-typed
expression of type `i64` in scope (possibly by referencing a name
bound as a size parameter).

### Unknown Sizes

There are cases where the type checker cannot assign a precise size
to the result of some operation. For example, `filter` has a type
roughly like:

```
filter : <[n], A>(A -> bool, [n]A) -> ?k. [k]A
```

The function returns an array of some existential size `k` that
cannot be known in advance.

When an application `filter(p, xs)` is encountered, the result is
typed `[k]A`, where `k` is a fresh unknown size that is considered
distinct from every other size in the program. It is sometimes
necessary to perform a size coercion (see Size Coercion) to convert
an unknown size to a known size.

In general, unknown sizes are produced whenever the true size
cannot be expressed. The following lists all sources of unknown
sizes.

#### Size going out of scope

An unknown size is created when a type references a name that has
gone out of scope:

```wyn
match …
case #some(c) -> replicate(c, 0)
```

The type of `replicate(c, 0)` is `[c]i32`, but since `c` is locally
bound, the type of the entire expression is `[k]i32` for some fresh
`k`.

#### Computed expression passed as a size argument

The type of `replicate(e, 0)` should be `[e]i32`, but if `e` is not
valid as a size expression this cannot be expressed. An unknown size
`k` is created and the size of the expression becomes `[k]i32`.

#### Compound expression used as range bound

While a simple range expression such as `0..<n` can be assigned type
`[n]i32`, a range expression `0..<(n+1)` produces an unknown size.

#### Complex slicing

Most complex array slicing, such as `xs[a..b]`, has an unknown size.
Exceptions are listed in the reference for slice expressions.

#### Complex ranges

Most complex ranges, such as `a..<b`, have an unknown size.
Exceptions exist for general ranges and "upto" ranges.

#### Existential size in function return type

Whenever the result of a function application has an existential
size, that size is replaced with a fresh unknown size variable.

For example, given `filter`'s type:

```
filter : <[n], A>(A -> bool, [n]A) -> ?k. [k]A
```

an application `filter(f, xs)` causes the type checker to invent a
fresh unknown size `k'`, and the actual type for that application
is `[k']A`.

#### Branches of if return arrays of different sizes

When an `if` (or `match`) expression has branches that return arrays
of different sizes, the differing sizes are replaced with fresh
unknown sizes. For example:

```wyn
if b then [[1, 2], [3, 4]]
     else [[5, 6]]
```

This expression has type `[k][2]i32` for some fresh `k`.

**Important**: the check for differing sizes is performed when first
encountering the `if` or `match` during type checking. At that point
the type checker may not yet realise that the two sizes are equal,
even though constraints later in the function force them to be.
Adding type annotations resolves this.

#### An array produced by a loop does not have a known size

If the size of some loop parameter is not maintained across a loop
iteration, the final result of the loop will contain unknown sizes.
Similarly to conditionals, the type checker may sometimes be too
conservative in concluding that a size might change during the loop;
adding type annotations to the loop parameter can resolve this.

### Size Coercion

Size coercion, written with `:>`, performs a runtime-checked
coercion of one size to another. It is the escape hatch from the
size type system — useful when a value has an unknown size that the
programmer knows is equal to some named size:

```wyn
def take_n<A>(n: i64, xs: []A) [n]A =
  xs[..n] :> [n]A
```

Here `xs[..n]` has an unknown size (slicing produces an existential
size; see Unknown Sizes), and `:> [n]A` asserts that the slice's
size is in fact `n`. The assertion is checked at run-time.

### Causality Restriction

Conceptually, size parameters are assigned their values by reading
the sizes of concrete values passed as parameters. Every size
parameter must therefore appear as the size of some parameter. The
following is an error:

```wyn
def f<[n]>(x: i32) i32 = i32.i64(n)   -- `n` is never bound by a param
```

The following is not an error:

```wyn
def f<[n]>(g: [n]i32 -> [n]i32) i32 = ???
```

…but using this function comes with a constraint: whenever an
application `f(x)` occurs, the value of the size parameter must be
inferable. The value must have been used as the size of an array
before the `f(x)` application is encountered. The notion of "before"
is subtle since there is no overall evaluation order on a Wyn
expression — only that a let-binding is evaluated before its body,
the argument to a function is evaluated before the function itself,
and the left operand of an operator is evaluated before the right.

The causality restriction only matters when a function has a size
parameter whose first use is not as a concrete array size. It does
not apply to uses of the following function, for example:

```wyn
def f<[n]>(arr: [n]i32, g: [n]i32 -> [n]i32) [n]i32 = g(arr)
```

…because the value of `n` can be read directly from `arr`'s size.

### Empty Array Literals

Just as with size-polymorphic functions, constructing an empty array
requires knowing the exact size of the (missing) elements. In the
following program the elements of `a` are constrained to have the
same type as the elements of `b`, but `b`'s element sizes are not
known at the time `a` is constructed:

```wyn
def main(b: bool, xs: []i32) bool =
  let a: [][]i32 = [] in
  let b = [filter(|x| x > 0, xs)] in
  a[0] == b[0]
```

The result is a type error.

### Sum Types

When constructing a value of a sum type, the compiler must still be
able to determine the size of the constructors that are not used.
The following is illegal:

```wyn
type sum = #foo([]i32) | #bar([]i32)

def main(xs: *[]i32) i32 =
  let v: sum = #foo(xs) in
  xs[0]
```

### Modules

When matching a module against a module type (see Modules), a
non-lifted abstract type (one declared with `type` rather than
`type^`) may not be implemented by a type abbreviation that contains
any existential sizes. This ensures that, given:

```wyn
module m : { type t } = …
```

an array of values of type `m.t` can always be constructed without
risking irregularity.

### Higher-order Functions

When a higher-order function expects a function argument whose
output is itself an array, the per-call output size must be the
same for every invocation. This is why `map` produces a regular
array: its function argument has type `A -> B`, where `B` must be a
fixed-size type, so every element of the input maps to a result of
the same shape and the output stays rectangular.

Operators whose result size genuinely varies per call — `filter`,
for example, where each invocation may keep a different number of
elements — produce existentially-sized results that do not feed
back into a `map`-like operator without first being materialised.

#### A function whose return type has an unknown size

If a function (named or anonymous) is inferred to have a return type
that contains an unknown size variable created within the function
body, that size variable is replaced with an existential size.
Usually this is harmless, but it means an expression like the
following is ill-typed:

```wyn
map(|xs: [m]i32| iota(length(xs)), xss)
```

Here `length(xs)` gives rise to some fresh size `k`. The lambda is
then assigned the type `[m]i32 -> [k]i32`, which is immediately
rewritten to `[m]i32 -> ?k. [k]i32` because `k` was generated inside
the lambda body. A function of this type cannot be passed to `map`,
as explained above. The fix is to bind the length to a name in the
enclosing scope before the lambda is constructed.

---

## In-place Updates

In-place updates do not produce observable side effects, but they
provide a way to update an array efficiently — the cost is
proportional to the size of the value(s) being written, not the size
of the full array.

The `a with [i] = v` construct (and its derived forms) performs an
in-place update. The compiler verifies that the original array `a` is
not used on any execution path following the update, and that no
alias of `a` is used either. Most language constructs produce fresh
arrays with no aliases; slicing is the main exception — a slice
aliases its source.

A function parameter may be marked **consuming** by prefixing its
type with `*`. A return type may be marked **alias-free** the same
way. For example:

```wyn
def modify(a: *[]i32, i: i32, x: i32) *[]i32 =
  a with [i] = a[i] + x
```

`*` is a property of a **signature only** — it may appear on function
parameter and return types, and nowhere else. It is not part of a
value's type: a `let` binding, a type ascription, or a plain (non-
function) `def` may not carry `*`. Uniqueness is never inferred, so a
`*` return does not make an unannotated parameter consuming — annotate
the parameter to consume it. A function that declares a `*` return
must actually produce an alias-free value: freshly allocated, or a
parameter it consumed (a `with` update on a consumed array qualifies);
returning an observing parameter or a global is an error.

Because a consuming function's effect cannot be tracked when the
function is used as a value, a **consuming function may not be passed
as a value** — a parameter, lambda parameter, or argument whose type
is a function mentioning `*` is rejected — and functions may not be
returned from `if` or `match`.

A parameter that is not consuming is called **observing**. The `*` in
`a: *[]i32` means `modify` takes ownership of `a`; no caller may
reference `a` (or any alias of it) after the call. This is what
permits the `with` expression to update in place. After a call
`modify(a, i, x)`, neither `a` nor any alias of `a` may be used on
any subsequent execution path.

If a `*` appears anywhere inside a tuple parameter type, the whole
parameter is considered consuming:

```wyn
def consumes_both(p: (*[]i32, []i32)) i32 = ???
```

This is usually not desirable. Prefer separate parameters:

```wyn
def consumes_first_arg(a: *[]i32, b: []i32) i32 = ???
```

For bulk in-place updates with multiple values, use the `scatter`
function from the prelude.

### Alias Analysis

The rules used to determine aliasing are intuitive in the intra-
procedural case. Aliases are associated with entire arrays. Aliases
of a record or tuple are tracked for each element, not for the record
or tuple itself. Most constructs produce fresh arrays with no
aliases; the main exceptions are `if`, `loop`, function calls, and
variable references.

After a binding `let a = b`, which simply assigns a new name to an
existing variable, `a` aliases `b`. Similarly for record projections
and pattern bindings.

The result of an `if` aliases the union of the aliases of its
branches.

The result of a `loop` aliases the initial values as well as any
aliases that the merge parameters may assume at the end of an
iteration, computed to a fixed point.

The aliases of a value returned from a function depend on whether the
return value is declared alias-free (with `*`). If it is, the value
has no aliases. Otherwise, it aliases all arguments passed for
non-consumed parameters.

#### Globals

A top-level constant is a shared, immutable value: it carries no
ownership, so it is never consumable. A function body may observe a
global freely — index it, pass it as an argument, fold over it — but
cannot consume it, and a function that returns a global as a
non-unique result exposes it only for observation. Consequently a
global can never satisfy a consuming parameter or an alias-free (`*`)
return.

```wyn
def table: [4]i32 = [1, 2, 3, 4]

def ok(i: i32) i32 = table[i]              -- observe: fine
def get(i: i32) [4]i32 = table            -- return for observation: fine

def bad(i: i32) i32 = consume(table)      -- error: cannot consume a global
def worse(i: i32) *[4]i32 = table         -- error: not alias-free
```

### In-place Updates and Higher-order Functions

Consumption interacts inflexibly with higher-order functions: the
language cannot control how many times a function argument is
applied, or to what, so it is not safe to pass a function that
consumes its argument. Two conservative rules govern this
interaction:

- In the expression `let p = e1 in …`, if any in-place update takes
  place inside `e1`, the value bound by `p` must not be or contain a
  function.
- A function that consumes one of its arguments may not be passed as
  a higher-order argument to another function.

---

## Modules

### Grammar

```ebnf
mod_bind      ::= "module" name mod_param* "=" [":" mod_type_exp] "=" mod_exp
mod_param     ::= "(" name ":" mod_type_exp ")"
mod_type_bind ::= "module" "type" name "=" mod_type_exp
```

Wyn supports an ML-style higher-order module system. Modules can contain types, functions, and other modules and module types. Module types are used to classify the contents of modules, and parametric modules are used to abstract over modules (essentially module-level functions). In Standard ML, modules, module types and parametric modules are called structs, signatures, and functors, respectively. Module names exist in the same name space as values, but module types are their own name space.

### Module Bindings

#### module m = mod_exp
Binds `m` to the module produced by the module expression `mod_exp`. Any name `x` in the module produced by `mod_exp` can then be accessed with `m.x`.

#### module m : mod_type_exp = mod_exp
Shorthand for `module m = mod_exp : mod_type_exp`.

#### module m mod_params... = mod_exp
Shorthand for `module m = \mod_params... -> mod_exp`. This produces a parametric module.

#### module type mt = mod_type_exp
Binds `mt` to the module type produced by the module type expression `mod_type_exp`.

### Module Expressions

```ebnf
mod_exp ::= qualname
            | mod_exp ":" mod_type_exp
            | "\" "(" mod_param* ")" [":" mod_type_exp] "->" mod_exp
            | mod_exp mod_exp
            | "(" mod_exp ")"
            | "{" dec* "}"
            | "import" stringlit
```

A module expression produces a module. Modules are collections of bindings produced by declarations (`dec`). In particular, a module may contain other modules or module types.

#### qualname
Evaluates to the module of the given name.

#### (mod_exp)
Evaluates to `mod_exp`.

#### mod_exp : mod_type_exp
Module ascription evaluates the module expression and the module type expression, verifies that the module implements the module type, then returns a module that exposes only the functionality described by the module type. This is how internal details of a module can be hidden.

As a slightly ad-hoc limitation, ascription is forbidden when a type substitution of size-lifted types occurs in a size appearing at the top level.

#### \(p: mt1): mt2 -> e
Constructs a parametric module (a function at the module level) that accepts a parameter of module type `mt1` and returns a module of type `mt2`. The latter is optional, but the parameter type is not.

#### e1 e2
Apply the parametric module `m1` to the module `m2`.

#### { decs }
Returns a module that contains the given definitions. The resulting module defines any name defined by any declaration that is not `local`, in particular including names made available via `open`.

#### import "foo"
Returns a module that contains the definitions of the file "foo" relative to the current file.

### Module Type Expressions

```ebnf
mod_type_exp ::= qualname
                 | "{" spec* "}"
                 | mod_type_exp "with" qualname type_param* "=" type
                 | "(" mod_type_exp ")"
                 | "(" name ":" mod_type_exp ")" "->" mod_type_exp
                 | mod_type_exp "->" mod_type_exp

spec ::= "val" name type_param* ":" type
         | "val" "(" symbol ")" ":" type
         | "val" symbol type_param* ":" type
         | ("type" | "type^" | "type~") name type_param* "=" type
         | ("type" | "type^" | "type~") name type_param*
         | "module" name ":" mod_type_exp
         | "include" mod_type_exp
         | "#[" attr "]" spec
```

Module types classify modules, with the only (unimportant) difference in expressivity being that modules can contain module types, but module types cannot specify that a module must contain a specific module type. They can specify of course that a module contains a submodule of a specific module type.

A module type expression can be the name of another module type, or a sequence of specifications, or specs, enclosed in curly braces. A spec can be a value spec, indicating the presence of a function or value, an abstract type spec, or a type abbreviation spec.

In a value spec, sizes in types on the left-hand side of a function arrow must not be anonymous. For example, this is forbidden:

```wyn
sig sum: []t -> t
```

Instead write:

```wyn
sig sum [n]: [n]t -> t
```

But this is allowed, because the empty size is not to the left of a function arrow:

```wyn
sig evens [n]: [n]i32 -> []i32
```

### Referencing Other Files

You can refer to external files in a Wyn file like this:

```wyn
import "file"
```

The above will include all non-local top-level definitions from `file.fut` is and make them available in the current file (but will not export them). The `.fut` extension is implied.

You can also include files from subdirectories:

```wyn
import "path/to/a/file"
```

The above will include the file `path/to/a/file.fut` relative to the including file.

Qualified imports are also possible, where a module is created for the file:

```wyn
module M = import "file"
```

In fact, a plain `import "file"` is equivalent to:

```wyn
local open import "file"
```

To re-export names from another file in the current module, use:

```wyn
open import "file"
```

> **DISCREPANCY:** The current compiler does not implement any of the
> three forms above as specified.
>
> - Plain `import "file"` is handled by `resolve_imports::run` as a
>   literal inline of the imported file's top-level decls into the
>   importer's declaration list. This matches the spec's
>   `open import "file"` (re-export) form, not the intended
>   `local open import "file"` (use without re-export) form.
> - `module M = import "file"` parses but errors at elaboration
>   ("Unsupported module expression type") — `module_manager`'s
>   `elaborate_module_body` has no case for
>   `ModuleExpression::Import`.
> - `open import "file"` parses but reaches the same elaboration
>   gap.
>
> See the ignored tests `bare_import_does_not_reexport_per_spec` and
> `qualified_module_import_per_spec` in
> `wyn-core/src/integration_tests.rs` for the intended behavior and
> implementation options. Remove this callout when both tests pass
> without `#[ignore]`.

---

## Attributes

Attributes attach narrowly defined metadata to declarations. An attribute is
valid only on the syntactic form named by the section that defines it. An
unknown attribute, an attribute in any other position, or an attribute with
arguments of the wrong form is a static error.

Attributes do not identify execution stages and do not connect stage interfaces
or resources. Vertex and fragment callback roles are specified by the invocation
operations under Unified Pipeline Entries and Stage Invocation; all other data
flow follows ordinary expression semantics.

### Grammar

```ebnf
attr ::= "linked" "(" stringlit ")"
```

### External Linkage

`#[linked("name")]` may be applied to an `extern` declaration. It declares
that the function body is supplied by the execution environment under the
given linkage name.

```wyn
#[linked("sha256_compress")]
extern sha256_compress(state: [8]u32, block: [16]u32) [8]u32
```

The supplied function must have exactly the declared Wyn function type. A
program that uses a linked declaration for which the execution environment
provides no matching definition cannot be executed.

## Vector Types

In addition to arrays, Wyn provides fixed-width vector types. They are
distinct from arrays: they have a fixed component count and componentwise
arithmetic.

Vector types use the naming convention `vecNT` where:
- `N` is the number of components (2, 3, or 4)
- `T` is the element type (i32, f32, etc.)

Common vector types:

| Component Type | 2-component | 3-component | 4-component |
|----------------|-------------|-------------|-------------|
| `i32` (signed) | `vec2i32`   | `vec3i32`   | `vec4i32`   |
| `f32` (32-bit float) | `vec2f32` | `vec3f32` | `vec4f32`  |

Vector types are distinct from array types and have different semantics:
- **Vectors** are fixed-size, optimized for SIMD operations
- **Arrays** are more general containers with runtime length operations

Example usage:
```wyn
let position: vec3f32 = @[1.0, 2.0, 3.0]
let color: vec4f32 = @[1.0, 0.0, 0.0, 1.0]
```

### Vector Swizzles

A vector's components are accessed with field syntax (`v.x`). Wyn
also supports swizzles: one to four letters drawn from a single
"swizzle set". The two sets and their component indices are:

| Set  | `0` | `1` | `2` | `3` |
|------|-----|-----|-----|-----|
| xyzw | `x` | `y` | `z` | `w` |
| rgba | `r` | `g` | `b` | `a` |

`rgba` is an alias set for `xyzw` (`r == x`, `g == y`, `b == z`,
`a == w`) — the two sets address the same underlying components and
produce identical values. Mixing letters from the two sets in one
swizzle (`.xg`, `.rbw`) is a type error.

A single-letter swizzle produces the scalar component; a 2-, 3-, or
4-letter swizzle produces a new vector of that length. Letters may
repeat and may be in any order.

Each referenced component must lie within the source vector's length —
`.z` (or `.b`) on a `vec2` is a type error.

```wyn
let v: vec4f32 = @[1.0, 2.0, 3.0, 4.0]
let first: f32       = v.x        -- 1.0
let alpha: f32       = v.a        -- 4.0   (same as v.w)
let rgb:   vec3f32   = v.rgb      -- (1.0, 2.0, 3.0)
let rev:   vec4f32   = v.wzyx     -- (4.0, 3.0, 2.0, 1.0)
let splat: vec3f32   = v.xxx      -- (1.0, 1.0, 1.0)
```

#### Swizzle Update via `with`

Wyn extends the `with` operator to vec swizzles, so the GLSL idiom
`dir.yz *= rot(angle)` translates directly:

```wyn
let v1 = v0 with .yz = e          -- replace v0.y, v0.z with e.x, e.y
let v2 = v1 with .yz *= m         -- compound: same as `with .yz = v1.yz * m`
let v3 = v2 with .x = scalar      -- single-component LHS takes a scalar RHS
```

The compound forms `*= += -= /=` desugar to
`target with .swizzle = target.swizzle <op> rhs`, with `target`
evaluated once. The result is always a fresh vec — wyn is
immutable, the original target is unchanged. Components on the
LHS must be **distinct** (`v with .xx = e` is rejected); the RHS
arity must match the swizzle length (a `vec2` for `.yz`, a scalar
for `.x`); and each component must be in range for the target's
size.

### Vector Constructors

Vectors are constructed with the `@[...]` literal syntax:

```wyn
let v1: vec3f32 = @[1.0, 2.0, 3.0]
let v2: vec4f32 = @[1.0, 0.0, 0.0, 1.0]
```

The element type is inferred from the arguments or the context.

### Vector Arithmetic and Scalar Broadcasting

The binary arithmetic operators `+`, `-`, `*`, and `/` apply
component-wise to vectors. When both operands are vectors they must
have the *same* vector type (same length and element type); the result
is that type and each component is combined independently:

```wyn
let a: vec3f32 = @[1.0, 2.0, 3.0]
let b: vec3f32 = @[4.0, 5.0, 6.0]
let s: vec3f32 = a + b            -- (5.0, 7.0, 9.0)
let p: vec3f32 = a * b            -- (4.0, 10.0, 18.0), component-wise
```

When one operand is a scalar, it is **broadcast** against the vector:
the scalar is applied to every component, and the result has the
vector's type. The scalar may appear on **either** side, so both
`v op scalar` and `scalar op v` are accepted:

```wyn
let v: vec3f32 = @[1.0, 2.0, 3.0]
let scaled: vec3f32 = v * 2.0         -- (2.0, 4.0, 6.0)
let shifted: vec3f32 = 3.0 - 2.0 * v  -- 3.0 - (2.0, 4.0, 6.0) = (1.0, -1.0, -3.0)
```

Broadcasting performs **no implicit numeric conversion**: the scalar's
type must equal the vector's element type. Mixing a `vec3f32` with an
`i32` scalar is a type error — write the scalar as `f32`. Likewise,
component-wise vector arithmetic between two vectors of different
element types or lengths is rejected.

For `*` specifically, matrix products (matrix×matrix, matrix×vector,
vector×matrix, matrix×scalar) take priority over component-wise
arithmetic; see [Matrix Types](#matrix-types).

---

## Matrix Types

Wyn provides built-in matrix types for graphics and compute workloads. A matrix value has a fixed row count R, a fixed column count C, and a scalar element type — written `matRxC<elem>`. Square matrices have a shorthand alias `matN<elem>`:

| Shape | Square shorthand | Rectangular form |
|-------|-----------------|------------------|
| 2×2   | `mat2f32`       | `mat2x2f32`      |
| 3×3   | `mat3f32`       | `mat3x3f32`      |
| 4×4   | `mat4f32`       | `mat4x4f32`      |
| R×C (R ≠ C) | n/a       | `matRxCf32`      |

Both the shorthand and rectangular form name the same type — `mat2f32 = mat2x2f32`.

Supported dimensions are R, C ∈ {2, 3, 4}. Supported element types are all of the SPIR-V scalar primitives: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`, `f64`. (In practice graphics code uses `f32`; the broader set is available for compute workloads that need it.)

### Matrix Literals

Matrices are written with the `@[[...], [...], ...]` literal syntax — outer brackets describe rows, inner brackets describe each row's components:

```wyn
let m: mat2x2f32 = @[[1.0, 0.0],
                     [0.0, 1.0]]

let rot: mat2f32 = @[[c, s],
                     [-s, c]]
```

Each inner array becomes one column of the matrix at SPIR-V emission — the literal above produces the GLSL equivalent `mat2(c, s, -s, c)` (column-major, matching GPU convention).

### Matrices in Arrays

Matrix types are valid array elements, so `[][N][M]T` arrays and SOAC inputs
work over matrix values in the same manner as they do over scalars.

---

## Unified Pipeline Entries and Stage Invocation

### Overall Architecture

A Wyn `entry` describes one complete operation invoked by the host. An entry
is the root of a pipeline, not an individual GPU shader. Its parameters are the
external inputs to the operation, its result is the external result of the
operation, and its body describes all data dependencies within the operation.

Work within an entry is expressed in two semantic contexts:

- The **orchestration context** evaluates the entry body and any ordinary
  functions called from it. It may use the stage invocation operations
  specified in this section.
- A **stage context** evaluates a callback passed to a stage invocation
  operation. A stage context has an invocation value supplied by that
  operation. It may call ordinary functions, but it may not initiate another
  stage invocation.

The context in which an ordinary `def` is evaluated follows its call site.
A function containing a stage invocation operation can therefore be used as
an orchestration helper, while a function that contains no such operation can
usually be used in either context.

An `entry` is not a function value. It cannot be named in an expression,
passed as an argument, or called by another Wyn declaration. Multiple entries
in one program denote independent operations exposed to the host.

A stage context is introduced when a stage invocation operation receives a
callback. The operation determines the callback's context:

| Invocation operation | Callback context | Callback argument | Result |
|---|---|---|---|
| `rasterize_*` | vertex | `vertex_invocation` | `raster<V>` |
| `shade`, `shade_with` | fragment | `fragment_invocation<V>` | render target |

These names denote predeclared operations with the special context rules above.
Name resolution otherwise follows the ordinary rules: if a program shadows one
of these names, a call to the shadowing declaration is an ordinary call and
does not introduce a stage.

### Entry Interfaces

The syntax of an entry is:

```ebnf
entry_bind  ::= "entry" name "(" [entry_param ("," entry_param)*] ")"
                type "=" exp
entry_param ::= name ":" type
```

An entry parameter or result may contain ordinary scalar, array, tuple, record,
and resource types. It may not contain an invocation type, `vertex<V>`,
`raster<V>`, or `draw`. Those types describe relationships within one
invocation of the entry and have no external value.

External resources occur as values in the structural parameter and result
types of the entry. They have no source-language binding numbers. In
particular, attributes such as `#[storage(...)]`,
`#[uniform(...)]`, `#[texture(...)]`, `#[sampler(...)]`, and
`#[view(...)]` are not valid on an entry interface.

An entry has ordinary value semantics. An external input can affect an
external result only through the expression in the entry body. A resource
that is updated by an entry is consumed and returned according to the usual
uniqueness rules.

### Ordinary Calls and Data Flow

Ordinary function application is the only way to pass values into a helper or
stage callback. Ordinary function results are the only way to pass values out.
A callback may capture values from its surrounding orchestration expression;
captures have the same meaning as explicit callback parameters.

For example, these two vertex callbacks are equivalent:

```wyn
def make_vertex(points: []vec3f32, transform: mat4f32,
                invocation: vertex_invocation) vertex<vec2f32> =
  let p = points[i64(invocation.vertex_index)] in
  vertex_output(transform * @[p.x, p.y, p.z, 1.0], p.xy)

rasterize_triangles(draw, |invocation|
  make_vertex(points, transform, invocation))

rasterize_triangles(draw, make_vertex(points, transform, _))
```

The final expression uses a function call section. As specified under Call
Sections, `make_vertex(points, transform, _)` is equivalent to
`|invocation| make_vertex(points, transform, invocation)`.

Passing an array, record, texture, or other value between two operations does
not expose storage identity or an address. Those notions are not part of the
semantics of a Wyn value.

### Invocation and Pipeline Types

The following built-in types are used by the invocation operations:

| Type | Meaning | Constructible by user code |
|---|---|---|
| `vertex_invocation` | indices for one requested vertex | no |
| `vertex<V>` | clip position and vertex-to-fragment payload | only with `vertex_output` |
| `raster<V>` | rasterized coverage carrying payload `V` | no |
| `fragment_invocation<V>` | one covered fragment and its interpolated payload | no |
| `fragment_output<C>` | color/depth/discard result of fragment processing | `#color`, `#depth`, `#discard` |
| `draw` | a direct, indexed, or indirect draw description | only with draw operations |
| `render_target<C>` | a render target with color shape `C` | no |

Invocation types are spellable so that named helper functions can state their
types. They have no literals or general constructors. An invocation value may
be passed to and returned from ordinary helper calls within its current stage
context, but it may not be stored in an array, included in an entry interface,
or captured by a different stage callback.

`vertex<V>`, `raster<V>`, `fragment_invocation<V>`, and
`render_target<C>` are opaque, one-argument built-in generic types.
`fragment_output<C>` is a predeclared generic sum type. Their arguments are
ordinary type arguments and participate in type inference in the same manner
as arguments to any other generic type.

### Array Computation

Ordinary computation within an entry is expressed with the same scalar,
aggregate, and second-order array expressions used in any other Wyn function.
There is no compute-stage callback, compute invocation value, or general
dispatch operation in the unified pipeline language.

For example, an element-wise update is an ordinary `map`:

```wyn
def update_particle(dt: f32, p: particle) particle =
  { position = p.position + dt * p.velocity,
    velocity = p.velocity }

let updated = map(update_particle(dt, _), particles)
```

When a computation needs an element index, the index is an ordinary value.
It can be obtained from an index array rather than from a platform invocation:

```wyn
let selected = map(|i| values[i], iota(length(values)))
```

The array operators determine the value, shape, and ordering semantics of
their results. The language does not expose whether an array expression is
executed by one compute shader, several compute shaders, or as part of another
operation.

The unified pipeline language does not expose workgroup identifiers, fixed
workgroup geometry, mutable workgroup storage, or workgroup barriers.

### Draw Invocation

A `draw` value describes which vertex invocations a rasterization operation
requests. It has no fields and no user-defined literal. The following
operations construct direct draws:

```wyn
direct_draw(vertex_count, instance_count)
direct_draw_from(vertex_count, instance_count,
                 first_vertex, first_instance)
indexed_draw(indices, instance_count)
indexed_draw_from(indices, index_count, instance_count,
                  first_index, vertex_offset, first_instance)
```

Counts and non-negative indices have type `u32`; `vertex_offset` has type
`i32`. `indices` must be a one-dimensional array of `u16` or `u32`.
`indexed_draw` requests every element of `indices`; the `_from` form requests
the stated subrange.

A direct draw supplies `first_vertex + i` as `vertex_index`. An indexed draw
reads the requested index element, adds `vertex_offset`, and supplies the
result as `vertex_index`. A negative result or a result greater than the
largest `u32` is a dynamic error. Both forms supply
`first_instance + instance` as `instance_index`.

Indirect draws obtain the counts and offsets from ordinary values:

```wyn
type draw_command = {
  vertex_count: u32,
  instance_count: u32,
  first_vertex: u32,
  first_instance: u32
}

type indexed_draw_command = {
  index_count: u32,
  instance_count: u32,
  first_index: u32,
  vertex_offset: i32,
  first_instance: u32
}

indirect_draw(command)
indexed_indirect_draw(indices, command)
indirect_draws(commands)
indexed_indirect_draws(indices, commands)
```

The plural forms describe one draw for each command in array order. Because a
command is an ordinary record, it may be computed earlier in the same entry
and passed directly to an indirect draw operation. Indirectness changes where
the draw description comes from; it does not change the type or calling
convention of the vertex callback.

### Vertex Invocation

A `vertex_invocation` exposes these read-only fields:

```wyn
invocation.vertex_index   : u32
invocation.instance_index : u32
invocation.draw_index     : u32
```

`draw_index` is zero for a singular draw and identifies the command for a
plural indirect draw. All other vertex data is supplied by ordinary arguments
or captures. There are no vertex-slot parameters.

A vertex callback must return `vertex<V>`, constructed by:

```wyn
vertex_output(position: vec4f32, value: V) vertex<V>
```

`position` is the homogeneous clip-space position. `value` is the complete
vertex-to-fragment payload. Returning one record or tuple is the way to pass
multiple varying values:

```wyn
type varying = { uv: vec2f32, normal: vec3f32, material: u32 }

vertex_output(clip_position,
              { uv = uv, normal = normal, material = material })
```

A vertex callback is evaluated only by a rasterization operation. A named
helper requiring `vertex_invocation` cannot be called in orchestration context
because no expression in that context can construct the required argument.

### Rasterization Invocation

Topology is selected by the invocation operation rather than by an attribute
or a separate topology value:

```wyn
rasterize_triangles(draw, callback)
rasterize_triangle_strip(draw, callback)
rasterize_lines(draw, callback)
rasterize_line_strip(draw, callback)
rasterize_points(draw, callback)
```

Each operation calls `callback` in vertex context for the invocations
described by `draw`. If the callback returns `vertex<V>`, the operation returns
`raster<V>`. An incomplete final primitive in a list topology is ignored.

The corresponding `_with` forms take explicit rasterization state as their
first argument:

```wyn
rasterize_triangles_with(state, draw, callback)
rasterize_triangle_strip_with(state, draw, callback)
rasterize_lines_with(state, draw, callback)
rasterize_line_strip_with(state, draw, callback)
rasterize_points_with(state, draw, callback)
```

`raster_state` is the following ordinary record type:

```wyn
type viewport = { origin: vec2f32, extent: vec2f32, depth: vec2f32 }
type scissor = { origin: vec2i32, extent: vec2u32 }

type raster_state = {
  viewport: #target | #custom(viewport),
  scissor: #target | #custom(scissor),
  front_face: #clockwise | #counter_clockwise,
  cull: #none | #front | #back,
  fill: #fill | #line | #point
}
```

`viewport.depth` contains the minimum and maximum depth. `#target` selects the
full extent of the render target later supplied to `shade`. The forms without
`_with` use `default_raster_state`, whose viewport and scissor are `#target`,
whose front face is counter-clockwise, whose culling mode is `#none`, and
whose fill mode is `#fill`.

#### Varying Payloads

The type argument `V` must be a varying payload. A varying payload is composed
recursively from numeric and Boolean scalars, vectors, matrices, tuples, and
records. It may not contain an array, sum, function, resource, invocation,
`draw`, or any instantiation of `vertex` or `raster`. The unit type `()` is a
valid payload when no user value is passed to the fragment callback.

Floating-point components are perspective-correct interpolated. Integer and
Boolean components are flat and take the first vertex's value. Matrix, tuple,
and record payloads apply these rules componentwise.

#### Raster Values

`raster<V>` is opaque and non-copyable. It has no literal, constructor, field
access, equality operation, or external representation. It may be bound to a
name and passed through ordinary orchestration helpers, but it must be
consumed exactly once by `shade` or `shade_with`. It cannot be placed in an
array or returned from an entry.

The `V` argument connects the vertex and fragment callbacks structurally.
There are no numbered varying locations and no separate link step between two
source declarations.

### Fragment Invocation

`shade` consumes rasterized coverage and a render target:

```wyn
shade(target, raster, callback)
shade_with(state, target, raster, callback)
```

If `target` has type `render_target<C>` and `raster` has type `raster<V>`,
the callback receives `fragment_invocation<V>` and returns `C` or
`fragment_output<C>`. The result of either operation has type
`render_target<C>`.

`C` may itself be a tuple or record. Such a target contains one color
attachment for each scalar or vector leaf, and the fragment callback returns
the colors with the same structure. All attachments have the same dimensions
and sample count. The target is consumed and its updated value is returned:

```wyn
let target' = shade(target, covered,
  |fragment| fragment.value)
```

Shading loads the current color and depth contents of the target. Samples for
which no fragment write is applied retain their previous values; neither
`shade` nor `shade_with` implicitly clears a target. A host may supply an
initialized or cleared target value when invoking the entry.

A `fragment_invocation<V>` exposes these read-only fields:

```wyn
invocation.value          : V
invocation.position       : vec4f32
invocation.front_facing   : bool
invocation.primitive_index: u32
invocation.sample_index   : u32
```

`value` is the payload after interpolation. `position.xy` is the target-space
fragment position, `position.z` is depth, and `position.w` is the reciprocal
homogeneous clip coordinate. For a single-sample target, `sample_index` is zero.

A direct `C` return writes those colors using the interpolated raster depth.
`fragment_output<C>` is the following predeclared generic sum type:

```wyn
type fragment_output<C> = #color(C) | #depth(C, f32) | #discard
```

`#color(colors)` is equivalent to returning `colors` directly.
`#depth(colors, depth)` replaces the interpolated depth. `#discard` writes
neither color nor depth. The expected callback result determines the otherwise
ambiguous type of `#discard`.

`fragment_state` is the following ordinary record type:

```wyn
type depth_test =
  #disabled | #never | #less | #less_equal | #equal |
  #greater_equal | #greater | #always

type blend_mode = #replace | #source_over | #add

type fragment_state = {
  depth_test: depth_test,
  depth_write: bool,
  blend: blend_mode,
  color_write: bool
}
```

`#replace` writes the source color and `#add` adds source and destination
componentwise. `#source_over` requires every color leaf of `C` to be a
four-component floating-point vector. For each leaf it computes
`result.rgb = source.rgb * source.a + destination.rgb * (1 - source.a)` and
`result.a = source.a + destination.a * (1 - source.a)`.

`shade_with` uses the supplied state. `shade` applies a default state whose
depth test is `#less`, whose depth and color writes are enabled, and whose
blend mode is `#replace`. If a render target has
no depth attachment, every depth test passes and depth writes have no effect.
`#disabled` also makes every depth test pass and prevents a depth write.

The fragment callback is evaluated once for every covered sample. Callback
evaluations have no specified order. For each sample, the returned depth is
tested against the target depth; if the test passes, enabled depth and color
writes are applied. Fragments addressing the same sample are applied in
`(draw_index, instance_index, primitive_index)` order.

Textures, samplers, scalars, records, and any other read-only values used by a
fragment callback are ordinary arguments or captures. There are no varying,
target, texture, sampler, or built-in attributes on the callback.

### Composition and Resource Flow

The result of any ordinary expression or invocation operation may be passed to
a later operation when its type permits. This includes:

- an array produced by ordinary array expressions captured by a vertex callback;
- a `draw_command` produced earlier in the entry passed to `indirect_draw`;
- a `raster<V>` returned by rasterization passed to `shade`; and
- a render target returned by one shading operation passed to another.

Within one entry invocation there is no syntax for assigning storage to an
intermediate value or connecting two operations by binding number. Across
entry invocations, persistent state is explicit: the host passes a prior
result as a later argument. There is no implicit previous-frame value.

### Static Restrictions

The following programs are ill-typed:

- a stage invocation operation evaluated in a stage context;
- `vertex_output` evaluated outside a vertex context;
- an invocation value constructed by a literal or constructor;
- an invocation value stored in an array or included in an entry interface;
- a vertex callback that does not return `vertex<V>`;
- a varying payload that does not satisfy the payload restrictions;
- a fragment callback whose result does not match its target color shape;
- a `raster<V>` that is copied, discarded, or returned from an entry; and
- an explicit resource-binding or inter-stage interface attribute.

Ordinary helper calls do not introduce a new stage invocation. A helper that
accepts the current invocation value remains part of that callback's stage
context. This permits a vertex or fragment computation to be decomposed into
arbitrarily many ordinary functions without turning those functions into
separate root entries.

### Complete Example

The following entry contains an array transformation, one vertex callback,
and one fragment callback:

```wyn
type particle = { position: vec3f32, velocity: vec3f32 }
type varying = { uv: vec2f32, speed: f32 }

def update_particle(dt: f32, p: particle) particle =
  { position = p.position + dt * p.velocity,
    velocity = p.velocity }

def particle_vertex(particles: []particle, view_projection: mat4f32,
                    invocation: vertex_invocation) vertex<varying> =
  let p = particles[i64(invocation.vertex_index)] in
  vertex_output(
    view_projection * @[p.position.x, p.position.y, p.position.z, 1.0],
    { uv = particle_uv(invocation.vertex_index),
      speed = length(p.velocity) })

def particle_fragment(albedo: texture2d, sampling: sampler,
                      invocation: fragment_invocation<varying>) vec4f32 =
  let base = texture_sample(albedo, sampling, invocation.value.uv, 0.0) in
  base * heat_tint(invocation.value.speed)

entry frame(particles: []particle,
            particle_count: u32,
            dt: f32,
            view_projection: mat4f32,
            albedo: texture2d,
            sampling: sampler,
            target: render_target<vec4f32>)
    ([]particle, render_target<vec4f32>) =
  let updated = map(update_particle(dt, _), particles) in
  let covered = rasterize_triangles(
    direct_draw(particle_count, 1u32),
    particle_vertex(updated, view_projection, _)) in
  let target' = shade(
    target,
    covered,
    particle_fragment(albedo, sampling, _)) in
  (updated, target')
```

`update_particle(dt, _)`, `particle_vertex(updated, view_projection, _)`, and
`particle_fragment(albedo, sampling, _)` are function call sections. The first
is an ordinary function value consumed by `map`. The invocation operations
receiving the latter two determine their vertex and fragment contexts. None of
the three helpers is an independently invocable root entry.

---

## Texture and Sampler Types

Wyn has two opaque resource types for image sampling:

| Type        | Meaning                                              |
|-------------|------------------------------------------------------|
| `texture2d` | A 2D, `f32`-sampled image.                           |
| `sampler`   | A filtering sampler.                                 |

`texture2d` and `sampler` values have no literals, equality, ordering, or
arithmetic operations. They may be entry parameters, ordinary function
arguments, or callback captures.

`texture2d` is monomorphic: its sampled type is `vec4f32`.

### Texture Operations

`texture_load(tex: texture2d, coord: vec2i32, lod: i32) -> vec4f32`
returns the texel at integer coordinate `coord` and mip level `lod` without
filtering.

`texture_sample(tex: texture2d, samp: sampler, uv: vec2f32, lod: f32) -> vec4f32`
returns a filtered sample at normalized coordinate `uv` and the
explicit mip level `lod`.

Both operations are referentially transparent. In particular,
`texture_sample` does not derive a mip level from neighboring invocations and
is valid in any stage context.

```wyn
def sample_surface(tex: texture2d, samp: sampler,
                   fragment: fragment_invocation<vec2f32>) vec4f32 =
  let filtered = texture_sample(tex, samp, fragment.value, 0.0) in
  let texel = texture_load(tex, @[0, 0], 0) in
  filtered + texel
```

---

## External Resources

### Resource Values

A resource value denotes data supplied at an entry boundary or returned from
an entry. Resource values have no literals, ordering, arithmetic, or observable
address. Except for operations defined for their resource type, they are
passed to functions and callbacks in the same manner as other values.

The following resource types are defined:

| Type | Access |
|---|---|
| `texture2d` | read-only texel loading and sampling |
| `sampler` | read-only sampling parameters |
| `render_target<C>` | color output updated by fragment shading |

Arrays are ordinary Wyn values rather than a distinct resource type. An array
entry parameter may be read by any operation that captures or receives it, and
an array entry result is an ordinary result. Scalar and record parameters are
likewise ordinary values. The language does not distinguish uniform and
storage parameters.

### Render Targets

`render_target<C>` is an opaque generic resource type. `C` is a color shape:
a numeric scalar or vector, or a tuple or record of color shapes. A render
target contains one color attachment for each scalar or vector leaf of `C` and
may also contain a depth attachment. It has no constructor, fields, or element
indexing operation.

A render target is non-copyable. `shade` and `shade_with` consume a render
target and return its updated value. The uniqueness rules therefore
make the order of successive updates explicit:

```wyn
let target1 = shade(target0, background, background_fragment) in
let target2 = shade(target1, foreground, foreground_fragment) in
target2
```

A target may be an entry parameter, an entry result, or a component of either.
It may not be stored in an array.

#### Reading a Render Target

The following operations read a render target without consuming it:

```wyn
target_load(target: render_target<C>, coord: vec2i32, sample: u32) C
target_sample(target: render_target<C>, sampler: sampler, uv: vec2f32) C
```

`target_load` reads one sample without filtering. `target_sample` filters each
color leaf of `C`; every leaf must have a floating-point scalar or vector type.
A target read may be captured by a later fragment callback. The uniqueness
rules reject a shading operation that both consumes a target and captures the
same target as a read-only input.

### External Aliasing

Two read-only resource parameters may denote the same external resource.
Parameters whose types are unique or non-copyable must satisfy the aliasing
rules for unique arguments. A program has no operation for testing whether two
resource arguments denote the same external object.

### Persistence Between Entry Invocations

Wyn provides no implicit persistent or previous-frame value. To preserve a
value across entry invocations, an entry returns it and the host supplies that
value as an argument to a later invocation. The same rule applies to arrays,
render targets, and any other externally representable result.

### No Explicit Resource Layout

Descriptor sets, binding numbers, storage classes, image layouts, and resource
transitions are not part of the Wyn language. The following forms are
therefore invalid:

```wyn
#[uniform(...)]
#[storage(...)]
#[texture(...)]
#[sampler(...)]
#[storage_image(...)]
#[view(...)]
```

Resource use is determined solely by the typed operations applied to a value.
Passing a resource to a helper, capturing it in a stage callback, or returning
an updated resource expresses all source-language data flow involving that
resource.

---

## Appendix: Wyn Compared to Other Functional Languages

This guide is intended for programmers who are familiar with other functional languages and want to start working with Wyn.

Wyn is a simple language with a complex compiler. Functional programming is fundamentally well suited to data parallelism, so Wyn's syntax and underlying concepts are taken directly from established functional languages such as Haskell and the ML family. While Wyn does add a few small conveniences (built-in array types) and one complicated and unusual feature (in-place updates via uniqueness types, see In-place Updates), a programmer familiar with a common functional language should be able to understand the meaning of a Wyn program and quickly begin writing their own programs. To speed up this process, we describe here some of the various quirks and unexpected limitations imposed by Wyn. We also recommended reading some of the example programs along with this guide. The guide does not cover all Wyn features worth knowing, so do also skim the Language Reference and the Glossary sections above.

### Basic Syntax

Wyn uses a keyword-based structure, with optional indentation solely for human readability. This aspect differs from Haskell and F#.

Names are lexically divided into identifiers and symbols:

- **Identifiers** begin with a letter or underscore and contain letters, numbers, underscores, and apostrophes.
- **Symbols** contain the characters found in the default operators (`+-*/%=!><|&^`).

All function and variable names must be identifiers, and all infix operators are symbols. An identifier can be used as an infix operator by enclosing it in backticks, as in Haskell.

Identifiers are case-sensitive, and there is no restriction on the case of the first letter (unlike Haskell and OCaml, but like Standard ML and Flix).

User-defined operators are possible, but the fixity of the operator depends on its name. Specifically, the fixity of a user-defined operator `op` is equal to the fixity of the built-in operator that is the longest prefix of `op`. For example, `<<=` would have the same fixity as `<<`, and `=<<` the same as `=`. This rule is the same as the rule found in OCaml and F#.

Top-level functions and values are defined with `def` as in Flix. Local variables are bound with `let`.

### Evaluation

Wyn is a completely pure language, with no cheating through monads, effect systems, or anything of the sort.

Evaluation is eager or call-by-value, like most non-Haskell languages. However, there is no defined evaluation order. Furthermore, the Wyn compiler is permitted to turn non-terminating programs into terminating programs, for example by removing dead code that might cause an error. Moreover, there is no way to handle errors within a Wyn program (no exceptions or similar); although errors are gracefully reported to whatever invokes the Wyn program.

The evaluation semantics are entirely sequential, with parallelism being solely an operational detail. Hence, race conditions are impossible. The Wyn compiler does not automatically go looking for parallelism. Only certain special constructs and built-in library functions (such as `map`, `reduce`, `scan`, and `filter`) may be executed in parallel.

Functions have a fixed number of arguments and every application must supply
all of them (although functions are not fully first class; see Types below).
Call sections such as `f(_, y)` construct explicit lambdas rather than
partially applying `f`. Although the `assert` construct looks like a function,
it is not.

Lambda terms are written as `|x| x + 2`.

A Wyn program is read top-down, and all functions must be declared in the order they are used, like Standard ML. Unlike just about all functional languages, recursive functions are not supported. Most of the time, you will use bulk array operations instead, but there is also a dedicated `loop` language construct, which is essentially syntactic sugar for tail recursive functions.

### Types

Wyn supports a range of integer types, floating point types, and booleans (see Primitive Types and Values). A numeric literal can be suffixed with its desired type, such as `1i8` for an eight-bit signed integer. Un-adorned numerals have their type inferred based on use. This only works for built-in numeric types.

Arrays are a built-in type. The type of an array containing elements of type `t` is written `[]t` (not `[t]` as in Haskell), and we may optionally annotate it with a size as `[n]t` (see Shape Declarations). Array values are written as `[1,2,3]`. Array indexing is written `a[i]` with no space allowed between the array name and the brace. Indexing of multi-dimensional arrays is done by chaining: `a[i][j]`. Arrays are 0-indexed.

All types can be combined in tuples as usual, as well as in structurally typed records, as in Standard ML and Flix. Non-recursive sum types are supported, and are also structurally typed. Abstract types are possible via the module system; see Modules.

If a variable `foo` is a record of type `{a: i32, b: bool}`, then we access field `a` with dot notation: `foo.a`. Tuples are a special case of records, where all the fields have a 0-indexed numeric label. For example, `(i32, bool)` is the same as `{0: i32, 1: bool}`, and can be indexed as `foo.1`.

Sum types are defined as constructors separated by a vertical bar (`|`). Constructor names always start with a `#`. For example, `#red | #blue(i32)` is a sum type with the constructors `#red` and `#blue`, where the latter has an `i32` as payload. The terms `#red` and `#blue(2)` produce values of this type. Constructor applications must always be fully saturated. Due to the structural type system, type annotations are sometimes necessary to resolve ambiguities. For example, the term `#blue(2)` can produce a value of any type that has an appropriate constructor.

Function types are written with the usual `a -> b` notation, and functions can be passed as arguments to other functions. However, there are some restrictions:

- A function cannot be put in an array (but a record or tuple is fine).
- A function cannot be returned from a branch.
- A function cannot be used as a loop parameter.

Function types interact with type parameters in a subtle way:

```wyn
def id 't (x: t) = x
```

This declaration defines a function `id` that has a type parameter `t`. Here, `t` is an unlifted type parameter, which is guaranteed never to be a function type, and so in the body of the function we could choose to put parameter values of type `t` in an array. However, it means that this identity function cannot be called on a functional value. Instead, we probably want a lifted type parameter:

```wyn
def id '^t (x: t) = x
```

Such lifted type parameters are not restricted from being instantiated with function types. On the other hand, in the function definition they are subject to the same restrictions as functional types.

Wyn supports Hindley-Milner type inference (with some restrictions), so we could also just write it as:

```wyn
def id x = x
```

Type abbreviations are possible:

```wyn
type foo = (i32, i32)
```

Type parameters are supported as well:

```wyn
type pair<A, B> = (A, B)
```

As with everything else, they are structurally typed, so the types
`pair<i32, bool>` and `(i32, bool)` are interchangeable. Most unusually,
this is also the case for sum types. The following two types are
interchangeable:

```wyn
type maybe<A> = #just(A) | #nothing

type option<A> = #nothing | #just(A)
```

Only for abstract types, where the definition has been hidden via the module system, do type names have any significance.

Size parameters can also be passed:

```wyn
type vector<[n], A> = [n]A
type i32matrix<[n], [m]> = [n]vector<[m], i32>
```

Type applications are fully saturated and first-order: aliases cannot be
partially applied and Wyn has no higher-kinded type parameters. Brackets on a
size argument distinguish it from an ordinary type argument.

---

*This specification defines the Wyn language. Implementations may be incomplete;
the language remains under active development.*
