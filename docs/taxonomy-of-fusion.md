# Taxonomy of fusion

This is a source-programmer's account of Wyn fusion: **what you write, and how the resulting computation is intended to be structured**. It covers the compiler's existing fusion families, their combinations, and useful extensions suggested by Futhark and other compilers. It deliberately does not explain the generic screma representation.

The main examples show the intended transformation. **A dagger (†) marks a current implementation shortcoming**, explained immediately below the example. Aspirations that require additional fusion capabilities are in a separate section. A successful compilation is evidence of generated structure, not a proof of runtime correctness or optimal performance.

## Reading the examples

Both columns use Wyn-like pseudocode. The right column shows the fused callbacks and the results they produce. Scheduling and storage details follow the examples in prose. Functions `f`, `g`, `p`, and `key` stand for suitably typed callbacks. Array lengths and compatible domains are assumed where a transformation needs them. Examples use pure arithmetic unless stated otherwise.

A few explanatory operators make fusion explicit without inventing temporary arrays:

* `reduce_map(xs, e, input, op)` computes each contribution with `input` and reduces with `op` and identity `e`.
* `reduce_map_emit` has the same arguments, but its callback returns `(contribution, element)`; the result is `(reduced_value, emitted_array)`.
* `scan_emit(xs, op, e, input=..., emit=...)` gives `emit` each original element and its **final inclusive prefix**. Returned values form an output array; an output action can instead write a destination directly.
* `filter_map` calls a callback that returns `Some(value)` or `None`, compacting the accepted values in input order. The compacted array carries its live length.
* `scatter_map`, `reduce_by_index_map`, and `bucket_scatter_map` compute each `(key, value)` inside the indexed operation. They retain that operation's destination, bounds, and collision semantics.

These names describe fused computations; they are not proposed Wyn built-ins. `out[k] <- v` denotes a scatter output action, and `scratch(n)` denotes an uninitialized destination of the required element type. Neither the notation nor one fused callback promises one GPU dispatch.

Three distinctions matter:

* **Vertical fusion:** an operation consumes another operation's elements. The consumer can use a scalar value directly instead of loading an intermediate array.
* **Horizontal fusion:** independent operations share an iteration domain and can execute together. Their result arrays may all still need storage.
* **Scheduling:** one fused computation can still require several GPU dispatches and internal buffers. Eliminating a source-level temporary does not eliminate synchronization scratch.

The normal parallel-entry schedules in this checkout are:

| Computation | Current generated organization |
|---|---|
| Maps | One element kernel, normally 64 lanes per workgroup, with a grid-stride loop. |
| Reductions | A cooperative chunk kernel, then a cooperative partial-result combine kernel. A returned scalar commonly adds a one-invocation publication kernel: **three dispatches** in the simple probes. |
| Scans | Chunk-local prefixes and chunk totals; a combine pass computes chunk carries; a final pass adds carries and writes the output: **three dispatches**. Full-input-size prefix scratch remains. |
| Filters | One 64-lane workgroup processes successive tiles, computes survivor ranks, and writes a compacted output plus its live length. **One dispatch does not imply scalable use of the whole GPU.** |
| Scatter / indexed reduction | An update kernel, with destination setup or copying if required. Integer reductions by index use supported atomics or a compare-exchange loop. |
| Bucket scatter | Clear counts/overflow, insert items, and publish results where necessary. The probes have three dispatches. |

The collective chunk grid and scratch sizes come from the scheduling decision. The current default uses 256 groups of 256 lanes; a workgroup loops over tiles when the input is larger. The combine stage is cooperative, not a single lane scanning all input chunks. These are scheduling details rather than additional fusion families.

Invocation-local array operations inside a callback have a different implementation, often serial loops and bounded local arrays. See [nested bodies](#nested-bodies-and-runtime-loops); a single outer dispatch is not evidence that the inner operations fused.

All families have legality conditions. Compatible source regions and iteration domains are required; captured completed results, intervening writes, extra observers, or an alternate dependency path can prevent contraction. The compiler exposes array-work helpers by inlining before this analysis; fusion itself does not generally cross an opaque function boundary. Horizontal fusion requires independence, and vertical fusion requires the right kind of connection: passing an array as a whole captured value is different from consuming its aligned elements. The planner chooses contractions greedily, so the availability of several pairwise rules is not a guarantee that every larger composition reaches the ideal result.

## Primitive-pair coverage

This matrix covers **direct element-stream connections** among all seven current primitive array-operation families. Rows produce; columns consume. `R` means reduction, `S` scan, `F` filter, `W` scatter, `H` reduce-by-index/histogram, and `B` bucket scatter.

`Yes` means an implemented family with a successful representative. `†` means restricted or incomplete; the relevant section gives the distinction. `—` means no general direct fusion rule was found. `Completion` means a reduced result is needed as a completed value, which is a different dependency from an element stream.

| Producer → consumer | Map | R | S | F | W | H | B |
|---|---|---|---|---|---|---|---|
| Map | Yes† | Yes | Yes | Yes† | Yes | Yes | Yes† |
| R | Completion | Completion | Completion | Completion | Completion | Completion | Completion |
| S | Yes | — | — | — | Yes† | — | — |
| F | Yes | Yes† | — | — | — | — | — |
| W | — | — | — | — | — | — | — |
| H | — | — | — | — | — | — | — |
| B | — | — | — | — | — | — | — |

The reduction row does not say that reductions always return scalars: array-valued reductions exist. It says the result becomes available after the collective finishes. Likewise, the output of a scatter or histogram is the **completed destination**, not the sequence of values being written. Substituting one update for a read of that destination is generally wrong.

For **horizontal fusion**, all six unordered pairs from `{map, reduce, scan}` are implemented: map+map, map+reduce, map+scan, reduce+reduce, reduce+scan, scan+scan. No general horizontal contraction was found for pairs involving filter, scatter, histogram, or bucket scatter. Their aspirational cases appear later.

## Vertical fusion: preparing and consuming elements

### 1. Map → map

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let ys = map(f, xs) in
map(g, ys)</code></pre></td>
<td valign="top"><pre><code>map(|x|
  let y = f(x) in
  g(y), xs)</code></pre></td>
</tr>
</tbody>
</table>

**Intermediate:** one scalar `y` per element; the `ys[n]` array disappears. Only the final map result needs an output array, with no second traversal. Longer map chains compose in the same way. The probe `map(|x|x*3, map(|x|x+17, xs))` produces one element kernel.

This extends to multi-input maps: `map2(h, map(f,xs), map(g,zs))` can compute the necessary producer elements at the consumer's index. Zips describe aligned inputs, not necessarily a separately stored array of pairs. Tuple-valued maps and field selection are covered by this mechanism; `unzip(map(|x|(x+17,x*3),xs))` produces one kernel writing two component outputs.

**† Nested-body shortcoming:** this is the intended elementwise rewrite at every suitable nesting level, but the tested `map(|row| map(g,map(f,row)), rows)` still emits two serial inner loops with a bounded local temporary. The top-level case works. See [nested bodies](#nested-bodies-and-runtime-loops).

### 2. Map → reduce

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let ys = map(f, xs) in
reduce(op, e, ys)</code></pre></td>
<td valign="top"><pre><code>reduce_map(xs, e, f, op)</code></pre></td>
</tr>
</tbody>
</table>

There is no full-size `ys`: `f(x)` feeds the reduction directly. The cooperative implementation reduces chunks, combines their partial results, and publishes the completed result when needed. Reduction partials remain. `all(p,xs)` and `any(p,xs)` are instances: compute a Boolean predicate and reduce it, without a Boolean array. Dot-product-style `reduce((+),0,map2((*),xs,zs))` is another instance.

The same producer may feed several independent reductions; [one producer, two reductions](#one-producer-two-reductions) shows how it is evaluated once per input element.

### 3. Map → scan

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let ys = map(f, xs) in
scan(op, e, ys)</code></pre></td>
<td valign="top"><pre><code>scan_emit(
  xs, op, e,
  input = f,
  emit = |x, prefix| prefix)</code></pre></td>
</tr>
</tbody>
</table>

The source `ys` array disappears. The current parallel implementation still writes local prefixes to an internal buffer, computes chunk carries, then combines each local prefix with its carry. Thus **map→scan fusion saves the map's array and dispatch; it does not turn this into a one-pass scan**.

### 4. Scan → map

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let ps = scan(op, e, xs) in
map(g, ps)</code></pre></td>
<td valign="top"><pre><code>scan_emit(
  xs, op, e,
  input = |x| x,
  emit = |x, prefix| g(prefix))</code></pre></td>
</tr>
</tbody>
</table>

The standalone map dispatch and final, consumer-only `ps` array disappear. Internal prefix scratch remains. The probe emits three kernels. If `ps` is also returned, the output phase writes both `ps[i] = p` and `ys[i] = g(p)`; that retained-output probe also emits three kernels.

This does **not** mean that `g` is folded into the scan's combining operator. The scan must still combine its original state correctly.

### 5. Map → filter

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let ys = map(f, xs) in
filter(p, ys)</code></pre></td>
<td valign="top"><pre><code>filter_map(xs, |x|
  let y = f(x) in
  if p(y) then Some(y) else None)</code></pre></td>
</tr>
</tbody>
</table>

There is one compaction computation, no materialized `ys`. The rank computation and live output length remain. Rejected elements still need `f` because the predicate observes its result.

**† Current recomputation:** the cooperative filter lowerer computes the mapped value for the predicate and computes it again when writing a survivor. The generated probe visibly contains both arithmetic expressions and two loads for surviving elements. The intermediate array is eliminated, but the ideal per-element register reuse is incomplete. The filter also currently uses one workgroup for the whole input.

The producer must have no other disqualifying observers. Returning `ys` as well is not handled by this element-consumer contraction in the same way as retained map→map.

### 6. Filter → map, including captured reads

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let ys = filter(p, xs) in
map(g, ys)</code></pre></td>
<td valign="top"><pre><code>filter_map(xs, |x|
  if p(x) then Some(g(x)) else None)</code></pre></td>
</tr>
</tbody>
</table>

`g` runs only on selected elements. This matters if it performs a read that is valid only after the predicate. A representative existing fixture filters indices by `xs[i] > 0`, then maps those indices to records built from captured `xs`; the record is constructed during compaction.

Pure callbacks and permitted captured read-only callbacks are supported, subject to ordering constraints. Writes and unknown effects cannot be moved through the filter this way. Extra unrelated element inputs, slices, raw observation of the filtered array, or intervening dependencies prevent the contraction.

Length observers are allowed:

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let ys = filter(p, xs) in
(map(g, ys), length(ys))</code></pre></td>
<td valign="top"><pre><code>let zs = filter_map(xs, |x|
  if p(x) then Some(g(x)) else None) in
(zs, length(zs))</code></pre></td>
</tr>
</tbody>
</table>

The probe produces one compaction kernel. Chained post-maps such as `map(h,map(g,filter(p,xs)))` also compile into that kernel. If the original `ys` is returned alongside the mapped result, the tested output instead has compaction followed by a separate map.

### 7. Filter → reduce

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let ys = filter(p, xs) in
reduce(op, e, ys)</code></pre></td>
<td valign="top"><pre><code>reduce_map(
  xs, e,
  |x| if p(x) then x else e,
  op)</code></pre></td>
</tr>
</tbody>
</table>

Compaction, survivor ranks, and the compacted array disappear. The neutral element makes rejected inputs contribute nothing. Several reductions of the same filtered stream can share this traversal. If the count is requested, add a count accumulator:

```text
reduce_map(
  xs, (e, 0),
  |x| if p(x) then (x, 1) else (e, 0),
  |(a, n), (b, m)| (op(a, b), n+m))
```

The sum/count and sum/max probes generate one chunk/combine pair, followed by scalar publication. This notation shows the logical predicate result; the emitted scalar code does not guarantee that an arbitrary predicate is evaluated only once across every accumulator.

Map preparation also composes: `reduce(op,e,filter(p,map(f,xs)))` computes `f(x)`, tests it, and contributes the accepted value, without either full-size intermediate.

**† Composition shortcoming:** `reduce(op,e,map(g,filter(p,xs)))` is intended to contribute `if p(x) then g(x) else e`, evaluating `g` only when selected. In the current checkout the simple integer probe fails during SSA lowering because the compacted producer no longer has scheduled storage. The pairwise rules exist, but this composition is incomplete. It belongs to this intended family, not to a new aspirational fusion type.

The raw compacted array must not remain observed outside the eligible reduction/count consumers.

### 8. Filter → length, including several length observers

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>length(filter(p, xs))</code></pre></td>
<td valign="top"><pre><code>reduce_map(
  xs, 0,
  |x| if p(x) then 1 else 0,
  (+))</code></pre></td>
</tr>
</tbody>
</table>

No element output, compaction, or survivor index is needed. Compatible direct length observers share the count result. The current count probe uses the reduction chunk/combine/publication schedule.

This is different from `length(map(f,xs))`: a map's length comes from its input without evaluating `f`; a filter's length depends on evaluating the predicate. Length-only map elimination is discussed under [metadata and dead outputs](#metadata-and-dead-output-elimination).

### 9. Map → scatter

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let is = map(key, xs) in
let vs = map(value, xs) in
scatter(dest, is, vs)</code></pre></td>
<td valign="top"><pre><code>scatter_map(
  dest, xs,
  |x| (key(x), value(x)))</code></pre></td>
</tr>
</tbody>
</table>

Either or both the keys and values may be produced by maps; maps may have further fused preparation. No key or value array is required merely to feed the write. The tested unique-destination program generates one scatter kernel.

The destination is a separate concern: existing untouched contents must survive unless scratch semantics permit omitting them. Fresh initialized destinations can require setup/copy work. Fusion does not change collision or out-of-bounds behavior. Conservative examples use unique in-bounds keys; Wyn permits equal-value collisions but does not promise a winner for conflicting values.

The current rule requires all relevant producer uses to belong to the consumer's direct inputs. Returning the mapped values separately blocks this contraction; the probe has a map followed by scatter.

### 10. Map → reduce-by-index / histogram

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let is = map(key, xs) in
let vs = map(value, xs) in
reduce_by_index(dest, op, e, is, vs)</code></pre></td>
<td valign="top"><pre><code>reduce_by_index_map(
  dest, op, e, xs,
  |x| (key(x), value(x)))</code></pre></td>
</tr>
</tbody>
</table>

The mapped arrays disappear. The destination bins remain and retain their initialization semantics. For supported integer operations, the final line is an atomic operation; other supported integer combiners use a compare-exchange loop. Other execution recipes may be ordered. Fusion of preparation does not itself choose a more sophisticated histogram algorithm.

`hist` is a source wrapper for this operation, not another fusion family. The map→histogram probe emits one atomic update kernel with key/value arithmetic in the kernel. A completed histogram followed by a map over its bins is a different, generally unfused connection.

### 11. Map → bucket scatter, including ranked nested maps

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let items = map(
  |x| (key(x), value(x)), xs) in
bucket_scatter_1d(dest, items)</code></pre></td>
<td valign="top"><pre><code>bucket_scatter_map(
  dest, xs,
  |x| (key(x), value(x)))</code></pre></td>
</tr>
</tbody>
</table>

The item array is eliminated. Destination storage, bucket counts, capacity checks, and overflow reporting remain. Counts describe all valid insertions, including those beyond capacity; they are not merely the number of stored items. Negative keys are discarded. Nonnegative invalid keys and exhausted capacity have the operation's overflow behavior.

There is also a dedicated multidimensional preparation path:

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let items = map(
  |a| map(|b| make_item(a,b), bs),
  as) in
bucket_scatter_2d(dest, items)</code></pre></td>
<td valign="top"><pre><code>bucket_scatter_map_2d(
  dest, as, bs,
  |a, b| make_item(a,b))</code></pre></td>
</tr>
</tbody>
</table>

`bucket_scatter_map_2d` calls its callback over the rectangular product of `as` and `bs`, inserting each item directly. Coordinate-dependent scalar bindings stay in the item computation. Compatible independent inputs supply the corresponding axes. No complete ranked item array is constructed. This applies to the supported ranked bucket operations, with static destination capacity and representable dimensions; it is not a general flattening rule for arbitrary nested maps.

**† Dynamic-size shortcoming:** the fixed-size 1D and rectangular 2D probes compile to clear/insertion/publication. The 1D probe with runtime-sized `xs:[]i32` fails with `host-computed launch has no capacity buffer`. The preparation-fusion intent is present, but that launch shape is incomplete.

### 12. Scan → scatter output action

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let flags = map(flag, xs) in
let prefixes = scan((+), 0,
  flags) in
let keys = map2(
  |x, prefix| dst(x, prefix),
  xs, prefixes) in
scatter(scratch(length(xs)),
  keys, xs)</code></pre></td>
<td valign="top"><pre><code>let out = scratch(length(xs))
scan_emit(
  xs, (+), 0,
  input = |x| flag(x),
  emit = |x, prefix|
    out[dst(x, prefix)] &lt;- x)
out</code></pre></td>
</tr>
</tbody>
</table>

**Intermediates:** the source `flags[n]`, `prefixes[n]`, and `keys[n]` arrays disappear. Each final prefix feeds the destination calculation and write directly. The standalone scatter dispatch can disappear too. Internal scan-prefix scratch still exists; maps preparing contributions or the output action can join this computation.

Current restrictions are substantial: the destination must be recognized as fresh scratch; the producer group cannot contain reductions; unrelated observations of its arrays block the contraction; the stream must be direct and unsliced; ordering and dependency checks must pass. Destination allocation cannot depend on a prefix result that is available only after the scan completes.

**† Current matching/setup shortcomings:** a scratch `replicate(length(xs),0)` probe does fuse the scatter into the third scan phase, but also emits an empty setup kernel: four dispatches instead of the desired three. The otherwise similar scratch `copy(xs)` probe misses the contraction and emits three scan phases plus a separate scatter. Returning the prefixes also prevents the current contraction. These are distinct from the successful scan-output-action path used by the radix sketch.

The radix sketch has five dispatches per pass: two reduction phases, then three scan phases with the scatter in the last phase. Its cross-phase bin computations and remaining prefix traffic are not eliminated merely by this fusion.

## Horizontal fusion: independent results from one traversal

The following table spells out all six implemented pairs. Reductions and scans retain their own combining operators and identities. Shared callbacks expose the contributions without separate mapped arrays.

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>(map(f, xs), map(g, xs))</code></pre></td>
<td valign="top"><pre><code>unzip(map(|x| (f(x), g(x)), xs))</code></pre></td>
</tr>
<tr>
<td valign="top"><pre><code>(map(f, xs), reduce(op, e, xs))</code></pre></td>
<td valign="top"><pre><code>let (r, ys) = reduce_map_emit(
  xs, e,
  |x| (x, f(x)),
  op) in
(ys, r)</code></pre></td>
</tr>
<tr>
<td valign="top"><pre><code>(map(f, xs), scan(op, e, xs))</code></pre></td>
<td valign="top"><pre><code>unzip(scan_emit(
  xs, op, e,
  input = |x| x,
  emit = |x, prefix| (f(x), prefix)))</code></pre></td>
</tr>
<tr>
<td valign="top"><pre><code>(reduce(opA, eA, xs),
 reduce(opB, eB, xs))</code></pre></td>
<td valign="top"><pre><code>reduce_map(
  xs, (eA, eB),
  |x| (x, x),
  |(a,b), (c,d)| (opA(a,c), opB(b,d)))</code></pre></td>
</tr>
<tr>
<td valign="top"><pre><code>(reduce(opA, eA, xs),
 scan(opB, eB, xs))</code></pre></td>
<td valign="top"><pre><code>reduce_scan_map(
  xs,
  reduce = (opA, eA, |x| x),
  scan = (opB, eB, |x| x))</code></pre></td>
</tr>
<tr>
<td valign="top"><pre><code>(scan(opA, eA, xs),
 scan(opB, eB, xs))</code></pre></td>
<td valign="top"><pre><code>unzip(scan_emit(
  xs,
  |(a,b), (c,d)| (opA(a,c), opB(b,d)),
  (eA, eB),
  input = |x| (x, x),
  emit = |x, prefixes| prefixes))</code></pre></td>
</tr>
</tbody>
</table>

`reduce_scan_map` returns `(total, prefixes)` from independent reduction and scan contributions over the same input. The reduction consumes input elements, not scan prefixes. `unzip` routes tuple components to separate output arrays; it need not materialize an array of tuples.

Map+map uses one element kernel. Map+reduce and reduce+reduce use a chunk/combine pair, with scalar publication when needed. Map+scan and scan+scan use three scan phases; mapped sibling outputs are written in the final phase.† Reduce+scan uses three collective phases plus scalar publication in the representative probe.

**† Same traversal is not necessarily one physical read across all phases.** With a scan present, ordinary mapped outputs are emitted during carry adjustment, and their preparation may reload input or recompute arithmetic done during the chunk phase. Returning `map(f,xs)` beside `scan(op,e,map(f,xs))` visibly computes `f` in both phases in the probe. The desired shared preparation does not currently imply a persistent register value across dispatches.

**† Equal-size distinct inputs:** the intended joint loop can read `xs[i]` and `zs[i]` even when these are different arrays. The production importer mainly identifies domains by input identity; the tested `[8]i32` inputs still produce two map kernels. The fixed-extent domain rule alone does not deliver this case end to end. Equal lengths also do not establish matching survivor positions for independently filtered arrays.

Independence is essential. These cannot use the sibling rule:

```text
s  = reduce((+),0,xs)
ys = map(|x| x+s,xs)          -- needs completed s
```

Sharing an input does not remove the completion dependency. Similarly, scan→scan is not horizontal scan+scan.

**† Dependent scan scheduling:** `scan((+),0,scan((+),0,xs))` currently fails with cyclic dispatch dependencies for both tested fixed-size and runtime-sized inputs. It should at least schedule two separate scans. This is a baseline lowering defect, independent of whether a stronger fusion is possible.

### One producer, two reductions

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let ys = map(f, xs) in
let sum = reduce((+), 0,
  map(g, ys)) in
let any = reduce((||), false,
  map(p, ys)) in
(sum, any)</code></pre></td>
<td valign="top"><pre><code>reduce_map(
  xs, (0, false),
  |x|
    let y = f(x) in
    (g(y), p(y)),
  |(s,a), (v,b)| (s+v, a||b))</code></pre></td>
</tr>
</tbody>
</table>

**Intermediates:** `ys`, `map(g, ys)`, and `map(p, ys)` disappear; `f(x)` is evaluated once. The compiler shares `y` and the input load across accumulators in the chunk phase. The two output accumulators and their combination work remain distinct.

If the original also returns `ys`, retain it in the same callback:

```text
let ((sum, any), ys) = reduce_map_emit(
  xs, (0, false),
  |x|
    let y = f(x) in
    ((g(y), p(y)), y),
  |(s,a), (v,b)| (s+v, a||b)) in
(sum, any, ys)
```

Here only `ys` needs a mapped output array; the two reduction contributions remain scalar values.

## Shared producers, retained results, and dead outputs

### Retaining an observable intermediate

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let a = map(f, xs) in
(a, map(g, a))</code></pre></td>
<td valign="top"><pre><code>unzip(map(|x|
  let y = f(x) in
  (y, g(y)), xs))</code></pre></td>
</tr>
</tbody>
</table>

The internal read of `a` disappears; its externally required array does not. The current probe produces one kernel. A mapped producer returned alongside its reduction similarly writes the producer during chunk reduction, avoiding a separate map traversal.

### Diamonds and several consumers

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let a = map(f, xs) in
let b = map(g, a) in
map2(h, a, b)</code></pre></td>
<td valign="top"><pre><code>map(|x|
  let a = f(x) in
  let b = g(a) in
  h(a, b), xs)</code></pre></td>
</tr>
</tbody>
</table>

The complete diamond can contract; both intermediate arrays disappear. The tested diamond emits one kernel. More generally, several consumers can first be grouped horizontally and then consume a shared producer vertically. The outside paths and observations still matter: arbitrary local contraction must not introduce a cycle or cross an intervening dependency.

### Sharing across a genuine barrier

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let a = map(f, xs) in
let s = reduce((+), 0, a) in
map(|y| y+s, a)</code></pre></td>
<td valign="top"><pre><code>let (s, a) = reduce_map_emit(
  xs, 0,
  |x| let y = f(x) in (y, y),
  (+)) in
map(|y| y+s, a)</code></pre></td>
</tr>
</tbody>
</table>

The `reduce_map_emit` callback uses the same scalar `y` as both its reduction contribution and its emitted array element. Here materializing `a` is a useful sharing decision. Removing its buffer would require recomputing `f` later. The current probe has two reduction phases followed by the map. It does not recompute the producer in the final map.

This should not be confused with recognizing two separate source expressions `map(f,xs)` as one shared producer. That broader coalescing/profitability problem remains open. In particular, the radix source has two separately written bin maps; they are independently fused into their consumers.

### Metadata and dead-output elimination

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>length(map(f, xs))</code></pre></td>
<td valign="top"><pre><code>length(xs)</code></pre></td>
</tr>
<tr>
<td valign="top"><pre><code>scatter(
  (#[scratch] map(f, xs)),
  is, vs)</code></pre></td>
<td valign="top"><pre><code>scatter(
  scratch(length(xs)),
  is, vs)</code></pre></td>
</tr>
<tr>
<td valign="top"><pre><code>let (live, dead) = unzip(
  map(|x| (f(x), g(x)), xs)) in
live</code></pre></td>
<td valign="top"><pre><code>map(f, xs)</code></pre></td>
</tr>
</tbody>
</table>

The compiler avoids emitting dead output expressions and demanding scratch initializer contents when only their shape is needed. This removes the dead radix `copy(xs)` initializer's full-array storage and writes. It does not remove all possible redundant stages.

**† Dispatch cleanup shortcoming:** `length(map(|x|x*3+17,xs))` has no map arithmetic or element buffer in the generated shader, but still has an empty element traversal followed by a scalar finish kernel. The dead scratch-output probe also retains a separate producer stage for the live value array. Dead-value elimination and dead-dispatch elimination are not yet equivalent.

## Other source forms that participate

### Generated inputs and wrappers

| Source form | Intended structure / classification |
|---|---|
| `map(f,iota(n))` / `tabulate(n,f)` | Generate the index in the element kernel; no stored iota array. |
| `map(f,replicate(n,k))` | Use `k` directly in each iteration; the probe emits one kernel. |
| `map2`, `zip`, `unzip`, tuple projections | Route aligned scalar components to/from fused bodies. They need not force a tuple-array temporary. |
| `copy(xs)` | Identity map; it participates in map fusion and can disappear when only scratch shape is required. |
| `all`, `any` | Predicate map→reduction. |
| `hist`, `spread` | Wrappers around indexed reduction or initialized scatter; destination setup still matters. |
| `reverse`, `rotate` | Indexing maps. A following ordinary map can consume their output; fusing an earlier computed array through their captured indexing is a different, incomplete indexed-producer case. |
| `partition(p,xs)` | Two filters in the current prelude; a shared traversal would evaluate the predicate once and route elements to two outputs.† |

**† Partition and sibling filters:** no horizontal filter rule exists yet. Moreover, returning `(filter(p,xs),filter(q,xs))` fails with `runtime-sized array requires storage` for both tested fixed-size and runtime-sized inputs. Correctly scheduling the two independent filters is a baseline lowering repair; combining them into one traversal is a further fusion goal.

### Conditional maps

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>if c then map(f, xs)
else map(g, xs)</code></pre></td>
<td valign="top"><pre><code>map(|x|
  if c then f(x) else g(x), xs)</code></pre></td>
</tr>
</tbody>
</table>

The compiler exposes a single pointwise producer while keeping the callback branch. Compatible branch domains and safely movable prefixes are required. Map chains immediately inside a branch may also be composed during this normalization. The probe emits one kernel. This does not authorize speculative execution of both callbacks or arbitrary fusion across unrelated control flow.

### Indexed demand and slices — intended, incomplete†

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let a = map(f, xs) in
a[3]</code></pre></td>
<td valign="top"><pre><code>f(xs[3])</code></pre></td>
</tr>
<tr>
<td valign="top"><pre><code>let a = map(f, xs) in
map(g, a[lo..hi])</code></pre></td>
<td valign="top"><pre><code>map(|x| g(f(x)), xs[lo..hi])</code></pre></td>
</tr>
<tr>
<td valign="top"><pre><code>let a = map(f, xs) in
map(|i| a[i], is)</code></pre></td>
<td valign="top"><pre><code>map(|i| f(xs[i]), is)</code></pre></td>
</tr>
</tbody>
</table>

**† None of these three probes eliminates the producer today.** The slice and gather examples produce two map kernels and store `a`; the constant-index example computes the full `a`, then reads one element in a finish kernel.

There is an indexed-expansion candidate in the planner, but the inspected production import initializes its indexed-use sets without populating them. Slice-chain vocabulary likewise does not establish a working source-to-lowering path. A stale comment in runtime-index normalization describes static-index fusion; the current pipeline and emitted code must take precedence over that comment.

The retained-array path reads `a[lo+j]` correctly, but still materializes the producer. Arbitrary gather fusion can duplicate `f` for repeated indices or skip unrequested elements, so it needs stronger reasoning than whole-stream map→map.

### Nested bodies and runtime loops

<table>
<thead>
<tr><th>Original</th><th>Fused</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>map(|row|
  reduce((+), 0, map(f, row)),
  rows)</code></pre></td>
<td valign="top"><pre><code>map(|row|
  reduce_map(row, 0, f, (+)),
  rows)</code></pre></td>
</tr>
</tbody>
</table>

This invocation-local map→reduce case does eliminate the mapped local array. The inner reduction is serial within an invocation; fusion and inner parallelism are separate decisions.

**† Nested map→map:** the corresponding map→map probe currently retains two serial inner loops and an intermediate `array<i32,8>`. It should have one loop writing `g(f(row[j]))`. A blanket claim that fusion is complete inside all callbacks would be false.

The compiler also supports host orchestration for a class of counted array-state loops. Each host iteration dispatches the iteration's fused kernels in dependency order and then swaps the carried buffers.

This lets a radix pass use multiple GPU threads and multiple dispatches, while passes remain sequential. It is not fusion between loop iterations, general loop interchange, or a guarantee of flattening arbitrary nested parallelism. Other control flow can remain invocation-local.

### Scalar publication and adjacent scalar work

Scalar output expressions can sometimes be attached to a stage that already has their inputs. A filter's live count can, for example, be used to write a small draw record in lane zero of compaction. Scalar-only scheduled work can also be grouped. These are useful dispatch eliminations adjacent to array fusion.

**† Availability restriction:** a cooperative reduction followed by scalar arithmetic, such as `reduce((+),0,xs)*3+17`, still produces a finish kernel in the probe. Do not infer from the epilogue machinery that every scalar consumer is already attached to the producing kernel.

## Aspirational fusion families and extensions

These extend the current capability rather than merely completing an already-selected lowering path. Each sketch states a source-visible goal. External sources establish the inspiration, not a claim that Futhark automatically optimizes every Wyn-like example below.

### A. Fusion through layout transformations

<table>
<thead>
<tr><th>Original</th><th>Aspirational fused form</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let prepared = map(
  |row| map(f, row), matrix) in
map(|row| map(g, row),
  transpose(prepared))</code></pre></td>
<td valign="top"><pre><code>tabulate(cols(matrix), |j|
  tabulate(rows(matrix), |i|
    g(f(matrix[i][j]))))</code></pre></td>
</tr>
</tbody>
</table>

This example assumes a rectangular matrix and elementwise `f` and `g`. The fused callback reads the original coordinates directly; neither `prepared` nor a transposed intermediate is stored.

Extend the intended slice handling to rank changes, reshapes, and permutations with explicit coordinate correspondence. Futhark moves compatible slices and rearrangements through mapped computations; its checked-in fusion implementation includes these transformations. Arbitrary reshapes of a collective's domain still need their own legality argument. [Futhark transformation rules at the inspected local revision](https://github.com/diku-dk/futhark/blob/3a5bc16d91763a1d5aa0c66dea0bd0b4287ab68a/src/Futhark/Optimise/Fusion/TryFusion.hs).

### B. General output actions, including retained results

<table>
<thead>
<tr><th>Original</th><th>Aspirational fused form</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let ps = scan(op, e, xs) in
(ps, scatter(
  dest, map(key, ps), map(value, ps)))</code></pre></td>
<td valign="top"><pre><code>let out = copy(dest)
let ps = scan_emit(
  xs, op, e,
  input = |x| x,
  emit = |x, prefix|
    out[key(prefix)] &lt;- value(prefix)
    prefix)
(ps, out)</code></pre></td>
</tr>
</tbody>
</table>

The output action both stores the scatter value and returns the prefix for `ps`. Copying `dest` represents preservation of initialized entries; ownership may permit reusing its storage.

Permit initialized destinations, still-observed prefixes, multiple destinations, and suitable output actions to compose without separate element traversals. Preserve initialization, aliases, write ordering, and collision contracts. This is broader than the current fresh-scratch, unobserved-producer restriction. Futhark's scan/scatter work attaches output actions to prefix production; a destination whose size depends on the completed scan can still block the transformation. [Futhark scan/scatter explanation](https://futhark-lang.org/blog/2026-03-24-scan-scatter-fusion.html).

### C. Several histograms or indexed destinations in one traversal

<table>
<thead>
<tr><th>Original</th><th>Aspirational fused form</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let h1 = hist(d1, op1, e1,
  map(k1, xs), map(v1, xs)) in
let h2 = hist(d2, op2, e2,
  map(k2, xs), map(v2, xs)) in
(h1, h2)</code></pre></td>
<td valign="top"><pre><code>hist2_map(
  (d1, op1, e1),
  (d2, op2, e2),
  xs, |x|
    ((k1(x), v1(x)),
     (k2(x), v2(x))))</code></pre></td>
</tr>
</tbody>
</table>

`hist2_map` routes the two `(key, value)` contributions from one callback to two independent indexed reductions. Each destination retains its own initialization and collision semantics.

Futhark has horizontal histogram composition. Wyn currently prepares each indexed consumer through map fusion but does not horizontally combine independent histogram operations. Sharing input preparation is useful even though destination bins and synchronization remain. [Futhark histogram rules](https://github.com/diku-dk/futhark/blob/3a5bc16d91763a1d5aa0c66dea0bd0b4287ab68a/src/Futhark/Optimise/Fusion/TryFusion.hs).

More general accumulation can emit a variable number of writes per input, or combine map outputs and updates in one traversal. Futhark's accumulator work and the accumulation effects it discusses in Dex provide a model for reasoning about such actions; no reading of an unfinished destination should be smuggled into the body. [Parallel accumulation](https://futhark-lang.org/blog/2026-02-23-accumulators.html).

### D. More compaction compositions

<table>
<thead>
<tr><th>Original</th><th>Aspirational fused form</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>filter(q, filter(p, xs))</code></pre></td>
<td valign="top"><pre><code>filter(|x|
  if p(x) then q(x) else false,
  xs)</code></pre></td>
</tr>
<tr>
<td valign="top"><pre><code>(filter(p, xs), filter(q, xs))</code></pre></td>
<td valign="top"><pre><code>filter_map2(xs, |x|
  (if p(x) then Some(x) else None,
   if q(x) then Some(x) else None))</code></pre></td>
</tr>
<tr>
<td valign="top"><pre><code>partition(p, xs)</code></pre></td>
<td valign="top"><pre><code>filter_map2(xs, |x|
  if p(x) then (Some(x), None)
  else (None, Some(x)))</code></pre></td>
</tr>
<tr>
<td valign="top"><pre><code>scan(op, e, filter(p, xs))</code></pre></td>
<td valign="top"><pre><code>filter_scan(
  xs, op, e,
  |x| if p(x) then Some(x) else None)</code></pre></td>
</tr>
</tbody>
</table>

`filter_map2` compacts two optional result streams from one callback; their output lengths are independent. `filter_scan` updates the prefix only for `Some(x)` and emits those selected prefixes in compacted order. These are aspirational pseudocode operators.

**† Current compaction chains:** filter→filter compiles as two separate compactions. Filter→scan has no direct fusion rule and currently fails with cyclic dispatch dependencies for both tested fixed-size and runtime-sized inputs. Correct separate compaction and scan scheduling must work even without fusion. The [partition and sibling-filter limitation](#generated-inputs-and-wrappers) also applies to the two-output cases above.

These need explicit compaction-domain and evaluation handling; equal result lengths are not enough. The first three are natural stream/compaction fusion goals. Futhark's scan/output-action formulation suggests the machinery for parallel compaction; staged stream fusion supplies a broader precedent for composing maps, filters, and nested streams. The scan-after-filter row is a proposed design, not an assertion of automatic Futhark support. [Stream Fusion, to Completeness](https://arxiv.org/abs/1612.06668).

Completing **filter→map→reduce** is the nearer implementation repair; its intended guarded reduction already belongs to the current taxonomy. Futhark's user guide also describes identity masking as an efficient alternative to actually filtering before a reduction, without promising that every source expression is automatically rewritten. [Filter/reduce guidance](https://futhark-lang.org/examples/filter-reduce.html).

### E. Chunked fusion across collective boundaries

<table>
<thead>
<tr><th>Original</th><th>Aspirational fused form</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>reduce(add, 0,
  map(g, scan(op, e, xs)))</code></pre></td>
<td valign="top"><pre><code>let step = |(prefix, total), x|
  let next = op(prefix, x) in
  (next, add(total, g(next))) in
let (_, result) = fold_chunks(
  xs, (e, 0),
  |state, chunk| fold(step, state, chunk)) in
result</code></pre></td>
</tr>
</tbody>
</table>

`fold_chunks` carries `(prefix, total)` between chunks. The inner fold spells out the required data dependence and does not allocate a prefix array. This is an ordered semantic sketch: obtaining parallel work requires suitable chunk summaries or a parallel scan within each chunk; executing the fold serially would not establish GPU scheduling parity.

A chunked representation can avoid a full completed-prefix array and preserve useful parallel work within chunks. It must retain the cross-chunk scan dependence; it is not equivalent to independent map fusion. Futhark's published streaming fusion rules explicitly discuss converting map/reduce/scan computations into compatible chunked forms. This is a direction for Wyn, with profitability and parallelism to evaluate, not a promise that all dependent collectives collapse into one parallel kernel. [Futhark PLDI 2017, streaming fusion](https://futhark-lang.org/publications/pldi17.pdf).

**† Current scan→reduce structure:** the tested returned-scalar case materializes a complete scan before running the reduction, for six dispatches in total. The full prefix array and collective boundary remain.

### F. Nested and segmented fusion

<table>
<thead>
<tr><th>Original</th><th>Aspirational fused form</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>map(|row| map(g, map(f, row)), rows)</code></pre></td>
<td valign="top"><pre><code>map(|row|
  map(|x| g(f(x)), row), rows)</code></pre></td>
</tr>
<tr>
<td valign="top"><pre><code>map(g, flatten(map(make_segment, xs)))</code></pre></td>
<td valign="top"><pre><code>concat_generate(xs, |x, emit|
  generate_segment(x, |y|
    emit(g(y))))</code></pre></td>
</tr>
</tbody>
</table>

In the second example, `make_segment(x)` collects the elements produced by `generate_segment(x, emit)`. `concat_generate` places each segment's emitted results in source order. Passing `g` into the producer's output callback removes the complete intermediate segments and flattened array; segment sizes and offsets may still need storage.

The first example completes ordinary fusion inside nested bodies. The second requires segmented-domain reasoning beyond today's primitive vocabulary. Flattened indices, empty segments, varying segment lengths, and cross-segment reductions/scans must remain correct. Futhark's nested-parallelism work motivates combining fusion with a separate flattening/scheduling choice; inner fusion alone does not require global flattening. [Futhark's nested-parallelism design](https://futhark-lang.org/publications/pldi17.pdf).

### G. Tiled producer/consumer fusion and selective recomputation

<table>
<thead>
<tr><th>Original</th><th>Aspirational fused form</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>let tmp =
  expensive_image_transform(image) in
stencil(tmp)</code></pre></td>
<td valign="top"><pre><code>map_tiles(shape(image), |tile|
  let halo = expand(tile, stencil_radius) in
  let local = tabulate_region(halo, |p|
    transformed_pixel(image, p)) in
  stencil_tile(local, tile))</code></pre></td>
</tr>
</tbody>
</table>

`transformed_pixel` computes one result of `expensive_image_transform`; `map_tiles` assembles the output tiles. Each tile keeps only the producer region needed by its stencil, including the halo. Boundary handling must match the original operations.

This can avoid a global intermediate while controlling recomputation and local memory. It generalizes element substitution to a consumer's required region. MLIR's structured fusion computes the producer region needed by a consumer tile using indexing maps; overlapping tiles may deliberately recompute values. The same cost question applies to arbitrary gathers and shared producers in Wyn. [MLIR producer/consumer fusion and rematerialization](https://mlir.llvm.org/docs/Tutorials/transform/Ch0/#producerconsumer-fusion-and-rematerialization).

This is substantially more than treating every indexing expression as an ordinary stream edge. Repeated indices, expensive producers, register pressure, and local storage can change which choice is best.

### H. Sharing and profitability across whole programs

<table>
<thead>
<tr><th>Original</th><th>One possible shared form</th></tr>
</thead>
<tbody>
<tr>
<td valign="top"><pre><code>(consumer_A(map(f, xs)),
 consumer_B(map(f, xs)))</code></pre></td>
<td valign="top"><pre><code>let ys = map(f, xs) in
(consumer_A(ys), consumer_B(ys))</code></pre></td>
</tr>
</tbody>
</table>

This form computes and retains one shared array. When compatible consumers can fuse, a shared callback may avoid that storage; when retention costs more than recomputation, keeping separate callbacks can be preferable.

Recognize equivalence where valid, then make a deliberate choice among these structures. The element cache shares the **same producer identity within a phase**, not arbitrary equivalent computations across phases. A global profitability decision should account for repeated work, stored bytes, dispatches, and resource pressure. MLIR's rematerialization discussion provides the concrete storage-versus-recomputation precedent; this row proposes a broader decision policy for Wyn, not an existing automatic solution there.

### Related scheduling goal: avoid full-size scan scratch

The desired scan→map or scan→scatter output action can also run in a scan algorithm that publishes globally correct prefixes without a separate full-array adjustment pass. NVIDIA's decoupled-lookback work provides a one-pass scan approach. This would make existing fusion more valuable by reducing internal traffic; it is **a scan implementation/synchronization change, not another source fusion type**. Portability and forward-progress requirements need an explicit design. [Single-pass parallel prefix scan](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back).

## Transformations that need additional laws

Several tempting rewrites are not generic fusion rules:

* `map(g,scatter(dest,is,vs))` cannot simply apply `g` to each written value: untouched destination elements also need `g`, and collisions must retain their meaning.
* Mapping completed histogram bins cannot generally move `g` to histogram contributions. A suitable homomorphism, including neutral and initial values, would be needed.
* A reduction result captured by a map must complete first. Some special algebraic reformulation may exist, but horizontal fusion is not it.
* Dependent scans cannot use the independent product-scan rule. Particular operators may admit a larger associative summary; that is a separately justified algebraic transformation.
* Moving work out of a filter guard can evaluate reads or partial functions on rejected inputs. Purity alone does not establish that this is safe.

Wyn's prelude currently states a commutativity requirement for its collective operators more broadly than some internal comments do. The examples here use compatible operations and do not resolve that language-contract discrepancy. Futhark's operator contracts should not silently be substituted for Wyn's.
