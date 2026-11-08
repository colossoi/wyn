# Unbound record capture during graphics source import

Compile only; no browser or simulation is needed:

```sh
target/release/wyn build fold/test/array_record_fragment_capture.wyn --graphics -t wgsl --max-warnings 0 -o /tmp/record-capture.wgsl
```

Fixed by master commit `9e4a1138` (Rebuild aggregate graphics captures from
external leaves). Verified with a rebuilt release compiler: this reproducer
now compiles successfully.

Expected: successful compilation. Previously on master `3beb084e`:

```text
error: egglog output: unbound TLC symbol SymbolId(304) during source import
```

The symbol number depends on surrounding definitions. The full Fold application
reports `SymbolId(1098)` after the two module-resolution fixes.

The single-file source constructs a record with an array and nested scalar
settings. Its vertex callback indexes the array, while its fragment callback
passes the nested settings to a scalar helper. Removing the array-bearing state
from the case allows compilation. Geometry is intentionally degenerate: this
is a compile-only reproducer, not a rendering test.

No compiler fix or application workaround is included.
