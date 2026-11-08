# Metal compiler failure for selected large array state

Compile and validate:

```sh
target/release/wyn build fold/test/metal_array_state_init.wyn -t wgsl -o /tmp/metal-array-state.wgsl
fold/tools/validate-wgsl/target/release/fold-validate-wgsl /tmp/metal-array-state.wgsl
```

Run with the user-authorized increased deadline:

```sh
python3 fold/deadline.py --seconds 30 extra/viz/target/release/viz pipeline /tmp/metal-array-state.wgsl --entry repro --headless --max-frames 1 --storage-bytes previous:83312
```

On the Radeon Pro 570 / macOS 13.7.8, the unchanged Viz binary fails while
creating the compute pipeline with `Compiler encountered an internal error`.
The equivalent temporary source was verified under the 30-second process-group
SIGKILL deadline. This is an actual Metal failure before the deadline, not a
timeout. No simulation executes.

The complete Fold application fails identically compiling
`fold__compute_0_compute`. This reduction contains only selection between
initialized and loaded arrays; no physics, rendering, browser, or solver.
The generated kernel materializes multiple large function-local arrays.
That code shape is observed; the precise Metal compiler defect is not established.
No generated shader patch or application workaround is included.
