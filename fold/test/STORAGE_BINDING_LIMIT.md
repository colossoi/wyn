# Generated kernel exceeds Metal storage-binding limit

Fixed by master commit `912ddff3` (Pack scalar output storage during ABI
allocation). Verified on Metal using the unchanged Viz binary: all 30 scalar
results are 1.0. The reduced kernel now has three storage bindings. Fold also
passes bind-group layout creation.

Compile:

```sh
target/release/wyn build fold/test/storage_binding_limit.wyn -t wgsl -o /tmp/storage-binding-limit.wgsl
```

The generated host program declares 32 storage buffers for one kernel:
30 scalar output buffers, one input buffer, and one input-length buffer.
WGSL validates, but the existing unchanged Viz binary rejects its layout on the
Radeon Pro 570 Metal adapter, whose per-stage storage-buffer limit is 31.

Verified with device access and a five-second process-group SIGKILL deadline:

```sh
python3 fold/deadline.py --seconds 5 extra/viz/target/release/viz pipeline /tmp/storage-binding-limit.wgsl --entry repro --headless --max-frames 1 --storage-bytes input:120
```

Actual error:

```text
Too many bindings of type StorageBuffers in Stage ShaderStages(COMPUTE), limit is 31, count was 32.
```

The full Fold application generates 37 storage buffers for its first kernel
and fails at the same validation point. Neither run dispatched a simulation
frame. Compiler HEAD was `32f7a1a9`. No application workaround or Viz source
change is included.
