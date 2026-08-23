# wyn-staged-ir

`wyn-staged-ir` owns the target-independent dataflow graph used after semantic
optimization and before physical recipe selection. It is independent of
`wyn-core`: stage bodies, origins, value types, and resident storage are opaque
generic payloads.

The graph has only two identities:

- `StageId` identifies an executable stage.
- `FlowId` identifies a typed resident value produced by one stage.

A finalized `StagedIr` guarantees that every resident flow has a valid
producer and at least one stage consumer or published output, every incidence
is recorded on both the stage and flow, and the stage graph is acyclic.
Resource-backed host inputs are represented separately from resident flows;
host outputs are resident flows marked as published. There are no separately
numbered input or output port objects.

Lowering constructs the graph with `StagedIrBuilder::add_stage`, `add_flow`,
`add_consumer`, `add_external_input`, and `publish`, then calls `finish` to
validate it. Finalized topology is private. Subsequent compiler passes may
mutate a stage body with `stage_body_mut`, and the phase transition may replace
all body payloads with `map_stage_bodies`; neither operation can invalidate the
graph.
