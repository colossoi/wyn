# Writing an EGIR Sub-Pass Rundown

A rundown explains one **sub-pass**: an independently meaningful analysis,
rewrite, validation, or other traversal of the program. It is not automatically
one rundown per inspector checkpoint. A checkpoint is an observable snapshot or
typestate boundary; the transition between two checkpoints may orchestrate
several sub-passes, and each of those sub-passes deserves its own rundown when
it has a distinct purpose.

Use these terms consistently:

- **Checkpoint** — a compiler state the inspector can select and compare. It is
  often, but not necessarily, a public Rust typestate.
- **Transition/orchestrator** — the entry point that consumes one checkpoint and
  produces the next while sequencing one or more sub-passes.
- **Sub-pass** — a function that makes a distinct full-program, per-definition,
  or fixpoint pass over the relevant tree/graph and establishes its own useful
  invariant.
- **Helper** — local machinery used during a sub-pass, such as visiting one
  node, constructing one record, or calculating one field. Helpers do not get
  separate rundowns merely because they are separate functions.

A fixpoint driver is one sub-pass when its repeated traversals jointly implement
one transformation. A verifier is also a sub-pass when it independently walks
the program and checks an invariant, but its rundown must describe validation,
not imply that it transforms the IR. When an orchestrator contains several
sub-passes, give the orchestrator only a short index or overview; do not replace
the individual rundowns with one blended checkpoint story.

Base every rundown on the current local compiler implementation. Inspect the
sub-pass, its enclosing transition, the state immediately before and after its
work, and the immediate consumer of its result. Do not infer semantics only
from names or from the inspector's rendering.

## Required structure

### 1. Important concepts

Start with a small, sub-pass-specific glossary, normally three to five concepts.
Define only terms needed to understand this transformation. Prefer concepts
such as "output route," "segmented space," or "publication" over generic
pipeline terms such as "EGIR," "metadata," or the name of the preceding
checkpoint.

Definitions should explain the concept's role in this sub-pass, not merely
expand an abbreviation.

### 2. Where the sub-pass runs

Name both the enclosing checkpoint transition and this sub-pass's exact place
inside it:

```text
Input checkpoint / typestate
    -> transition_orchestrator
       1. earlier_sub_pass
       2. sub_pass_being_explained   <-- this rundown
       3. later_sub_pass
Output checkpoint / typestate
```

If no public typestate or inspector snapshot exists between adjacent
sub-passes, name the internal invariant each one receives and establishes. Do
not attribute all changes visible at the output checkpoint to the selected
sub-pass. If the inspector exposes intermediate snapshots, identify the exact
before and after snapshots to compare.

Describe the selected sub-pass in semantic units. Distinguish among:

- changes to the graph or operations themselves;
- in-tree metadata stored on operations;
- sidecar metadata stored on entries or the program;
- analysis results that are merely derived;
- validation that changes no IR; and
- allocation, scheduling, or physicalization decisions deferred to later
  sub-passes.

Call out important things that deliberately do not change.

### 3. Example Wyn program

Give one copy-pasteable program for the inspector before narrating the
transformation. Keep it to a few lines while exercising as many characteristic
branches of the selected sub-pass as possible. Prefer an example where two
related operations receive meaningfully different metadata over a larger
program containing many redundant operations.

Verify that the example compiles through the current local pipeline and
actually exhibits the claimed transformation. Do not provide a hypothetical
example as though it were tested. Refer back to its definitions and operations
throughout the remaining sections so the explanation of why and improvement
stays grounded in something the reader can see.

### 4. Why it changes and what improves

This is the main section. For every important new or changed field, answer:

1. What was implicit, unknown, or unsafe before?
2. Why is this sub-pass the right boundary to resolve it?
3. Which immediate downstream sub-pass or consumer uses the result?
4. What would that consumer otherwise need to rediscover or conservatively
   assume?

Do not describe a field change as an improvement merely because the after
state contains more metadata. Explain the compiler property it enables: for
example, legal fusion, observable-output preservation, resource-conflict
ordering, deferred allocation, or deterministic provenance.

If a fact appears to be available earlier or useful only later, say so. Discuss
whether it could move backward into construction or forward into planning, and
identify the genuinely nonlocal analysis—if any—that requires this boundary.

For a validation sub-pass, explain which invariant it checks, why that
invariant becomes required at this point, and which later code is therefore
allowed to rely on it. Do not invent a before/after mutation for a verifier.

### 5. Before and after

Show compact, disciplined fragments of the inspector syntax. Use the same
field names and recursive record/variant grammar as `IR_SYNTAX.md`; do not
invent an explanatory mini-language.

Cover both in-tree and sidecar state when the sub-pass changes both. Pair each
before/after fragment with a short explanation of why it differs. Avoid dumping
large listings or raw Rust `Debug` output.

The fragments must isolate this sub-pass. When only checkpoint-level snapshots
are available, explicitly mark changes belonging to earlier or later sibling
sub-passes and do not present the aggregate diff as this sub-pass's work. A
validation sub-pass should instead show representative valid and invalid shapes
or the stable state it certifies.

### 6. What to look for

End with concrete comparison instructions. Tell the reader:

- which checkpoint and, when available, sub-pass snapshot to select;
- which definition and operations to locate;
- the exact fields or route entries expected to change;
- which identities, operands, or bodies should remain stable;
- how corresponding bindings, slots, values, or resources should match across
  the listing;
- which visible differences belong to sibling sub-passes in the same
  transition; and
- when a field is visible only in the operation detail rather than inline.

Prefer observable checks such as "the final Screma gains `output_slots: [0]`"
over vague prompts such as "notice the richer semantic state."

## Quality checks

Before presenting a rundown, confirm that it:

- covers one independently meaningful sub-pass, not merely one checkpoint;
- names the enclosing checkpoint transition and neighboring sub-passes;
- uses only a few concepts and that all are specific to the sub-pass;
- leads with why and improvement rather than implementation chronology;
- separates semantic facts from representation and allocation choices;
- distinguishes in-tree data from sidecars;
- explains unchanged fields when their stability is meaningful;
- identifies the first downstream consumer of each important result;
- does not claim sibling sub-pass changes as its own;
- treats verifier traversals as validation rather than transformation;
- does not claim that derived metadata changes runtime behavior by itself;
- does not fabricate a natural-language transformation summary that would
  require an LLM to generate reliably;
- uses an example that has been checked against the local compiler; and
- gives enough precise inspection cues that the reader can independently
  confirm every major claim in the before/after panes.

## Compact template

````markdown
# `<sub_pass_name>` rundown

## Important concepts
- Three to five sub-pass-specific definitions.

## Where the sub-pass runs
- Input checkpoint -> orchestrator -> output checkpoint.
- Neighboring sub-passes and this sub-pass's position among them.
- Internal before/after invariant when there is no intermediate checkpoint.
- Changed graph, in-tree metadata, sidecars, or validation result.
- Explicit non-goals and changes owned by sibling sub-passes.

## Example
```wyn
entry ...
```

## Why this is an improvement
- One reason per meaningful change.
- Immediate downstream sub-pass or consumer.
- Any questionable phase placement or tradeoff.

## Before and after
- Small syntax fragments that isolate this sub-pass.
- For validation, representative shapes or the stable state being certified.

## What to look for
- Exact checkpoint/sub-pass snapshots, operations, fields, relationships, and
  intentional non-changes.
````
