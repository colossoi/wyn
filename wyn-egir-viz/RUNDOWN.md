# Writing an EGIR Pass Rundown

A pass rundown explains one inspector checkpoint to a reader who understands
programming and compilers but does not yet know that pass's internal model. It
is not a source-code tour or a list of changed Rust fields. Its central job is
to explain why the transformation exists and what becomes easier, safer, or
more explicit afterward.

Base every rundown on the current local compiler implementation. Inspect the
pass, its input and output typestates, the constructors of the before state,
and the immediate consumers of the after state. Do not infer semantics only
from names or from the inspector's rendering.

## Required structure

### 1. Important concepts

Start with a small, pass-specific glossary, normally three to five concepts.
Define only terms needed to understand this transformation. Prefer concepts
such as "output route," "segmented space," or "publication" over generic
pipeline terms such as "EGIR," "metadata," or the name of the preceding pass.

Definitions should explain the concept's role in this pass, not merely expand
an abbreviation.

### 2. What the pass does

State the exact phase boundary and the invariant established by the pass:

```text
Input typestate
    -> pass_name
Output typestate
```

Describe the transformation in semantic units. Distinguish among:

- changes to the graph or operations themselves;
- in-tree metadata stored on operations;
- sidecar metadata stored on entries or the program;
- analysis results that are merely derived; and
- allocation, scheduling, or physicalization decisions deferred to later
  passes.

Call out important things that deliberately do not change.

### 3. Example Wyn program

Give one copy-pasteable program for the inspector before narrating the
transformation. Keep it to a few lines while exercising as many characteristic
branches of the pass as possible. Prefer an example where two related
operations receive meaningfully different metadata over a larger program
containing many redundant operations.

Verify that the example compiles through the current local pipeline and
actually exhibits the claimed transformation. Do not provide a hypothetical
example as though it were tested. Refer back to its definitions and operations
throughout the remaining sections so the explanation of why and improvement
stays grounded in something the reader can see.

### 4. Why it changes and what improves

This is the main section. For every important new or changed field, answer:

1. What was implicit, unknown, or unsafe before?
2. Why is this pass the right boundary to resolve it?
3. Which immediate downstream pass consumes the result?
4. What would that consumer otherwise need to rediscover or conservatively
   assume?

Do not describe a field change as an improvement merely because the after
state contains more metadata. Explain the compiler property it enables: for
example, legal fusion, observable-output preservation, resource-conflict
ordering, deferred allocation, or deterministic provenance.

If a fact appears to be available earlier or useful only later, say so. Discuss
whether it could move backward into construction or forward into planning, and
identify the genuinely nonlocal analysis—if any—that requires this boundary.

### 5. Before and after

Show compact, disciplined fragments of the inspector syntax. Use the same
field names and recursive record/variant grammar as `IR_SYNTAX.md`; do not
invent an explanatory mini-language.

Cover both in-tree and sidecar state when the pass changes both. Pair each
before/after fragment with a short explanation of why it differs. Avoid dumping
large listings or raw Rust `Debug` output.

### 6. What to look for

End with concrete comparison instructions. Tell the reader:

- which pass to select;
- which definition and operations to locate;
- the exact fields or route entries expected to change;
- which identities, operands, or bodies should remain stable;
- how corresponding bindings, slots, values, or resources should match across
  the listing; and
- when a field is visible only in the operation detail rather than inline.

Prefer observable checks such as "the final Screma gains `output_slots: [0]`"
over vague prompts such as "notice the richer semantic state."

## Quality checks

Before presenting a rundown, confirm that it:

- uses only a few concepts and that all are specific to the pass;
- leads with why and improvement rather than implementation chronology;
- separates semantic facts from representation and allocation choices;
- distinguishes in-tree data from sidecars;
- explains unchanged fields when their stability is meaningful;
- identifies the first downstream consumer of each important after-state fact;
- does not claim that derived metadata changes runtime behavior by itself;
- does not fabricate a natural-language transformation summary that would
  require an LLM to generate reliably;
- uses an example that has been checked against the local compiler; and
- gives enough precise inspection cues that the reader can independently
  confirm every major claim in the before/after panes.

## Compact template

````markdown
# `<pass_name>` rundown

## Important concepts
- Three to five pass-specific definitions.

## What the pass does
- Input typestate -> output typestate.
- Changed graph, in-tree metadata, and sidecars.
- Explicit non-goals.

## Example
```wyn
entry ...
```

## Why this is an improvement
- One reason per meaningful change.
- Immediate downstream consumers.
- Any questionable phase placement or tradeoff.

## Before and after
- Small syntax fragments with explanations.

## What to look for
- Exact operations, fields, relationships, and intentional non-changes.
````
