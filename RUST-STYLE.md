# Rust Style Guide

This guide records the conventions for Rust code in the Wyn repository. Follow
`rustfmt.toml` for mechanical formatting; the rules below cover choices that a
formatter cannot make.

## Imports and paths

- Prefer module-level `use` declarations over inline module paths.
- Use an inline module path only when at least one of the following is true:
  - it disambiguates a name collision;
  - it is a one-time use of an item from the standard library (`std`, `core`, or
    `alloc`).
- Keep imports as narrow as practical. Do not use glob imports outside a
  prelude or another module deliberately designed for glob import.
- Do not introduce aliases merely to shorten a name. Use an alias when it
  resolves ambiguity or communicates the role of the imported item.

## Errors, panics, and invariants

- In non-test code, every use of `Option::unwrap`, `Option::expect`,
  `Result::unwrap`, or `Result::expect` requires the user's explicit approval.
  This includes indirect results such as poisoned lock guards.
- Prefer `?`, structured error types, and useful context at subsystem
  boundaries. Wyn uses `thiserror` for structured errors.
- When a required `Option` value is missing, prefer a diverging
  `let Some(value) = option else { return Err(...); };` over
  `option.ok_or(...)?` or `option.ok_or_else(...)?`. This keeps the exceptional
  path explicit and the successful binding on the main path.
- Use `Option` only when absence is an expected state. Use `Result` when the
  caller needs to distinguish failure from absence.
- Do not silently discard a `Result`. Handle it, propagate it, or document why
  ignoring it is correct.
- Assertions are for programmer invariants, not recoverable input errors. Make
  assertion messages describe the violated invariant and include the values
  needed to diagnose it.
- Avoid sentinel values when the type system can represent the states directly
  with an enum or `Option`.

## Types and APIs

- Whenever practical, make invalid states impossible to represent.
- Prefer borrowing in APIs when the callee does not need ownership. Do not add
  `clone` merely to satisfy the borrow checker without first understanding the
  ownership relationship.
- Accept slices, `&str`, and iterators when callers should not need to allocate
  a particular owned collection.
- Preserve exhaustive matching when practical. A wildcard arm should not hide
  newly added enum variants that require deliberate handling.

## Control flow and implementation

- Prefer early returns and `let ... else` for invalid or exceptional paths so
  the main path remains easy to follow.
- Keep functions focused. Extract a helper when it gives a concept a useful
  name or isolates an invariant, not solely to reduce line count.
- Avoid cleverness in compiler passes. Make phase assumptions, state
  transitions, and ordering requirements visible in names, types, or concise
  comments.

## New code and compiler layers

- Before adding roughly 100 or more lines of code, first consider whether the
  required functionality can be obtained by making existing code more general
  or reusing existing functionality.
- Much of the compiler is structured at two levels:
  1. the representation, builder, and mutation APIs for a low-level IR;
  2. higher-level transformation operations that implement the business logic
     of compiler passes.
- Respect the boundary between these levels. Low-level APIs should provide
  reusable operations and preserve representation invariants; higher-level
  transformations should orchestrate those operations to implement a pass.
- When adding functionality, first consider whether it should be expressed as
  orchestration of functions already available through the lower-level API.
  Extend that API when a reusable primitive is genuinely missing.

## Unsafe code

- Do not introduce `unsafe` code without the user's explicit approval.
- Every approved unsafe block must have a nearby `SAFETY:` comment explaining
  the invariants that make the operation sound and who maintains them.

## Lint suppression

- Do not weaken workspace-wide lint or compiler settings to accommodate one
  local issue.

## Documentation and comments

- First and foremost, comments should document the current code structure and
  usage. Do not explain how things used to work or how they have changed for
  the better.
- Document public APIs whose contract is not obvious from their name and type.
  Include important error conditions, panics, and safety requirements.
- Keep TODOs actionable: state what remains and, when useful, what currently
  prevents it. Do not leave commented-out code in place of version history.

## Tests

- Put unit tests in a separate sibling `<module>_tests.rs` file, registered as
  a child of the implementation module:

  ```rust
  #[cfg(test)]
  #[path = "module_tests.rs"]
  mod module_tests;
  ```

- Add a regression test for every fixed bug when a stable, focused test is
  practical.
- Whenever possible, test through the highest-level API and avoid relying on
  intermediate compiler types.
- Name tests after the behavior or scenario they establish, especially the
  boundary or failure case.

## Tooling

- Format Rust changes with the repository's `rustfmt.toml`.
- Keep code warning-free under the repository's normal checks.
- Run the narrowest relevant tests while iterating, then the appropriate crate
  or workspace checks before handing off a change.
