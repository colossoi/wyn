# Choosing where a test belongs

Choose the smallest test boundary that establishes the required behavior. Keep
small source/error examples in table-driven unit tests with a helper that reports
the failing source. Use source fixtures for broader compiler scenarios, especially
when the CLI, generated artifacts, or backend validation are part of the contract.

## Unit-test time limit

**No individual unit test should take more than three seconds of wall-clock time.**
This applies to the entire test function, including setup and every example,
backend, or optimization mode in a loop—not three seconds per iteration. Measure
execution after compilation, using the normal development test profile; record
concurrency and resource limits when comparing timings.

An over-budget test must be made smaller/faster or moved to an appropriate
integration suite. Preserve the property it checks: a successful build cannot
replace an assertion about ownership, instruction structure, or evaluation order.
For an expensive internal invariant, reduce its setup to the relevant phase or
provide a suitable dedicated integration harness. Do not hide expensive work by
ignoring the test or splitting one slow operation into superficial assertions.

The three-second rule is the placement policy, not an automatically enforced
Cargo timeout. Existing slow unit tests need follow-up; the current suite is not
yet compliant. Resource caps for a whole test executable are a separate safeguard.

## Test locations

| Kind | Location and runner | What belongs here |
| --- | --- | --- |
| Internal unit tests | Inline `#[cfg(test)]` modules and `*_tests.rs` / `tests.rs` modules within a crate; `cargo test -p CRATE --lib` (or its binary tests). | Small algorithm/data-structure contracts, malformed internal representations, and phase-specific assertions. Tiny source rejection collections also belong here. Example: literal-expansion rejection cases in `wyn-core/src/literal_expansion_tests.rs`. Keep the complete test under three seconds. |
| Public-library integration tests | A crate's `tests/*.rs`, using its public API; `cargo test -p CRATE --test NAME`. | Behavior spanning library components without compiler-private API access. This is a conventional destination; it does not imply that every crate already has such a suite. |
| Documentation tests | Executable Rust documentation examples; `cargo test --doc -p CRATE`. | Small public API usage examples that must remain correct. |
| CLI check/diagnostics | `tests/compiler/cases/*.wyn`, driven by `wyn/tests/compiler_fixtures.rs`. | Broader source acceptance, rejection, warnings, or inferred-type probes. Use `command = "check"` when lowering and output are irrelevant. Example: grouped `pattern_bindings.wyn`. |
| Shader build/validation | The same source fixtures, with `command = "build"` and explicit targets. | End-to-end lowering/backend regressions. Example: `entry_lowering_regressions.wyn`. Successful builds require nonempty output, shader validation, and an affirmative artifact-content assertion for every target. |
| Emitted artifacts | Artifact expectations in those fixtures; dedicated Rust integration code only when the data format cannot express the assertion. | Host entry names, required/forbidden generated operations, sidecar files, target-specific output, and path handling. Example: `cli_graphics_paths.wyn`. Successful validation alone does not establish these properties. |
| Packages/filesystems | `tests/module-packages/cases/*/case.toml` plus package trees; `wyn/tests/package_manager_functional.rs` and `local_modules_cli.rs`. | Dependency resolution, imports, filesystem layout, source attribution, and CLI package behavior. Use temporary directories. See [package fixtures](tests/module-packages/README.md). |
| Host execution | `wyn-host-interp/src/interp_tests.rs`, its test backend, and source-driven `wyn-core/src/host_interp_tests.rs`. | Actual host-program values, shapes, control flow, and backend events. Compiler-success assertions are insufficient. There is not yet a general execution mode in the `.wyn` fixture runner. |
| GPU execution | `wyn-host-interp/src/gpu/tests.rs`, `extra/tephra`, and `scripts/test_*_gpu.ps1`. | Shader numerical results, resource usage, or synchronization requiring a GPU. Requires the appropriate adapter/backend; preserve documented opt-in/ignored status and prerequisites. |
| Product/protocol tests | Their owning crates and tools: `wyn-analyzer` (LSP), `wyn-wasm` (compiler API/browser target), `extra/wyn-doc` and `extra/mdbook-wyn-highlight` (documentation processing), `extra/treereduce-wyn` (reducer), both Tree-sitter Rust manifests, and `extra/viz` (validation/runtime CLI). | Test the actual product boundary. A compiler check does not substitute for an LSP response, browser build, preprocessor protocol, grammar result, or renderer behavior. These standalone crates need their own Cargo commands. |
| Shared source assets and sweeps | `testfiles/`, `testfiles/regressions/`, `testfiles/playground/`, package test programs, and `scripts/validate_testfiles.*` / `scripts/run_tests.sh`. | Reusable programs and broad compile/validation or execution sweeps. A source asset alone is not a test: identify its runner and assertions. Shader sweeps need `spirv-val` or the WGSL validator; GPU execution has additional requirements. |

## Fixture and assertion guidance

The [compiler fixture guide](tests/compiler/README.md) defines the leading `--!`
TOML comment block and runner. Run it with:

```sh
cargo test -p wyn --test compiler_fixtures
```

- Combine related positive scenarios into existing `.wyn` files where appropriate.
  Ensure every build scenario is reachable and check its emitted entry.
- Keep independent negative integration inputs separate when the first failure
  would mask the others. Tiny negatives can instead share a unit-test helper that
  invokes the frontend independently for every source, within the time budget.
- Match diagnostic categories and the reason relevant to the regression. Avoid
  internal IDs, debug token/type formatting, and unrelated advice wording. Preserve
  exact error/warning counts, and use `warnings = []` when silence is required.
- `???` holes can expose an expression's inferred type through CLI diagnostics
  (exit 2). Constrain the hole through the expression being tested; do not annotate
  it with the very type whose inference is under test. Printed types cannot replace
  assertions about hidden representation, identity, or ownership facts.
- Keep target and option coverage explicit. Compile without `-O` unless the test
  specifically concerns optimization. Build success is not numerical execution.
- Preserve ignored tests and their reasons during migration. Do not silently
  enable unsupported cases or discard assertions that lack a replacement.

See [Building and Testing](README.md#building-and-testing) for workspace,
standalone-crate, WASM, and shader-validation gates. The CLI fixture validator uses
`spirv-val` when available and Naga as its absence fallback; the standalone SPIR-V
sweep specifically requires `spirv-val`.
