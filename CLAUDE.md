# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wyn is a minimal compiler for a Futhark-like language that generates SPIR-V code. It's structured as a Rust workspace with two crates: the compiler library and a CLI driver.

## HIGH PRIORITY DIRECTIVE

Wyn DOES NOT support partial function application.  If you find yourself implementing partial function application at any level of the compiler, you're probably doing something wrong.

## Commands

Build the project:
```bash
cargo build
```

Run tests:
```bash
cargo test
```

Compile a Wyn source file:
```bash
cargo run --bin wyn -- compile test.wyn -o test.spv
```

Check a source file without generating output:
```bash
cargo run --bin wyn -- check test.wyn
```

## Architecture

### Compilation Pipeline
1. **Lexer** (`wyn-core/src/lexer.rs`): Tokenizes input using nom combinators
2. **Parser** (`wyn-core/src/parser.rs`): Builds AST from tokens
3. **Type Checker** (`wyn-core/src/types/checker.rs`): Hindley-Milner type inference
4. **TLC** (`wyn-core/src/tlc/`): Typed Lambda Calculus IR, defunctionalization, monomorphization
5. **SSA** (`wyn-core/src/ssa/types.rs`): SSA form with basic blocks and block parameters
6. **Code Generator** (`wyn-core/src/spirv/mod.rs`): Generates SPIR-V from SSA

### Key Design Decisions
- **Error Handling**: Uses thiserror for structured error types
- **Parsing**: nom for combinator-based parsing (primarily for lexer)
- **Type System**: Hindley-Milner inference with polytype crate
- **SPIR-V Generation**: rspirv provides safe SPIR-V building APIs

### Adding New Features
- Language features start in `lexer.rs` (tokens) and `ast.rs` (AST nodes)
- Parser rules go in `parser.rs`
- Type checking logic in `types/checker.rs`
- TLC transformation in `tlc/mod.rs`
- SPIR-V generation in `spirv/mod.rs`
- All new syntax elements should have unit tests

### Test Organization
- **Unit tests live in a separate `<modname>_tests.rs` file**, NOT inline in the
  source file. For `foo.rs`, tests go in `foo_tests.rs` registered as a child
  module of `foo.rs` itself via `#[path]`:
  ```rust
  #[cfg(test)]
  #[path = "foo_tests.rs"]
  mod foo_tests;
  ```
  Placing the declaration inside `foo.rs` (rather than in the parent `mod.rs`)
  makes the test module a child of `foo`, so it sees `foo`'s private items for
  free — no `pub(super)` / `pub(crate)` plumbing. This keeps source files
  focused on implementation and keeps test churn out of source diffs.
- Existing examples: `tlc/defunctionalize.rs`, `tlc/fusion.rs`,
  `egir/from_tlc.rs`, `ssa/builder.rs`.

### Visualizing SPIR-V Output
```bash
cd viz && cargo run vf ../complete_shader_example.spv --vertex vertex_main --fragment fragment_main
```

### Minimizing a Failing Wyn Program (treereduce-wyn)

When a Wyn test file triggers a compiler bug and the source is too
large to debug directly, use **treereduce-wyn** to shrink it to the
smallest program that still reproduces the bug. Source:
`extra/treereduce-wyn/` (binary: `extra/treereduce-wyn/target/release/treereduce-wyn`).

**Workflow:**

1. **Build both the wyn compiler and the reducer** (one-time):
   ```bash
   cargo build --release                                                    # wyn → target/release/wyn
   (cd extra/treereduce-wyn && cargo build --release)                       # reducer
   ```

2. **Write an interestingness script** that exits 0 iff the bug still
   reproduces. It should pattern-match a stable substring of the
   error, not the full message (line numbers and function names drift
   during reduction). Save as e.g. `/tmp/interesting.sh`:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   WYN="${WYN:-./target/release/wyn}"
   candidate="${1:--}"
   if [[ "$candidate" == "-" ]]; then
     tmp=$(mktemp /tmp/tr_wyn_XXXXXX.wyn)
     trap 'rm -f "$tmp"' EXIT
     cat > "$tmp"
     candidate="$tmp"
   fi
   output=$("$WYN" compile --fill-holes "$candidate" -o /dev/null 2>&1 || true)
   grep -q "STABLE_ERROR_SUBSTRING" <<< "$output"
   ```
   `chmod +x /tmp/interesting.sh`. Always test on the original file
   (`bash /tmp/interesting.sh path/to/bug.wyn; echo $?` → 0) before
   launching the reducer.

   Always pass `--fill-holes` in the script; the reducer substitutes
   `???` (type-hole) as the universal polymorphic replacement for
   every candidate rewrite, and without `--fill-holes` the compiler
   exits 2 at the type-check gate before the target bug can fire.

3. **Run the reducer** from the repo root (so the script's default
   `./target/release/wyn` resolves):
   ```bash
   ./extra/treereduce-wyn/target/release/treereduce-wyn \
     -v --stable --stats \
     -s path/to/bug.wyn \
     -o /tmp/bug_min.wyn \
     --on-parse-error ignore \
     -j 4 \
     -- bash /tmp/interesting.sh @@
   ```
   `--stable` iterates passes until a pass makes zero progress (true
   fixpoint). `-j 4` parallelizes. `-v` is important — without it
   treereduce only emits the final stats block, so a running job
   looks stalled. Typical run: 2–10 minutes depending on input size
   and how many rewrites the table allows.

4. **Iterate on the replacement table** if the reducer's floor looks
   too high. The table lives in
   `extra/treereduce-wyn/src/main.rs` and maps tree-sitter node kinds
   to replacement strings. Current vocabulary covers every expression
   kind mapped to `???`, every pattern kind mapped to `_`, and
   top-level `def_declaration` / `binding_declaration` mapped to `""`
   (deletion). If a bug's minimum repro is structurally blocked (e.g.
   the reducer can't delete a load-bearing def because removing it
   breaks type-check), you may need to extend the table — **and
   regenerate the parser** if you change `grammar.js`:
   ```bash
   (cd extra/tree-sitter-wyn && tree-sitter generate)       # rebuild parser.c
   (cd extra/treereduce-wyn && cargo build --release)       # rebuild reducer
   ```

5. **The reducer has known limitations**:
   - It can only *delete* and *substitute*, not *synthesize*. If the
     bug requires a specific expression to remain reachable (e.g.
     `f32.sqrt` for the sqrt-panic demo), any substitution that
     disconnects the call graph from that expression will be rejected
     by interestingness — leaving the call chain intact in the output.
   - It deterministically hits a fixpoint; re-running with the same
     table and interestingness produces the same output. To go
     further: relax the interestingness signature, extend the table,
     or do a manual polish pass.

**See also:**
- `extra/treereduce-wyn/interesting.sh` — the committed sqrt-panic
  script; good template.
- Prior reduction results for the sqrt-panic demo and SoA-tuple
  ArrayWith bug yielded ~2KB repros from ~12KB sources.
- Compile a source file to SPIR-V or GLSL

     Usage: wyn compile [OPTIONS] <FILE>

     Arguments:
       <FILE>
               Input source file

     Options:
       -o, --output <FILE>
               Output file (defaults to input name with .spv or .glsl extension)

       -t, --target <TARGET>
               Target output format

               Possible values:
               - spirv:     SPIR-V binary (default)
               - glsl:      GLSL source code
               - shadertoy: GLSL for Shadertoy (fragment shader only, mainImage entry point)

               [default: spirv]

           --output-annotated <FILE>
               Output annotated source code with block IDs and locations

           --output-tlc <FILE>
               Output typed lambda calculus representation

           --output-mir <FILE>
               Output MIR (SSA post-EGIR, pre-backend-lowering)

       -v, --verbose
               Print verbose output

       -h, --help
               Print help (see a summary with '-h')
