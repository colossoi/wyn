# viz

GPU shader runner for Wyn-compiled SPIR-V modules.

## Commands

```
viz pipeline <shader.spv>    # Run from the pipeline descriptor JSON
                             # (defaults to the sibling <shader>.json that
                             # `wyn compile` writes; -p overrides).
                             # Graphics descriptor -> interactive window;
                             # compute-only -> headless with --input/--output.
viz run <shader.spv>         # Alias for `pipeline`
viz validate <shader.spv>    # Validate a SPIR-V module (headless naga)
viz info                     # Show GPU device info
viz testpattern              # Render a built-in test pattern
```

## Miner

The Bitcoin miner moved to **tephra** (the Vulkan compute runner) — see
`extra/tephra/README.md` for the `tephra mine` subcommand, genesis-block
verification, and benchmarking.
