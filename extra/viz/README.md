# viz

GPU shader runner for Wyn-compiled SPIR-V modules.

## Commands

```
viz vf <shader.spv>          # Render with vertex + fragment shaders
viz compute <shader.spv>     # Run a compute shader (headless)
viz run <shader.spv> -p <desc.json>  # Run from a pipeline descriptor JSON
viz validate <shader.spv>    # Validate a SPIR-V module
viz info                     # Show GPU device info
viz testpattern              # Render a built-in test pattern
```

## Miner

The Bitcoin miner moved to **tephra** (the Vulkan compute runner) — see
`extra/tephra/README.md` for the `tephra mine` subcommand, genesis-block
verification, and benchmarking.
