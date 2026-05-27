# tephra

Minimal `ash`-based Vulkan runner for Wyn-compiled SPIR-V compute shaders.

## Commands

```
tephra run <shader.spv> [OPTIONS]        # Run a single-buffer compute shader
tephra pipeline <config.json>            # Run a multi-buffer compute pipeline
tephra mine <miner.spv> --header-hex <HEX> [OPTIONS]   # Bitcoin double-SHA256 miner
```

## Miner

The `mine` subcommand drives the Bitcoin double-SHA256 miner — a Wyn `reduce`
over a nonce range, lowered to a two-stage compute pipeline (`*_phase1_chunks`
→ `*_phase2_combine`). Each thread hashes a slice of the nonce range; the
reduce keeps the lowest hash (and its nonce), and a hit is reported when a hash
falls below the header's difficulty target.

```
tephra mine [OPTIONS] --header-hex <HEX> <SHADER>

Arguments:
  <SHADER>  Path to the linked miner SPIR-V module (a sibling `.json` pipeline
            descriptor must sit next to it, e.g. miner.spv + miner.json)

Options:
      --header-hex <HEX>     Raw block header hex (76 bytes = 152 hex chars,
                             everything except the 4-byte nonce)
  -n, --nonces <N>           Number of nonces to try [default: 1024]
      --nonce-offset <N>     Starting nonce offset [default: 0]
      --workgroups <N>       Workgroups per dispatch (each 64 threads)
                             [default: 1024]
  -c, --chunk-size <N>       Max nonces per GPU dispatch (chunks the range to
                             dodge GPU watchdog timeouts) [default: 262144]
  -v, --verbose              Print per-chunk progress
```

### Building the miner shader

```bash
bash scripts/build_miner.sh   # from the repo root
```

This compiles `testfiles/miner.wyn`, assembles `lib/sha256_compress.spvasm`,
links them with `spirv-link`, and validates the result — producing
`testfiles/miner.spv` (and the `testfiles/miner.json` pipeline descriptor the
runner reads alongside it).

### Verify the genesis block

Bitcoin's genesis block (block 0, 2009-01-03) hashes to
`000000000019d6689c085ae165831e93…` under the difficulty-1 target
`00000000ffff0000…`. Its known nonce is `2083236893`. Run a small window at
that offset and the miner reports the hit immediately:

```bash
cd extra/tephra
cargo run --release -- mine ../../testfiles/miner.spv -v \
  --nonce-offset 2083236893 --nonces 64 \
  --header-hex 0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d
```

The 76-byte header above is the genesis header with the nonce stripped:
version `01000000`, 32 zero bytes (no previous block), the merkle root, the
timestamp `29ab5f49` (1231006505), and the bits `ffff001d` (compact target
`0x1d00ffff`).

### Benchmark (~50M nonces)

For a throughput measurement, scan a large nonce range. At the difficulty-1
target a hit is astronomically unlikely in 50M nonces, so the run scans the
whole range and prints the wall-clock hash rate:

```bash
cd extra/tephra
cargo run --release -- mine ../../testfiles/miner.spv \
  --nonce-offset 0 --nonces 50000000 \
  --header-hex 0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d
```

Prints e.g. `Mined 50000000 nonces in 1.85s (27000000 H/s wall clock)`.
Tune `--workgroups` (grid width) and `--chunk-size` (nonces per dispatch) for
your GPU; larger chunks amortize dispatch overhead but risk the OS GPU
watchdog killing a too-long dispatch.
