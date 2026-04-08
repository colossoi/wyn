# viz

GPU shader runner for Wyn-compiled SPIR-V modules.

## Commands

```
viz vf <shader.spv>          # Render with vertex + fragment shaders
viz compute <shader.spv>     # Run a compute shader (headless)
viz run <shader.spv> -p desc # Run from a pipeline descriptor JSON
viz miner [shader.spv]       # Run the Bitcoin miner shader
viz validate <shader.spv>    # Validate a SPIR-V module
viz info                     # Show GPU device info
viz testpattern              # Render a built-in test pattern
```

## Miner

The `miner` command runs `testfiles/miner.spv` (a Bitcoin double-SHA256 compute
shader) and reports nonces whose hashes meet a given difficulty target.

```
viz miner [OPTIONS] [PATH]

Options:
  --header-hex <HEX>     76-byte raw header hex (152 chars, everything except nonce)
  -n, --nonces <N>       Number of nonces to try [default: 1024]
  --nonce-offset <N>     Starting nonce offset (BE u32) [default: 0]
  -d, --difficulty <N>   Leading zero bytes required [default: 1]
  --workgroups <N>       Override workgroup count (default: nonces/64)
  -v, --verbose          Print all hashes
```

### Example: verify Bitcoin block 170

Block 170 is the first Bitcoin block containing a real transaction (Satoshi
to Hal Finney, 2009-01-12). Its header hashes to
`00000000d1145790a8694403d4063f323d499e655c83426834d4ce2f8dd4a2ee`
(4 leading zero bytes).

The raw 76-byte header (everything except the nonce) in hex:

```
0100000055bd840a78798ad0da853f68974f3d183e2bd1db6a842c1feecf222a00000000ff104ccb05421ab93e63f8c3ce5c2c2e9dbb37de2764b3a3175c8166562cac7d51b96a49ffff001d
```

The known nonce bytes are `283e9e70` (BE u32 = 675282544).

**Verify the known solution** (run 1 nonce at the exact offset):

```bash
viz miner ../../testfiles/miner.spv -n 1 -d 4 \
  --nonce-offset 675282544 \
  --header-hex 0100000055bd840a78798ad0da853f68974f3d183e2bd1db6a842c1feecf222a00000000ff104ccb05421ab93e63f8c3ce5c2c2e9dbb37de2764b3a3175c8166562cac7d51b96a49ffff001d
```

Expected output:

```
Mined 1 nonces in ...
1 hit(s) found:
  nonce  675282544 -> 00000000...
```

**Search for the solution** (scan a range around the known nonce):

```bash
viz miner ../../testfiles/miner.spv -n 65536 -d 4 \
  --nonce-offset 675270000 \
  --header-hex 0100000055bd840a78798ad0da853f68974f3d183e2bd1db6a842c1feecf222a00000000ff104ccb05421ab93e63f8c3ce5c2c2e9dbb37de2764b3a3175c8166562cac7d51b96a49ffff001d
```

### Building the miner shader

```bash
bash scripts/build_miner.sh
```

This compiles `testfiles/miner.wyn`, assembles `lib/sha256_compress.spvasm`,
links them with `spirv-link`, and validates the result.
