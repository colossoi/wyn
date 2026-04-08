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
  --header <HEX>         19 comma-separated hex u32 words (default: zeros)
  -n, --nonces <N>       Number of nonces to try [default: 1024]
  --nonce-offset <N>     Starting nonce [default: 0]
  -d, --difficulty <N>   Leading zero bytes required [default: 1]
  --workgroups <N>       Override workgroup count (default: nonces/64)
  -v, --verbose          Print details
```

### Example: verify Bitcoin block 170

Block 170 is the first Bitcoin block containing a real transaction (Satoshi
to Hal Finney, 2009-01-12). Its header hashes to
`00000000d1145790a8694403d4063f323d499e655c83426834d4ce2f8dd4a2ee`
(4 leading zero bytes).

The 76-byte header base (everything except the nonce) as 19 little-endian
u32 words:

```
00000001,0a84bd55,d08a7978,683f85da,183d4f97,dbd12b3e,1f2c846a,2a22cfee,00000000,cb4c10ff,b91a4205,c3f8633e,2e2c5cce,de37bb9d,a3b36427,66815c17,7dac2c56,496ab951,1d00ffff
```

The known nonce is 1889418792 (`0x709e3e28`).

**Verify the known solution** (run 1 nonce at the exact offset):

```bash
viz miner ../../testfiles/miner.spv -n 1 -d 4 \
  --nonce-offset 1889418792 \
  --header 00000001,0a84bd55,d08a7978,683f85da,183d4f97,dbd12b3e,1f2c846a,2a22cfee,00000000,cb4c10ff,b91a4205,c3f8633e,2e2c5cce,de37bb9d,a3b36427,66815c17,7dac2c56,496ab951,1d00ffff
```

Expected output:

```
Mined 1 nonces in ...
1 hit(s) found:
  nonce 1889418792 -> 00000000...
```

**Search for the solution** (scan a range around the known nonce):

```bash
viz miner ../../testfiles/miner.spv -n 65536 -d 4 \
  --nonce-offset 1889400000 \
  --header 00000001,0a84bd55,d08a7978,683f85da,183d4f97,dbd12b3e,1f2c846a,2a22cfee,00000000,cb4c10ff,b91a4205,c3f8633e,2e2c5cce,de37bb9d,a3b36427,66815c17,7dac2c56,496ab951,1d00ffff
```

### Building the miner shader

```bash
bash scripts/build_miner.sh
```

This compiles `testfiles/miner.wyn`, assembles `lib/sha256_compress.spvasm`,
links them with `spirv-link`, and validates the result.
