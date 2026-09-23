# Native fusion fact import

Fusion facts are inserted through one `EGraph::update` using native constructors,
sets, vectors, and unions. The imported checkpoint retains that graph for fusion
planning. Generated fact text, its parsing, and per-command evaluation are gone.
Static schemas, rules, and schedules remain in egglog. Fact selection is unchanged;
restricting scalar-purity facts to relevant callbacks is separate work.

Native insertion propagates errors through the existing fact sink. Sets and
vectors containing e-class values rebuild after unions; numeric containers do
not. A focused regression checks repeated set writes, union rebuilding, and
preservation of numeric members. Existing import tests inspect native graph facts.

## Performance

Compared with `773ecf04`, with Stage 3 and the earlier timing changes stashed.
Identical frozen tinyporto source and package contents, release compilers, WGSL,
`--graphics --max-warnings 0 --verbose --output-mir`, without `-O`. Three runs per
compiler, interleaved, with this task's builds and tests stopped.

| Work | HEAD | Native import |
| --- | ---: | ---: |
| Whole compile | 4.13 s | 3.77 s |
| TLC import plus fusion | 487.709 ms | 96.503 ms |
| Arithmetic EqSat | 1,215.249 ms | 1,233.439 ms |
| Scheduling | 904.609 ms | 912.288 ms |
| Shader entries | 25 | 25 |

Each row is independently medianed. Import now occurs during TLC conversion,
so TLC import and fusion are combined for the comparison. Native schema loading
takes 2.515 ms and fact analysis/insertion takes 17.111 ms. The combined import
and fusion reduction is 391 ms (80%); total median improves by 360 ms (9%).
Totals were 4.11/4.13/4.19 s at HEAD and 3.76/3.77/4.25 s with native import.
The slower third native sample did not increase import/fusion time (94 ms).

An earlier batch using the compiler produced by `cargo test --release` measured
4.40 s versus 4.00 s, and 498 ms versus 98 ms for import plus fusion. The table
above uses the subsequent ordinary `cargo build --release -p wyn` executable.

Tinyporto's WGSL and MIR are byte-for-byte identical to HEAD. The host program
is identical after normalizing its generated shader filename.

## Validation

- `cargo test --release -p wyn-core -p wyn`: 1,431 passed, 18 ignored.
- Tracked testfiles: 107 SPIR-V passed; 106 WGSL passed, with the existing
  linked-SPIR-V `miner` skip.
- Generated Rust host GPU tests passed for SPIR-V and WGSL on Radeon RX 580
  Vulkan, covering batching, filtering, and filter post-maps. Default backend
  selection crashed before reporting an adapter; the explicit Vulkan run passed.
- Formatting and whitespace checks passed. No egglog rules changed.

Local artifacts are under `tmp/tinyporto-fusion-recheck/`: the reproduction
script is `benchmark-native-import.ps1`; samples are in
`native-import-timings.json`; the earlier batch is saved as
`native-import-test-build-timings.json`. Compiler, testfile, and GPU logs use
the `native-import-` prefix.

Stage 3 and the earlier detailed timers remain in stash
`9d54d78fe1d5288c0bd979b37044564d7ac61f4d`, named
`codex: Stage 3 scalar epilogues and egglog sub-pass timings before native fusion import`.
