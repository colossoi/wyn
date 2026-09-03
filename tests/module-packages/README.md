# Local package functional-test fixtures

This directory holds complete package trees used by the package-manager
functional tests. Every dependency in these fixtures is a local `path`
dependency. The runner validates that constraint before it executes Wyn.

The Cargo integration test `wyn/tests/package_manager_functional.rs` is the
test binary. It:

1. enumerate cases below `cases/`;
2. validate that every dependency source is local;
3. copy each case to a fresh temporary directory;
4. runs the installed test build of `wyn` through `CARGO_BIN_EXE_wyn`;
5. compare its exit status and diagnostic or graph output with the case
   expectation; and
6. verify that the checked-in fixture tree was unchanged.

The table-driven case runner rejects manifests containing GitHub, URL, or
registry sources before it starts Wyn. A separate functional test synthesizes
a GitHub manifest and a completed unpacked cache entry, then checks the package
through the CLI without making a network request.

`scaffold/` is a template for new cases and is skipped by the runner. Copy it
to `cases/<case-name>/`, edit its package trees, and record the expected result
in `case.toml`.

The convenience entry points are:

```text
scripts/test_local_packages.sh
scripts/test_local_packages.ps1
```

Both invoke:

```text
cargo test -p wyn --test package_manager_functional -- --nocapture
```

Both scripts invoke the Cargo test binary directly.
