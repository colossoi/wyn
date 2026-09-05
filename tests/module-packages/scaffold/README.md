# Functional-test case scaffold

Copy this directory to `../cases/<case-name>/`. The `app` package depends on
the sibling `deps/example` package through a local path. Keep every dependency
inside the copied case tree so the runner can relocate the entire case into a
temporary directory.

`case.toml` describes the package root, command, and expected result. Failure
cases may add stable diagnostic fragments once the runner's expectation schema
is implemented.
