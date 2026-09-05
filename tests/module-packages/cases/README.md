# Functional-test cases

Each child directory is one relocatable package-manager scenario. A case owns
its root package, every dependency package, and a `case.toml` expectation. The
test runner copies the child directory as a unit before running Wyn. The
runner rejects every dependency source other than a local path.

Begin with the sibling `scaffold/` template. Prefer one behavior per case and
assert stable user-visible output rather than temporary paths or internal IDs.

The initial suite should include:

- one local dependency;
- a local dependency chain and diamond;
- identical relative module paths in distinct packages;
- package-local alias scoping;
- a missing dependency alias;
- direct and indirect import cycles;
- a relative path that escapes its package root;
- locally constructed conflicting major requirements; and
- deterministic results from repeated resolution.
