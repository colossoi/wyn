Please treat git submodule directories like wyn-web/wyn as read-only.

Follow TESTS.md when adding compiler tests. Source-to-SSA and backend regressions
belong in public integration tests or CLI fixtures. Unit tests should exercise a
small phase or algorithm and finish within three seconds. Run the enforced unit
gate with `python scripts/test_units.py`; never add or broaden a timing exception
without explicit user approval. Renaming or removing existing exceptions is fine.
When changing or renaming a slow test, consider reducing it below three seconds;
when a new test trips the gate, first consider whether it belongs in integration
coverage.

Run integration tests through `python scripts/test_integrations.py`: each test
case has a hard five-minute deadline; timeout is an immediate gate failure and
must terminate its compiler descendants. Direct Cargo commands do not enforce it.
