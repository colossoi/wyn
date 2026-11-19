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

For each full-gate round, keep the logs in a new directory under `tmp/`. Create
an empty `workspace.jsonl` at the start so interrupted rounds are visible. Write
one JSON object per check to `workspace.jsonl`, `integrations.jsonl`,
`standalone.jsonl`, `spirv.jsonl`, or `wgsl.jsonl`. Use the check names in
`scripts/report_full_gates.py` for workspace checks. Include `name`, `seconds`,
and `exit_code`, and save the output as `<name>.log`. After the round ends,
including a failed or interrupted round, run `python scripts/report_full_gates.py`
and link `tmp/full-gates.html` in the report. Keep targeted rechecks out of those
five streams. An optional `run.json` can provide a `title` and the path to a
separate `integration_recheck` log.
