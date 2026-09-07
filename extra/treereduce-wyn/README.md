# Using `treereduce-wyn`

`treereduce-wyn` minimizes a single Wyn source file while repeatedly running a
command that decides whether the candidate still reproduces the bug. It uses
Wyn's Tree-sitter grammar for structural reductions, tries small concrete
expressions, and falls back to `???` where useful.

The reducer does not decide what counts as the same bug. That contract belongs
to the interestingness command supplied on the command line.

## Install

From the repository root:

```console
cargo install --locked --path wyn
cargo install --locked --path extra/treereduce-wyn
```

Both executables are installed into Cargo's binary directory, normally
`~/.cargo/bin`. Ensure that directory is on `PATH`, then verify the install:

```console
wyn --help
treereduce-wyn --version
```

Re-run the two `cargo install` commands after updating the repository. Add
`--force` if Cargo reports that the same package version is already installed.

## The interestingness contract

For every candidate, the reducer launches the command after `--`. The command
must:

1. inspect the candidate;
2. return exit code 0 only when the target bug is still present; and
3. return a nonzero exit code for a successful compile, a different error, a
   timeout, or any other uninteresting result.

Match a stable and distinctive fragment of the diagnostic. Do not match line
numbers, temporary paths, or the whole message because they change as the
source shrinks.

The argument `@@.wyn` is replaced with a unique temporary file containing the
candidate. Keep the `.wyn` suffix: the Wyn CLI uses it to recognize a source
file. If the command contains no `@@` argument, the candidate is sent to its
standard input instead.

Always run the predicate manually on the original source and on a known
negative case before starting a reduction.

## Recommended workflow

Suppose `bug.wyn` fails with the stable text `STABLE_ERROR_SUBSTRING`.

### PowerShell

Create `interesting.ps1`:

```powershell
param(
    [Parameter(Mandatory = $true)]
    [string] $Candidate
)

$lines = & wyn check $Candidate 2>&1
$status = $LASTEXITCODE
$output = $lines | Out-String

if ($status -ne 0 -and $output.Contains("STABLE_ERROR_SUBSTRING")) {
    exit 0
}
exit 1
```

Confirm the original is interesting, then reduce it:

```powershell
powershell.exe -NoProfile -File .\interesting.ps1 .\bug.wyn
$LASTEXITCODE # must be 0

treereduce-wyn `
  -v --stable --stats `
  --source .\bug.wyn `
  --output .\bug.min.wyn `
  --timeout 30 `
  -- powershell.exe -NoProfile -File .\interesting.ps1 @@.wyn
```

### Bash

Create `interesting.sh`:

```bash
#!/usr/bin/env bash
set -u

candidate=$1
if output=$(wyn check "$candidate" 2>&1); then
  status=0
else
  status=$?
fi

if (( status != 0 )) && grep -Fq -- "STABLE_ERROR_SUBSTRING" <<<"$output"; then
  exit 0
fi
exit 1
```

Confirm the original is interesting, then reduce it:

```bash
chmod +x ./interesting.sh
./interesting.sh ./bug.wyn
echo $? # must be 0

treereduce-wyn \
  -v --stable --stats \
  --source ./bug.wyn \
  --output ./bug.min.wyn \
  --timeout 30 \
  -- ./interesting.sh @@.wyn
```

Use `wyn build --fill-holes` instead of `wyn check` when the bug occurs after
type checking or the reduced candidate contains `???`. Give each invocation a
unique output path derived from the candidate path if the build reaches code
generation; `--jobs` may run several checks concurrently. The committed
[`interesting.sh`](./interesting.sh) is a build-failure example.

After reduction, run the predicate once more on `bug.min.wyn` and inspect the
file before replacing or deleting the original.

## Direct command checks

For broad predicates, a separate script is optional. This example preserves
any `wyn check` failure with exit code 1:

```console
treereduce-wyn -v --stable --stats \
  --interesting-exit-code 1 \
  --source bug.wyn --output bug.min.wyn \
  -- wyn check @@.wyn
```

The positive selectors `--interesting-exit-code`, `--interesting-stdout`, and
`--interesting-stderr` are combined with **OR**, while an
`--uninteresting-stdout` or `--uninteresting-stderr` match vetoes the result.
Use a script when the predicate needs **AND** logic, such as a nonzero exit code
and a particular diagnostic. If no selector is supplied, exit code 0 is
interesting by default.

## Useful options

- `--stable` repeats outer passes until no pass reduces the byte size. This is
  the normal choice for a thorough first run.
- `--slow` also runs to a size fixpoint, tries non-optional syntax deletions,
  and lowers the generic minimum reduction to one byte. Try it when `--stable`
  stops too early.
- `--fast` performs one outer pass with a four-byte minimum.
- `-j N` controls parallel checks in the generic reduction pass. The structural
  pass is sequential. Check scripts and their output files must be safe for
  concurrent execution.
- `--comments remove` is the default and strips comments before verification.
  `--comments keep` requires every original comment to survive unchanged.
- `--on-parse-error warn|ignore|error` controls handling of parse errors in the
  original source; the default is `warn`.
- `--timeout SECONDS` limits each interestingness invocation.
- `--temp-dir DIR` chooses where `@@` files are created.
- `--min-reduction BYTES` sets the generic pass's minimum accepted shrinkage.
- `-v`, `-vv`, and `-vvv` progressively increase diagnostic output.
- `--stats` prints starting and final sizes, pass counts, structural attempt
  counts, and elapsed time.
- `--output -` writes the reduced source to standard output. Diagnostic and
  stats output may still be present, so a named output file is usually safer.

Run `treereduce-wyn --help` for the complete option list.

## What it can reduce

The custom structural pass can promote useful children out of `if`, `let`,
calls, tuples, records, matches, unary expressions, and binary expressions. It
can remove elements from comma-separated lists while repairing separators. It
also tries concrete replacements such as `0`, `1`, booleans, unit, and empty
collections before the generic `???` fallback. The predicate/compiler is the
type-compatibility and bug-preservation oracle for every change.

## Limitations and troubleshooting

- Reduction is single-file and syntax-driven. It does not rewrite imported
  modules or use Wyn's resolved dependency graph. Run from a working directory
  where unchanged imports and package files remain available.
- `initial test case is not interesting` means the predicate rejected the
  source after the selected comment policy. Run the exact predicate command by
  hand and inspect its exit code and captured output.
- If Wyn rejects the temporary input as an unknown file type, use `@@.wyn`
  rather than bare `@@`.
- If `???` prevents the target compiler phase from running, use
  `wyn build --fill-holes` in the predicate. Some inferred types cannot be
  default-filled, so those candidates will correctly be rejected.
- If the result remains large, try `--slow`, tighten or relax the diagnostic
  signature as appropriate, or add a structural rule in `src/main.rs`.
- A flaky, stateful, or overly broad predicate produces unreliable minima.
  Eliminate external state and test both positive and negative cases first.
