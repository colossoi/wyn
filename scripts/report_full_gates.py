"""Turn full-gate JSONL logs under tmp/ into a standalone HTML report."""

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote


GROUPS = {
    "workspace": (
        ("Formatting", "format"),
        ("Workspace build", "workspace-build"),
        ("Test runner checks", "unit-runner-tests", "integration-runner-tests"),
        ("Unit tests", "unit-tests"),
        ("Doctests", "doctests"),
        ("Clippy", "clippy"),
    ),
    "integrations": (("Integration tests",),),
    "standalone": (("Standalone tools and WASM",),),
    "spirv": (("SPIR-V testfiles",),),
    "wgsl": (("WGSL testfiles",),),
}

STYLES = """
:root { color-scheme: light; font: 16px/1.5 system-ui, sans-serif; color: #152539;
        background: #f0f4f7; }
body { margin: auto; max-width: 66rem; padding: 2rem 1rem 4rem; }
h1 { font-size: clamp(2rem, 5vw, 3.25rem); line-height: 1.1; margin: 0 0 .75rem; }
h2 { margin: 0 0 .5rem; font-size: 1.35rem; }
h3, p { margin: .35rem 0; }
.muted { color: #4d6176; }
.panel, details { background: white; border: 1px solid #cbd7e0; border-radius: .65rem; }
.panel { padding: 1.25rem; margin: 1.5rem 0 2rem; border-top: 4px solid #28617a; }
details { margin: .6rem 0; padding: .75rem 1rem; }
summary { cursor: pointer; font-weight: 600; overflow-wrap: anywhere; }
.status { display: inline-block; border-radius: .3rem; padding: .1rem .5rem;
          font-size: .9rem; font-weight: 700; white-space: nowrap; }
.passed { background: #d8eee4; color: #15573c; }
.failed { background: #f8e0e3; color: #8c2637; }
.incomplete { background: #f4e7c8; color: #654700; }
.table { overflow-x: auto; margin-top: .85rem; }
table { border-collapse: collapse; width: 100%; text-align: left; }
th, td { border-bottom: 1px solid #e3e9ee; padding: .65rem .5rem; vertical-align: top; }
thead { background: #f0f4f7; }
td:nth-child(3) { white-space: nowrap; font-variant-numeric: tabular-nums; }
a { color: #145a80; }
code { font-size: .88em; overflow-wrap: anywhere; }
:focus-visible { outline: 3px solid #28617a; outline-offset: 3px; }
.history > summary { font-size: 1.35rem; }
.matrix th:not(:first-child), .matrix td { text-align: center; min-width: 6.75rem; }
.matrix th small { display: block; font-weight: normal; }
.matrix td { vertical-align: middle; }
.cell-time { white-space: nowrap; font-variant-numeric: tabular-nums; }
.matrix-result { display: inline-grid; grid-template-columns: 1.1rem 9ch;
                 gap: .35rem; align-items: center; }
.matrix-result .cell-time { text-align: right; }
.dot { display: inline-grid; place-items: center; border-radius: 50%;
       width: 1.1rem; height: 1.1rem; color: white; font-size: .75rem; line-height: 1; }
.dot.passed { background: #25754f; }
.dot.failed { background: #a42a3b; }
.dot.incomplete { background: #805b13; }
.legend { font-size: .875rem; display: flex; flex-wrap: wrap; gap: .4rem 1rem; }
"""


@dataclass
class Check:
    name: str
    status: str
    elapsed: str
    detail: str


@dataclass
class Run:
    stamp: float
    title: str
    date: datetime
    status: str
    body: str
    checks: list[Check]


def text(path):
    return path.read_text(encoding="utf-8-sig", errors="replace") if path.is_file() else ""


def link(path, base, label):
    relative = Path(os.path.relpath(path, base)).as_posix()
    return f'<a href="{quote(relative)}">{escape(label)}</a>'


def badge(status):
    return f'<span class="status {status.lower()}">{status}</span>'


def duration(seconds):
    minutes, remainder = divmod(round(seconds, 1), 60)
    return f"{int(minutes)}m {remainder:04.1f}s" if minutes else f"{remainder:.1f}s"


def status_for(rows, complete):
    if any(row["exit_code"] != 0 for row in rows):
        return "Failed"
    return "Passed" if rows and complete else "Incomplete"


def integration_detail(log):
    passed = len(re.findall(r"^\d+\.\d+s ok: ", log, re.MULTILINE))
    first = re.search(r"^\d+\.\d+s FAILED: (.+)$", log, re.MULTILINE)
    if first:
        name = first[1].split("::tests/compiler/cases/")[-1]
        detail = f"Stopped at its first failure after {passed} integration tests passed."
        return f"{detail} {name}"
    if passed:
        return "1 integration test passed." if passed == 1 else f"{passed} integration tests passed."
    return ""


def render_run(directory, base):
    metadata = json.loads(text(directory / "run.json") or "{}")
    paths = [directory / f"{group}.jsonl" for group in GROUPS]
    stamp = max(path.stat().st_mtime for path in paths if path.is_file())
    date = datetime.fromtimestamp(stamp, timezone.utc).astimezone()
    title = metadata.get("title", directory.name)
    checks = []
    table = []
    interrupted_streams = []
    for stream, sections in GROUPS.items():
        records = []
        journal = directory / f"{stream}.jsonl"
        for line in text(journal).splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                interrupted_streams.append(journal)
                break
        by_name = {row["name"]: row for row in records}
        for heading, *names in sections:
            if names:
                rows = [by_name[name] for name in names if name in by_name]
                complete = len(rows) == len(names)
            elif stream == "standalone":
                rows = list(by_name.values())
                complete = len(rows) >= metadata.get("standalone_checks", 21)
            else:
                rows = records[-1:]
                complete = bool(rows) and journal not in interrupted_streams
            status = status_for(rows, complete)
            seconds = duration(sum(row["seconds"] for row in rows)) if rows else "—"
            logs = [(row, directory / row.get("log", f"{row['name']}.log")) for row in rows]
            detail = ""
            if logs and stream == "integrations":
                detail = integration_detail(text(logs[-1][1]))
            elif logs and stream in ("spirv", "wgsl"):
                counts = re.search(r"^Results: (.+)$", text(logs[-1][1]), re.MULTILINE)
                if counts:
                    detail = counts[1]
            if stream == "standalone":
                failed = [row["name"] for row in rows if row["exit_code"] != 0]
                detail = ", ".join(failed)
            checks.append(Check(heading, status, seconds, detail))
            available = [(row, path) for row, path in logs if path.is_file()]
            links = " · ".join(link(path, base, row["name"]) for row, path in available)
            if len(available) > 3:
                links = f"<details><summary>{len(available)} logs</summary>{links}</details>"
            table.append(
                f'<tr><th scope="row">{heading}</th><td>{badge(status)}</td>'
                f'<td>{seconds}</td><td>{escape(detail)} {links}</td></tr>'
            )
    statuses = {check.status for check in checks}
    if interrupted_streams:
        statuses.add("Incomplete")
    overall = "Failed" if "Failed" in statuses else "Incomplete" if "Incomplete" in statuses else "Passed"
    body = (
        '<div class="table"><table><thead><tr><th scope="col">Check</th>'
        '<th scope="col">Result</th><th scope="col">Time</th><th scope="col">Details</th>'
        f"</tr></thead><tbody>{''.join(table)}</tbody></table></div>"
    )
    if interrupted_streams:
        journals = ", ".join(link(path, base, path.name) for path in interrupted_streams)
        body += f"<p>Incomplete journal record in {journals}.</p>"
    if recheck := metadata.get("integration_recheck"):
        path = directory / recheck
        log = text(path)
        passed = len(re.findall(r"^\d+\.\d+s ok: ", log, re.MULTILINE))
        failed = len(re.findall(r"^\d+\.\d+s FAILED: ", log, re.MULTILINE))
        if passed + failed:
            body += (
                f"<p>In a separate integration recheck, {failed} tests failed and {passed} passed. "
                f"{link(path, base, 'Recheck log')}</p>"
            )
    if note := metadata.get("note"):
        body += f'<p class="muted">{escape(note)}</p>'
    return Run(stamp, title, date, overall, body, checks)


def dot(status):
    symbol = {"Passed": "✓", "Failed": "×", "Incomplete": "–"}[status]
    return f'<span class="dot {status.lower()}" aria-hidden="true">{symbol}</span>'


def history_table(runs):
    headers = []
    for index, run in enumerate(runs, start=1):
        label = "Latest" if index == 1 else f"Run {index}"
        clock = run.date.strftime("%I:%M %p").lstrip("0")
        date = f"{run.date:%b} {run.date.day} · {clock}"
        headers.append(
            f'<th scope="col" title="{escape(run.title)}">{label}<small class="muted">{date}</small></th>'
        )
    rows = []
    for checks in zip(*(run.checks for run in runs)):
        cells = []
        for check in checks:
            tooltip = f"{check.status}. {check.elapsed}. {check.detail}".strip()
            cells.append(
                f'<td title="{escape(tooltip)}" aria-label="{escape(tooltip)}">'
                f'<span class="matrix-result">{dot(check.status)}'
                f'<span class="cell-time">{check.elapsed}</span></span></td>'
            )
        rows.append(f'<tr><th scope="row">{checks[0].name}</th>{"".join(cells)}</tr>')
    legend = "".join(f"<span>{dot(status)} {status}</span>" for status in ("Passed", "Failed", "Incomplete"))
    return (
        f'<details open class="history"><summary>Last {len(runs)} runs</summary>'
        f'<p class="legend">{legend}</p><div class="table"><table class="matrix">'
        f'<thead><tr><th scope="col">Suite</th>{"".join(headers)}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></details>'
    )


def render(base):
    candidates = {path.parent for group in GROUPS for path in base.glob(f"*/{group}.jsonl")}
    runs = sorted((render_run(path, base) for path in candidates), key=lambda run: run.stamp, reverse=True)[:5]
    latest = "<p>No runs recorded yet.</p>"
    history = ""
    if runs:
        run = runs[0]
        date = f"{run.date:%b} {run.date.day}, {run.date:%Y at %I:%M %p}".replace(" at 0", " at ")
        latest = (
            f"<p>{badge(run.status)} {escape(run.title)} "
            f'<span class="muted">· {date}</span></p>{run.body}'
        )
        history = history_table(runs)
    return f"""<!DOCTYPE html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wyn full gates</title>
<style>{STYLES}</style>
<main>
<h1>Wyn full gates</h1>
<section class="panel"><h2>Latest run</h2>{latest}</section>
<section>{history}</section>
</main>
</html>
"""


def main():
    base = Path(__file__).resolve().parents[1] / "tmp"
    base.mkdir(exist_ok=True)
    output = base / "full-gates.html"
    html = render(base)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=base, delete=False) as temporary:
        temporary.write(html)
    try:
        os.replace(temporary.name, output)
    finally:
        Path(temporary.name).unlink(missing_ok=True)
    print(output)


if __name__ == "__main__":
    main()
