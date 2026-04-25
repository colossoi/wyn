#!/usr/bin/env python3
"""Split SPECIFICATION.md into per-chapter files under docs/src/.

The canonical spec lives at the repo root in SPECIFICATION.md as a
single Markdown file. mdBook expects one file per chapter plus a
SUMMARY.md outline; this script produces both from the canonical
source. Run before `mdbook build` / `mdbook serve`.

Each H2 (`## Title`) becomes one chapter file. The first H2 is the
prefix (introduction) chapter — listed without a bullet in
SUMMARY.md so mdBook renders it before the numbered chapters.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC = REPO / "SPECIFICATION.md"
OUT = REPO / "docs" / "src"


def slug(title: str) -> str:
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "section"


def main() -> None:
    if not SPEC.exists():
        sys.exit(f"missing {SPEC}")
    lines = SPEC.read_text().splitlines()

    h2_idx = [i for i, ln in enumerate(lines) if re.match(r"^## ", ln)]
    if not h2_idx:
        sys.exit("expected SPECIFICATION.md to contain `## ` sections")

    # Build (title, slug, body) for each H2 chunk.
    chapters = []
    seen_slugs: dict[str, int] = {}
    for pos, start in enumerate(h2_idx):
        end = h2_idx[pos + 1] if pos + 1 < len(h2_idx) else len(lines)
        title = lines[start][3:].strip()
        sl = slug(title)
        # Disambiguate any slug collision deterministically.
        if sl in seen_slugs:
            seen_slugs[sl] += 1
            sl = f"{sl}-{seen_slugs[sl]}"
        else:
            seen_slugs[sl] = 1
        body = "\n".join(lines[start:end]).rstrip() + "\n"
        chapters.append((title, sl, body))

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.md"):
        old.unlink()

    summary = ["# Summary", ""]
    for i, (title, sl, body) in enumerate(chapters):
        fname = f"{sl}.md"
        (OUT / fname).write_text(body)
        if i == 0:
            # Prefix chapter — no bullet, mdBook renders before numbered list.
            summary.append(f"[{title}]({fname})")
            summary.append("")
        else:
            summary.append(f"- [{title}]({fname})")
    summary.append("")
    (OUT / "SUMMARY.md").write_text("\n".join(summary))

    rel = OUT.relative_to(REPO)
    print(f"wrote {len(chapters)} chapters → {rel}/")


if __name__ == "__main__":
    main()
