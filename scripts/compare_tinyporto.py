#!/usr/bin/env python3
"""Compile a pinned Tinyporto snapshot with named Wyn binaries and count SPIR-V.

Example:
  python3 scripts/compare_tinyporto.py --tinyporto-repo ../tinyporto \
    --output-dir /tmp/tinyporto-comparison \
    --compiler baseline=/tmp/wyn-baseline --compiler current=target/debug/wyn

The source and package snapshots are exported from Git, never from working trees.
Each entry's count includes its reachable functions once, including function
headers, parameters, labels, and ends; the module count includes all instructions.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import tarfile


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def snapshot(repo, revision, destination, paths):
    commit = git(repo, "rev-parse", f"{revision}^{{commit}}").decode().strip()
    destination.mkdir(parents=True, exist_ok=True)
    archive = git(repo, "archive", commit, *paths)
    with tarfile.open(fileobj=io.BytesIO(archive)) as entries:
        entries.extractall(destination, filter="data")
    return commit


def counts(path):
    data = path.read_bytes()
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if words[0] != 0x07230203:
        raise ValueError(f"Not little-endian SPIR-V: {path}")
    functions, entries = {}, {}
    current = None
    offset, total = 5, 0
    while offset < len(words):
        size, opcode = words[offset] >> 16, words[offset] & 0xFFFF
        if not size or offset + size > len(words):
            raise ValueError(f"Invalid instruction at word {offset}")
        args = words[offset + 1:offset + size]
        total += 1
        if opcode == 15:  # OpEntryPoint
            name = struct.pack(f"<{len(args) - 2}I", *args[2:]).split(b"\0", 1)[0].decode()
            entries[name] = args[1]
        if opcode == 54:  # OpFunction
            current = args[1]
            functions[current] = {"instructions": 0, "calls": set()}
        if current is not None:
            functions[current]["instructions"] += 1
            if opcode == 57:  # OpFunctionCall
                functions[current]["calls"].add(args[2])
        if opcode == 56:  # OpFunctionEnd
            current = None
        offset += size

    def reachable(root):
        seen, pending = set(), [root]
        while pending:
            function = pending.pop()
            if function not in seen:
                seen.add(function)
                pending.extend(functions[function]["calls"])
        return sum(functions[f]["instructions"] for f in seen)

    return {"module": total, "entries": {name: reachable(root) for name, root in entries.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tinyporto-repo", type=Path, required=True)
    parser.add_argument("--tinyporto-revision", default="32ec4b8")
    parser.add_argument("--wyn-repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--packages-revision", default="0cff30a2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler", action="append", required=True, metavar="LABEL=BINARY")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("--output-dir must not exist; use a fresh directory for each comparison")
    output.mkdir(parents=True)
    source = output / "tinyporto"
    source_commit = snapshot(args.tinyporto_repo, args.tinyporto_revision, source,
                             ["wyn.toml", "wyn", "pkg", "assets/vehicles/fiat-500/scene"])
    packages_commit = snapshot(args.wyn_repo, args.packages_revision, output / "wyn", ["pkg"])
    results = {"tinyporto_commit": source_commit, "packages_commit": packages_commit,
               "flags": ["--graphics", "-O", "--target", "spirv", "--target-double", "rust-wgpu"],
               "compilers": {}}
    for compiler in args.compiler:
        label, binary = compiler.split("=", 1)
        if not label or Path(label).name != label or label in results["compilers"]:
            parser.error(f"Invalid or duplicate compiler label: {label}")
        directory = output / label
        directory.mkdir()
        command = [str(Path(binary).resolve()), "build", str(source / "wyn/main.wyn"),
                   *results["flags"], "-o", str(directory / "main.spv")]
        print(f"Compiling {label}...", flush=True)
        with (directory / "compile.log").open("w") as log:
            subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
        results["compilers"][label] = {
            "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
            **counts(directory / "main.spv"),
        }
        (output / "counts.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({label: results["compilers"][label]}), flush=True)


if __name__ == "__main__":
    main()
