#!/usr/bin/env -S deno run --allow-read
// Sanity-checks a Wyn pipeline descriptor's frame graph.
//
//   deno run --allow-read scripts/check_descriptor.ts <descriptor.json>
//   deno run scripts/check_descriptor.ts --selftest
//
// Catches the class of bug where a render-target WRITE and its downstream READ
// end up as two separate resources (write keyed by `#[target(name)]`, read
// keyed by the texture param name) — so the producer→consumer edge never forms
// and a derived schedule is wrong. Exits non-zero if any error-level issue is
// found.

type FrameAccess = { resource: number; role?: string };
type FramePass = {
  name: string;
  kind: string;
  reads?: FrameAccess[];
  writes?: FrameAccess[];
  depends_on?: number[];
};
type FrameResource = {
  name: string;
  kind: string;
  bindings?: { pipeline_index: number; name: string; kind: string }[];
};
type FrameGraph = { passes?: FramePass[]; resources?: FrameResource[] };
type Descriptor = { frame_graph?: FrameGraph };

type Issue = { level: "error" | "warn"; msg: string };

// Terminal sinks a host presents rather than another pass reading them.
const TERMINAL_TARGETS = new Set(["screen", "surface", "swapchain"]);

export function checkDescriptor(desc: Descriptor): Issue[] {
  const issues: Issue[] = [];
  const g = desc.frame_graph ?? {};
  const passes = g.passes ?? [];
  const resources = g.resources ?? [];
  const n = resources.length;

  const err = (msg: string) => issues.push({ level: "error", msg });
  const warn = (msg: string) => issues.push({ level: "warn", msg });

  // 1. Structural: every access index and depends_on target is in range.
  passes.forEach((p, pi) => {
    for (const a of [...(p.reads ?? []), ...(p.writes ?? [])]) {
      if (a.resource < 0 || a.resource >= n) {
        err(`pass ${pi} (${p.name}) references resource ${a.resource}, out of [0,${n})`);
      }
    }
    for (const d of p.depends_on ?? []) {
      if (d < 0 || d >= passes.length) err(`pass ${pi} (${p.name}) depends_on ${d}, out of range`);
      if (d === pi) err(`pass ${pi} (${p.name}) depends on itself`);
    }
  });

  // 2. depends_on must be acyclic.
  {
    const WHITE = 0, GRAY = 1, BLACK = 2;
    const color = new Array(passes.length).fill(WHITE);
    const stack: number[] = [];
    const dfs = (u: number): boolean => {
      color[u] = GRAY;
      stack.push(u);
      for (const v of passes[u].depends_on ?? []) {
        if (v < 0 || v >= passes.length) continue;
        if (color[v] === GRAY) {
          err(`cycle in depends_on: ${[...stack, v].join(" -> ")}`);
          return false;
        }
        if (color[v] === WHITE && !dfs(v)) return false;
      }
      stack.pop();
      color[u] = BLACK;
      return true;
    };
    for (let i = 0; i < passes.length; i++) if (color[i] === WHITE) dfs(i);
  }

  // 3. Dependency soundness: if a pass reads a resource that some pass writes,
  //    a writer of it must be in the reader's transitive depends_on. A missing
  //    edge means the schedule can run the reader before its producer.
  const writersOf = new Map<number, number[]>();
  const readersOf = new Map<number, number[]>();
  passes.forEach((p, pi) => {
    for (const a of p.writes ?? []) (writersOf.get(a.resource) ?? writersOf.set(a.resource, []).get(a.resource)!).push(pi);
    for (const a of p.reads ?? []) (readersOf.get(a.resource) ?? readersOf.set(a.resource, []).get(a.resource)!).push(pi);
  });
  const ancestors = (start: number): Set<number> => {
    const seen = new Set<number>();
    const work = [...(passes[start].depends_on ?? [])];
    while (work.length) {
      const u = work.pop()!;
      if (seen.has(u) || u < 0 || u >= passes.length) continue;
      seen.add(u);
      for (const v of passes[u].depends_on ?? []) work.push(v);
    }
    return seen;
  };
  passes.forEach((p, pi) => {
    const anc = ancestors(pi);
    for (const a of p.reads ?? []) {
      const writers = (writersOf.get(a.resource) ?? []).filter((w) => w !== pi);
      if (writers.length > 0 && !writers.some((w) => anc.has(w))) {
        err(
          `pass ${pi} (${p.name}) reads resource ${a.resource} (${resources[a.resource]?.name}) ` +
            `written by pass(es) [${writers.join(",")}] but none are in its depends_on — ` +
            `schedule could run it before its producer`,
        );
      }
    }
  });

  // 4. The name-mismatch signal. A render-target write with no reader, alongside
  //    a read with no writer, is the classic "write keyed by target name, read
  //    keyed by param name" split — the same logical resource under two names.
  const danglingWrites: number[] = [];
  const danglingReads: number[] = [];
  for (let r = 0; r < n; r++) {
    const written = (writersOf.get(r) ?? []).length > 0;
    const read = (readersOf.get(r) ?? []).length > 0;
    const name = resources[r].name;
    if (written && !read && !TERMINAL_TARGETS.has(name)) danglingWrites.push(r);
    if (read && !written) danglingReads.push(r);
  }
  for (const r of danglingWrites) {
    warn(`resource ${r} (${resources[r].name}, ${resources[r].kind}) is written but never read — a dangling producer`);
  }
  for (const r of danglingReads) {
    // A host input legitimately has no in-graph producer; flag only textures,
    // which for a render pipeline usually SHOULD have an in-frame producer.
    const level = resources[r].kind === "texture" ? "error" : "warn";
    issues.push({
      level,
      msg: `resource ${r} (${resources[r].name}, ${resources[r].kind}) is read but never written` +
        (level === "error"
          ? " — a sampled texture with no producer; likely a render target read under a different name than its #[target(...)] write"
          : " — host must supply it"),
    });
  }
  if (danglingWrites.length && danglingReads.length) {
    const w = danglingWrites.map((r) => resources[r].name).join(", ");
    const rd = danglingReads.map((r) => resources[r].name).join(", ");
    warn(`possible name mismatch: writes with no reader {${w}} and reads with no writer {${rd}} may be the same logical resources`);
  }

  return issues;
}

function report(label: string, issues: Issue[]): boolean {
  const errors = issues.filter((i) => i.level === "error");
  if (issues.length === 0) {
    console.log(`✓ ${label}: no issues`);
    return true;
  }
  console.log(`${errors.length ? "✗" : "⚠"} ${label}: ${errors.length} error(s), ${issues.length - errors.length} warning(s)`);
  for (const i of issues) console.log(`  [${i.level}] ${i.msg}`);
  return errors.length === 0;
}

function selftest(): boolean {
  // Fixture A — matched: fragment writes `scene_depth`, compute reads
  // `scene_depth`. One resource, edge forms. Must be clean.
  const matched: Descriptor = {
    frame_graph: {
      resources: [{ name: "scene_depth", kind: "texture", bindings: [{ pipeline_index: 1, name: "scene_depth", kind: "texture" }] }],
      passes: [
        { name: "scene_fragment", kind: "fragment", writes: [{ resource: 0 }] },
        { name: "occ_reduce", kind: "compute", reads: [{ resource: 0 }], depends_on: [0] },
      ],
    },
  };
  // Fixture B — mismatched: write keyed `scene_depth`, read keyed `sd`. Two
  // resources, no edge. Must be flagged (dangling write + unproduced texture read).
  const mismatched: Descriptor = {
    frame_graph: {
      resources: [
        { name: "scene_depth", kind: "texture" },
        { name: "sd", kind: "texture", bindings: [{ pipeline_index: 1, name: "sd", kind: "texture" }] },
      ],
      passes: [
        { name: "scene_fragment", kind: "fragment", writes: [{ resource: 0 }] },
        { name: "occ_reduce", kind: "compute", reads: [{ resource: 1 }] },
      ],
    },
  };
  // Fixture C — missing edge: reader shares the resource but lacks depends_on.
  const missingEdge: Descriptor = {
    frame_graph: {
      resources: [{ name: "scene_depth", kind: "texture" }],
      passes: [
        { name: "scene_fragment", kind: "fragment", writes: [{ resource: 0 }] },
        { name: "occ_reduce", kind: "compute", reads: [{ resource: 0 }] },
      ],
    },
  };

  let ok = true;
  const a = checkDescriptor(matched);
  ok = report("matched (expect clean)", a) && ok;
  ok = (a.length === 0) === true && ok;

  const b = checkDescriptor(mismatched);
  report("mismatched (expect error)", b);
  const bFlagged = b.some((i) => i.level === "error" && /read but never written/.test(i.msg));
  console.log(bFlagged ? "  ✓ mismatch detected" : "  ✗ mismatch NOT detected");
  ok = bFlagged && ok;

  const c = checkDescriptor(missingEdge);
  report("missing-edge (expect error)", c);
  const cFlagged = c.some((i) => i.level === "error" && /before its producer/.test(i.msg));
  console.log(cFlagged ? "  ✓ missing edge detected" : "  ✗ missing edge NOT detected");
  ok = cFlagged && ok;

  return ok;
}

if (import.meta.main) {
  const args = Deno.args;
  if (args[0] === "--selftest") {
    Deno.exit(selftest() ? 0 : 1);
  }
  if (args.length !== 1) {
    console.error("usage: check_descriptor.ts <descriptor.json> | --selftest");
    Deno.exit(2);
  }
  const desc = JSON.parse(Deno.readTextFileSync(args[0])) as Descriptor;
  const ok = report(args[0], checkDescriptor(desc));
  Deno.exit(ok ? 0 : 1);
}
