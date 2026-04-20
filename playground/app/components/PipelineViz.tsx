// Compile-pipeline visualization.
//
// Shows every entry point (vertex / fragment / compute) in the compiled
// program as a "kernel" node, plus the module-scope uniforms and storage
// buffers it reads or writes. Intended as a quick overview of the data
// flow through a multi-stage compute pipeline — e.g. a reduce that
// partitions work into a first-pass (map+partial reduce into `partials`)
// and a second-pass (reduce `partials` → `result`).
//
// The interface metadata comes straight from the WGSL compile result
// (`ProgramInterface`). We don't round-trip the WGSL text — the
// structured data is the single source of truth.

import type { EntryBinding, EntryInterface, ProgramInterface, ResourceBinding } from "~/lib/wasm";

interface PipelineVizProps {
  iface: ProgramInterface | null;
}

export function PipelineViz({ iface }: PipelineVizProps) {
  if (!iface) {
    return (
      <div style={placeholderStyle}>
        No pipeline metadata — compile a program to see its stages.
      </div>
    );
  }
  if (iface.entries.length === 0) {
    return (
      <div style={placeholderStyle}>Program has no entry points.</div>
    );
  }
  return (
    <div style={rootStyle}>
      <Legend />
      <div style={entriesListStyle}>
        {iface.entries.map((e) => (
          <EntryCard key={e.name} entry={e} />
        ))}
      </div>
      <ResourceSection title="Uniforms" resources={iface.uniforms} />
      <ResourceSection title="Storage buffers" resources={iface.storage} />
    </div>
  );
}

function Legend() {
  return (
    <div style={legendStyle}>
      <span style={{ ...kindBadge("vertex"), marginRight: 8 }}>vertex</span>
      <span style={{ ...kindBadge("fragment"), marginRight: 8 }}>fragment</span>
      <span style={{ ...kindBadge("compute"), marginRight: 8 }}>compute</span>
    </div>
  );
}

function EntryCard({ entry }: { entry: EntryInterface }) {
  return (
    <div style={entryCardStyle}>
      <div style={entryHeaderStyle}>
        <span style={kindBadge(entry.kind)}>{entry.kind}</span>
        <span style={entryNameStyle}>{entry.name}</span>
        {entry.workgroup_size && (
          <span style={workgroupStyle}>
            @workgroup_size({entry.workgroup_size.join(", ")})
          </span>
        )}
      </div>
      <div style={ioGridStyle}>
        <BindingColumn title="inputs" bindings={entry.inputs} />
        <BindingColumn title="outputs" bindings={entry.outputs} />
      </div>
    </div>
  );
}

function BindingColumn({ title, bindings }: { title: string; bindings: EntryBinding[] }) {
  return (
    <div>
      <div style={columnTitleStyle}>{title}</div>
      {bindings.length === 0 ? (
        <div style={emptyStyle}>—</div>
      ) : (
        <ul style={listStyle}>
          {bindings.map((b, i) => (
            <li key={`${b.name}-${i}`} style={bindingRowStyle}>
              <span style={bindingNameStyle}>{b.name}</span>
              <span style={bindingDecoStyle}>{b.decoration}</span>
              <span style={bindingTypeStyle}>: {b.ty}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ResourceSection({ title, resources }: { title: string; resources: ResourceBinding[] }) {
  if (resources.length === 0) return null;
  return (
    <div style={resourceSectionStyle}>
      <div style={sectionHeaderStyle}>{title}</div>
      <ul style={listStyle}>
        {resources.map((r) => (
          <li key={`${r.set}-${r.binding}-${r.name}`} style={bindingRowStyle}>
            <span style={resourceSlotStyle}>
              @group({r.set}) @binding({r.binding})
            </span>
            <span style={bindingNameStyle}>{r.name}</span>
            <span style={bindingTypeStyle}>: {r.ty}</span>
            {r.access && <span style={accessStyle}>({r.access})</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}

// --- styles (inline to avoid CSS churn) ----------------------------------

const rootStyle: React.CSSProperties = {
  padding: "12px 16px",
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 12,
  color: "#d4d4d4",
  overflowY: "auto",
  height: "100%",
};
const placeholderStyle: React.CSSProperties = {
  padding: "16px",
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 12,
  color: "#888",
};
const legendStyle: React.CSSProperties = {
  marginBottom: 12,
};
const entriesListStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 8,
  marginBottom: 16,
};
const entryCardStyle: React.CSSProperties = {
  border: "1px solid #3c3c3c",
  borderRadius: 4,
  padding: "10px 12px",
  background: "#2a2a2a",
};
const entryHeaderStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  marginBottom: 8,
};
const entryNameStyle: React.CSSProperties = {
  fontWeight: 600,
  color: "#e0e0e0",
};
const workgroupStyle: React.CSSProperties = {
  color: "#888",
  fontSize: 11,
};
const ioGridStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: 12,
};
const columnTitleStyle: React.CSSProperties = {
  color: "#888",
  fontSize: 11,
  textTransform: "uppercase",
  marginBottom: 4,
  letterSpacing: "0.05em",
};
const emptyStyle: React.CSSProperties = {
  color: "#555",
  fontStyle: "italic",
};
const listStyle: React.CSSProperties = {
  listStyle: "none",
  padding: 0,
  margin: 0,
};
const bindingRowStyle: React.CSSProperties = {
  display: "flex",
  gap: 6,
  alignItems: "baseline",
  flexWrap: "wrap",
  padding: "2px 0",
};
const bindingNameStyle: React.CSSProperties = { color: "#9cdcfe" };
const bindingDecoStyle: React.CSSProperties = { color: "#ce9178", fontSize: 11 };
const bindingTypeStyle: React.CSSProperties = { color: "#4ec9b0" };
const resourceSectionStyle: React.CSSProperties = {
  marginTop: 12,
  padding: "8px 12px",
  border: "1px solid #3c3c3c",
  borderRadius: 4,
  background: "#252525",
};
const sectionHeaderStyle: React.CSSProperties = {
  fontWeight: 600,
  color: "#e0e0e0",
  marginBottom: 4,
};
const resourceSlotStyle: React.CSSProperties = {
  color: "#888",
  fontSize: 11,
};
const accessStyle: React.CSSProperties = {
  color: "#888",
  fontSize: 11,
};

function kindBadge(kind: string): React.CSSProperties {
  const colors: Record<string, string> = {
    vertex: "#4ec9b0",
    fragment: "#c586c0",
    compute: "#dcdcaa",
  };
  const bg = colors[kind] || "#888";
  return {
    display: "inline-block",
    padding: "2px 6px",
    borderRadius: 3,
    background: bg,
    color: "#000",
    fontWeight: 600,
    fontSize: 10,
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  };
}
