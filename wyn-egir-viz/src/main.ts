import initWasm, {
  init_compiler,
  inspect_pass,
} from "./wasm-pkg/wyn_egir_viz_wasm.js";
import appPackage from "../package.json";
import wasmPackage from "./wasm-pkg/package.json";
import "./style.css";

type Side = "before" | "after";
type PassId = "egir::optimize_semantics" | "egir::realize_outputs";

const passInfo: Record<PassId, { before: string; after: string }> = {
  "egir::optimize_semantics": {
    before: "Segmented EGIR",
    after: "Optimized EGIR",
  },
  "egir::realize_outputs": {
    before: "Converted EGIR",
    after: "Output writers recorded",
  },
};

const passStorageKey = "wyn-egir-viz:pass";
const savedPass = localStorage.getItem(passStorageKey);
const initialPass: PassId = savedPass && savedPass in passInfo
  ? savedPass as PassId
  : "egir::optimize_semantics";

interface SourceSpan {
  start_line: number;
  start_col: number;
  end_line: number;
  end_col: number;
}

interface VizError {
  message: string;
  span?: SourceSpan;
}

interface GraphGroup {
  id: string;
  label: string;
  kind: string;
  outputs: GraphOutput[];
}

interface GraphOutput {
  slot: number;
  ty: string;
  routes: GraphOutputRoute[];
}

interface GraphOutputRoute {
  source_block: string;
  source_value: string;
  writers: GraphOutputWriter[];
}

interface GraphOutputWriter {
  kind: "value" | "effect";
  id: string;
}

interface GraphNode {
  id: string;
  group: string;
  label: string;
  category: string;
  variant: string;
  detail: string;
  ty?: string;
  span?: SourceSpan;
  operation?: GraphOperation;
}

interface GraphReference {
  id: string;
  kind: "value" | "view" | "place";
}

interface GraphOperandGroup {
  role: string;
  values: GraphReference[];
}

interface GraphRegion {
  role: string;
  symbol?: string;
  identity: boolean;
  captures: GraphReference[];
  parameter_types: string[];
  result_types: string[];
}

interface GraphResult {
  path: number[];
  ty: string;
  destination: "return_value" | "place" | "bounded_place";
  references: GraphReference[];
}

interface GraphOperation {
  operand_groups: GraphOperandGroup[];
  regions: GraphRegion[];
  results: GraphResult[];
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  kind: string;
}

interface GraphTerminator {
  kind: string;
  values: string[];
  targets: string[];
  target_args: string[][];
}

interface GraphBlock {
  id: string;
  group: string;
  params: string[];
  operations: string[];
  terminator: GraphTerminator;
}

interface GraphSnapshot {
  groups: GraphGroup[];
  nodes: GraphNode[];
  edges: GraphEdge[];
  blocks: GraphBlock[];
}

interface NodeRelation {
  before: string[];
  after: string[];
}

interface InspectResult {
  success: boolean;
  pass: string;
  before?: GraphSnapshot;
  after?: GraphSnapshot;
  relations: NodeRelation[];
  error?: VizError;
}

interface Selection {
  side: Side;
  id: string;
}

interface Names {
  values: Map<string, string>;
  places: Map<string, string>;
  blocks: Map<string, string>;
}

const app = document.querySelector<HTMLDivElement>("#app")!;

app.innerHTML = `
  <main class="app-shell">
    <header class="topbar">
      <div class="brand-block">
        <span class="eyebrow">Wyn compiler</span>
        <h1>EGIR Pass Inspector</h1>
        <span class="version-info" title="Viewer and compiler bundle versions">viz ${appPackage.version} · wasm ${wasmPackage.version}</span>
      </div>
      <label class="pass-block">
        <span>Pass</span>
        <select id="pass-select" aria-label="Compiler pass">
          <option value="egir::optimize_semantics"${initialPass === "egir::optimize_semantics" ? " selected" : ""}>egir::optimize_semantics</option>
          <option value="egir::realize_outputs"${initialPass === "egir::realize_outputs" ? " selected" : ""}>egir::realize_outputs</option>
        </select>
      </label>
      <button class="run-button" id="run-button" type="button" disabled>
        <span>Run pass</span>
        <kbd>Ctrl ↵</kbd>
      </button>
    </header>

    <section class="source-panel" aria-labelledby="source-heading">
      <div class="section-heading">
        <h2 id="source-heading">Source</h2>
        <span class="status" id="status" aria-live="polite">Compiler loading…</span>
      </div>
      <textarea
        id="source-editor"
        spellcheck="false"
        aria-label="Wyn source"
        placeholder="Paste Wyn source here…"
      ></textarea>
      <button class="error-message" id="error-message" type="button" hidden></button>
    </section>

    <div
      class="source-resizer"
      id="source-resizer"
      role="separator"
      aria-label="Resize source editor"
      aria-orientation="horizontal"
      aria-valuemin="120"
      tabindex="0"
    ><span aria-hidden="true"></span></div>

    <section class="comparison" aria-label="EGIR before and after comparison">
      ${listingPane("before", "01", "Before", passInfo[initialPass].before)}
      ${listingPane("after", "02", "After", passInfo[initialPass].after)}
    </section>

    <aside class="node-detail" id="node-detail" aria-live="polite" hidden>
      <div class="detail-heading">
        <div>
          <span id="detail-side"></span>
          <strong id="detail-label"></strong>
          <code id="detail-type"></code>
        </div>
        <button id="close-detail" type="button">Close</button>
      </div>
      <pre id="detail-body"></pre>
    </aside>
  </main>
`;

function listingPane(side: Side, step: string, title: string, subtitle: string): string {
  return `
    <article class="listing-pane" data-side="${side}">
      <div class="pane-heading">
        <div><span class="step">${step}</span><h2>${title}</h2></div>
        <span id="${side}-stage">${subtitle}</span>
      </div>
      <div class="listing-scroll" id="${side}-scroll">
        <div class="listing-empty" id="${side}-empty">
          <span class="empty-code" aria-hidden="true">bb0():<br>&nbsp;&nbsp;…<br>&nbsp;&nbsp;return</span>
          <p>${
            side === "before"
              ? "Run the pass to inspect EGIR as a program listing."
              : "Definitions and uses stay linked across snapshots."
          }</p>
        </div>
        <div class="listing-root" id="${side}-listing" hidden></div>
      </div>
    </article>
  `;
}

const editor = document.querySelector<HTMLTextAreaElement>("#source-editor")!;
const appShell = document.querySelector<HTMLElement>(".app-shell")!;
const topbar = document.querySelector<HTMLElement>(".topbar")!;
const sourcePanel = document.querySelector<HTMLElement>(".source-panel")!;
const sourceResizer = document.querySelector<HTMLElement>("#source-resizer")!;
const passSelect = document.querySelector<HTMLSelectElement>("#pass-select")!;
const runButton = document.querySelector<HTMLButtonElement>("#run-button")!;
const status = document.querySelector<HTMLSpanElement>("#status")!;
const errorMessage = document.querySelector<HTMLButtonElement>("#error-message")!;
const detail = document.querySelector<HTMLElement>("#node-detail")!;
const detailSide = document.querySelector<HTMLSpanElement>("#detail-side")!;
const detailLabel = document.querySelector<HTMLElement>("#detail-label")!;
const detailType = document.querySelector<HTMLElement>("#detail-type")!;
const detailBody = document.querySelector<HTMLElement>("#detail-body")!;
const listings: Record<Side, HTMLElement> = {
  before: document.querySelector<HTMLElement>("#before-listing")!,
  after: document.querySelector<HTMLElement>("#after-listing")!,
};
const scrollers: Record<Side, HTMLElement> = {
  before: document.querySelector<HTMLElement>("#before-scroll")!,
  after: document.querySelector<HTMLElement>("#after-scroll")!,
};
const stageLabels: Record<Side, HTMLElement> = {
  before: document.querySelector<HTMLElement>("#before-stage")!,
  after: document.querySelector<HTMLElement>("#after-stage")!,
};

let result: InspectResult | undefined;
let selection: Selection | undefined;
let wasmReady = false;
let compiling = false;

const sourceHeightKey = "wyn-egir-viz:source-height";
const savedSourceHeight = Number(localStorage.getItem(sourceHeightKey));
if (Number.isFinite(savedSourceHeight) && savedSourceHeight > 0) {
  setSourceHeight(savedSourceHeight, false);
} else {
  updateSourceResizerValue();
}

const savedSource = localStorage.getItem("wyn-egir-viz:source");
if (savedSource !== null) editor.value = savedSource;

editor.addEventListener("input", () => {
  localStorage.setItem("wyn-egir-viz:source", editor.value);
  clearError();
});
editor.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    void runPass();
  }
});
runButton.addEventListener("click", () => void runPass());
passSelect.addEventListener("change", () => {
  const pass = passSelect.value as PassId;
  localStorage.setItem(passStorageKey, pass);
  stageLabels.before.textContent = passInfo[pass].before;
  stageLabels.after.textContent = passInfo[pass].after;
  clearComparison();
  status.textContent = "Ready";
});
errorMessage.addEventListener("click", () => focusError(result?.error?.span));
document.querySelector<HTMLButtonElement>("#close-detail")!.addEventListener("click", clearSelection);

let resizeStartY = 0;
let resizeStartHeight = 0;
sourceResizer.addEventListener("pointerdown", (event) => {
  resizeStartY = event.clientY;
  resizeStartHeight = sourcePanel.getBoundingClientRect().height;
  sourceResizer.setPointerCapture(event.pointerId);
  document.body.classList.add("is-resizing-source");
});
sourceResizer.addEventListener("pointermove", (event) => {
  if (!sourceResizer.hasPointerCapture(event.pointerId)) return;
  setSourceHeight(resizeStartHeight + event.clientY - resizeStartY, false);
});
sourceResizer.addEventListener("pointerup", (event) => {
  if (!sourceResizer.hasPointerCapture(event.pointerId)) return;
  sourceResizer.releasePointerCapture(event.pointerId);
  document.body.classList.remove("is-resizing-source");
  localStorage.setItem(sourceHeightKey, String(Math.round(sourcePanel.getBoundingClientRect().height)));
});
sourceResizer.addEventListener("keydown", (event) => {
  if (event.key !== "ArrowUp" && event.key !== "ArrowDown" && event.key !== "Home" && event.key !== "End") return;
  event.preventDefault();
  const step = event.shiftKey ? 40 : 10;
  const current = sourcePanel.getBoundingClientRect().height;
  const next = event.key === "Home"
    ? 120
    : event.key === "End"
      ? maxSourceHeight()
      : current + (event.key === "ArrowDown" ? step : -step);
  setSourceHeight(next);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && selection) clearSelection();
});

for (const side of ["before", "after"] as const) {
  scrollers[side].addEventListener("click", (event) => {
    const definition = (event.target as Element).closest<HTMLElement>("[data-definition-group]");
    if (definition) {
      alignDefinition(side, definition.dataset.definitionGroup!);
      return;
    }
    const target = (event.target as Element).closest<HTMLElement>("[data-ref-id], [data-node-id]");
    const id = target?.dataset.refId ?? target?.dataset.nodeId;
    if (id) {
      selectNode(side, id);
    } else {
      clearSelection();
    }
  });
  scrollers[side].addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const target = (event.target as Element).closest<HTMLElement>("[data-ref-id], [data-node-id]");
    const id = target?.dataset.refId ?? target?.dataset.nodeId;
    if (id) {
      event.preventDefault();
      selectNode(side, id);
    }
  });
}

window.addEventListener("resize", () => {
  if (appShell.style.getPropertyValue("--source-height")) {
    setSourceHeight(sourcePanel.getBoundingClientRect().height, false);
  } else {
    updateSourceResizerValue();
  }
  requestAnimationFrame(drawDependencyGutters);
});

void initializeCompiler();

async function initializeCompiler(): Promise<void> {
  try {
    await initWasm();
    wasmReady = init_compiler();
    if (!wasmReady) throw new Error("The compiler prelude could not be initialized.");
    status.textContent = "Ready";
    runButton.disabled = false;
  } catch (error) {
    status.textContent = "Compiler unavailable";
    showError({ message: error instanceof Error ? error.message : String(error) });
  }
}

async function runPass(): Promise<void> {
  if (!wasmReady || compiling) return;
  compiling = true;
  runButton.disabled = true;
  passSelect.disabled = true;
  status.textContent = "Running pass…";
  clearError();
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

  try {
    result = inspect_pass(editor.value, passSelect.value) as InspectResult;
    if (!result.success || !result.before || !result.after) {
      showError(result.error ?? { message: "Compilation failed without a diagnostic." });
      status.textContent = "Compile failed";
      return;
    }
    selection = undefined;
    detail.hidden = true;
    renderListing("before", result.before);
    renderListing("after", result.after);
    scrollers.before.scrollTop = 0;
    scrollers.after.scrollTop = 0;
    requestAnimationFrame(drawDependencyGutters);
    status.textContent = "Ready";
  } catch (error) {
    showError({ message: error instanceof Error ? error.message : String(error) });
    status.textContent = "Compile failed";
  } finally {
    compiling = false;
    runButton.disabled = !wasmReady;
    passSelect.disabled = !wasmReady;
  }
}

function clearComparison(): void {
  result = undefined;
  selection = undefined;
  detail.hidden = true;
  for (const side of ["before", "after"] as const) {
    listings[side].hidden = true;
    listings[side].replaceChildren();
    document.querySelector<HTMLElement>(`#${side}-empty`)!.hidden = false;
  }
}

function renderListing(side: Side, snapshot: GraphSnapshot): void {
  const listing = listings[side];
  document.querySelector<HTMLElement>(`#${side}-empty`)!.hidden = true;
  listing.hidden = false;
  listing.innerHTML = snapshot.groups.map((group) => renderBody(snapshot, group)).join("");
}

function renderBody(snapshot: GraphSnapshot, group: GraphGroup): string {
  const groupNodes = snapshot.nodes.filter((node) => node.group === group.id);
  const groupBlocks = snapshot.blocks.filter((block) => block.group === group.id);
  const names = buildNames(groupNodes, groupBlocks);
  const parameters = groupNodes
    .filter(isFunctionParameter)
    .sort((a, b) => parameterIndex(a) - parameterIndex(b));
  const signature = parameters
    .map((node) => typedDefinition(node, definitionName(node, names)))
    .join(`<span class="punct">, </span>`);
  const producedResults = new Set(
    snapshot.edges.filter((edge) => edge.kind === "result").map((edge) => edge.target),
  );
  const valueLines = groupNodes
    .filter((node) => node.category === "value" && node.variant !== "parameter" && !producedResults.has(node.id))
    .sort((a, b) => nameIndex(names.values.get(a.id)).localeCompare(nameIndex(names.values.get(b.id)), undefined, { numeric: true }))
    .map((node) => renderValueLine(snapshot, node, names))
    .join("");
  const placeLines = groupNodes
    .filter((node) => node.category === "place" && node.variant !== "parameter" && !producedResults.has(node.id))
    .sort((a, b) => nameIndex(names.places.get(a.id)).localeCompare(nameIndex(names.places.get(b.id)), undefined, { numeric: true }))
    .map((node) => renderPlaceLine(node, names))
    .join("");
  const blockLines = groupBlocks.map((block) => renderBlock(snapshot, block, names)).join("");
  const outputRoutes = renderOutputRoutes(group.outputs ?? [], names);

  return `
    <section class="ir-body" data-group-id="${escapeHtml(group.id)}">
      <svg class="dependency-gutter" aria-hidden="true"></svg>
      <div class="body-heading">
        <span class="body-kind">${escapeHtml(group.kind)}</span>
        <button class="definition-name" type="button" data-definition-group="${escapeHtml(group.id)}">${escapeHtml(group.label.replace(/^(entry|fn|const)\s+/, ""))}</button>
        <span class="signature"><span class="punct">(</span>${signature}<span class="punct">)</span></span>
        <span class="brace">{</span>
      </div>
      ${outputRoutes}
      ${valueLines ? `<div class="ir-comment">; pure sea</div>${valueLines}` : ""}
      ${placeLines ? `<div class="ir-comment">; places</div>${placeLines}` : ""}
      ${blockLines}
      <div class="body-close">}</div>
    </section>
  `;
}

function isFunctionParameter(node: GraphNode): boolean {
  return (node.category === "value" || node.category === "place")
    && node.variant === "parameter"
    && /^param \d+$/.test(node.label);
}

function parameterIndex(node: GraphNode): number {
  return Number(node.label.match(/\d+$/)?.[0] ?? Number.MAX_SAFE_INTEGER);
}

function typedDefinition(node: GraphNode, name: string): string {
  return `${definitionToken(node.id, name)}<span class="punct">:</span> ${typeToken(node.ty)}`;
}

function definitionName(node: GraphNode, names: Names): string {
  return node.category === "place"
    ? names.places.get(node.id) ?? "&?"
    : names.values.get(node.id) ?? "%?";
}

function buildNames(nodes: GraphNode[], blocks: GraphBlock[]): Names {
  const references = nodes.flatMap((node) => node.operation
    ? [
        ...node.operation.operand_groups.flatMap((group) => group.values),
        ...node.operation.regions.flatMap((region) => region.captures),
        ...(node.operation.results ?? []).flatMap((result) => result.references),
      ]
    : []);
  const viewIds = new Set(references.filter((reference) => reference.kind === "view").map((reference) => reference.id));
  const values = new Map<string, string>();
  nodes
    .filter((node) => node.category === "value")
    .sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }))
    .forEach((node, index) => values.set(node.id, `${viewIds.has(node.id) ? "~" : "%"}${index}`));
  const places = new Map<string, string>();
  [...new Set([
    ...nodes.filter((node) => node.category === "place").map((node) => node.id),
    ...references.filter((reference) => reference.kind === "place").map((reference) => reference.id),
  ])]
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
    .forEach((id, index) => places.set(id, `&${index}`));
  const blockNames = new Map<string, string>();
  blocks.forEach((block, index) => blockNames.set(block.id, `bb${index}`));
  return { values, places, blocks: blockNames };
}

function renderValueLine(snapshot: GraphSnapshot, node: GraphNode, names: Names): string {
  const dependencies = snapshot.edges
    .filter((edge) => edge.target === node.id && edge.kind === "value")
    .map((edge) => edge.source);
  const operands = dependencies.map((id) => refToken(id, names.values.get(id) ?? "?", "value-ref")).join(", ");
  const rhs = node.variant === "parameter"
    ? `<span class="ir-keyword">param</span>`
    : node.variant === "constant"
      ? `<span class="ir-literal">${escapeHtml(node.label)}</span>`
      : `<span class="ir-op">${escapeHtml(node.label)}</span>${operands ? `(${operands})` : ""}`;
  return codeLine(
    node.id,
    `${definitionToken(node.id, names.values.get(node.id) ?? "%?")}<span class="punct">:</span> ${typeToken(node.ty)} <span class="punct">=</span> ${rhs}`,
    `line-${node.variant}`,
  );
}

function renderPlaceLine(node: GraphNode, names: Names): string {
  const fields = (node.operation?.operand_groups ?? []).map((group) =>
    irField(group.role, renderReferenceList(group.values, names)));
  const rhs = `<span class="ir-op">${escapeHtml(node.label)}</span>${fields.length ? variantFields(fields) : ""}`;
  return codeLine(
    node.id,
    `${definitionToken(node.id, names.places.get(node.id) ?? "&?")}<span class="punct">:</span> ${typeToken(node.ty)} <span class="punct">=</span> ${rhs}`,
    `place-line line-${node.variant}`,
  );
}

function renderBlock(snapshot: GraphSnapshot, block: GraphBlock, names: Names): string {
  const params = block.params
    .map((id) => {
      const node = snapshot.nodes.find((candidate) => candidate.id === id);
      return node
        ? typedDefinition(node, names.values.get(id) ?? "%?")
        : definitionToken(id, names.values.get(id) ?? "%?");
    })
    .join(`<span class="punct">, </span>`);
  const header = codeLine(
    block.id,
    `<span class="block-name">${escapeHtml(names.blocks.get(block.id) ?? "bb?")}</span><span class="punct">(</span>${params}<span class="punct">):</span>`,
    "block-label",
  );
  const operations = block.operations
    .map((id) => {
      const node = snapshot.nodes.find((candidate) => candidate.id === id);
      if (!node) return "";
      const inputs = snapshot.edges
        .filter((edge) => edge.target === id && edge.kind === "operand")
        .map((edge) => edge.source);
      const outputs = snapshot.edges
        .filter((edge) => edge.source === id && edge.kind === "result")
        .map((edge) => edge.target);
      const lhs = outputs.length
        ? `${outputs.map((output) => {
            const resultNode = snapshot.nodes.find((candidate) => candidate.id === output);
            return resultNode
              ? typedDefinition(resultNode, definitionName(resultNode, names))
              : definitionToken(output, names.values.get(output) ?? names.places.get(output) ?? "?");
          }).join(", ")} <span class="punct">=</span> `
        : "";
      if (node.operation) {
        return renderStructuredOperation(id, node, lhs, names);
      }
      const args = inputs.map((value) => refToken(value, names.values.get(value) ?? "%?", "value-ref")).join(", ");
      return codeLine(
        id,
        `${lhs}<span class="ir-op op-${escapeHtml(node.variant)}">${escapeHtml(node.label)}</span><span class="punct">(</span>${args}<span class="punct">)</span>`,
        `effect-line line-${node.variant}`,
      );
    })
    .join("");
  return `<div class="basic-block">${header}${operations}${renderTerminator(block, names)}</div>`;
}

function renderStructuredOperation(id: string, node: GraphNode, lhs: string, names: Names): string {
  const operation = node.operation!;
  const fields = node.label === "soac.screma"
    ? renderScremaFields(operation, names)
    : node.label === "soac.filter"
      ? renderFilterFields(operation, names)
      : node.label === "soac.hist"
        ? renderHistFields(operation, names)
        : renderGenericOperationFields(operation, names);
  return codeLine(
    id,
    `${lhs}<span class="ir-op op-${escapeHtml(node.variant)}">${escapeHtml(node.label)}</span><span class="punct">(</span>${fields}${sourceRow(0, `<span class="punct">)</span>`)}`,
    `effect-line structured-effect line-${node.variant}`,
  );
}

function renderScremaFields(operation: GraphOperation, names: Names): string {
  const pre = operation.regions.find((region) => region.role === "pre");
  const post = operation.regions.find((region) => region.role === "post");
  const scans = operation.regions.filter((region) => /^scan\[\d+\]$/.test(region.role));
  const reductions = operation.regions.filter((region) => /^reduce\[\d+\]$/.test(region.role));
  const scanTerms = scans.map((region) => variantTerm("scan", [
    irField("operator", renderRegion(region, names)),
    irField("neutral", renderReferenceList(groupValues(operation, `${region.role}.neutral`), names)),
  ]));
  const reductionTerms = reductions.map((region) => variantTerm("reduce", [
    irField("operator", renderRegion(region, names)),
    irField("neutral", renderReferenceList(groupValues(operation, `${region.role}.neutral`), names)),
  ]));
  return [
    sourceRow(1, comma(irField("inputs", renderReferenceList(groupValues(operation, "inputs"), names)))),
    sourceRow(1, comma(irField("results", renderResultList(operation.results ?? [], names)))),
    sourceRow(1, `${irField("form", `<span class="punct">{</span>`)}`),
    sourceRow(2, comma(irField("pre", pre ? renderRegion(pre, names) : missingTerm()))),
    sourceRow(2, comma(irField("scans", listTerm(scanTerms)))),
    sourceRow(2, comma(irField("reductions", listTerm(reductionTerms)))),
    sourceRow(2, irField("post", post ? renderRegion(post, names) : missingTerm())),
    sourceRow(1, `<span class="punct">}</span>`),
  ].join("");
}

function renderFilterFields(operation: GraphOperation, names: Names): string {
  const map = operation.regions.find((region) => region.role === "map");
  const predicate = operation.regions.find((region) => region.role === "predicate");
  return [
    sourceRow(1, comma(irField("inputs", renderReferenceList(groupValues(operation, "inputs"), names)))),
    sourceRow(1, comma(irField("results", renderResultList(operation.results ?? [], names)))),
    sourceRow(1, irField("body", `<span class="punct">{</span>`)),
    sourceRow(2, comma(irField("map", map ? renderRegion(map, names) : missingTerm()))),
    sourceRow(2, irField("predicate", predicate ? renderRegion(predicate, names) : missingTerm())),
    sourceRow(1, `<span class="punct">}</span>`),
  ].join("");
}

function renderHistFields(operation: GraphOperation, names: Names): string {
  const bucket = operation.regions.find((region) => region.role === "bucket");
  const otherGroups = operation.operand_groups.filter((group) => group.role !== "inputs");
  const reducers = operation.regions.filter((region) => region.role !== "bucket");
  return [
    sourceRow(1, comma(irField("inputs", renderReferenceList(groupValues(operation, "inputs"), names)))),
    sourceRow(1, comma(irField("results", renderResultList(operation.results ?? [], names)))),
    sourceRow(1, irField("form", `<span class="punct">{</span>`)),
    sourceRow(2, comma(irField("bucket", bucket ? renderRegion(bucket, names) : missingTerm()))),
    sourceRow(2, comma(irField("operands", recordTerm(otherGroups.map((group) =>
      irField(group.role, renderReferenceList(group.values, names))))))),
    sourceRow(2, irField("reducers", listTerm(reducers.map((region) =>
      variantTerm("region", [irField("role", literalTerm(region.role)), irField("body", renderRegion(region, names))]))))),
    sourceRow(1, `<span class="punct">}</span>`),
  ].join("");
}

function renderGenericOperationFields(operation: GraphOperation, names: Names): string {
  return joinFields([
    ...operation.operand_groups.map((group) => irField(group.role, renderReferenceList(group.values, names))),
    ...((operation.results ?? []).length
      ? [irField("results", renderResultList(operation.results, names))]
      : []),
    ...(operation.regions.length
      ? [irField("regions", recordTerm(operation.regions.map((region) => irField(region.role, renderRegion(region, names)))))]
      : []),
  ]);
}

function renderResultList(results: GraphResult[], names: Names): string {
  return listTerm(results.map((result) => variantTerm("result", [
    irField("path", listTerm(result.path.map(numberTerm))),
    irField("type", typeToken(result.ty)),
    irField("destination", renderResultDestination(result, names)),
  ])));
}

function renderResultDestination(result: GraphResult, names: Names): string {
  const references = result.references.map((reference) => renderGraphReference(reference, names));
  switch (result.destination) {
    case "return_value":
      return variantTerm("return_value", [irField("value", references[0] ?? missingTerm())]);
    case "place":
      return variantTerm("place", [irField("storage", references[0] ?? missingTerm())]);
    case "bounded_place":
      return variantTerm("bounded_place", [
        irField("storage", references[0] ?? missingTerm()),
        irField("length", references[1] ?? missingTerm()),
      ]);
  }
  return missingTerm();
}

function renderOutputRoutes(outputs: GraphOutput[], names: Names): string {
  const routes = outputs.flatMap((output) => output.routes.map((route) => {
    const source = variantTerm("source", [
      irField("block", refToken(route.source_block, names.blocks.get(route.source_block) ?? "bb?", "block-ref")),
      irField("value", refToken(route.source_value, names.values.get(route.source_value) ?? "%?", "value-ref")),
    ]);
    const writers = listTerm(route.writers.map((writer) => writer.kind === "value"
      ? refToken(writer.id, names.values.get(writer.id) ?? "%?", "value-ref")
      : `<span class="effect-token">${escapeHtml(writer.id)}</span>`));
    return `<div class="ir-metadata-line">${sourceRow(0, `<span class="ir-keyword">output</span> <span class="ir-literal">${output.slot}</span><span class="punct">:</span> ${typeToken(output.ty)} <span class="punct">=</span> <span class="ir-keyword">route</span><span class="punct">(</span>`)}${sourceRow(1, comma(irField("source", source)))}${sourceRow(1, irField("writers", writers))}${sourceRow(0, `<span class="punct">)</span>`)}</div>`;
  }));
  return routes.length ? `<div class="ir-comment">; output routes</div>${routes.join("")}` : "";
}

function groupValues(operation: GraphOperation, role: string): GraphReference[] {
  return operation.operand_groups.find((group) => group.role === role)?.values ?? [];
}

function renderRegion(region: GraphRegion, names: Names): string {
  const body = region.identity
    ? `<span class="ir-keyword">identity</span>`
    : `<span class="ir-symbol">@${escapeHtml(region.symbol ?? "?")}</span>${variantFields([
        irField("captures", renderReferenceList(region.captures, names)),
      ])}`;
  const parameters = region.parameter_types.map((type) => typeToken(type)).join(`<span class="punct">, </span>`);
  const results = region.result_types.map((type) => typeToken(type)).join(`<span class="punct">, </span>`);
  return `${body} <span class="punct">:</span> <span class="punct">(</span>${parameters}<span class="punct">) -&gt; (</span>${results}<span class="punct">)</span>`;
}

function renderReferenceList(references: GraphReference[], names: Names): string {
  return listTerm(references.map((reference) => renderGraphReference(reference, names)));
}

function renderGraphReference(reference: GraphReference, names: Names): string {
  const label = reference.kind === "place"
    ? names.places.get(reference.id) ?? "&?"
    : withReferenceSigil(names.values.get(reference.id) ?? "%?", reference.kind);
  return refToken(reference.id, label, reference.kind === "place" ? "place-ref" : "value-ref");
}

function withReferenceSigil(label: string, kind: GraphReference["kind"]): string {
  const bare = label.replace(/^[%~&]/, "");
  return `${kind === "view" ? "~" : kind === "place" ? "&" : "%"}${bare}`;
}

function irField(name: string, value: string): string {
  return `<span class="ir-field">${escapeHtml(name)}</span><span class="punct">:</span> ${value}`;
}

function joinFields(fields: string[]): string {
  return fields.join(`<span class="punct">, </span>`);
}

function listTerm(items: string[]): string {
  return `<span class="punct">[</span>${items.join(`<span class="punct">, </span>`)}<span class="punct">]</span>`;
}

function recordTerm(fields: string[]): string {
  return `<span class="punct">{ </span>${joinFields(fields)}<span class="punct"> }</span>`;
}

function comma(value: string): string {
  return `${value}<span class="punct">,</span>`;
}

function sourceRow(indent: number, content: string): string {
  return `<span class="ir-source-row ir-indent-${indent}">${content}</span>`;
}

function variantTerm(name: string, fields: string[]): string {
  return `<span class="ir-keyword">${escapeHtml(name)}</span>${variantFields(fields)}`;
}

function variantFields(fields: string[]): string {
  return `<span class="punct">(</span>${joinFields(fields)}<span class="punct">)</span>`;
}

function literalTerm(value: string): string {
  return `<span class="ir-literal">${escapeHtml(JSON.stringify(value))}</span>`;
}

function numberTerm(value: number): string {
  return `<span class="ir-literal">${value}</span>`;
}

function missingTerm(): string {
  return `<span class="ir-error">missing</span>`;
}

function renderTerminator(block: GraphBlock, names: Names): string {
  const term = block.terminator;
  let text: string;
  switch (term.kind) {
    case "return":
      text = `<span class="ir-keyword">return</span>${term.values.length ? ` ${term.values.map((id) => refToken(id, names.values.get(id) ?? "%?", "value-ref")).join(", ")}` : ""}`;
      break;
    case "branch":
      text = `<span class="ir-keyword">br</span> ${targetWithArgs(term.targets[0], term.target_args[0], names)}`;
      break;
    case "cond_branch":
      text = `<span class="ir-keyword">br_if</span> ${refToken(term.values[0], names.values.get(term.values[0]) ?? "%?", "value-ref")}, ${targetWithArgs(term.targets[0], term.target_args[0], names)}, ${targetWithArgs(term.targets[1], term.target_args[1], names)}`;
      break;
    default:
      text = `<span class="ir-keyword">unreachable</span>`;
  }
  return `<div class="code-line terminator-line" data-terminator-for="${escapeHtml(block.id)}"><span class="line-content">${text}</span></div>`;
}

function targetWithArgs(target: string, args: string[], names: Names): string {
  const renderedArgs = (args ?? []).map((id) => refToken(id, names.values.get(id) ?? "%?", "value-ref")).join(", ");
  return `${refToken(target, names.blocks.get(target) ?? "bb?", "block-ref")}<span class="punct">(</span>${renderedArgs}<span class="punct">)</span>`;
}

function codeLine(id: string, content: string, className: string): string {
  return `<div class="code-line ${className}" data-node-id="${escapeHtml(id)}" tabindex="0"><span class="line-content">${content}</span></div>`;
}

function definitionToken(id: string, label: string): string {
  return `<span class="definition" data-ref-id="${escapeHtml(id)}" tabindex="0">${escapeHtml(label)}</span>`;
}

function refToken(id: string | undefined, label: string, className: string): string {
  if (!id) return `<span class="${className}">${escapeHtml(label)}</span>`;
  return `<span class="${className}" data-ref-id="${escapeHtml(id)}" tabindex="0">${escapeHtml(label)}</span>`;
}

function typeToken(type?: string): string {
  return `<span class="ir-type">${escapeHtml(type ?? "_")}</span>`;
}

function drawDependencyGutters(): void {
  if (!result?.before || !result.after) return;
  drawGutter("before", result.before);
  drawGutter("after", result.after);
  updateHighlights();
}

function drawGutter(side: Side, snapshot: GraphSnapshot): void {
  for (const body of listings[side].querySelectorAll<HTMLElement>(".ir-body")) {
    const group = body.dataset.groupId!;
    const overlay = body.querySelector<SVGSVGElement>(".dependency-gutter")!;
    const bodyRect = body.getBoundingClientRect();
    overlay.setAttribute("viewBox", `0 0 ${body.clientWidth} ${body.scrollHeight}`);
    overlay.setAttribute("width", String(body.clientWidth));
    overlay.setAttribute("height", String(body.scrollHeight));
    let lane = 0;
    overlay.innerHTML = snapshot.edges
      .filter((edge) => nodeGroup(snapshot, edge.source) === group && nodeGroup(snapshot, edge.target) === group)
      .filter((edge) => !["block", "sequence"].includes(edge.kind))
      .map((edge) => {
        const source = lineForId(body, edge.source);
        const target = lineForId(body, edge.target);
        if (!source || !target || source === target) return "";
        const sourceRect = source.getBoundingClientRect();
        const targetRect = target.getBoundingClientRect();
        const sourceY = sourceRect.top - bodyRect.top + sourceRect.height / 2;
        const targetY = targetRect.top - bodyRect.top + targetRect.height / 2;
        const x = 18 + (lane++ % 6) * 7;
        const startX = 77;
        const path = `M ${startX} ${sourceY} C ${x} ${sourceY}, ${x} ${targetY}, ${startX} ${targetY}`;
        return `<path class="dependency edge-${escapeHtml(edge.kind)}" data-source="${escapeHtml(edge.source)}" data-target="${escapeHtml(edge.target)}" d="${path}"></path>`;
      })
      .join("");
  }
}

function nodeGroup(snapshot: GraphSnapshot, id: string): string | undefined {
  return snapshot.nodes.find((node) => node.id === id)?.group ?? snapshot.blocks.find((block) => block.id === id)?.group;
}

function lineForId(body: HTMLElement, id: string): HTMLElement | null {
  const definition = body.querySelector<HTMLElement>(`[data-node-id="${cssEscape(id)}"]`);
  if (definition) return definition;
  for (const reference of body.querySelectorAll<HTMLElement>(`[data-ref-id="${cssEscape(id)}"]`)) {
    const line = reference.closest<HTMLElement>(".code-line, .body-heading");
    if (line) return line;
  }
  return body.querySelector<HTMLElement>(`[data-terminator-for="${cssEscape(id)}"]`);
}

function selectNode(side: Side, id: string): void {
  if (selection?.side === side && selection.id === id) {
    clearSelection();
    return;
  }
  selection = { side, id };
  const snapshot = result?.[side];
  const node = snapshot?.nodes.find((candidate) => candidate.id === id);
  const block = snapshot?.blocks.find((candidate) => candidate.id === id);
  if (!node && !block) return;
  detail.hidden = false;
  detailSide.textContent = side;
  detailLabel.textContent = node?.label ?? blockName(snapshot!, block!.id);
  detailType.textContent = node?.ty ?? node?.variant ?? "basic block";
  detailBody.textContent = node?.detail ?? formatBlockDetail(block!);
  updateHighlights();
}

function clearSelection(): void {
  if (!selection && detail.hidden) return;
  selection = undefined;
  detail.hidden = true;
  updateHighlights();
}

function maxSourceHeight(): number {
  return Math.max(120, window.innerHeight - topbar.getBoundingClientRect().height - sourceResizer.offsetHeight - 160);
}

function setSourceHeight(height: number, persist = true): void {
  const clamped = Math.min(Math.max(120, height), maxSourceHeight());
  appShell.style.setProperty("--source-height", `${Math.round(clamped)}px`);
  updateSourceResizerValue();
  if (persist) localStorage.setItem(sourceHeightKey, String(Math.round(clamped)));
}

function updateSourceResizerValue(): void {
  sourceResizer.setAttribute("aria-valuemax", String(Math.round(maxSourceHeight())));
  sourceResizer.setAttribute("aria-valuenow", String(Math.round(sourcePanel.getBoundingClientRect().height)));
}

function blockName(snapshot: GraphSnapshot, id: string): string {
  const block = snapshot.blocks.find((candidate) => candidate.id === id);
  if (!block) return "basic block";
  const index = snapshot.blocks.filter((candidate) => candidate.group === block.group).indexOf(block);
  return `bb${index}`;
}

function formatBlockDetail(block: GraphBlock): string {
  return `parameters: ${block.params.length}\noperations: ${block.operations.length}\nterminator: ${block.terminator.kind}`;
}

function updateHighlights(): void {
  const related = relatedSelection();
  for (const side of ["before", "after"] as const) {
    for (const line of listings[side].querySelectorAll<HTMLElement>("[data-node-id]")) {
      const id = line.dataset.nodeId!;
      line.classList.toggle("is-selected", selection?.side === side && selection.id === id);
      line.classList.toggle("is-related", related[side].has(id));
      line.classList.toggle("is-dimmed", Boolean(selection) && !related[side].has(id));
    }
    for (const token of listings[side].querySelectorAll<HTMLElement>("[data-ref-id]")) {
      const id = token.dataset.refId!;
      token.classList.toggle("is-selected-ref", selection?.side === side && selection.id === id);
      token.classList.toggle("is-related-ref", related[side].has(id));
    }
    for (const path of listings[side].querySelectorAll<SVGPathElement>(".dependency")) {
      const touches = related[side].has(path.dataset.source!) || related[side].has(path.dataset.target!);
      path.classList.toggle("is-related", touches);
      path.classList.toggle("is-dimmed", Boolean(selection) && !touches);
    }
  }
}

function relatedSelection(): Record<Side, Set<string>> {
  const related: Record<Side, Set<string>> = { before: new Set(), after: new Set() };
  const current = selection;
  if (!current || !result) return related;
  related[current.side].add(current.id);
  const other: Side = current.side === "before" ? "after" : "before";
  if (result[other]?.nodes.some((node) => node.id === current.id) || result[other]?.blocks.some((block) => block.id === current.id)) {
    related[other].add(current.id);
  }
  let changed = true;
  while (changed) {
    changed = false;
    for (const relation of result.relations) {
      const touches = relation.before.some((id) => related.before.has(id)) || relation.after.some((id) => related.after.has(id));
      if (!touches) continue;
      for (const id of relation.before) {
        if (!related.before.has(id)) {
          related.before.add(id);
          changed = true;
        }
      }
      for (const id of relation.after) {
        if (!related.after.has(id)) {
          related.after.add(id);
          changed = true;
        }
      }
    }
  }
  return related;
}

function alignDefinition(source: Side, group: string): void {
  const target: Side = source === "before" ? "after" : "before";
  const counterpart = listings[target].querySelector<HTMLElement>(
    `.ir-body[data-group-id="${cssEscape(group)}"]`,
  );
  if (!counterpart) return;
  const scroller = scrollers[target];
  const top = counterpart.getBoundingClientRect().top
    - scroller.getBoundingClientRect().top
    + scroller.scrollTop;
  scroller.scrollTo({ top: Math.max(0, top - 8), behavior: "smooth" });
}

function showError(error: VizError): void {
  errorMessage.hidden = false;
  errorMessage.textContent = error.span ? `${error.message} · ${formatSpan(error.span)}` : error.message;
  errorMessage.classList.toggle("has-location", Boolean(error.span));
  if (error.span) focusError(error.span);
}

function clearError(): void {
  errorMessage.hidden = true;
  errorMessage.textContent = "";
}

function focusError(span?: SourceSpan): void {
  if (!span) return;
  const start = sourceOffset(editor.value, span.start_line, span.start_col);
  const end = sourceOffset(editor.value, span.end_line, span.end_col);
  editor.focus();
  editor.setSelectionRange(start, Math.max(start + 1, end));
}

function sourceOffset(source: string, line: number, column: number): number {
  const lines = source.split("\n");
  const lineIndex = Math.min(Math.max(line - 1, 0), Math.max(0, lines.length - 1));
  let offset = 0;
  for (let index = 0; index < lineIndex; index += 1) offset += lines[index].length + 1;
  return offset + Math.min(Math.max(column - 1, 0), lines[lineIndex]?.length ?? 0);
}

function formatSpan(span: SourceSpan): string {
  return span.start_line === span.end_line
    ? `${span.start_line}:${span.start_col}–${span.end_col}`
    : `${span.start_line}:${span.start_col}–${span.end_line}:${span.end_col}`;
}

function nameIndex(value?: string): string {
  return value ?? "%999999";
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function cssEscape(value: string): string {
  return CSS.escape(value);
}
