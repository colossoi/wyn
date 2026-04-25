import { useEffect, useMemo, useRef, useState } from "react";
import type { CompileResultWgsl, ErrorInfo } from "~/lib/wasm";
import {
  availableColorSpaces,
  createRenderPipeline,
  setColorSpace,
  setupContext,
  startRenderLoop,
  type RenderLoop,
  type WebGPUContext,
} from "~/lib/webgpu";
import { IRTree } from "./IRTree";
import { PipelineViz } from "./PipelineViz";

const COLOR_SPACE_LABELS: Record<PredefinedColorSpace, string> = {
  srgb: "sRGB",
  "display-p3": "Display P3",
};

type Tab = "output" | "tlc" | "mir" | "wgsl";

interface PreviewProps {
  result: CompileResultWgsl | null;
  errorInfo: ErrorInfo | null;
  /** Called when the user clicks an error item — jumps the editor cursor. */
  onErrorClick: (location: NonNullable<ErrorInfo["location"]>) => void;
}

export function Preview({ result, errorInfo, onErrorClick }: PreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Store the WebGPU context in state (not a ref) so the pipeline-creation
  // effect re-runs once the async `setupContext` resolves. A ref would race
  // against the first `result.wgsl` update and leave the canvas black on
  // cold start whenever the WASM compile finishes before the GPU adapter.
  const [ctx, setCtx] = useState<WebGPUContext | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("output");
  const [fps, setFps] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState<number>(0);
  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(false);
  pausedRef.current = paused;
  const [pipelineOpen, setPipelineOpen] = useState(true);
  const [resolution, setResolution] = useState<{
    width: number;
    height: number;
  }>({
    width: 640,
    height: 360,
  });
  const [gpuError, setGpuError] = useState<string | null>(null);
  const [colorspace, setColorspace] = useState<PredefinedColorSpace>("srgb");
  // Probed once on mount — the dropdown only offers what the display
  // can actually render. `availableColorSpaces` reads `matchMedia` and
  // is safe to call repeatedly, but useMemo keeps the option list
  // referentially stable for React reconciliation.
  const colorSpaceOptions = useMemo<PredefinedColorSpace[]>(
    () => availableColorSpaces(),
    [],
  );

  // Initialize WebGPU context once.
  useEffect(() => {
    if (!canvasRef.current) return;
    let cancelled = false;
    setupContext(canvasRef.current)
      .then((c) => {
        if (cancelled) return;
        setCtx(c);
      })
      .catch((e) => {
        if (cancelled) return;
        setGpuError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Track container size → drive canvas backing-store dimensions and the
  // reported resolution. iResolution picks up the new size automatically
  // because the render loop reads canvas.width/height each frame.
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = canvas?.parentElement;
    if (!canvas || !container) return;
    const apply = () => {
      const w = Math.max(1, Math.floor(container.clientWidth));
      const h = Math.max(1, Math.floor(container.clientHeight));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        setResolution({ width: w, height: h });
      }
    };
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  // (Re)create pipeline + run shader whenever the WGSL text or the GPU
  // context changes. `ctx` becoming non-null (setupContext's promise
  // resolved) has to trigger a re-run even when the WGSL text is the one
  // from a compile that finished before the adapter was ready.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (
      !ctx ||
      !canvas ||
      !result?.success ||
      !result.wgsl ||
      !result.interface
    )
      return;

    let loop: RenderLoop | null = null;
    setGpuError(null);
    try {
      const res = createRenderPipeline(ctx, result.wgsl, result.interface);
      loop = startRenderLoop(
        ctx,
        canvas,
        res,
        setFps,
        setElapsed,
        () => pausedRef.current,
      );
    } catch (e) {
      setGpuError(e instanceof Error ? e.message : String(e));
    }
    return () => {
      loop?.stop();
    };
  }, [ctx, result?.wgsl]);

  const renderableEntry = result?.interface?.entries.some(
    (e) => e.kind === "vertex" || e.kind === "fragment",
  );

  return (
    <div className="preview-panel">
      <div className="preview-window">
        <div className="panel-header">
          <span>Preview</span>
        </div>
        <div className="canvas-container">
          <canvas ref={canvasRef} id="canvas" />
        </div>
        <div className="canvas-bottom-bar">
          <div className="cbb-section cbb-left">
            <div>{fps !== null ? `${fps.toFixed(1)} FPS` : "-- FPS"}</div>
            <div>{elapsed.toFixed(1)}s</div>
          </div>
          <div className="cbb-section cbb-center">
            <button
              type="button"
              className={`cbb-btn ${paused ? "play" : "pause"}`}
              title={paused ? "Play" : "Pause"}
              onClick={() => setPaused((p) => !p)}
            >
              {paused ? "▶" : "❚❚"}
            </button>
            <button
              type="button"
              className="cbb-btn reset"
              title="Reset (not wired)"
              onClick={() => {}}
            >
              ↺
            </button>
            <button
              type="button"
              className="cbb-btn rec"
              title="Recording not implemented yet"
              disabled
            >
              REC
            </button>
          </div>
          <div className="cbb-section cbb-right">
            <div className="resolution">
              {resolution.width}×{resolution.height}
            </div>
            <button
              type="button"
              className="cbb-btn fullscreen"
              title="Fullscreen"
              onClick={() => {
                const el = canvasRef.current;
                if (!el) return;
                if (document.fullscreenElement === el)
                  document.exitFullscreen();
                else el.requestFullscreen();
              }}
            >
              ⛶
            </button>
          </div>
        </div>
      </div>
      <div className="widget-row">
        <div className="widget">
          <div className="widget-title">Settings</div>
          <div className="settings-grid">
            <div className="settings-row">
              <span>HDR</span>
              <select
                className="ctrl-select"
                value={colorspace}
                disabled={colorSpaceOptions.length < 2}
                onChange={(e) => {
                  const cs = e.target.value as PredefinedColorSpace;
                  setColorspace(cs);
                  if (ctx) setColorSpace(ctx, cs);
                }}
              >
                {colorSpaceOptions.map((cs) => (
                  <option key={cs} value={cs}>
                    {COLOR_SPACE_LABELS[cs]}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
        <div className="widget">
          <div className="widget-title">Entry points</div>
          <div className="entries-list">
            {result?.interface?.entries.length ? (
              result.interface.entries.map((e) => (
                <div className="entry-row" key={e.name}>
                  <span className={`entry-dot ${e.kind}`} />
                  <span>{e.name}</span>
                </div>
              ))
            ) : (
              <div className="widget-empty">—</div>
            )}
          </div>
        </div>
        <div className="widget">
          <div className="widget-title">Stats</div>
          <div className="stats-grid">
            <div className="stats-row">
              <span>FPS</span>
              <span>{fps ?? "—"}</span>
            </div>
            <div className="stats-row">
              <span>Resolution</span>
              <span>
                {resolution.width}×{resolution.height}
              </span>
            </div>
          </div>
        </div>
      </div>
      <div className={`pipeline-widget ${pipelineOpen ? "open" : "closed"}`}>
        <button
          type="button"
          className="pipeline-widget-header"
          onClick={() => setPipelineOpen((o) => !o)}
          aria-expanded={pipelineOpen}
        >
          <span className="chevron">{pipelineOpen ? "▼" : "▶"}</span>
          <span>Pipeline</span>
        </button>
        {pipelineOpen && (
          <div className="pipeline-widget-body">
            <PipelineViz iface={result?.interface ?? null} />
          </div>
        )}
      </div>
      <div className="resize-handle-h" />
      <div className="output-panel">
        <div className="tab-bar">
          {(["output", "tlc", "mir", "wgsl"] as Tab[]).map((t) => (
            <div
              key={t}
              className={`tab ${activeTab === t ? "active" : ""}`}
              onClick={() => setActiveTab(t)}
            >
              {tabLabel(t)}
            </div>
          ))}
        </div>
        <div className="tab-content active">
          {activeTab === "output" && (
            <OutputPane
              result={result}
              errorInfo={errorInfo}
              gpuError={gpuError}
              renderable={!!renderableEntry}
              onErrorClick={onErrorClick}
            />
          )}
          {activeTab === "tlc" && <IRTree nodes={result?.tlc} />}
          {activeTab === "mir" && (
            <pre style={monoPanel}>{result?.mir ?? ""}</pre>
          )}
          {activeTab === "wgsl" && (
            <pre style={monoPanel}>{result?.wgsl ?? ""}</pre>
          )}
        </div>
      </div>
    </div>
  );
}

const monoPanel: React.CSSProperties = {
  padding: "12px 16px",
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 12,
};

function tabLabel(t: Tab): string {
  switch (t) {
    case "output":
      return "Output";
    case "tlc":
      return "TLC";
    case "mir":
      return "MIR";
    case "wgsl":
      return "WGSL";
  }
}

interface OutputPaneProps {
  result: CompileResultWgsl | null;
  errorInfo: ErrorInfo | null;
  gpuError: string | null;
  renderable: boolean;
  onErrorClick: (location: NonNullable<ErrorInfo["location"]>) => void;
}

function OutputPane({
  result,
  errorInfo,
  gpuError,
  renderable,
  onErrorClick,
}: OutputPaneProps) {
  if (errorInfo) {
    return (
      <div id="output" className="error">
        <div
          className="error-item"
          onClick={() => errorInfo.location && onErrorClick(errorInfo.location)}
          style={errorInfo.location ? { cursor: "pointer" } : undefined}
        >
          {errorInfo.location && (
            <div className="error-location">
              Line {errorInfo.location.start_line}, Column{" "}
              {errorInfo.location.start_col}
            </div>
          )}
          <div className="error-message">{errorInfo.message}</div>
        </div>
      </div>
    );
  }
  if (gpuError && renderable) {
    return (
      <div id="output" className="error">
        <div className="error-item">
          <div className="error-message">WebGPU error: {gpuError}</div>
        </div>
      </div>
    );
  }
  if (result?.success) {
    if (!renderable) {
      return (
        <div id="output" className="success">
          Compile successful — compute-only program. See the Pipeline tab for
          stages.
        </div>
      );
    }
    return (
      <div id="output" className="success">
        Compilation successful!
      </div>
    );
  }
  return (
    <div id="output">
      Press "Compile &amp; Run" or Ctrl+Enter to compile your shader.
    </div>
  );
}
