import { useEffect, useRef, useState } from "react";
import type { CompileResultWgsl, ErrorInfo } from "~/lib/wasm";
import {
  createRenderPipeline,
  setupContext,
  startRenderLoop,
  type RenderLoop,
  type WebGPUContext,
} from "~/lib/webgpu";
import { IRTree } from "./IRTree";
import { PipelineViz } from "./PipelineViz";

type Tab = "output" | "pipeline" | "tlc" | "mir" | "wgsl";

interface PreviewProps {
  result: CompileResultWgsl | null;
  errorInfo: ErrorInfo | null;
  /** Called when the user clicks an error item — jumps the editor cursor. */
  onErrorClick: (location: NonNullable<ErrorInfo["location"]>) => void;
}

export function Preview({ result, errorInfo, onErrorClick }: PreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<WebGPUContext | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("output");
  const [fps, setFps] = useState<number | null>(null);
  const [gpuError, setGpuError] = useState<string | null>(null);

  // Initialize WebGPU context once.
  useEffect(() => {
    if (!canvasRef.current) return;
    let cancelled = false;
    setupContext(canvasRef.current)
      .then((ctx) => {
        if (cancelled) return;
        ctxRef.current = ctx;
      })
      .catch((e) => {
        if (cancelled) return;
        setGpuError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // (Re)create pipeline + run shader whenever the WGSL text changes.
  useEffect(() => {
    const ctx = ctxRef.current;
    const canvas = canvasRef.current;
    if (!ctx || !canvas || !result?.success || !result.wgsl || !result.interface) return;

    let loop: RenderLoop | null = null;
    setGpuError(null);
    try {
      const res = createRenderPipeline(ctx, result.wgsl, result.interface);
      loop = startRenderLoop(ctx, canvas, res, setFps);
    } catch (e) {
      // Compute-only programs throw "requires @vertex and @fragment" — that's
      // expected; the pipeline viz still shows the stages. Only surface the
      // error in the Output pane (not as a render failure).
      setGpuError(e instanceof Error ? e.message : String(e));
    }
    return () => {
      loop?.stop();
    };
  }, [result?.wgsl]);

  const renderableEntry = result?.interface?.entries.some(
    (e) => e.kind === "vertex" || e.kind === "fragment",
  );

  return (
    <div className="preview-panel">
      <div className="panel-header">
        <span>Preview</span>
      </div>
      <div className="canvas-container">
        <canvas ref={canvasRef} id="canvas" width={640} height={360} />
        <div className="fps">{fps !== null ? `${fps} FPS` : "-- FPS"}</div>
      </div>
      <div className="resize-handle-h" />
      <div className="output-panel">
        <div className="tab-bar">
          {(["output", "pipeline", "tlc", "mir", "wgsl"] as Tab[]).map((t) => (
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
          {activeTab === "pipeline" && <PipelineViz iface={result?.interface ?? null} />}
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
    case "pipeline":
      return "Pipeline";
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
              Line {errorInfo.location.start_line}, Column {errorInfo.location.start_col}
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
          Compile successful — compute-only program. See the Pipeline tab for stages.
        </div>
      );
    }
    return (
      <div id="output" className="success">
        Compilation successful!
      </div>
    );
  }
  return <div id="output">Press "Compile &amp; Run" or Ctrl+Enter to compile your shader.</div>;
}
