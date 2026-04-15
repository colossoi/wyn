import { useEffect, useRef, useState } from "react";
import type { CompileResult, ErrorInfo } from "~/lib/wasm";
import { createProgram, setupContext, startRenderLoop, wrapForWebGL2 } from "~/lib/webgl";
import { IRTree } from "./IRTree";

type Tab = "output" | "tlc" | "initial-mir" | "final-mir" | "glsl";

interface PreviewProps {
  result: CompileResult | null;
  errorInfo: ErrorInfo | null;
  /** Called when the user clicks an error item — jumps the editor cursor. */
  onErrorClick: (location: NonNullable<ErrorInfo["location"]>) => void;
}

export function Preview({ result, errorInfo, onErrorClick }: PreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("output");
  const [fps, setFps] = useState<number | null>(null);
  const [glError, setGlError] = useState<string | null>(null);

  // Init WebGL context once.
  useEffect(() => {
    if (!canvasRef.current) return;
    const gl = setupContext(canvasRef.current);
    if (!gl) {
      setGlError("WebGL2 not supported");
      return;
    }
    glRef.current = gl;
  }, []);

  // (Re)compile + run shader whenever the result.glsl changes.
  useEffect(() => {
    const gl = glRef.current;
    const canvas = canvasRef.current;
    if (!gl || !canvas || !result?.success || !result.glsl) return;

    let program: WebGLProgram | null = null;
    let loop: ReturnType<typeof startRenderLoop> | null = null;
    setGlError(null);
    try {
      program = createProgram(gl, wrapForWebGL2(result.glsl));
      loop = startRenderLoop(gl, canvas, program, setFps);
    } catch (e) {
      setGlError(e instanceof Error ? e.message : String(e));
    }

    return () => {
      loop?.stop();
      if (program && gl) gl.deleteProgram(program);
    };
  }, [result?.glsl]);

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
          {(["output", "tlc", "initial-mir", "final-mir", "glsl"] as Tab[]).map((t) => (
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
            <OutputPane result={result} errorInfo={errorInfo} glError={glError} onErrorClick={onErrorClick} />
          )}
          {activeTab === "tlc" && <IRTree nodes={result?.tlc} />}
          {activeTab === "initial-mir" && <IRTree nodes={result?.initial_mir} />}
          {activeTab === "final-mir" && <IRTree nodes={result?.final_mir} />}
          {activeTab === "glsl" && (
            <pre style={{ padding: "12px 16px", fontFamily: "'JetBrains Mono', monospace", fontSize: 12 }}>
              {result?.glsl ?? ""}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}

function tabLabel(t: Tab): string {
  switch (t) {
    case "output": return "Output";
    case "tlc": return "TLC";
    case "initial-mir": return "Initial MIR";
    case "final-mir": return "Final MIR";
    case "glsl": return "GLSL";
  }
}

interface OutputPaneProps {
  result: CompileResult | null;
  errorInfo: ErrorInfo | null;
  glError: string | null;
  onErrorClick: (location: NonNullable<ErrorInfo["location"]>) => void;
}

function OutputPane({ result, errorInfo, glError, onErrorClick }: OutputPaneProps) {
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
  if (glError) {
    return (
      <div id="output" className="error">
        <div className="error-item">
          <div className="error-message">GLSL error: {glError}</div>
        </div>
      </div>
    );
  }
  if (result?.success) {
    return <div id="output" className="success">Compilation successful!</div>;
  }
  return <div id="output">Press "Compile &amp; Run" or Ctrl+Enter to compile your shader.</div>;
}
