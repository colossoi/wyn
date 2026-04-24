// The interactive editor + preview shell.
//
// Three callsites feed it:
//   - `routes/home.tsx` (logged-in): starts on the built-in example, `slug`
//     is null, `canSave` is true.
//   - `routes/shader.tsx`: starts on the loaded shader source; `slug` is the
//     URL slug; `canSave` is true iff the viewer owns the shader.
//   - Anonymous viewers of `/s/:slug` hit the same `shader.tsx` path with
//     `canSave=false`.
//
// The WASM compiler runs client-only. If `initialSource` is provided it's
// used directly; otherwise the hardcoded example program is loaded.

import { useCallback, useEffect, useRef, useState } from "react";
import { useFetcher, useNavigate } from "react-router";
import { Editor } from "~/components/Editor";
import { Preview } from "~/components/Preview";
import { StatusBar, type Status } from "~/components/StatusBar";
import type { CompileResultWgsl, ErrorInfo, WynWasm } from "~/lib/wasm";
import { initWasm } from "~/lib/wasm";

export interface PlaygroundProps {
  /** If provided, this source populates the editor on mount. Otherwise the
   *  WASM module's `get_example_program()` fallback is used. */
  initialSource?: string | null;
  /** URL slug of the loaded shader, or null for a fresh editor. */
  slug?: string | null;
  /** Whether the current viewer can save changes. */
  canSave: boolean;
  /** Tooltip text when `canSave` is false (e.g., "Sign in to save"). */
  saveDisabledReason?: string;
}

interface SaveResponse {
  slug?: string;
}

export function Playground({
  initialSource = null,
  slug = null,
  canSave,
  saveDisabledReason,
}: PlaygroundProps) {
  const [wasm, setWasm] = useState<WynWasm | null>(null);
  const [source, setSource] = useState<string>("");
  const [result, setResult] = useState<CompileResultWgsl | null>(null);
  const [errorInfo, setErrorInfo] = useState<ErrorInfo | null>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [statusText, setStatusText] = useState<string>("Loading...");
  const editorRef = useRef<any>(null);
  const sourceRef = useRef<string>("");
  sourceRef.current = source;

  const saveFetcher = useFetcher<SaveResponse>();
  const navigate = useNavigate();

  // Bootstrap WASM, load initial source (prop or fallback), auto-compile.
  useEffect(() => {
    let cancelled = false;
    initWasm()
      .then((w) => {
        if (cancelled) return;
        setWasm(w);
        const src = initialSource ?? w.get_example_program();
        setSource(src);
        if (editorRef.current) editorRef.current.setValue(src);
        compile(w, src);
        setStatus("ready");
        setStatusText("Ready");
      })
      .catch((e) => {
        setStatus("error");
        setStatusText("Error");
        setErrorInfo({ message: `Initialization error: ${e}`, location: null });
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const compile = useCallback((w: WynWasm, src: string) => {
    setStatus("compiling");
    setStatusText("Compiling...");
    try {
      const r = w.compile_to_wgsl(src);
      setResult(r);
      if (!r.success) {
        const err = r.error || {
          message: "Unknown compilation error",
          location: null,
        };
        setErrorInfo(err);
        setStatus("error");
        setStatusText("Error");
      } else {
        setErrorInfo(null);
        setStatus("ready");
        setStatusText("Running");
      }
    } catch (e) {
      setErrorInfo({ message: `Compile threw: ${e}`, location: null });
      setStatus("error");
      setStatusText("Error");
    }
  }, []);

  const handleCompile = useCallback(() => {
    if (!wasm) return;
    compile(wasm, sourceRef.current);
  }, [wasm, compile]);

  const handleEditorMount = useCallback((editor: any) => {
    editorRef.current = editor;
    if (sourceRef.current) editor.setValue(sourceRef.current);
  }, []);

  const handleErrorClick = useCallback(
    (loc: NonNullable<ErrorInfo["location"]>) => {
      const editor = editorRef.current;
      if (!editor) return;
      const line = loc.start_line - 1;
      const col = loc.start_col - 1;
      editor.setCursor({ line, ch: col });
      editor.focus();
      editor.scrollIntoView({ line, ch: col }, 100);
    },
    [],
  );

  const saving = saveFetcher.state !== "idle";

  const handleSave = useCallback(() => {
    if (!canSave || saving) return;
    const body = JSON.stringify({ source: sourceRef.current });
    if (slug) {
      saveFetcher.submit(body, {
        method: "put",
        action: `/api/shaders/${slug}`,
        encType: "application/json",
      });
    } else {
      saveFetcher.submit(body, {
        method: "post",
        action: "/api/shaders",
        encType: "application/json",
      });
    }
  }, [canSave, saving, slug, saveFetcher]);

  // When a create succeeds, navigate to the newly-assigned slug URL.
  useEffect(() => {
    const data = saveFetcher.data;
    if (saveFetcher.state === "idle" && data?.slug && !slug) {
      navigate(`/s/${data.slug}`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [saveFetcher.state, saveFetcher.data]);

  return (
    <>
      <main className="main-content">
        <div className="editor-panel">
          <div className="panel-header">
            <span>Wyn Source</span>
            <div className="panel-toolbar">
              <StatusBar status={status} text={statusText} />
              <button
                type="button"
                onClick={handleCompile}
                disabled={!wasm || saving}
              >
                Compile &amp; Run
              </button>
              <button
                type="button"
                className="btn-secondary"
                onClick={handleSave}
                disabled={!canSave || saving}
                title={!canSave ? saveDisabledReason : undefined}
              >
                {saving ? "Saving…" : "Save"}
              </button>
            </div>
          </div>
          <Editor
            initialValue=""
            errorLocation={errorInfo?.location ?? null}
            onChange={setSource}
            onCompile={handleCompile}
            onMount={handleEditorMount}
          />
        </div>
        <Preview
          result={result}
          errorInfo={errorInfo}
          onErrorClick={handleErrorClick}
        />
      </main>
    </>
  );
}
