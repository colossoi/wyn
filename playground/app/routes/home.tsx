import { useCallback, useEffect, useRef, useState } from "react";
import type { Route } from "./+types/home";
import { Editor } from "~/components/Editor";
import { Preview } from "~/components/Preview";
import { StatusBar, type Status } from "~/components/StatusBar";
import type { CompileResultWgsl, ErrorInfo, WynWasm } from "~/lib/wasm";
import { initWasm } from "~/lib/wasm";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Wyn Playground" },
    { name: "description", content: "Browser-based playground for the Wyn shader language." },
  ];
}

export default function Home() {
  const [wasm, setWasm] = useState<WynWasm | null>(null);
  const [source, setSource] = useState<string>("");
  const [result, setResult] = useState<CompileResultWgsl | null>(null);
  const [errorInfo, setErrorInfo] = useState<ErrorInfo | null>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [statusText, setStatusText] = useState<string>("Loading...");
  const editorRef = useRef<any>(null);
  const sourceRef = useRef<string>("");
  sourceRef.current = source;

  // Bootstrap WASM, load example, auto-compile.
  useEffect(() => {
    let cancelled = false;
    initWasm().then((w) => {
      if (cancelled) return;
      setWasm(w);
      const example = w.get_example_program();
      setSource(example);
      // Push value into the editor too (it has its own internal state).
      if (editorRef.current) editorRef.current.setValue(example);
      compile(w, example);
      setStatus("ready");
      setStatusText("Ready");
    }).catch((e) => {
      setStatus("error");
      setStatusText("Error");
      setErrorInfo({ message: `Initialization error: ${e}`, location: null });
    });
    return () => { cancelled = true; };
  }, []);

  const compile = useCallback((w: WynWasm, src: string) => {
    setStatus("compiling");
    setStatusText("Compiling...");
    try {
      const r = w.compile_to_wgsl(src);
      setResult(r);
      if (!r.success) {
        const err = r.error || { message: "Unknown compilation error", location: null };
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

  const handleErrorClick = useCallback((loc: NonNullable<ErrorInfo["location"]>) => {
    const editor = editorRef.current;
    if (!editor) return;
    const line = loc.start_line - 1;
    const col = loc.start_col - 1;
    editor.setCursor({ line, ch: col });
    editor.focus();
    editor.scrollIntoView({ line, ch: col }, 100);
  }, []);

  return (
    <>
      <header>
        <div className="logo">Wyn Playground</div>
        <div className="toolbar">
          <StatusBar status={status} text={statusText} />
          <button onClick={handleCompile} disabled={!wasm}>Compile &amp; Run</button>
        </div>
      </header>
      <main className="main-content">
        <div className="editor-panel">
          <div className="panel-header"><span>Wyn Source</span></div>
          <Editor
            initialValue=""
            errorLocation={errorInfo?.location ?? null}
            onChange={setSource}
            onCompile={handleCompile}
            onMount={handleEditorMount}
          />
        </div>
        <Preview result={result} errorInfo={errorInfo} onErrorClick={handleErrorClick} />
      </main>
    </>
  );
}
