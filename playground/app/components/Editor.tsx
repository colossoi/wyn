import { useEffect, useRef } from "react";
import type { ErrorLocation } from "~/lib/wasm";

// CodeMirror 5 from CDN, set on window by the script tag injected via root.tsx links.
// Hand-typed minimal interface for the bits we use.
declare global {
  interface Window {
    CodeMirror?: any;
  }
}

interface EditorProps {
  initialValue: string;
  errorLocation: ErrorLocation | null;
  onChange: (value: string) => void;
  onCompile: () => void;
  onMount: (editor: any) => void;
}

export function Editor({ initialValue, errorLocation, onChange, onCompile, onMount }: EditorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<any>(null);
  const callbacksRef = useRef({ onChange, onCompile, onMount });
  callbacksRef.current = { onChange, onCompile, onMount };

  // Mount CodeMirror once.
  useEffect(() => {
    if (!containerRef.current) return;
    const CM = window.CodeMirror;
    if (!CM) {
      console.error("CodeMirror not loaded on window");
      return;
    }
    const editor = CM(containerRef.current, {
      value: initialValue,
      mode: null,
      theme: "monokai",
      lineNumbers: true,
      tabSize: 2,
      indentWithTabs: false,
      lineWrapping: false,
      autofocus: true,
      gutters: ["CodeMirror-linenumbers"],
      scrollbarStyle: "native",
    });
    editor.setOption("extraKeys", {
      "Ctrl-Enter": () => callbacksRef.current.onCompile(),
      "Cmd-Enter": () => callbacksRef.current.onCompile(),
    });
    editor.on("change", (instance: any) => callbacksRef.current.onChange(instance.getValue()));
    editorRef.current = editor;
    callbacksRef.current.onMount(editor);
    return () => {
      // CodeMirror 5 has no destroy; clear container so React unmount is clean.
      if (containerRef.current) containerRef.current.innerHTML = "";
      editorRef.current = null;
    };
  }, []);

  // Mirror errorLocation onto the editor as line highlight + wavy underline.
  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;

    const markers: any[] = [];
    const lineHandles: any[] = [];

    if (errorLocation) {
      const startLine = errorLocation.start_line - 1;
      const startCol = errorLocation.start_col - 1;
      const endLine = errorLocation.end_line - 1;
      const endCol = errorLocation.end_col - 1;
      lineHandles.push(editor.addLineClass(startLine, "background", "line-error"));
      markers.push(editor.markText(
        { line: startLine, ch: startCol },
        { line: endLine, ch: endCol },
        { className: "cm-error-underline" },
      ));
      editor.scrollIntoView({ line: startLine, ch: 0 }, 100);
    }

    return () => {
      for (const m of markers) m.clear();
      for (const h of lineHandles) editor.removeLineClass(h, "background", "line-error");
    };
  }, [errorLocation]);

  return <div ref={containerRef} className="editor-container" />;
}
