import { useEffect, useImperativeHandle, useRef } from "react";
import { EditorState, StateEffect, StateField } from "@codemirror/state";
import { EditorView, Decoration, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter } from "@codemirror/view";
import type { DecorationSet } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { indentUnit } from "@codemirror/language";
import { searchKeymap } from "@codemirror/search";
import { monokai } from "@uiw/codemirror-theme-monokai";
import { wyn } from "~/lib/wyn-lang";
import type { ErrorLocation } from "~/lib/wasm";

export interface EditorHandle {
  setValue(value: string): void;
  setCursor(pos: { line: number; ch: number }): void;
  focus(): void;
  scrollIntoView(pos: { line: number; ch: number }, margin?: number): void;
  view: EditorView;
}

interface EditorProps {
  initialValue: string;
  errorLocation: ErrorLocation | null;
  onChange: (value: string) => void;
  onCompile: () => void;
  onMount: (handle: EditorHandle) => void;
}

// Convert a {line, ch} position (0-indexed) to an absolute document offset,
// clamped to the document bounds.
function posToOffset(view: EditorView, line: number, ch: number): number {
  const doc = view.state.doc;
  const lineNo = Math.max(1, Math.min(doc.lines, line + 1));
  const l = doc.line(lineNo);
  return Math.min(l.to, l.from + Math.max(0, ch));
}

// Error decorations: a line-background highlight plus a wavy underline on
// the offending range. Driven by a StateEffect so React can push updates
// without recreating the view.
const setErrorEffect = StateEffect.define<ErrorLocation | null>();

const errorField = StateField.define<DecorationSet>({
  create: () => Decoration.none,
  update(deco, tr) {
    deco = deco.map(tr.changes);
    for (const e of tr.effects) {
      if (e.is(setErrorEffect)) {
        const loc = e.value;
        if (!loc) {
          deco = Decoration.none;
          continue;
        }
        const doc = tr.state.doc;
        const startLine = Math.max(1, Math.min(doc.lines, loc.start_line));
        const endLine = Math.max(1, Math.min(doc.lines, loc.end_line));
        const startLineInfo = doc.line(startLine);
        const endLineInfo = doc.line(endLine);
        const from = Math.min(startLineInfo.to, startLineInfo.from + Math.max(0, loc.start_col - 1));
        const to = Math.min(endLineInfo.to, endLineInfo.from + Math.max(0, loc.end_col - 1));
        const builder = [
          Decoration.line({ class: "cm-line-error" }).range(startLineInfo.from),
        ];
        if (to > from) {
          builder.push(
            Decoration.mark({ class: "cm-error-underline" }).range(from, to),
          );
        }
        deco = Decoration.set(builder, /*sort=*/ true);
      }
    }
    return deco;
  },
  provide: (f) => EditorView.decorations.from(f),
});

export function Editor({ initialValue, errorLocation, onChange, onCompile, onMount }: EditorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<EditorView | null>(null);
  const callbacksRef = useRef({ onChange, onCompile, onMount });
  callbacksRef.current = { onChange, onCompile, onMount };

  // Mount the editor once. initialValue is only used at first mount; later
  // content updates happen via handle.setValue.
  useEffect(() => {
    if (!containerRef.current) return;

    const compileCmd = () => {
      callbacksRef.current.onCompile();
      return true;
    };

    const state = EditorState.create({
      doc: initialValue,
      extensions: [
        lineNumbers(),
        highlightActiveLine(),
        highlightActiveLineGutter(),
        history(),
        indentUnit.of("  "),
        EditorState.tabSize.of(2),
        keymap.of([
          { key: "Mod-Enter", run: compileCmd },
          indentWithTab,
          ...defaultKeymap,
          ...historyKeymap,
          ...searchKeymap,
        ]),
        errorField,
        EditorView.updateListener.of((u) => {
          if (u.docChanged) callbacksRef.current.onChange(u.state.doc.toString());
        }),
        // Language must come before the theme so syntax-tag styles can layer
        // on top of the theme's defaults rather than being overwritten.
        wyn(),
        monokai,
      ],
    });

    const view = new EditorView({ state, parent: containerRef.current });
    viewRef.current = view;

    const handle: EditorHandle = {
      view,
      setValue(value: string) {
        view.dispatch({
          changes: { from: 0, to: view.state.doc.length, insert: value },
        });
      },
      setCursor(pos) {
        const offset = posToOffset(view, pos.line, pos.ch);
        view.dispatch({ selection: { anchor: offset } });
      },
      focus() {
        view.focus();
      },
      scrollIntoView(pos, _margin) {
        const offset = posToOffset(view, pos.line, pos.ch);
        view.dispatch({ effects: EditorView.scrollIntoView(offset, { y: "center" }) });
      },
    };

    view.focus();
    callbacksRef.current.onMount(handle);

    return () => {
      view.destroy();
      viewRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push error-location changes into the editor via StateEffect.
  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    view.dispatch({ effects: setErrorEffect.of(errorLocation) });
    if (errorLocation) {
      const offset = posToOffset(view, errorLocation.start_line - 1, 0);
      view.dispatch({ effects: EditorView.scrollIntoView(offset, { y: "center" }) });
    }
  }, [errorLocation]);

  return <div ref={containerRef} className="editor-container" />;
}
