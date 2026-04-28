// Modal dialog for submitting a bug / feedback report. Posts the
// current source, an optional canvas screenshot, the chosen category,
// and a free-form comment to /api/reports. Anonymous submissions are
// allowed; the server attaches user info from the session cookie when
// present.

import { useCallback, useEffect, useRef, useState } from "react";
import { useFetcher } from "react-router";

export type ReportCategory = "compiler" | "playground" | "other";

const CATEGORIES: { value: ReportCategory; label: string }[] = [
  { value: "compiler", label: "Compiler issue" },
  { value: "playground", label: "Playground issue" },
  { value: "other", label: "Other" },
];

const MAX_COMMENT_CHARS = 4000;

interface ReportDialogProps {
  open: boolean;
  onClose: () => void;
  /** The current editor source — submitted as-is, not the saved copy. */
  getSource: () => string;
  /** Slug of the saved shader being viewed, if any. */
  slug: string | null;
}

interface SubmitResponse {
  id?: number;
}

export function ReportDialog({ open, onClose, getSource, slug }: ReportDialogProps) {
  const [category, setCategory] = useState<ReportCategory>("compiler");
  const [comment, setComment] = useState<string>("");
  const [screenshot, setScreenshot] = useState<string | null>(null);
  const [includeScreenshot, setIncludeScreenshot] = useState<boolean>(true);
  const [capturing, setCapturing] = useState<boolean>(false);
  const fetcher = useFetcher<SubmitResponse>();
  const submitting = fetcher.state !== "idle";
  const submittedId = fetcher.data?.id ?? null;
  const dialogRef = useRef<HTMLDivElement>(null);

  // Capture a screenshot when the dialog opens. Fail-soft — leaves
  // `screenshot` null if the canvas isn't there or readback fails.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setCapturing(true);
    setScreenshot(null);
    captureCanvas()
      .then((data) => {
        if (!cancelled) setScreenshot(data);
      })
      .finally(() => {
        if (!cancelled) setCapturing(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  // Reset transient state on close so the next open starts fresh,
  // *except* the category which persists across opens — the user
  // typically files several reports of the same flavor in a row.
  useEffect(() => {
    if (!open) {
      setComment("");
      setScreenshot(null);
    }
  }, [open]);

  // Esc to close.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  // Auto-close shortly after a successful submission so the user sees
  // the "Submitted" state but isn't stuck having to click again.
  useEffect(() => {
    if (submittedId == null) return;
    const t = setTimeout(() => onClose(), 1200);
    return () => clearTimeout(t);
  }, [submittedId, onClose]);

  const handleSubmit = useCallback(() => {
    if (submitting) return;
    const body = JSON.stringify({
      category,
      comment: comment.slice(0, MAX_COMMENT_CHARS),
      source: getSource(),
      screenshot: includeScreenshot ? screenshot : null,
      slug,
    });
    fetcher.submit(body, {
      method: "post",
      action: "/api/reports",
      encType: "application/json",
    });
  }, [submitting, category, comment, getSource, includeScreenshot, screenshot, slug, fetcher]);

  if (!open) return null;

  return (
    <div className="report-overlay" onClick={onClose}>
      <div
        className="report-dialog"
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="report-dialog-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="report-header">
          <h2 id="report-dialog-title">Report an issue</h2>
          <button
            type="button"
            className="btn-ghost"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <div className="report-body">
          <label className="report-field">
            <span className="report-label">Category</span>
            <select
              className="report-select"
              value={category}
              onChange={(e) => setCategory(e.target.value as ReportCategory)}
              disabled={submitting || submittedId != null}
            >
              {CATEGORIES.map((c) => (
                <option key={c.value} value={c.value}>
                  {c.label}
                </option>
              ))}
            </select>
          </label>
          <label className="report-field">
            <span className="report-label">
              Comment
              <span className="report-count">
                {comment.length}/{MAX_COMMENT_CHARS}
              </span>
            </span>
            <textarea
              className="report-textarea"
              value={comment}
              onChange={(e) => setComment(e.target.value.slice(0, MAX_COMMENT_CHARS))}
              placeholder="What happened? What did you expect? Reproduction steps?"
              rows={10}
              maxLength={MAX_COMMENT_CHARS}
              disabled={submitting || submittedId != null}
              autoFocus
            />
          </label>
          <div className="report-field">
            <label className="report-screenshot-toggle">
              <input
                type="checkbox"
                checked={includeScreenshot}
                onChange={(e) => setIncludeScreenshot(e.target.checked)}
                disabled={submitting || submittedId != null || !screenshot}
              />
              <span>
                Include screenshot of the visualization
                {capturing && " (capturing…)"}
                {!capturing && !screenshot && " (no canvas to capture)"}
              </span>
            </label>
            {screenshot && includeScreenshot && (
              <img
                className="report-screenshot-preview"
                src={screenshot}
                alt="Visualization screenshot"
              />
            )}
          </div>
          <div className="report-meta">
            Includes the current editor source ({getSource().length.toLocaleString()} chars)
            {slug && ` and the shader slug (${slug})`}.
          </div>
        </div>
        <div className="report-footer">
          {submittedId != null ? (
            <span className="report-success">Submitted — thank you!</span>
          ) : (
            <>
              <button
                type="button"
                className="btn-secondary"
                onClick={onClose}
                disabled={submitting}
              >
                Cancel
              </button>
              <button type="button" onClick={handleSubmit} disabled={submitting}>
                {submitting ? "Submitting…" : "Submit report"}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// Snapshot the visualization canvas into a JPEG data URL. We aim for a
// modestly higher resolution than the save-time thumbnail (which is
// 320x180) so the reporter has a usable image for triage; capped at
// ~640px on the long edge to keep the payload well under the server's
// MAX_SCREENSHOT_BYTES limit.
async function captureCanvas(): Promise<string | null> {
  try {
    const canvas = document.getElementById("canvas");
    if (!(canvas instanceof HTMLCanvasElement)) return null;
    await new Promise<void>((r) => requestAnimationFrame(() => r()));
    const MAX = 640;
    const sw = canvas.width;
    const sh = canvas.height;
    if (sw === 0 || sh === 0) return null;
    const scale = Math.min(1, MAX / Math.max(sw, sh));
    const dw = Math.max(1, Math.round(sw * scale));
    const dh = Math.max(1, Math.round(sh * scale));
    const off = document.createElement("canvas");
    off.width = dw;
    off.height = dh;
    const g = off.getContext("2d");
    if (!g) return null;
    g.fillStyle = "#000";
    g.fillRect(0, 0, dw, dh);
    g.drawImage(canvas, 0, 0, dw, dh);
    return off.toDataURL("image/jpeg", 0.8);
  } catch {
    return null;
  }
}
