// POST /api/reports — submit a bug/feedback report.
//
// Body: JSON {
//   category:    "compiler" | "playground" | "other",
//   comment:     string,         // <= MAX_COMMENT_CHARS
//   source:      string,         // <= MAX_SOURCE_BYTES
//   screenshot?: string | null,  // data:image/* URL, <= MAX_SCREENSHOT_BYTES
//   slug?:       string | null,  // optional context: which saved shader
// }
// Responses:
//   200 { id }   — created
//   400          — malformed body / missing fields / bad category
//   403          — cross-origin
//   413          — payload too large
//   415          — not JSON
//
// Anonymous reports are accepted; if a session cookie is present we
// attach the user id + login for follow-up.

import type { Route } from "./+types/api.reports";
import { createReport, type ReportCategory } from "~/lib/db.server";
import { getSession } from "~/lib/session.server";

const MAX_SOURCE_BYTES = 256 * 1024;
const MAX_SCREENSHOT_BYTES = 512 * 1024; // ~384 KiB raw — bigger than thumbnails since we want a usable image
const MAX_COMMENT_CHARS = 4000;
const MAX_USER_AGENT_CHARS = 512;
const VALID_CATEGORIES: ReportCategory[] = ["compiler", "playground", "other"];

function originMatches(request: Request): boolean {
  const origin = request.headers.get("Origin");
  if (!origin) return true;
  return origin === new URL(request.url).origin;
}

export async function action({ request, context }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return new Response("method not allowed", { status: 405 });
  }
  if (!originMatches(request)) {
    return new Response("bad origin", { status: 403 });
  }

  const env = context.cloudflare.env;
  const contentType = request.headers.get("Content-Type") ?? "";
  if (!contentType.includes("application/json")) {
    return new Response("expected JSON body", { status: 415 });
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return new Response("malformed JSON", { status: 400 });
  }

  const categoryRaw = (body as { category?: unknown })?.category;
  if (
    typeof categoryRaw !== "string" ||
    !VALID_CATEGORIES.includes(categoryRaw as ReportCategory)
  ) {
    return new Response("bad category", { status: 400 });
  }
  const category = categoryRaw as ReportCategory;

  const sourceRaw = (body as { source?: unknown })?.source;
  if (typeof sourceRaw !== "string") {
    return new Response("missing source", { status: 400 });
  }
  if (new TextEncoder().encode(sourceRaw).byteLength > MAX_SOURCE_BYTES) {
    return new Response("source too large", { status: 413 });
  }

  const commentRaw = (body as { comment?: unknown })?.comment;
  const comment =
    typeof commentRaw === "string" ? commentRaw.slice(0, MAX_COMMENT_CHARS) : "";

  const screenshotRaw = (body as { screenshot?: unknown })?.screenshot;
  let screenshot: string | null = null;
  if (typeof screenshotRaw === "string" && screenshotRaw.startsWith("data:image/")) {
    if (screenshotRaw.length > MAX_SCREENSHOT_BYTES) {
      return new Response("screenshot too large", { status: 413 });
    }
    screenshot = screenshotRaw;
  }

  const slugRaw = (body as { slug?: unknown })?.slug;
  const shaderSlug =
    typeof slugRaw === "string" && /^[A-Za-z0-9]{1,16}$/.test(slugRaw)
      ? slugRaw
      : null;

  const userAgent =
    request.headers.get("User-Agent")?.slice(0, MAX_USER_AGENT_CHARS) ?? null;

  const session = await getSession(request, env);

  const id = await createReport(env, {
    category,
    comment,
    source: sourceRaw,
    screenshot,
    userId: session?.userId ?? null,
    userLogin: session?.login ?? null,
    shaderSlug,
    userAgent,
  });
  return Response.json({ id });
}
