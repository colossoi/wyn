// PUT /api/shaders/:slug — update an existing shader. Owner-only.
//
// Body: JSON { source: string, thumbnail?: string | null, title?: string | null }
// Responses:
//   200 { slug }  — updated
//   401           — not logged in
//   403           — logged in but not the owner (or cross-origin)
//   404           — slug doesn't exist
//   413           — source too large
//   415           — not JSON
//   400           — malformed body / missing source

import type { Route } from "./+types/api.shader";
import { getShader, updateShaderSource } from "~/lib/db.server";
import { getSession } from "~/lib/session.server";

const MAX_SOURCE_BYTES = 256 * 1024;
const MAX_THUMBNAIL_BYTES = 64 * 1024;

function originMatches(request: Request): boolean {
  const origin = request.headers.get("Origin");
  if (!origin) return true;
  return origin === new URL(request.url).origin;
}

export async function action({ request, params, context }: Route.ActionArgs) {
  if (request.method !== "PUT") {
    return new Response("method not allowed", { status: 405 });
  }
  if (!originMatches(request)) {
    return new Response("bad origin", { status: 403 });
  }

  const env = context.cloudflare.env;
  const session = await getSession(request, env);
  if (!session) {
    return new Response("sign in to save", { status: 401 });
  }

  const slug = params.slug;
  const shader = await getShader(env, slug);
  if (!shader) {
    return new Response("no such shader", { status: 404 });
  }
  if (shader.owner_id !== session.userId) {
    return new Response("not the owner", { status: 403 });
  }

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
  const source = (body as { source?: unknown })?.source;
  if (typeof source !== "string") {
    return new Response("missing source", { status: 400 });
  }
  if (new TextEncoder().encode(source).byteLength > MAX_SOURCE_BYTES) {
    return new Response("source too large", { status: 413 });
  }

  const thumbnailRaw = (body as { thumbnail?: unknown })?.thumbnail;
  let thumbnail: string | null | undefined = undefined;
  if (typeof thumbnailRaw === "string" && thumbnailRaw.startsWith("data:image/")) {
    if (thumbnailRaw.length > MAX_THUMBNAIL_BYTES) {
      return new Response("thumbnail too large", { status: 413 });
    }
    thumbnail = thumbnailRaw;
  } else if (thumbnailRaw === null) {
    thumbnail = null;
  }

  // Same convention as thumbnail: undefined → leave column untouched,
  // explicit null → clear, string → sanitize and set.
  const titleRaw = (body as { title?: unknown })?.title;
  let title: string | null | undefined = undefined;
  if (typeof titleRaw === "string") {
    const trimmed = titleRaw.replace(/\s+/g, " ").trim();
    title = trimmed ? trimmed.slice(0, 120) : null;
  } else if (titleRaw === null) {
    title = null;
  }

  await updateShaderSource(env, slug, source, thumbnail, title);
  return Response.json({ slug });
}
