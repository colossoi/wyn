// PUT /api/shaders/:slug — update an existing shader. Owner-only.
//
// Body: JSON { source: string }
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

  await updateShaderSource(env, slug, source);
  return Response.json({ slug });
}
