// POST /api/shaders — create a new shader for the current user.
//
// Body: JSON { source: string }
// Responses:
//   200 { slug: "..." }  — created
//   401                  — not logged in
//   413                  — source too large
//   415                  — not JSON
//   400                  — malformed body / missing source
//   403                  — cross-origin request

import type { Route } from "./+types/api.shaders";
import { createShader } from "~/lib/db.server";
import { getSession } from "~/lib/session.server";

const MAX_SOURCE_BYTES = 256 * 1024;

function originMatches(request: Request): boolean {
  const origin = request.headers.get("Origin");
  if (!origin) {
    // Non-browser clients (curl, server-to-server) won't send Origin.
    // We still require a session cookie below, so skipping this check
    // for them is safe.
    return true;
  }
  const expected = new URL(request.url).origin;
  return origin === expected;
}

export async function action({ request, context }: Route.ActionArgs) {
  if (request.method !== "POST") {
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

  const slug = await createShader(env, session.userId, source);
  return Response.json({ slug });
}
