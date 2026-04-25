// POST /api/featured — admin-only. Manage the `featured_shaders` table.
//
// Body: JSON `{ slug: string, action: "set" | "remove", rank?: number }`
// Responses:
//   200 { slug, action }  — applied
//   400                   — malformed body / missing slug / unknown action
//   401                   — not logged in
//   403                   — logged in but not an admin (or cross-origin)
//   404                   — slug doesn't exist (FK rejects on `set`)
//   415                   — not JSON

import type { Route } from "./+types/api.featured";
import {
  getShader,
  isAdmin,
  removeFeatured,
  setFeatured,
} from "~/lib/db.server";
import { getSession } from "~/lib/session.server";

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
  const session = await getSession(request, env);
  if (!session) {
    return new Response("sign in required", { status: 401 });
  }
  if (!isAdmin(env, session.login)) {
    return new Response("admin only", { status: 403 });
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
  const slug = (body as { slug?: unknown })?.slug;
  const action = (body as { action?: unknown })?.action;
  if (typeof slug !== "string" || !slug) {
    return new Response("missing slug", { status: 400 });
  }
  if (action !== "set" && action !== "remove") {
    return new Response("action must be 'set' or 'remove'", { status: 400 });
  }

  // Verify the slug exists before touching the featured table — gives a
  // clean 404 instead of an opaque FK constraint error.
  const shader = await getShader(env, slug);
  if (!shader) {
    return new Response("no such shader", { status: 404 });
  }

  if (action === "set") {
    const rank = (body as { rank?: unknown })?.rank;
    const r = typeof rank === "number" && Number.isFinite(rank) ? rank : 0;
    await setFeatured(env, slug, r, session.userId);
  } else {
    await removeFeatured(env, slug);
  }
  return Response.json({ slug, action });
}
