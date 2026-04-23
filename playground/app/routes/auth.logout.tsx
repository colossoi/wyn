// POST /auth/logout — destroy the KV session row and clear the cookie.
//
// Exposed as an `action` (POST) so a simple `<form method="post">` in the
// header works; no GET logout (to avoid accidental prefetch / XSS-triggered
// sign-out).

import type { Route } from "./+types/auth.logout";
import { redirect } from "react-router";
import { destroySession } from "~/lib/session.server";

export async function action({ request, context }: Route.ActionArgs) {
  const env = context.cloudflare.env;
  const setCookie = await destroySession(request, env);
  return redirect("/", { headers: { "Set-Cookie": setCookie } });
}

// If someone hits /auth/logout with GET, send them home rather than 404.
export async function loader() {
  return redirect("/");
}
