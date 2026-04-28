// POST /auth/code — request a one-time login code.
//
// Body: form-encoded `contact=<email>` (kind=email implicit for now;
// SMS will pass `kind=phone`). Issues a fresh code, sends it via the
// `send_email` binding, redirects to /auth/code/verify with the
// contact + kind in the query string.
//
// We don't reveal whether the email is registered: the verify page
// renders identically and a successful verification will create the
// user row if it doesn't already exist.

import type { Route } from "./+types/auth.code";
import { redirect } from "react-router";
import { sendLoginCode, isLikelyEmail } from "~/lib/email.server";
import { issueLoginCode, type ContactKind } from "~/lib/login_codes.server";

const RESEND_COOLDOWN_SECONDS = 30;

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
  const form = await request.formData();
  const contactRaw = form.get("contact");
  const kindRaw = form.get("kind") ?? "email";
  if (typeof contactRaw !== "string" || typeof kindRaw !== "string") {
    return new Response("missing contact", { status: 400 });
  }
  if (kindRaw !== "email") {
    return new Response("unsupported contact kind", { status: 400 });
  }
  const kind: ContactKind = "email";
  const contact = contactRaw.trim().toLowerCase();
  if (!isLikelyEmail(contact)) {
    return new Response("invalid email", { status: 400 });
  }

  // Per-contact rate limit: refuse re-requests within the cooldown
  // window. The login_codes row's created_at is reset on each issue,
  // so this naturally throttles a single email regardless of how many
  // verify-page tabs the user has open.
  const recent = await env.DB.prepare(
    `SELECT created_at FROM login_codes WHERE contact = ? AND contact_kind = ?`,
  )
    .bind(contact, kind)
    .first<{ created_at: number }>();
  const now = Math.floor(Date.now() / 1000);
  if (recent && now - recent.created_at < RESEND_COOLDOWN_SECONDS) {
    return new Response("please wait before requesting another code", {
      status: 429,
    });
  }

  const issued = await issueLoginCode(env, contact, kind);
  await sendLoginCode(env, contact, issued.code, issued.expiresAt);

  const params = new URLSearchParams({ contact, kind });
  return redirect(`/auth/code/verify?${params.toString()}`);
}

// Hitting GET on this URL doesn't make sense — the form is on /auth/login.
export async function loader() {
  return redirect("/auth/login");
}
