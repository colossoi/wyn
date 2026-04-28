// /auth/code/verify — submit the code that was emailed (or, eventually,
// SMSed). On success, find-or-create the user row, issue a session
// cookie, and redirect to /p/new.

import type { Route } from "./+types/auth.code.verify";
import { Form, redirect, useSearchParams } from "react-router";
import { upsertUserByEmail } from "~/lib/db.server";
import { isLikelyEmail } from "~/lib/email.server";
import { verifyLoginCode, type ContactKind } from "~/lib/login_codes.server";
import { createSession } from "~/lib/session.server";

export function meta({}: Route.MetaArgs) {
  return [{ title: "Enter your code — Wyn Playground" }];
}

interface ActionData {
  error?: string;
}

function originMatches(request: Request): boolean {
  const origin = request.headers.get("Origin");
  if (!origin) return true;
  return origin === new URL(request.url).origin;
}

export async function loader() {
  // The form posts back here; loader has nothing to do.
  return null;
}

export async function action({
  request,
  context,
}: Route.ActionArgs): Promise<Response | ActionData> {
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
  const codeRaw = form.get("code");
  if (
    typeof contactRaw !== "string" ||
    typeof kindRaw !== "string" ||
    typeof codeRaw !== "string"
  ) {
    return { error: "Missing fields." };
  }
  if (kindRaw !== "email") {
    return { error: "Unsupported contact kind." };
  }
  const kind: ContactKind = "email";
  const contact = contactRaw.trim().toLowerCase();
  if (!isLikelyEmail(contact)) {
    return { error: "Invalid email." };
  }
  const code = codeRaw.replace(/\s+/g, "");
  if (!/^\d{6}$/.test(code)) {
    return { error: "Code must be 6 digits." };
  }

  const result = await verifyLoginCode(env, contact, kind, code);
  if (!result.ok) {
    return { error: errorMessage(result.reason) };
  }

  const user = await upsertUserByEmail(env, contact);
  const sessionCookie = await createSession(env, {
    userId: user.id,
    login: user.login,
    avatarUrl: null,
    createdAt: Math.floor(Date.now() / 1000),
  });
  return redirect("/p/new", {
    headers: { "Set-Cookie": sessionCookie },
  });
}

function errorMessage(reason: string): string {
  switch (reason) {
    case "no_code":
      return "No code outstanding for that contact. Request a new one.";
    case "expired":
      return "That code has expired. Request a new one.";
    case "too_many_attempts":
      return "Too many wrong attempts. Request a new code.";
    case "bad_code":
    default:
      return "Incorrect code.";
  }
}

export default function VerifyCode({ actionData }: Route.ComponentProps) {
  const [searchParams] = useSearchParams();
  const contact = searchParams.get("contact") ?? "";
  const kind = searchParams.get("kind") ?? "email";
  const error = (actionData as ActionData | undefined)?.error ?? null;

  return (
    <div className="auth-shell">
      <div className="auth-card">
        <h1 className="auth-title">Enter your code</h1>
        <p className="auth-subtitle">
          We sent a 6-digit code to <strong>{contact || "your address"}</strong>.
          It expires in 10 minutes.
        </p>
        <Form method="post" className="auth-email-form">
          <input type="hidden" name="contact" value={contact} />
          <input type="hidden" name="kind" value={kind} />
          <label className="auth-field">
            <span>Code</span>
            <input
              type="text"
              name="code"
              required
              autoComplete="one-time-code"
              inputMode="numeric"
              pattern="[0-9]{6}"
              maxLength={6}
              spellCheck={false}
              placeholder="123456"
              className="auth-input auth-code-input"
              autoFocus
            />
          </label>
          {error && <div className="auth-error">{error}</div>}
          <button type="submit" className="auth-submit">
            Verify
          </button>
        </Form>
        <p className="auth-footnote">
          Didn't get it?{" "}
          <Form
            method="post"
            action="/auth/code"
            style={{ display: "inline" }}
          >
            <input type="hidden" name="contact" value={contact} />
            <input type="hidden" name="kind" value={kind} />
            <button type="submit" className="auth-link-button">
              Send another code
            </button>
          </Form>
        </p>
      </div>
    </div>
  );
}
