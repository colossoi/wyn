// /auth/login — entry point for both auth providers. The Header's
// "Sign in" link routes here instead of going straight to GitHub, so
// users without a GitHub account have a path in.

import type { Route } from "./+types/auth.login";
import { Form, Link, redirect } from "react-router";
import { getSession } from "~/lib/session.server";

export function meta({}: Route.MetaArgs) {
  return [{ title: "Sign in — Wyn Playground" }];
}

export async function loader({ request, context }: Route.LoaderArgs) {
  const env = context.cloudflare.env;
  const session = await getSession(request, env);
  if (session) throw redirect("/p/new");
  return null;
}

export default function Login() {
  return (
    <div className="auth-shell">
      <div className="auth-card">
        <h1 className="auth-title">Sign in</h1>
        <p className="auth-subtitle">
          You only need an account to save shaders. Viewing shared links
          works without one.
        </p>
        <a href="/auth/github" className="auth-provider auth-provider-github">
          Continue with GitHub
        </a>
        <div className="auth-divider"><span>or</span></div>
        <Form method="post" action="/auth/code" className="auth-email-form">
          <label className="auth-field">
            <span>Email</span>
            <input
              type="email"
              name="contact"
              required
              autoComplete="email"
              spellCheck={false}
              placeholder="you@example.com"
              className="auth-input"
            />
          </label>
          <button type="submit" className="auth-submit">
            Email me a sign-in code
          </button>
        </Form>
        <p className="auth-footnote">
          <Link to="/" className="auth-back">← Back to home</Link>
        </p>
      </div>
    </div>
  );
}
