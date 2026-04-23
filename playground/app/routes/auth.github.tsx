// GET /auth/github — start of the OAuth 2.0 authorization-code flow.
//
// Mint a random `state`, stash it in a signed short-lived cookie scoped to
// `/auth`, and redirect to GitHub. The callback route verifies the cookie
// before trusting the returned `state`.

import type { Route } from "./+types/auth.github";
import { redirect } from "react-router";
import { authorizeUrl } from "~/lib/github.server";
import { signedCookie } from "~/lib/session.server";

const STATE_COOKIE = "__oauth_state";
const STATE_TTL_SECONDS = 600;

function randomState(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  let b64 = btoa(String.fromCharCode(...bytes));
  return b64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export async function loader({ request, context }: Route.LoaderArgs) {
  const env = context.cloudflare.env;
  const state = randomState();
  const origin = new URL(request.url).origin;
  const redirectUri = `${origin}/auth/github/callback`;

  const cookie = await signedCookie(
    STATE_COOKIE,
    state,
    env.SESSION_COOKIE_SECRET,
    STATE_TTL_SECONDS,
  );

  return redirect(authorizeUrl(env, redirectUri, state), {
    headers: { "Set-Cookie": cookie },
  });
}
