// GET /auth/github/callback — finish the OAuth dance.
//
// Verify the state cookie (CSRF), exchange `code` for an access token, fetch
// the GitHub user, upsert into D1, issue a session cookie, redirect home.

import type { Route } from "./+types/auth.callback";
import { redirect } from "react-router";
import { exchangeCode, fetchUser } from "~/lib/github.server";
import { upsertUser } from "~/lib/db.server";
import {
  clearSignedCookie,
  createSession,
  readSignedCookie,
} from "~/lib/session.server";

const STATE_COOKIE = "__oauth_state";

export async function loader({ request, context }: Route.LoaderArgs) {
  const env = context.cloudflare.env;
  const url = new URL(request.url);
  const code = url.searchParams.get("code");
  const returnedState = url.searchParams.get("state");

  if (!code || !returnedState) {
    throw new Response("missing code or state", { status: 400 });
  }

  const expectedState = await readSignedCookie(
    request,
    STATE_COOKIE,
    env.SESSION_COOKIE_SECRET,
  );
  if (!expectedState || expectedState !== returnedState) {
    throw new Response("invalid oauth state", { status: 400 });
  }

  const redirectUri = `${url.origin}/auth/github/callback`;
  const accessToken = await exchangeCode(env, code, redirectUri);
  const user = await fetchUser(accessToken);

  await upsertUser(env, user);

  const sessionCookie = await createSession(env, {
    userId: user.id,
    login: user.login,
    avatarUrl: user.avatarUrl,
    createdAt: Math.floor(Date.now() / 1000),
  });

  // Two Set-Cookie headers: the session (write) and the state cookie (clear).
  const headers = new Headers();
  headers.append("Set-Cookie", sessionCookie);
  headers.append("Set-Cookie", clearSignedCookie(STATE_COOKIE));

  return redirect("/new", { headers });
}
