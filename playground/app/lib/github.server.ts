// GitHub OAuth 2.0 authorization-code helpers.
//
// Flow (per auth.github.tsx / auth.callback.tsx):
//   1. mint `state`, stash in a signed short-lived cookie, redirect to
//      `authorizeUrl(state, redirectUri)`.
//   2. on callback, verify the state cookie, then exchange `code` for an
//      access token via `exchangeCode`, then fetch the GitHub user via
//      `fetchUser`.
//   3. upsert into D1 and issue a session.

import type { Env } from "../../workers/env";

const AUTHORIZE = "https://github.com/login/oauth/authorize";
const TOKEN = "https://github.com/login/oauth/access_token";
const USER_API = "https://api.github.com/user";

export interface GithubUser {
  id: number;
  login: string;
  avatarUrl: string | null;
}

export function authorizeUrl(env: Env, redirectUri: string, state: string): string {
  const qs = new URLSearchParams({
    client_id: env.GITHUB_CLIENT_ID,
    redirect_uri: redirectUri,
    scope: "read:user",
    state,
    allow_signup: "true",
  });
  return `${AUTHORIZE}?${qs}`;
}

export async function exchangeCode(
  env: Env,
  code: string,
  redirectUri: string,
): Promise<string> {
  const resp = await fetch(TOKEN, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: new URLSearchParams({
      client_id: env.GITHUB_CLIENT_ID,
      client_secret: env.GITHUB_CLIENT_SECRET,
      code,
      redirect_uri: redirectUri,
    }),
  });
  if (!resp.ok) {
    throw new Error(`GitHub token exchange failed: ${resp.status}`);
  }
  const body = (await resp.json()) as { access_token?: string; error?: string };
  if (!body.access_token) {
    throw new Error(`GitHub token exchange: ${body.error ?? "missing access_token"}`);
  }
  return body.access_token;
}

export async function fetchUser(accessToken: string): Promise<GithubUser> {
  const resp = await fetch(USER_API, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
      Accept: "application/vnd.github+json",
      "User-Agent": "wyn-playground",
    },
  });
  if (!resp.ok) {
    throw new Error(`GitHub /user failed: ${resp.status}`);
  }
  const body = (await resp.json()) as {
    id: number;
    login: string;
    avatar_url: string | null;
  };
  return {
    id: body.id,
    login: body.login,
    avatarUrl: body.avatar_url,
  };
}
