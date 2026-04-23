// Session handling: opaque session ID in an HttpOnly cookie, backed by KV.
//
// Layout:
//   Cookie __wyn_session = <32-byte random, base64url>
//   KV key  sess:<id>    = JSON { userId, login, avatarUrl, createdAt }
//
// A 30-day TTL is set on the KV entry at creation; we don't extend on
// activity (sliding expiry is a nice-to-have, not needed for this pass).
// Logout deletes the KV entry AND clears the cookie.

import type { Env } from "../../workers/env";

export interface Session {
  userId: number;
  login: string;
  avatarUrl: string | null;
  createdAt: number;
}

const SESSION_COOKIE = "__wyn_session";
const SESSION_TTL_SECONDS = 60 * 60 * 24 * 30; // 30 days

function sessionCookie(id: string, maxAge: number): string {
  const attrs = [
    `${SESSION_COOKIE}=${id}`,
    "HttpOnly",
    "Secure",
    "SameSite=Lax",
    "Path=/",
    `Max-Age=${maxAge}`,
  ];
  return attrs.join("; ");
}

function randomId(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  // base64url
  let b64 = btoa(String.fromCharCode(...bytes));
  b64 = b64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  return b64;
}

// Cookie header parser. `request.headers.get("Cookie")` returns the raw
// header; split by `; ` and find the one we want. Accepts zero or multiple
// cookies; silently ignores malformed pairs.
function readCookie(request: Request, name: string): string | null {
  const header = request.headers.get("Cookie");
  if (!header) return null;
  for (const pair of header.split(/;\s*/)) {
    const eq = pair.indexOf("=");
    if (eq < 0) continue;
    if (pair.slice(0, eq) === name) {
      return decodeURIComponent(pair.slice(eq + 1));
    }
  }
  return null;
}

// Load the active session, or null if the cookie is missing / KV entry gone.
// Never throws — a bad cookie just means "not logged in".
export async function getSession(request: Request, env: Env): Promise<Session | null> {
  const id = readCookie(request, SESSION_COOKIE);
  if (!id) return null;
  try {
    const raw = await env.SESSIONS.get(`sess:${id}`, "json");
    if (!raw || typeof raw !== "object") return null;
    // Trust the shape we wrote.
    return raw as Session;
  } catch {
    return null;
  }
}

// Create a new session for `user` and return the Set-Cookie header value.
// The caller attaches it to the redirect/response.
export async function createSession(env: Env, user: Session): Promise<string> {
  const id = randomId();
  await env.SESSIONS.put(`sess:${id}`, JSON.stringify(user), {
    expirationTtl: SESSION_TTL_SECONDS,
  });
  return sessionCookie(id, SESSION_TTL_SECONDS);
}

// Destroy the session referenced by the request cookie (if any) and return
// a Set-Cookie header that clears the cookie in the browser.
export async function destroySession(request: Request, env: Env): Promise<string> {
  const id = readCookie(request, SESSION_COOKIE);
  if (id) {
    try {
      await env.SESSIONS.delete(`sess:${id}`);
    } catch {
      // Best-effort — worst case the session entry is orphaned in KV until TTL.
    }
  }
  return sessionCookie("", 0);
}

// ----------------------------------------------------------------------------
// Signed short-lived cookies (used for OAuth `state`).
//
// HMAC-SHA256 over the cookie value with `SESSION_COOKIE_SECRET` as the key.
// The stored cookie value is `<payload>.<signature>` where both halves are
// base64url; if the signature doesn't verify we treat the cookie as absent.
// ----------------------------------------------------------------------------

function toBase64Url(bytes: ArrayBuffer | Uint8Array): string {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let b64 = btoa(String.fromCharCode(...u8));
  return b64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(s: string): Uint8Array {
  s = s.replace(/-/g, "+").replace(/_/g, "/");
  while (s.length % 4) s += "=";
  const bin = atob(s);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}

async function hmacKey(secret: string): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"],
  );
}

async function sign(secret: string, payload: string): Promise<string> {
  const key = await hmacKey(secret);
  const sig = await crypto.subtle.sign(
    "HMAC",
    key,
    new TextEncoder().encode(payload),
  );
  return toBase64Url(sig);
}

async function verify(secret: string, payload: string, signature: string): Promise<boolean> {
  try {
    const key = await hmacKey(secret);
    const sig = fromBase64Url(signature);
    return await crypto.subtle.verify(
      "HMAC",
      key,
      sig.buffer.slice(sig.byteOffset, sig.byteOffset + sig.byteLength) as ArrayBuffer,
      new TextEncoder().encode(payload),
    );
  } catch {
    return false;
  }
}

// Build a Set-Cookie for a signed short-lived cookie.
// Path is narrowed to /auth so the cookie doesn't go out on every request.
export async function signedCookie(
  name: string,
  value: string,
  secret: string,
  maxAgeSeconds: number,
): Promise<string> {
  const payload = toBase64Url(new TextEncoder().encode(value));
  const sig = await sign(secret, payload);
  const combined = `${payload}.${sig}`;
  return [
    `${name}=${combined}`,
    "HttpOnly",
    "Secure",
    "SameSite=Lax",
    "Path=/auth",
    `Max-Age=${maxAgeSeconds}`,
  ].join("; ");
}

// Read + verify a signed short-lived cookie. Returns the original value or
// null. Also tolerates a missing cookie.
export async function readSignedCookie(
  request: Request,
  name: string,
  secret: string,
): Promise<string | null> {
  const raw = readCookie(request, name);
  if (!raw) return null;
  const dot = raw.indexOf(".");
  if (dot < 0) return null;
  const payload = raw.slice(0, dot);
  const sig = raw.slice(dot + 1);
  if (!(await verify(secret, payload, sig))) return null;
  try {
    return new TextDecoder().decode(fromBase64Url(payload));
  } catch {
    return null;
  }
}

// Clear a signed cookie (immediate expiry).
export function clearSignedCookie(name: string): string {
  return [
    `${name}=`,
    "HttpOnly",
    "Secure",
    "SameSite=Lax",
    "Path=/auth",
    "Max-Age=0",
  ].join("; ");
}
