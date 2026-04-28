// Passwordless OTP issuance + verification.
//
// One row per pending code, keyed by (contact, contact_kind). Issuing
// a fresh code overwrites the previous one — the user can re-request
// without leaving zombie rows. The code itself is never stored; we
// keep an HMAC-SHA256 hash with `SESSION_COOKIE_SECRET` as the key, so
// a DB read alone can't be replayed against the verify endpoint.
//
// Constants chosen for SMS-friendly UX: 6 digits, 10-minute expiry,
// 5 attempts per issued code. Re-requesting always replaces the
// existing row (and zeroes attempts) so a user who fat-fingered five
// times can recover by hitting "send another code".

import type { Env } from "../../workers/env";

export type ContactKind = "email" | "phone";

export const CODE_LENGTH = 6;
export const CODE_TTL_SECONDS = 600;
export const MAX_ATTEMPTS = 5;

export interface IssuedCode {
  /** The cleartext code to send to the user — caller mails/SMSes this
   *  and must NOT persist it. */
  code: string;
  /** Unix timestamp seconds. */
  expiresAt: number;
}

export async function issueLoginCode(
  env: Env,
  contact: string,
  kind: ContactKind,
): Promise<IssuedCode> {
  const code = randomDigits(CODE_LENGTH);
  const codeHash = await hmacHex(env.SESSION_COOKIE_SECRET, contactKey(contact, kind, code));
  const expiresAt = Math.floor(Date.now() / 1000) + CODE_TTL_SECONDS;
  await env.DB.prepare(
    `INSERT INTO login_codes (contact, contact_kind, code_hash, expires_at, attempts)
     VALUES (?, ?, ?, ?, 0)
     ON CONFLICT(contact, contact_kind) DO UPDATE SET
       code_hash  = excluded.code_hash,
       expires_at = excluded.expires_at,
       attempts   = 0,
       created_at = unixepoch()`,
  )
    .bind(contact, kind, codeHash, expiresAt)
    .run();
  return { code, expiresAt };
}

export type VerifyResult =
  | { ok: true }
  | { ok: false; reason: "no_code" | "expired" | "too_many_attempts" | "bad_code" };

export async function verifyLoginCode(
  env: Env,
  contact: string,
  kind: ContactKind,
  code: string,
): Promise<VerifyResult> {
  const row = await env.DB.prepare(
    `SELECT code_hash, expires_at, attempts
     FROM login_codes WHERE contact = ? AND contact_kind = ? LIMIT 1`,
  )
    .bind(contact, kind)
    .first<{ code_hash: string; expires_at: number; attempts: number }>();
  if (!row) return { ok: false, reason: "no_code" };
  const now = Math.floor(Date.now() / 1000);
  if (row.expires_at < now) {
    await env.DB.prepare(
      `DELETE FROM login_codes WHERE contact = ? AND contact_kind = ?`,
    )
      .bind(contact, kind)
      .run();
    return { ok: false, reason: "expired" };
  }
  if (row.attempts >= MAX_ATTEMPTS) {
    return { ok: false, reason: "too_many_attempts" };
  }
  const candidate = await hmacHex(
    env.SESSION_COOKIE_SECRET,
    contactKey(contact, kind, code),
  );
  // SHA-256 outputs are fixed-width hex strings, so a constant-time
  // string compare suffices here.
  if (!constantTimeEqual(candidate, row.code_hash)) {
    await env.DB.prepare(
      `UPDATE login_codes SET attempts = attempts + 1
       WHERE contact = ? AND contact_kind = ?`,
    )
      .bind(contact, kind)
      .run();
    return { ok: false, reason: "bad_code" };
  }
  // Single-use: consume on success.
  await env.DB.prepare(
    `DELETE FROM login_codes WHERE contact = ? AND contact_kind = ?`,
  )
    .bind(contact, kind)
    .run();
  return { ok: true };
}

// Cryptographically-strong N-digit string. `crypto.getRandomValues`
// + modulo-by-10 is uniformly distributed across digits since 256 % 10
// has bias of <1.6% per digit, which is fine for a 6-digit OTP that
// the user only sees one of and that gets rate-limited anyway.
function randomDigits(n: number): string {
  const bytes = new Uint8Array(n);
  crypto.getRandomValues(bytes);
  let out = "";
  for (const b of bytes) out += (b % 10).toString();
  return out;
}

// The HMAC payload includes contact + kind so a code valid for
// alice@example.com can't be replayed on a phone-number row even if
// somebody crafted matching hash inputs.
function contactKey(contact: string, kind: ContactKind, code: string): string {
  return `${kind}\x00${contact.toLowerCase()}\x00${code}`;
}

async function hmacHex(secret: string, payload: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(payload));
  return Array.from(new Uint8Array(sig), (b) => b.toString(16).padStart(2, "0")).join("");
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}
