// Outbound transactional mail via the Cloudflare `send_email` binding.
//
// Cloudflare's binding takes a fully-formed RFC 5322 message; we build
// it by hand here (no `mimetext` dep) for a one-shot plain-text use
// case. The binding requires DKIM on the sender domain — set up in
// the Cloudflare Email Routing dashboard for `LOGIN_FROM_EMAIL`'s
// domain.

import { EmailMessage } from "cloudflare:email";
import type { Env } from "../../workers/env";

export async function sendLoginCode(
  env: Env,
  recipient: string,
  code: string,
  expiresAtSec: number,
): Promise<void> {
  if (!isLikelyEmail(recipient)) {
    // Caller should have validated; this is a belt-and-braces guard
    // because the address goes into header lines unescaped.
    throw new Error("sendLoginCode: refusing malformed recipient");
  }
  const fromEmail = env.LOGIN_FROM_EMAIL;
  const fromName = env.LOGIN_FROM_NAME;
  if (!isLikelyEmail(fromEmail)) {
    throw new Error("sendLoginCode: LOGIN_FROM_EMAIL is malformed");
  }
  const ttlSec = Math.max(0, expiresAtSec - Math.floor(Date.now() / 1000));
  const ttlMin = Math.max(1, Math.round(ttlSec / 60));

  const subject = `Your Wyn Playground login code: ${code}`;
  const body =
    `Your Wyn Playground login code is:\r\n` +
    `\r\n` +
    `    ${code}\r\n` +
    `\r\n` +
    `It expires in ${ttlMin} minute${ttlMin === 1 ? "" : "s"}.\r\n` +
    `\r\n` +
    `If you didn't request this code, you can safely ignore this email.\r\n`;

  const date = new Date().toUTCString();
  const fromDomain = fromEmail.slice(fromEmail.indexOf("@") + 1);
  const messageId = `<${crypto.randomUUID()}@${fromDomain}>`;

  const headers = [
    `From: "${escapeQuoted(fromName)}" <${fromEmail}>`,
    `To: <${recipient}>`,
    `Subject: ${escapeHeader(subject)}`,
    `Date: ${date}`,
    `Message-ID: ${messageId}`,
    `Auto-Submitted: auto-generated`,
    `MIME-Version: 1.0`,
    `Content-Type: text/plain; charset=utf-8`,
    `Content-Transfer-Encoding: 7bit`,
  ];
  const raw = headers.join("\r\n") + "\r\n\r\n" + body;

  const message = new EmailMessage(fromEmail, recipient, raw);
  await env.MAILER.send(message);
}

// Loose check sufficient to reject header-injection attempts and
// obvious malformed input. Real validity is enforced by the receiving
// MTA — there's no point matching the full RFC 5322 grammar here.
export function isLikelyEmail(s: string): boolean {
  if (s.length > 254) return false;
  if (/[\s<>"]/.test(s)) return false;
  const at = s.indexOf("@");
  if (at <= 0 || at !== s.lastIndexOf("@")) return false;
  if (at === s.length - 1) return false;
  if (!/\./.test(s.slice(at + 1))) return false;
  return true;
}

// Strip CR/LF — the only header-injection vector that matters here.
// Anything else is the caller's problem.
function escapeHeader(s: string): string {
  return s.replace(/[\r\n]+/g, " ");
}

function escapeQuoted(s: string): string {
  return escapeHeader(s).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}
