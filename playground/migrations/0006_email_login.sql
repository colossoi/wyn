-- Passwordless OTP login (email today, phone future) alongside GitHub OAuth.
--
-- Two changes:
--   1. Rebuild `users` so identity is provider-agnostic. The old schema
--      treated `users.id` as the GitHub numeric id, which doesn't fit
--      OTP-only users. New shape: an autoincrement internal id, with
--      `github_id` and `email` as nullable, unique provider columns.
--      Existing rows preserve their id (= old github numeric id), so
--      session cookies and shaders.owner_id keep pointing at the same
--      user. A future `phone` column slots in alongside `email` when
--      SMS-based login lands.
--   2. Add `login_codes` — one row per pending OTP, replaced on
--      re-request, deleted on successful verify or expiry. Keyed by
--      `(contact, contact_kind)` so the same plumbing carries email
--      and phone OTPs.
--
--   wrangler d1 execute wyn --local  --file=./migrations/0006_email_login.sql
--   wrangler d1 execute wyn --remote --file=./migrations/0006_email_login.sql

CREATE TABLE users_new (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  github_id   INTEGER UNIQUE,
  email       TEXT    UNIQUE COLLATE NOCASE,
  login       TEXT NOT NULL UNIQUE COLLATE NOCASE,
  avatar_url  TEXT,
  created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
  updated_at  INTEGER NOT NULL DEFAULT (unixepoch()),
  -- A row must have at least one identity column populated.
  CHECK (github_id IS NOT NULL OR email IS NOT NULL)
);

INSERT INTO users_new (id, github_id, login, avatar_url, created_at, updated_at)
SELECT id, id, login, avatar_url, created_at, updated_at FROM users;

DROP TABLE users;
ALTER TABLE users_new RENAME TO users;

-- Partial indexes: only non-null provider ids participate.
CREATE INDEX idx_users_github_id ON users(github_id) WHERE github_id IS NOT NULL;
CREATE INDEX idx_users_email     ON users(email)     WHERE email     IS NOT NULL;

CREATE TABLE login_codes (
  contact       TEXT NOT NULL COLLATE NOCASE,         -- email address or E.164 phone
  contact_kind  TEXT NOT NULL CHECK (contact_kind IN ('email', 'phone')),
  code_hash     TEXT NOT NULL,                        -- HMAC-SHA256 hex of the OTP
  expires_at    INTEGER NOT NULL,
  attempts      INTEGER NOT NULL DEFAULT 0,
  created_at    INTEGER NOT NULL DEFAULT (unixepoch()),
  PRIMARY KEY (contact, contact_kind)
);

CREATE INDEX idx_login_codes_expires ON login_codes(expires_at);
