-- D1 schema for the playground.
--
-- Run once per environment:
--   wrangler d1 execute wyn --local  --file=./migrations/0001_init.sql
--   wrangler d1 execute wyn --remote --file=./migrations/0001_init.sql

CREATE TABLE users (
  id         INTEGER PRIMARY KEY,                     -- GitHub numeric id
  login      TEXT NOT NULL UNIQUE COLLATE NOCASE,
  avatar_url TEXT,
  created_at INTEGER NOT NULL DEFAULT (unixepoch()),
  updated_at INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE shaders (
  slug       TEXT PRIMARY KEY,                        -- 8 chars, [A-Za-z0-9]
  owner_id   INTEGER NOT NULL REFERENCES users(id),
  title      TEXT,                                    -- nullable; UI deferred
  source     TEXT NOT NULL,                           -- <= 256 KiB, enforced in action
  created_at INTEGER NOT NULL DEFAULT (unixepoch()),
  updated_at INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX idx_shaders_owner ON shaders(owner_id, updated_at DESC);
