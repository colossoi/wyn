-- User-submitted bug reports / feedback. Each row captures a snapshot
-- of the source the user was working on, an optional screenshot of the
-- visualization area, a free-form comment, and a category bucket so we
-- can triage compiler vs. playground UI vs. other feedback. Reports
-- are accepted from anonymous viewers too — `user_id` is nullable.
--
--   wrangler d1 execute wyn --local  --file=./migrations/0005_reports.sql
--   wrangler d1 execute wyn --remote --file=./migrations/0005_reports.sql

CREATE TABLE reports (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  category    TEXT NOT NULL CHECK (category IN ('compiler', 'playground', 'other')),
  comment     TEXT NOT NULL DEFAULT '',
  source      TEXT NOT NULL,
  screenshot  TEXT,                                          -- data:image/jpeg URL, or NULL
  user_id     INTEGER REFERENCES users(id) ON DELETE SET NULL,
  user_login  TEXT,                                          -- denormalized snapshot for display
  shader_slug TEXT REFERENCES shaders(slug) ON DELETE SET NULL,
  user_agent  TEXT,
  created_at  INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX idx_reports_created  ON reports(created_at DESC);
CREATE INDEX idx_reports_category ON reports(category, created_at DESC);
