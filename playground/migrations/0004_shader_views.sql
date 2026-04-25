-- Per-shader view counter. Incremented on every GET /s/:slug; the
-- /popular page sorts by views DESC. ON DELETE CASCADE so removing a
-- shader auto-removes its counter row.
--
--   wrangler d1 execute wyn --local  --file=./migrations/0004_shader_views.sql
--   wrangler d1 execute wyn --remote --file=./migrations/0004_shader_views.sql

CREATE TABLE shader_views (
  slug       TEXT PRIMARY KEY REFERENCES shaders(slug) ON DELETE CASCADE,
  views      INTEGER NOT NULL DEFAULT 0,
  last_view  INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX idx_shader_views_count ON shader_views(views DESC);
