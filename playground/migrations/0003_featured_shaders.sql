-- Featured shaders — manually curated, ordered by `rank` (lower wins).
-- Currently the home page uses only the top entry; expanding to top-N
-- later is a single SELECT change. ON DELETE CASCADE so removing a
-- shader auto-removes its featured entry.
--
--   wrangler d1 execute wyn --local  --file=./migrations/0003_featured_shaders.sql
--   wrangler d1 execute wyn --remote --file=./migrations/0003_featured_shaders.sql

CREATE TABLE featured_shaders (
  slug      TEXT PRIMARY KEY REFERENCES shaders(slug) ON DELETE CASCADE,
  rank      INTEGER NOT NULL DEFAULT 0,
  added_at  INTEGER NOT NULL DEFAULT (unixepoch()),
  added_by  INTEGER REFERENCES users(id)
);

CREATE INDEX idx_featured_rank ON featured_shaders(rank ASC, added_at DESC);
