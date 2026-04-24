-- Add a thumbnail column to shaders. Captured at save time as a small
-- JPEG data URL (downscaled to ~320x180); nullable so old rows and
-- compute-only programs stay valid.
--
--   wrangler d1 execute wyn --local  --file=./migrations/0002_thumbnail.sql
--   wrangler d1 execute wyn --remote --file=./migrations/0002_thumbnail.sql

ALTER TABLE shaders ADD COLUMN thumbnail TEXT;
