# Deploying the Wyn Playground

The playground runs on Cloudflare Workers with three bindings:

- `DB` — D1 database (users + shaders + featured_shaders)
- `SESSIONS` — KV namespace (signed-cookie session store)
- `ASSETS` — static asset directory (`playground/dist/client/`)

## One-time setup (already done for the current deployment)

The current production deployment was bootstrapped with:

```bash
cd playground

# Cloudflare resources
wrangler kv namespace create SESSIONS    # → id pasted into wrangler.jsonc
wrangler d1 create wyn                   # → database_id pasted into wrangler.jsonc

# Initial schema on remote D1
wrangler d1 execute wyn --remote --file=./migrations/0001_init.sql

# Two GitHub OAuth Apps registered at https://github.com/settings/developers
#   * dev:  callback http://localhost:5173/auth/github/callback
#   * prod: callback https://<deployed-domain>/auth/github/callback
# Dev client id is pasted into wrangler.jsonc :: vars.GITHUB_CLIENT_ID;
# the prod app's client id is wired via a wrangler env or a separate
# wrangler.jsonc entry when ready.

# Secrets (these set the values on the deployed Worker; .dev.vars covers
# the local `wrangler dev` environment with a separate copy)
wrangler secret put GITHUB_CLIENT_SECRET
wrangler secret put SESSION_COOKIE_SECRET    # any long random; openssl rand -base64 48
```

If you're standing up a fresh environment, repeat all of the above
once. The IDs that go into `wrangler.jsonc` are public-safe (resource
identifiers, not auth tokens).

## Prerequisites for every deploy

- Node + npm
- Rust toolchain (for `cargo build` of the wasm crate)
- `wasm-pack` — `cargo install wasm-pack`
- `mdbook` — `cargo install mdbook`
- Python 3 (used by `scripts/split_spec.py` to chunk the spec)
- `wrangler` — installed by `npm install` in `playground/`

Authenticate wrangler once: `wrangler login`.

## Routine deploy

From `playground/`:

```bash
# 1. Apply any new D1 migrations to remote BEFORE deploying — backwards-
#    compatible additions (new tables / nullable columns) only. The
#    list of migration files lives in playground/migrations/; check
#    `git log -- migrations/` against what's been run on prod.
wrangler d1 execute wyn --remote --file=./migrations/0002_thumbnail.sql
wrangler d1 execute wyn --remote --file=./migrations/0003_featured_shaders.sql

# 2. Full build (wasm → spec → react-router). Order matters: spec
#    output goes to public/spec/ so vite copies it into dist/client/
#    during react-router build.
npm run build

# 3. Deploy the worker + assets
wrangler deploy
```

That's it. Cloudflare's assets binding picks up everything under
`dist/client/`, including the static mdBook output at `dist/client/spec/`,
and serves it directly without invoking the worker.

## Sanity checks after deploy

- `https://<domain>/` — landing page renders, ripple background animates,
  Featured shader card shows the curated entry (if `featured_shaders`
  has rows) or the most recently saved shader (fallback).
- `https://<domain>/spec/` — mdBook shows the spec.
- `https://<domain>/auth/github` — kicks off the OAuth flow against the
  prod GitHub OAuth App.
- `https://<domain>/u/<your-login>` — your saved shaders.
- Sign in, save a shader, hit the **★ Feature** toggle (admin only) on
  `/s/:slug`, reload `/` — the featured card should be the one you just
  flagged.

## Local dev loop

```bash
cd playground
npm run dev          # vite, hot reload, /spec/ served from public/spec/
                     # if you've run `npm run build:spec` at least once
```

For the cloudflare-dev experience (D1 + KV bindings, asset routing as in
prod), use `wrangler dev` instead — `npm run preview`.

`.dev.vars` (gitignored) holds local copies of `GITHUB_CLIENT_SECRET` and
`SESSION_COOKIE_SECRET`. Generate the latter with
`openssl rand -base64 48`.

## Updating just the spec

Editing `SPECIFICATION.md` doesn't require a full app rebuild:

```bash
npm run build:spec   # python split + mdbook build → public/spec/
```

`vite` (via `npm run dev`) hot-reloads the new files. For prod, you
still need a full `npm run build && wrangler deploy` so the new spec
files land in `dist/client/`.

## Migrations protocol

- Every migration file in `playground/migrations/` is forward-only,
  backwards-compatible (additive tables / nullable columns).
- Apply locally first: `wrangler d1 execute wyn --local --file=./migrations/<n>_*.sql`.
- Apply remotely *before* `wrangler deploy` so the new code can rely on
  the new schema from the first request after rollout.
- Never re-run an already-applied migration; `wrangler d1` doesn't track
  state and a re-run errors on duplicate columns / tables.

## Admin

Admin privileges are gated on `wrangler.jsonc :: vars.ADMIN_LOGINS`
(comma-separated GitHub logins, case-insensitive). Adding a new admin:

1. Edit `vars.ADMIN_LOGINS` in `wrangler.jsonc`.
2. `wrangler deploy` (no migration needed; this is a runtime var).

Admins get a **★ Feature** / **☆ Feature** toggle in the toolbar on
`/s/:slug` pages, which manages rows in `featured_shaders` via
`POST /api/featured`.
