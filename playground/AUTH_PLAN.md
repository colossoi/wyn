# Playground: GitHub login + save/load public shaders

## Context

The Wyn playground (`playground/`) is currently a pure client-side
SPA: it loads a hardcoded example program, runs the WASM compiler
in the browser, and has no notion of users, sessions, or
persistence. Every reload loses the source. There's no way to
share a shader except copy-pasting the source.

We want: let users sign in with GitHub, save their shader under a
short public URL, and share that URL. All shaders are
world-readable; only the owner can edit. Fork / version history
are out of scope for this pass.

This is a net-new server-side layer built on top of the existing
Cloudflare-Worker-SSR React Router app — no auth, cookies,
database, or bindings exist today.

**Decided with the user:**
- Anonymous visitors freely compile and run; login gates only
  save.
- URL scheme: `/s/<slug>` where slug is an 8-char random
  `[A-Za-z0-9]` string.
- `/` when logged out → landing page. `/` when logged in → blank
  playground.
- `/u/<login>` → that user's saved shaders.
- Only the latest saved version is kept; no history.
- Fork / "Save as new" from a non-owned shader is **deferred**;
  Save button just disables with "Fork coming soon".

## Current stack (from exploration)

- `playground/workers/app.ts` — Worker fetch handler that hands
  every request to the React Router SSR entry, passing
  `{ cloudflare: { env, ctx } }` as the route context. Ready to
  carry bindings.
- `playground/wrangler.jsonc` — **zero bindings** today (no KV,
  D1, R2, secrets). Need to provision.
- `playground/app/routes.ts` — single index route.
- `playground/app/routes/home.tsx` — all state + WASM bootstrap
  lives here, client-only. No loaders / actions anywhere in the
  codebase.
- `@react-router/cloudflare` is already a dep; use its cookie
  helpers.
- No existing `cookie`, `session`, or `oauth` references
  anywhere — clean slate.

## Design

### OAuth flow — React Router loaders/actions, opaque-session-ID in KV

GitHub's authorization-code flow, implemented as three route
modules so they share the same `context.cloudflare.env` as the
rest of the app (no separate raw Worker handlers):

- `GET /auth/github` — mint 32 random bytes as `state`, stash in
  a signed `__oauth_state` cookie (`HttpOnly; Secure;
  SameSite=Lax; Path=/auth; Max-Age=600`), redirect to
  `https://github.com/login/oauth/authorize?client_id=…&redirect_uri=${ORIGIN}/auth/github/callback&scope=read:user&state=…&allow_signup=true`.
- `GET /auth/github/callback` — verify state cookie, clear it,
  POST to `https://github.com/login/oauth/access_token`, call
  `https://api.github.com/user`, upsert `users`, mint a session,
  set `__wyn_session` cookie, redirect to `/`.
- `POST /auth/logout` — delete KV entry, clear cookie,
  redirect `/`.

**Session = opaque 32-byte base64url random in an HttpOnly
cookie; KV holds `sess:<id> → {userId, login, avatarUrl,
createdAt}` with 30-day TTL.** Chosen over JWT because we want
immediate server-side revocation on logout without maintaining a
denylist, and KV reads are sub-ms so the per-request lookup is
free. Chosen over a signed-inline cookie because we want the
`users` table authoritative.

### Secrets + config

`playground/wrangler.jsonc` additions (after `observability`):

```jsonc
"vars": { "GITHUB_CLIENT_ID": "<public app id>" },
"kv_namespaces": [{ "binding": "SESSIONS", "id": "<from create>" }],
"d1_databases": [{ "binding": "DB", "database_name": "wyn", "database_id": "<from create>" }]
```

Setup commands the user will run once:

```
cd playground
wrangler kv namespace create SESSIONS
wrangler d1 create wyn
wrangler secret put GITHUB_CLIENT_SECRET
wrangler secret put SESSION_COOKIE_SECRET   # HMAC key for signed state cookie
wrangler d1 execute wyn --remote --file=./migrations/0001_init.sql
```

Register two GitHub OAuth Apps (dev + prod) at
github.com/settings/developers with callback
`<origin>/auth/github/callback`.

### D1 schema — `playground/migrations/0001_init.sql`

```sql
CREATE TABLE users (
  id         INTEGER PRIMARY KEY,           -- GitHub numeric id
  login      TEXT NOT NULL UNIQUE COLLATE NOCASE,
  avatar_url TEXT,
  created_at INTEGER NOT NULL DEFAULT (unixepoch()),
  updated_at INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE shaders (
  slug       TEXT PRIMARY KEY,              -- 8 chars, [A-Za-z0-9]
  owner_id   INTEGER NOT NULL REFERENCES users(id),
  title      TEXT,                          -- nullable; UI deferred
  source     TEXT NOT NULL,                 -- <= 256 KiB (enforced in action)
  created_at INTEGER NOT NULL DEFAULT (unixepoch()),
  updated_at INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX idx_shaders_owner ON shaders(owner_id, updated_at DESC);
```

Slug generation: 8 chars from a 62-char alphabet via
`crypto.getRandomValues`, retry on `UNIQUE` violation.

### Routes / loaders / actions

`playground/app/routes.ts`:

```ts
export default [
  index("routes/home.tsx"),
  route("s/:slug",               "routes/shader.tsx"),
  route("u/:login",              "routes/user.tsx"),
  route("auth/github",           "routes/auth.github.tsx"),
  route("auth/github/callback",  "routes/auth.callback.tsx"),
  route("auth/logout",           "routes/auth.logout.tsx"),
  route("api/shaders",           "routes/api.shaders.tsx"),    // POST create
  route("api/shaders/:slug",     "routes/api.shader.tsx"),     // PUT update
] satisfies RouteConfig;
```

- `root.tsx` — add a `loader` returning `{ session }` (nullable).
  `<Header>` reads it via `useRouteLoaderData("root")`.
- `home.tsx` — loader returns the root session. Body delegates:
  logged-out → `<Landing>`, logged-in → `<Playground
  initialSource={defaultExample} canSave={true} slug={null} />`.
- `shader.tsx` — loader `SELECT … WHERE slug=?`, 404 if missing;
  returns `{ shader, session, isOwner }`. Renders `<Playground
  initialSource={shader.source} slug={shader.slug}
  canSave={isOwner} />`.
- `user.tsx` — loader joins `users`+`shaders`, returns list.
  Dead-simple list UI for this pass.
- `api.shaders.tsx` (POST) — 401 if no session; validate
  `source` (string, ≤ 256 KiB); generate slug; insert; return
  `{ slug }`.
- `api.shader.tsx` (PUT) — 401 if no session; 403 if
  `owner_id !== session.userId`; 413 if body too large; update
  `source` + `updated_at`.

### Frontend changes

- Extract playground body from `routes/home.tsx` into a new
  `app/components/Playground.tsx` taking
  `{ initialSource, slug?, canSave }`. Keep the WASM bootstrap
  and CodeMirror wiring exactly as today.
- New `app/components/Header.tsx` — logo + (a) `Sign in with
  GitHub` `<a href="/auth/github">` when logged out, or (b)
  avatar + `<form method="post" action="/auth/logout">` when
  logged in. Mounted in `root.tsx`'s `<Layout>`.
- New `app/components/Landing.tsx` — marketing blurb and big
  sign-in CTA.
- Toolbar in `Playground.tsx` gains a `Save` button.
  `useFetcher` posts to `/api/shaders` (create) or
  `/api/shaders/:slug` (update). Disabled with tooltip when
  `!canSave`: "Sign in to save" when logged out, "Fork coming
  soon" when logged in but not owner. On successful create,
  `navigate(\`/s/\${slug}\`)`.
- Server/client handoff: the existing `useEffect` that loads the
  example program becomes "use `initialSource` if provided,
  otherwise call `get_example_program()`." WASM init stays
  client-only.

### Security

- CSRF: same-origin React Router posts with `SameSite=Lax`
  session cookies are not reachable from cross-site forms; that's
  the CSRF defense. Add `Origin` header check on `/api/*`
  actions as defense-in-depth.
- Source size: 256 KiB cap, reject with 413.
- Cookies: `__wyn_session`: `HttpOnly; Secure; SameSite=Lax;
  Path=/; Max-Age=2592000`. `__oauth_state`: same attributes,
  `Path=/auth; Max-Age=600`; HMAC-signed with
  `SESSION_COOKIE_SECRET`.
- Session validation centralized in
  `app/lib/session.server.ts :: getSession(request, env)`.
  Every owner-sensitive loader/action calls it — never trust the
  client for `isOwner`.

### Files to add

- `playground/app/lib/session.server.ts` — cookie parse/serialize, KV CRUD.
- `playground/app/lib/github.server.ts` — OAuth helpers.
- `playground/app/lib/db.server.ts` — D1 helpers + slug generator.
- `playground/app/components/Playground.tsx`
- `playground/app/components/Header.tsx`
- `playground/app/components/Landing.tsx`
- `playground/app/routes/shader.tsx`
- `playground/app/routes/user.tsx`
- `playground/app/routes/auth.github.tsx`
- `playground/app/routes/auth.callback.tsx`
- `playground/app/routes/auth.logout.tsx`
- `playground/app/routes/api.shaders.tsx`
- `playground/app/routes/api.shader.tsx`
- `playground/migrations/0001_init.sql`
- `playground/workers/env.d.ts` — augment `Env` with `DB`,
  `SESSIONS`, `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`,
  `SESSION_COOKIE_SECRET`.

### Files to modify

- `playground/app/routes.ts`
- `playground/app/root.tsx` — add loader, mount `<Header>`.
- `playground/app/routes/home.tsx` — delegate to
  `<Landing>` / `<Playground>`.
- `playground/wrangler.jsonc` — bindings + vars.
- `playground/workers/app.ts` — unchanged; the `env` type
  argument just widens from `unknown` to the new `Env`.

No new npm deps expected (Web Crypto handles HMAC + random;
`@react-router/cloudflare` already provides cookie helpers).

## Verification

Local: `wrangler d1 create wyn --local`, `wrangler kv namespace
create SESSIONS --local`, `wrangler d1 execute wyn --local
--file=./migrations/0001_init.sql`, then `npm run dev`. Register
a dev-only GitHub OAuth App with `http://localhost:5173/auth/github/callback`.

End-to-end manual flow:

1. Visit `/` signed-out → landing page renders.
2. Click `Sign in with GitHub` → GitHub consent → callback →
   redirect to `/` with playground visible, avatar in header.
3. Paste a source, click `Save` → redirect to
   `/s/<slug>`; reload to confirm it's persisted.
4. Log out; reload the same `/s/<slug>` URL → source renders,
   Save is disabled with "Sign in to save".
5. Incognito window (no session): `curl -X PUT
   /api/shaders/<slug> -d '{"source":"x"}'` → 401. Log in as a
   different account and repeat → 403.
6. Visit `/u/<login>` → lists the saved shader.
7. Tamper with the `__wyn_session` cookie in DevTools → next
   load treats you as logged out.
8. Oversize source (>256 KiB) → 413 from the save action.

Run `npm run typecheck` and `npm run build` to catch regressions.
The wyn-core test suite is not affected.

## Out of scope / flagged for later

- Fork / "Save as new" from a non-owned shader.
- Version history / diffs / revert.
- Per-user and per-IP rate limits (Cloudflare Rate Limiting or
  a KV counter — add once there's traffic).
- User-chosen shader titles + descriptions (column exists, UI
  deferred).
- Account deletion / data export.
- Public shader discovery / search / trending list.
- Sliding session expiry refresh.
- Multiple OAuth providers.
