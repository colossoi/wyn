// Cloudflare bindings exposed to the Worker + React Router loaders/actions.
//
// The playground's SSR layer accesses these via
// `context.cloudflare.env` in every loader / action. Local dev is driven by
// `wrangler dev` with `--local` bindings; prod is driven by wrangler's
// `wrangler secret put` + `wrangler d1 create` / `wrangler kv namespace create`
// entries in `wrangler.jsonc`.

export interface Env {
  // D1 (SQLite): users + shaders tables. Migration in `migrations/0001_init.sql`.
  DB: D1Database;

  // KV namespace for session records. Key `sess:<id>` → JSON of
  // `{ userId, login, avatarUrl, createdAt }` with a 30-day TTL.
  SESSIONS: KVNamespace;

  // GitHub OAuth App credentials. The client ID is public-safe and lives
  // in `wrangler.jsonc :: vars`; the secret is injected via
  // `wrangler secret put GITHUB_CLIENT_SECRET`.
  GITHUB_CLIENT_ID: string;
  GITHUB_CLIENT_SECRET: string;

  // HMAC key (UTF-8 string) used to sign the short-lived `__oauth_state`
  // cookie. Injected via `wrangler secret put SESSION_COOKIE_SECRET`.
  SESSION_COOKIE_SECRET: string;

  // Comma-separated list of GitHub logins that get admin privileges
  // (e.g., managing the featured-shaders table). Case-insensitive.
  // Public-safe — lives in `wrangler.jsonc :: vars`.
  ADMIN_LOGINS: string;
}

declare module "react-router" {
  interface AppLoadContext {
    cloudflare: {
      env: Env;
      ctx: ExecutionContext;
    };
  }
}
