// D1 query helpers + slug generation.

import type { Env } from "../../workers/env";

export interface UserRow {
  id: number;
  login: string;
  avatar_url: string | null;
  created_at: number;
  updated_at: number;
}

export interface ShaderRow {
  slug: string;
  owner_id: number;
  title: string | null;
  source: string;
  thumbnail: string | null;
  created_at: number;
  updated_at: number;
}

/** Lightweight row for listing; excludes `source` (potentially large). */
export interface ShaderListRow {
  slug: string;
  owner_id: number;
  title: string | null;
  thumbnail: string | null;
  created_at: number;
  updated_at: number;
}

// ----------------------------------------------------------------------------
// Users
// ----------------------------------------------------------------------------

// Upsert the user row that mirrors the GitHub profile. Called from the
// OAuth callback. `login` and `avatar_url` may change on GitHub over
// time; `github_id` is stable. Returns the *internal* user id (the
// autoincrement primary key) — distinct from `github_id` for users
// created post-migration. Caller stores the internal id in the session.
export async function upsertUser(
  env: Env,
  user: { id: number; login: string; avatarUrl: string | null },
): Promise<number> {
  await env.DB.prepare(
    `INSERT INTO users (github_id, login, avatar_url)
     VALUES (?, ?, ?)
     ON CONFLICT(github_id) DO UPDATE SET
       login      = excluded.login,
       avatar_url = excluded.avatar_url,
       updated_at = unixepoch()`,
  )
    .bind(user.id, user.login, user.avatarUrl)
    .run();
  const row = await env.DB.prepare(
    `SELECT id FROM users WHERE github_id = ? LIMIT 1`,
  )
    .bind(user.id)
    .first<{ id: number }>();
  if (!row) throw new Error("upsertUser: row vanished after insert");
  return row.id;
}

// Find-or-create the user row keyed by email. The `login` for new
// rows is derived from the email local part and de-duplicated against
// the existing `login` UNIQUE column with a short random suffix.
// Returns { id, login } — id is stored in the session, login is used
// for /u/:login URLs.
export async function upsertUserByEmail(
  env: Env,
  email: string,
): Promise<{ id: number; login: string }> {
  const existing = await env.DB.prepare(
    `SELECT id, login FROM users WHERE email = ? COLLATE NOCASE LIMIT 1`,
  )
    .bind(email)
    .first<{ id: number; login: string }>();
  if (existing) return existing;

  for (let attempt = 0; attempt < 6; attempt++) {
    const candidate = candidateLogin(email, attempt);
    try {
      const result = await env.DB.prepare(
        `INSERT INTO users (email, login) VALUES (?, ?)`,
      )
        .bind(email, candidate)
        .run();
      const id = (result as unknown as { meta?: { last_row_id?: number } }).meta
        ?.last_row_id;
      if (typeof id !== "number") {
        throw new Error("upsertUserByEmail: D1 did not return last_row_id");
      }
      return { id, login: candidate };
    } catch (err) {
      const msg = (err as Error).message ?? "";
      if (/unique|constraint/i.test(msg)) continue;
      throw err;
    }
  }
  throw new Error("upsertUserByEmail: exhausted login candidates");
}

// Slugify the email's local part down to [a-z0-9-]; on retry, append a
// 4-char random suffix so the next candidate has a fresh address space.
function candidateLogin(email: string, attempt: number): string {
  const at = email.indexOf("@");
  const local = (at > 0 ? email.slice(0, at) : email).toLowerCase();
  const slug = local.replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "user";
  const trimmed = slug.slice(0, 24);
  if (attempt === 0) return trimmed;
  const bytes = new Uint8Array(2);
  crypto.getRandomValues(bytes);
  const suffix = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${trimmed}-${suffix}`;
}

export async function getUserByLogin(env: Env, login: string): Promise<UserRow | null> {
  const row = await env.DB.prepare(
    `SELECT id, login, avatar_url, created_at, updated_at
     FROM users WHERE login = ? COLLATE NOCASE LIMIT 1`,
  )
    .bind(login)
    .first<UserRow>();
  return row ?? null;
}

// ----------------------------------------------------------------------------
// Shaders
// ----------------------------------------------------------------------------

export async function getShader(env: Env, slug: string): Promise<ShaderRow | null> {
  const row = await env.DB.prepare(
    `SELECT slug, owner_id, title, source, thumbnail, created_at, updated_at
     FROM shaders WHERE slug = ? LIMIT 1`,
  )
    .bind(slug)
    .first<ShaderRow>();
  return row ?? null;
}

export async function listShadersByOwner(
  env: Env,
  ownerId: number,
): Promise<ShaderListRow[]> {
  const result = await env.DB.prepare(
    `SELECT slug, owner_id, title, thumbnail, created_at, updated_at
     FROM shaders WHERE owner_id = ? ORDER BY updated_at DESC`,
  )
    .bind(ownerId)
    .all<ShaderListRow>();
  return result.results ?? [];
}

// Insert a new shader with a freshly generated slug. Retries on the
// astronomically unlikely UNIQUE collision.
export async function createShader(
  env: Env,
  ownerId: number,
  source: string,
  thumbnail: string | null = null,
  title: string | null = null,
): Promise<string> {
  for (let attempt = 0; attempt < 5; attempt++) {
    const slug = generateSlug();
    try {
      await env.DB.prepare(
        `INSERT INTO shaders (slug, owner_id, source, thumbnail, title) VALUES (?, ?, ?, ?, ?)`,
      )
        .bind(slug, ownerId, source, thumbnail, title)
        .run();
      return slug;
    } catch (err) {
      // D1 surfaces UNIQUE violations as errors with "UNIQUE" / "constraint"
      // in the message. Retry on those, re-throw anything else.
      const msg = (err as Error).message ?? "";
      if (/unique|constraint/i.test(msg)) continue;
      throw err;
    }
  }
  throw new Error("createShader: exhausted slug retries");
}

/** Update a saved shader. `undefined` for either optional field means
 *  "leave the column untouched"; explicit `null` clears it. The
 *  per-column UPDATE assembly keeps the SQL surface predictable. */
export async function updateShaderSource(
  env: Env,
  slug: string,
  source: string,
  thumbnail: string | null | undefined = undefined,
  title: string | null | undefined = undefined,
): Promise<void> {
  const sets: string[] = ["source = ?", "updated_at = unixepoch()"];
  const binds: (string | null)[] = [source];
  if (thumbnail !== undefined) {
    sets.push("thumbnail = ?");
    binds.push(thumbnail);
  }
  if (title !== undefined) {
    sets.push("title = ?");
    binds.push(title);
  }
  binds.push(slug);
  await env.DB.prepare(`UPDATE shaders SET ${sets.join(", ")} WHERE slug = ?`)
    .bind(...binds)
    .run();
}

// ----------------------------------------------------------------------------
// Slug generator
// ----------------------------------------------------------------------------

// 8 chars from a 62-char alphabet → 62^8 ≈ 2.18e14 possible slugs.
// Collision probability at 1M shaders: ~2.3e-3, per-insert retry < 1e-14.
const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

export function generateSlug(): string {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  let out = "";
  for (const b of bytes) {
    out += ALPHABET[b % ALPHABET.length];
  }
  return out;
}

// ----------------------------------------------------------------------------
// Featured shaders
// ----------------------------------------------------------------------------

export interface FeaturedRow {
  slug: string;
  title: string | null;
  thumbnail: string | null;
  rank: number;
}

/** Fetch up to `limit` featured shaders by `rank` ascending, joined to
 *  the shaders table for title + thumbnail. */
export async function listFeaturedShaders(
  env: Env,
  limit: number = 10,
): Promise<FeaturedRow[]> {
  const result = await env.DB.prepare(
    `SELECT s.slug, s.title, s.thumbnail, f.rank
     FROM featured_shaders f
     JOIN shaders s ON s.slug = f.slug
     ORDER BY f.rank ASC, f.added_at DESC
     LIMIT ?`,
  )
    .bind(limit)
    .all<FeaturedRow>();
  return result.results ?? [];
}

/** Top-ranked featured shader, or null when nothing is featured. */
export async function getTopFeaturedShader(env: Env): Promise<FeaturedRow | null> {
  const rows = await listFeaturedShaders(env, 1);
  return rows[0] ?? null;
}

/** Promote (or re-rank) a shader as featured. Idempotent — re-calling
 *  with the same slug overwrites the rank. Errors if the slug doesn't
 *  exist in `shaders` (FK constraint). */
export async function setFeatured(
  env: Env,
  slug: string,
  rank: number,
  addedBy: number,
): Promise<void> {
  await env.DB.prepare(
    `INSERT INTO featured_shaders (slug, rank, added_by) VALUES (?, ?, ?)
     ON CONFLICT(slug) DO UPDATE SET
       rank      = excluded.rank,
       added_at  = unixepoch(),
       added_by  = excluded.added_by`,
  )
    .bind(slug, rank, addedBy)
    .run();
}

export async function removeFeatured(env: Env, slug: string): Promise<void> {
  await env.DB.prepare(`DELETE FROM featured_shaders WHERE slug = ?`).bind(slug).run();
}

// ----------------------------------------------------------------------------
// Shader views (popularity counter)
// ----------------------------------------------------------------------------

export interface PopularRow {
  slug: string;
  title: string | null;
  thumbnail: string | null;
  views: number;
}

/** Increment the view count for a shader. Idempotent on missing rows
 *  (UPSERT with `views = 1` on first hit). Caller should fire-and-
 *  forget via `ctx.waitUntil` so the page render isn't blocked on the
 *  D1 write. */
export async function incrementShaderView(env: Env, slug: string): Promise<void> {
  await env.DB.prepare(
    `INSERT INTO shader_views (slug, views) VALUES (?, 1)
     ON CONFLICT(slug) DO UPDATE SET
       views     = views + 1,
       last_view = unixepoch()`,
  )
    .bind(slug)
    .run();
}

/** Top `limit` shaders by view count, joined to the shaders table for
 *  display metadata. LEFT JOIN so shaders that have never been viewed
 *  still appear (with views=0) when the table is sparse — the
 *  `idx_shader_views_count` index keeps the sort cheap. */
export async function listPopularShaders(
  env: Env,
  limit: number = 24,
): Promise<PopularRow[]> {
  const result = await env.DB.prepare(
    `SELECT s.slug, s.title, s.thumbnail, COALESCE(v.views, 0) as views
     FROM shaders s
     LEFT JOIN shader_views v ON s.slug = v.slug
     ORDER BY COALESCE(v.views, 0) DESC, s.updated_at DESC
     LIMIT ?`,
  )
    .bind(limit)
    .all<PopularRow>();
  return result.results ?? [];
}

// ----------------------------------------------------------------------------
// Reports (user-submitted bug reports / feedback)
// ----------------------------------------------------------------------------

export type ReportCategory = "compiler" | "playground" | "other";

export interface NewReport {
  category: ReportCategory;
  comment: string;
  source: string;
  screenshot: string | null;
  userId: number | null;
  userLogin: string | null;
  shaderSlug: string | null;
  userAgent: string | null;
}

export async function createReport(env: Env, r: NewReport): Promise<number> {
  const result = await env.DB.prepare(
    `INSERT INTO reports
       (category, comment, source, screenshot, user_id, user_login, shader_slug, user_agent)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
  )
    .bind(
      r.category,
      r.comment,
      r.source,
      r.screenshot,
      r.userId,
      r.userLogin,
      r.shaderSlug,
      r.userAgent,
    )
    .run();
  // D1 surfaces the inserted rowid via `meta.last_row_id`.
  const id = (result as unknown as { meta?: { last_row_id?: number } }).meta
    ?.last_row_id;
  if (typeof id !== "number") {
    throw new Error("createReport: D1 did not return last_row_id");
  }
  return id;
}

// ----------------------------------------------------------------------------
// Admin check
// ----------------------------------------------------------------------------

/** Comma-separated logins from the `ADMIN_LOGINS` env var have admin
 *  privileges across the app — currently: managing the featured-shaders
 *  table. Case-insensitive match against the GitHub login. Whitespace
 *  around entries is tolerated. */
export function isAdmin(env: Env, login: string | null | undefined): boolean {
  if (!login) return false;
  const list = (env.ADMIN_LOGINS ?? "").split(",");
  const target = login.toLowerCase();
  return list.some((entry) => entry.trim().toLowerCase() === target);
}
