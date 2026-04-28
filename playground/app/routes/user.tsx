// /u/:login — list of shaders owned by a given user.
//
// Grid of cards with procedural cover art derived from the slug, title +
// slug, and both created/modified timestamps as relative-time with
// absolute tooltips. Matches the landing page's visual language.

import type { Route } from "./+types/user";
import { Link, useRouteLoaderData } from "react-router";
import { getUserByLogin, listShadersByOwner } from "~/lib/db.server";
import type { Session } from "~/lib/session.server";

export function meta({ params }: Route.MetaArgs) {
  return [{ title: `${params.login} · Wyn Playground` }];
}

export async function loader({ params, context }: Route.LoaderArgs) {
  const env = context.cloudflare.env;
  const user = await getUserByLogin(env, params.login);
  if (!user) {
    throw new Response("not found", { status: 404 });
  }
  const shaders = await listShadersByOwner(env, user.id);
  return {
    user: {
      login: user.login,
      avatarUrl: user.avatar_url,
    },
    shaders: shaders.map((s) => ({
      slug: s.slug,
      title: s.title,
      thumbnail: s.thumbnail,
      createdAt: s.created_at,
      updatedAt: s.updated_at,
    })),
  };
}

interface RootData {
  session: Session | null;
}

export default function UserRoute({ loaderData }: Route.ComponentProps) {
  const { user, shaders } = loaderData;
  const rootData = useRouteLoaderData("root") as RootData | undefined;
  const isOwner = rootData?.session?.login === user.login;

  return (
    <main className="user-page">
      <section className="user-hero">
        {user.avatarUrl && (
          <img
            src={user.avatarUrl}
            alt=""
            width={72}
            height={72}
            className="user-hero-avatar"
          />
        )}
        <div className="user-hero-text">
          <div className="user-hero-eyebrow">
            {isOwner ? "Your shaders" : "Shaders by"}
          </div>
          <h1 className="user-hero-title">{user.login}</h1>
          <div className="user-hero-count">
            {shaders.length === 0
              ? "No saved shaders yet"
              : `${shaders.length} saved shader${shaders.length === 1 ? "" : "s"}`}
          </div>
        </div>
        {isOwner && (
          <Link to="/" className="user-hero-cta">
            + New shader
          </Link>
        )}
      </section>

      {shaders.length === 0 ? (
        <EmptyState isOwner={isOwner} />
      ) : (
        <ul className="shader-grid">
          {shaders.map((s) => (
            <li key={s.slug}>
              <ShaderCard
                slug={s.slug}
                title={s.title}
                thumbnail={s.thumbnail}
                createdAt={s.createdAt}
                updatedAt={s.updatedAt}
              />
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}

// ---------------------------------------------------------------------------

interface ShaderCardProps {
  slug: string;
  title: string | null;
  thumbnail: string | null;
  createdAt: number;
  updatedAt: number;
}

function ShaderCard({ slug, title, thumbnail, createdAt, updatedAt }: ShaderCardProps) {
  const displayTitle = title?.trim() || "Untitled shader";
  const edited = createdAt !== updatedAt;
  return (
    <Link to={`/p/s/${slug}`} className="shader-card">
      <div className="shader-cover" style={thumbnail ? undefined : coverStyle(slug)}>
        {thumbnail && (
          <img src={thumbnail} alt="" className="shader-cover-img" loading="lazy" />
        )}
        <div className="shader-cover-slug">{slug}</div>
      </div>
      <div className="shader-card-body">
        <div className="shader-card-title">{displayTitle}</div>
        <dl className="shader-card-meta">
          <div>
            <dt>Created</dt>
            <dd title={absolute(createdAt)}>{relative(createdAt)}</dd>
          </div>
          <div>
            <dt>{edited ? "Modified" : "Not yet modified"}</dt>
            <dd title={absolute(updatedAt)}>
              {edited ? relative(updatedAt) : "—"}
            </dd>
          </div>
        </dl>
      </div>
    </Link>
  );
}

function EmptyState({ isOwner }: { isOwner: boolean }) {
  return (
    <div className="user-empty">
      <div className="user-empty-icon">✨</div>
      <h2>{isOwner ? "No shaders yet" : "Nothing to see here"}</h2>
      <p>
        {isOwner
          ? "Save a shader from the playground and it'll show up here with a shareable link."
          : "This user hasn't saved any public shaders yet."}
      </p>
      {isOwner && (
        <Link to="/" className="user-empty-cta">
          Open the playground
        </Link>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Cover-art generation: a deterministic three-color radial-gradient mix
// keyed off the slug, so each card has a unique but stable visual identity
// without needing a real thumbnail pipeline.
// ---------------------------------------------------------------------------

function hash32(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

function coverStyle(slug: string): React.CSSProperties {
  const h = hash32(slug);
  // Pick three analogous hues around a random base so colors harmonize.
  const baseHue = h % 360;
  const hue1 = baseHue;
  const hue2 = (baseHue + 40 + ((h >> 8) % 40)) % 360;
  const hue3 = (baseHue + 200 + ((h >> 16) % 60)) % 360;
  const x1 = 10 + ((h >> 3) % 40);
  const y1 = 10 + ((h >> 11) % 40);
  const x2 = 55 + ((h >> 7) % 35);
  const y2 = 55 + ((h >> 17) % 35);
  return {
    background: `
      radial-gradient(circle at ${x1}% ${y1}%, hsla(${hue1}, 80%, 65%, 0.95), transparent 60%),
      radial-gradient(circle at ${x2}% ${y2}%, hsla(${hue2}, 75%, 55%, 0.85), transparent 55%),
      linear-gradient(135deg, hsl(${hue3}, 40%, 22%) 0%, hsl(${(hue3 + 30) % 360}, 45%, 12%) 100%)
    `,
  };
}

// ---------------------------------------------------------------------------
// Time formatting. Renders a coarse relative string ("2 days ago") with
// the full local timestamp available on hover via `title`.
// ---------------------------------------------------------------------------

function relative(unixSeconds: number): string {
  const diffMs = Date.now() - unixSeconds * 1000;
  const sec = Math.floor(diffMs / 1000);
  if (sec < 45) return "just now";
  const min = Math.floor(sec / 60);
  if (min < 2) return "a minute ago";
  if (min < 60) return `${min} minutes ago`;
  const hr = Math.floor(min / 60);
  if (hr < 2) return "an hour ago";
  if (hr < 24) return `${hr} hours ago`;
  const day = Math.floor(hr / 24);
  if (day < 2) return "yesterday";
  if (day < 30) return `${day} days ago`;
  const mo = Math.floor(day / 30);
  if (mo < 2) return "a month ago";
  if (mo < 12) return `${mo} months ago`;
  const yr = Math.floor(mo / 12);
  return yr < 2 ? "a year ago" : `${yr} years ago`;
}

function absolute(unixSeconds: number): string {
  return new Date(unixSeconds * 1000).toLocaleString();
}
