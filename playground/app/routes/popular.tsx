// /popular — top N shaders by view count, no pagination.
//
// View counts come from the `shader_views` table (incremented on every
// /s/:slug visit). LEFT JOIN'd to shaders so the grid always fills out
// even when the counter table is sparse — never-viewed shaders fall to
// the bottom by `updated_at DESC`.

import type { Route } from "./+types/popular";
import { Link } from "react-router";
import { listPopularShaders } from "~/lib/db.server";

const POPULAR_LIMIT = 24;

export function meta() {
  return [
    { title: "Popular shaders · Wyn Playground" },
    {
      name: "description",
      content: "The most-viewed shaders on the Wyn playground.",
    },
  ];
}

export async function loader({ context }: Route.LoaderArgs) {
  const shaders = await listPopularShaders(context.cloudflare.env, POPULAR_LIMIT);
  return { shaders };
}

export default function PopularRoute({ loaderData }: Route.ComponentProps) {
  const { shaders } = loaderData;
  return (
    <main className="user-page">
      <section className="user-hero">
        <div className="user-hero-text">
          <div className="user-hero-eyebrow">Popular shaders</div>
          <h1 className="user-hero-title">Most viewed</h1>
          <div className="user-hero-count">
            {shaders.length === 0
              ? "No shaders yet"
              : `Top ${shaders.length}`}
          </div>
        </div>
      </section>

      {shaders.length === 0 ? (
        <div className="user-empty">
          <div className="user-empty-icon">✨</div>
          <h2>Nothing to see here yet</h2>
          <p>Save a shader and share its link — the most-viewed ones land here.</p>
        </div>
      ) : (
        <ul className="shader-grid">
          {shaders.map((s) => (
            <li key={s.slug}>
              <PopularCard
                slug={s.slug}
                title={s.title}
                thumbnail={s.thumbnail}
                views={s.views}
              />
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}

interface PopularCardProps {
  slug: string;
  title: string | null;
  thumbnail: string | null;
  views: number;
}

function PopularCard({ slug, title, thumbnail, views }: PopularCardProps) {
  const displayTitle = title?.trim() || "Untitled shader";
  return (
    <Link to={`/s/${slug}`} className="shader-card">
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
            <dt>Views</dt>
            <dd>{formatViews(views)}</dd>
          </div>
        </dl>
      </div>
    </Link>
  );
}

function formatViews(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}K`;
  return `${(n / 1_000_000).toFixed(1)}M`;
}

// Same hash-derived gradient cover as user.tsx — kept in-line for now;
// can extract to a shared util if a third caller appears.
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
