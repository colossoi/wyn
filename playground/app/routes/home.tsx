// `/` — logged-out viewers see the landing page; logged-in viewers
// redirect to /new (the fresh-editor route), so the canonical
// "compose a new shader" URL stays the same regardless of session
// state.

import type { Route } from "./+types/home";
import { redirect } from "react-router";
import { Landing, type FeaturedShader } from "~/components/Landing";
import { getTopFeaturedShader } from "~/lib/db.server";
import { getSession } from "~/lib/session.server";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Wyn Playground" },
    {
      name: "description",
      content: "Browser-based playground for the Wyn shader language.",
    },
  ];
}

export async function loader({ request, context }: Route.LoaderArgs) {
  const env = context.cloudflare.env;
  const session = await getSession(request, env);
  if (session) {
    // Signed-in viewers skip the marketing page — bounce to the
    // fresh-editor route. Same target as the Header's "New shader"
    // link, so the post-login destination is consistent.
    throw redirect("/new");
  }

  // Featured-shader resolution: consult the curated `featured_shaders`
  // table first; if empty, fall back to the most recently updated
  // shader so the landing always has *something* to showcase.
  let featured: FeaturedShader | null = null;
  const top = await getTopFeaturedShader(env);
  if (top) {
    featured = { slug: top.slug, title: top.title, thumbnail: top.thumbnail };
  } else {
    const row = await env.DB.prepare(
      `SELECT slug, title, thumbnail FROM shaders
       ORDER BY updated_at DESC LIMIT 1`,
    )
      .first<{ slug: string; title: string | null; thumbnail: string | null }>();
    if (row) {
      featured = {
        slug: row.slug,
        title: row.title,
        thumbnail: row.thumbnail,
      };
    }
  }
  return { featured };
}

export default function Home({ loaderData }: Route.ComponentProps) {
  return <Landing featured={loaderData.featured} />;
}
