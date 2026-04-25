// `/` — logged-out viewers see the landing page; logged-in users go straight
// to a blank playground (no slug, save-enabled).

import type { Route } from "./+types/home";
import { Landing, type FeaturedShader } from "~/components/Landing";
import { Playground } from "~/components/Playground";
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
  // Surface the most recently updated public shader as the landing-page
  // "Featured shader" — auto-rotates to whatever someone saved last.
  // Logged-in viewers don't see the landing, so skip the query.
  let featured: FeaturedShader | null = null;
  if (!session) {
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
  return { signedIn: !!session, featured };
}

export default function Home({ loaderData }: Route.ComponentProps) {
  if (!loaderData.signedIn) {
    return <Landing featured={loaderData.featured} />;
  }
  return (
    <Playground
      initialSource={null}
      slug={null}
      canSave={true}
    />
  );
}
