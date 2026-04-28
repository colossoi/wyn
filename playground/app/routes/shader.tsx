// /p/s/:slug — view (and optionally edit) a saved shader.

import type { Route } from "./+types/shader";
import { Playground } from "~/components/Playground";
import {
  getShader,
  incrementShaderView,
  isAdmin,
  listFeaturedShaders,
} from "~/lib/db.server";
import { getSession } from "~/lib/session.server";

export function meta({ data }: Route.MetaArgs) {
  const slug = data?.slug ?? "";
  return [
    { title: `Wyn Playground · ${slug}` },
    {
      name: "description",
      content: `Shader ${slug} on the Wyn playground.`,
    },
  ];
}

export async function loader({ params, request, context }: Route.LoaderArgs) {
  const env = context.cloudflare.env;
  const shader = await getShader(env, params.slug);
  if (!shader) {
    throw new Response("not found", { status: 404 });
  }
  // Bump the visit counter without blocking the response. `waitUntil`
  // keeps the worker alive past the response so the D1 write can
  // finish; the page renders immediately. Counts include duplicate
  // tabs / refreshes — bot inflation is a TODO.
  context.cloudflare.ctx.waitUntil(incrementShaderView(env, shader.slug));

  const session = await getSession(request, env);
  const isOwner = session?.userId === shader.owner_id;
  const admin = isAdmin(env, session?.login ?? null);
  // Whether this shader is currently in the featured set — only
  // queried when an admin is viewing, since regular viewers don't
  // need the bit.
  let isFeatured = false;
  if (admin) {
    const featured = await listFeaturedShaders(env, 100);
    isFeatured = featured.some((f) => f.slug === shader.slug);
  }

  return {
    slug: shader.slug,
    source: shader.source,
    title: shader.title,
    isOwner,
    isSignedIn: !!session,
    isAdmin: admin,
    isFeatured,
  };
}

export default function ShaderRoute({ loaderData }: Route.ComponentProps) {
  const { slug, source, title, isOwner, isSignedIn, isAdmin, isFeatured } =
    loaderData;
  const saveDisabledReason = !isSignedIn
    ? "Sign in to save"
    : "Fork coming soon";
  return (
    <Playground
      initialSource={source}
      initialTitle={title}
      slug={slug}
      canSave={isOwner}
      saveDisabledReason={saveDisabledReason}
      adminControls={isAdmin ? { isFeatured } : null}
    />
  );
}
