// /s/:slug — view (and optionally edit) a saved shader.

import type { Route } from "./+types/shader";
import { Playground } from "~/components/Playground";
import { getShader } from "~/lib/db.server";
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
  const session = await getSession(request, env);
  const isOwner = session?.userId === shader.owner_id;

  return {
    slug: shader.slug,
    source: shader.source,
    isOwner,
    isSignedIn: !!session,
  };
}

export default function ShaderRoute({ loaderData }: Route.ComponentProps) {
  const { slug, source, isOwner, isSignedIn } = loaderData;
  const saveDisabledReason = !isSignedIn
    ? "Sign in to save"
    : "Fork coming soon";
  return (
    <Playground
      initialSource={source}
      slug={slug}
      canSave={isOwner}
      saveDisabledReason={saveDisabledReason}
    />
  );
}
