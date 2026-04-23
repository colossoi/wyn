// `/` — logged-out viewers see the landing page; logged-in users go straight
// to a blank playground (no slug, save-enabled).

import type { Route } from "./+types/home";
import { Landing } from "~/components/Landing";
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
  const session = await getSession(request, context.cloudflare.env);
  return { signedIn: !!session };
}

export default function Home({ loaderData }: Route.ComponentProps) {
  if (!loaderData.signedIn) {
    return <Landing />;
  }
  return (
    <Playground
      initialSource={null}
      slug={null}
      canSave={true}
    />
  );
}
