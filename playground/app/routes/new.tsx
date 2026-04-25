// /new — fresh editor. Save-enabled for signed-in users; signed-out
// viewers can still play with the editor, just can't persist.

import type { Route } from "./+types/new";
import { Playground } from "~/components/Playground";
import { getSession } from "~/lib/session.server";

export function meta() {
  return [
    { title: "New shader · Wyn Playground" },
    {
      name: "description",
      content: "Compose a new Wyn shader in the playground.",
    },
  ];
}

export async function loader({ request, context }: Route.LoaderArgs) {
  const session = await getSession(request, context.cloudflare.env);
  return { isSignedIn: !!session };
}

export default function NewShaderRoute({ loaderData }: Route.ComponentProps) {
  const { isSignedIn } = loaderData;
  return (
    <Playground
      initialSource={null}
      slug={null}
      canSave={isSignedIn}
      saveDisabledReason={isSignedIn ? undefined : "Sign in to save"}
    />
  );
}
