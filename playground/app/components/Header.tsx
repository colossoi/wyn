// Global header — logo + auth controls. Rendered from `root.tsx` so it's
// visible on every route.
//
// Reads the current session (if any) from the root route's loader data.

import { useRouteLoaderData, Link, Form } from "react-router";
import type { Session } from "~/lib/session.server";

interface RootData {
  session: Session | null;
}

export function Header() {
  const data = useRouteLoaderData("root") as RootData | undefined;
  const session = data?.session ?? null;

  return (
    <header>
      <Link to="/" className="logo">
        Wyn Playground
      </Link>
      <div className="header-right">
        {session ? <SignedIn session={session} /> : <SignedOut />}
      </div>
    </header>
  );
}

function SignedIn({ session }: { session: Session }) {
  return (
    <div className="header-user">
      <Link to={`/u/${session.login}`} className="user-chip">
        {session.avatarUrl && (
          <img
            src={session.avatarUrl}
            alt=""
            width={24}
            height={24}
            className="user-avatar"
          />
        )}
        <span>{session.login}</span>
      </Link>
      <Form method="post" action="/auth/logout">
        <button type="submit">Sign out</button>
      </Form>
    </div>
  );
}

function SignedOut() {
  return (
    <a href="/auth/github" className="signin-link">
      Sign in with GitHub
    </a>
  );
}
