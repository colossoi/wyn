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
        <nav className="header-nav">
          {/* `reloadDocument` so clicking from another /p/new view — or
              from /p/s/<slug> with the same Playground component still
              mounted — always drops into a fresh editor. Without the
              full reload, Playground's once-only init effect would
              leave the prior source loaded. */}
          <Link to="/p/new" reloadDocument className="header-nav-link">
            New shader
          </Link>
          <Link to="/p/popular" className="header-nav-link">
            Popular
          </Link>
          <a href="/spec/" className="header-nav-link">
            Spec
          </a>
          {/* Future: Docs, Tutorials, etc. drop in here. */}
        </nav>
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
        <button type="submit" className="btn-ghost">Sign out</button>
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
