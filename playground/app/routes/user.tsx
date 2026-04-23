// /u/:login — list of shaders owned by a given user. Dead-simple list for
// this pass; richer UI (thumbnails, descriptions) deferred.

import type { Route } from "./+types/user";
import { Link } from "react-router";
import { getUserByLogin, listShadersByOwner } from "~/lib/db.server";

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
      updatedAt: s.updated_at,
    })),
  };
}

export default function UserRoute({ loaderData }: Route.ComponentProps) {
  const { user, shaders } = loaderData;

  return (
    <main className="user-page">
      <header className="user-header">
        {user.avatarUrl && (
          <img
            src={user.avatarUrl}
            alt=""
            width={48}
            height={48}
            className="user-avatar"
          />
        )}
        <h1>{user.login}</h1>
      </header>
      {shaders.length === 0 ? (
        <p className="user-empty">No saved shaders yet.</p>
      ) : (
        <ul className="shader-list">
          {shaders.map((s) => (
            <li key={s.slug}>
              <Link to={`/s/${s.slug}`}>
                <span className="shader-slug">{s.slug}</span>
                {s.title && <span className="shader-title">{s.title}</span>}
                <span className="shader-updated">
                  {new Date(s.updatedAt * 1000).toLocaleString()}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
