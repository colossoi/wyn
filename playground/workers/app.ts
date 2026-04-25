import { createRequestHandler } from "react-router";
import type { Env } from "./env";

// Dynamic import of the React Router server-build virtual module — the
// shape the @cloudflare/vite-plugin's React Router integration expects.
// A static `import * as build from "virtual:..."` confuses the worker
// bundler at deploy time (it tries to resolve the virtual module
// without going through the SSR environment first).
const handler = createRequestHandler(
  () => import("virtual:react-router/server-build" as string),
  import.meta.env.MODE,
);

export default {
  fetch(request: Request, env: Env, ctx: ExecutionContext) {
    return handler(request, { cloudflare: { env, ctx } });
  },
} satisfies ExportedHandler<Env>;
