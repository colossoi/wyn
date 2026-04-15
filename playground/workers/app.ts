import { createRequestHandler } from "react-router";

// The virtual module is provided by @cloudflare/vite-plugin's React Router integration.
// @ts-ignore - virtual module
import * as build from "virtual:react-router/server-build";

const handler = createRequestHandler(build, import.meta.env.MODE);

export default {
  fetch(request: Request, env: unknown, ctx: ExecutionContext) {
    return handler(request, { cloudflare: { env, ctx } });
  },
} satisfies ExportedHandler<unknown>;
