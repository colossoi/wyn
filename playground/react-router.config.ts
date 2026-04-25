import type { Config } from "@react-router/dev/config";

export default {
  ssr: true,
  // Keep `dist/` as the build root — wrangler.jsonc's assets.directory
  // points at dist/client/.
  buildDirectory: "dist",
  future: {
    // Required for the @cloudflare/vite-plugin integration: switches
    // React Router's vite plugin to the v8 environment API, which lets
    // the Cloudflare plugin own the SSR environment's outDir so the
    // worker bundle can resolve `virtual:react-router/server-build`
    // at deploy time. Without this, RR's plugin reads the SSR manifest
    // from a hardcoded path that doesn't match where the env actually
    // wrote it, and the build fails with ENOENT on manifest.json.
    v8_viteEnvironmentApi: true,
  },
} satisfies Config;
