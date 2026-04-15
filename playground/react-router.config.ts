import type { Config } from "@react-router/dev/config";

export default {
  ssr: true,
  // Align with @cloudflare/vite-plugin's default `dist/` output so the SSR
  // build can find the client manifest at dist/client/.vite/manifest.json.
  buildDirectory: "dist",
} satisfies Config;
