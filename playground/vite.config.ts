import { reactRouter } from "@react-router/dev/vite";
import { cloudflare } from "@cloudflare/vite-plugin";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [
    // `viteEnvironment.name: "ssr"` tells the Cloudflare plugin to bundle
    // the worker entry into React Router's existing SSR environment
    // instead of spawning a separate "workers" environment. Without
    // this, the worker bundle can't resolve
    // `virtual:react-router/server-build` at deploy time.
    cloudflare({ viteEnvironment: { name: "ssr" } }),
    reactRouter(),
  ],
  resolve: {
    tsconfigPaths: true,
  },
});
