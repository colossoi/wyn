import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  index("routes/home.tsx"),
  route("new", "routes/new.tsx"),
  route("popular", "routes/popular.tsx"),
  route("s/:slug", "routes/shader.tsx"),
  route("u/:login", "routes/user.tsx"),
  route("auth/github", "routes/auth.github.tsx"),
  route("auth/github/callback", "routes/auth.callback.tsx"),
  route("auth/logout", "routes/auth.logout.tsx"),
  route("api/shaders", "routes/api.shaders.tsx"),
  route("api/shaders/:slug", "routes/api.shader.tsx"),
  route("api/featured", "routes/api.featured.tsx"),
  route("api/reports", "routes/api.reports.tsx"),
] satisfies RouteConfig;
