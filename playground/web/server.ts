const PORT = 8080;

const MIME_TYPES: Record<string, string> = {
  ".html": "text/html",
  ".css": "text/css",
  ".js": "application/javascript",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};

function getMimeType(path: string): string {
  const ext = path.substring(path.lastIndexOf("."));
  return MIME_TYPES[ext] || "application/octet-stream";
}

async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);
  let path = url.pathname;

  // Default to index.html
  if (path === "/") {
    path = "/index.html";
  }

  const filePath = `./static${path}`;

  try {
    const file = await Deno.readFile(filePath);
    const contentType = getMimeType(path);

    return new Response(file, {
      headers: {
        "Content-Type": contentType,
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
      },
    });
  } catch {
    return new Response("Not Found", { status: 404 });
  }
}

console.log(`Wyn Playground running at http://localhost:${PORT}/`);
Deno.serve({ port: PORT }, handler);
