// Landing page shown at `/` when no session is present.
//
// A WebGL2 fragment shader fills the viewport behind the hero — same
// visual family as a Wyn fragment shader, so the page doubles as a
// demo of what the playground produces. Falls back to a CSS gradient
// if WebGL isn't available.

import { useEffect, useRef } from "react";

export function Landing() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const cleanup = startBackgroundShader(canvas);
    return cleanup;
  }, []);

  return (
    <main className="landing">
      <canvas ref={canvasRef} className="landing-bg" aria-hidden="true" />
      <div className="landing-overlay" />
      <section className="landing-hero">
        <div className="landing-eyebrow">A GPU language for the browser</div>
        <h1 className="landing-title">
          Write shaders in <span className="landing-title-accent">Wyn</span>.
          <br />
          Run them on your GPU.
        </h1>
        <p className="landing-tagline">
          Wyn is a Futhark-inspired language that compiles to WGSL and SPIR-V.
          The playground gives you a live editor, multi-stage pipeline
          visualization, and shareable links — all in your browser.
        </p>
        <div className="landing-cta-row">
          <a href="/auth/github" className="landing-cta landing-cta-primary">
            Sign in with GitHub
          </a>
          <a href="#features" className="landing-cta landing-cta-secondary">
            Learn more
          </a>
        </div>
        <p className="landing-note">
          Viewing a shared shader doesn't require an account — sign in is only
          needed to save your own.
        </p>
      </section>

      <section id="features" className="landing-features">
        <div className="landing-feature">
          <div className="landing-feature-icon">⚡</div>
          <h3>Browser-native</h3>
          <p>
            The compiler itself runs in WebAssembly — no server round-trip,
            no install, no CLI to set up.
          </p>
        </div>
        <div className="landing-feature">
          <div className="landing-feature-icon">🧩</div>
          <h3>Multi-stage pipelines</h3>
          <p>
            Write one file. Wyn automatically parallelizes and splits
            fragment-invariant reductions into compute pre-passes.
          </p>
        </div>
        <div className="landing-feature">
          <div className="landing-feature-icon">🔗</div>
          <h3>Share with a link</h3>
          <p>
            Every saved shader gets a permanent URL anyone can open — no
            account required to view.
          </p>
        </div>
      </section>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Background shader — a WebGL2 fragment that paints orbiting ripple
// centers through an Iñigo Quilez cosine palette. Runs at half resolution
// and gives up silently if WebGL2 isn't available.
// ---------------------------------------------------------------------------

function startBackgroundShader(canvas: HTMLCanvasElement): () => void {
  const gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
  if (!gl) {
    // No WebGL2 — the CSS gradient fallback is already visible behind.
    return () => {};
  }

  const vertexSrc = `#version 300 es
    in vec2 a_pos;
    void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }
  `;
  const fragmentSrc = `#version 300 es
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;
    out vec4 fragColor;

    vec3 palette(float t) {
      vec3 a = vec3(0.5);
      vec3 b = vec3(0.5);
      vec3 c = vec3(1.0);
      vec3 d = vec3(0.15, 0.35, 0.60);
      return a + b * cos(6.28318 * (c * t + d));
    }

    void main() {
      vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
      vec2 c1 = vec2(cos(u_time * 0.37) * 0.7, sin(u_time * 0.53) * 0.5);
      vec2 c2 = vec2(cos(u_time * 0.71 + 2.1) * 0.6, sin(u_time * 0.29) * 0.6);
      vec2 c3 = vec2(sin(u_time * 0.47 + 4.2) * 0.8, cos(u_time * 0.61) * 0.5);
      float d1 = length(uv - c1);
      float d2 = length(uv - c2);
      float d3 = length(uv - c3);
      float w = sin(d1 * 8.0 - u_time * 0.9)
              + sin(d2 * 6.0 - u_time * 0.6)
              + sin(d3 * 10.0 + u_time * 0.7);
      vec3 col = palette(w * 0.12 + 0.45);
      // Desaturate + dim for legibility behind hero text.
      float lum = dot(col, vec3(0.299, 0.587, 0.114));
      col = mix(vec3(lum), col, 0.55) * 0.55;
      fragColor = vec4(col, 1.0);
    }
  `;

  function compile(type: number, src: string): WebGLShader | null {
    const sh = gl!.createShader(type);
    if (!sh) return null;
    gl!.shaderSource(sh, src);
    gl!.compileShader(sh);
    if (!gl!.getShaderParameter(sh, gl!.COMPILE_STATUS)) {
      gl!.deleteShader(sh);
      return null;
    }
    return sh;
  }

  const vs = compile(gl.VERTEX_SHADER, vertexSrc);
  const fs = compile(gl.FRAGMENT_SHADER, fragmentSrc);
  if (!vs || !fs) return () => {};
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    return () => {};
  }

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 3, -1, -1, 3]),
    gl.STATIC_DRAW,
  );
  const posLoc = gl.getAttribLocation(prog, "a_pos");
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

  const resLoc = gl.getUniformLocation(prog, "u_resolution");
  const timeLoc = gl.getUniformLocation(prog, "u_time");

  let raf = 0;
  let running = true;
  const start = performance.now();

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = Math.floor(window.innerWidth * dpr * 0.5);
    const h = Math.floor(window.innerHeight * dpr * 0.5);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
  };
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);

  const tick = () => {
    if (!running) return;
    const t = (performance.now() - start) / 1000;
    gl.useProgram(prog);
    gl.uniform2f(resLoc, canvas.width, canvas.height);
    gl.uniform1f(timeLoc, t);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);

  return () => {
    running = false;
    cancelAnimationFrame(raf);
    ro.disconnect();
    gl.deleteProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    gl.deleteBuffer(buf);
    gl.deleteVertexArray(vao);
  };
}
