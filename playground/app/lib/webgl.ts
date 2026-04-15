// WebGL2 helpers for the shader preview.
// Pure functions + a minimal RAF-loop builder; no React in here.

const VERTEX_SHADER_SOURCE = `#version 300 es
in vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}`;

export function setupContext(canvas: HTMLCanvasElement): WebGL2RenderingContext | null {
  const gl = canvas.getContext("webgl2");
  if (!gl) return null;

  const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  return gl;
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("createShader returned null");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const error = gl.getShaderInfoLog(shader) || "unknown shader error";
    gl.deleteShader(shader);
    throw new Error(error);
  }
  return shader;
}

export function createProgram(gl: WebGL2RenderingContext, fragmentSource: string): WebGLProgram {
  const vertex = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER_SOURCE);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  if (!program) throw new Error("createProgram returned null");
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const error = gl.getProgramInfoLog(program) || "unknown link error";
    gl.deleteProgram(program);
    throw new Error(error);
  }
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  return program;
}

/** Wrap Shadertoy GLSL into a complete WebGL2 fragment shader. */
export function wrapForWebGL2(shadertoyGlsl: string): string {
  return `#version 300 es
precision highp float;

uniform vec3 iResolution;
uniform float iTime;
uniform vec4 iMouse;

out vec4 outColor;

${shadertoyGlsl}

void main() {
    vec4 fragColor;
    mainImage(fragColor, gl_FragCoord.xy);
    outColor = fragColor;
}
`;
}

export interface RenderLoop {
  stop: () => void;
}

export function startRenderLoop(
  gl: WebGL2RenderingContext,
  canvas: HTMLCanvasElement,
  program: WebGLProgram,
  onFps: (fps: number) => void,
): RenderLoop {
  let animationId: number | null = null;
  let startTime = performance.now();
  let frameCount = 0;
  let lastFpsUpdate = startTime;

  const resolutionLoc = gl.getUniformLocation(program, "iResolution");
  const timeLoc = gl.getUniformLocation(program, "iTime");
  const mouseLoc = gl.getUniformLocation(program, "iMouse");
  const positionLoc = gl.getAttribLocation(program, "a_position");

  function tick(time: number) {
    const currentTime = (time - startTime) / 1000;

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(program);

    if (resolutionLoc) gl.uniform3f(resolutionLoc, canvas.width, canvas.height, 1.0);
    if (timeLoc) gl.uniform1f(timeLoc, currentTime);
    if (mouseLoc) gl.uniform4f(mouseLoc, 0, 0, 0, 0);

    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    frameCount++;
    if (time - lastFpsUpdate > 1000) {
      const fps = Math.round((frameCount * 1000) / (time - lastFpsUpdate));
      onFps(fps);
      frameCount = 0;
      lastFpsUpdate = time;
    }

    animationId = requestAnimationFrame(tick);
  }

  animationId = requestAnimationFrame(tick);

  return {
    stop() {
      if (animationId !== null) {
        cancelAnimationFrame(animationId);
        animationId = null;
      }
    },
  };
}
