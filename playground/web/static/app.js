import wasmInit, {
  version,
  init_compiler,
  compile_with_ir,
  get_example_program,
} from "./pkg/wyn_wasm.js";

// DOM elements
const editorContainer = document.getElementById("editor");
const canvas = document.getElementById("canvas");
const output = document.getElementById("output");
const compileBtn = document.getElementById("compile-btn");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const fpsDisplay = document.getElementById("fps");
const loadingOverlay = document.getElementById("loading");
const tlcTree = document.getElementById("tlc-tree");
const initialMirTree = document.getElementById("initial-mir-tree");
const finalMirTree = document.getElementById("final-mir-tree");
const glslOutput = document.getElementById("glsl-output");

// CodeMirror editor instance
let editor = null;

// Error markers state
let errorMarkers = [];
let errorLineHandles = [];

// WebGL state
let gl = null;
let program = null;
let animationId = null;
let startTime = 0;
let frameCount = 0;
let lastFpsUpdate = 0;

// Vertex shader for fullscreen quad
const vertexShaderSource = `#version 300 es
in vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}`;

// Initialize CodeMirror editor
function initEditor() {
  editor = CodeMirror(editorContainer, {
    value: "",
    mode: null, // Plain text, no syntax highlighting
    theme: "monokai",
    lineNumbers: true,
    tabSize: 2,
    indentWithTabs: false,
    lineWrapping: false,
    autofocus: true,
    gutters: ["CodeMirror-linenumbers"],
    scrollbarStyle: "native",
  });

  // Ctrl/Cmd+Enter to compile
  editor.setOption("extraKeys", {
    "Ctrl-Enter": compileAndRun,
    "Cmd-Enter": compileAndRun,
  });
}

// Initialize tabs
function initTabs() {
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      // Deactivate all tabs
      tabs.forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

      // Activate clicked tab
      tab.classList.add("active");
      const tabId = tab.dataset.tab;
      document.getElementById(`tab-${tabId}`).classList.add("active");
    });
  });
}

// Initialize resize handle for output panel
function initResizeHandle() {
  const resizeHandle = document.getElementById("resize-handle");
  const outputPanel = document.querySelector(".output-panel");
  const previewPanel = document.querySelector(".preview-panel");

  let isResizing = false;
  let startY = 0;
  let startHeight = 0;

  resizeHandle.addEventListener("mousedown", (e) => {
    isResizing = true;
    startY = e.clientY;
    startHeight = outputPanel.offsetHeight;
    resizeHandle.classList.add("dragging");
    document.body.style.cursor = "ns-resize";
    document.body.style.userSelect = "none";
    e.preventDefault();
  });

  document.addEventListener("mousemove", (e) => {
    if (!isResizing) return;

    const deltaY = startY - e.clientY;
    const newHeight = Math.max(100, Math.min(startHeight + deltaY, previewPanel.offsetHeight - 100));
    outputPanel.style.height = `${newHeight}px`;
  });

  document.addEventListener("mouseup", () => {
    if (isResizing) {
      isResizing = false;
      resizeHandle.classList.remove("dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }
  });
}

// Clear all error markers
function clearErrors() {
  // Clear text markers
  for (const marker of errorMarkers) {
    marker.clear();
  }
  errorMarkers = [];

  // Clear line background highlights
  for (const handle of errorLineHandles) {
    editor.removeLineClass(handle, "background", "line-error");
  }
  errorLineHandles = [];
}

// Mark an error in the editor using structured location data
function markError(location) {
  if (!location) return;

  const startLine = location.start_line - 1; // CodeMirror is 0-indexed
  const startCol = location.start_col - 1;
  const endLine = location.end_line - 1;
  const endCol = location.end_col - 1;

  // Add line background highlight
  const lineHandle = editor.addLineClass(
    startLine,
    "background",
    "line-error",
  );
  errorLineHandles.push(lineHandle);

  // Add text marker for underline
  const marker = editor.markText(
    { line: startLine, ch: startCol },
    { line: endLine, ch: endCol },
    { className: "cm-error-underline" },
  );
  errorMarkers.push(marker);

  // Scroll to error line
  editor.scrollIntoView({ line: startLine, ch: 0 }, 100);
}

// Initialize WebGL
function initWebGL() {
  gl = canvas.getContext("webgl2");
  if (!gl) {
    setOutput("WebGL2 not supported", true);
    return false;
  }

  // Create fullscreen quad
  const positions = new Float32Array([
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    1,
  ]);

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

  return true;
}

// Compile shader
function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const error = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(error);
  }

  return shader;
}

// Create shader program
function createProgram(vertexSource, fragmentSource) {
  const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);

  const prog = gl.createProgram();
  gl.attachShader(prog, vertexShader);
  gl.attachShader(prog, fragmentShader);
  gl.linkProgram(prog);

  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const error = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(error);
  }

  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);

  return prog;
}

// Render frame
function render(time) {
  if (!program) return;

  const currentTime = (time - startTime) / 1000;

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.useProgram(program);

  // Set uniforms
  const resolutionLoc = gl.getUniformLocation(program, "iResolution");
  const timeLoc = gl.getUniformLocation(program, "iTime");
  const mouseLoc = gl.getUniformLocation(program, "iMouse");

  if (resolutionLoc) {
    gl.uniform3f(resolutionLoc, canvas.width, canvas.height, 1.0);
  }
  if (timeLoc) {
    gl.uniform1f(timeLoc, currentTime);
  }
  if (mouseLoc) {
    gl.uniform4f(mouseLoc, 0, 0, 0, 0);
  }

  // Set up vertex attribute
  const positionLoc = gl.getAttribLocation(program, "a_position");
  gl.enableVertexAttribArray(positionLoc);
  gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

  // Draw
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  // Update FPS
  frameCount++;
  if (time - lastFpsUpdate > 1000) {
    const fps = Math.round((frameCount * 1000) / (time - lastFpsUpdate));
    fpsDisplay.textContent = `${fps} FPS`;
    frameCount = 0;
    lastFpsUpdate = time;
  }

  animationId = requestAnimationFrame(render);
}

// Stop rendering
function stopRendering() {
  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
}

// Start rendering
function startRendering() {
  stopRendering();
  startTime = performance.now();
  frameCount = 0;
  lastFpsUpdate = startTime;
  animationId = requestAnimationFrame(render);
}

// Set output message
function setOutput(message, isError = false) {
  output.textContent = message;
  output.className = isError ? "error" : "success";
}

// Show error as a clickable item using structured error data
function showErrorItem(errorInfo) {
  output.innerHTML = "";
  output.className = "error";

  const item = document.createElement("div");
  item.className = "error-item";

  const location = errorInfo.location;
  if (location) {
    const locationDiv = document.createElement("div");
    locationDiv.className = "error-location";
    locationDiv.textContent =
      `Line ${location.start_line}, Column ${location.start_col}`;
    item.appendChild(locationDiv);

    // Click to jump to error
    item.addEventListener("click", () => {
      const line = location.start_line - 1;
      const col = location.start_col - 1;
      editor.setCursor({ line, ch: col });
      editor.focus();
      editor.scrollIntoView({ line, ch: col }, 100);
    });
  }

  const messageDiv = document.createElement("div");
  messageDiv.className = "error-message";
  messageDiv.textContent = errorInfo.message;
  item.appendChild(messageDiv);

  output.appendChild(item);
}

// Set status
function setStatus(status, text) {
  statusDot.className = `status-dot ${status}`;
  statusText.textContent = text;
}

// Wrap Shadertoy GLSL for WebGL2
function wrapForWebGL2(shadertoyGlsl) {
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

// Render a tree node recursively
function renderTreeNode(node, depth = 0) {
  const container = document.createElement("div");
  container.className = "tree-node";
  if (depth === 0) {
    container.style.marginLeft = "0";
  }

  const hasChildren = node.children && node.children.length > 0;

  const label = document.createElement("div");
  label.className = "tree-node-label";

  const toggle = document.createElement("span");
  toggle.className = `tree-toggle ${hasChildren ? "expanded" : "leaf"}`;
  label.appendChild(toggle);

  const text = document.createElement("span");
  text.className = "tree-text";
  text.textContent = node.name;
  label.appendChild(text);

  container.appendChild(label);

  if (hasChildren) {
    const childrenContainer = document.createElement("div");
    childrenContainer.className = "tree-children";

    for (const child of node.children) {
      childrenContainer.appendChild(renderTreeNode(child, depth + 1));
    }

    container.appendChild(childrenContainer);

    // Toggle collapse/expand
    label.addEventListener("click", (e) => {
      e.stopPropagation();
      const isExpanded = toggle.classList.contains("expanded");
      toggle.classList.toggle("expanded", !isExpanded);
      toggle.classList.toggle("collapsed", isExpanded);
      childrenContainer.classList.toggle("collapsed", isExpanded);
    });
  }

  return container;
}

// Render tree data into a container
function renderTree(container, nodes) {
  container.innerHTML = "";
  if (!nodes || nodes.length === 0) {
    container.textContent = "(empty)";
    return;
  }
  for (const node of nodes) {
    container.appendChild(renderTreeNode(node));
  }
}

// Compile and run
function compileAndRun() {
  const source = editor.getValue();

  // Clear previous errors
  clearErrors();

  setStatus("compiling", "Compiling...");
  compileBtn.disabled = true;

  try {
    // Compile Wyn to GLSL with IR trees
    const result = compile_with_ir(source);

    if (!result.success) {
      const errorInfo = result.error || {
        message: "Unknown compilation error",
        location: null,
      };
      markError(errorInfo.location);
      showErrorItem(errorInfo);
      setStatus("error", "Error");
      stopRendering();
      compileBtn.disabled = false;

      // Clear IR views on error
      tlcTree.innerHTML = "";
      initialMirTree.innerHTML = "";
      finalMirTree.innerHTML = "";
      glslOutput.textContent = "";
      return;
    }

    const shadertoyGlsl = result.glsl;
    const glslSource = wrapForWebGL2(shadertoyGlsl);
    console.log("=== Generated GLSL (wrapped for WebGL2) ===\n" + glslSource);

    // Update output tab
    setOutput("Compilation successful!", false);

    // Update IR tabs
    renderTree(tlcTree, result.tlc);
    renderTree(initialMirTree, result.initial_mir);
    renderTree(finalMirTree, result.final_mir);
    glslOutput.textContent = shadertoyGlsl;

    // Clean up old program
    if (program) {
      gl.deleteProgram(program);
      program = null;
    }

    // Create new program
    try {
      program = createProgram(vertexShaderSource, glslSource);
    } catch (glError) {
      showErrorItem({
        message: `GLSL error: ${glError.message}`,
        location: null,
      });
      setStatus("error", "GLSL Error");
      stopRendering();
      compileBtn.disabled = false;
      return;
    }

    setStatus("ready", "Running");
    startRendering();
  } finally {
    compileBtn.disabled = false;
  }
}

// Initialize
async function main() {
  try {
    // Initialize WASM with cache-busting nonce
    await wasmInit(`./pkg/wyn_wasm_bg.wasm?v=${Date.now()}`);

    // Log compiler version
    console.log("Wyn compiler version:", version());

    // Pre-initialize the compiler cache (parses prelude files)
    if (!init_compiler()) {
      throw new Error("Failed to initialize compiler");
    }

    // Initialize CodeMirror editor
    initEditor();

    // Initialize tabs
    initTabs();

    // Initialize resize handle
    initResizeHandle();

    // Initialize WebGL
    if (!initWebGL()) {
      throw new Error("Failed to initialize WebGL");
    }

    // Load example program
    editor.setValue(get_example_program());

    // Set up event listeners
    compileBtn.addEventListener("click", compileAndRun);

    // Hide loading overlay
    loadingOverlay.classList.add("hidden");

    // Enable compile button
    compileBtn.disabled = false;
    setStatus("ready", "Ready");
    setOutput('Press "Compile & Run" or Ctrl+Enter to compile your shader.');

    // Auto-compile on load
    compileAndRun();
  } catch (error) {
    setOutput(`Initialization error: ${error.message}`, true);
    setStatus("error", "Error");
    loadingOverlay.classList.add("hidden");
  }
}

main();
