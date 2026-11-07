# Codevember 2026 / 01 / Fold

An endless rainbow plastic sheet, simulated as cloth and rendered with three.js. Drag a fold to pull it around, or press **Space** to pause. Open **Settings** to adjust the material, simulation, and color grading.

Startup and **Reset simulation** begin with a straight hanging sheet whose lower edge just touches the floor. New material feeds in from above and starts folding on contact.

Plain JavaScript and import maps. No build step or npm dependencies. Three.js is pinned to **0.187.0**, including its addons and bundled lil-gui; modules load from jsDelivr, so an internet connection is required.

## Run locally

Use Python 3.10 or newer:

```sh
python3 scripts/serve.py
```

Open **http://localhost:8000** in a browser with WebGPU enabled. The development server binds to localhost and disables caching so edits appear on reload. Use `--port 8002` if port 8000 is occupied.

You can also run `npm start`, or serve the project with another static HTTP server. For hosting, copy `index.html`, `style.css`, and the root JavaScript files to an HTTPS static host. Keep their relative paths intact; opening `index.html` through `file://` is not supported.

## Implementation

- `main.js` creates the studio, fixed side camera, GUI, and render passes.
- `cloth-simulation.js` runs CPU XPBD at 120 steps per second, with stretch, shear, bending, particle contact, floor/wall constraints, and soft patch grabbing.
- `sheet.js` interpolates the simulation into a smooth shell with rounded edges and the selected thickness.
- `cloth-drag.js` raycasts pointer input onto a camera-facing drag plane.
- `viewport-walls.js` derives screen-edge collision planes and an adaptive cloth-length budget. The camera looks along X; front/back walls sit at X = ±2.85. Old material fades and sinks before rows are recycled.
- `plastic.js` shades the rainbow shell and implements refraction through `backdropNode`. An opaque depth/normal pass and a peeled depth/color pass supply the next visible surface. Roughness selects blurred mip levels.
- `grading.js` applies display-space grading after ACES filmic tone mapping. GTAO and denoising precede tone mapping; FXAA follows grading.

The default renderer caps device pixel ratio at 1. The Resolution setting multiplies that by 0.65, 1, or 1.5. GTAO uses full resolution by default. The material does not use SSR or OIT.

This is an interactive visual experiment: particle-based self-contact and single-layer screen-space refraction are approximations. Tight folds can intersect, and refraction cannot recover surfaces outside the view or behind additional occluded layers. The simulator's fixed timesteps make its motion independent of the capture rate; rendered pixels can still differ between GPU/browser implementations.

## Tests

With Node.js 22 or newer:

```sh
npm test
```

The tests cover fixed-step consistency, feeding, contact, thickness clearance, screen/depth walls, recycling during resizing, and grabbing. The optional capture receiver tests use Python's standard library:

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Frame-by-frame video export

Install `ffmpeg` and `ffprobe` on PATH. Start the local web server, then start a separate capture receiver:

```sh
python3 scripts/capture-server.py exports/fold.mp4
```

Omit the filename to generate a timestamped name. Existing videos and partial files are never overwritten. `--previews` saves selected PNG frames beside the MP4. If the page is served on another port, pass its origin, for example `--origin http://localhost:8002`.

In a **separate tab** of the experiment, set the browser's responsive viewport to **1024 × 1024** and leave Rendering → Resolution at **Medium**. Keep that viewport fixed until capture finishes. Run this in the browser console:

```js
const { captureVideo } = await import('./scripts/capture-video.js');
await captureVideo();
```

The default export is 60 seconds at 30 fps. It starts at simulation time zero, with no warm-up. Drags begin at 10, 20, 30, 40, and 50 seconds, using different visible patches and directions. At 10 and 30 seconds, the falling section is pulled back and forth in three 0.8-second strokes; the other drags each last 0.8 seconds. The video UI is enlarged by √2 with the same edge spacing.

Capture advances the simulation, renders, snapshots a lossless PNG, and waits for the receiver's acknowledgement before proceeding. FFmpeg consumes those ordered frames without real-time recording or frame-rate conversion. The receiver decodes the finished MP4 to verify dimensions, duration, and frame count, and writes a JSON report containing source-frame hashes and drag timings. Images that happen to be identical are still preserved as separate frames.

For a short check, use `await captureVideo({ duration: 13 })`. Only drags that fit fully within the duration are included. To inspect progress, evaluate `window.captureProgress`; to cancel, call `window.captureProgress.cancel()`. The interactive render loop and UI are restored after completion, cancellation, or failure. Restart the receiver with a new filename for another capture.

Generated videos, reports, preview frames, and local reference footage are excluded by `.gitignore` and are not needed to run the experiment.

## References

The cloth solver follows the compliant constraint formulation in [XPBD: Position-Based Simulation of Compliant Constrained Dynamics](https://matthias-research.github.io/pages/publications/XPBD.pdf). Rendering uses [three.js](https://threejs.org/) and its included addons.
