import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ClothSimulation } from '../cloth-simulation.js';
import { viewportWallPlanes, clothLengthForViewport } from '../viewport-walls.js';

const small = () => new ClothSimulation({ columns: 6, maxRows: 48, width: 1.8, height: 3 });
const run = (sim, seconds, settings, frame = 1 / 60) => {
  for (let i = 0; i < Math.round(seconds / frame); i++) sim.advance(frame, settings);
};
const fed = () => { const sim = small(); run(sim, 2, { speed: 1 }); return sim; };

test('startup stays empty until feeding begins, then grows from above', () => {
  const sim = small();
  assert.equal(sim.activeRows, 0);
  assert.equal(sim.beginGrab(0), false);
  run(sim, 1, { speed: 0 });
  assert.equal(sim.activeRows, 0);
  assert.equal(sim.emittedRows, 0);
  run(sim, 0.5, { speed: 1 });
  assert.equal(sim.activeRows, sim.emittedRows + 2);
  assert.ok(sim.activeRows < Math.ceil(sim.height / sim.spacing));
  for (let i = 0; i < sim.activeRows * sim.stride; i++) {
    assert.ok(sim.positions[i * 3 + 1] > sim.height / 2, 'new cloth must still be above the floor');
  }
  assert.ok(sim.positions.every(Number.isFinite));
});

test('fixed-step cloth gives the same result at 30 and 60 fps', () => {
  const a = small(), b = small();
  run(a, 2, {}, 1 / 30);
  run(b, 2, {}, 1 / 60);
  assert.equal(a.emittedRows, b.emittedRows);
  for (let i = 0; i < a.positions.length; i++) assert.ok(Math.abs(a.positions[i] - b.positions[i]) < 1e-10);
});

test('the inlet sways smoothly as one straight strip and repeats after reset', () => {
  const sim = small();
  const sample = () => {
    const offsets = [];
    let previous = 0;
    for (let frame = 0; frame < 360; frame++) {
      sim.advance(1 / 60, { speed: 2 });
      const z = sim.positions[2];
      assert.ok(Math.abs(z) < sim.width * 0.26, 'inlet sway must stay bounded');
      assert.ok(Math.abs(z - previous) < sim.width * 0.004, 'inlet must not jump between frames');
      for (let i = 0; i < sim.stride * 2; i++) assert.equal(sim.positions[i * 3 + 2], z);
      offsets.push(z);
      previous = z;
    }
    return offsets;
  };
  const first = sample();
  assert.ok(Math.max(...first) - Math.min(...first) > sim.width * 0.15, 'source should visibly move');
  sim.reset();
  assert.deepEqual(sample(), first);
});

test('a hanging reset places the tip without advancing time or stretching rows', () => {
  const sim = small();
  const settings = { tipHeight: 1.47, speed: 2 };
  sim.reset(settings);
  const initial = sim.positions.slice();
  const last = (sim.activeRows - 1) * sim.stride * 3;
  assert.ok(Math.abs(sim.positions[last + 1] - settings.tipHeight) < 1e-9);
  assert.equal(sim.time, 0);
  assert.equal(sim.emittedRows, 0);
  assert.ok(sim.velocities.every(v => v === 0));
  for (let row = 1; row < sim.activeRows; row++) {
    const i = row * sim.stride * 3, previous = i - sim.stride * 3;
    assert.ok(Math.abs(sim.positions[previous + 1] - sim.positions[i + 1] - sim.spacing) < 1e-9);
    assert.ok(sim.birth[row] <= sim.birth[row - 1]);
  }
  run(sim, 3, { speed: 2 });
  sim.reset(settings);
  assert.deepEqual(sim.positions, initial);
});

test('a floor-contact reset accounts for the selected thickness', () => {
  const sim = small();
  for (const thickness of [.02, .35, .5]) {
    sim.reset({ tipHeight: thickness / 2, speed: 2 });
    const tip = sim.positions[(sim.activeRows - 1) * sim.stride * 3 + 1];
    assert.ok(Math.abs(tip - thickness / 2) < 1e-9);
    run(sim, .5, { thickness, speed: 2 });
    assert.ok(sim.positions.every(Number.isFinite));
  }
});

test('stopping the feed leaves gravity active and does not create new material', () => {
  const sim = fed();
  const emitted = sim.emittedRows;
  const before = sim.positions.slice();
  run(sim, 1, { speed: 0 });
  assert.equal(sim.emittedRows, emitted);
  assert.ok(Math.abs(sim.positions[1] - (sim.height + sim.spacing - sim.feed)) < 1e-9);
  assert.ok(sim.positions.some((v, i) => Math.abs(v - before[i]) > 0.02));
});

test('self-contact separates non-neighboring particles, including coincident particles', () => {
  for (const separation of [0, 0.1]) {
    const sim = fed(), count = sim.activeRows * sim.stride;
    for (let i = 0; i < count; i++) sim.positions.set([i * 4, 20, 0], i * 3);
    const a = 3 * sim.stride, b = 9 * sim.stride;
    sim.positions.set([0, 1, 0], a * 3);
    sim.positions.set([0, 1 + separation, 0], b * 3);
    sim.weights.fill(1);
    sim.solveContacts(count, 0.4);
    const distance = Math.hypot(...[0, 1, 2].map(k => sim.positions[a * 3 + k] - sim.positions[b * 3 + k]));
    assert.ok(distance >= 0.4 - 1e-9);
  }
});

test('feeding and recycling stay bounded across thickness and bending settings', () => {
  for (const [thickness, stiffness] of [[0.02, 0], [0.2, 0.5], [0.5, 1]]) {
    const sim = small(), positions = sim.positions;
    run(sim, 15, { speed: 2, thickness, stiffness });
    assert.equal(sim.activeRows, sim.maxRows);
    assert.ok(sim.emittedRows > sim.maxRows * 2);
    assert.equal(sim.positions, positions, 'recycling must reuse its storage');
    assert.ok(sim.positions.every(Number.isFinite));
    assert.ok(sim.positions.every(v => Math.abs(v) < 20));
    for (let row = 2; row < sim.maxRows - 22; row++) for (let col = 0; col < sim.stride; col++) {
      assert.ok(sim.positions[(row * sim.stride + col) * 3 + 1] >= thickness * 0.5 - 1e-9, 'active cloth must stay above the floor');
    }
    let maxStretch = 0;
    for (let i = 0; i < sim.edges.length; i += 4) {
      if (sim.edges[i + 3] !== 0) continue;
      const a = sim.edges[i] * 3, b = sim.edges[i + 1] * 3;
      const length = Math.hypot(...[0, 1, 2].map(k => sim.positions[a + k] - sim.positions[b + k]));
      maxStretch = Math.max(maxStretch, length / sim.edges[i + 2]);
    }
    assert.ok(maxStretch < 1.2, `excessive stretch at thickness ${thickness}: ${maxStretch}`);
    assert.ok(sim.positions.some((v, i) => i % 3 === 1 && v < 0), 'old cloth must enter the recycling zone');
    for (let r = 1; r < sim.activeRows; r++) assert.ok(sim.birth[r] <= sim.birth[r - 1], 'rainbow birth times must follow the material');
  }
});

test('viewport planes follow perspective edges and aspect ratio', () => {
  for (const aspect of [0.5, 1, 2]) {
    const sy = 1 / Math.tan(55 * Math.PI / 360), sx = sy / aspect;
    // Perspective camera at (0, 0, 10), looking along -Z.
    const matrix = [sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, -1, -1, 0, 0, 9.8, 10];
    const walls = viewportWallPlanes(matrix);
    const points = [[-10 / sx, 0, 0], [10 / sx, 0, 0], [0, -10 / sy, 0]];
    for (let wall = 0; wall < 3; wall++) {
      const k = wall * 4, point = points[wall];
      assert.ok(Math.abs(walls[k] * point[0] + walls[k + 1] * point[1] + walls[k + 2] * point[2] + walls[k + 3]) < 1e-10);
      assert.ok(Math.abs(Math.hypot(walls[k], walls[k + 1], walls[k + 2]) - 1) < 1e-10);
      assert.ok(walls[k + 3] > 0, 'center of the view should lie inside');
    }
  }
});

test('viewport walls contain thick particles and resizing does not add velocity', () => {
  const sim = fed(), count = sim.activeRows * sim.stride;
  sim.setViewportWalls(new Float64Array([1, 0, 0, 1, -1, 0, 0, 1, 0, 1, 0, 0]));
  const a = sim.stride * 3 * 3;
  sim.positions.set([2, -1, 0], a);
  sim.velocities.set([5, -2, 0], a);
  sim.solveWalls(count, 0.4, true);
  assert.ok(sim.positions[a] <= 0.775 + 1e-9);
  assert.ok(sim.positions[a + 1] >= 0.225 - 1e-9);
  assert.equal(sim.velocities[a], 0);
  assert.equal(sim.velocities[a + 1], 0);
  sim.setViewportWalls(new Float64Array([1, 0, 0, 0.7, -1, 0, 0, 0.7, 0, 1, 0, 0]));
  sim.solveWalls(count, 0.4, true);
  assert.ok(sim.positions[a] <= 0.475 + 1e-9);
  assert.equal(sim.velocities[a], 0);
});

test('cloth length follows available folding area and material thickness', () => {
  const walls = width => new Float64Array([0, 0, 1, width / 2, 0, 0, -1, width / 2, 0, 1, 0, 0]);
  const settings = { height: 7.5, spacing: 3.8 / 12, thickness: 0.35, maxLength: 60 };
  const narrow = clothLengthForViewport(walls(5), settings);
  const square = clothLengthForViewport(walls(8), settings);
  const wide = clothLengthForViewport(walls(12), settings);
  assert.ok(narrow < square && square < wide);
  assert.ok(square < 40, 'the square view should retain considerably less than the old 60-unit cloth');
  assert.ok(clothLengthForViewport(walls(8), { ...settings, thickness: 0.5 }) < square);
  assert.ok(clothLengthForViewport(walls(8), { ...settings, thickness: 0.02 }) > square);
  assert.equal(clothLengthForViewport(walls(100), settings), settings.maxLength);
});

test('shrinking the cloth budget fades rows before removing them', () => {
  const sim = new ClothSimulation({ columns: 6, maxRows: 72, width: 1.8, height: 3 });
  run(sim, 8, { speed: 2 });
  const positions = sim.positions, retained = sim.retainedRows;
  const before = sim.visibilityAt(sim.maxRows - 7);
  sim.setRetainedLength(10);
  assert.equal(sim.retainedRows, retained, 'a resize must not instantly cut the cloth');
  assert.equal(sim.visibilityAt(sim.maxRows - 7), before);
  const previousBirth = sim.birth.slice();
  const activeRows = sim.activeRows;
  sim.advance(1 / 120, { speed: 0 });
  assert.ok(Math.abs(sim.visibilityAt(sim.maxRows - 7) - before) < 0.03);
  assert.deepEqual(sim.birth, previousBirth, 'resizing must not move material through the inlet');
  assert.equal(sim.activeRows, activeRows);
  for (let i = 0; i < 600; i++) {
    const oldActive = sim.activeRows;
    sim.advance(1 / 120, { speed: 0 });
    for (let row = sim.activeRows; row < oldActive; row++) assert.equal(sim.visibilityAt(row), 0);
  }
  assert.ok(Math.abs(sim.retainedRows - sim.targetRetainedRows) < 1e-9);
  assert.equal(sim.activeRows, Math.ceil(sim.retainedRows));
  assert.equal(sim.positions, positions);
  assert.ok(sim.positions.every(Number.isFinite));
  assert.ok(sim.velocities.every(Number.isFinite));

  const shortRows = sim.activeRows, hidden = sim.visibilityAt(shortRows - 2);
  sim.setRetainedLength(100);
  run(sim, 1, { speed: 0 });
  assert.equal(sim.activeRows, shortRows, 'stopped feeding must not resurrect a removed tail');
  assert.equal(sim.visibilityAt(shortRows - 2), hidden);
  run(sim, 2, { speed: 1 });
  assert.ok(sim.activeRows > shortRows);
  assert.ok(sim.activeRows <= Math.ceil(sim.retainedRows));
  sim.reset();
  assert.equal(sim.retainedRows, sim.targetRetainedRows);
});

test('adaptive recycling remains independent of display frame rate', () => {
  const a = small(), b = small();
  for (const sim of [a, b]) { run(sim, 3, {}); sim.setRetainedLength(10); }
  run(a, 2, {}, 1 / 30);
  run(b, 2, {}, 1 / 60);
  assert.equal(a.retainedRows, b.retainedRows);
  assert.equal(a.activeRows, b.activeRows);
  assert.deepEqual(a.positions, b.positions);
});

test('recycling and visibility remain continuous when rows are inserted', () => {
  const sim = small();
  for (const row of [sim.maxRows - 20, sim.maxRows - 8, sim.maxRows - 4]) {
    sim.feed = sim.spacing * (1 - 1e-8);
    const sink = sim.recycleAmount(row), visibility = sim.visibilityAt(row);
    sim.feed = 0;
    assert.ok(Math.abs(sink - sim.recycleAmount(row + 1)) < 1e-8);
    assert.ok(Math.abs(visibility - sim.visibilityAt(row + 1)) < 1e-8);
  }
  sim.feed = 0;
  assert.equal(sim.visibilityAt(sim.maxRows - 2), 0, 'last segment must be invisible before deletion');
  assert.equal(sim.visibilityAt(0), 1);
  assert.ok(sim.recycleAmount(sim.maxRows - 8) > 0);
});

test('front/back walls leave a quarter of the cloth depth beyond each inlet edge', () => {
  const sim = fed(), count = sim.activeRows * sim.stride;
  assert.ok(Math.abs(sim.depthHalfExtent - sim.width / 2 - sim.width / 4) < 1e-9);
  for (const thickness of [0.02, 0.35, 0.5]) {
    for (const sign of [-1, 1]) {
      const i = 3 * sim.stride * 3;
      sim.positions.set([sign * (sim.width + 2), 1, 0.6], i);
      sim.velocities.set([sign * 5, 2, 3], i);
      sim.solveWalls(count, thickness, true);
      assert.ok(Math.abs(sim.positions[i]) <= sim.depthHalfExtent - thickness / 2 - 0.025 + 1e-9);
      assert.equal(sim.positions[i + 1], 1);
      assert.equal(sim.positions[i + 2], 0.6);
      assert.equal(sim.velocities[i], 0);
      assert.equal(sim.velocities[i + 1], 2);
      assert.equal(sim.velocities[i + 2], 3);
    }
  }
  // Recycling may pass through the bottom, but cannot escape in depth.
  sim.activeRows = sim.maxRows;
  const tail = (sim.maxRows - 1) * sim.stride * 3;
  sim.positions.set([sim.width + 2, -1, 0], tail);
  sim.solveWalls(sim.count, 0.35);
  assert.ok(sim.positions[tail] < sim.depthHalfExtent);
  assert.equal(sim.positions[tail + 1], -1);
});

test('a grab outside the front/back walls stays bounded while feeding', () => {
  const sim = fed();
  sim.beginGrab(5 * sim.stride + 3);
  sim.moveGrab([sim.width * 3, 1, 0.3]);
  run(sim, 2, { speed: 1, thickness: 0.35 });
  for (let i = sim.stride * 2; i < sim.activeRows * sim.stride; i++) {
    assert.ok(Math.abs(sim.positions[i * 3]) <= sim.depthHalfExtent - 0.2 + 1e-9);
  }
  assert.ok(sim.positions.every(Number.isFinite));
});

test('reset empties the cloth and clears feed history', () => {
  const sim = small(), initial = sim.positions.slice();
  run(sim, 1, {});
  sim.reset();
  assert.deepEqual(sim.positions, initial);
  assert.equal(sim.time, 0);
  assert.equal(sim.emittedRows, 0);
  assert.equal(sim.activeRows, 0);
  assert.ok(sim.birth.every(v => v === 0));
  assert.equal(sim.beginGrab(5 * sim.stride), false);
  assert.ok(sim.velocities.every(v => v === 0));
});

test('grabbing pulls a local cloth patch and releases it back to gravity', () => {
  const sim = fed(), index = 6 * sim.stride + 3;
  assert.equal(sim.beginGrab(0), false, 'the feed attachment stays pinned');
  assert.equal(sim.beginGrab(index), true);
  sim.moveGrab([0, 1.2, 0.7]);
  run(sim, 1, { speed: 0 });
  const held = Array.from(sim.positions.subarray(index * 3, index * 3 + 3));
  assert.ok(Math.hypot(held[0], held[1] - 1.2, held[2] - 0.7) < 0.05);
  assert.ok(Math.abs(sim.positions[1] - (sim.height + sim.spacing - sim.feed)) < 1e-9);
  sim.endGrab();
  run(sim, 1, { speed: 0 });
  assert.equal(sim.grab, null);
  assert.ok(Math.hypot(...held.map((value, axis) => sim.positions[index * 3 + axis] - value)) > 0.05);
  assert.ok(sim.positions.every(Number.isFinite));
});

test('a grab follows the same material across feed insertions and clears on reset', () => {
  const sim = fed(), index = 5 * sim.stride + 3;
  const emitted = sim.emittedRows;
  const birth = sim.birth[5];
  sim.beginGrab(index);
  run(sim, 0.5, { speed: 1 });
  assert.equal(sim.grab.index, index + (sim.emittedRows - emitted) * sim.stride);
  assert.equal(sim.birth[Math.floor(sim.grab.index / sim.stride)], birth);
  sim.moveGrab([Infinity, 0, 0]);
  assert.ok(sim.grab.target.every(Number.isFinite));
  sim.reset();
  assert.equal(sim.grab, null);
});

test('invalid grab indices and targets cannot poison the simulation', () => {
  const sim = fed();
  for (const index of [-1, NaN, Infinity, 2.5, sim.count]) assert.equal(sim.beginGrab(index), false);
  sim.beginGrab(5 * sim.stride + 3);
  const target = sim.grab.target.slice();
  for (const value of [null, [], [1, 2], [1, 2, 3, 4], [1, NaN, 2], [Infinity, 0, 0]]) {
    sim.moveGrab(value);
    assert.deepEqual(sim.grab.target, target);
  }
  run(sim, 0.1, { speed: 0 });
  assert.ok(sim.positions.every(Number.isFinite));
});
