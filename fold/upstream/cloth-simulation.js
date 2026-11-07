// Fixed-step XPBD cloth: stretch/shear edges, compliant two-edge bending,
// particle self-contact, and a floor. Only the feed boundary is prescribed.
// Constraint formulation: Macklin et al., XPBD (2016):
// https://matthias-research.github.io/pages/publications/XPBD.pdf
export class ClothSimulation {
  constructor({ columns = 12, maxRows = 191, width = 3.8, height = 7.5 } = {}) {
    this.columns = columns;
    this.stride = columns + 1;
    this.maxRows = maxRows;
    this.targetRetainedRows = maxRows;
    this.width = width;
    // The camera looks along X: the inlet width is the cloth's depth in view.
    // Keep a quarter of that depth as free space beyond each inlet edge.
    this.depthHalfExtent = width * 0.75;
    this.height = height;
    this.spacing = width / columns;
    this.count = this.stride * maxRows;
    this.positions = new Float64Array(this.count * 3);
    this.previous = new Float64Array(this.count * 3);
    this.velocities = new Float64Array(this.count * 3);
    this.birth = new Float64Array(maxRows);
    this.weights = new Float64Array(this.count);
    this.head = new Int32Array(8192);
    this.next = new Int32Array(this.count);
    this.cells = new Int32Array(this.count * 3);
    this.wallPlanes = new Float64Array(12);
    this.wallCount = 0;
    this.wallsChanged = false;
    this.rowVisibility = new Float64Array(maxRows);
    const edges = [];
    const add = (a, b, length, type) => edges.push(a, b, length, type);
    for (let r = 0; r < maxRows; r++) for (let c = 0; c <= columns; c++) {
      const i = r * this.stride + c;
      if (c < columns) add(i, i + 1, this.spacing, 0);
      if (r + 1 < maxRows) add(i, i + this.stride, this.spacing, 0);
      if (r + 1 < maxRows && c < columns) {
        add(i, i + this.stride + 1, this.spacing * Math.SQRT2, 1);
        add(i + 1, i + this.stride, this.spacing * Math.SQRT2, 1);
      }
      if (c + 2 <= columns) add(i, i + 2, this.spacing * 2, 2);
      if (r + 2 < maxRows) add(i, i + this.stride * 2, this.spacing * 2, 2);
    }
    this.edges = new Float64Array(edges);
    this.lambdas = new Float64Array(edges.length / 4);
    this.reset();
  }

  reset({ tipHeight = null, speed = 1 } = {}) {
    this.retainedRows = this.targetRetainedRows;
    this.grab = null;
    this.time = 0;
    this.accumulator = 0;
    this.feed = 0;
    this.emittedRows = 0;
    this.rowVisibility.fill(1);
    this.wallsChanged = true;
    this.activeRows = 0;
    this.positions.fill(0);
    this.velocities.fill(0);
    this.birth.fill(0);
    this.weights.fill(0);
    this.lambdas.fill(0);
    if (Number.isFinite(tipHeight)) {
      const length = Math.max(0, this.height - tipHeight);
      const wholeRows = Math.floor(length / this.spacing);
      this.activeRows = Math.min(this.maxRows, wholeRows + 2);
      // Use the feed phase for the fractional row so every edge starts at its
      // rest length, including when the screen midpoint falls between rows.
      this.feed = Math.max(0, length - wholeRows * this.spacing);
      const feedRate = speed > 0 ? speed * 1.5 : 1.5;
      for (let r = 0; r < this.activeRows; r++) {
        this.birth[r] = -(r * this.spacing + this.feed) / feedRate;
        const bow = r < 2 ? 0 : 0.005 * ((r - 1) / (this.activeRows - 2)) ** 4;
        for (let col = 0; col < this.stride; col++) {
          const i = (r * this.stride + col) * 3;
          this.positions[i] = col * this.spacing - this.width * 0.5;
          this.positions[i + 1] = this.height + (1 - r) * this.spacing - this.feed;
          this.positions[i + 2] = bow;
        }
      }
    }
    this.previous.set(this.positions);
  }

  beginGrab(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.activeRows * this.stride) return false;
    const row = Math.floor(index / this.stride), col = index % this.stride;
    if (row < 2 || row >= this.activeRows || this.visibilityAt(row) < 0.1) return false;
    const center = Array.from(this.positions.subarray(index * 3, index * 3 + 3));
    const particles = [];
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      const r = row + dr, c = col + dc;
      if (r < 2 || r >= this.activeRows || c < 0 || c >= this.stride) continue;
      const i = r * this.stride + c;
      particles.push({ index: i, offset: center.map((v, axis) => this.positions[i * 3 + axis] - v),
        compliance: dr === 0 && dc === 0 ? 0.000002 : 0.00008 * (dr * dr + dc * dc), lambda: [0, 0, 0] });
    }
    this.grab = { index, target: center.slice(), anchor: center.slice(), particles };
    return true;
  }

  moveGrab(target) {
    if (this.grab && target?.length === 3 && Array.from(target).every(Number.isFinite)) {
      this.grab.target = Array.from(target);
    }
  }

  endGrab() {
    this.grab = null;
  }

  prepareGrab(dt, thickness) {
    const grab = this.grab;
    if (!grab) return;
    if (grab.index >= this.activeRows * this.stride || this.visibilityAt(Math.floor(grab.index / this.stride)) < 0.05) {
      this.endGrab();
      return;
    }
    grab.target[1] = Math.max(thickness * 0.5, grab.target[1]);
    const dx = grab.target[0] - grab.anchor[0], dy = grab.target[1] - grab.anchor[1], dz = grab.target[2] - grab.anchor[2];
    // A bounded, smoothed anchor prevents fast mouse/touch motion from injecting
    // a single-frame impulse into the stretch constraints.
    const distance = Math.hypot(dx, dy, dz);
    const blend = Math.min(1 - Math.exp(-30 * dt), 8 * dt / Math.max(distance, 1e-9));
    grab.anchor[0] += dx * blend; grab.anchor[1] += dy * blend; grab.anchor[2] += dz * blend;
    for (const particle of grab.particles) particle.lambda.fill(0);
  }

  solveGrab(dt) {
    if (!this.grab) return;
    const { anchor, particles } = this.grab;
    for (const particle of particles) {
      if (particle.index >= this.activeRows * this.stride) continue;
      const weight = this.weights[particle.index];
      const alpha = particle.compliance / (dt * dt);
      for (let axis = 0; axis < 3; axis++) {
        const i = particle.index * 3 + axis;
        const error = this.positions[i] - anchor[axis] - particle.offset[axis];
        const dl = (-error - alpha * particle.lambda[axis]) / (weight + alpha);
        particle.lambda[axis] += dl;
        this.positions[i] += weight * dl;
      }
    }
  }

  setViewportWalls(planes) {
    if (this.wallCount === 3 && planes.every((value, i) => Math.abs(value - this.wallPlanes[i]) < 1e-9)) return;
    this.wallPlanes.set(planes);
    this.wallCount = 3;
    this.wallsChanged = true;
  }

  setRetainedLength(length) {
    if (!Number.isFinite(length)) return;
    const minimum = Math.min(this.maxRows, Math.ceil(this.height / this.spacing) + 24);
    this.targetRetainedRows = Math.max(minimum, Math.min(this.maxRows, length / this.spacing + 1));
    if (this.activeRows === 0) this.retainedRows = this.targetRetainedRows;
  }

  recycleAmount(row) {
    const age = row + this.feed / this.spacing;
    const t = Math.max(0, Math.min(1, (age - (this.retainedRows - 22)) / 20));
    return t * t * (3 - 2 * t);
  }

  visibilityAt(row) {
    const age = row + this.feed / this.spacing;
    // Leave two fully invisible rows at the tail, hiding endpoint changes
    // when its final row is removed and the spline gets a new end tangent.
    const t = Math.max(0, Math.min(1, (age - (this.retainedRows - 10)) / 7));
    return 1 - t * t * (3 - 2 * t);
  }

  advance(delta, { speed = 1, thickness = 0.2, stiffness = 0.5 } = {}) {
    this.accumulator += Math.min(Math.max(delta, 0), 0.05);
    const dt = 1 / 120;
    while (this.accumulator + 1e-10 >= dt) {
      this.step(dt, speed, thickness, stiffness);
      this.accumulator -= dt;
    }
  }

  step(dt, speed, thickness, stiffness) {
    this.time += dt;
    // Uneven frequencies and phases give the inlet an irregular sideways sway.
    // Use simulation time so resets and offline captures repeat the same motion.
    // Subtract the starting phases to keep the initial hanging sheet centered.
    const t = this.time;
    const sourceZ = this.width * (
      0.12 * Math.sin(t * 0.71) +
      0.045 * (Math.sin(t * 1.23 + 1.7) - Math.sin(1.7)) +
      0.025 * (Math.sin(t * 2.13 + 4.2) - Math.sin(4.2))
    );
    if (this.activeRows === 0) {
      if (speed <= 0) return;
      // An empty reset creates only the attachment rows when feeding starts.
      this.activeRows = 2;
      for (let r = 0; r < 2; r++) {
        this.birth[r] = this.time;
        for (let c = 0; c < this.stride; c++) {
          const i = (r * this.stride + c) * 3;
          this.positions[i] = c * this.spacing - this.width * 0.5;
          this.positions[i + 1] = this.height + (1 - r) * this.spacing;
          this.positions[i + 2] = sourceZ;
        }
      }
    }
    // Move the fade boundary gradually when the window shrinks. When it grows,
    // allow only newly fed material to extend the tail, avoiding resurrection
    // of rows that have already disappeared (including when feeding is stopped).
    if (this.retainedRows > this.targetRetainedRows) {
      this.retainedRows = Math.max(this.targetRetainedRows, this.retainedRows - 12 * dt);
    } else {
      this.retainedRows = Math.min(this.targetRetainedRows, this.retainedRows + speed * 1.5 * dt / this.spacing);
    }
    this.feed += speed * 1.5 * dt;
    while (this.feed >= this.spacing) {
      this.feed -= this.spacing;
      const stride3 = this.stride * 3;
      this.positions.copyWithin(stride3, 0, this.positions.length - stride3);
      this.velocities.copyWithin(stride3, 0, this.velocities.length - stride3);
      this.velocities.fill(0, 0, stride3);
      this.birth.copyWithin(1, 0, this.maxRows - 1);
      this.birth[0] = this.time;
      this.activeRows = Math.min(this.maxRows, this.activeRows + 1);
      if (this.emittedRows === 0 && this.activeRows === 3) {
        // A tiny bow on the leading edge breaks perfect vertical symmetry.
        for (let c = 0; c < this.stride; c++) this.positions[(2 * this.stride + c) * 3 + 2] += 0.005;
      }
      this.emittedRows++;
      if (this.grab) {
        this.grab.index += this.stride;
        for (const particle of this.grab.particles) particle.index += this.stride;
      }
    }
    // The final two rows are already invisible before they leave the solver.
    this.activeRows = Math.min(this.activeRows, Math.ceil(this.retainedRows));
    const count = this.activeRows * this.stride;
    const p = this.positions, v = this.velocities, old = this.previous;
    for (let row = 0; row < this.activeRows; row++) this.rowVisibility[row] = this.visibilityAt(row);
    if (this.wallsChanged) {
      // Reposition for a resized/orbited viewport before saving old positions,
      // so a camera movement does not become a large artificial velocity.
      this.solveWalls(count, thickness, true);
      this.wallsChanged = false;
    }
    const damping = Math.exp(-1.3 * dt);
    this.weights.fill(1, 0, count);
    this.weights.fill(0, 0, this.stride * 2);
    // Two feed rows prescribe a vertical inlet; everything below is free.
    for (let r = 0; r < 2; r++) for (let c = 0; c < this.stride; c++) {
      const i = (r * this.stride + c) * 3;
      p[i] = c * this.spacing - this.width * 0.5;
      p[i + 1] = this.height + (1 - r) * this.spacing - this.feed;
      p[i + 2] = sourceZ;
    }
    old.set(p);
    for (let i = this.stride * 2; i < count; i++) {
      const j = i * 3;
      v[j] *= damping;
      v[j + 1] = v[j + 1] * damping - 9.81 * dt;
      v[j + 2] *= damping;
      p[j] += v[j] * dt;
      p[j + 1] += v[j + 1] * dt;
      p[j + 2] += v[j + 2] * dt;
    }
    this.lambdas.fill(0);
    this.prepareGrab(dt, thickness);
    const bendCompliance = 0.00002 * 10 ** ((1 - stiffness) * 3);
    for (let iteration = 0; iteration < 4; iteration++) {
      this.solveEdges(count, dt, bendCompliance, iteration % 2 === 1);
      this.solveGrab(dt);
      if (iteration % 2 === 1) this.solveContacts(count, thickness);
      this.solveFloor(count, thickness);
      this.solveWalls(count, thickness);
    }
    for (let i = this.stride * 2; i < count; i++) {
      const j = i * 3;
      v[j] = (p[j] - old[j]) / dt;
      v[j + 1] = (p[j + 1] - old[j + 1]) / dt;
      v[j + 2] = (p[j + 2] - old[j + 2]) / dt;
    }
  }

  solveEdges(count, dt, bendCompliance, reverse) {
    const p = this.positions, w = this.weights, e = this.edges;
    const length = e.length / 4;
    for (let k = 0; k < length; k++) {
      const n = reverse ? length - 1 - k : k, offset = n * 4;
      const a = e[offset], b = e[offset + 1];
      if (a >= count || b >= count || w[a] + w[b] === 0) continue;
      const ia = a * 3, ib = b * 3;
      const dx = p[ia] - p[ib], dy = p[ia + 1] - p[ib + 1], dz = p[ia + 2] - p[ib + 2];
      const distance = Math.hypot(dx, dy, dz);
      if (distance < 1e-9) continue;
      const type = e[offset + 3];
      const alpha = (type === 2 ? bendCompliance : type === 1 ? 0.0000005 : 0.0000001) / (dt * dt);
      const dl = (-(distance - e[offset + 2]) - alpha * this.lambdas[n]) / (w[a] + w[b] + alpha);
      this.lambdas[n] += dl;
      const sa = dl * w[a] / distance, sb = dl * w[b] / distance;
      p[ia] += dx * sa; p[ia + 1] += dy * sa; p[ia + 2] += dz * sa;
      p[ib] -= dx * sb; p[ib + 1] -= dy * sb; p[ib + 2] -= dz * sb;
    }
  }

  hash(x, y, z) {
    return (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) & (this.head.length - 1);
  }

  solveContacts(count, thickness) {
    // Particle contact is an approximation; a grid-sized minimum clearance
    // prevents thin cloth from passing through gaps between contact samples.
    const separation = Math.max(thickness, this.spacing * 0.85);
    const d2 = separation * separation;
    const p = this.positions, w = this.weights, cells = this.cells;
    const excluded = Math.max(2, Math.ceil(thickness / this.spacing));
    this.head.fill(-1);
    for (let i = 0; i < count; i++) {
      const j = i * 3;
      const x = Math.floor(p[j] / separation), y = Math.floor(p[j + 1] / separation), z = Math.floor(p[j + 2] / separation);
      cells[j] = x; cells[j + 1] = y; cells[j + 2] = z;
      const bucket = this.hash(x, y, z);
      this.next[i] = this.head[bucket]; this.head[bucket] = i;
    }
    for (let a = this.stride * 2; a < count; a++) {
      const ia = a * 3, row = Math.floor(a / this.stride), col = a % this.stride;
      for (let x = -1; x <= 1; x++) for (let y = -1; y <= 1; y++) for (let z = -1; z <= 1; z++) {
        const cx = cells[ia] + x, cy = cells[ia + 1] + y, cz = cells[ia + 2] + z;
        for (let b = this.head[this.hash(cx, cy, cz)]; b !== -1; b = this.next[b]) {
          if (b <= a || (Math.abs(row - Math.floor(b / this.stride)) <= excluded && Math.abs(col - b % this.stride) <= excluded)) continue;
          const ib = b * 3;
          if (cells[ib] !== cx || cells[ib + 1] !== cy || cells[ib + 2] !== cz) continue;
          let dx = p[ia] - p[ib], dy = p[ia + 1] - p[ib + 1], dz = p[ia + 2] - p[ib + 2];
          const squared = dx * dx + dy * dy + dz * dz;
          if (squared >= d2) continue;
          // Release support continuously as the old cloth disappears.
          const contact = separation * Math.min(this.rowVisibility[row], this.rowVisibility[Math.floor(b / this.stride)]);
          if (squared >= contact * contact) continue;
          let distance = Math.sqrt(squared), penetration = contact - distance;
          if (distance < 1e-8) { dx = 0; dy = 1; dz = 0; distance = 1; penetration = contact; }
          const correction = penetration / (distance * (w[a] + w[b]));
          p[ia] += dx * correction * w[a]; p[ia + 1] += dy * correction * w[a]; p[ia + 2] += dz * correction * w[a];
          p[ib] -= dx * correction * w[b]; p[ib + 1] -= dy * correction * w[b]; p[ib + 2] -= dz * correction * w[b];
        }
      }
    }
  }

  solveFloor(count, thickness) {
    const p = this.positions, old = this.previous;
    for (let i = this.stride * 2; i < count; i++) {
      const j = i * 3, row = Math.floor(i / this.stride);
      // The oldest rows sink out through the recycling zone before reuse.
      const recycle = this.recycleAmount(row);
      const floor = thickness * 0.5 - recycle * 1.5;
      if (recycle > 0) p[j + 1] -= recycle * 0.001;
      if (p[j + 1] < floor) {
        p[j + 1] = floor;
        p[j] = old[j] + (p[j] - old[j]) * 0.7;
        p[j + 2] = old[j + 2] + (p[j + 2] - old[j + 2]) * 0.7;
      }
    }
  }

  solveWalls(count, thickness, moving = false) {
    const p = this.positions, v = this.velocities, planes = this.wallPlanes;
    // Extra clearance accommodates the smooth render surface between particles.
    const clearance = thickness * 0.5 + 0.025;
    for (let i = this.stride * 2; i < count; i++) {
      const j = i * 3, row = Math.floor(i / this.stride);
      for (let sweep = 0; sweep < 2; sweep++) for (let wall = 0; wall < this.wallCount + 2; wall++) {
        // Let the retiring tail leave through the bottom edge.
        if (wall === 2 && wall < this.wallCount && this.recycleAmount(row) > 0) continue;
        // Front/back walls stay fixed around the inlet, independently of the
        // window size and the changing bounds of the folded pile.
        const isDepthWall = wall >= this.wallCount;
        const k = wall * 4;
        const nx = isDepthWall ? (wall === this.wallCount ? 1 : -1) : planes[k];
        const ny = isDepthWall ? 0 : planes[k + 1];
        const nz = isDepthWall ? 0 : planes[k + 2];
        const d = isDepthWall ? this.depthHalfExtent : planes[k + 3];
        const distance = p[j] * nx + p[j + 1] * ny + p[j + 2] * nz + d;
        if (distance >= clearance) continue;
        const correction = clearance - distance;
        p[j] += nx * correction; p[j + 1] += ny * correction; p[j + 2] += nz * correction;
        if (moving) {
          const outward = Math.min(0, v[j] * nx + v[j + 1] * ny + v[j + 2] * nz);
          v[j] -= nx * outward; v[j + 1] -= ny * outward; v[j + 2] -= nz * outward;
        }
      }
    }
  }
}
