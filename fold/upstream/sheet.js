import * as THREE from 'three';
import { ClothSimulation } from './cloth-simulation.js';

// A smooth thick shell follows the simulated cloth grid. GPU topology stays
// fixed as new particle rows feed in and old rows are recycled.
export class FallingSheet {
  constructor(material) {
    this.simulation = new ClothSimulation();
    this.columns = 24;
    this.rows = (this.simulation.maxRows - 1) * 3;
    this.surface = new THREE.PlaneGeometry(1, 1, this.columns, this.rows);
    const count = this.surface.attributes.position.count;
    const stride = this.columns + 1;
    // Clockwise perimeter, with each corner included once.
    this.perimeter = [];
    for (let c = 0; c < this.columns; c++) this.perimeter.push(c);
    for (let r = 0; r < this.rows; r++) this.perimeter.push(r * stride + this.columns);
    for (let c = this.columns; c > 0; c--) this.perimeter.push(this.rows * stride + c);
    for (let r = this.rows; r > 0; r--) this.perimeter.push(r * stride);
    this.edgeSegments = 6;
    const perimeterCount = this.perimeter.length;
    const vertexCount = count * 2 + perimeterCount * (this.edgeSegments - 1);
    const uv = new Float32Array(vertexCount * 2);
    uv.set(this.surface.attributes.uv.array);
    uv.set(this.surface.attributes.uv.array, count * 2);
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3));
    this.geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3));
    this.geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    this.geometry.setAttribute('emissionTime', new THREE.BufferAttribute(new Float32Array(vertexCount), 1));
    this.geometry.setAttribute('clothVisibility', new THREE.BufferAttribute(new Float32Array(vertexCount), 1));
    const indices = Array.from(this.surface.index.array);
    for (let i = 0; i < this.surface.index.count; i += 3) {
      indices.push(count + this.surface.index.array[i], count + this.surface.index.array[i + 2], count + this.surface.index.array[i + 1]);
    }
    const rimIndex = (ring, i) => ring === 0 ? this.perimeter[i] : ring === this.edgeSegments ? count + this.perimeter[i] : count * 2 + (ring - 1) * perimeterCount + i;
    for (let ring = 0; ring < this.edgeSegments; ring++) {
      for (let i = 0; i < perimeterCount; i++) {
        const next = (i + 1) % perimeterCount;
        const a = rimIndex(ring, i), b = rimIndex(ring + 1, i);
        const c = rimIndex(ring, next), d = rimIndex(ring + 1, next);
        indices.push(a, c, b, c, d, b);
      }
    }
    for (let ring = 1; ring < this.edgeSegments; ring++) {
      for (let i = 0; i < perimeterCount; i++) {
        const source = this.perimeter[i];
        const target = rimIndex(ring, i);
        uv[target * 2] = this.surface.attributes.uv.getX(source);
        uv[target * 2 + 1] = this.surface.attributes.uv.getY(source);
      }
    }
    this.geometry.setIndex(indices);
    this.geometry.attributes.position.setUsage(THREE.DynamicDrawUsage);
    this.geometry.attributes.normal.setUsage(THREE.DynamicDrawUsage);
    this.edgeTangent = new THREE.Vector3();
    this.edgeNormal = new THREE.Vector3();
    this.edgeOutward = new THREE.Vector3();
    this.meshes = [];
    this.mesh = this.createMesh(material);
    this.update(0);
  }

  createMesh(material) {
    const mesh = new THREE.Mesh(this.geometry, material);
    mesh.frustumCulled = false;
    mesh.visible = this.simulation.activeRows >= 2;
    this.meshes.push(mesh);
    return mesh;
  }

  update(delta, settings = {}) {
    const thickness = settings.thickness ?? 0.2;
    const sim = this.simulation;
    sim.advance(delta, settings);
    for (const mesh of this.meshes) mesh.visible = sim.activeRows >= 2;
    // All render passes share this geometry, so an empty simulation must draw
    // nothing in the beauty, depth/normal, and refraction passes alike.
    if (sim.activeRows < 2) {
      this.geometry.setDrawRange(0, 0);
      return;
    }
    this.geometry.setDrawRange(0, Infinity);
    const positions = this.surface.attributes.position;
    const emission = this.geometry.attributes.emissionTime;
    const visibility = this.geometry.attributes.clothVisibility;
    const sample = (row, col, axis) => sim.positions[(Math.max(0, Math.min(sim.activeRows - 1, row)) * sim.stride + col) * 3 + axis];
    const interpolate = (row, col, axis, t) => {
      const a = sample(row - 1, col, axis), b = sample(row, col, axis);
      const c = sample(row + 1, col, axis), d = sample(row + 2, col, axis);
      return b + 0.5 * t * (c - a + t * (2 * a - 5 * b + 4 * c - d + t * (3 * (b - c) + d - a)));
    };
    for (let row = 0; row <= this.rows; row++) {
      const sourceRow = row / this.rows * (sim.activeRows - 1);
      const r = Math.floor(sourceRow), t = sourceRow - r;
      const birth = sim.birth[r] * (1 - t) + sim.birth[Math.min(r + 1, sim.activeRows - 1)] * t;
      const fade = sim.visibilityAt(sourceRow);
      for (let col = 0; col <= this.columns; col++) {
        const sourceCol = col / this.columns * sim.columns;
        const c = Math.floor(sourceCol), f = sourceCol - c, next = Math.min(c + 1, sim.columns);
        const i = row * (this.columns + 1) + col;
        for (let axis = 0; axis < 3; axis++) {
          positions.array[i * 3 + axis] = interpolate(r, c, axis, t) * (1 - f) + interpolate(r, next, axis, t) * f;
        }
        emission.setX(i, birth);
        emission.setX(positions.count + i, birth);
        visibility.setX(i, fade);
        visibility.setX(positions.count + i, fade);
      }
    }
    this.surface.computeVertexNormals();
    const sourceNormals = this.surface.attributes.normal;
    const shellPositions = this.geometry.attributes.position;
    const shellNormals = this.geometry.attributes.normal;
    const half = thickness * 0.5;
    const count = positions.count;
    for (let i = 0; i < count; i++) {
      const x = positions.getX(i), y = positions.getY(i), z = positions.getZ(i);
      const nx = sourceNormals.getX(i), ny = sourceNormals.getY(i), nz = sourceNormals.getZ(i);
      shellPositions.setXYZ(i, x + nx * half, y + ny * half, z + nz * half);
      shellPositions.setXYZ(count + i, x - nx * half, y - ny * half, z - nz * half);
      shellNormals.setXYZ(i, nx, ny, nz);
      shellNormals.setXYZ(count + i, -nx, -ny, -nz);
    }
    // Semicircular rims connect the two faces, catching a continuous highlight.
    const length = this.perimeter.length;
    for (let i = 0; i < length; i++) {
      const source = this.perimeter[i];
      const previous = this.perimeter[(i + length - 1) % length];
      const next = this.perimeter[(i + 1) % length];
      this.edgeTangent.set(positions.getX(next) - positions.getX(previous), positions.getY(next) - positions.getY(previous), positions.getZ(next) - positions.getZ(previous)).normalize();
      this.edgeNormal.fromBufferAttribute(sourceNormals, source);
      this.edgeOutward.crossVectors(this.edgeNormal, this.edgeTangent).normalize();
      for (let ring = 1; ring < this.edgeSegments; ring++) {
        const angle = ring / this.edgeSegments * Math.PI;
        const a = Math.cos(angle), b = Math.sin(angle);
        const nx = this.edgeNormal.x * a + this.edgeOutward.x * b;
        const ny = this.edgeNormal.y * a + this.edgeOutward.y * b;
        const nz = this.edgeNormal.z * a + this.edgeOutward.z * b;
        const target = count * 2 + (ring - 1) * length + i;
        emission.setX(target, emission.getX(source));
        visibility.setX(target, visibility.getX(source));
        shellPositions.setXYZ(target, positions.getX(source) + nx * half, positions.getY(source) + ny * half, positions.getZ(source) + nz * half);
        shellNormals.setXYZ(target, nx, ny, nz);
      }
    }
    shellPositions.needsUpdate = true;
    shellNormals.needsUpdate = true;
    emission.needsUpdate = true;
    visibility.needsUpdate = true;
  }
}
