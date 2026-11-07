// Inward-facing world-space planes for the left, right, and bottom edges.
// Extracting clip-space x/y boundaries works for both WebGPU and WebGL depth.
// The top stays open for the cloth inlet.
export function viewportWallPlanes(matrix, result = new Float64Array(12)) {
  for (let wall = 0; wall < 3; wall++) {
    const axis = wall === 2 ? 1 : 0;
    const sign = wall === 1 ? -1 : 1;
    const offset = wall * 4;
    for (let component = 0; component < 4; component++) {
      result[offset + component] = matrix[component * 4 + 3] + sign * matrix[component * 4 + axis];
    }
    const length = Math.hypot(result[offset], result[offset + 1], result[offset + 2]);
    for (let component = 0; component < 4; component++) result[offset + component] /= length;
  }
  return result;
}

// Estimate how much ribbon fits in the lower part of the side view at X=0.
// Leave room between curls instead of filling the whole view with solid cloth.
export function clothLengthForViewport(planes, { height, spacing, thickness, maxLength }) {
  const clearance = thickness * 0.5 + 0.025;
  const bottom = planes[9] > 1e-6 ? Math.max(0, (clearance - planes[11]) / planes[9]) : 0;
  const top = height * 0.4;
  const widthAt = y => {
    let low = -Infinity, high = Infinity;
    for (let wall = 0; wall < 2; wall++) {
      const k = wall * 4, nz = planes[k + 2];
      if (Math.abs(nz) < 1e-6) return Infinity;
      const z = (clearance - planes[k + 1] * y - planes[k + 3]) / nz;
      if (nz > 0) low = Math.max(low, z);
      else high = Math.min(high, z);
    }
    return Math.max(0, high - low);
  };
  const area = (widthAt(bottom) + widthAt(top)) * 0.5 * Math.max(0, top - bottom);
  if (!Number.isFinite(area)) return maxLength;
  const contactSpacing = Math.max(thickness, spacing * 0.85);
  // Reserve the hanging inlet plus the hidden rows used for smooth recycling.
  return Math.min(maxLength, height + area * 0.35 / contactSpacing + 22 * spacing);
}
