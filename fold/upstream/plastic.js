import * as THREE from 'three/webgpu';
import { Fn, If, Loop, Break, bool, int, float, vec3, attribute, mix, fract, abs, clamp, normalView, positionView, positionWorld, normalize, refract, dot, pow, depth, screenUV, textureSize, log2, max, smoothstep, cameraProjectionMatrix, cameraProjectionMatrixInverse, getViewPosition, getScreenPosition } from 'three/tsl';

export function createPlastic(uniforms) {
  const material = new THREE.MeshPhysicalNodeMaterial({
    roughness: 0, metalness: 0, clearcoat: 0.15, clearcoatRoughness: 0.075,
    side: THREE.FrontSide, ior: 1.46, iridescence: 0.12,
    iridescenceIOR: 1.3, iridescenceThicknessMinimum: 180, iridescenceThicknessMaximum: 420
  });
  // Birth time travels with the particles, so stretching and folding do not
  // slide the rainbow through the cloth.
  const emissionTime = attribute('emissionTime', 'float');
  const hue = emissionTime.mul(0.065).add(0.05);
  const rainbow = clamp(abs(fract(hue.add(vec3(0, 2 / 3, 1 / 3))).mul(6).sub(3)).sub(1), 0, 1);
  const smoothRainbow = rainbow.mul(rainbow).mul(float(3).sub(rainbow.mul(2)));
  const pigment = mix(vec3(1), mix(vec3(0.025), vec3(0.7), smoothRainbow), uniforms.tint);
  material.colorNode = pigment;
  material.roughnessNode = uniforms.roughness;
  material.iorNode = uniforms.ior;
  // Dithered coverage lets the retiring tail fade in every depth-writing pass
  // without transparent-object sorting or a separate compositing pipeline.
  material.alphaHash = true;
  material.opacityNode = attribute('clothVisibility', 'float');
  // Recycle the tail at the floor in beauty, depth/normal, and refraction
  // passes, including when the camera or a refracted ray looks from below.
  material.maskNode = positionWorld.y.greaterThanEqual(0);
  return { material, pigment };
}

// The depth-writing shell samples the next visible layer for refraction.
export function addRefraction(material, pigment, uniforms, frontDepth, behindColor, behindDepth) {
  const nearest = abs(depth.sub(frontDepth.sample(screenUV).r)).lessThan(0.00001);
  // Replace only the diffuse/transmitted contribution. Three.js adds physical
  // specular lighting, clearcoat, and fog afterward.
  material.backdropAlphaNode = nearest.select(uniforms.refraction, float(0));
  material.backdropNode = Fn(() => {
    const result = vec3(0).toVar();
    If(nearest.and(uniforms.refraction.greaterThan(0)), () => {
      const origin = positionView.toVar();
      const incident = normalize(origin).toVar();
      const inside = refract(incident, normalView, uniforms.ior.reciprocal()).toVar();
      // Approximate the shell locally as a parallel slab: bend on entry, then
      // resume the incident direction on exit. The optical displacement comes
      // from the actual thickness, rather than bending through several metres
      // of empty air behind the plastic.
      const insideDistance = uniforms.thickness.div(abs(dot(inside, normalView)).max(0.2));
      const exit = origin.add(inside.mul(insideDistance)).toVar();
      const end = exit.add(incident.mul(8)).toVar();
      const startUV = getScreenPosition(exit, cameraProjectionMatrix).toVar();
      const endUV = getScreenPosition(end, cameraProjectionMatrix).toVar();
      const size = textureSize(behindColor);
      const pixelDelta = endUV.sub(startUV).mul(size).abs();
      const steps = max(pixelDelta.x, pixelDelta.y).ceil().clamp(16, 384).toVar();
      const startK = exit.z.negate().reciprocal();
      const endK = end.z.negate().reciprocal();
      const startQ = exit.mul(startK);
      const endQ = end.mul(endK);
      const hitUV = screenUV.toVar();
      const travel = insideDistance.toVar();
      const confidence = float(0).toVar();
      const previousS = float(0).toVar();
      const previousPoint = exit.toVar();
      const previousInFront = bool(true).toVar();
      Loop({ start: 1, end: 385, type: 'int', condition: '<' }, ({ i }) => {
        If(i.greaterThan(int(steps)), () => { Break(); });
        const s = float(i).div(steps).toVar();
        const coord = mix(startUV, endUV, s).toVar();
        If(coord.x.lessThan(0.001).or(coord.y.lessThan(0.001)).or(coord.x.greaterThan(0.999)).or(coord.y.greaterThan(0.999)), () => { Break(); });
        // Perspective-correct interpolation along a one-pixel screen-space ray.
        const point = mix(startQ, endQ, s).div(mix(startK, endK, s)).toVar();
        const d = behindDepth.sample(coord).level(0).r;
        const surface = getViewPosition(coord, d, cameraProjectionMatrixInverse).toVar();
        const pixelWorld = abs(surface.z).mul(2).div(float(size.y).mul(abs(cameraProjectionMatrix.element(1).y)));
        const tolerance = max(pixelWorld.mul(2), 0.008).toVar();
        const error = surface.z.sub(point.z).toVar();
        const stepLength = point.sub(previousPoint).length();
        If(d.lessThan(0.99999).and(error.greaterThanEqual(0)).and(previousInFront)
          .and(error.lessThan(max(tolerance, stepLength.mul(1.5))))
          .and(surface.z.lessThan(origin.z.sub(0.002))), () => {
          const low = previousS.toVar(), high = s.toVar();
          Loop(8, () => {
            const mid = low.add(high).mul(0.5).toVar();
            const midUV = mix(startUV, endUV, mid);
            const midPoint = mix(startQ, endQ, mid).div(mix(startK, endK, mid));
            const surfaceZ = getViewPosition(midUV, behindDepth.sample(midUV).level(0).r, cameraProjectionMatrixInverse).z;
            If(midPoint.z.lessThan(surfaceZ), () => { high.assign(mid); }).Else(() => { low.assign(mid); });
          });
          const hitS = low.add(high).mul(0.5).toVar();
          const candidateUV = mix(startUV, endUV, hitS).toVar();
          const hitPoint = mix(startQ, endQ, hitS).div(mix(startK, endK, hitS)).toVar();
          const hitSurface = getViewPosition(candidateUV, behindDepth.sample(candidateUV).level(0).r, cameraProjectionMatrixInverse);
          const residual = abs(hitPoint.z.sub(hitSurface.z)).toVar();
          // A silhouette discontinuity can bracket a false hit. Reject it
          // instead of sampling an unrelated fold or foreground surface.
          If(residual.lessThan(tolerance), () => {
            hitUV.assign(candidateUV);
            travel.assign(hitPoint.sub(origin).length());
            confidence.assign(float(1).sub(smoothstep(tolerance.mul(0.25), tolerance, residual)));
            Break();
          });
        });
        previousInFront.assign(point.z.greaterThan(surface.z));
        previousS.assign(s);
        previousPoint.assign(point);
      });
      const edgeDistance = hitUV.x.min(hitUV.y).min(hitUV.x.oneMinus()).min(hitUV.y.oneMinus());
      confidence.mulAssign(smoothstep(0, 0.025, edgeDistance));
      // Roughness opens a refraction cone. Sample its projected footprint from
      // the peeled color mip chain: distant folds blur more, and the footprint
      // stays consistent when the render resolution or camera zoom changes.
      const coneRadius = uniforms.roughness.pow(2).mul(travel.add(0.35)).mul(0.65);
      const pixelRadius = coneRadius.mul(abs(cameraProjectionMatrix.element(1).y))
        .mul(float(textureSize(behindColor).y)).div(abs(origin.z).mul(2).max(0.001));
      const lod = log2(pixelRadius.mul(2).add(1));
      const fallback = behindColor.sample(screenUV).level(lod).rgb;
      const behind = mix(fallback, behindColor.sample(hitUV).level(lod).rgb, confidence);
      const f0 = uniforms.ior.sub(1).div(uniforms.ior.add(1)).pow(2);
      const fresnel = f0.add(pow(float(1).sub(abs(dot(normalize(origin.negate()), normalView)).clamp()), 5).mul(f0.oneMinus()));
      const transmitted = behind.mul(mix(vec3(1), pigment, 0.95));
      result.assign(transmitted.mul(fresnel.oneMinus()));
    });
    return result;
  })();
}
