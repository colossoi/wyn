import { vec3, vec4, exp2, dot, mix, clamp } from 'three/tsl';

// Display-referred grading, after tone mapping and before antialiasing.
export function colorGrade(source, uniforms) {
  const balanced = source.rgb.mul(exp2(vec3(0.22, 0, -0.22).mul(uniforms.temperature)));
  const luminance = dot(balanced, vec3(0.2126, 0.7152, 0.0722));
  const saturated = mix(vec3(luminance), balanced, uniforms.saturation);
  return vec4(clamp(saturated.sub(0.5).mul(uniforms.contrast).add(0.5), 0, 1), source.a);
}
