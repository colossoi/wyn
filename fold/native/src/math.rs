pub type V = [f64; 3];
pub fn add(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] + b[i])
}
pub fn sub(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] - b[i])
}
pub fn mul(a: V, b: f64) -> V {
    a.map(|v| v * b)
}
pub fn dot(a: V, b: V) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
pub fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub fn unit(a: V) -> V {
    let length = dot(a, a).sqrt();
    if length > 0.0 {
        mul(a, 1.0 / length)
    } else {
        [0.0; 3]
    }
}
pub const EYE: V = [7.0, 1.0, 0.0];
pub fn basis() -> (V, V, V) {
    let forward = unit(sub([0.0, 3.0, 0.0], EYE));
    let right = unit(cross(forward, [0.0, 1.0, 0.0]));
    (right, cross(right, forward), forward)
}
pub fn ray(x: f64, y: f64, aspect: f64) -> V {
    let (right, up, forward) = basis();
    let t = (55.0f64.to_radians() * 0.5).tan();
    unit(add(forward, add(mul(right, x * t * aspect), mul(up, y * t))))
}
pub fn planes(aspect: f64) -> [[f64; 4]; 3] {
    let (r, u, f) = basis();
    let t = (55.0f64.to_radians() * 0.5).tan();
    [
        unit(add(mul(f, t * aspect), r)),
        unit(sub(mul(f, t * aspect), r)),
        unit(add(mul(f, t), u)),
    ]
    .map(|n| [n[0], n[1], n[2], -dot(n, EYE)])
}
