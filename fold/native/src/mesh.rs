// MIT, Copyright (c) 2026 mrdoob. Port of upstream/sheet.js shell construction.
use crate::{math::*, simulation::Simulation};
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub p: [f32; 3],
    pub n: [f32; 3],
    pub birth: f32,
    pub fade: f32,
}
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub surface_indices: Vec<u32>,
    pub perimeter: Vec<usize>,
}
const C: usize = 24;
const R: usize = 570;
const S: usize = 25;
const COUNT: usize = (R + 1) * S;
impl Mesh {
    pub fn new() -> Self {
        let mut surface_indices = Vec::new();
        // PlaneGeometry's exact triangle winding and diagonal.
        for r in 0..R {
            for c in 0..C {
                let a = (r * S + c) as u32;
                let b = a + S as u32;
                let d = a + 1;
                let e = b + 1;
                surface_indices.extend([a, b, d, b, e, d]);
            }
        }
        let mut indices = surface_indices.clone();
        for t in surface_indices.chunks_exact(3) {
            indices.extend([COUNT as u32 + t[0], COUNT as u32 + t[2], COUNT as u32 + t[1]]);
        }
        let mut perimeter = Vec::new();
        for c in 0..C {
            perimeter.push(c)
        }
        for r in 0..R {
            perimeter.push(r * S + C)
        }
        for c in (1..=C).rev() {
            perimeter.push(R * S + c)
        }
        for r in (1..=R).rev() {
            perimeter.push(r * S)
        }
        let len = perimeter.len();
        let index = |ring: usize, i: usize| -> u32 {
            if ring == 0 {
                perimeter[i] as u32
            } else if ring == 6 {
                (COUNT + perimeter[i]) as u32
            } else {
                (COUNT * 2 + (ring - 1) * len + i) as u32
            }
        };
        for ring in 0..6 {
            for i in 0..len {
                let next = (i + 1) % len;
                let a = index(ring, i);
                let b = index(ring + 1, i);
                let c = index(ring, next);
                let d = index(ring + 1, next);
                indices.extend([a, c, b, c, d, b]);
            }
        }
        Self {
            vertices: vec![Vertex::zeroed(); COUNT * 2 + len * 5],
            indices,
            surface_indices,
            perimeter,
        }
    }
    pub fn update(&mut self, sim: &Simulation, thickness: f64) {
        if sim.active < 2 {
            return;
        }
        let mut positions = vec![[0.0; 3]; COUNT];
        let mut normals = vec![[0.0; 3]; COUNT];
        let sample = |r: i32, c: usize| sim.p[r.clamp(0, sim.active as i32 - 1) as usize * 13 + c];
        let interpolate = |r: i32, c: usize, t: f64| -> V {
            let a = sample(r - 1, c);
            let b = sample(r, c);
            let d = sample(r + 1, c);
            let e = sample(r + 2, c);
            std::array::from_fn(|i| {
                b[i] + 0.5
                    * t
                    * (d[i] - a[i]
                        + t * (2.0 * a[i] - 5.0 * b[i] + 4.0 * d[i] - e[i]
                            + t * (3.0 * (b[i] - d[i]) + e[i] - a[i])))
            })
        };
        for row in 0..=R {
            let source = row as f64 / R as f64 * (sim.active - 1) as f64;
            let r = source.floor() as usize;
            let t = source - r as f64;
            let birth = sim.birth[r] * (1.0 - t) + sim.birth[(r + 1).min(sim.active - 1)] * t;
            let fade = sim.visibility(source);
            for col in 0..=C {
                let c = col / 2;
                let f = (col % 2) as f64 * 0.5;
                let i = row * S + col;
                positions[i] = add(
                    mul(interpolate(r as i32, c, t), 1.0 - f),
                    mul(interpolate(r as i32, (c + 1).min(12), t), f),
                );
                // Upstream position/normal attributes round to Float32Array before normals.
                positions[i] = positions[i].map(|x| x as f32 as f64);
                for index in [i, COUNT + i] {
                    self.vertices[index].birth = birth as f32;
                    self.vertices[index].fade = fade as f32;
                }
            }
        }
        for triangle in self.surface_indices.chunks_exact(3) {
            let a = triangle[0] as usize;
            let b = triangle[1] as usize;
            let c = triangle[2] as usize;
            let n = cross(sub(positions[c], positions[b]), sub(positions[a], positions[b]));
            for i in [a, b, c] {
                normals[i] = add(normals[i], n).map(|x| x as f32 as f64);
            }
        }
        for i in 0..COUNT {
            normals[i] = unit(normals[i]).map(|x| x as f32 as f64);
            for side in 0..2 {
                let n = mul(normals[i], if side == 0 { 1.0 } else { -1.0 });
                let p = add(positions[i], mul(n, thickness * 0.5));
                self.vertices[side * COUNT + i].p = p.map(|x| x as f32);
                self.vertices[side * COUNT + i].n = n.map(|x| x as f32);
            }
        }
        let len = self.perimeter.len();
        for i in 0..len {
            let source = self.perimeter[i];
            let previous = self.perimeter[(i + len - 1) % len];
            let next = self.perimeter[(i + 1) % len];
            let tangent = unit(sub(positions[next], positions[previous]));
            let outward = unit(cross(normals[source], tangent));
            for ring in 1..6 {
                let angle = ring as f64 / 6.0 * std::f64::consts::PI;
                let n = add(mul(normals[source], angle.cos()), mul(outward, angle.sin()));
                let target = COUNT * 2 + (ring - 1) * len + i;
                self.vertices[target] = Vertex {
                    p: add(positions[source], mul(n, thickness * 0.5)).map(|x| x as f32),
                    n: n.map(|x| x as f32),
                    birth: self.vertices[source].birth,
                    fade: self.vertices[source].fade,
                };
            }
        }
    }
    pub fn pick(&self, origin: V, direction: V) -> Option<(V, usize, usize)> {
        let mut hit = None;
        let mut best = f64::INFINITY;
        for triangle in self.surface_indices.chunks_exact(3) {
            let p = [triangle[0], triangle[1], triangle[2]]
                .map(|i| self.vertices[i as usize].p.map(|x| x as f64));
            let e1 = sub(p[1], p[0]);
            let e2 = sub(p[2], p[0]);
            let h = cross(direction, e2);
            let determinant = dot(e1, h);
            if determinant.abs() < 1e-9 {
                continue;
            }
            let s = sub(origin, p[0]);
            let u = dot(s, h) / determinant;
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            let q = cross(s, e1);
            let v = dot(direction, q) / determinant;
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let distance = dot(e2, q) / determinant;
            if distance <= 0.0 || distance >= best {
                continue;
            }
            let point = add(origin, mul(direction, distance));
            if point[1] < 0.0 {
                continue;
            }
            best = distance;
            let indices = [triangle[0] as usize, triangle[1] as usize, triangle[2] as usize];
            let row = indices
                .iter()
                .map(|i| (i / S) as f64)
                .zip([1.0 - u - v, u, v])
                .map(|(i, w)| i * w)
                .sum::<f64>()
                / R as f64;
            let col = indices
                .iter()
                .map(|i| (i % S) as f64)
                .zip([1.0 - u - v, u, v])
                .map(|(i, w)| i * w)
                .sum::<f64>()
                / C as f64;
            hit = Some((point, (row * 1000000.0) as usize, (col * 12.0).round() as usize));
        }
        hit
    }
}
use bytemuck::Zeroable;
