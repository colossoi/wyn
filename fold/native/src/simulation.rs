// MIT, Copyright (c) 2026 mrdoob. Native host lifecycle from upstream/cloth-simulation.js.
// Constraint equations execute exclusively in the compiled Wyn kernel.
use crate::math::*;
pub const N: usize = 2483;
pub const ROWS: usize = 191;
pub const STRIDE: usize = 13;
pub const SPACING: f64 = 3.8 / 12.0;
pub const DT: f64 = 1.0 / 120.0;
#[derive(serde::Serialize, serde::Deserialize)]
pub struct GrabParticle {
    pub index: usize,
    pub offset: V,
    pub compliance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_host_matches_upstream_across_feeding_grabbing_resize_and_retirement() {
        let mut sim = Simulation::new();
        sim.viewport(1.0, 0.35);
        let mut resized = Simulation::new();
        resized.viewport(1.4, 0.35);
        let input = serde_json::json!({"walls":sim.walls,"retained":sim.target_retained,"resized":resized.walls,"resizedRetained":resized.target_retained});
        let output = std::process::Command::new("node")
            .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/../test/host_oracle.mjs"))
            .arg(input.to_string())
            .output()
            .expect("Node upstream oracle");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        sim.reset(0.35, 2.0);
        for i in 0..90 {
            if i == 10 {
                assert!(sim.begin_grab(10 * 13 + 6));
                sim.grab.as_mut().unwrap().target = [0.2, 3.5, 0.7];
            }
            if i == 40 {
                sim.grab = None;
            }
            if i == 50 {
                sim.viewport(1.4, 0.35);
            }
            sim.prepare(2.0, 0.35);
            sim.finish();
        }
        for (name, actual) in [
            ("positions", sim.p.iter().flatten().copied().collect::<Vec<_>>()),
            ("velocities", sim.v.iter().flatten().copied().collect()),
            ("birth", sim.birth.clone()),
            ("edges", sim.edges.iter().flatten().map(|&x| x as f64).collect()),
        ] {
            let values = expected[name].as_array().unwrap();
            assert_eq!(actual.len(), values.len());
            let error =
                actual.iter().zip(values).map(|(a, b)| (a - b.as_f64().unwrap()).abs()).fold(0.0, f64::max);
            assert!(error < 1e-9, "{name}: error {error}");
        }
        assert_eq!(sim.active, expected["active"].as_u64().unwrap() as usize);
        assert_eq!(sim.emitted, expected["emitted"].as_u64().unwrap() as usize);
        for (name, actual) in [("feed", sim.feed), ("time", sim.time), ("retained", sim.retained)] {
            assert!(
                (actual - expected[name].as_f64().unwrap()).abs() < 1e-12,
                "{name}"
            );
        }
    }
}
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Grab {
    pub index: usize,
    pub target: V,
    pub anchor: V,
    pub particles: Vec<GrabParticle>,
}
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Simulation {
    pub p: Vec<V>,
    pub old: Vec<V>,
    pub v: Vec<V>,
    pub birth: Vec<f64>,
    pub weights: Vec<f64>,
    pub edges: Vec<[f32; 4]>,
    pub active: usize,
    pub retained: f64,
    pub target_retained: f64,
    pub time: f64,
    pub feed: f64,
    pub emitted: usize,
    pub accumulator: f64,
    pub walls: [[f64; 4]; 3],
    pub walls_changed: bool,
    pub grab: Option<Grab>,
}
impl Simulation {
    pub fn new() -> Self {
        let mut edges = Vec::new();
        for r in 0..ROWS {
            for c in 0..STRIDE {
                let i = r * STRIDE + c;
                let mut edge = |a: usize, b: usize, length: f64, kind: f32| {
                    edges.push([a as f32, b as f32, length as f32, kind])
                };
                if c < 12 {
                    edge(i, i + 1, SPACING, 0.0)
                }
                if r + 1 < ROWS {
                    edge(i, i + STRIDE, SPACING, 0.0)
                }
                if r + 1 < ROWS && c < 12 {
                    edge(i, i + STRIDE + 1, SPACING * 2.0f64.sqrt(), 1.0);
                    edge(i + 1, i + STRIDE, SPACING * 2.0f64.sqrt(), 1.0)
                }
                if c + 2 <= 12 {
                    edge(i, i + 2, SPACING * 2.0, 2.0)
                }
                if r + 2 < ROWS {
                    edge(i, i + STRIDE * 2, SPACING * 2.0, 2.0)
                }
            }
        }
        assert_eq!(edges.len(), 13880);
        Self {
            p: vec![[0.0; 3]; N],
            old: vec![[0.0; 3]; N],
            v: vec![[0.0; 3]; N],
            birth: vec![0.0; ROWS],
            weights: vec![0.0; N],
            edges,
            active: 0,
            retained: 191.0,
            target_retained: 191.0,
            time: 0.0,
            feed: 0.0,
            emitted: 0,
            accumulator: 0.0,
            walls: planes(1.0),
            walls_changed: true,
            grab: None,
        }
    }
    pub fn reset(&mut self, thickness: f64, speed: f64) {
        self.retained = self.target_retained;
        self.grab = None;
        self.time = 0.0;
        self.accumulator = 0.0;
        self.feed = 0.0;
        self.emitted = 0;
        self.p.fill([0.0; 3]);
        self.v.fill([0.0; 3]);
        self.birth.fill(0.0);
        self.weights.fill(0.0);
        self.walls_changed = true;
        let length = (7.5 - thickness * 0.5).max(0.0);
        let whole = (length / SPACING).floor();
        self.active = ROWS.min(whole as usize + 2);
        self.feed = (length - whole * SPACING).max(0.0);
        let rate = if speed > 0.0 { speed * 1.5 } else { 1.5 };
        for r in 0..self.active {
            self.birth[r] = -(r as f64 * SPACING + self.feed) / rate;
            let bow = if r < 2 { 0.0 } else { 0.005 * ((r - 1) as f64 / (self.active - 2) as f64).powi(4) };
            for c in 0..STRIDE {
                self.p[r * STRIDE + c] = [
                    c as f64 * SPACING - 1.9,
                    7.5 + (1.0 - r as f64) * SPACING - self.feed,
                    bow,
                ];
            }
        }
        self.old.clone_from(&self.p);
    }
    pub fn visibility(&self, row: f64) -> f64 {
        let t = ((row + self.feed / SPACING - (self.retained - 10.0)) / 7.0).clamp(0.0, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    }
    pub fn recycle(&self, row: f64) -> f64 {
        let t = ((row + self.feed / SPACING - (self.retained - 22.0)) / 20.0).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
    pub fn viewport(&mut self, aspect: f64, thickness: f64) {
        let planes = planes(aspect);
        if self.walls.iter().flatten().zip(planes.iter().flatten()).any(|(a, b)| (a - b).abs() >= 1e-9) {
            self.walls = planes;
            self.walls_changed = true;
        }
        let clearance = thickness * 0.5 + 0.025;
        let bottom =
            if planes[2][1] > 1e-6 { ((clearance - planes[2][3]) / planes[2][1]).max(0.0) } else { 0.0 };
        let width = |y: f64| {
            let mut low = f64::NEG_INFINITY;
            let mut high = f64::INFINITY;
            for p in &planes[..2] {
                if p[2].abs() < 1e-6 {
                    return f64::INFINITY;
                }
                let z = (clearance - p[1] * y - p[3]) / p[2];
                if p[2] > 0.0 {
                    low = low.max(z)
                } else {
                    high = high.min(z)
                }
            }
            (high - low).max(0.0)
        };
        let area = (width(bottom) + width(3.0)) * 0.5 * (3.0 - bottom).max(0.0);
        let length = if area.is_finite() {
            (7.5 + area * 0.35 / thickness.max(SPACING * 0.85) + 22.0 * SPACING).min(190.0 * SPACING)
        } else {
            190.0 * SPACING
        };
        let minimum = (7.5 / SPACING).ceil() + 24.0;
        self.target_retained = (length / SPACING + 1.0).clamp(minimum, 191.0);
    }
    fn moving_walls(&mut self, thickness: f64) {
        let clearance = thickness * 0.5 + 0.025;
        for i in 26..self.active * STRIDE {
            for _ in 0..2 {
                for wall in 0..5 {
                    if wall == 2 && self.recycle((i / STRIDE) as f64) > 0.0 {
                        continue;
                    }
                    let plane = if wall < 3 {
                        self.walls[wall]
                    } else {
                        [if wall == 3 { 1.0 } else { -1.0 }, 0.0, 0.0, 2.85]
                    };
                    let n = [plane[0], plane[1], plane[2]];
                    let distance = dot(self.p[i], n) + plane[3];
                    if distance < clearance {
                        self.p[i] = add(self.p[i], mul(n, clearance - distance));
                        self.v[i] = sub(self.v[i], mul(n, dot(self.v[i], n).min(0.0)));
                    }
                }
            }
        }
    }
    pub fn prepare(&mut self, speed: f64, thickness: f64) {
        self.time += DT;
        let t = self.time;
        let source_z = 3.8
            * (0.12 * (t * 0.71).sin()
                + 0.045 * ((t * 1.23 + 1.7).sin() - 1.7f64.sin())
                + 0.025 * ((t * 2.13 + 4.2).sin() - 4.2f64.sin()));
        if self.active == 0 {
            if speed <= 0.0 {
                return;
            }
            self.active = 2;
            for r in 0..2 {
                self.birth[r] = self.time;
                for c in 0..STRIDE {
                    self.p[r * STRIDE + c] = [
                        c as f64 * SPACING - 1.9,
                        7.5 + (1.0 - r as f64) * SPACING,
                        source_z,
                    ];
                }
            }
        }
        if self.retained > self.target_retained {
            self.retained = self.target_retained.max(self.retained - 12.0 * DT)
        } else {
            self.retained = self.target_retained.min(self.retained + speed * 1.5 * DT / SPACING)
        }
        self.feed += speed * 1.5 * DT;
        while self.feed >= SPACING {
            self.feed -= SPACING;
            self.p.copy_within(..N - STRIDE, STRIDE);
            self.v.copy_within(..N - STRIDE, STRIDE);
            self.v[..STRIDE].fill([0.0; 3]);
            self.birth.copy_within(..ROWS - 1, 1);
            self.birth[0] = self.time;
            self.active = ROWS.min(self.active + 1);
            if self.emitted == 0 && self.active == 3 {
                for c in 0..STRIDE {
                    self.p[26 + c][2] += 0.005;
                }
            }
            self.emitted += 1;
            if let Some(grab) = &mut self.grab {
                grab.index += STRIDE;
                for p in &mut grab.particles {
                    p.index += STRIDE;
                }
            }
        }
        self.active = self.active.min(self.retained.ceil() as usize);
        if self.walls_changed {
            self.moving_walls(thickness);
            self.walls_changed = false;
        }
        self.weights[..self.active * STRIDE].fill(1.0);
        self.weights[..26].fill(0.0);
        for r in 0..2 {
            for c in 0..STRIDE {
                self.p[r * STRIDE + c] = [
                    c as f64 * SPACING - 1.9,
                    7.5 + (1.0 - r as f64) * SPACING - self.feed,
                    source_z,
                ];
            }
        }
        self.old.clone_from(&self.p);
        let damping = (-1.3 * DT).exp();
        for i in 26..self.active * STRIDE {
            self.v[i] = mul(self.v[i], damping);
            self.v[i][1] -= 9.81 * DT;
            self.p[i] = add(self.p[i], mul(self.v[i], DT));
        }
        let cancel = self.grab.as_ref().is_some_and(|g| {
            g.index >= self.active * STRIDE || self.visibility((g.index / STRIDE) as f64) < 0.05
        });
        if cancel {
            self.grab = None
        }
        if let Some(g) = &mut self.grab {
            g.target[1] = g.target[1].max(thickness * 0.5);
            let delta = sub(g.target, g.anchor);
            let distance = dot(delta, delta).sqrt();
            let blend = (1.0 - (-30.0 * DT).exp()).min(8.0 * DT / distance.max(1e-9));
            g.anchor = add(g.anchor, mul(delta, blend));
        }
    }
    pub fn finish(&mut self) {
        for i in 26..self.active * STRIDE {
            self.v[i] = mul(sub(self.p[i], self.old[i]), 1.0 / DT)
        }
    }
    pub fn begin_grab(&mut self, index: usize) -> bool {
        let row = index / STRIDE;
        let col = index % STRIDE;
        if row < 2 || row >= self.active || self.visibility(row as f64) < 0.1 {
            return false;
        }
        let center = self.p[index];
        let mut particles = Vec::new();
        for dr in -1i32..=1 {
            for dc in -1i32..=1 {
                let r = row as i32 + dr;
                let c = col as i32 + dc;
                if r < 2 || r >= self.active as i32 || !(0..13).contains(&c) {
                    continue;
                }
                let i = r as usize * STRIDE + c as usize;
                particles.push(GrabParticle {
                    index: i,
                    offset: sub(self.p[i], center),
                    compliance: if dr == 0 && dc == 0 {
                        0.000002
                    } else {
                        0.00008 * (dr * dr + dc * dc) as f64
                    },
                });
            }
        }
        self.grab = Some(Grab {
            index,
            target: center,
            anchor: center,
            particles,
        });
        true
    }
    pub fn transport(
        &self,
        thickness: f64,
        stiffness: f64,
    ) -> (Vec<[f32; 4]>, Vec<[f32; 4]>, [f32; 44], [[f32; 4]; 9]) {
        let p = (0..N)
            .map(|i| {
                [
                    self.p[i][0] as f32,
                    self.p[i][1] as f32,
                    self.p[i][2] as f32,
                    self.weights[i] as f32,
                ]
            })
            .collect();
        let old = self.old.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, 0.0]).collect();
        let mut params = [0.0; 44];
        for i in 0..12 {
            params[i] = self.walls[i / 4][i % 4] as f32;
        }
        params[12..20].copy_from_slice(&[
            (self.active * STRIDE) as f32,
            DT as f32,
            thickness as f32,
            stiffness as f32,
            self.feed as f32,
            self.retained as f32,
            3.0,
            0.0,
        ]);
        let mut offsets = [[0.0; 4]; 9];
        if let Some(g) = &self.grab {
            for a in 0..3 {
                params[20 + a] = g.anchor[a] as f32
            }
            params[23] = g.particles.len() as f32;
            for (i, p) in g.particles.iter().enumerate() {
                for a in 0..3 {
                    offsets[i][a] = p.offset[a] as f32
                }
                params[24 + i * 2] = p.index as f32;
                params[25 + i * 2] = p.compliance as f32;
            }
        }
        (p, old, params, offsets)
    }
}
