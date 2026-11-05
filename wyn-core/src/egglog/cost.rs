//! Portable schedule ranking. Facts, candidate enumeration, and hardware rates
//! are separate so calibration never changes legality or a kernel recipe.

#[derive(Clone, Copy, Debug)]
pub struct CostModel {
    pub compute_ops_per_ns: f64,
    pub memory_bytes_per_ns: f64,
    pub barrier_ns: f64,
    pub launch_ns: f64,
    pub readback_ns: f64,
    pub resident_workgroups: u32,
}

impl Default for CostModel {
    fn default() -> Self {
        // Conservative portable priors, not measurements of a particular GPU.
        Self {
            compute_ops_per_ns: 256.0,
            memory_bytes_per_ns: 64.0,
            barrier_ns: 20.0,
            launch_ns: 5000.0,
            readback_ns: 100_000.0,
            resident_workgroups: 256,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CollectiveWork {
    pub elements: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub state_bytes: u64,
    pub map_ops: u64,
    pub combine_ops: u64,
    pub scan: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct CollectiveLaunch {
    pub groups: u32,
    pub chunk_width: u32,
    pub carry_width: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CostEstimate {
    pub compute_ns: f64,
    pub memory_ns: f64,
    pub synchronization_ns: f64,
    pub launch_ns: f64,
    pub global_bytes: u64,
    pub temporary_bytes: u64,
}

impl CostEstimate {
    pub fn total_ns(self) -> f64 {
        self.compute_ns.max(self.memory_ns) + self.synchronization_ns + self.launch_ns
    }
}

impl CostModel {
    pub fn validate(self) -> Result<Self, &'static str> {
        if !self.compute_ops_per_ns.is_finite()
            || self.compute_ops_per_ns <= 0.0
            || !self.memory_bytes_per_ns.is_finite()
            || self.memory_bytes_per_ns <= 0.0
            || !self.barrier_ns.is_finite()
            || self.barrier_ns < 0.0
            || !self.launch_ns.is_finite()
            || self.launch_ns < 0.0
            || !self.readback_ns.is_finite()
            || self.readback_ns < 0.0
            || self.resident_workgroups == 0
        {
            return Err("GPU costs require finite positive throughput/residency and nonnegative latency");
        }
        Ok(self)
    }

    /// Keep a bounded loop local when its iteration costs less than even one
    /// dispatch (plus the predicate launch/readback for conditional control).
    pub fn prefer_local_loop(self, scalar_ops: u64, state_bytes: u64, conditional: bool) -> bool {
        let lanes = f64::from(self.resident_workgroups) * 32.0;
        let local = (scalar_ops as f64 * lanes / self.compute_ops_per_ns)
            .max(state_bytes as f64 * 2.0 * lanes / self.memory_bytes_per_ns);
        let dispatch = self.launch_ns + if conditional { self.launch_ns + self.readback_ns } else { 0.0 };
        local < dispatch
    }

    pub fn estimate(self, work: CollectiveWork, launch: CollectiveLaunch) -> CostEstimate {
        let groups = u64::from(launch.groups.max(1));
        let width = u64::from(launch.chunk_width.max(1));
        let carry = u64::from(launch.carry_width.max(1));
        let n = work.elements;
        let steps = u64::from(launch.chunk_width.max(1).ilog2());
        let carry_steps = u64::from(launch.carry_width.max(1).ilog2());
        let resident = u64::from(self.resident_workgroups.max(1));
        let occupancy = (groups.min(resident) as f64 / resident as f64).max(1.0 / resident as f64);
        let rounds = n.div_ceil(groups * width).max(1);
        let waves = groups.div_ceil(resident);
        let passes = if work.scan { 3u64 } else { 2u64 };
        let temporary_bytes = groups.saturating_mul(work.state_bytes).saturating_mul(passes);
        let global_bytes = n
            .saturating_mul(
                work.input_bytes.saturating_add(work.output_bytes).saturating_add(if work.scan {
                    work.state_bytes.saturating_mul(2)
                } else {
                    0
                }),
            )
            .saturating_add(temporary_bytes.saturating_mul(2));
        let element_ops = n as f64 * (work.map_ops + work.combine_ops * steps.max(1)) as f64;
        let carry_ops = (carry * carry_steps.max(1)) as f64 * work.combine_ops as f64;
        CostEstimate {
            compute_ns: element_ops / (self.compute_ops_per_ns * occupancy)
                + carry_ops / (self.compute_ops_per_ns / resident as f64),
            memory_ns: global_bytes as f64 / (self.memory_bytes_per_ns * occupancy),
            synchronization_ns: ((rounds * waves * steps + carry_steps) * 2) as f64 * self.barrier_ns,
            launch_ns: passes as f64 * self.launch_ns,
            global_bytes,
            temporary_bytes,
        }
    }

    pub fn select(
        self,
        work: CollectiveWork,
        chunk_width: u32,
        maximum_groups: u32,
    ) -> Result<CollectiveLaunch, &'static str> {
        self.validate()?;
        if !chunk_width.is_power_of_two() || maximum_groups == 0 || maximum_groups > 256 {
            return Err("portable collectives require a power-of-two width and 1..=256 carry groups");
        }
        let useful =
            work.elements.div_ceil(u64::from(chunk_width)).max(1).min(u64::from(maximum_groups)) as u32;
        // All candidates use the same portable recipe and fit its carry stage.
        (1..=useful)
            .map(|groups| CollectiveLaunch {
                groups,
                chunk_width,
                carry_width: groups.next_power_of_two().max(32),
            })
            .min_by(|a, b| {
                self.estimate(work, *a)
                    .total_ns()
                    .total_cmp(&self.estimate(work, *b).total_ns())
                    .then_with(|| b.groups.cmp(&a.groups))
            })
            .ok_or("no legal collective launch")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_loop_guard_accounts_for_work_memory_and_readbacks() {
        let model = CostModel::default();
        assert!(model.prefer_local_loop(80, 20, true));
        assert!(!model.prefer_local_loop(1_000_000, 20, true));
        assert!(!model.prefer_local_loop(80, 1_000_000, true));
        let immediate = CostModel {
            launch_ns: 0.0,
            readback_ns: 0.0,
            ..model
        };
        assert!(!immediate.prefer_local_loop(80, 20, true));
        assert!(CostModel {
            readback_ns: f64::NAN,
            ..model
        }
        .validate()
        .is_err());
    }

    #[test]
    fn each_cost_component_responds_to_its_inputs() {
        let model = CostModel::default();
        let work = CollectiveWork {
            elements: 8192,
            input_bytes: 8,
            output_bytes: 8,
            state_bytes: 64,
            map_ops: 16,
            combine_ops: 8,
            scan: true,
        };
        let launch = CollectiveLaunch {
            groups: 128,
            chunk_width: 64,
            carry_width: 128,
        };
        let base = model.estimate(work, launch);
        assert!(
            model.estimate(CollectiveWork { map_ops: 64, ..work }, launch).compute_ns > base.compute_ns
        );
        assert!(
            model
                .estimate(
                    CollectiveWork {
                        input_bytes: 64,
                        ..work
                    },
                    launch
                )
                .memory_ns
                > base.memory_ns
        );
        assert_eq!(
            CostModel {
                barrier_ns: model.barrier_ns * 2.0,
                ..model
            }
            .estimate(work, launch)
            .synchronization_ns,
            base.synchronization_ns * 2.0
        );
        assert_eq!(
            CostModel {
                launch_ns: model.launch_ns * 2.0,
                ..model
            }
            .estimate(work, launch)
            .launch_ns,
            base.launch_ns * 2.0
        );
        assert!(CostModel {
            memory_bytes_per_ns: f64::NAN,
            ..model
        }
        .validate()
        .is_err());
        for n in [0, 1, 257, 4320, 100_000] {
            let choice = model.select(CollectiveWork { elements: n, ..work }, 64, 256).unwrap();
            assert!(choice.groups >= 1 && choice.groups <= 256);
            assert!(choice.carry_width.is_power_of_two() && choice.carry_width >= choice.groups);
        }
        assert!(model.select(work, 0, 256).is_err());
        assert!(model.select(work, 64, 0).is_err());
    }
}
