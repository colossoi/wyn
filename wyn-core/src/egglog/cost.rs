//! Portable schedule ranking. Facts, candidate enumeration, and hardware rates
//! are separate so calibration never changes legality or a kernel recipe.

use egglog_engine::ast::Span;
use egglog_engine::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog_engine::prelude::BaseSort;
use egglog_engine::sort::{BigIntSort, BoolSort, I64Sort, Z};
use egglog_engine::{Core, EGraph, Primitive, PurePrim, PureState, Value};

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

    pub fn estimate(self, work: CollectiveWork, chunk_width: u32, groups: u32) -> f64 {
        let carry_width = groups.next_power_of_two().max(32);
        let groups = u64::from(groups.max(1));
        let width = u64::from(chunk_width.max(1));
        let carry = u64::from(carry_width.max(1));
        let n = work.elements;
        let steps = u64::from(chunk_width.max(1).ilog2());
        let carry_steps = u64::from(carry_width.max(1).ilog2());
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
        let compute = element_ops / (self.compute_ops_per_ns * occupancy)
            + carry_ops / (self.compute_ops_per_ns / resident as f64);
        let memory = global_bytes as f64 / (self.memory_bytes_per_ns * occupancy);
        let synchronization = ((rounds * waves * steps + carry_steps) * 2) as f64 * self.barrier_ns;
        compute.max(memory) + synchronization + passes as f64 * self.launch_ns
    }

    pub fn select(
        self,
        work: CollectiveWork,
        chunk_width: u32,
        maximum_groups: u32,
    ) -> Result<u32, &'static str> {
        self.validate()?;
        if !chunk_width.is_power_of_two() || maximum_groups == 0 || maximum_groups > 256 {
            return Err("portable collectives require a power-of-two width and 1..=256 carry groups");
        }
        let useful =
            work.elements.div_ceil(u64::from(chunk_width)).max(1).min(u64::from(maximum_groups)) as u32;
        // All candidates use the same portable recipe and fit its carry stage.
        (1..=useful)
            .min_by(|a, b| {
                self.estimate(work, chunk_width, *a)
                    .total_cmp(&self.estimate(work, chunk_width, *b))
                    .then_with(|| b.cmp(a))
            })
            .ok_or("no legal collective launch")
    }
}

// Only calibrated arithmetic crosses this boundary; primitives cannot read graph facts.
pub(super) fn register(graph: &mut EGraph, costs: CostModel, collective: bool) {
    graph.add_pure_primitive(Rank { costs, collective }, None);
}
#[derive(Clone)]
struct Rank {
    costs: CostModel,
    collective: bool,
}
impl Primitive for Rank {
    fn name(&self) -> &str {
        if self.collective {
            "wyn-collective-groups"
        } else {
            "wyn-local-loop"
        }
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let mut sorts = vec![BigIntSort.to_arcsort(); if self.collective { 6 } else { 2 }];
        sorts.push(BoolSort.to_arcsort());
        if self.collective {
            sorts.extend([I64Sort.to_arcsort(), I64Sort.to_arcsort()]);
        } else {
            sorts.push(BoolSort.to_arcsort());
        }
        SimpleTypeConstraint::new(self.name(), sorts, span.clone()).into_box()
    }
}
impl PurePrim for Rank {
    fn apply<'a, 'db>(&self, state: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let number = |value| u64::try_from(&*state.value_to_base::<Z>(value)).ok();
        if self.collective {
            let &[n, input, output, bytes, map, combine, scan, width] = args else {
                unreachable!("invalid collective cost arity")
            };
            let work = CollectiveWork {
                elements: number(n)?,
                input_bytes: number(input)?,
                output_bytes: number(output)?,
                state_bytes: number(bytes)?,
                map_ops: number(map)?,
                combine_ops: number(combine)?,
                scan: state.value_to_base(scan),
            };
            let width = u32::try_from(state.value_to_base::<i64>(width)).ok()?;
            let groups = self.costs.select(work, width, 256).ok()?;
            Some(state.base_to_value(i64::from(groups)))
        } else {
            let &[work, bytes, conditional] = args else {
                unreachable!("invalid loop cost arity")
            };
            Some(state.base_to_value(self.costs.prefer_local_loop(
                number(work)?,
                number(bytes)?,
                state.value_to_base(conditional),
            )))
        }
    }
}

#[cfg(test)]
#[path = "cost_tests.rs"]
mod tests;
