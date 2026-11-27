use super::{resource, string};
use crate::{
    Allocation, BufferSource, DispatchLoopControl, Entry, HostError, Operation, Program, ResourceId,
    ScalarExpr, ScalarType,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

struct Carry {
    current: ResourceId,
    next: ResourceId,
    bytes: u64,
}

struct DeviceLoop {
    maximum: u32,
    predicate: usize,
    carried: Vec<Carry>,
    dispatches: Vec<(usize, [u32; 3])>,
}

impl Program {
    fn device_loop(
        &self,
        pipeline: usize,
        region: usize,
        test: &[Operation],
        body: &[Operation],
    ) -> Result<Option<DeviceLoop>, HostError> {
        let loop_ = &self.interface.dispatch_loops[region];
        let DispatchLoopControl::While {
            initial_condition: Some(true),
            condition:
                ScalarExpr::Read {
                    source,
                    offset: 0,
                    ty: ScalarType::Bool,
                },
            maximum: Some(maximum @ 1..=16),
        } = &loop_.control
        else {
            return Ok(None);
        };
        if !test.is_empty() {
            return Ok(None);
        }
        let mut allocations = BTreeMap::new();
        let mut dispatches = Vec::new();
        for operation in body {
            match operation {
                Operation::Allocate(Allocation::Buffer { resource, bytes }) => {
                    let Some(bytes) = bytes.constant()?.and_then(|bytes| u64::try_from(bytes).ok()) else {
                        return Ok(None);
                    };
                    if bytes == 0 || bytes % 4 != 0 || allocations.insert(*resource, bytes).is_some() {
                        return Ok(None);
                    }
                }
                Operation::Dispatch {
                    pipeline: owner,
                    stage,
                    groups,
                } if *owner == pipeline => {
                    let mut dimensions = [0; 3];
                    for (out, expr) in dimensions.iter_mut().zip(groups) {
                        let Some(group @ 1..=65_535) = expr.constant()? else {
                            return Ok(None);
                        };
                        *out = group as u32;
                    }
                    dispatches.push((*stage, dimensions));
                }
                _ => return Ok(None),
            }
        }
        // Limit extra per-stage bindings to the storage limit already required
        // by at least one dispatch in the original loop.
        let max_bindings = dispatches
            .iter()
            .map(|(stage, _)| {
                self.parameter_indices(pipeline, Some(*stage))
                    .into_iter()
                    .filter(|&index| {
                        matches!(
                            self.bindings(pipeline)[index],
                            crate::Binding::StorageBuffer { .. }
                        )
                    })
                    .count()
            })
            .max()
            .unwrap_or(0);
        let predicate = self.scalar_resource(pipeline, source)?;
        let mut carried = Vec::new();
        let mut unique = BTreeSet::new();
        let mut predicate_index = None;
        for carry in &loop_.carried {
            let (BufferSource::Binding(initial), BufferSource::Binding(next)) =
                (&carry.initial, &carry.next)
            else {
                return Ok(None);
            };
            let current = self.scalar_resource(pipeline, &carry.binding)?;
            let initial = self.scalar_resource(pipeline, initial)?;
            let next = self.scalar_resource(pipeline, next)?;
            let Some(bytes) = allocations.remove(&next) else {
                return Ok(None);
            };
            if allocations.contains_key(&initial) || !unique.insert(current) || !unique.insert(next) {
                return Ok(None);
            }
            if current == predicate {
                if bytes != 4 {
                    return Ok(None);
                }
                predicate_index = Some(carried.len());
            }
            carried.push(Carry { current, next, bytes });
        }
        let Some(predicate) = predicate_index else {
            return Ok(None);
        };
        if !allocations.is_empty()
            || dispatches.is_empty()
            || carried.len() + 3 > max_bindings
            || carried.len() * 2 > max_bindings
        {
            return Ok(None);
        }
        Ok(Some(DeviceLoop {
            maximum: *maximum,
            predicate,
            carried,
            dispatches,
        }))
    }

    pub(super) fn write_device_loop_kernels(
        &self,
        out: &mut String,
        operations: &[Operation],
    ) -> Result<(), HostError> {
        for operation in operations {
            if let Operation::Loop {
                pipeline,
                region,
                test,
                body,
            } = operation
            {
                if let Some(plan) = self.device_loop(*pipeline, *region, test, body)? {
                    plan.write_kernels(out, *region)?;
                } else {
                    self.write_device_loop_kernels(out, test)?;
                    self.write_device_loop_kernels(out, body)?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn write_device_loop(
        &self,
        out: &mut String,
        entry: &Entry,
        pipeline: usize,
        region: usize,
        test: &[Operation],
        body: &[Operation],
    ) -> Result<bool, HostError> {
        let Some(plan) = self.device_loop(pipeline, region, test, body)? else {
            return Ok(false);
        };
        let repeated = &self.interface.dispatch_loops[region];
        self.write_buffer_bindings(
            out,
            entry,
            pipeline,
            repeated.carried.iter().map(|c| (&c.binding, &c.initial)),
        )?;
        for operation in body {
            if let Operation::Allocate(allocation) = operation {
                self.write_allocation(out, allocation)?;
            }
        }
        let bytes = 12 * (plan.dispatches.len() + 1);
        writeln!(out, "    (let* (")?;
        for (i, carry) in plan.carried.iter().enumerate() {
            writeln!(out, "      (alternate-{i} (gpu-alloc (i64 {})))", carry.bytes)?;
        }
        for round in 1..=plan.maximum {
            writeln!(out, "      (grid-{round} (gpu-alloc (i64 {bytes})))")?;
        }
        writeln!(out, "      (publish-grid (gpu-alloc (i64 12))))")?;
        let mut primary = Vec::new();
        let mut alternate = Vec::new();
        for (i, carry) in plan.carried.iter().enumerate() {
            primary.push(resource(carry.next));
            alternate.push(format!("alternate-{i}"));
        }
        writeln!(out, "    (let* (")?;
        self.write_views(out, pipeline)?;
        writeln!(out, "    )")?;
        for round in 0..plan.maximum {
            let (prior, next) =
                if round % 2 == 0 { (&alternate, &primary) } else { (&primary, &alternate) };
            let mut remap = BTreeMap::new();
            for (i, carry) in plan.carried.iter().enumerate() {
                let current = resource(carry.current);
                remap.insert(
                    current.clone(),
                    if round == 0 { current } else { prior[i].clone() },
                );
                remap.insert(resource(carry.next), next[i].clone());
            }
            for (slot, (stage, size)) in plan.dispatches.iter().enumerate() {
                let args = self.arguments(pipeline, Some(*stage))?;
                let args = args
                    .split_whitespace()
                    .map(|name| remap.get(name).map_or(name, String::as_str))
                    .collect::<Vec<_>>()
                    .join(" ");
                let launch = launch(round, slot, *size);
                writeln!(
                    out,
                    "      (gpu-dispatch 'kernel-{pipeline}-{stage} {launch} :args (list {args}))"
                )?;
            }
            let launch = launch(round, plan.dispatches.len(), [1, 1, 1]);
            let parity = if round % 2 == 0 { "primary" } else { "alternate" };
            writeln!(out, "      (gpu-dispatch 'device-loop-{region}-{parity} {launch} :args (list {} grid-{} publish-grid {}))",
                next[plan.predicate], round+1, prior.join(" "))?;
        }
        let copy_arguments = alternate
            .iter()
            .zip(&primary)
            .flat_map(|(a, b)| [a.as_str(), b.as_str()])
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(out, "      (gpu-dispatch 'device-loop-{region}-publish :indirect (list publish-grid (i64 0)) :args (list {copy_arguments}))")?;
        writeln!(out, "    )")?;
        self.write_buffer_bindings(
            out,
            entry,
            pipeline,
            repeated.carried.iter().map(|c| (&c.binding, &c.next)),
        )?;
        for (i, _) in alternate.iter().enumerate() {
            writeln!(out, "    (gpu-free alternate-{i})")?;
        }
        for round in 1..=plan.maximum {
            writeln!(out, "    (gpu-free grid-{round})")?;
        }
        writeln!(out, "    (gpu-free publish-grid))")?;
        Ok(true)
    }
}

fn launch(round: u32, slot: usize, [x, y, z]: [u32; 3]) -> String {
    if round == 0 {
        format!(":groups (list (i64 {x}) (i64 {y}) (i64 {z}))")
    } else {
        format!(":indirect (list grid-{round} (i64 {}))", slot * 12)
    }
}

impl DeviceLoop {
    fn groups(&self) -> u32 {
        // Grid-stride copies also cover large state with at most 65535 groups.
        self.carried
            .iter()
            .map(|carry| (carry.bytes / 4).div_ceil(128).min(65535) as u32)
            .max()
            .unwrap_or(1)
    }

    fn write_kernels(&self, out: &mut String, region: usize) -> Result<(), HostError> {
        let groups = self.groups();
        let prefix = format!("device-loop-{region}");
        let mut source = String::new();
        for (i, name) in ["predicate", "grid", "selected"].iter().enumerate() {
            let access = if i == 0 { "read" } else { "read_write" };
            writeln!(
                source,
                "@group(0) @binding({i}) var<storage, {access}> {name}: array<u32>;"
            )?;
        }
        for i in 0..self.carried.len() * 2 {
            // The same module contains the two disjoint kernel ABIs.
            writeln!(
                source,
                "@group(1) @binding({i}) var<storage, read_write> carry_{i}: array<u32>;"
            )?;
        }
        for (name, publish) in [("primary", 0), ("alternate", groups)] {
            writeln!(
                source,
                "@compute @workgroup_size(128) fn {name}(@builtin(global_invocation_id) id: vec3<u32>) {{"
            )?;
            writeln!(source, "  let enabled = predicate[0] != 0u; if (id.x == 0u) {{")?;
            writeln!(
                source,
                "    selected[0] = {publish}u; selected[1] = 1u; selected[2] = 1u;"
            )?;
            writeln!(source, "    if (enabled) {{")?;
            for (i, [x, y, z]) in
                self.dispatches.iter().map(|(_, size)| *size).chain([[1, 1, 1]]).enumerate()
            {
                writeln!(
                    source,
                    "      grid[{}] = {x}u; grid[{}] = {y}u; grid[{}] = {z}u;",
                    i * 3,
                    i * 3 + 1,
                    i * 3 + 2
                )?;
            }
            writeln!(source, "    }} }}")?;
            writeln!(source, "  if (enabled) {{")?;
            for (i, carry) in self.carried.iter().enumerate() {
                writeln!(
                    source,
                    "    for (var j = id.x; j < {}u; j += 128u) {{ carry_{i}[j] = 0u; }}",
                    carry.bytes / 4
                )?;
            }
            writeln!(source, "  }} }}")?;
        }
        writeln!(
            source,
            "@compute @workgroup_size(128) fn publish(@builtin(global_invocation_id) id: vec3<u32>) {{"
        )?;
        for (i, carry) in self.carried.iter().enumerate() {
            writeln!(
                source,
                "  for (var j = id.x; j < {}u; j += {}u) {{ carry_{}[j] = carry_{}[j]; }}",
                carry.bytes / 4,
                groups * 128,
                i * 2 + 1,
                i * 2
            )?;
        }
        writeln!(source, "}}")?;
        writeln!(
            out,
            "(define-gpu-module '{prefix} :format :wgsl :source {})",
            string(&source)
        )?;
        for name in ["primary", "alternate", "publish"] {
            let publishing = name == "publish";
            let count = if publishing { self.carried.len() * 2 } else { 3 + self.carried.len() };
            writeln!(
                out,
                "(define-gpu-kernel '{prefix}-{name} :module '{prefix} :entry {}",
                string(name)
            )?;
            writeln!(out, "  :workgroup-size '(128 1 1) :parameters '(")?;
            for i in 0..count {
                let access = if i == 0 && !publishing { ":read" } else { ":read-write" };
                writeln!(out, "    (arg-{i} :buffer {access})")?;
            }
            writeln!(out, "  ) :abi '(")?;
            for i in 0..count {
                let (set, slot) = if publishing {
                    (1, i)
                } else if i < 3 {
                    (0, i)
                } else {
                    (1, i - 3)
                };
                writeln!(out, "    (arg-{i} :storage {set} {slot})")?;
            }
            writeln!(out, "  ))")?;
        }
        Ok(())
    }
}
