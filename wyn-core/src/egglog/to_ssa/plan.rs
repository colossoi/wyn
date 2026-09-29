//! Native plan identities mapped to final dispatch and storage metadata.
use super::read::Facts;
use super::{error, OptimizeError, Optimized, Program};
use crate::binding_layout::extract_sampler_binding;
use crate::binding_layout::extract_storage_binding;
use crate::binding_layout::extract_texture_binding;
use crate::binding_layout::extract_uniform_binding;
use crate::ssa::types::AtomicOp;
use crate::tlc::extract_lambda_params_ref;
use crate::tlc::DefMeta;
use crate::types::strip_existentials;
use crate::types::Type;
use crate::types::TypeName;
use crate::{BindingRef, LookupMap, LookupSet, SymbolId};
use egglog_engine::{sort::S, Value};

#[derive(Clone)]
pub(super) struct Stage {
    pub key: Value,
    pub operation: Value,
    pub owner: SymbolId,
    pub phase: String,
    pub extent: Value,
    pub width: u32,
}
#[derive(Clone)]
pub(super) struct Buffer {
    pub binding: BindingRef,
    pub name: String,
    pub element: Type,
    pub extent: Value,
}
#[derive(Clone)]
pub(super) struct Output {
    pub owner: SymbolId,
    pub source: Value,
    pub ty: Type,
    pub resource: Value,
    pub copy: bool,
    pub writer: Option<Value>,
}
pub(super) struct Plan<'a, 'source> {
    pub outputs: Vec<Output>,
    pub stages: Vec<Stage>,
    pub buffers: LookupMap<Value, Buffer>,
    pub next_binding: u32,
    query: Facts<'a, 'source>,
}
impl<'a, 'source> Plan<'a, 'source> {
    pub fn group(&self, value: Value) -> Option<Value> {
        self.query.lookup("SsaGroup", (value,))
    }
    pub fn source(&self, value: Value) -> Option<Value> {
        self.query.lookup("SsaSource", (value,))
    }
    pub fn expr(&self, value: Value) -> Option<Value> {
        self.query.lookup("SsaExprSource", (value,))
    }
    pub fn value_ref(&self, value: Value) -> Option<Value> {
        self.query.lookup("SsaValueRef", (value,))
    }
    pub fn external(&self, value: Value) -> Option<Value> {
        self.query.lookup("SsaExternal", (value,))
    }
    pub fn live_length(&self, value: Value) -> Option<Value> {
        self.query.lookup("SsaLiveLength", (value,))
    }
    pub fn domain(&self, value: Value) -> Option<Value> {
        self.query.lookup("SsaDomain", (value,))
    }
    pub fn backing(&self, value: Value) -> Option<Value> {
        self.query.lookup("Backing", (value,))
    }
    pub fn owner(&self, value: Value) -> Option<Value> {
        self.query.lookup("Owner", (value,))
    }
    pub fn members(&self, value: Value) -> Vec<Value> {
        self.query.set("SsaMembers", (value,))
    }
    pub fn scalar_group(&self, value: Value) -> Vec<Value> {
        let members = self.query.set("SsaScalarGroup", (value,));
        if members.is_empty() {
            vec![value]
        } else {
            members
        }
    }
    pub fn dependencies(&self, value: Value) -> Vec<Value> {
        self.query.set("SsaDependencies", (value,))
    }
    pub fn resources(&self, value: Value) -> Vec<Value> {
        self.query.set("SsaOperationResources", (value,))
    }
    pub fn capacity_sources(&self, value: Value) -> Vec<Value> {
        self.query.set("SsaCapacitySources", (value,))
    }
    pub fn pure(&self, value: Value) -> bool {
        self.query.flag("SourcePure", value)
    }
    pub fn member(&self, plan: Value, op: Value) -> bool {
        self.query.contains("PlanMember", (plan, op))
    }
    pub fn view_length(&self, value: Value) -> Option<Value> {
        self.query.lookup(
            "SsaViewLength",
            (self.query.program.identities.values.get(&value)?,),
        )
    }
    pub fn bucket_axis(&self, op: Value, axis: i64) -> Option<Value> {
        self.query.lookup("SsaBucketAxis", (op, axis))
    }
    pub fn slot(&self, op: Value, role: &str, index: i64) -> Option<Value> {
        self.query.lookup("SsaSlot", (op, role, index))
    }
    pub fn pinned(&self, id: i64) -> Option<BindingRef> {
        Some(BindingRef::new(
            self.query.integer(self.query.lookup("SsaPinnedSet", (id,))?) as u32,
            self.query.integer(self.query.lookup("SsaPinnedBinding", (id,))?) as u32,
        ))
    }
    pub fn atomic(&self, op: Value) -> Option<AtomicOp> {
        let value = self.query.lookup("SsaAtomic", (op,))?;
        for (name, update) in [
            ("AtomicAdd", AtomicOp::Add),
            ("AtomicAnd", AtomicOp::And),
            ("AtomicOr", AtomicOp::Or),
            ("AtomicXor", AtomicOp::Xor),
            ("AtomicCas", AtomicOp::CompareExchange),
        ] {
            if self.query.enode(name, value).is_some() {
                return Some(update);
            }
        }
        None
    }
    pub fn extent(&self, value: Value) -> Option<(&'static str, Vec<Value>)> {
        for name in ["Fixed", "Length", "Scalar", "ChunkCount", "Product", "Stored"] {
            if let Some(fields) = self.query.enode(name, value) {
                return Some((name, fields));
            }
        }
        None
    }
    pub fn results(&self, plan: Value) -> Vec<(String, i64, Value)> {
        let mut results = Vec::new();
        for role in ["array", "scan", "total"] {
            let count = self
                .query
                .lookup("SsaResultCount", (plan, role))
                .map(|v| self.query.integer(v))
                .unwrap_or(0);
            for i in 0..count {
                if let Some(value) = self.query.lookup("SsaResultAt", (plan, role, i)) {
                    results.push((role.into(), i, value));
                }
            }
        }
        results
    }
    pub fn counts(&self, plan: Value) -> Vec<(i64, Value)> {
        let count = self.query.lookup("SsaCountCount", (plan,)).map(|v| self.query.integer(v)).unwrap_or(0);
        (0..count).filter_map(|i| self.query.lookup("SsaCountAt", (plan, i)).map(|v| (i, v))).collect()
    }
    pub fn read(program: &'a Program<'source, Optimized>) -> Result<Self, OptimizeError> {
        let graph = &program.graph;
        let mut plan = Self {
            outputs: Vec::new(),
            stages: Vec::new(),
            buffers: LookupMap::default(),
            next_binding: 0,
            query: Facts { program },
        };
        graph.constructor_enodes("AbiOutputBinding", |r| {
            if graph.value_to_base::<i64>(r.children[1]) == 0 {
                plan.next_binding =
                    plan.next_binding.max(graph.value_to_base::<i64>(r.children[2]) as u32 + 1);
            }
        })?;
        graph.constructor_enodes("InputBinding", |r| {
            if graph.value_to_base::<i64>(r.children[0]) == 0 {
                plan.next_binding =
                    plan.next_binding.max(graph.value_to_base::<i64>(r.children[1]) as u32 + 1);
            }
        })?;
        for definition in &program.source.defs {
            if let DefMeta::EntryPoint(entry) = &definition.meta {
                for parameter in &entry.declaration.params {
                    for binding in [
                        extract_uniform_binding(parameter),
                        extract_storage_binding(parameter),
                        extract_texture_binding(parameter),
                        extract_sampler_binding(parameter),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        if binding.set == 0 {
                            plan.next_binding = plan.next_binding.max(binding.binding + 1);
                        }
                    }
                }
            }
        }
        let mut buffers = Vec::new();
        graph.constructor_enodes("PlannedBuffer", |r| {
            buffers.push((
                r.children[0],
                graph.value_to_base::<i64>(r.children[1]),
                r.children[2],
            ));
        })?;
        buffers.sort_by_key(|r| r.0);
        for (key, ty, extent) in buffers {
            plan.buffers.insert(
                key,
                Buffer {
                    binding: BindingRef {
                        set: 0,
                        binding: plan.next_binding,
                    },
                    name: String::new(),
                    element: program.identities.types.resolve(ty).clone(),
                    extent,
                },
            );
            plan.next_binding += 1;
        }
        let mut stages = Vec::new();
        let mut stage_rows = Vec::new();
        graph.constructor_enodes("PlannedStage", |r| stage_rows.push(r.children.to_vec()))?;
        for row in stage_rows {
            let r = row.as_slice();
            let Some(operation) =
                plan.query.constructor("OperationId", (graph.value_to_base::<i64>(r[1]),))
            else {
                return Err(error("missing stage operation"));
            };
            stages.push(Stage {
                key: r[0],
                operation,
                phase: graph.value_to_base::<S>(r[2]).to_string(),
                owner: *program.identities.symbols.resolve(graph.value_to_base::<i64>(r[3])),
                extent: r[4],
                width: graph.value_to_base::<i64>(r[5]) as u32,
            });
        }
        stages.sort_by_key(|s| s.key);
        let mut emitted = LookupSet::default();
        while !stages.is_empty() {
            let Some(index) =
                stages.iter().position(|s| plan.dependencies(s.key).iter().all(|d| emitted.contains(d)))
            else {
                return Err(error("dispatch plan contains a cycle"));
            };
            let stage = stages.remove(index);
            emitted.insert(stage.key);
            plan.stages.push(stage);
        }
        let mut outputs = Vec::new();
        let mut output_rows = Vec::new();
        graph.constructor_enodes("SourceOutput", |r| output_rows.push(r.children.to_vec()))?;
        for row in output_rows {
            let r = row.as_slice();
            if let Some(resource) =
                plan.query.lookup("SsaOutputBacking", (graph.value_to_base::<i64>(r[0]),))
            {
                let Some(ty) = plan.query.ty(r[3]).cloned() else {
                    return Err(error("output has no source type"));
                };
                outputs.push((
                    graph.value_to_base::<i64>(r[0]),
                    Output {
                        owner: *program.identities.symbols.resolve(graph.value_to_base::<i64>(r[1])),
                        source: r[2],
                        ty,
                        resource,
                        copy: plan.query.contains("CopyOutput", (graph.value_to_base::<i64>(r[0]),)),
                        writer: plan.query.lookup("SsaOutputWriter", (graph.value_to_base::<i64>(r[0]),)),
                    },
                ));
            }
        }
        outputs.sort_by_key(|r| r.0);
        let mut used = std::collections::BTreeSet::new();
        for definition in &program.source.defs {
            if let DefMeta::EntryPoint(entry) = &definition.meta {
                used.extend(entry.declaration.params.iter().map(|p| p.name.clone()));
            }
        }
        let mut indices = LookupMap::<SymbolId, usize>::default();
        for (id, output) in &outputs {
            let index = indices.entry(output.owner).or_default();
            let backing = plan.backing(output.resource).unwrap_or(output.resource);
            let pinned = plan.pinned(*id);
            if let Some(buffer) = plan.buffers.get_mut(&backing) {
                if let Some(binding) = pinned {
                    buffer.binding = binding;
                }
                let definition = program.source.defs.iter().find(|d| d.name == output.owner);
                if let Some(definition) = definition {
                    let (body, _) = extract_lambda_params_ref(&definition.body);
                    let owner =
                        program.source.symbols.get(output.owner).map(String::as_str).unwrap_or("entry");
                    let field = match strip_existentials(&body.ty) {
                        Type::Constructed(TypeName::Record(names), _) => names.0[*index].clone(),
                        Type::Constructed(TypeName::Tuple(_), _) => {
                            format!("result_{index}")
                        }
                        _ => "output".into(),
                    };
                    buffer.name = if pinned.is_some() {
                        format!("{owner}_output_{index}")
                    } else {
                        unique(format!("{owner}_{field}"), &mut used)
                    };
                }
            }
            *index += 1;
        }
        for stage in &plan.stages {
            let owner = program.source.symbols.get(stage.owner).map(String::as_str).unwrap_or("entry");
            let mut operations = plan.scalar_group(stage.operation);
            operations.push(stage.operation);
            for resource in operations.into_iter().flat_map(|op| plan.resources(op)).collect::<Vec<_>>() {
                let backing = plan.backing(resource).unwrap_or(resource);
                if let Some(buffer) = plan.buffers.get_mut(&backing) {
                    if buffer.name.is_empty() {
                        buffer.name = unique(format!("{owner}_scratch"), &mut used);
                    }
                }
            }
        }
        for buffer in plan.buffers.values_mut() {
            if buffer.name.is_empty() {
                buffer.name = unique(format!("scratch_{}", buffer.binding.binding), &mut used);
            }
        }
        plan.outputs = outputs.into_iter().map(|(_, o)| o).collect();
        Ok(plan)
    }
}

pub(super) fn unique(base: String, used: &mut std::collections::BTreeSet<String>) -> String {
    let mut name = base.clone();
    let mut index = 2;
    while !used.insert(name.clone()) {
        name = format!("{base}_{index}");
        index += 1;
    }
    name
}
