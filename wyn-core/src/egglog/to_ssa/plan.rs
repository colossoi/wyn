//! Native plan identities mapped to final dispatch and storage metadata.
use super::read::Facts;
use super::{error, OptimizeError, Optimized, Program};
use crate::egglog::query::Query;
use crate::interface::{EntryOutput, EntryParamBindingKind};
use crate::ssa::types::AtomicOp;
use crate::tlc::extract_lambda_params_ref;
use crate::tlc::DefMeta;
use crate::types::{strip_existentials, Type, TypeName};
use crate::{BindingRef, LookupMap, SymbolId};
use egglog_engine::{sort::S, Value};
use wyn_base::IdSource;
use wyn_graph::topo_sort_by_dependencies;

pub(in crate::egglog) struct Plan<'a, 'source> {
    query: Facts<'a, 'source>,
    bindings: LookupMap<Value, BindingRef>,
    pub captures: LookupMap<egglog_engine::TermId, BindingRef>,
    pub entry_outputs: LookupMap<SymbolId, Vec<EntryOutput>>,
}
impl<'a, 'source> Plan<'a, 'source> {
    pub fn new(program: &'a Program<'source, Optimized>) -> Result<Self, OptimizeError> {
        let query = Facts { program };
        let mut reserved = std::collections::BTreeSet::new();
        let mut entry_outputs = LookupMap::default();
        for definition in &program.source.defs {
            let DefMeta::EntryPoint(entry) = &definition.meta else {
                continue;
            };
            let mut output_bindings = IdSource::new();
            let mut reserve_input = |binding: BindingRef| {
                reserved.insert(binding);
                if binding.set == 0 {
                    while output_bindings.peek_id() <= binding.binding {
                        output_bindings.next_id();
                    }
                }
            };
            for parameter in &entry.declaration.params {
                for (set, binding) in parameter.attributes.iter().filter_map(|a| a.binding_slot()) {
                    reserve_input(BindingRef::new(set, binding));
                }
            }
            for parameter in entry.data.param_bindings.iter().flatten() {
                match &parameter.kind {
                    EntryParamBindingKind::Single { binding, .. } => reserve_input(*binding),
                    EntryParamBindingKind::TupleOfViews(fields) => {
                        for field in fields {
                            reserve_input(field.binding);
                        }
                    }
                }
            }
            let outputs = crate::egglog::abi::outputs(program, definition.name, &mut output_bindings)?;
            for output in &outputs {
                if let Some(binding) = output.storage_binding() {
                    reserved.insert(binding);
                }
            }
            entry_outputs.insert(definition.name, outputs);
        }
        let mut allocations = Vec::new();
        program.graph.constructor_enodes("PlannedBuffer", |row| allocations.push(row.children[0]))?;
        allocations.sort();
        allocations.dedup();
        let mut bindings = LookupMap::default();
        for &value in &allocations {
            if let Some(binding) = query.lookup("PhysicalBinding", (value,)) {
                let binding = query.binding(binding)?;
                reserved.insert(binding);
                bindings.insert(value, binding);
            }
        }
        // TODO: Consider assigning generated buffer and capture bindings in
        // Egglog; this ABI choice might fit better alongside PhysicalBinding.
        let mut next = 0;
        for value in allocations {
            if bindings.contains_key(&value) {
                continue;
            }
            while reserved.contains(&BindingRef::new(0, next)) {
                next += 1;
            }
            let binding = BindingRef::new(0, next);
            reserved.insert(binding);
            bindings.insert(value, binding);
        }
        let mut captures = LookupMap::default();
        for &term in program.stage.captures.keys() {
            while reserved.contains(&BindingRef::new(0, next)) {
                next += 1;
            }
            let binding = BindingRef::new(0, next);
            reserved.insert(binding);
            captures.insert(term, binding);
        }
        Ok(Self {
            query,
            bindings,
            captures,
            entry_outputs,
        })
    }

    pub fn stages(&self, owner: SymbolId) -> Result<Vec<Value>, OptimizeError> {
        let Some(owner) = self.query.program.identities.symbols.get(&owner) else {
            return Err(error("entry identity missing"));
        };
        let mut stages = Vec::new();
        self.query.program.graph.constructor_enodes("PhaseOwner", |row| {
            if self.query.integer(row.children[1]) == owner {
                stages.push(row.children[0]);
            }
        })?;
        stages.sort();
        topo_sort_by_dependencies(stages, |key, out| out.extend(self.dependencies(key)))
            .map_err(|_| error("dispatch plan contains a cycle"))
    }

    pub fn phase_operation(&self, stage: Value) -> Result<Value, OptimizeError> {
        let fields = self.query.enode("Stage", stage).ok_or_else(|| error("stage identity missing"))?;
        Ok(fields[0])
    }

    pub fn phase_name(&self, stage: Value) -> Result<String, OptimizeError> {
        let fields = self.query.enode("Stage", stage).ok_or_else(|| error("stage identity missing"))?;
        Ok(self.query.program.graph.value_to_base::<S>(fields[1]).to_string())
    }

    pub fn phase_extent(&self, stage: Value) -> Result<Value, OptimizeError> {
        let row = Query(&self.query.program.graph)
            .row("PhaseDomain", |r| r[0] == stage)?
            .ok_or_else(|| error("stage domain missing"))?;
        Ok(row[1])
    }

    pub fn phase_width(&self, stage: Value) -> Result<u32, OptimizeError> {
        let row = Query(&self.query.program.graph)
            .row("PhaseDomain", |r| r[0] == stage)?
            .ok_or_else(|| error("stage domain missing"))?;
        self.query.positive(row[2], "stage width")
    }

    pub fn grid(&self, key: Value) -> Result<Option<(u32, u32, u32)>, OptimizeError> {
        let query = Query(&self.query.program.graph);
        let grid = query.required("PhaseGrid", (key,))?;
        if query.enode("AutomaticGrid", grid)?.is_some() {
            return Ok(None);
        }
        let grid = self.query.grid(grid)?;
        if [grid.0, grid.1, grid.2].into_iter().any(|axis| axis > 65_535) {
            return Err(error("dispatch grid axis must be in 1..=65535"));
        }
        Ok(Some(grid))
    }

    pub fn buffer(&self, key: Value) -> Result<Option<(BindingRef, &'a Type, Value)>, OptimizeError> {
        let query = Query(&self.query.program.graph);
        let Some(allocation) = query.row("PlannedBuffer", |r| r[0] == key)? else {
            return Ok(None);
        };
        let Some(&binding) = self.bindings.get(&key) else {
            return Err(error("allocation has no physical ABI"));
        };
        Ok(Some((
            binding,
            self.query.program.identities.types.resolve(self.query.integer(allocation[1])),
            allocation[2],
        )))
    }

    pub fn buffer_name(&self, key: Value) -> Result<String, OptimizeError> {
        let query = Query(&self.query.program.graph);
        let owner = query.required("AllocationOwner", (key,))?;
        let symbol = self.query.program.identities.symbols.resolve(self.query.integer(owner));
        let Some(owner) = self.query.program.source.symbols.get(*symbol) else {
            return Err(error("allocation owner name missing"));
        };
        let Some((binding, _, _)) = self.buffer(key)? else {
            return Err(error("allocation missing"));
        };
        if query.lookup("PhysicalBinding", (key,))?.is_some() {
            for (index, output) in self.outputs(*symbol)?.into_iter().enumerate() {
                if query.contains(
                    "AbiOutputBinding",
                    (output, i64::from(binding.set), i64::from(binding.binding)),
                )? {
                    return Ok(format!("{owner}_output_{index}"));
                }
            }
            return Err(error("pinned allocation has no output declaration"));
        }
        for (index, output) in self.outputs(*symbol)?.into_iter().enumerate() {
            let (_, _, resource) = self.output(output)?;
            if self.backing(resource) != Some(key) {
                continue;
            }
            let Some(definition) = self.query.program.source.defs.iter().find(|d| d.name == *symbol) else {
                return Err(error("output definition missing"));
            };
            let (body, _) = extract_lambda_params_ref(&definition.body);
            let field = match strip_existentials(&body.ty) {
                Type::Constructed(TypeName::Record(names), _) => names.0[index].clone(),
                Type::Constructed(TypeName::Tuple(_), _) => format!("result_{index}"),
                _ => "output".into(),
            };
            return Ok(format!("{owner}_{field}"));
        }
        Ok(format!("{owner}_scratch_{}_{}", binding.set, binding.binding))
    }

    pub fn outputs(&self, owner: SymbolId) -> Result<Vec<i64>, OptimizeError> {
        let Some(owner) = self.query.program.identities.symbols.get(&owner) else {
            return Err(error("output owner missing"));
        };
        let mut outputs = Vec::new();
        self.query.program.graph.constructor_enodes("SourceOutput", |row| {
            let id = self.query.integer(row.children[0]);
            if self.query.integer(row.children[1]) == owner
                && self.query.lookup("SsaOutputBacking", (id,)).is_some()
            {
                outputs.push(id);
            }
        })?;
        outputs.sort();
        Ok(outputs)
    }

    pub fn output(&self, id: i64) -> Result<(Value, &'a Type, Value), OptimizeError> {
        let query = Query(&self.query.program.graph);
        let Some(row) = query.row("SourceOutput", |r| self.query.integer(r[0]) == id)? else {
            return Err(error("output declaration missing"));
        };
        let Some(ty) = self.query.ty(row[3]) else {
            return Err(error("output type missing"));
        };
        Ok((row[2], ty, query.required("SsaOutputBacking", (id,))?))
    }
    pub fn group(&self, value: Value) -> Option<Value> {
        self.query.lookup("SsaGroup", (value,))
    }
    pub fn source(&self, value: Value) -> Option<Value> {
        self.query.lookup("SourceOperationValue", (value,))
    }
    pub fn expr(&self, value: Value) -> Option<Value> {
        self.query.lookup("SsaExprSource", (value,))
    }
    pub fn value_ref(&self, value: Value) -> Option<Value> {
        self.query.lookup("SsaValueRef", (value,))
    }
    pub fn value_read(&self, source: Value) -> Option<Value> {
        self.query.lookup("ValueRead", (self.query.lookup("SourceExprKey", (source,))?,))
    }
    pub fn access(&self, source: Value) -> Option<Value> {
        self.query.lookup(
            "SelectedAccess",
            (self.query.lookup("SourceExprKey", (source,))?,),
        )
    }
    pub fn view_extent(&self, source: Value) -> Option<Value> {
        self.query.lookup("LogicalExtent", (source,))
    }
    pub fn external(&self, value: Value) -> Option<Value> {
        self.expr(self.query.enode("Source", value)?[0])
    }
    pub fn live_length(&self, value: Value) -> Option<Value> {
        self.query.lookup("LiveLength", (value,))
    }
    pub fn capacity(&self, value: Value) -> Option<Value> {
        self.query.lookup("CapacityExtent", (value,))
    }
    pub fn reuse_source(&self, value: Value) -> Option<Value> {
        self.expr(self.query.lookup("SameBacking", (value,))?)
    }
    pub fn domain(&self, value: Value) -> Option<Value> {
        self.query.lookup("SsaDomain", (value,))
    }
    pub fn backing(&self, value: Value) -> Option<Value> {
        self.query.lookup("Backing", (value,))
    }
    pub fn filter(&self, plan: Value) -> Result<Value, OptimizeError> {
        let Some(operation) = self.query.lookup("SsaFilter", (plan,)) else {
            return Err(error("filter plan has no filter operation"));
        };
        Ok(operation)
    }
    pub fn scalar_group(&self, value: Value) -> Vec<Value> {
        self.query.set("SsaScalarGroup", (value,))
    }
    pub fn dependencies(&self, value: Value) -> Vec<Value> {
        self.query.set("SsaDependencies", (value,))
    }
    pub fn member(&self, plan: Value, op: Value) -> bool {
        self.query.contains("PlanMember", (plan, op))
    }
    pub fn bucket_axis(&self, op: Value, axis: i64) -> Option<Value> {
        self.query.lookup("SourceBucketAxis", (op, axis))
    }
    pub fn slot(&self, op: Value, role: &str, index: i64) -> Option<Value> {
        self.query.lookup("SsaSlot", (op, role, index))
    }
    pub fn atomic(&self, op: Value) -> Option<AtomicOp> {
        let recipe = self.query.lookup("Plan", (op,))?;
        let value = self.query.enode("Atomic", recipe)?[0];
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
        for name in [
            "Fixed",
            "Length",
            "Scalar",
            "ChunkCount",
            "Product",
            "Difference",
            "Stored",
        ] {
            if let Some(fields) = self.query.enode(name, value) {
                return Some((name, fields));
            }
        }
        None
    }
    pub fn results(&self, plan: Value) -> Result<Vec<(String, i64, Value)>, OptimizeError> {
        let graph = &self.query.program.graph;
        let mut results = Vec::new();
        graph.constructor_enodes("PlanResult", |row| {
            if row.children[0] == plan {
                results.push((
                    graph.value_to_base::<S>(row.children[1]).to_string(),
                    self.query.integer(row.children[2]),
                    row.children[3],
                ));
            }
        })?;
        results.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        Ok(results)
    }
    pub fn counts(&self, plan: Value) -> Result<Vec<(i64, Value)>, OptimizeError> {
        let mut indices = Vec::new();
        self.query.program.graph.constructor_enodes("PlanCountSlot", |row| {
            if row.children[0] == plan {
                indices.push(self.query.integer(row.children[2]));
            }
        })?;
        indices.sort();
        indices
            .into_iter()
            .map(|index| {
                let Some(value) = self.query.lookup("SsaCountAt", (plan, index)) else {
                    return Err(error("count slot has no selected operation"));
                };
                Ok((index, value))
            })
            .collect()
    }
}

pub(in crate::egglog) fn unique(base: String, used: &mut std::collections::BTreeSet<String>) -> String {
    let mut name = base.clone();
    let mut index = 2;
    while !used.insert(name.clone()) {
        name = format!("{base}_{index}");
        index += 1;
    }
    name
}
