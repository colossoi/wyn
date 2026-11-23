//! Import the live body graph and publish changes to its direct operands.
use super::ir::Code;
use crate::ast::Span;
use crate::egglog::graph::{ArgumentSite, Graph};
use crate::egglog::{analysis, output_error, query::Query, OptimizeError};
use crate::memory_ir::{
    BodyRole, Definition, Function, MemoryIr, Operation, OperationId, Parameter, Region, RegionEdge,
    RegionId, UseSite, ValueRef,
};
use crate::{LookupMap, StableMap};
use egglog_engine::Value;
use slotmap::SecondaryMap;

pub(in crate::egglog) struct BodyGraph {
    pub ir: MemoryIr<Code, Value, Value>,
    pub operations: SecondaryMap<OperationId, Value>,
    pub regions: SecondaryMap<RegionId, Value>,
    pub names: LookupMap<ValueRef, Value>,
    pub operation_ids: LookupMap<Value, OperationId>,
    pub refs: LookupMap<Value, ValueRef>,
}

impl BodyGraph {
    pub fn read(query: Query<'_>) -> Result<Self, OptimizeError> {
        let mut ir = MemoryIr::default();
        let mut regions = SecondaryMap::new();
        let mut region_ids = StableMap::new();
        let mut operations = SecondaryMap::new();
        let mut operation_ids = LookupMap::new();
        let mut names = LookupMap::new();
        let mut refs = LookupMap::new();
        let mut add_region = |region| {
            let id = ir.regions.insert(Region::default());
            regions.insert(id, region);
            region_ids.insert(region, id);
        };
        query.for_each("LiveRegion", |row| {
            add_region(row[0]);
            Ok(())
        })?;
        let mut functions = LookupMap::new();
        query.for_each("LiveFunction", |row| {
            let function = if let Some(key) = query.0.root_of(row[0]) {
                Function::Defined(region_ids[&key])
            } else {
                Function::External
            };
            let id = ir.functions.insert(function);
            if query.0.entry_order(row[0]).is_some() {
                ir.entries.push(id);
            }
            functions.insert(row[0], id);
            Ok(())
        })?;
        for (&region, &id) in &region_ids {
            for (index, name) in query.0.parameters(region).into_iter().enumerate() {
                let value = ValueRef::Parameter(id, index);
                refs.insert(name, value);
                names.insert(value, name);
                ir.regions[id].parameters.push(Parameter {
                    ty: analysis::typed(query, name)?,
                    diet: query.required("ParameterDiet", (region, index as i64))?,
                });
            }
            for &op in query.0.region_order(region)? {
                let code = query.0.code(op)?;
                let outputs = query.0.outputs(op);
                let definition = match code {
                    Code::Call => Definition::Call(functions[&query.0.call_target(op)?]),
                    code => Definition::Other(code),
                };
                let span = query.0.span(op).unwrap_or_else(Span::generated);
                let key = ir.append(
                    id,
                    Operation {
                        definition,
                        inputs: Vec::new(),
                        bodies: Vec::new(),
                        span,
                        result_types: outputs
                            .iter()
                            .map(|&name| analysis::typed(query, name))
                            .collect::<Result<_, _>>()?,
                    },
                );
                operation_ids.insert(op, key);
                operations.insert(key, op);
                for (index, name) in outputs.into_iter().enumerate() {
                    let value = ValueRef::Result(key, index);
                    refs.insert(name, value);
                    names.insert(value, name);
                }
            }
        }
        let resolve = |name: Value| -> Result<ValueRef, OptimizeError> {
            let Some(&value) = refs.get(&name) else {
                return Err(output_error(format!(
                    "memory operand {name:?} has no live definition"
                )));
            };
            Ok(value)
        };
        for (id, &op) in operations.iter() {
            ir.operations[id].inputs =
                query.0.inputs(op).into_iter().map(resolve).collect::<Result<_, _>>()?;
        }
        for (&region, &id) in &region_ids {
            ir.regions[id].yields =
                query.0.yield_values(region).into_iter().map(resolve).collect::<Result<_, _>>()?;
        }
        for (op, &key) in &operations {
            for (slot, child) in query.0.controls(key) {
                let incoming =
                    query.0.arguments(key, slot).into_iter().map(resolve).collect::<Result<_, _>>()?;
                ir.operations[op].bodies.push(RegionEdge {
                    role: BodyRole::Control(slot as usize),
                    region: region_ids[&child],
                    arguments: incoming,
                });
            }
        }
        for (op, &key) in &operations {
            for (bodies, role) in [
                (query.0.reducers(key), BodyRole::Reducer as fn(usize) -> BodyRole),
                (query.0.combiners(key), BodyRole::Combiner),
            ] {
                for (slot, child) in bodies {
                    let index =
                        usize::try_from(slot).map_err(|_| output_error("negative callback index"))?;
                    ir.operations[op].bodies.push(RegionEdge {
                        role: role(index),
                        region: region_ids[&child],
                        arguments: Vec::new(),
                    });
                }
            }
        }
        Ok(Self {
            ir,
            names,
            regions,
            operations,
            operation_ids,
            refs,
        })
    }

    pub fn publish_edges(
        &self,
        graph: &mut Graph,
        edits: Vec<(UseSite, ValueRef)>,
    ) -> Result<(), OptimizeError> {
        let mut yields = Vec::new();
        let mut arguments = Vec::new();
        let mut inputs = Vec::new();
        for (site, value) in edits {
            match site {
                UseSite::Input(op, i) => {
                    inputs.push((self.operations[op], i as i64, self.names[&value]));
                }
                UseSite::Yield(region, i) => {
                    yields.push((self.regions[region], i as i64, self.names[&value]));
                }
                UseSite::Argument(op, BodyRole::Control(edge), i) => {
                    arguments.push((
                        ArgumentSite(self.operations[op], edge as i64, i),
                        self.names[&value],
                    ));
                }
                UseSite::Argument(_, _, _) => {
                    return Err(output_error("combiner arguments use collective metadata"))
                }
            }
        }
        graph.update(|mut sink| {
            for (op, slot, value) in inputs {
                sink.set_input(op, slot, value)?;
            }
            for (site, value) in arguments {
                sink.set_argument(site, value)?;
            }
            for (region, slot, value) in yields {
                sink.set_yield(region, slot, value)?;
            }
            Ok(())
        })?;
        Ok(())
    }
}
