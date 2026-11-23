//! Import the live body graph and publish changes to its direct operands.
use super::{edit, ir::Code};
use crate::ast::Span;
use crate::egglog::graph::Graph;
use crate::egglog::{analysis, output_error, query::Query, timing, OptimizeError};
use crate::memory_ir::{
    BodyRole, Definition, ForwardedValues, Function, MemoryIr, Operation, OperationId, Parameter, Region,
    RegionEdge, RegionId, UseSite, ValueRef,
};
use crate::{LookupMap, StableMap};
use egglog_engine::{RawValues, Value, Write};
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
    pub fn get_or_read<'a>(
        slot: &'a mut Option<Self>,
        query: Query<'_>,
    ) -> Result<&'a mut Self, OptimizeError> {
        match slot {
            Some(body) => Ok(body),
            slot @ None => Ok(slot.insert(Self::read(query)?)),
        }
    }

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
            let function = if let Some(key) = query.lookup("FunctionRoot", row[0])? {
                Function::Defined(region_ids[&key])
            } else {
                Function::External
            };
            let id = ir.functions.insert(function);
            if query.lookup("EntryOrder", row[0])?.is_some() {
                ir.entries.push(id);
            }
            functions.insert(row[0], id);
            Ok(())
        })?;
        for (&region, &id) in &region_ids {
            for (index, name) in query.sequence("RegionParameter", region)?.into_iter().enumerate() {
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
                let outputs = query.sequence("Output", op)?;
                let definition = match code {
                    Code::Call => Definition::Call(functions[&query.required("CallTarget", op)?]),
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
                query.sequence("Input", op)?.into_iter().map(resolve).collect::<Result<_, _>>()?;
        }
        for (&region, &id) in &region_ids {
            ir.regions[id].yields = query
                .sequence("RegionYield", region)?
                .into_iter()
                .map(resolve)
                .collect::<Result<_, _>>()?;
        }
        let mut arguments = LookupMap::<(Value, Value), Vec<(i64, ValueRef)>>::new();
        query.for_function("EdgeArgument", |keys, name| {
            if operation_ids.contains_key(&keys[0]) {
                arguments
                    .entry((keys[0], keys[1]))
                    .or_default()
                    .push((query.0.value_to_base(keys[2]), resolve(name)?));
            }
            Ok(())
        })?;
        for (table, role) in [
            ("ControlRegion", BodyRole::Control as fn(usize) -> BodyRole),
            ("ResultReducer", BodyRole::Reducer),
            ("AccumulatorCombiner", BodyRole::Combiner),
        ] {
            query.for_function(table, |keys, child| {
                let Some(&op) = operation_ids.get(&keys[0]) else {
                    return Ok(());
                };
                let index = usize::try_from(query.0.value_to_base::<i64>(keys[1]))
                    .map_err(|_| output_error("negative region index"))?;
                let role = role(index);
                let mut incoming = Vec::new();
                if let BodyRole::Control(_) = role {
                    let mut args = arguments.remove(&(keys[0], keys[1])).unwrap_or_default();
                    args.sort_unstable_by_key(|&(index, _)| index);
                    for (expected, (slot, name)) in args.into_iter().enumerate() {
                        if slot != expected as i64 {
                            return Err(output_error("missing region argument"));
                        }
                        incoming.push(name);
                    }
                }
                ir.operations[op].bodies.push(RegionEdge {
                    role,
                    region: region_ids[&child],
                    arguments: incoming,
                });
                Ok(())
            })?;
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

    pub fn publish_substitutions(
        &self,
        graph: &mut Graph,
        values: &LookupMap<ValueRef, ValueRef>,
        uses: Vec<(UseSite, ValueRef)>,
    ) -> Result<(), OptimizeError> {
        let names: Vec<_> = values
            .iter()
            .filter_map(|(old, new)| {
                let (old, new) = (self.names[old], self.names[new]);
                (old != new).then_some((old, new))
            })
            .collect();
        self.publish_edges(graph, uses)?;
        if !names.is_empty() {
            edit::replace_attributes(graph, &names)?;
        }
        Ok(())
    }

    pub fn publish_edges(
        &self,
        graph: &mut Graph,
        edits: Vec<(UseSite, ValueRef)>,
    ) -> Result<(), OptimizeError> {
        let mut rows = Vec::with_capacity(edits.len());
        for (site, value) in edits {
            let (table, keys) = match site {
                UseSite::Input(op, i) => {
                    ("Input", vec![self.operations[op], graph.base_to_value(i as i64)])
                }
                UseSite::Yield(region, i) => (
                    "RegionYield",
                    vec![self.regions[region], graph.base_to_value(i as i64)],
                ),
                UseSite::Argument(op, BodyRole::Control(edge), i) => (
                    "EdgeArgument",
                    vec![
                        self.operations[op],
                        graph.base_to_value(edge as i64),
                        graph.base_to_value(i as i64),
                    ],
                ),
                UseSite::Argument(_, _, _) => {
                    return Err(output_error("combiner arguments use collective metadata"))
                }
            };
            rows.push((table, keys, self.names[&value]));
        }
        graph.update(|mut sink| {
            for (table, keys, value) in rows {
                sink.remove(table, RawValues(keys.clone()))?;
                sink.set(table, RawValues(keys), value)?;
            }
            Ok(())
        })?;
        Ok(())
    }
}

pub(in crate::egglog) fn forward_equivalents(
    graph: &mut Graph,
    body: &mut Option<BodyGraph>,
) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    let mut replacements = LookupMap::new();
    query.for_each("ForwardCandidate", |row| {
        replacements.entry(row[0]).or_insert(row[1]);
        Ok(())
    })?;
    if replacements.is_empty() {
        return Ok(false);
    }
    let _timing = timing::span("Rust value forwarding");
    let mut body = match body.take() {
        Some(body) => body,
        None => BodyGraph::read(query)?,
    };
    let resolve = |name| {
        let Some(&value) = body.refs.get(&name) else {
            return Err(output_error("forwarded operand has no live memory definition"));
        };
        Ok(value)
    };
    let equivalents = replacements
        .into_iter()
        .map(|(old, new)| Ok((resolve(old)?, resolve(new)?)))
        .collect::<Result<Vec<_>, OptimizeError>>()?;
    let redirected = body.ir.forward_values(equivalents).map_err(|err| output_error(err.to_string()))?;
    if publish_forwarding(graph, body, redirected)? {
        return Ok(true);
    }
    analysis::ensure(graph)?;
    edit::clear(graph)?;
    Ok(false)
}

fn publish_forwarding(
    graph: &mut Graph,
    body: BodyGraph,
    redirected: ForwardedValues,
) -> Result<bool, OptimizeError> {
    let removed = redirected
        .values
        .keys()
        .filter_map(|old| match *old {
            ValueRef::Result(op, _) if !body.ir.operations.contains_key(op) => Some(body.operations[op]),
            _ => None,
        })
        .collect::<Vec<_>>();
    body.publish_substitutions(graph, &redirected.values, redirected.uses)?;
    if removed.is_empty() {
        return Ok(false);
    }
    edit::remove(graph, &removed)?;
    Ok(true)
}
