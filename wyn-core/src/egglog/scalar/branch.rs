//! Splice proven branch arms into their enclosing Rust regions.
use super::fold;
use crate::egglog::dataflow::{edit, ir::Code};
use crate::egglog::graph::{Graph, Origin};
use crate::egglog::{output_error, query::Query, representation, timing, OptimizeError};
use crate::types::Type;
use crate::LookupMap;
use egglog_engine::Value;
use wyn_graph::ReplacementForest;

pub(super) fn apply(graph: &mut Graph) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    let mut selected = LookupMap::<Value, Value>::new();
    query.for_each("BranchCandidate", |row| {
        selected.insert(row[0], row[1]);
        Ok(())
    })?;
    if selected.is_empty() {
        return Ok(false);
    }
    splice(graph, query.live_roots()?, selected)
}

/// Evaluate only the inputs needed to decide a known branch before solving proofs.
pub(in crate::egglog) fn before_analysis(
    graph: &mut Graph,
    roots: &[Value],
) -> Result<bool, OptimizeError> {
    let _timing = timing::span("Rust static branch conditions");
    let query = Query(graph);
    let mut selected = LookupMap::new();
    let mut constants = Constants {
        query,
        values: LookupMap::new(),
    };
    query.for_each("BranchOperation", |row| {
        let branch = row[0];
        if let Some(arm) = constants.arm(branch)? {
            selected.insert(branch, arm);
        }
        Ok(())
    })?;
    if selected.is_empty() {
        return Ok(false);
    }
    splice(graph, roots.to_vec(), selected)
}

fn splice(
    graph: &mut Graph,
    mut pending: Vec<Value>,
    selected: LookupMap<Value, Value>,
) -> Result<bool, OptimizeError> {
    let _timing = timing::span("Rust constant branch splicing");
    let query = Query(graph);
    let mut replacements = ReplacementForest::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();
    while let Some(region) = pending.pop() {
        let mut operations = ordered(query, region)?;
        operations.reverse();
        let mut remaining = Vec::with_capacity(operations.len());
        let mut spliced = false;
        while let Some(operation) = operations.pop() {
            let Some(&chosen) = selected.get(&operation) else {
                pending.extend(graph.attached_regions(operation));
                remaining.push(operation);
                continue;
            };
            let Some((edge, _)) = graph.controls(operation).find(|&(_, arm)| arm == chosen) else {
                return Err(output_error("constant branch proof names a foreign arm"));
            };
            let arguments = graph.arguments(operation, edge);
            let params = graph.parameters(chosen);
            let outputs = graph.outputs(operation);
            let yields = graph.yield_values(chosen);
            if arguments.len() != params.len() || outputs.len() != yields.len() {
                return Err(output_error(
                    "constant branch has an inconsistent region signature",
                ));
            }
            for (old, new) in params.into_iter().zip(arguments).chain(outputs.into_iter().zip(yields)) {
                replacements.replace(old, new).map_err(|error| output_error(error.to_string()))?;
            }
            // A nested branch in the selected arm now occupies this block.
            operations.extend(ordered(query, chosen)?.into_iter().rev());
            removed.push(operation);
            spliced = true;
        }
        if spliced {
            changed.push((region, remaining));
        }
    }
    if removed.is_empty() {
        return Ok(false);
    }
    let replacements: Vec<_> = replacements.into_map().into_iter().collect();
    edit::replace(graph, &replacements)?;
    graph.update(|mut sink| {
        for (region, operations) in changed {
            for (order, operation) in operations.into_iter().enumerate() {
                sink.set_region(operation, region);
                sink.set_order(operation, order as i64);
            }
        }
        Ok(())
    })?;
    edit::remove(graph, &removed)?;
    Ok(true)
}

fn ordered(query: Query<'_>, region: Value) -> Result<Vec<Value>, OptimizeError> {
    if !query.0.dirty_orders.contains(&region) {
        if let Some(operations) = query.0.region_orders.get(&region) {
            return Ok(operations.clone());
        }
    }
    query.operations(region)
}

struct Constants<'a> {
    query: Query<'a>,
    values: LookupMap<Value, Option<i64>>,
}

impl Constants<'_> {
    fn arm(&mut self, branch: Value) -> Result<Option<Value>, OptimizeError> {
        let Some(condition) = self.query.0.input_at(branch, 0) else {
            return Ok(None);
        };
        let slot = match self.value(condition)? {
            Some(1) => 0,
            Some(0) => 1,
            _ => return Ok(None),
        };
        Ok(self.query.0.control_at(branch, slot))
    }

    fn value(&mut self, name: Value) -> Result<Option<i64>, OptimizeError> {
        if let Some(&value) = self.values.get(&name) {
            return Ok(value);
        }
        self.values.insert(name, None);
        let result = self.read_value(name)?;
        self.values.insert(name, result);
        Ok(result)
    }

    fn read_value(&mut self, name: Value) -> Result<Option<i64>, OptimizeError> {
        let graph = self.query.0;
        match graph.origin(name) {
            Some(Origin::Result(operation, slot)) => match graph.code(operation)? {
                Code::Constant { bits, .. } => Ok(Some(bits)),
                Code::Scalar(payload) => {
                    let inputs = graph.inputs(operation);
                    let Some(&first) = inputs.first() else {
                        return Ok(None);
                    };
                    let mut bits = Vec::with_capacity(inputs.len());
                    for operand in &inputs {
                        let Some(bits_value) = self.value(*operand)? else {
                            return Ok(None);
                        };
                        bits.push(bits_value);
                    }
                    let input = self.ty(first)?;
                    for operand in &inputs[1..] {
                        if self.ty(*operand)? != input {
                            return Ok(None);
                        }
                    }
                    Ok(fold::evaluate_typed(payload, &input, &self.ty(name)?, &bits))
                }
                Code::Branch => {
                    if let Some(arm) = self.arm(operation)? {
                        if let Some(value) = graph.yield_at(arm, slot as i64) {
                            return self.value(value);
                        }
                    }
                    Ok(None)
                }
                _ => Ok(None),
            },
            Some(Origin::Parameter(region, slot)) => {
                if let Some(owner) = graph.region_owner(region) {
                    if graph.code(owner)? == Code::Branch {
                        if let Some((edge, _)) = graph.controls(owner).find(|&(_, body)| body == region) {
                            if let Some(arg) = graph.arguments(owner, edge).get(slot) {
                                return self.value(*arg);
                            }
                        }
                    }
                }
                Ok(None)
            }
            None => Ok(None),
        }
    }

    fn ty(&self, name: Value) -> Result<Type, OptimizeError> {
        let info = crate::egglog::analysis::typed(self.query, name)?;
        representation::value_type(self.query, info)
    }
}
