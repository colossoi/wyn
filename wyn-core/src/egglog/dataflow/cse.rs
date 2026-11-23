//! Match repeatable expressions against the definitions visible at each occurrence.
use super::{body::BodyGraph, ir::Code};
use crate::egglog::graph::Formula;
use crate::egglog::graph::Graph;
use crate::egglog::{query::Query, OptimizeError};
use crate::memory_ir::{Definition, Function, OperationId, RegionId, ValueRef};
use crate::LookupMap;
use egglog_engine::{Value, Write};

#[derive(Clone, Eq, Hash, PartialEq)]
struct Expression {
    code: Code,
    operands: Vec<ValueRef>,
    index: Option<Value>,
    ty: Value,
}

#[derive(Default)]
pub(in crate::egglog) struct Reuse {
    visible: LookupMap<Expression, ValueRef>,
    pub replacements: LookupMap<ValueRef, ValueRef>,
}

impl Reuse {
    fn visit(
        &mut self,
        query: Query<'_>,
        body: &BodyGraph,
        operation: OperationId,
        replace: bool,
    ) -> Result<Option<Expression>, OptimizeError> {
        let op = &body.ir.operations[operation];
        let (Definition::Other(code), [ty]) = (&op.definition, op.result_types.as_slice()) else {
            return Ok(None);
        };
        let key = body.operations[operation];
        let output = ValueRef::Result(operation, 0);
        if !code.has_expression_identity()
            || !query.contains("RepeatableOperation", key)?
            || query.value_name(body.names[&output])?.is_none()
        {
            return Ok(None);
        }
        let expression = Expression {
            code: code.clone(),
            operands: op
                .inputs
                .iter()
                .map(|value| self.replacements.get(value).copied().unwrap_or(*value))
                .collect(),
            index: query.0.formula_at(key, Formula::Index),
            ty: *ty,
        };
        if let Some(&existing) = self.visible.get(&expression) {
            if replace {
                self.replacements.insert(output, existing);
            }
            Ok(None)
        } else {
            self.visible.insert(expression.clone(), output);
            Ok(Some(expression))
        }
    }

    pub fn scope(
        &mut self,
        query: Query<'_>,
        body: &BodyGraph,
        region: RegionId,
        prefixes: &[(RegionId, usize)],
    ) -> Result<(), OptimizeError> {
        for &(ancestor, end) in prefixes {
            for &op in &body.ir.regions[ancestor].operations[..end] {
                self.visit(query, body, op, false)?;
            }
        }
        self.region(query, body, region)?;
        self.visible.clear();
        Ok(())
    }

    fn region(
        &mut self,
        query: Query<'_>,
        body: &BodyGraph,
        region: RegionId,
    ) -> Result<(), OptimizeError> {
        let mut definitions = Vec::new();
        for &op in &body.ir.regions[region].operations {
            for edge in &body.ir.operations[op].bodies {
                self.region(query, body, edge.region)?;
            }
            definitions.extend(self.visit(query, body, op, true)?);
        }
        for expression in definitions {
            self.visible.remove(&expression);
        }
        Ok(())
    }
}

pub(in crate::egglog) fn publish(graph: &mut Graph, body: &BodyGraph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut reuse = Reuse::default();
    for function in body.ir.functions.values() {
        if let Function::Defined(region) = *function {
            reuse.scope(query, body, region, &[])?;
        }
    }
    graph.update(|mut sink| {
        for (old, new) in reuse.replacements {
            sink.add("ForwardCandidate", (body.names[&old], body.names[&new]))?;
        }
        Ok(())
    })?;
    Ok(())
}
