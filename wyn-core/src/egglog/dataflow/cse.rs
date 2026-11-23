//! Match repeatable expressions against the definitions visible at each occurrence.
use super::{body::BodyGraph, ir::Code};
use crate::egglog::graph::Formula;
use crate::egglog::graph::Graph;
use crate::egglog::{query::Query, OptimizeError};
use crate::memory_ir::{Definition, OperationId, RegionId, ValueRef};
use crate::LookupMap;
use egglog_engine::{Value, Write};
use std::hash::Hash;

#[derive(Clone, Eq, Hash, PartialEq)]
struct Expression<K> {
    code: Code,
    operands: Vec<K>,
    index: Option<Value>,
    ty: Value,
}

pub(in crate::egglog) struct Reuse<K = ValueRef> {
    visible: LookupMap<Expression<K>, K>,
    pub replacements: LookupMap<K, K>,
}

impl<K> Default for Reuse<K> {
    fn default() -> Self {
        Self {
            visible: LookupMap::new(),
            replacements: LookupMap::new(),
        }
    }
}

impl<K: Copy + Eq + Hash> Reuse<K> {
    fn observe(
        &mut self,
        mut expression: Expression<K>,
        output: K,
        replace: bool,
    ) -> Option<Expression<K>> {
        for operand in &mut expression.operands {
            if let Some(&existing) = self.replacements.get(operand) {
                *operand = existing;
            }
        }
        if let Some(&existing) = self.visible.get(&expression) {
            if replace {
                self.replacements.insert(output, existing);
            }
            None
        } else {
            self.visible.insert(expression.clone(), output);
            Some(expression)
        }
    }
}

impl Reuse<ValueRef> {
    fn visit(
        &mut self,
        query: Query<'_>,
        body: &BodyGraph,
        operation: OperationId,
        replace: bool,
    ) -> Result<Option<Expression<ValueRef>>, OptimizeError> {
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
        Ok(self.observe(
            Expression {
                code: code.clone(),
                operands: op.inputs.clone(),
                index: query.0.formula_at(key, Formula::Index),
                ty: *ty,
            },
            output,
            replace,
        ))
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

impl Reuse<Value> {
    fn execution_region(&mut self, query: Query<'_>, region: Value) -> Result<(), OptimizeError> {
        let mut definitions = Vec::new();
        for &operation in query.0.region_order(region)? {
            for child in query.0.attached_regions(operation) {
                self.execution_region(query, child)?;
            }
            let code = query.0.code(operation)?;
            if !code.has_expression_identity()
                || !query.contains("RepeatableOperation", operation)?
                || query.0.output_count(operation) != 1
            {
                continue;
            }
            let output = query.0.output(operation, 0)?;
            if query.value_name(output)?.is_none() {
                continue;
            }
            let expression = Expression {
                code,
                operands: query.0.inputs(operation),
                index: query.0.formula_at(operation, Formula::Index),
                ty: crate::egglog::analysis::typed(query, output)?,
            };
            definitions.extend(self.observe(expression, output, true));
        }
        for expression in definitions {
            self.visible.remove(&expression);
        }
        Ok(())
    }
}

pub(in crate::egglog) fn publish(graph: &mut Graph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut reuse = Reuse::<Value>::default();
    for region in query.live_roots()? {
        reuse.execution_region(query, region)?;
    }
    graph.update(|mut sink| {
        for (old, new) in reuse.replacements {
            sink.add("ForwardCandidate", (old, new))?;
        }
        Ok(())
    })?;
    Ok(())
}
