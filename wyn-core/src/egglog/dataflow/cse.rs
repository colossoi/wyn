//! Match repeatable expressions against the definitions visible at each occurrence.
use super::{body::BodyGraph, ir::Code};
use crate::egglog::graph::Formula;
use crate::egglog::graph::Graph;
use crate::egglog::{query::Query, OptimizeError};
use crate::memory_ir::{Definition, OperationId, RegionId, ValueRef};
use crate::{LookupMap, LookupSet};
use egglog_engine::{Value, Write};
use std::borrow::Cow;
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
    fn execution_region(
        &mut self,
        query: Query<'_>,
        region: Value,
        products: &mut LookupMap<Value, Vec<Value>>,
    ) -> Result<(), OptimizeError> {
        let mut definitions = Vec::new();
        let graph = query.0;
        let operations = match graph.region_orders.get(&region) {
            Some(ordered) if !graph.dirty_orders.contains(&region) => Cow::Borrowed(ordered.as_slice()),
            _ => Cow::Owned(query.operations(region)?),
        };
        for &operation in operations.iter() {
            for child in query.0.attached_regions(operation) {
                self.execution_region(query, child, products)?;
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
            let operands = graph.inputs(operation);
            let ty = crate::egglog::analysis::typed(query, output)?;
            if let (Code::Project { index }, [input]) = (&code, operands.as_slice()) {
                let product = self.replacements.get(input).copied().unwrap_or(*input);
                if let Some(field) = products.get(&product).and_then(|fields| fields.get(*index)) {
                    if query.value_name(*field)?.is_some()
                        && crate::egglog::analysis::typed(query, *field)? == ty
                    {
                        self.replacements.insert(output, *field);
                        continue;
                    }
                }
            }
            let expression = Expression {
                code,
                operands,
                index: graph.formula_at(operation, Formula::Index),
                ty,
            };
            if let Some(expression) = self.observe(expression, output, true) {
                if expression.code == Code::Product {
                    products.insert(output, expression.operands.clone());
                }
                definitions.push((output, expression));
            }
        }
        for (output, expression) in definitions {
            if expression.code == Code::Product {
                products.remove(&output);
            }
            self.visible.remove(&expression);
        }
        Ok(())
    }
}

/// Forget completed searches in the lexical roots touched by the edit journal.
pub(in crate::egglog) fn invalidate(graph: &mut Graph) {
    if graph.clean_expression_roots.is_empty() {
        return;
    }
    let changes = &graph.proof_changes;
    let mut regions = changes.regions.clone();
    regions.extend(&graph.dirty_orders);
    {
        let structure = graph.structure().read();
        regions.extend(changes.operations.iter().filter_map(|op| structure.operations.get(op)?.region));
    }
    let mut visited = LookupSet::new();
    for mut region in regions {
        while visited.insert(region) {
            graph.clean_expression_roots.remove(&region);
            let Some(parent) = graph.region_owner(region).and_then(|op| graph.region_of(op)) else {
                break;
            };
            region = parent;
        }
    }
}

pub(in crate::egglog) fn find(
    graph: &mut Graph,
    roots: Vec<Value>,
) -> Result<LookupMap<Value, Value>, OptimizeError> {
    invalidate(graph);
    let mut reuse = Reuse::<Value>::default();
    let mut products = LookupMap::new();
    for region in roots {
        if graph.clean_expression_roots.contains(&region) {
            continue;
        }
        let previous = reuse.replacements.len();
        reuse.execution_region(Query(graph), region, &mut products)?;
        if reuse.replacements.len() == previous {
            graph.clean_expression_roots.insert(region);
        }
    }
    Ok(reuse.replacements)
}

pub(in crate::egglog) fn publish(graph: &mut Graph) -> Result<(), OptimizeError> {
    let roots = Query(graph).live_roots()?;
    let replacements = find(graph, roots)?;
    graph.update(|mut sink| {
        for (old, new) in replacements {
            sink.add("ForwardCandidate", (old, new))?;
        }
        Ok(())
    })?;
    Ok(())
}
