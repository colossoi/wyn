//! Reuse scalar readbacks until the resource is written or rebound.
use super::{Allocation, Entry, Expr, HostError, Operation, Program, ResourceId, ScalarExpr, ScalarSource};
use crate::ScalarType;
use std::collections::BTreeMap;

#[derive(Clone, Default)]
struct Readbacks {
    values: BTreeMap<(ResourceId, u32, ScalarType), String>,
    next: usize,
}

impl Readbacks {
    fn scalar(&mut self, value: &mut ScalarExpr, bindings: &mut Vec<Operation>) {
        match value {
            ScalarExpr::Read {
                source: ScalarSource::Resource(resource),
                offset,
                ty,
            } => {
                let key = (*resource, *offset, *ty);
                let name = if let Some(name) = self.values.get(&key) {
                    name.clone()
                } else {
                    let name = format!("host-readback-{}", self.next);
                    self.next += 1;
                    bindings.push(Operation::Bind {
                        name: name.clone(),
                        value: value.clone(),
                    });
                    self.values.insert(key, name.clone());
                    name
                };
                *value = ScalarExpr::Local(name);
            }
            ScalarExpr::Apply { args, .. } | ScalarExpr::Tuple(args) => {
                for arg in args {
                    self.scalar(arg, bindings);
                }
            }
            ScalarExpr::Field { tuple, .. } => self.scalar(tuple, bindings),
            ScalarExpr::If { condition, yes, no } => {
                self.scalar(condition, bindings);
                self.scoped(yes);
                self.scoped(no);
            }
            ScalarExpr::Let { bindings, result } => {
                let mut scope = self.clone();
                let mut rewritten = Vec::new();
                for (name, mut value) in std::mem::take(bindings) {
                    scope.lexical(&mut value, &mut rewritten);
                    rewritten.push((name, value));
                }
                scope.lexical(result, &mut rewritten);
                *bindings = rewritten;
                self.next = scope.next;
            }
            ScalarExpr::Loop {
                initial,
                condition,
                step,
                ..
            } => {
                self.scalar(initial, bindings);
                self.scoped(condition);
                self.scoped(step);
            }
            _ => {}
        }
    }

    fn lexical(&mut self, value: &mut ScalarExpr, bindings: &mut Vec<(String, ScalarExpr)>) {
        let mut operations = Vec::new();
        self.scalar(value, &mut operations);
        bindings.extend(operations.into_iter().map(|operation| match operation {
            Operation::Bind { name, value } => (name, value),
            _ => unreachable!("scalar readback generation only emits bindings"),
        }));
    }

    fn scoped(&mut self, value: &mut ScalarExpr) {
        let mut scope = self.clone();
        let mut bindings = Vec::new();
        scope.lexical(value, &mut bindings);
        if !bindings.is_empty() {
            *value = ScalarExpr::Let {
                bindings,
                result: Box::new(value.clone()),
            };
        }
        self.next = scope.next;
    }

    fn expression(&mut self, expression: &mut Expr, bindings: &mut Vec<Operation>) {
        match expression {
            Expr::Scalar(value) => self.scalar(value, bindings),
            Expr::ReadScalar {
                resource,
                offset,
                signed,
            } => {
                let mut value = ScalarExpr::Read {
                    source: ScalarSource::Resource(*resource),
                    offset: *offset,
                    ty: if *signed { ScalarType::I32 } else { ScalarType::U32 },
                };
                self.scalar(&mut value, bindings);
                *expression = Expr::Scalar(value);
            }
            Expr::Add(a, b)
            | Expr::Subtract(a, b)
            | Expr::Multiply(a, b)
            | Expr::Floor(a, b)
            | Expr::Ceiling(a, b)
            | Expr::Min(a, b)
            | Expr::Max(a, b) => {
                self.expression(a, bindings);
                self.expression(b, bindings);
            }
            _ => {}
        }
    }

    fn operations(&mut self, program: &Program, operations: &mut Vec<Operation>) -> Result<(), HostError> {
        for mut operation in std::mem::take(operations) {
            match &mut operation {
                Operation::Allocate(allocation) => {
                    match allocation {
                        Allocation::Buffer { bytes, .. } => self.expression(bytes, operations),
                        Allocation::Texture { width, height, .. } => {
                            self.expression(width, operations);
                            self.expression(height, operations);
                        }
                    }
                    self.values.retain(|(resource, ..), _| *resource != allocation.resource());
                }
                Operation::Loop { test, body, .. } => {
                    // A loop may rebind carried buffers, and its test executes after every body.
                    self.values.clear();
                    self.operations(program, test)?;
                    self.values.clear();
                    self.operations(program, body)?;
                    self.values.clear();
                }
                _ => {
                    let writes = program.operation_dependencies(&operation)?.writes;
                    if let Operation::Dispatch { groups, .. } = &mut operation {
                        for group in groups {
                            self.expression(group, operations);
                        }
                    }
                    if let Operation::Scalar { value, .. } = &mut operation {
                        self.scalar(value, operations);
                    }
                    self.values.retain(|(resource, ..), _| !writes.contains(resource));
                }
            }
            operations.push(operation);
        }
        Ok(())
    }
}

impl Program {
    pub(super) fn cache_readbacks(&self, entry: &mut Entry) -> Result<(), HostError> {
        let mut reads = Readbacks::default();
        reads.operations(self, &mut entry.operations)?;
        for result in &mut entry.source_results {
            for expression in result.shape.expressions_mut() {
                reads.expression(expression, &mut entry.operations);
            }
        }
        Ok(())
    }
}
