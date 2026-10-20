//! Value identity and evaluation dependencies for a typed TLC body.
//! Rebuild this snapshot after transformations; source term IDs are not updated.

use super::{ArrayExpr, LoopKind, Payload, SoacOp, Term, TermId, TermKind, VarRef};
use crate::ast::Span;
use crate::types::Type;
use crate::{LookupMap, LookupSet, StableMap, SymbolId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Value(usize);

#[derive(Clone, Copy)]
pub(crate) enum Source {
    Input(SymbolId),
    Term(TermId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Origin {
    pub producer: SymbolId,
    pub path: Vec<usize>,
}

enum Node {
    Input(SymbolId),
    Binding(Value),
    Aggregate(Vec<Value>),
    Projection(Value, usize),
    RecordUpdate {
        record: Value,
        index: usize,
        replacement: Value,
    },
    Choice {
        condition: Value,
        yes: Value,
        no: Value,
    },
    Computed(Vec<Value>),
}

#[derive(Clone, Copy)]
struct Expression {
    value: Value,
    evaluation: Value,
}

pub(crate) struct Dependencies(Vec<bool>);

struct Reference {
    id: TermId,
    ty: Type,
    span: Span,
}

#[derive(Default)]
pub(crate) struct ValueGraph {
    // Operands precede consumers; lexical binders have globally unique symbols.
    nodes: Vec<Node>,
    sources: Vec<Source>,
    bindings: LookupMap<SymbolId, Expression>,
    terms: LookupMap<TermId, Expression>,
    producers: LookupMap<Value, SymbolId>,
    references: StableMap<SymbolId, Reference>,
    locals: LookupSet<SymbolId>,
}

impl ValueGraph {
    pub fn build<C: Payload, S: Payload>(term: &Term<C, S>) -> (Self, Value) {
        let mut graph = Self::default();
        let result = graph.import(term).value;
        (graph, result)
    }

    /// First references in traversal order, excluding this body's binders and
    /// caller-supplied parameters/globals. Returned-value dependencies are insufficient.
    pub fn free_variables<C: Payload, S: Payload>(
        &self,
        mut excluded: impl FnMut(SymbolId) -> bool,
    ) -> Vec<Term<C, S>> {
        self.references
            .iter()
            .filter(|(symbol, _)| !self.locals.contains(symbol) && !excluded(**symbol))
            .map(|(&symbol, reference)| Term {
                id: reference.id,
                ty: reference.ty.clone(),
                span: reference.span,
                kind: TermKind::Var(VarRef::Symbol(symbol)),
            })
            .collect()
    }

    pub fn free_symbols(&self) -> impl Iterator<Item = SymbolId> + '_ {
        self.references.keys().copied().filter(|symbol| !self.locals.contains(symbol))
    }

    /// Includes control inputs and evaluated operands, even if their values are discarded.
    /// Dependence alone does not prove purity or permission to move an expression.
    pub fn dependencies(&self, mut selected: impl FnMut(SymbolId) -> bool) -> Dependencies {
        let mut depends = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            let dependent = match node {
                Node::Input(symbol) => selected(*symbol),
                Node::Binding(value) => depends[value.0],
                Node::Aggregate(inputs) | Node::Computed(inputs) => inputs.iter().any(|v| depends[v.0]),
                Node::Projection(input, field) => {
                    let projected = self.project(*input, &[*field]).unwrap_or(*input);
                    debug_assert!(projected.0 < index);
                    depends[projected.0]
                }
                Node::RecordUpdate {
                    record, replacement, ..
                } => depends[record.0] || depends[replacement.0],
                Node::Choice { condition, yes, no } => {
                    depends[condition.0] || depends[yes.0] || depends[no.0]
                }
            };
            depends.push(dependent);
        }
        Dependencies(depends)
    }

    pub fn dependent_bindings(&self, selected: impl FnMut(SymbolId) -> bool) -> LookupSet<SymbolId> {
        let depends = self.dependencies(selected);
        self.bindings
            .iter()
            .filter_map(|(&symbol, expr)| depends.0[expr.value.0].then_some(symbol))
            .collect()
    }

    pub fn evaluation_depends(&self, term: TermId, depends: &Dependencies) -> bool {
        // A missing term cannot justify hoisting it.
        self.terms.get(&term).is_none_or(|expr| depends.0[expr.evaluation.0])
    }

    pub fn evaluation_uses<C: Payload, S: Payload>(
        term: &Term<C, S>,
        selected: impl FnMut(SymbolId) -> bool,
    ) -> bool {
        let (graph, _) = Self::build(term);
        graph.evaluation_depends(term.id, &graph.dependencies(selected))
    }

    pub fn array_depends<C: Payload, S: Payload>(
        &self,
        array: &ArrayExpr<C, S>,
        depends: &Dependencies,
    ) -> bool {
        match array {
            ArrayExpr::Var(VarRef::Symbol(symbol), _) => {
                self.bindings.get(symbol).is_none_or(|expr| depends.0[expr.evaluation.0])
            }
            ArrayExpr::Var(VarRef::Builtin { .. }, _) => false,
            ArrayExpr::Zip(parts) => parts.iter().any(|part| self.array_depends(part, depends)),
            ArrayExpr::Literal(terms) => terms.iter().any(|term| self.evaluation_depends(term.id, depends)),
            ArrayExpr::Range { start, len, step } => {
                self.evaluation_depends(start.id, depends)
                    || self.evaluation_depends(len.id, depends)
                    || step.as_deref().is_some_and(|step| self.evaluation_depends(step.id, depends))
            }
        }
    }

    /// Resolve only structural forwarding, without moving values across choices.
    fn project(&self, value: Value, fields: &[usize]) -> Option<Value> {
        match &self.nodes[value.0] {
            Node::Binding(value) => self.project(*value, fields),
            Node::Projection(input, index) => match self.project(*input, &[*index]) {
                Some(field) => self.project(field, fields),
                None => fields.is_empty().then_some(value),
            },
            Node::Aggregate(values) if !fields.is_empty() => {
                self.project(*values.get(fields[0])?, &fields[1..])
            }
            Node::RecordUpdate {
                record,
                index,
                replacement,
            } if !fields.is_empty() => {
                if fields[0] == *index {
                    self.project(*replacement, &fields[1..])
                } else {
                    self.project(*record, fields)
                }
            }
            _ if fields.is_empty() => Some(value),
            _ => None,
        }
    }

    pub fn projected_source(&self, source: Source, fields: &[usize]) -> Option<Source> {
        let expr = match source {
            Source::Input(symbol) => self.bindings.get(&symbol)?,
            Source::Term(term) => self.terms.get(&term)?,
        };
        let value = self.project(expr.value, fields)?;
        Some(self.sources[value.0])
    }

    /// An input forwarded unchanged, including choices whose arms agree.
    pub fn input_origin(&self, value: Value, fields: &[usize]) -> Option<SymbolId> {
        match &self.nodes[value.0] {
            Node::Input(symbol) if fields.is_empty() => Some(*symbol),
            Node::Binding(value) => self.input_origin(*value, fields),
            Node::Aggregate(values) if !fields.is_empty() => {
                self.input_origin(*values.get(fields[0])?, &fields[1..])
            }
            Node::RecordUpdate {
                record,
                index,
                replacement,
            } if !fields.is_empty() => {
                if fields[0] == *index {
                    self.input_origin(*replacement, &fields[1..])
                } else {
                    self.input_origin(*record, fields)
                }
            }
            Node::Projection(input, index) => {
                let mut path = vec![*index];
                path.extend_from_slice(fields);
                self.input_origin(*input, &path)
            }
            Node::Choice { yes, no, .. } => {
                let origin = self.input_origin(*yes, fields)?;
                (*yes == *no || self.input_origin(*no, fields) == Some(origin)).then_some(origin)
            }
            _ => None,
        }
    }

    /// Mark a materialization boundary unless forwarding preserves its evaluation.
    pub fn name_producer(&mut self, symbol: SymbolId) {
        let Some(&expr) = self.bindings.get(&symbol) else {
            return;
        };
        if self.forwarding_origin(expr).is_none() {
            self.producers.insert(expr.value, symbol);
        }
    }

    pub fn binding_origin(&self, symbol: SymbolId, fields: &[usize]) -> Option<Origin> {
        self.origin(self.bindings.get(&symbol)?.value, fields)
    }

    /// An existing producer that can replace this term without dropping evaluation.
    pub fn term_origin(&self, term: TermId) -> Option<Origin> {
        self.forwarding_origin(*self.terms.get(&term)?)
    }

    fn forwarding_origin(&self, expr: Expression) -> Option<Origin> {
        let origin = self.origin(expr.value, &[])?;
        let producer = self.bindings.get(&origin.producer)?;
        (expr.evaluation == producer.evaluation).then_some(origin)
    }

    fn origin(&self, value: Value, fields: &[usize]) -> Option<Origin> {
        if let Some(&producer) = self.producers.get(&value) {
            return Some(Origin {
                producer,
                path: fields.to_vec(),
            });
        }
        if let Node::Binding(value) = &self.nodes[value.0] {
            return self.origin(*value, fields);
        }
        if let Node::Projection(input, index) = &self.nodes[value.0] {
            let mut path = vec![*index];
            path.extend_from_slice(fields);
            return self.origin(*input, &path);
        }
        None
    }

    fn push(&mut self, node: Node, source: Source) -> Value {
        let value = Value(self.nodes.len());
        self.nodes.push(node);
        self.sources.push(source);
        value
    }

    fn symbol(&mut self, symbol: SymbolId) -> Expression {
        if let Some(&expr) = self.bindings.get(&symbol) {
            return expr;
        }
        let value = self.push(Node::Input(symbol), Source::Input(symbol));
        let expr = Expression {
            value,
            evaluation: value,
        };
        self.bindings.insert(symbol, expr);
        expr
    }

    fn import<C: Payload, S: Payload>(&mut self, term: &Term<C, S>) -> Expression {
        // Alpha-renamed binders are unique within a body, so membership suffices
        // for whole-body capture queries without copying environments at each let.
        match &term.kind {
            TermKind::Let { name, .. } => {
                self.locals.insert(*name);
            }
            TermKind::Lambda(lambda) => self.locals.extend(lambda.params.iter().map(|(name, _)| *name)),
            TermKind::Loop {
                loop_var,
                init_bindings,
                kind,
                ..
            } => {
                self.locals.insert(*loop_var);
                self.locals.extend(init_bindings.iter().map(|(name, _, _)| *name));
                if let LoopKind::For { var, .. } | LoopKind::ForRange { var, .. } = kind {
                    self.locals.insert(*var);
                }
            }
            TermKind::Soac(soac) => {
                let body = match soac {
                    SoacOp::Map { lam, .. }
                    | SoacOp::Scatter { lam, .. }
                    | SoacOp::BucketScatter { lam, .. } => lam,
                    SoacOp::Reduce { op, .. }
                    | SoacOp::Scan { op, .. }
                    | SoacOp::ReduceByIndex { op, .. } => op,
                    SoacOp::Filter { pred, .. } => pred,
                };
                self.locals.extend(body.lam.params.iter().map(|(name, _)| *name));
            }
            _ => {}
        }
        let source = Source::Term(term.id);
        let update = super::record_update_parts(term);
        let expr = match &term.kind {
            TermKind::Var(VarRef::Symbol(symbol))
            | TermKind::ArrayExpr(ArrayExpr::Var(VarRef::Symbol(symbol), _)) => {
                self.references.entry(*symbol).or_insert_with(|| Reference {
                    id: term.id,
                    ty: term.ty.clone(),
                    span: term.span,
                });
                self.symbol(*symbol)
            }
            TermKind::Coerce { inner, .. } => self.import(inner),
            TermKind::App { .. } if update.is_some() => {
                let (record, index, replacement) = update.unwrap();
                let record = self.import(record);
                let replacement = self.import(replacement);
                let value = self.push(
                    Node::RecordUpdate {
                        record: record.value,
                        index,
                        replacement: replacement.value,
                    },
                    source,
                );
                self.evaluated(value, &[record, replacement], source)
            }
            TermKind::Let { name, rhs, body, .. } => {
                let rhs = self.import(rhs);
                // Keep the binding boundary without copying its computation.
                let value = self.push(Node::Binding(rhs.value), Source::Input(*name));
                self.bindings.insert(
                    *name,
                    Expression {
                        value,
                        evaluation: rhs.evaluation,
                    },
                );
                let body = self.import(body);
                let evaluation = self.push(Node::Computed(vec![rhs.evaluation, body.evaluation]), source);
                Expression {
                    value: body.value,
                    evaluation,
                }
            }
            TermKind::Tuple(fields)
            | TermKind::VecLit(fields)
            | TermKind::ArrayExpr(ArrayExpr::Literal(fields)) => {
                let fields: Vec<_> = fields.iter().map(|field| self.import(field)).collect();
                let value = self.push(
                    Node::Aggregate(fields.iter().map(|field| field.value).collect()),
                    source,
                );
                self.evaluated(value, &fields, source)
            }
            TermKind::TupleProj { tuple, idx } => {
                let input = self.import(tuple);
                let value = self.push(Node::Projection(input.value, *idx), source);
                Expression {
                    value,
                    evaluation: input.evaluation,
                }
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let condition = self.import(cond);
                let yes = self.import(then_branch);
                let no = self.import(else_branch);
                let value = self.push(
                    Node::Choice {
                        condition: condition.value,
                        yes: yes.value,
                        no: no.value,
                    },
                    source,
                );
                self.evaluated(value, &[condition, yes, no], source)
            }
            _ => {
                // shortcut: calls and loops retain operand dependencies; result
                // summaries and loop-carried edges need a separate extension.
                let mut inputs = Vec::new();
                term.for_each_child(&mut |child| inputs.push(self.import(child).evaluation));
                if let TermKind::Soac(
                    SoacOp::Scatter { dest, .. }
                    | SoacOp::BucketScatter { dest, .. }
                    | SoacOp::ReduceByIndex { dest, .. },
                ) = &term.kind
                {
                    inputs.push(self.symbol(dest.id).evaluation);
                }
                let value = self.push(Node::Computed(inputs), source);
                Expression {
                    value,
                    evaluation: value,
                }
            }
        };
        if term.id != TermId::SYNTHETIC {
            self.terms.insert(term.id, expr);
        }
        expr
    }

    fn evaluated(&mut self, value: Value, operands: &[Expression], source: Source) -> Expression {
        let evaluation = if operands.iter().all(|expr| expr.value == expr.evaluation) {
            value
        } else {
            self.push(
                Node::Computed(operands.iter().map(|expr| expr.evaluation).collect()),
                source,
            )
        };
        Expression { value, evaluation }
    }
}

#[cfg(test)]
#[path = "value_graph_tests.rs"]
mod tests;
