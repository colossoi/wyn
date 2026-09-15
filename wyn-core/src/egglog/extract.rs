//! Decode relational graph facts and one atomic fusion candidate. Values
//! retain their interned IDs; the current rules introduce no new pure values.

use super::data::{
    Array, AssociatedData, ExprId, OperationId, OperationKind, Reduction, RegionId, Scan, ScremaForm,
    SoacBody, TypeId,
};
use super::optimize::OptimizeError;
use crate::{types::SoacOwnership, LookupMap};
use egglog_engine::{ast::Literal, EGraph, Term, TermDag};
use std::collections::{BTreeMap, BTreeSet};

// These are indices in egglog's temporary extraction DAG, never compiler IDs.
type Node = egglog_engine::TermId;
type Result<T> = std::result::Result<T, OptimizeError>;

/// Select one whole candidate. A fresh snapshot after every contraction prevents
/// stale use/safety proofs and incompatible independently selected actions.
pub(super) fn fuse_one(graph: &EGraph, data: &mut AssociatedData) -> Result<bool> {
    let (rows, _, mut dag) = graph.function_to_dag("Expression", usize::MAX, false)?;
    let mut expressions = LookupMap::new();
    {
        let reader = Reader {
            dag: &dag,
            expressions: LookupMap::new(),
        };
        for row in rows {
            let args = reader.app(row, "Expression", 2)?;
            let id: ExprId = reader.key(args[0], "ExprId")?;
            expressions.entry(args[1]).and_modify(|old: &mut ExprId| *old = (*old).min(id)).or_insert(id);
        }
    }
    let (roots, _, candidates) = graph.function_to_dag("FusionCandidate", usize::MAX, false)?;
    let mut imported = LookupMap::new();
    let roots: Vec<_> =
        roots.into_iter().map(|root| import(&candidates, root, &mut dag, &mut imported)).collect();
    let reader = Reader {
        dag: &dag,
        expressions,
    };
    let mut choices = Vec::new();
    for root in roots {
        let args = reader.app(root, "FusionCandidate", 4)?;
        let region: RegionId = reader.key(args[0], "RegionId")?;
        let producer: OperationId = reader.key(args[1], "OperationId")?;
        let consumer: OperationId = reader.key(args[2], "OperationId")?;
        choices.push((region, producer, consumer, args[3]));
    }
    choices.sort_by_key(|&(r, p, c, _)| (r, p, c));
    let Some((region, producer, consumer, action)) = choices.into_iter().next() else {
        return Ok(false);
    };
    let args = reader.app(action, "ScremaApp", 3)?;
    let kind = OperationKind::Screma {
        form: reader.form(args[0])?,
        inputs: reader.arrays(args[1])?,
        ownership: reader
            .list(args[2], "NoOwnerships", "OwnershipsCons")?
            .into_iter()
            .map(|node| reader.ownership(node))
            .collect::<Result<_>>()?,
    };
    let Some(scope) = data.regions.get(region) else {
        return Err(invalid("fusion region has no sidecar record"));
    };
    if producer == consumer || !scope.members.contains(&producer) || !scope.members.contains(&consumer) {
        return Err(invalid("fusion candidate does not belong to the selected graph"));
    }
    let Some(record) = data.operations.get_mut(consumer) else {
        return Err(invalid("fusion consumer has no sidecar record"));
    };
    record.kind = kind;
    // Keep the producer record and any dead observers in the fact base. With
    // its sole live use absorbed, backward reachability stops selecting it.
    // Dead-operation elimination belongs to readout, not this mutation.
    Ok(true)
}

/// Only facts about the final selected graph, not speculative candidates.
pub(super) fn facts(graph: &EGraph) -> Result<String> {
    let mut output = String::new();
    for relation in [
        "DependsOn",
        "ResultDependsOn",
        "EffectBefore",
        "RequiredEffect",
        "LiveOperation",
    ] {
        let (rows, _, dag) = graph.function_to_dag(relation, usize::MAX, false)?;
        let mut rows: Vec<_> = rows.into_iter().map(|row| dag.to_string(row)).collect();
        rows.sort();
        for row in rows {
            output.push_str(&row);
            output.push('\n');
        }
    }
    Ok(output)
}

/// Liveness walked backward from region outputs and required effects. Order
/// only that induced graph; membership is not a schedule. Enclosing executions
/// satisfy edges to outer regions, keeping branch and loop work in its scope.
pub(super) fn schedules(
    graph: &EGraph,
    data: &AssociatedData,
) -> Result<BTreeMap<RegionId, Vec<OperationId>>> {
    let (rows, _, dag) = graph.function_to_dag("LiveOperation", usize::MAX, false)?;
    let reader = Reader {
        dag: &dag,
        expressions: LookupMap::new(),
    };
    let mut live = BTreeMap::<RegionId, BTreeSet<OperationId>>::new();
    for row in rows {
        let args = reader.app(row, "LiveOperation", 2)?;
        let region: RegionId = reader.key(args[0], "RegionId")?;
        let op: OperationId = reader.key(args[1], "OperationId")?;
        if !data.regions.get(region).is_some_and(|r| r.members.contains(&op)) {
            return Err(invalid("live operation is not a member of its region"));
        }
        live.entry(region).or_default().insert(op);
    }
    let mut edges = BTreeSet::<(OperationId, OperationId)>::new();
    for relation in ["DependsOn", "EffectBefore"] {
        let (rows, _, dag) = graph.function_to_dag(relation, usize::MAX, false)?;
        let reader = Reader {
            dag: &dag,
            expressions: LookupMap::new(),
        };
        for row in rows {
            let args = reader.app(row, relation, 2)?;
            let a: OperationId = reader.key(args[0], "OperationId")?;
            let b: OperationId = reader.key(args[1], "OperationId")?;
            edges.insert(if relation == "DependsOn" { (b, a) } else { (a, b) });
        }
    }
    let mut result = BTreeMap::new();
    for (region, members) in live {
        let mut indegree: BTreeMap<_, usize> = members.iter().map(|&op| (op, 0)).collect();
        let mut successors = BTreeMap::<OperationId, Vec<OperationId>>::new();
        for &(before, after) in &edges {
            if members.contains(&before) && members.contains(&after) {
                if let Some(count) = indegree.get_mut(&after) {
                    *count += 1;
                }
                successors.entry(before).or_default().push(after);
            }
        }
        let mut ready: BTreeSet<_> =
            indegree.iter().filter_map(|(&op, &n)| (n == 0).then_some(op)).collect();
        let mut ordered = Vec::new();
        while let Some(op) = ready.pop_first() {
            ordered.push(op);
            for &next in successors.get(&op).into_iter().flatten() {
                if let Some(count) = indegree.get_mut(&next) {
                    *count -= 1;
                    if *count == 0 {
                        ready.insert(next);
                    }
                }
            }
        }
        if ordered.len() != members.len() {
            return Err(invalid("cycle in the selected execution graph"));
        }
        result.insert(region, ordered);
    }
    Ok(result)
}

fn import(
    source: &TermDag,
    node: Node,
    target: &mut TermDag,
    imported: &mut LookupMap<Node, Node>,
) -> Node {
    if let Some(&id) = imported.get(&node) {
        return id;
    }
    let id = match source.get(node) {
        Term::App(name, args) => {
            let args = args.iter().map(|&node| import(source, node, target, imported)).collect();
            target.app(name.clone(), args)
        }
        Term::Lit(value) => target.lit(value.clone()),
        Term::Var(name) => target.var(name.clone()),
    };
    imported.insert(node, id);
    id
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::Extraction(message.into())
}

struct Reader<'a> {
    dag: &'a TermDag,
    expressions: LookupMap<Node, ExprId>,
}

impl Reader<'_> {
    fn app(&self, node: Node, expected: &str, arity: usize) -> Result<&[Node]> {
        match self.dag.get(node) {
            Term::App(name, args) if name == expected && args.len() == arity => Ok(args),
            _ => Err(invalid(&format!(
                "expected {expected}/{arity} in extracted graph"
            ))),
        }
    }
    fn key<Id: From<u32>>(&self, node: Node, name: &str) -> Result<Id> {
        let args = self.app(node, name, 1)?;
        let Term::Lit(Literal::Int(value)) = self.dag.get(args[0]) else {
            return Err(invalid("expected an integer sidecar ID"));
        };
        let value = u32::try_from(*value).map_err(|_| invalid("sidecar ID is out of range"))?;
        Ok(Id::from(value))
    }
    fn list(&self, mut node: Node, nil: &str, cons: &str) -> Result<Vec<Node>> {
        let mut values = Vec::new();
        loop {
            if self.app(node, nil, 0).is_ok() {
                return Ok(values);
            }
            let args = self.app(node, cons, 2)?;
            values.push(args[0]);
            node = args[1];
        }
    }
    fn expr(&self, node: Node) -> Result<ExprId> {
        let Some(&id) = self.expressions.get(&node) else {
            return Err(invalid("fusion introduced an unregistered expression"));
        };
        Ok(id)
    }
    fn exprs(&self, node: Node) -> Result<Vec<ExprId>> {
        self.list(node, "NoExprs", "ExprsCons")?.into_iter().map(|node| self.expr(node)).collect()
    }
    fn types(&self, node: Node) -> Result<Vec<TypeId>> {
        self.list(node, "NoTypes", "TypesCons")?.into_iter().map(|node| self.key(node, "TypeId")).collect()
    }
    fn body(&self, node: Node) -> Result<SoacBody> {
        let Term::App(name, _) = self.dag.get(node) else {
            return Err(invalid("expected a SOAC body"));
        };
        Ok(match name.as_str() {
            "RouteBody" => {
                let args = self.app(node, name, 2)?;
                let indices = self
                    .list(args[1], "NoIndices", "IndicesCons")?
                    .into_iter()
                    .map(|node| {
                        let Term::Lit(Literal::Int(index)) = self.dag.get(node) else {
                            return Err(invalid("expected a body argument index"));
                        };
                        usize::try_from(*index).map_err(|_| invalid("body argument index is out of range"))
                    })
                    .collect::<Result<_>>()?;
                SoacBody::Route {
                    parameters: self.types(args[0])?,
                    indices,
                }
            }
            "ParallelBody" => {
                let args = self.app(node, name, 2)?;
                SoacBody::Parallel {
                    left: Box::new(self.body(args[0])?),
                    right: Box::new(self.body(args[1])?),
                }
            }
            "ComposeBody" => {
                let args = self.app(node, name, 2)?;
                SoacBody::Compose {
                    first: Box::new(self.body(args[0])?),
                    then: Box::new(self.body(args[1])?),
                }
            }
            "ApplyBody" => {
                let args = self.app(node, name, 4)?;
                SoacBody::Apply {
                    region: self.key(args[0], "RegionId")?,
                    parameters: self.types(args[1])?,
                    results: self.types(args[2])?,
                    captures: self.exprs(args[3])?,
                }
            }
            "IdentityBody" => SoacBody::Identity(self.types(self.app(node, name, 1)?[0])?),
            _ => return Err(invalid("unknown SOAC body in extracted graph")),
        })
    }
    fn form(&self, node: Node) -> Result<ScremaForm> {
        let args = self.app(node, "Screma", 4)?;
        let scans = self
            .list(args[1], "NoCollectives", "CollectivesCons")?
            .into_iter()
            .map(|node| {
                let args = self.app(node, "Scan", 2)?;
                Ok(Scan {
                    operator: self.body(args[0])?,
                    neutral: self.exprs(args[1])?,
                })
            })
            .collect::<Result<_>>()?;
        let reductions = self
            .list(args[2], "NoCollectives", "CollectivesCons")?
            .into_iter()
            .map(|node| {
                let args = self.app(node, "Reduction", 3)?;
                let Term::Lit(Literal::Bool(commutative)) = self.dag.get(args[2]) else {
                    return Err(invalid("expected reduction commutativity"));
                };
                Ok(Reduction {
                    operator: self.body(args[0])?,
                    neutral: self.exprs(args[1])?,
                    commutative: *commutative,
                })
            })
            .collect::<Result<_>>()?;
        Ok(ScremaForm {
            pre: self.body(args[0])?,
            scans,
            reductions,
            post: self.body(args[3])?,
        })
    }
    fn arrays(&self, node: Node) -> Result<Vec<Array>> {
        self.list(node, "NoArrays", "ArraysCons")?.into_iter().map(|node| self.array(node)).collect()
    }
    fn array(&self, node: Node) -> Result<Array> {
        let Term::App(name, _) = self.dag.get(node) else {
            return Err(invalid("expected an array input"));
        };
        Ok(match name.as_str() {
            "ArrayInput" => Array::Value(self.expr(self.app(node, name, 1)?[0])?),
            "Zip" => Array::Zip(self.arrays(self.app(node, name, 1)?[0])?),
            "ArrayLiteral" => Array::Literal(self.exprs(self.app(node, name, 1)?[0])?),
            "Range" => {
                let args = self.app(node, name, 3)?;
                let step = if self.app(args[2], "NoExpr", 0).is_ok() {
                    None
                } else {
                    Some(self.expr(self.app(args[2], "SomeExpr", 1)?[0])?)
                };
                Array::Range {
                    start: self.expr(args[0])?,
                    len: self.expr(args[1])?,
                    step,
                }
            }
            _ => return Err(invalid("unknown array input in extracted graph")),
        })
    }
    fn ownership(&self, node: Node) -> Result<SoacOwnership> {
        if self.app(node, "Fresh", 0).is_ok() {
            return Ok(SoacOwnership::Fresh);
        }
        self.app(node, "UniqueInput", 0)?;
        Ok(SoacOwnership::UniqueInput)
    }
}
