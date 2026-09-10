//! Symbolic scalar recipes. Only source/catalog identities and owned facts live here.
use crate::SortedSet;
use wyn_base::IdArena;
use wyn_fusion::PortId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct TypeId(u32);
impl From<u32> for TypeId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct SourceLambdaId(u32);
impl From<u32> for SourceLambdaId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct LambdaId(u32);
impl From<u32> for LambdaId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct NodeId(u32);
impl From<u32> for NodeId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct CallId(u32);
impl From<u32> for CallId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct PrimitiveId(u32);
impl From<u32> for PrimitiveId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Shape {
    Other,
    Tuple,
    Project(usize),
    Union,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Capture {
    pub value: PortId,
    pub ty: TypeId,
}

#[derive(Clone, Debug)]
pub(super) enum Code {
    Source(SourceLambdaId),
    Expression(LambdaId),
}

#[derive(Clone, Debug)]
pub(super) struct Body {
    pub code: Code,
    pub captures: Vec<Capture>,
}

#[derive(Clone, Debug)]
pub(super) struct Dependency {
    pub inputs: SortedSet<usize>,
    pub projectable: bool,
}

#[derive(Clone, Debug)]
pub(super) struct Lambda {
    pub original: Option<SourceLambdaId>,
    pub body: Option<Body>,
    pub parameter_types: Vec<TypeId>,
    pub result_types: Vec<TypeId>,
    pub dependencies: Vec<Dependency>,
}

impl Lambda {
    pub fn identity(types: Vec<TypeId>) -> Self {
        let dependencies = (0..types.len())
            .map(|index| Dependency {
                inputs: SortedSet::from([index]),
                projectable: true,
            })
            .collect();
        Self {
            original: None,
            body: None,
            parameter_types: types.clone(),
            result_types: types,
            dependencies,
        }
    }
    pub fn is_identity(&self) -> bool {
        self.body.is_none()
    }
    pub fn seg_body(&self) -> Option<&Body> {
        self.body.as_ref()
    }
    pub fn captures(&self) -> &[Capture] {
        self.body.as_ref().map_or(&[], |body| &body.captures)
    }
    pub fn capture_count(&self) -> usize {
        self.captures().len()
    }
    pub fn projectable(&self, results: impl IntoIterator<Item = usize>) -> bool {
        results
            .into_iter()
            .all(|slot| self.dependencies.get(slot).is_some_and(|dependency| dependency.projectable))
    }
    pub fn depends_on(&self, results: impl IntoIterator<Item = usize>, parameters: &[usize]) -> bool {
        results.into_iter().any(|slot| {
            self.dependencies.get(slot).is_none_or(|dependency| {
                parameters.iter().any(|parameter| dependency.inputs.contains(parameter))
            })
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Neutral {
    Value(PortId),
    Zero(TypeId),
}

#[derive(Clone, Debug)]
pub(super) struct Scan {
    pub operator: Lambda,
    pub neutral: Vec<Neutral>,
}
#[derive(Clone, Debug)]
pub(super) struct Reduce {
    pub operator: Lambda,
    pub neutral: Vec<Neutral>,
    pub commutative: bool,
}
#[derive(Clone, Debug)]
pub(super) struct ScremaForm {
    pub pre: Lambda,
    pub scans: Vec<Scan>,
    pub reductions: Vec<Reduce>,
    pub post: Lambda,
}
impl ScremaForm {
    pub fn scan_input_count(&self) -> usize {
        self.scans.iter().map(|scan| scan.neutral.len()).sum()
    }
    pub fn reduction_input_count(&self) -> usize {
        self.reductions.iter().map(|reduce| reduce.neutral.len()).sum()
    }
    pub fn operator_input_count(&self) -> usize {
        self.scan_input_count() + self.reduction_input_count()
    }
    pub fn reduction_result_count(&self) -> usize {
        self.reductions.iter().map(|reduce| reduce.operator.result_types.len()).sum()
    }
    pub fn result_count(&self) -> usize {
        self.reduction_result_count() + self.post.result_types.len()
    }
    pub fn mapped_types(&self) -> Option<&[TypeId]> {
        self.pre.result_types.get(self.operator_input_count()..)
    }
    pub fn is_map(&self) -> bool {
        self.scans.is_empty() && self.reductions.is_empty()
    }
    pub fn captures(&self) -> impl Iterator<Item = Capture> + '_ {
        self.pre
            .captures()
            .iter()
            .chain(self.post.captures())
            .chain(self.scans.iter().flat_map(|scan| scan.operator.captures()))
            .chain(self.reductions.iter().flat_map(|reduce| reduce.operator.captures()))
            .copied()
    }
}

#[derive(Clone, Debug)]
pub(super) enum NodeKind {
    Parameter(usize),
    CallResult(CallId, usize),
    Integer(u32),
    Add(NodeId, NodeId),
    Select {
        predicate: NodeId,
        yes: NodeId,
        no: NodeId,
    },
    Primitive {
        id: PrimitiveId,
        shape: Shape,
        operands: Vec<NodeId>,
    },
}
#[derive(Clone, Debug)]
pub(super) struct Node {
    pub kind: NodeKind,
    pub ty: TypeId,
    pub dependency: Dependency,
}
#[derive(Clone, Debug)]
pub(super) struct Call {
    pub lambda: Lambda,
    pub arguments: Vec<Option<NodeId>>,
    pub results: Vec<usize>,
}
#[derive(Clone, Debug, Default)]
pub(super) struct Expression {
    pub label: String,
    pub nodes: IdArena<NodeId, Node>,
    pub calls: IdArena<CallId, Call>,
    pub results: Vec<NodeId>,
}

impl Expression {
    /// Project the symbolic DAG using the same checked result dependencies
    /// composed during construction. Calls retain only their demanded lanes.
    pub fn selected_calls(&self, selected: &[usize]) -> crate::LookupMap<CallId, SortedSet<usize>> {
        let mut calls = crate::LookupMap::<CallId, SortedSet<usize>>::new();
        let mut visited = crate::LookupSet::new();
        let mut pending =
            selected.iter().filter_map(|slot| self.results.get(*slot).copied()).collect::<Vec<_>>();
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            match &self.nodes[id].kind {
                NodeKind::CallResult(id, slot) => {
                    calls.entry(*id).or_default().insert(*slot);
                    let call = &self.calls[*id];
                    for input in &call.lambda.dependencies[call.results[*slot]].inputs {
                        pending.extend(call.arguments[*input]);
                    }
                }
                NodeKind::Add(left, right) => pending.extend([*left, *right]),
                NodeKind::Select { predicate, yes, no } => pending.extend([*predicate, *yes, *no]),
                NodeKind::Primitive { operands, .. } => pending.extend(operands),
                NodeKind::Parameter(_) | NodeKind::Integer(_) => {}
            }
        }
        calls
    }
}
#[derive(Clone, Debug, Default)]
pub(super) struct Recipes {
    pub expressions: IdArena<LambdaId, Expression>,
}

pub(super) struct Builder {
    expression: Expression,
    parameter_types: Vec<TypeId>,
    captures: Vec<Capture>,
    pub arguments: Vec<NodeId>,
}
impl Builder {
    pub fn new(parameter_types: Vec<TypeId>, captures: Vec<Capture>) -> Self {
        let mut expression = Expression::default();
        let arguments = parameter_types
            .iter()
            .copied()
            .chain(captures.iter().map(|capture| capture.ty))
            .enumerate()
            .map(|(index, ty)| {
                expression.nodes.alloc(Node {
                    kind: NodeKind::Parameter(index),
                    ty,
                    dependency: Dependency {
                        inputs: SortedSet::from([index]),
                        projectable: true,
                    },
                })
            })
            .collect();
        Self {
            expression,
            parameter_types,
            captures,
            arguments,
        }
    }

    pub fn invoke(
        &mut self,
        recipes: &Recipes,
        lambda: &Lambda,
        arguments: Vec<NodeId>,
    ) -> Option<Vec<NodeId>> {
        self.project(
            recipes,
            lambda,
            &arguments.into_iter().map(Some).collect::<Vec<_>>(),
            &(0..lambda.result_types.len()).collect::<Vec<_>>(),
        )
    }

    pub fn project(
        &mut self,
        recipes: &Recipes,
        lambda: &Lambda,
        arguments: &[Option<NodeId>],
        results: &[usize],
    ) -> Option<Vec<NodeId>> {
        if arguments.len() != lambda.parameter_types.len() + lambda.capture_count() {
            return None;
        }
        if results.iter().any(|slot| *slot >= lambda.result_types.len()) {
            return None;
        }
        for slot in results {
            if lambda.dependencies[*slot]
                .inputs
                .iter()
                .any(|index| arguments.get(*index).is_none_or(Option::is_none))
            {
                return None;
            }
        }
        if lambda.is_identity() {
            return results.iter().map(|slot| arguments[*slot]).collect();
        }
        if let Some(Body {
            code: Code::Expression(id),
            ..
        }) = &lambda.body
        {
            let expression = &recipes.expressions[*id];
            let mut memo = crate::LookupMap::new();
            return results
                .iter()
                .map(|slot| {
                    self.substitute(
                        recipes,
                        expression,
                        expression.results[*slot],
                        arguments,
                        &mut memo,
                    )
                })
                .collect();
        }
        let call = self.expression.calls.alloc(Call {
            lambda: lambda.clone(),
            arguments: arguments.to_vec(),
            results: results.to_vec(),
        });
        Some(
            results
                .iter()
                .enumerate()
                .map(|(output, slot)| {
                    let source = &lambda.dependencies[*slot];
                    let mut dependency = Dependency {
                        inputs: SortedSet::new(),
                        projectable: source.projectable,
                    };
                    for argument in &source.inputs {
                        let node = &self.expression.nodes[arguments[*argument].unwrap()];
                        dependency.inputs.extend(&node.dependency.inputs);
                        dependency.projectable &= node.dependency.projectable;
                    }
                    self.expression.nodes.alloc(Node {
                        kind: NodeKind::CallResult(call, output),
                        ty: lambda.result_types[*slot],
                        dependency,
                    })
                })
                .collect(),
        )
    }

    fn substitute(
        &mut self,
        recipes: &Recipes,
        source: &Expression,
        node: NodeId,
        arguments: &[Option<NodeId>],
        memo: &mut crate::LookupMap<NodeId, NodeId>,
    ) -> Option<NodeId> {
        if let Some(node) = memo.get(&node) {
            return Some(*node);
        }
        let original = &source.nodes[node];
        let ty = original.ty;
        let value = match &original.kind {
            NodeKind::Parameter(slot) => arguments[*slot]?,
            NodeKind::CallResult(id, slot) => {
                let call = &source.calls[*id];
                let selected = call.results[*slot];
                let needed = &call.lambda.dependencies[selected].inputs;
                let args = call
                    .arguments
                    .iter()
                    .enumerate()
                    .map(|(index, node)| {
                        if needed.contains(&index) {
                            node.and_then(|node| self.substitute(recipes, source, node, arguments, memo))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                self.project(recipes, &call.lambda, &args, &[selected])?[0]
            }
            NodeKind::Integer(value) => self.integer(*value, ty),
            NodeKind::Add(left, right) => {
                let left = self.substitute(recipes, source, *left, arguments, memo)?;
                let right = self.substitute(recipes, source, *right, arguments, memo)?;
                self.add(left, right, ty)
            }
            NodeKind::Select { predicate, yes, no } => {
                let predicate = self.substitute(recipes, source, *predicate, arguments, memo)?;
                let yes = self.substitute(recipes, source, *yes, arguments, memo)?;
                let no = self.substitute(recipes, source, *no, arguments, memo)?;
                self.select(predicate, yes, no, ty)
            }
            NodeKind::Primitive { id, shape, operands } => {
                if let Shape::Project(slot) = shape {
                    if let Some(input) = operands.first() {
                        if let NodeKind::Primitive {
                            shape: Shape::Tuple,
                            operands: fields,
                            ..
                        } = &source.nodes[*input].kind
                        {
                            return self.substitute(recipes, source, *fields.get(*slot)?, arguments, memo);
                        }
                    }
                }
                let operands = operands
                    .iter()
                    .map(|node| self.substitute(recipes, source, *node, arguments, memo))
                    .collect::<Option<Vec<_>>>()?;
                self.primitive(*id, *shape, operands, ty)?
            }
        };
        memo.insert(node, value);
        Some(value)
    }

    pub fn primitive(
        &mut self,
        id: PrimitiveId,
        shape: Shape,
        operands: Vec<NodeId>,
        ty: TypeId,
    ) -> Option<NodeId> {
        if let Shape::Project(slot) = shape {
            if let Some(operand) = operands.first() {
                if let NodeKind::Primitive {
                    shape: Shape::Tuple,
                    operands,
                    ..
                } = &self.expression.nodes[*operand].kind
                {
                    return operands.get(slot).copied();
                }
            }
        }
        Some(self.combine(
            NodeKind::Primitive {
                id,
                shape,
                operands: operands.clone(),
            },
            &operands,
            ty,
        ))
    }

    pub fn ty(&self, node: NodeId) -> TypeId {
        self.expression.nodes[node].ty
    }

    pub fn integer(&mut self, value: u32, ty: TypeId) -> NodeId {
        self.expression.nodes.alloc(Node {
            kind: NodeKind::Integer(value),
            ty,
            dependency: Dependency {
                inputs: SortedSet::new(),
                projectable: true,
            },
        })
    }

    fn combine(&mut self, kind: NodeKind, inputs: &[NodeId], ty: TypeId) -> NodeId {
        let dependency = Dependency {
            inputs: inputs
                .iter()
                .flat_map(|node| self.expression.nodes[*node].dependency.inputs.iter().copied())
                .collect(),
            projectable: inputs.iter().all(|node| self.expression.nodes[*node].dependency.projectable),
        };
        self.expression.nodes.alloc(Node { kind, ty, dependency })
    }

    pub fn add(&mut self, left: NodeId, right: NodeId, ty: TypeId) -> NodeId {
        self.combine(NodeKind::Add(left, right), &[left, right], ty)
    }

    pub fn select(&mut self, predicate: NodeId, yes: NodeId, no: NodeId, ty: TypeId) -> NodeId {
        self.combine(NodeKind::Select { predicate, yes, no }, &[predicate, yes, no], ty)
    }

    pub fn finish(self, recipes: &mut Recipes, results: Vec<NodeId>) -> Lambda {
        self.finish_named(recipes, results, "fusion")
    }

    pub fn finish_named(mut self, recipes: &mut Recipes, results: Vec<NodeId>, label: &str) -> Lambda {
        self.expression.label = label.to_owned();
        let result_types = results.iter().map(|node| self.expression.nodes[*node].ty).collect();
        if self.captures.is_empty() && results == self.arguments {
            return Lambda::identity(self.parameter_types);
        }
        let dependencies =
            results.iter().map(|node| self.expression.nodes[*node].dependency.clone()).collect();
        self.expression.results = results;
        let id = recipes.expressions.alloc(self.expression);
        Lambda {
            original: None,
            body: Some(Body {
                code: Code::Expression(id),
                captures: self.captures,
            }),
            parameter_types: self.parameter_types,
            result_types,
            dependencies,
        }
    }
}
