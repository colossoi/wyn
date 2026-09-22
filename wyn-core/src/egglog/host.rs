//! Select sequential scalar evaluations and publish their dispatch captures.
use super::abi::u32_type;
use super::data::{
    BlockId, BufferId, ExprId, ExprKind, LoopKind, OperationId, OperationKind, ParameterId, RegionId,
};
use super::dependencies::analyze;
use super::{OptimizeError, Program, Scheduled, Value};
use crate::builtins::lowering::PrimOp;
use crate::builtins::{by_id, BuiltinLowering, Purity};
use crate::host::{
    Access, Binding, BufferLen, BufferUsage, ModuleInterface, Pipeline, ScalarExpr, ScalarSource,
    ScalarTask, ScalarType,
};
use crate::interface::{EntryInputKind, EntryKind, StorageBindingDecl, StorageRole};
use crate::ssa::types::EntryPoint;
use crate::types::{Type, TypeName};
use crate::{BindingRef, EntryId};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
pub(super) struct Host {
    pub captures: BTreeMap<(BlockId, ExprId), BindingRef>,
    pub inputs: BTreeMap<BlockId, BTreeSet<ParameterId>>,
    tasks: Vec<ScalarTask>,
    declarations: BTreeMap<BlockId, Vec<StorageBindingDecl>>,
}

pub(super) fn plan(data: &Program<Scheduled>) -> Result<Host, OptimizeError> {
    let mut result = Host::default();
    let schedules = analyze(data).schedules(data)?;
    let Some(mut binding) = data
        .state
        .abi
        .bindings
        .values()
        .map(|b| b.binding.binding)
        .chain(
            data.state
                .abi
                .inputs
                .values()
                .flatten()
                .filter_map(|i| i.descriptor_binding().map(|b| b.binding)),
        )
        .max()
        .unwrap_or(0)
        .checked_add(1)
    else {
        return Ok(result);
    };
    for dispatch in data.state.dispatches.values() {
        let root = dispatch.kernel;
        let stage = &data.state.abi.entry_names[&root];
        for &expression in &dispatch.captures {
            if !matches!(
                data.expressions[expression].kind,
                ExprKind::PureApp { .. } | ExprKind::If { .. } | ExprKind::Coerce(_)
            ) {
                continue;
            }
            let Some(ty) = scalar_type(&data.types[data.expressions[expression].ty].ty) else {
                continue;
            };
            let mut lower = Lower::new(data, &schedules);
            let Some(value) = lower.expression(expression) else {
                continue;
            };
            let slot = BindingRef::new(0, binding);
            let Some(next) = binding.checked_add(1) else { break };
            binding = next;
            result.captures.insert((root, expression), slot);
            let value = lower.finish(value);
            result.inputs.entry(root).or_default().extend(lower.inputs);
            result.declarations.entry(root).or_default().push(StorageBindingDecl {
                binding: slot,
                role: StorageRole::Input,
                logical_resource: Some(format!("{stage}_host_{}", expression.as_u32())),
                elem_ty: if ty == ScalarType::Bool {
                    u32_type()
                } else {
                    data.types[data.expressions[expression].ty].ty.clone()
                },
                length: Some(BufferLen::Fixed { bytes: 4 }),
            });
            result.tasks.push(ScalarTask {
                stage: stage.clone(),
                destination: ScalarSource::Binding {
                    set: 0,
                    binding: slot.binding,
                },
                offset: 0,
                ty,
                value,
                replaces_dispatch: false,
            });
        }
    }
    // Scalar stages have a single materialized result. Their source operation
    // can execute on the host when every dependency has a scalar host meaning.
    for dispatch in data.state.dispatches.values() {
        let root = dispatch.kernel;
        if data.state.blocks[root].interface.as_ref().is_none_or(|f| f.name != "scalar") {
            continue;
        }
        for (&op, materialized) in &data.state.materialized {
            let Some(ty) = scalar_type(&data.types[data.operations[op].ty].ty) else {
                continue;
            };
            let Some(buffer) = singleton_buffer(materialized) else {
                continue;
            };
            if !dispatch.writes.contains(&buffer) {
                continue;
            }
            let mut lower = Lower::new(data, &schedules);
            let Some(value) = lower.operation(op) else {
                continue;
            };
            let slot = data.state.abi.buffer_bindings[&buffer];
            let value = lower.finish(value);
            result.inputs.entry(root).or_default().extend(lower.inputs);
            result.tasks.push(ScalarTask {
                stage: data.state.abi.entry_names[&root].clone(),
                destination: source(slot),
                offset: 0,
                ty,
                value,
                replaces_dispatch: true,
            });
        }
    }
    for &(root, owner, _, finish) in &data.state.abi.roots {
        if !finish
            || data.entries[owner].declaration.entry_kind != EntryKind::Compute
            || data.entries[owner].declaration.compute_dispatch.is_some()
        {
            continue;
        }
        let outputs: Vec<_> = data.state.outputs.values().filter(|o| o.entry == owner && o.copy).collect();
        if outputs.is_empty() {
            continue;
        }
        let mut lower = Lower::new(data, &schedules);
        let tasks: Option<Vec<_>> = outputs
            .into_iter()
            .map(|output| {
                let ty = scalar_type(&data.types[data.expressions[output.expression].ty].ty)?;
                let value = lower.expression(output.expression)?;
                let slot = *data.state.abi.buffer_bindings.get(&output.buffer?)?;
                Some(ScalarTask {
                    stage: data.state.abi.entry_names[&root].clone(),
                    destination: source(slot),
                    offset: 0,
                    ty,
                    value: lower.finish(value),
                    replaces_dispatch: true,
                })
            })
            .collect();
        if let Some(tasks) = tasks {
            result.tasks.extend(tasks);
            result.inputs.entry(root).or_default().extend(lower.inputs);
        }
    }
    Ok(result)
}

impl Host {
    pub fn publish(
        self,
        pipeline: &mut ModuleInterface,
        entries: &mut [EntryPoint],
    ) -> Result<(), OptimizeError> {
        for (root, declarations) in self.declarations {
            let Some(entry) = entries.iter_mut().find(|e| e.id == EntryId::from(root.as_u32())) else {
                return Err(OptimizeError::Output(format!(
                    "missing host capture entry {root:?}"
                )));
            };
            let mut published = false;
            for p in &mut pipeline.pipelines {
                let Pipeline::Compute(p) = p else { continue };
                let Some(stage) = p.stages.iter_mut().find(|s| s.entry_point == entry.name) else {
                    continue;
                };
                published = true;
                for declaration in &declarations {
                    let Some(name) = &declaration.logical_resource else {
                        return Err(OptimizeError::Output("host capture has no resource name".into()));
                    };
                    let index = p.bindings.len();
                    stage.uses.record(index, Access::ReadOnly);
                    p.bindings.push(Binding::StorageBuffer {
                        set: declaration.binding.set,
                        binding: declaration.binding.binding,
                        access: Access::ReadOnly,
                        usage: BufferUsage::Intermediate,
                        name: name.clone(),
                        resource: declaration.logical_resource.clone(),
                        length: declaration.length.clone(),
                        members: vec![],
                    });
                }
            }
            if !published {
                return Err(OptimizeError::Output(format!(
                    "missing host capture stage {}",
                    entry.name
                )));
            }
            entry.storage_bindings.extend(declarations);
        }
        pipeline.scalar_tasks = self.tasks;
        pipeline.rebuild_frame_graph();
        Ok(())
    }
}

fn source(binding: BindingRef) -> ScalarSource {
    ScalarSource::Binding {
        set: binding.set,
        binding: binding.binding,
    }
}
fn singleton_buffer(value: &Value) -> Option<BufferId> {
    let Value::Primitive("index", args) = value else {
        return None;
    };
    let [Value::Buffer(buffer), Value::Int(0)] = args.as_slice() else {
        return None;
    };
    Some(*buffer)
}
fn scalar_type(ty: &Type) -> Option<ScalarType> {
    match ty {
        Type::Constructed(TypeName::Bool, _) => Some(ScalarType::Bool),
        Type::Constructed(TypeName::Int(32), _) => Some(ScalarType::I32),
        Type::Constructed(TypeName::UInt(32), _) => Some(ScalarType::U32),
        Type::Constructed(TypeName::Float(32), _) => Some(ScalarType::F32),
        _ => None,
    }
}

struct Lower<'a> {
    data: &'a Program<Scheduled>,
    schedules: &'a BTreeMap<RegionId, Vec<OperationId>>,
    inputs: BTreeSet<ParameterId>,
    reads: BTreeMap<ParameterId, ScalarExpr>,
    parameters: BTreeMap<ParameterId, ScalarExpr>,
    values: BTreeMap<ExprId, ScalarExpr>,
    operations: BTreeMap<OperationId, ScalarExpr>,
    bindings: Vec<(String, ScalarExpr)>,
    active: BTreeSet<OperationId>,
}
impl<'a> Lower<'a> {
    fn new(data: &'a Program<Scheduled>, schedules: &'a BTreeMap<RegionId, Vec<OperationId>>) -> Self {
        Self {
            data,
            schedules,
            inputs: BTreeSet::new(),
            reads: BTreeMap::new(),
            parameters: BTreeMap::new(),
            values: BTreeMap::new(),
            operations: BTreeMap::new(),
            bindings: vec![],
            active: BTreeSet::new(),
        }
    }
    fn finish(&self, result: ScalarExpr) -> ScalarExpr {
        ScalarExpr::Let {
            bindings: self
                .reads
                .iter()
                .map(|(id, value)| (format!("host-input-{}", id.as_u32()), value.clone()))
                .chain(self.bindings.iter().cloned())
                .collect(),
            result: Box::new(result),
        }
    }
    fn expression(&mut self, id: ExprId) -> Option<ScalarExpr> {
        if let Some(value) = self.values.get(&id) {
            return Some(value.clone());
        }
        let value = self.expression_body(id)?;
        if matches!(
            value,
            ScalarExpr::I32(_)
                | ScalarExpr::U32(_)
                | ScalarExpr::F32(_)
                | ScalarExpr::Bool(_)
                | ScalarExpr::Local(_)
        ) {
            return Some(value);
        }
        let name = format!("host-value-{}", id.as_u32());
        self.bindings.push((name.clone(), value));
        let value = ScalarExpr::Local(name);
        self.values.insert(id, value.clone());
        Some(value)
    }
    // A branch or invocation owns its bindings. Only values dominating that
    // scope can be reused; expressions from an untaken arm stay unevaluated.
    fn scope(&mut self, lower: impl FnOnce(&mut Self) -> Option<ScalarExpr>) -> Option<ScalarExpr> {
        let bindings = std::mem::take(&mut self.bindings);
        let values = self.values.clone();
        let operations = self.operations.clone();
        let result = lower(self);
        let local = std::mem::replace(&mut self.bindings, bindings);
        self.values = values;
        self.operations = operations;
        result.map(|result| ScalarExpr::Let {
            bindings: local,
            result: Box::new(result),
        })
    }
    fn expression_body(&mut self, id: ExprId) -> Option<ScalarExpr> {
        let data = self.data;
        let expression = &data.expressions[id];
        let ty = &data.types[expression.ty].ty;
        Some(match &expression.kind {
            ExprKind::Int(n) => match scalar_type(ty)? {
                ScalarType::I32 => ScalarExpr::I32(n.parse().ok()?),
                ScalarType::U32 => ScalarExpr::U32(n.parse().ok()?),
                _ => return None,
            },
            ExprKind::FloatBits(bits) if scalar_type(ty) == Some(ScalarType::F32) => ScalarExpr::F32(*bits),
            ExprKind::Bool(b) => ScalarExpr::Bool(*b),
            ExprKind::Unit => ScalarExpr::Tuple(vec![]),
            ExprKind::Tuple(fields) => {
                ScalarExpr::Tuple(fields.iter().map(|e| self.expression(*e)).collect::<Option<_>>()?)
            }
            ExprKind::Project { tuple, index } => {
                // SOAC results can be split across buffers. Read only the
                // selected scalar field, without requiring its sibling arrays.
                if let ExprKind::OperationResult(op) = data.expressions[*tuple].kind {
                    if let Some(Value::Tuple(fields)) = data.state.materialized.get(&op) {
                        let buffer = singleton_buffer(fields.get(*index)?)?;
                        return Some(ScalarExpr::Read {
                            source: source(*data.state.abi.buffer_bindings.get(&buffer)?),
                            offset: 0,
                            ty: scalar_type(ty)?,
                        });
                    }
                }
                ScalarExpr::Field {
                    tuple: Box::new(self.expression(*tuple)?),
                    index: *index,
                }
            }
            ExprKind::Parameter(p) => {
                if let Some(value) = self.parameters.get(p) {
                    return Some(value.clone());
                }
                let [input] = data.state.abi.inputs.get(p)?.as_slice() else {
                    return None;
                };
                let ty = scalar_type(ty)?;
                let source = match &input.kind {
                    EntryInputKind::PushConstant { slot } => ScalarSource::PushConstant {
                        name: input.name.clone(),
                        offset: slot.offset,
                    },
                    EntryInputKind::Uniform { binding } => source(*binding),
                    _ => return None,
                };
                self.inputs.insert(*p);
                self.reads.insert(
                    *p,
                    ScalarExpr::Read {
                        source,
                        offset: 0,
                        ty,
                    },
                );
                ScalarExpr::Local(format!("host-input-{}", p.as_u32()))
            }
            ExprKind::Coerce(e) => {
                let from = scalar_type(&data.types[data.expressions[*e].ty].ty)?;
                let to = scalar_type(ty)?;
                let value = self.expression(*e)?;
                if from == to {
                    value
                } else {
                    apply(&format!("to-{}", to.name()), from, vec![value])
                }
            }
            ExprKind::PureApp { function, args } => {
                let first = *args.first()?;
                let operand = scalar_type(&data.types[data.expressions[first].ty].ty)?;
                let op = match &data.expressions[*function].kind {
                    ExprKind::BinOp(op) | ExprKind::UnOp(op) => match op.as_str() {
                        "+" => "add",
                        "-" if args.len() == 1 => "neg",
                        "-" => "sub",
                        "*" => "mul",
                        "/" => "div",
                        "%" => "rem",
                        "==" => "eq",
                        "!=" => "ne",
                        "<" => "lt",
                        "<=" => "le",
                        ">" => "gt",
                        ">=" => "ge",
                        "&" | "&&" => "and",
                        "|" | "||" => "or",
                        "^" => "xor",
                        "<<" => "shl",
                        ">>" => "shr",
                        "!" | "~" => "not",
                        _ => return None,
                    }
                    .to_owned(),
                    ExprKind::Builtin(b) => {
                        let builtin = &data.builtins[*b];
                        let definition = by_id(builtin.builtin);
                        if definition.raw.purity != Purity::Pure {
                            return None;
                        }
                        builtin_op(
                            &definition.overloads().get(builtin.overload_idx)?.lowering,
                            scalar_type(ty)?,
                        )?
                        .to_owned()
                    }
                    _ => return None,
                };
                apply(
                    &op,
                    operand,
                    args.iter().map(|e| self.expression(*e)).collect::<Option<_>>()?,
                )
            }
            ExprKind::If {
                condition,
                then_value,
                else_value,
            } => ScalarExpr::If {
                condition: Box::new(self.expression(*condition)?),
                yes: Box::new(self.scope(|s| s.expression(*then_value))?),
                no: Box::new(self.scope(|s| s.expression(*else_value))?),
            },
            ExprKind::OperationResult(op) => {
                if let Some(value) = self.operations.get(op) {
                    value.clone()
                } else if let Some(buffer) = data.state.materialized.get(op).and_then(singleton_buffer) {
                    ScalarExpr::Read {
                        source: source(*data.state.abi.buffer_bindings.get(&buffer)?),
                        offset: 0,
                        ty: scalar_type(ty)?,
                    }
                } else {
                    self.operation(*op)?
                }
            }
            _ => return None,
        })
    }
    fn region(&mut self, region: RegionId) -> Option<ScalarExpr> {
        // Preserve eager evaluation and effect order, including operations
        // whose values are discarded. Unsupported work keeps the region on GPU.
        for &operation in self.schedules.get(&region).into_iter().flatten() {
            self.operation(operation)?;
        }
        let results = &self.data.regions[region].results;
        if results.len() == 1 {
            self.expression(results[0])
        } else {
            Some(ScalarExpr::Tuple(
                results.iter().map(|e| self.expression(*e)).collect::<Option<_>>()?,
            ))
        }
    }
    fn operation(&mut self, id: OperationId) -> Option<ScalarExpr> {
        if let Some(value) = self.operations.get(&id) {
            return Some(value.clone());
        }
        if !self.active.insert(id) {
            return None;
        }
        let parameters = self.parameters.clone();
        let values = std::mem::take(&mut self.values);
        let result = self.scope(|s| s.operation_body(id));
        self.values = values;
        self.parameters = parameters;
        self.active.remove(&id);
        let value = result?;
        let name = format!("host-operation-{}", id.as_u32());
        self.bindings.push((name.clone(), value));
        let value = ScalarExpr::Local(name);
        self.operations.insert(id, value.clone());
        Some(value)
    }
    fn operation_body(&mut self, id: OperationId) -> Option<ScalarExpr> {
        let data = self.data;
        Some(match &data.operations[id].kind {
            OperationKind::If {
                condition,
                then_region,
                else_region,
            } => ScalarExpr::If {
                condition: Box::new(self.expression(*condition)?),
                yes: Box::new(self.scope(|s| s.region(*then_region))?),
                no: Box::new(self.scope(|s| s.region(*else_region))?),
            },
            OperationKind::Loop {
                init,
                header,
                kind,
                body,
            } => {
                let initial = self.expression(*init)?;
                let name = format!("host-loop-{}", id.as_u32());
                let local = ScalarExpr::Local(name.clone());
                let params = &data.regions[*header].parameters;
                self.parameters.insert(*params.first()?, field(local.clone(), 0));
                if let Some(&p) = params.get(1) {
                    self.parameters.insert(p, field(local.clone(), 1));
                }
                let condition = match kind {
                    LoopKind::While => self.scope(|s| s.region(*header))?,
                    LoopKind::ForRange(bound) => apply(
                        "lt",
                        ScalarType::I32,
                        vec![field(local.clone(), 1), self.expression(*bound)?],
                    ),
                    LoopKind::For(_) => return None,
                };
                let step = ScalarExpr::Tuple(vec![
                    self.scope(|s| s.region(*body))?,
                    apply("add", ScalarType::I32, vec![field(local, 1), ScalarExpr::I32(1)]),
                ]);
                field(
                    ScalarExpr::Loop {
                        name,
                        initial: Box::new(ScalarExpr::Tuple(vec![initial, ScalarExpr::I32(0)])),
                        condition: Box::new(condition),
                        step: Box::new(step),
                    },
                    0,
                )
            }
            OperationKind::Call { function, args } => {
                let (region, captures) = match &data.expressions[*function].kind {
                    ExprKind::Global(symbol) => (
                        data.definitions.values().find(|d| d.symbol == *symbol)?.body,
                        vec![],
                    ),
                    ExprKind::Lambda(r) => (*r, vec![]),
                    ExprKind::Closure { code, captures, .. } => (
                        data.definitions.values().find(|d| d.symbol == *code)?.body,
                        captures.clone(),
                    ),
                    _ => return None,
                };
                let values: Vec<_> =
                    args.iter().chain(&captures).map(|e| self.expression(*e)).collect::<Option<_>>()?;
                if values.len() != data.regions[region].parameters.len() {
                    return None;
                }
                let mut bindings = vec![];
                for (i, (&param, value)) in data.regions[region].parameters.iter().zip(values).enumerate() {
                    let name = format!("host-call-{}-{i}", id.as_u32());
                    bindings.push((name.clone(), value));
                    self.parameters.insert(param, ScalarExpr::Local(name));
                }
                ScalarExpr::Let {
                    bindings,
                    result: Box::new(self.scope(|s| s.region(region))?),
                }
            }
            _ => return None,
        })
    }
}
fn field(tuple: ScalarExpr, index: usize) -> ScalarExpr {
    ScalarExpr::Field {
        tuple: Box::new(tuple),
        index,
    }
}
fn apply(op: &str, ty: ScalarType, args: Vec<ScalarExpr>) -> ScalarExpr {
    ScalarExpr::Apply {
        op: op.into(),
        ty,
        args,
    }
}
fn builtin_op(lowering: &BuiltinLowering, result: ScalarType) -> Option<&'static str> {
    Some(match lowering {
        BuiltinLowering::PrimOp(PrimOp::GlslExt(ext)) => match ext {
            1 => "round",
            2 => "round-even",
            3 => "trunc",
            4 | 5 => "abs",
            6 | 7 => "sign",
            8 => "floor",
            9 => "ceil",
            10 => "fract",
            11 => "radians",
            12 => "degrees",
            13 => "sin",
            14 => "cos",
            15 => "tan",
            16 => "asin",
            17 => "acos",
            18 => "atan",
            19 => "sinh",
            20 => "cosh",
            21 => "tanh",
            22 => "asinh",
            23 => "acosh",
            24 => "atanh",
            25 => "atan2",
            26 => "pow",
            27 => "exp",
            28 => "log",
            29 => "exp2",
            30 => "log2",
            31 => "sqrt",
            32 => "rsqrt",
            37..=39 => "min",
            40..=42 => "max",
            _ => return None,
        },
        BuiltinLowering::PrimOp(
            PrimOp::FPToSI
            | PrimOp::FPToUI
            | PrimOp::SIToFP
            | PrimOp::UIToFP
            | PrimOp::SConvert
            | PrimOp::UConvert
            | PrimOp::FPConvert,
        ) => match result {
            ScalarType::I32 => "to-i32",
            ScalarType::U32 => "to-u32",
            ScalarType::F32 => "to-f32",
            ScalarType::Bool => return None,
        },
        BuiltinLowering::PrimOp(PrimOp::IsNan) => "isnan",
        BuiltinLowering::PrimOp(PrimOp::IsInf) => "isinf",
        _ => return None,
    })
}
