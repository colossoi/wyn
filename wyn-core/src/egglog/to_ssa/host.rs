//! Publish sequential host expressions using imported lexical identities.
use super::{plan::Output, Compiler};
use crate::builtins::catalog;
use crate::builtins::lowering::PrimOp;
use crate::builtins::{by_id, BuiltinLowering, Purity};
use crate::egglog::bindings::Bindings;
use crate::egglog::source::Term;
use crate::egglog::to_ssa::interface;
use crate::host::BufferLen;
use crate::host::{ModuleInterface, ScalarExpr, ScalarSource, ScalarTask, ScalarType};
use crate::interface::StorageBindingDecl;
use crate::interface::StorageRole;
use crate::interface::{EntryInputKind, StorageLayout};
use crate::op::OpTag;
use crate::op::PureViewSource;
use crate::ssa::layout::block_layout;
use crate::ssa::types::EntryPoint;
use crate::tlc::{LoopKind, TermKind, VarRef};
use crate::types::buffer_tag;
use crate::types::{Type, TypeName};
use crate::BindingRef;
use crate::LookupMap;
use egglog_engine::Value;

pub(super) fn publish(compiler: &Compiler<'_, '_>, entries: &[EntryPoint], module: &mut ModuleInterface) {
    module.scalar_tasks.extend(compiler.host_tasks.iter().cloned());
    for entry in entries {
        let Some((owner, stage)) = compiler.entry_origins.get(&entry.id) else {
            continue;
        };
        if let Some(stage) = stage {
            if stage.phase != "scalar" {
                continue;
            }
            let operations = compiler.plan.scalar_group(stage.operation);
            let mut targets = Vec::new();
            for operation in operations {
                if let (Some(source), Some(resource)) = (
                    compiler.plan.source(operation),
                    compiler.plan.slot(operation, "scalar", 0),
                ) {
                    let backing = compiler.plan.backing(resource).unwrap_or(resource);
                    if let Some(buffer) = compiler.plan.buffers.get(&backing) {
                        targets.push((source, buffer));
                    }
                }
            }
            for output in compiler
                .plan
                .outputs
                .iter()
                .filter(|output| output.copy && output.writer == Some(stage.key))
            {
                if let Some(buffer) = compiler.plan.buffers.get(&output.resource) {
                    targets.push((output.source, buffer));
                }
            }
            let tasks: Option<Vec<_>> = targets
                .into_iter()
                .map(|(source, buffer)| {
                    let ty = scalar_type(compiler.facts.source_type(source)?)?;
                    let value = expression(compiler, source, &LookupMap::default())?;
                    Some(ScalarTask {
                        stage: entry.name.clone(),
                        destination: ScalarSource::Binding {
                            set: buffer.binding.set,
                            binding: buffer.binding.binding,
                        },
                        offset: 0,
                        ty,
                        value,
                        replaces_dispatch: true,
                    })
                })
                .collect();
            if let Some(tasks) = tasks {
                module.scalar_tasks.extend(tasks);
            }
            continue;
        }
        if compiler.facts.definition(*owner).is_some_and(|scope| compiler.facts.device_region(scope)) {
            continue;
        }
        let outputs: Vec<&Output> = compiler
            .plan
            .outputs
            .iter()
            .filter(|output| output.owner == *owner && output.copy && output.writer.is_none())
            .collect();
        if outputs.is_empty() {
            continue;
        }
        let mut lower = Lower {
            compiler,
            values: Bindings::default(),
            bindings: vec![],
            next: 0,
        };
        let tasks: Option<Vec<_>> = outputs
            .into_iter()
            .map(|output| {
                let ty = scalar_type(&output.ty)?;
                let buffer = compiler.plan.buffers.get(&output.resource)?;
                let value = lower.expression(output.source)?;
                Some(ScalarTask {
                    stage: entry.name.clone(),
                    destination: ScalarSource::Binding {
                        set: buffer.binding.set,
                        binding: buffer.binding.binding,
                    },
                    offset: 0,
                    ty,
                    value: lower.finish(value),
                    replaces_dispatch: true,
                })
            })
            .collect();
        if let Some(tasks) = tasks {
            module.scalar_tasks.extend(tasks);
        }
    }
}

struct Lower<'a, 'p, 'source> {
    compiler: &'a Compiler<'p, 'source>,
    values: Bindings<Value, ScalarExpr>,
    bindings: Vec<(String, ScalarExpr)>,
    next: usize,
}
impl<'source> Lower<'_, '_, 'source> {
    fn finish(&mut self, value: ScalarExpr) -> ScalarExpr {
        ScalarExpr::Let {
            bindings: std::mem::take(&mut self.bindings),
            result: Box::new(value),
        }
    }
    fn scope(&mut self, emit: impl FnOnce(&mut Self) -> Option<ScalarExpr>) -> Option<ScalarExpr> {
        let outer = std::mem::take(&mut self.bindings);
        let values = self.values.checkpoint();
        let result = emit(self).map(|value| self.finish(value));
        self.bindings = outer;
        self.values.restore(values);
        result
    }
    fn source(&mut self, scope: Value, term: &'source Term) -> Option<ScalarExpr> {
        let &value = self.compiler.program.identities.occurrences.get(&(scope, term.id))?;
        self.expression(value)
    }
    fn expression(&mut self, source: Value) -> Option<ScalarExpr> {
        if let Some(value) = self.values.get(&source) {
            return Some(value.clone());
        }
        let value = self.body(source)?;
        let name = format!("value{}", self.next);
        self.next += 1;
        self.bindings.push((name.clone(), value));
        let local = ScalarExpr::Local(name);
        self.values.insert(source, local.clone());
        Some(local)
    }
    fn body(&mut self, source: Value) -> Option<ScalarExpr> {
        if let Some(actual) = self.compiler.facts.alias(source) {
            return self.expression(actual);
        }
        if let Some(input) = self.compiler.input_interfaces.get(&source) {
            let binding = match &input.kind {
                EntryInputKind::PushConstant { slot } => ScalarSource::PushConstant {
                    name: input.name.clone(),
                    offset: slot.offset,
                },
                EntryInputKind::Uniform { binding } => ScalarSource::Binding {
                    set: binding.set,
                    binding: binding.binding,
                },
                _ => return None,
            };
            let ty = self.compiler.facts.source_type(source)?;
            return read(binding, 0, ty);
        }
        if let Some((parent, index)) = self.compiler.facts.projection(source) {
            return Some(field(self.expression(parent)?, index));
        }
        if let Some(symbol) = self.compiler.facts.global_symbol(source) {
            let scope = self.compiler.facts.definition(symbol)?;
            return self.call(scope, vec![]);
        }
        let &(term, scope) = self.compiler.program.identities.origins.get(&source)?;
        match &term.kind {
            TermKind::IntLit(text) => match scalar_type(&term.ty)? {
                ScalarType::I32 => Some(ScalarExpr::I32(text.parse().ok()?)),
                ScalarType::U32 => Some(ScalarExpr::U32(text.parse().ok()?)),
                _ => None,
            },
            TermKind::FloatLit(value) => Some(ScalarExpr::F32(value.to_bits())),
            TermKind::BoolLit(value) => Some(ScalarExpr::Bool(*value)),
            TermKind::UnitLit => Some(ScalarExpr::Tuple(vec![])),
            TermKind::Tuple(fields) => Some(ScalarExpr::Tuple(
                fields.iter().map(|term| self.source(scope, term)).collect::<Option<_>>()?,
            )),
            TermKind::Coerce { inner, .. } => {
                let from = scalar_type(&inner.ty)?;
                let to = scalar_type(&term.ty)?;
                let value = self.source(scope, inner)?;
                Some(if from == to {
                    value
                } else {
                    apply(&format!("to-{}", to.name()), from, vec![value])
                })
            }
            TermKind::App { func, args } => {
                if matches!(func.kind, TermKind::Var(VarRef::Builtin { id, .. }) if id == catalog().known().length)
                {
                    let array = args.first()?;
                    let &source = self.compiler.program.identities.occurrences.get(&(scope, array.id))?;
                    return array_length(self.compiler, source);
                }
                let values =
                    args.iter().map(|term| self.source(scope, term)).collect::<Option<Vec<_>>>()?;
                let op = match &func.kind {
                    TermKind::BinOp(op) => Some(operator(op.op.symbol(), args.len())?.to_owned()),
                    TermKind::UnOp(op) => Some(operator(op.op.symbol(), args.len())?.to_owned()),
                    TermKind::Var(VarRef::Builtin { id, overload_idx }) => {
                        let builtin = by_id(*id);
                        if builtin.raw.purity != Purity::Pure {
                            return None;
                        }
                        Some(
                            builtin_op(
                                &builtin.overloads().get(*overload_idx)?.lowering,
                                scalar_type(&term.ty)?,
                            )?
                            .to_owned(),
                        )
                    }
                    _ => None,
                };
                if let Some(op) = op {
                    Some(apply(&op, scalar_type(&args.first()?.ty)?, values))
                } else {
                    let function = *self.compiler.program.identities.occurrences.get(&(scope, func.id))?;
                    let callee = self.compiler.facts.callable(function)?;
                    self.call(callee, values)
                }
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let (yes, no) = self.compiler.facts.branches(source)?;
                Some(ScalarExpr::If {
                    condition: Box::new(self.source(scope, cond)?),
                    yes: Box::new(self.scope(|lower| lower.source(yes, then_branch))?),
                    no: Box::new(self.scope(|lower| lower.source(no, else_branch))?),
                })
            }
            TermKind::Loop { init, kind, body, .. } => {
                let (header, iteration) = self.compiler.facts.loops(source)?;
                let accumulator = self.compiler.facts.loop_state(header)?;
                let initial = self.source(scope, init)?;
                let name = format!("loop{}", self.next);
                self.next += 1;
                let state = ScalarExpr::Local(name.clone());
                match kind {
                    LoopKind::ForRange { bound, .. } => {
                        let bound = self.source(scope, bound)?;
                        let index = self.compiler.facts.iteration(iteration)?;
                        let condition = apply("lt", ScalarType::I32, vec![field(state.clone(), 1), bound]);
                        let step = self.scope(|lower| {
                            lower.values.insert(accumulator, field(state.clone(), 0));
                            lower.values.insert(index, field(state.clone(), 1));
                            let value = lower.source(iteration, body)?;
                            Some(ScalarExpr::Tuple(vec![
                                value,
                                apply(
                                    "add",
                                    ScalarType::I32,
                                    vec![field(state.clone(), 1), ScalarExpr::I32(1)],
                                ),
                            ]))
                        })?;
                        Some(field(
                            ScalarExpr::Loop {
                                name,
                                initial: Box::new(ScalarExpr::Tuple(vec![initial, ScalarExpr::I32(0)])),
                                condition: Box::new(condition),
                                step: Box::new(step),
                            },
                            0,
                        ))
                    }
                    LoopKind::While { cond } => {
                        let condition = self.scope(|lower| {
                            lower.values.insert(accumulator, state.clone());
                            lower.source(header, cond)
                        })?;
                        let step = self.scope(|lower| {
                            lower.values.insert(accumulator, state.clone());
                            lower.source(iteration, body)
                        })?;
                        Some(ScalarExpr::Loop {
                            name,
                            initial: Box::new(initial),
                            condition: Box::new(condition),
                            step: Box::new(step),
                        })
                    }
                    LoopKind::For { .. } => None,
                }
            }
            _ => None,
        }
    }
    fn call(&mut self, scope: Value, arguments: Vec<ScalarExpr>) -> Option<ScalarExpr> {
        let result = self.compiler.facts.result(scope)?;
        self.scope(|lower| {
            for (i, value) in arguments.into_iter().enumerate() {
                lower.values.insert(lower.compiler.facts.parameter(scope, i as i64)?, value);
            }
            lower.expression(result)
        })
    }
}
fn scalar_type(ty: &Type) -> Option<ScalarType> {
    match ty {
        Type::Constructed(TypeName::Int(32), _) => Some(ScalarType::I32),
        Type::Constructed(TypeName::UInt(32), _) => Some(ScalarType::U32),
        Type::Constructed(TypeName::Float(32), _) => Some(ScalarType::F32),
        Type::Constructed(TypeName::Bool, _) => Some(ScalarType::Bool),
        _ => None,
    }
}
fn read(source: ScalarSource, offset: u32, ty: &Type) -> Option<ScalarExpr> {
    if let Some(ty) = scalar_type(ty) {
        return Some(ScalarExpr::Read { source, offset, ty });
    }
    if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = ty {
        let layout = block_layout(ty, StorageLayout::Std140)?;
        return Some(ScalarExpr::Tuple(
            fields
                .iter()
                .enumerate()
                .map(|(i, ty)| read(source.clone(), offset + layout.member_offsets[i], ty))
                .collect::<Option<_>>()?,
        ));
    }
    None
}
fn operator(op: &str, arity: usize) -> Option<&'static str> {
    Some(match op {
        "+" => "add",
        "-" if arity == 1 => "neg",
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
    })
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
        BuiltinLowering::PrimOp(PrimOp::Select) => "select",
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

/// Host expressions are a final output language. Admission here uses the
/// selected scalar DAG and egglog's safety proof, within one kernel invocation.
pub(super) fn capture_scalar(
    body: &mut super::Body<'_, '_, '_>,
    term: egglog_engine::TermId,
) -> Result<Option<super::Typed>, super::OptimizeError> {
    let Some(stage) = body.host_stage.clone() else {
        return Ok(None);
    };
    let selected = &body.compiler.program.stage.selected;
    if !body.compiler.facts.total(selected.values[term]) {
        return Ok(None);
    }
    let egglog_engine::Term::App(name, fields) = selected.dag.get(term) else {
        return Ok(None);
    };
    if !matches!(
        name.as_str(),
        "ScalarUnary" | "ScalarBinary" | "ScalarOp" | "ScalarCoerce"
    ) {
        return Ok(None);
    }
    let Some(ty) = body.compiler.facts.ty(selected.values[fields[1]]).cloned() else {
        return Ok(None);
    };
    let Some(scalar) = scalar_type(&ty) else {
        return Ok(None);
    };
    let binding = if let Some(&binding) = body.host_scalar_bindings.get(&term) {
        binding
    } else {
        let mut lower = Lower {
            compiler: body.compiler,
            values: body.host_arguments.clone().into(),
            bindings: vec![],
            next: 0,
        };
        let Some(value) = lower.selected(term) else {
            return Ok(None);
        };
        let value = lower.finish(value);
        let binding = BindingRef::new(0, body.compiler.plan.next_binding);
        body.compiler.plan.next_binding += 1;
        let element = interface::storage_type(&ty)?;
        body.capture_bindings.push(StorageBindingDecl {
            binding,
            role: StorageRole::Input,
            logical_resource: Some(format!("{stage}_capture_{}", binding.binding)),
            elem_ty: element,
            length: Some(BufferLen::Fixed { bytes: 4 }),
        });
        body.compiler.host_tasks.push(ScalarTask {
            stage,
            destination: ScalarSource::Binding {
                set: binding.set,
                binding: binding.binding,
            },
            offset: 0,
            ty: scalar,
            value,
            replaces_dispatch: false,
        });
        body.host_scalar_bindings.insert(term, binding);
        binding
    };
    let zero = body.literal("0", &crate::types::i32())?;
    let one = body.literal("1", &crate::types::i32())?;
    let element = interface::storage_type(&ty)?;
    let view = body.op(
        OpTag::StorageView(PureViewSource::Storage(binding)),
        vec![zero.clone(), one],
        interface::view_type(&element, buffer_tag(binding)),
    )?;
    let loaded = body.index(view, zero)?;
    Ok(Some(body.cast(loaded, &ty)?))
}
pub(super) fn expression(
    compiler: &Compiler<'_, '_>,
    source: Value,
    arguments: &LookupMap<Value, ScalarExpr>,
) -> Option<ScalarExpr> {
    let mut lower = Lower {
        compiler,
        values: Bindings::default(),
        bindings: vec![],
        next: 0,
    };
    for (&source, value) in arguments {
        lower.values.insert(source, value.clone());
    }
    let value = lower.expression(source)?;
    Some(lower.finish(value))
}
impl Lower<'_, '_, '_> {
    fn selected(&mut self, term: egglog_engine::TermId) -> Option<ScalarExpr> {
        let program = self.compiler.program;
        let selected = &program.stage.selected;
        let (name, f) = selected.app(term).ok()?;
        let ty = self.compiler.facts.ty(selected.values[f[1]])?;
        match name {
            "ScalarLeaf" => self.expression(selected.values[f[2]]),
            "ScalarParameter" => self.expression(
                self.compiler.facts.parameter(selected.values[f[2]], selected.integer(f[3]).ok()?)?,
            ),
            "ScalarLiteral" => Some(match scalar_type(ty)? {
                ScalarType::I32 => ScalarExpr::I32(selected.text(f[2]).ok()?.parse().ok()?),
                ScalarType::U32 => ScalarExpr::U32(selected.text(f[2]).ok()?.parse().ok()?),
                ScalarType::F32 => ScalarExpr::F32(selected.text(f[2]).ok()?.parse().ok()?),
                ScalarType::Bool => ScalarExpr::Bool(selected.text(f[2]).ok()? == "true"),
            }),
            "ScalarUnary" | "ScalarBinary" | "ScalarOp" => {
                let terms = selected.operation_arguments(term).ok()?;
                let (_, first) = selected.app(*terms.first()?).ok()?;
                let argument_type = scalar_type(self.compiler.facts.ty(selected.values[first[1]])?)?;
                let op = match selected.operator(f[2], terms.len()).ok()? {
                    OpTag::Intrinsic { id, overload_idx } => builtin_op(
                        &catalog().get(id).overloads().get(overload_idx)?.lowering,
                        scalar_type(ty)?,
                    )?,
                    OpTag::UnaryOp(op) => operator(op.symbol(), 1)?,
                    OpTag::BinOp(op) => operator(op.symbol(), 2)?,
                    _ => return None,
                };
                let args = terms.into_iter().map(|term| self.selected(term)).collect::<Option<Vec<_>>>()?;
                Some(apply(op, argument_type, args))
            }
            "ScalarCoerce" => {
                let (_, operand) = selected.app(f[2]).ok()?;
                let from = scalar_type(self.compiler.facts.ty(selected.values[operand[1]])?)?;
                Some(apply(
                    &format!("to-{}", scalar_type(ty)?.name()),
                    from,
                    vec![self.selected(f[2])?],
                ))
            }
            "ScalarProject" => Some(field(self.selected(f[2])?, selected.integer(f[3]).ok()? as usize)),
            _ => None,
        }
    }
}

/// Evaluate an array length without reading GPU contents.
pub(super) fn array_length(compiler: &Compiler<'_, '_>, source: Value) -> Option<ScalarExpr> {
    match super::sizes::source_size(compiler, source, true).ok()? {
        crate::host::SizeExpr::Integer(n) => Some(ScalarExpr::I32(i32::try_from(n).ok()?)),
        crate::host::SizeExpr::BufferLength { set, binding, stride } => Some(ScalarExpr::BufferLength {
            source: ScalarSource::Binding { set, binding },
            stride,
        }),
        _ => None,
    }
}
