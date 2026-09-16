//! Inspection-only SSA handoff. Computations and kernel control flow are real;
//! the shader interface is provisional. See TODOs below before executing output.

use super::blocks::{Exit, FunctionKind, Instruction, Storage, Value};
use super::data::{AssociatedData, BlockId, BufferId, ExprId, OperationId, OperationKind, ParameterId};
use super::OptimizeError;
use crate::ast::Span;
use crate::flow::ControlHeader;
use crate::interface::{StorageBindingDecl, StorageRole};
use crate::op::{OpTag, PureViewSource};
use crate::ssa::{
    self,
    builder::FuncBuilder,
    types::{FuncBody, InstKind, PlaceId, Terminator, ValueRef},
};
use crate::types::{self, Type, TypeExt, TypeName};
use crate::{BindingRef, CodegenTarget, EntryId, FunctionId, LoweringProfile, SchedulePolicy};
use std::collections::{BTreeMap, BTreeSet};

mod control;
mod values;

/// Lower scheduled kernels and their helpers to backend-bound SSA for inspection.
///
/// TODO(egglog-interface): replace synthetic group-0 bindings and placeholder
/// lengths with the actual entry/storage/uniform ABI and allocation sizes.
/// TODO(egglog-host): publish the host CFG, allocations, grids, and repeated or
/// conditional launches. Every kernel is emitted independently for now.
/// No runtime pipeline descriptor is published by this provisional adapter.
pub fn to_ssa(
    data: &AssociatedData,
    target: CodegenTarget,
) -> Result<ssa::stage::Elaborated, OptimizeError> {
    if data.blocks.is_empty() && !data.entries.is_empty() {
        return Err(error("schedule the program before lowering to SSA"));
    }
    let mut compiler = Compiler {
        data,
        functions: vec![],
        specializations: vec![],
        bindings: vec![],
        slots: BTreeMap::new(),
    };
    let mut roots: Vec<_> = data
        .blocks
        .iter()
        .filter_map(|(&id, b)| {
            b.interface.as_ref().and_then(|f| match f.kind {
                FunctionKind::Kernel(_) => Some(id),
                FunctionKind::Entry(_) if !contains_dispatch(data, id) => Some(id),
                _ => None,
            })
        })
        .collect();
    roots.sort();
    let mut entries = vec![];
    for root in roots {
        let Some(interface) = &data.blocks[root].interface else {
            return Err(error("missing entry interface"));
        };
        let size = match interface.kind {
            FunctionKind::Kernel(size) => size,
            _ => [1, 1, 1],
        };
        let mut lower = Body::new(&mut compiler, root, &[], size[0])?;
        // Input captures keep their identities, but use provisional storage
        // slots instead of an authored uniform/buffer interface.
        for instruction in &data.bodies[data.blocks[root].body].instructions {
            let (name, id, ty) = match instruction {
                Instruction::BindExpression(id, Value::Local(name)) => {
                    (name, Some(*id), data.types[data.expressions[*id].ty].ty.clone())
                }
                Instruction::BindParameter(p, Value::Local(name)) => {
                    (name, None, data.types[data.parameters[*p].ty].ty.clone())
                }
                _ => continue,
            };
            let key = id.map_or_else(
                || format!("entry{}_{}", root.as_u32(), name),
                |id| format!("capture{}", id.as_u32()),
            );
            let value = if let Some(id) = id { lower.seed(id)? } else { lower.placeholder(&key, &ty)? };
            lower.environment.locals.insert(name.clone(), value);
        }
        lower.visit(root)?;
        let (mut body, _) = lower.finish()?;
        // Shader return values need an observable sink, even for scalar-only
        // source entries. TODO(egglog-interface): route actual authored outputs.
        compiler.store_entry_results(&mut body, root)?;
        entries.push(ssa::types::EntryPoint {
            id: EntryId::from(root.as_u32()),
            name: format!(
                "egg_{}_{}",
                if matches!(interface.kind, FunctionKind::Kernel(_)) { "kernel" } else { "entry" },
                root.as_u32()
            ),
            body,
            execution_model: crate::flow::ExecutionModel::Compute {
                local_size: (size[0], size[1], size[2]),
            },
            inputs: vec![],
            parameter_inputs: vec![],
            outputs: vec![],
            storage_bindings: vec![],
            stage_descriptor_storage_accesses: Default::default(),
            pipeline_storage_accesses: Default::default(),
            span: Span::generated(),
        });
    }
    // TODO(egglog-interface): each entry should declare only its actual uses;
    // all preview entries currently share the same conservative binding table.
    for entry in &mut entries {
        entry.storage_bindings = compiler.bindings.clone();
    }
    Ok(ssa::Program::bare(compiler.functions, entries, vec![])
        .with_context::<ssa::stage::ElaboratedTag, _>(ssa::context::BackendGlobal {
            pipeline: Default::default(),
            physical_kernels: Default::default(),
            profile: LoweringProfile::new(target, SchedulePolicy::Parallel),
        }))
}

fn error(message: impl Into<String>) -> OptimizeError {
    OptimizeError::Output(format!("egglog to SSA: {}", message.into()))
}
fn builder_error(e: ssa::builder::BuilderError) -> OptimizeError {
    error(e.to_string())
}

fn contains_dispatch(data: &AssociatedData, root: BlockId) -> bool {
    let mut pending = vec![root];
    let mut seen = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        let b = &data.blocks[id];
        for i in &data.bodies[b.body].instructions {
            match i {
                Instruction::Dispatch(_) => return true,
                Instruction::Call { function, .. } => pending.push(*function),
                _ => {}
            }
        }
        match &b.exit {
            Exit::Jump(e) => pending.push(e.target),
            Exit::Branch { yes, no, .. } => pending.extend([yes.target, no.target]),
            Exit::Return(_) => {}
        }
    }
    false
}

struct Specialization {
    source: BlockId,
    parameters: Vec<Type>,
    id: FunctionId,
    result: Option<Vec<Type>>,
}
struct Compiler<'a> {
    data: &'a AssociatedData,
    functions: Vec<ssa::types::Function>,
    specializations: Vec<Specialization>,
    bindings: Vec<StorageBindingDecl>,
    slots: BTreeMap<String, BindingRef>,
}
impl Compiler<'_> {
    fn binding(&mut self, key: String, element: Type) -> Result<BindingRef, OptimizeError> {
        if let Some(&binding) = self.slots.get(&key) {
            return Ok(binding);
        }
        let index = u32::try_from(self.bindings.len()).map_err(|_| error("too many preview bindings"))?;
        let binding = BindingRef::new(0, index);
        self.bindings.push(StorageBindingDecl {
            binding,
            elem_ty: element,
            role: StorageRole::Intermediate,
            logical_resource: Some(key.clone()),
            length: None,
        });
        self.slots.insert(key, binding);
        Ok(binding)
    }
    fn function(
        &mut self,
        source: BlockId,
        parameters: Vec<Type>,
    ) -> Result<(FunctionId, Vec<Type>), OptimizeError> {
        if let Some(s) =
            self.specializations.iter().find(|s| s.source == source && s.parameters == parameters)
        {
            let Some(result) = &s.result else {
                return Err(error("TODO: recursive helper signatures"));
            };
            return Ok((s.id, result.clone()));
        }
        let index = self.specializations.len();
        let id = FunctionId::from(u32::try_from(index).map_err(|_| error("too many helper functions"))?);
        self.specializations.push(Specialization {
            source,
            parameters: parameters.clone(),
            id,
            result: None,
        });
        let mut lower = Body::new(self, source, &parameters, 1)?;
        lower.visit(source)?;
        let (body, result) = lower.finish()?;
        self.specializations[index].result = Some(result.clone());
        self.functions.push(ssa::types::Function {
            id,
            name: format!("egg_helper_{}_{index}", source.as_u32()),
            body,
            span: Span::generated(),
            linkage_name: None,
        });
        Ok((id, result))
    }
    fn store_entry_results(&mut self, body: &mut FuncBody, root: BlockId) -> Result<(), OptimizeError> {
        // Reuse the normal SSA storage operations; this only supplies the
        // provisional output ABI, not a second shader text generator.
        let returns: Vec<_> = body
            .inner
            .blocks
            .iter()
            .filter_map(|(b, data)| match data.term {
                Terminator::Return(Some(v)) => Some((b, v)),
                _ => None,
            })
            .collect();
        for (block, value) in returns {
            let ty = body.return_ty.clone();
            if ty == types::unit() {
                body.inner.blocks[block].term = Terminator::Return(None);
                continue;
            }
            let binding = self.binding(format!("result{}", root.as_u32()), ty.clone())?;
            let view_ty = types::view_array_of(&ty, types::buffer_tag(binding));
            let view = body.inner.append_inst(
                block,
                InstKind::Op {
                    tag: OpTag::StorageView(PureViewSource::Storage(binding)),
                    operands: vec![uint(0), uint(1)],
                },
                view_ty,
            );
            let place = body.places.insert(ssa::types::PlaceInfo {
                elem_ty: ty,
                origin: ssa::types::PlaceOrigin::Instruction,
            });
            body.inner.append_void_inst(
                block,
                InstKind::ViewIndex {
                    view: view.into(),
                    index: uint(0),
                    result: place,
                },
            );
            body.inner.append_void_inst(block, InstKind::Store { place, value });
            body.inner.blocks[block].term = Terminator::Return(None);
        }
        body.return_ty = types::unit();
        Ok(())
    }
}

#[derive(Clone)]
struct Typed {
    value: ValueRef,
    ty: Type,
}
#[derive(Clone, Default)]
struct Environment {
    locals: BTreeMap<String, Typed>,
    parameters: BTreeMap<ParameterId, Typed>,
    expressions: BTreeMap<ExprId, Typed>,
    operations: BTreeMap<OperationId, Typed>,
    buffers: BTreeMap<BufferId, (PlaceId, Type)>,
}
struct Body<'a, 'b> {
    compiler: &'a mut Compiler<'b>,
    builder: FuncBuilder,
    environment: Environment,
    blocks: BTreeMap<BlockId, crate::flow::BlockId>,
    entered: BTreeSet<BlockId>,
    active: BTreeSet<BlockId>,
    continuing: BTreeMap<BlockId, crate::flow::BlockId>,
    parameter_types: BTreeMap<BlockId, Vec<Type>>,
    return_types: Option<Vec<Type>>,
    width: u32,
}
impl<'a, 'b> Body<'a, 'b> {
    fn new(
        compiler: &'a mut Compiler<'b>,
        entry: BlockId,
        types: &[Type],
        width: u32,
    ) -> Result<Self, OptimizeError> {
        let names = &compiler.data.blocks[entry].parameters;
        if !types.is_empty() && names.len() != types.len() {
            return Err(error("function parameter arity"));
        }
        let mut builder = FuncBuilder::new(
            types.iter().cloned().zip(names.iter().cloned()).collect(),
            types::unit(),
        );
        let mut environment = Environment::default();
        for (i, (name, ty)) in names.iter().zip(types).enumerate() {
            environment.locals.insert(
                name.clone(),
                Typed {
                    value: builder.get_param(i).into(),
                    ty: ty.clone(),
                },
            );
        }
        let mut blocks = BTreeMap::from([(entry, builder.entry())]);
        for (&id, b) in &compiler.data.blocks {
            if b.function == entry && id != entry {
                blocks.insert(id, builder.create_block());
            }
        }
        Ok(Self {
            compiler,
            builder,
            environment,
            blocks,
            entered: BTreeSet::new(),
            active: BTreeSet::new(),
            continuing: BTreeMap::new(),
            parameter_types: BTreeMap::new(),
            return_types: None,
            width,
        })
    }
    fn finish(mut self) -> Result<(FuncBody, Vec<Type>), OptimizeError> {
        for (&source, &cont) in &self.continuing {
            self.builder.switch_to_block_unchecked(cont);
            let args =
                self.builder.func().blocks[cont].params.iter().copied().map(ValueRef::from).collect();
            self.builder
                .terminate(Terminator::Branch {
                    target: self.blocks[&source],
                    args,
                })
                .map_err(builder_error)?;
            let Some(exit) = self.compiler.data.blocks[source].loop_exit else {
                return Err(error("loop exit missing"));
            };
            self.builder.set_control_header(
                self.blocks[&source],
                ControlHeader::Loop {
                    merge: self.blocks[&exit],
                    continue_block: cont,
                },
            );
        }
        let types = self.return_types.unwrap_or_default();
        let mut body = self.builder.finish().map_err(builder_error)?;
        body.return_ty = result_type(&types);
        control::annotate_selections(&mut body)?;
        Ok((body, types))
    }
    fn visit(&mut self, id: BlockId) -> Result<(), OptimizeError> {
        if !self.entered.insert(id) {
            return Ok(());
        }
        self.active.insert(id);
        self.builder.switch_to_block_unchecked(self.blocks[&id]);
        let data = self.compiler.data;
        for (name, &value) in
            data.blocks[id].parameters.iter().zip(&self.builder.func().blocks[self.blocks[&id]].params)
        {
            self.environment.locals.insert(
                name.clone(),
                Typed {
                    value: value.into(),
                    ty: self.builder.func().value_type(value).clone(),
                },
            );
        }
        for instruction in &data.bodies[data.blocks[id].body].instructions {
            self.instruction(instruction)?;
        }
        match &data.blocks[id].exit {
            Exit::Return(body) => {
                let values = self.values(&data.bodies[*body].results)?;
                let types: Vec<_> = values.iter().map(|v| v.ty.clone()).collect();
                if self.return_types.as_ref().is_some_and(|known| *known != types) {
                    return Err(error("inconsistent function return types"));
                }
                self.return_types = Some(types);
                let value = if values.is_empty() { None } else { Some(self.pack(values)?.value) };
                self.builder.terminate(Terminator::Return(value)).map_err(builder_error)?;
            }
            Exit::Jump(edge) => {
                let values = self.values(&data.bodies[edge.arguments].results)?;
                let (target, args) = self.edge(edge.target, values)?;
                self.builder.terminate(Terminator::Branch { target, args }).map_err(builder_error)?;
                self.visit(edge.target)?;
            }
            Exit::Branch { condition, yes, no } => {
                let values = self.values(&data.bodies[*condition].results)?;
                let [condition] = values.as_slice() else {
                    return Err(error("branch condition arity"));
                };
                let y = self.values(&data.bodies[yes.arguments].results)?;
                let n = self.values(&data.bodies[no.arguments].results)?;
                // The current backend requires a single conditional header.
                // TODO(egglog-control): outline branching loop tests into helpers.
                if data.blocks[id].loop_exit.is_some()
                    && self.builder.current_block() != Some(self.blocks[&id])
                {
                    return Err(error("TODO: branching expression in loop header"));
                }
                let (then_target, then_args) = self.edge(yes.target, y)?;
                let (else_target, else_args) = self.edge(no.target, n)?;
                self.builder
                    .terminate(Terminator::CondBranch {
                        cond: condition.value,
                        then_target,
                        then_args,
                        else_target,
                        else_args,
                    })
                    .map_err(builder_error)?;
                let environment = self.environment.clone();
                self.visit(yes.target)?;
                self.environment = environment;
                self.visit(no.target)?;
            }
        }
        self.active.remove(&id);
        Ok(())
    }
    fn edge(
        &mut self,
        target: BlockId,
        values: Vec<Typed>,
    ) -> Result<(crate::flow::BlockId, Vec<ValueRef>), OptimizeError> {
        if self.compiler.data.blocks[target].parameters.len() != values.len() {
            return Err(error("block argument arity"));
        }
        let types: Vec<_> = values.iter().map(|v| v.ty.clone()).collect();
        if let Some(known) = self.parameter_types.get(&target) {
            if *known != types {
                return Err(error(format!(
                    "block {target:?} argument types differ: {known:?} vs {types:?}"
                )));
            }
        } else {
            for ty in &types {
                self.builder.add_block_param(self.blocks[&target], ty.clone());
            }
            self.parameter_types.insert(target, types.clone());
        }
        let block = if self.active.contains(&target) {
            if self.compiler.data.blocks[target].loop_exit.is_none() {
                return Err(error("cycle without loop metadata"));
            }
            if let Some(&b) = self.continuing.get(&target) {
                b
            } else {
                let (b, _) = self.builder.create_block_with_params(types);
                self.continuing.insert(target, b);
                b
            }
        } else {
            self.blocks[&target]
        };
        Ok((block, values.into_iter().map(|v| v.value).collect()))
    }
    fn instruction(&mut self, instruction: &Instruction) -> Result<(), OptimizeError> {
        match instruction {
            Instruction::BindParameter(id, value) => {
                let v = self.value(value)?;
                let ty = &self.compiler.data.types[self.compiler.data.parameters[*id].ty].ty;
                let v = self.cast(v, ty)?;
                self.environment.parameters.insert(*id, v);
            }
            Instruction::BindExpression(id, value) => {
                let v = self.value(value)?;
                self.environment.expressions.insert(*id, v);
            }
            Instruction::BindResult(id, value) => {
                let v = self.value(value)?;
                self.environment.operations.insert(*id, v);
            }
            Instruction::Call {
                function,
                arguments,
                results,
            } => {
                let args = self.values(arguments)?;
                let (id, types) =
                    self.compiler.function(*function, args.iter().map(|v| v.ty.clone()).collect())?;
                if results.len() != types.len() {
                    return Err(error("call result arity"));
                }
                let value = self.op(OpTag::Call(id), args, result_type(&types))?;
                if results.len() == 1 {
                    self.environment.locals.insert(results[0].clone(), value);
                } else {
                    for (i, name) in results.iter().enumerate() {
                        let field = self.field(value.clone(), i)?;
                        self.environment.locals.insert(name.clone(), field);
                    }
                }
            }
            Instruction::Evaluate(op) => {
                let data = self.compiler.data;
                let value = match &data.operations[*op].kind {
                    OperationKind::Index { array, index } => {
                        let a = self.expression(*array)?;
                        let i = self.expression(*index)?;
                        self.index(a, i)?
                    }
                    OperationKind::Call { function, args } => {
                        let args =
                            args.iter().map(|&id| self.expression(id)).collect::<Result<Vec<_>, _>>()?;
                        self.apply(*function, args, data.types[data.operations[*op].ty].ty.clone())?
                    }
                    other => return Err(error(format!("TODO: lower source execution {other:?}"))),
                };
                self.environment.operations.insert(*op, value);
            }
            Instruction::Load {
                result,
                buffer,
                index,
            } => {
                let a = self.value(buffer)?;
                let i = self.value(index)?;
                let value = self.index(a, i)?;
                self.environment.locals.insert(result.clone(), value);
            }
            Instruction::Store { buffer, index, value } => {
                let i = self.value(index)?;
                let v = self.value(value)?;
                let (place, ty) = if let Value::Buffer(id) = buffer {
                    if let Some((p, ty)) = self.environment.buffers.get(id).cloned() {
                        let Some(elem) = ty.elem_type() else {
                            return Err(error("local buffer element type"));
                        };
                        let ty = elem.clone();
                        let result = self.builder.new_place(ty.clone());
                        self.builder
                            .push_void_inst(InstKind::PlaceIndex {
                                place: p,
                                index: i.value,
                                result,
                            })
                            .map_err(builder_error)?;
                        (result, ty)
                    } else {
                        let a = self.value(buffer)?;
                        self.index_place(a, i)?
                    }
                } else {
                    let a = self.value(buffer)?;
                    self.index_place(a, i)?
                };
                let v = self.cast(v, &ty)?;
                self.builder
                    .push_void_inst(InstKind::Store {
                        place,
                        value: v.value,
                    })
                    .map_err(builder_error)?;
            }
            Instruction::Allocate(id) => {
                let buffer = &self.compiler.data.buffers[*id];
                if buffer.storage == Storage::Function {
                    // TODO(egglog-interface): dynamic invocation-local allocation.
                    let count = if let Value::Int(n) = buffer.length { n.max(1) as usize } else { 64 };
                    let ty = types::sized_array(count, concrete(&buffer.element)?);
                    let place = self.builder.new_place(ty.clone());
                    self.builder
                        .push_void_inst(InstKind::Alloca {
                            elem_ty: ty.clone(),
                            result: place,
                        })
                        .map_err(builder_error)?;
                    self.environment.buffers.insert(*id, (place, ty));
                }
            }
            Instruction::Dispatch(_) => return Err(error("host dispatches do not belong in shader SSA")),
        }
        Ok(())
    }
    fn op(
        &mut self,
        tag: OpTag<BindingRef, FunctionId>,
        args: Vec<Typed>,
        ty: Type,
    ) -> Result<Typed, OptimizeError> {
        let value = self
            .builder
            .push_inst(
                InstKind::Op {
                    tag,
                    operands: args.into_iter().map(|v| v.value).collect(),
                },
                ty.clone(),
            )
            .map_err(builder_error)?;
        Ok(Typed {
            value: value.into(),
            ty,
        })
    }
    fn pack(&mut self, values: Vec<Typed>) -> Result<Typed, OptimizeError> {
        if values.len() == 1 {
            return Ok(values[0].clone());
        }
        self.tuple(values)
    }
    fn tuple(&mut self, values: Vec<Typed>) -> Result<Typed, OptimizeError> {
        let ty = types::tuple(values.iter().map(|v| v.ty.clone()).collect());
        self.op(OpTag::Tuple(values.len()), values, ty)
    }
    fn values(&mut self, values: &[Value]) -> Result<Vec<Typed>, OptimizeError> {
        values.iter().map(|v| self.value(v)).collect()
    }
}
fn result_type(types: &[Type]) -> Type {
    match types {
        [] => types::unit(),
        [ty] => ty.clone(),
        _ => types::tuple(types.to_vec()),
    }
}
fn uint(n: u32) -> ValueRef {
    ValueRef::Const(ssa::types::ConstantValue::U32(n))
}
fn u32_type() -> Type {
    Type::Constructed(TypeName::UInt(32), vec![])
}
fn concrete(ty: &Type) -> Result<Type, OptimizeError> {
    if let Some(element) = ty.elem_type().filter(|_| ty.is_array()) {
        let count = match ty.array_size() {
            Some(Type::Constructed(TypeName::Size(n), _)) => *n as usize,
            _ => 64,
        };
        return Ok(types::sized_array(count.max(1), concrete(element)?));
    }
    match ty {
        Type::Constructed(name, args) => Ok(Type::Constructed(
            name.clone(),
            args.iter().map(concrete).collect::<Result<_, _>>()?,
        )),
        Type::Variable(_) => Err(error("unresolved scalar type")),
    }
}

#[cfg(test)]
#[path = "to_ssa_tests.rs"]
mod tests;
