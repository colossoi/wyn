//! SSA handoff from scheduled blocks and the derived resource plan.

use super::blocks::{Exit, FunctionKind, Instruction, Storage, Value};
use super::data::{AssociatedData, BlockId, BufferId, ExprId, OperationId, OperationKind, ParameterId};
use super::OptimizeError;
use crate::ast::Span;
use crate::flow::ControlHeader;
use crate::interface::lowering::build_entry_outputs;
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
mod interface;
mod publish;
mod values;

/// Lower scheduled kernels with their authored inputs and planned storage.
/// Static compute pipelines publish the same resources and dispatch order.
/// TODO: the runtime descriptor cannot yet execute host branches/repeated launches.
pub fn to_ssa(
    data: &AssociatedData,
    target: CodegenTarget,
) -> Result<ssa::stage::Elaborated, OptimizeError> {
    let _timing = super::timing::span("to SSA");
    if data.blocks.is_empty() && !data.entries.is_empty() {
        return Err(error("schedule the program before lowering to SSA"));
    }
    let results = data
        .blocks
        .values()
        .filter(|b| {
            matches!(
                data.blocks[b.function].interface.as_ref().map(|f| &f.kind),
                Some(FunctionKind::Host | FunctionKind::Entry(_))
            )
        })
        .flat_map(|b| &data.bodies[b.body].instructions)
        .filter_map(|i| match i {
            Instruction::BindResult(op, value) if materialized(value) => Some((*op, value.clone())),
            _ => None,
        })
        .collect();
    let mut origins = BTreeMap::new();
    for origin in data.origins.values().filter(|origin| origin.span.module().is_some()) {
        origins.entry(origin.expression).or_insert(origin.span);
    }
    let mut compiler = Compiler {
        origins,
        placements: super::scalar::placement_index(data),
        data,
        functions: vec![],
        specializations: vec![],
        inputs: interface::inputs(data)?,
        bindings: BTreeMap::new(),
        results,
        used: BTreeSet::new(),
    };
    super::timing::time("assign physical bindings", || compiler.allocate_bindings())?;
    let owners: BTreeMap<_, _> = data.dispatches.values().map(|d| (d.kernel, d.owner)).collect();
    let mut roots = vec![];
    for (&id, block) in &data.blocks {
        let Some(f) = &block.interface else {
            continue;
        };
        match f.kind {
            FunctionKind::Kernel(size) => roots.push((id, owners[&id], size, false)),
            FunctionKind::Entry(entry) => {
                if contains_dispatch(data, id) && !static_host(data, id) {
                    return Err(error(format!(
                        "TODO: runtime publication of conditional or repeated host dispatches in {} ({id:?})",
                        data.entries[entry].declaration.name
                    )));
                }
                if data.outputs.values().any(|o| o.entry == entry && o.scalar)
                    || !contains_dispatch(data, id)
                {
                    roots.push((id, entry, [1, 1, 1], true));
                }
            }
            _ => {}
        }
    }
    roots.sort_by_key(|r| r.0);
    let mut entries = vec![];
    for &(root, owner, size, finish) in &roots {
        let _entry = super::timing::span("lower entry or kernel");
        compiler.used.clear();
        let mut lower = Body::new(&mut compiler, root, &[], size[0])?;
        lower.entry = Some(owner);
        let declaration = &data.entries[owner].declaration;
        let compute = !finish || declaration.entry_kind == crate::interface::EntryKind::Compute;
        lower.finish_outputs = finish && compute;
        lower.graphics_outputs = !compute;
        // Root parameters are source ABI values. Kernel captures are resolved
        // lazily, so a combine phase does not declare unused input arrays.
        if finish {
            for instruction in &data.bodies[data.blocks[root].body].instructions {
                if let Instruction::BindParameter(p, Value::Local(name)) = instruction {
                    let value = lower.input(*p)?;
                    lower.environment.locals.insert(name.clone(), value);
                }
            }
        }
        lower.visit(root).map_err(|e| error(format!("{} ({root:?}): {e}", declaration.name)))?;
        let inputs = lower.inputs.clone();
        let (body, return_types) = lower.finish()?;
        let outputs = if compute {
            vec![]
        } else {
            build_entry_outputs(
                declaration,
                &result_type(&return_types),
                &[],
                &inputs,
                false,
                &mut wyn_base::IdSource::new(),
            )
            .map_err(|e| error(e.to_string()))?
        };
        let storage_bindings =
            compiler.used.iter().filter_map(|id| compiler.bindings.get(id).cloned()).collect();
        entries.push(ssa::types::EntryPoint {
            id: EntryId::from(root.as_u32()),
            name: entry_name(root),
            body,
            execution_model: if compute {
                crate::flow::ExecutionModel::Compute {
                    local_size: (size[0], size[1], size[2]),
                }
            } else {
                match declaration.entry_kind {
                    crate::interface::EntryKind::Vertex => crate::flow::ExecutionModel::Vertex,
                    crate::interface::EntryKind::Fragment => crate::flow::ExecutionModel::Fragment,
                    _ => return Err(error("unextracted graphics root")),
                }
            },
            parameter_inputs: (0..inputs.len()).map(|i| vec![i]).collect(),
            inputs,
            outputs,
            storage_bindings,
            stage_descriptor_storage_accesses: Default::default(),
            pipeline_storage_accesses: Default::default(),
            span: Span::generated(),
        });
    }
    let (pipeline, physical_kernels) = compiler.publish(&mut entries, &roots)?;
    Ok(ssa::Program::bare(compiler.functions, entries, vec![])
        .with_context::<ssa::stage::ElaboratedTag, _>(ssa::context::BackendGlobal {
            pipeline,
            physical_kernels,
            profile: LoweringProfile::new(target, SchedulePolicy::Parallel),
        }))
}

fn entry_name(root: BlockId) -> String {
    format!("egg_kernel_{}", root.as_u32())
}

fn materialized(value: &Value) -> bool {
    match value {
        Value::Buffer(_) => true,
        Value::Field(value, _) => materialized(value),
        Value::Tuple(values) | Value::Primitive(_, values) => values.iter().any(materialized),
        _ => false,
    }
}

/// The existing runtime executes a static sequence. Do not advertise a
/// conditional or repeated launch as an unconditional executable pipeline.
fn static_host(data: &AssociatedData, root: BlockId) -> bool {
    // Scalar memory reads need their own stage if a later launch can overwrite
    // their inputs. Until those stages exist, moving the read to the final
    // publication kernel would change the observed value.
    let mut pending = vec![root];
    let mut read = false;
    while let Some(id) = pending.pop() {
        let block = &data.blocks[id];
        for instruction in &data.bodies[block.body].instructions {
            match instruction {
                Instruction::Evaluate(_) | Instruction::Call { .. } => read = true,
                Instruction::Dispatch(_) if read => return false,
                _ => {}
            }
        }
        if let Exit::Jump(edge) = &block.exit {
            pending.push(edge.target);
        }
        if matches!(block.exit, Exit::Branch { .. }) {
            return false;
        }
    }
    data.blocks.iter().filter(|(_, b)| b.function == root).all(|(_, b)| {
        !matches!(b.exit, Exit::Branch { .. })
            && !data.bodies[b.body].instructions.iter().any(|i| matches!(i, Instruction::Call { .. }))
    })
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
    origins: BTreeMap<ExprId, Span>,
    placements: BTreeMap<super::PlacementSite, Vec<ExprId>>,
    data: &'a AssociatedData,
    functions: Vec<ssa::types::Function>,
    specializations: Vec<Specialization>,
    inputs: BTreeMap<ParameterId, Vec<crate::interface::EntryInput>>,
    bindings: BTreeMap<BufferId, StorageBindingDecl>,
    results: BTreeMap<OperationId, Value>,
    used: BTreeSet<BufferId>,
}
impl Compiler<'_> {
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
    entry: Option<super::EntryId>,
    finish_outputs: bool,
    graphics_outputs: bool,
    inputs: Vec<crate::interface::EntryInput>,
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
            entry: None,
            finish_outputs: false,
            graphics_outputs: false,
            inputs: vec![],
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
        body.return_ty = if self.graphics_outputs { types::unit() } else { result_type(&types) };
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
                let values = if self.finish_outputs {
                    let owner = self.entry.unwrap();
                    for output in data.outputs.values().filter(|o| o.entry == owner && o.scalar) {
                        let buffer = output.buffer.ok_or_else(|| error("missing planned output buffer"))?;
                        self.instruction(&Instruction::Store {
                            buffer: Value::Buffer(buffer),
                            index: Value::Int(0),
                            value: Value::Source(output.expression),
                        })?;
                    }
                    vec![]
                } else {
                    self.values(&data.bodies[*body].results)?
                };
                let types: Vec<_> = values.iter().map(|v| v.ty.clone()).collect();
                if self.return_types.as_ref().is_some_and(|known| *known != types) {
                    return Err(error("inconsistent function return types"));
                }
                self.return_types = Some(types);
                let value = if values.is_empty() {
                    None
                } else if self.graphics_outputs {
                    let packed = self.pack(values)?;
                    let fields = match &packed.ty {
                        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => (0..fields
                            .len())
                            .map(|i| self.field(packed.clone(), i))
                            .collect::<Result<Vec<_>, _>>()?,
                        _ => vec![packed],
                    };
                    for (index, field) in fields.into_iter().enumerate() {
                        let result = self.builder.new_place(field.ty);
                        self.builder
                            .push_void_inst(InstKind::OutputSlot { index, result })
                            .map_err(builder_error)?;
                        self.builder
                            .push_void_inst(InstKind::Store {
                                place: result,
                                value: field.value,
                            })
                            .map_err(builder_error)?;
                    }
                    None
                } else {
                    Some(self.pack(values)?.value)
                };
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
                if matches!(value, Value::Local(name) if !self.environment.locals.contains_key(name))
                    && self.entry.is_some()
                {
                    return Ok(());
                }
                self.environment.expressions.remove(id);
                let v = self.value(value)?;
                self.environment.expressions.insert(*id, v);
            }
            Instruction::BindResult(id, value) => {
                if self.finish_outputs && self.compiler.results.contains_key(id) {
                    return Ok(());
                }
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
                    let count = self
                        .compiler
                        .constant(&buffer.length)
                        .ok_or_else(|| error("TODO: dynamic invocation-local allocation"))?
                        .max(1) as usize;
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
            Instruction::Dispatch(_) if self.finish_outputs => {}
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
            _ => return Err(error("runtime-sized array requires a storage view")),
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
