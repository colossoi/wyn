//! SSA handoff from scheduled blocks and the derived resource plan.
use super::abi::{concrete, u32_type};
use super::blocks::{Control, Exit, Instruction, Storage, Value};
use super::scalar::placement_index;
use super::timing::span;
use super::{OptimizeError, PlacementSite};
use crate::ast::Span;
use crate::egglog::data::{BlockId, BufferId, ExprId, ExternId, OperationId, OperationKind, ParameterId};
use crate::egglog::{Program, Scheduled};
use crate::flow::{ControlHeader, ExecutionModel};
use crate::interface::lowering::build_entry_outputs;
use crate::interface::{EntryInput, EntryKind};
use crate::op::{OpTag, PureViewSource};
use crate::pipeline_descriptor::DispatchSize;
use crate::ssa::builder::{BuilderError, FuncBuilder};
use crate::ssa::context::BackendGlobal;
use crate::ssa::stage::{Elaborated, ElaboratedTag};
use crate::ssa::types::{
    AtomicOp, ConstantValue, EntryPoint, FuncBody, Function, InstKind, PlaceId, Terminator, ValueRef,
};
use crate::types::{bool_type, sized_array, unit, Type, TypeExt, TypeName};
use crate::{ssa, types, BindingRef, CodegenTarget, EntryId, FunctionId, LoweringProfile, SchedulePolicy};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use wyn_base::IdSource;

mod values;

/// Lower scheduled kernels with their authored inputs and planned storage.
/// Static compute pipelines publish the same resources and dispatch order.
/// TODO: the runtime descriptor cannot yet execute host branches/repeated launches.
pub fn to_ssa(data: &Program<Scheduled>, target: CodegenTarget) -> Result<Elaborated, OptimizeError> {
    let _timing = span("egglog to SSA");
    if let Some(root) = data.state.unsupported_host {
        return Err(error(format!(
            "TODO: runtime publication of conditional or repeated host dispatches ({root:?})"
        )));
    }
    let mut origins = BTreeMap::new();
    for origin in data.origins.values().filter(|origin| origin.span.module().is_some()) {
        origins.entry(origin.expression).or_insert(origin.span);
    }
    let mut compiler = Compiler {
        origins,
        placements: placement_index(&data.ir, &data.state.placements),
        data,
        functions: vec![],
        externs: BTreeMap::new(),
        specializations: HashMap::new(),
        active: HashSet::new(),
        used: BTreeSet::new(),
    };
    let roots = &data.state.abi.roots;
    let mut entries = vec![];
    for &(root, owner, size, finish) in roots {
        compiler.used.clear();
        let mut lower = Body::new(&mut compiler, root, &[], size[0])?;
        lower.entry = Some(owner);
        let declaration = &data.entries[owner].declaration;
        let compute = !finish || declaration.entry_kind == EntryKind::Compute;
        lower.finish_outputs = finish && compute;
        lower.graphics_outputs = !compute;
        // Root parameters are source ABI values. Kernel captures are resolved
        // lazily, so a combine phase does not declare unused input arrays.
        if finish {
            for instruction in &data.state.bodies[data.state.blocks[root].body].instructions {
                if let Instruction::BindParameter(p, Value::Local(name)) = instruction {
                    let value = lower.input(*p)?;
                    lower.environment.locals.insert(name.clone(), value);
                }
            }
        }
        lower.visit(root, None).map_err(|e| error(format!("{} ({root:?}): {e}", declaration.name)))?;
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
                &mut IdSource::new(),
            )
            .map_err(|e| error(e.to_string()))?
        };
        let storage_bindings = compiler
            .used
            .iter()
            .filter_map(|id| compiler.data.state.abi.bindings.get(id).cloned())
            .collect();
        entries.push(EntryPoint {
            id: EntryId::from(root.as_u32()),
            name: entry_name(root),
            body,
            execution_model: if compute {
                ExecutionModel::Compute {
                    local_size: (size[0], size[1], size[2]),
                }
            } else {
                match declaration.entry_kind {
                    EntryKind::Vertex => ExecutionModel::Vertex,
                    EntryKind::Fragment => ExecutionModel::Fragment,
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
    let pipeline = super::publish::publish(
        &data.state.abi,
        &data.entries,
        &data.symbols,
        &data.state.outputs,
        &mut entries,
    )?;
    Ok(
        ssa::Program::bare(compiler.functions, entries, vec![]).with_context::<ElaboratedTag, _>(
            BackendGlobal {
                pipeline,
                physical_kernels: data.state.physical_kernels.clone(),
                profile: LoweringProfile::new(target, SchedulePolicy::Parallel),
            },
        ),
    )
}

fn entry_name(root: BlockId) -> String {
    format!("egg_kernel_{}", root.as_u32())
}

fn error(message: impl Into<String>) -> OptimizeError {
    OptimizeError::Output(format!("egglog to SSA: {}", message.into()))
}
fn builder_error(e: BuilderError) -> OptimizeError {
    error(e.to_string())
}

struct Compiler<'a> {
    origins: BTreeMap<ExprId, Span>,
    placements: BTreeMap<PlacementSite, Vec<ExprId>>,
    data: &'a Program<Scheduled>,
    functions: Vec<Function>,
    externs: BTreeMap<ExternId, FunctionId>,
    specializations: HashMap<(BlockId, Vec<Type>), (FunctionId, BTreeSet<BufferId>)>,
    active: HashSet<BlockId>,
    used: BTreeSet<BufferId>,
}
impl Compiler<'_> {
    // Emit each specialization once. Its signature comes from the actual SSA
    // body, and cached resource uses belong to every shader that calls it.
    fn function(&mut self, source: BlockId, parameters: Vec<Type>) -> Result<FunctionId, OptimizeError> {
        let key = (source, parameters);
        if let Some((id, buffers)) = self.specializations.get(&key) {
            self.used.extend(buffers);
            return Ok(*id);
        }
        if !self.active.insert(source) {
            return Err(error("recursive device helper"));
        }
        let caller_buffers = std::mem::take(&mut self.used);
        let mut lower = Body::new(self, source, &key.1, 1)?;
        lower.visit(source, None)?;
        let (body, results) = lower.finish()?;
        if self.data.state.blocks[source].interface.as_ref().is_none_or(|f| f.results != results.len()) {
            return Err(error("helper result arity"));
        }
        let index = u32::try_from(self.functions.len()).map_err(|_| error("too many helpers"))?;
        let id = FunctionId::from(index);
        self.functions.push(Function {
            id,
            name: format!("egg_helper_{}_{}", source.as_u32(), index),
            body,
            span: Span::generated(),
            linkage_name: None,
        });
        let buffers = std::mem::replace(&mut self.used, caller_buffers);
        self.used.extend(&buffers);
        self.specializations.insert(key, (id, buffers));
        self.active.remove(&source);
        Ok(id)
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
    parameter_types: BTreeMap<BlockId, Vec<Type>>,
    return_types: Option<Vec<Type>>,
    width: u32,
    grid_yz: [u32; 2],
    entry: Option<super::EntryId>,
    finish_outputs: bool,
    graphics_outputs: bool,
    inputs: Vec<EntryInput>,
}
impl<'a, 'b> Body<'a, 'b> {
    fn new(
        compiler: &'a mut Compiler<'b>,
        entry: BlockId,
        types: &[Type],
        width: u32,
    ) -> Result<Self, OptimizeError> {
        let names = &compiler.data.state.blocks[entry].parameters;
        if !types.is_empty() && names.len() != types.len() {
            return Err(error("function parameter arity"));
        }
        let mut builder =
            FuncBuilder::new(types.iter().cloned().zip(names.iter().cloned()).collect(), unit());
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
        let Some(function) = &compiler.data.state.blocks[entry].interface else {
            return Err(error("helper has no function interface"));
        };
        for &id in &function.blocks {
            if id != entry {
                blocks.insert(id, builder.create_block());
            }
        }
        let grid_yz = match compiler.data.state.abi.dispatch_sizes.get(&entry) {
            Some(DispatchSize::Fixed { y, z, .. }) => [*y, *z],
            _ => [1, 1],
        };
        Ok(Self {
            compiler,
            builder,
            environment,
            blocks,
            entered: BTreeSet::new(),
            parameter_types: BTreeMap::new(),
            return_types: None,
            width,
            grid_yz,
            entry: None,
            finish_outputs: false,
            graphics_outputs: false,
            inputs: vec![],
        })
    }
    fn finish(self) -> Result<(FuncBody, Vec<Type>), OptimizeError> {
        let types = self.return_types.unwrap_or_default();
        let mut body = self.builder.finish().map_err(builder_error)?;
        body.return_ty = if self.graphics_outputs { unit() } else { result_type(&types) };
        Ok((body, types))
    }
    fn visit(&mut self, id: BlockId, stop: Option<BlockId>) -> Result<(), OptimizeError> {
        if Some(id) == stop || !self.entered.insert(id) {
            return Ok(());
        }
        self.builder.switch_to_block_unchecked(self.blocks[&id]);
        let data = self.compiler.data;
        for (name, &value) in data.state.blocks[id]
            .parameters
            .iter()
            .zip(&self.builder.func().blocks[self.blocks[&id]].params)
        {
            self.environment.locals.insert(
                name.clone(),
                Typed {
                    value: value.into(),
                    ty: self.builder.func().value_type(value).clone(),
                },
            );
        }
        if let Some(control @ Control::Loop { .. }) = data.state.blocks[id].control {
            let control = match control {
                Control::Selection { .. } => unreachable!(),
                Control::Loop { merge, continuing } => ControlHeader::Loop {
                    merge: self.blocks[&merge],
                    continue_block: self.blocks[&continuing],
                },
            };
            self.builder.set_control_header(self.blocks[&id], control);
        }
        for instruction in &data.state.bodies[data.state.blocks[id].body].instructions {
            self.instruction(instruction)?;
        }
        match &data.state.blocks[id].exit {
            Exit::Return(body) => {
                let values = if self.finish_outputs {
                    vec![]
                } else {
                    self.values(&data.state.bodies[*body].results)?
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
                let values = self.values(&data.state.bodies[edge.arguments].results)?;
                let (target, args) = self.edge(edge.target, values)?;
                self.builder.terminate(Terminator::Branch { target, args }).map_err(builder_error)?;
                self.visit(edge.target, stop)?;
            }
            Exit::Branch { condition, yes, no } => {
                let values = self.values(&data.state.bodies[*condition].results)?;
                let [condition] = values.as_slice() else {
                    return Err(error("branch condition arity"));
                };
                let y = self.values(&data.state.bodies[yes.arguments].results)?;
                let n = self.values(&data.state.bodies[no.arguments].results)?;
                let (then_target, then_args) = self.edge(yes.target, y)?;
                let (else_target, else_args) = self.edge(no.target, n)?;
                if let Some(Control::Selection { merge }) = data.state.blocks[id].control {
                    let Some(header) = self.builder.current_block() else {
                        return Err(error("missing selection header"));
                    };
                    self.builder.set_control_header(
                        header,
                        ControlHeader::Selection {
                            merge: self.blocks[&merge],
                        },
                    );
                }
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
                let merge = match data.state.blocks[id].control {
                    Some(Control::Selection { merge }) => Some(merge),
                    _ => None,
                };
                self.visit(yes.target, merge.or(stop))?;
                self.environment = environment.clone();
                self.visit(no.target, merge.or(stop))?;
                if let Some(merge) = merge {
                    // Neither arm's local bindings dominate the join. Only
                    // its explicit block parameters carry values out of it.
                    self.environment = environment;
                    self.visit(merge, stop)?;
                }
            }
        }
        Ok(())
    }
    fn edge(
        &mut self,
        target: BlockId,
        values: Vec<Typed>,
    ) -> Result<(crate::flow::BlockId, Vec<ValueRef>), OptimizeError> {
        if self.compiler.data.state.blocks[target].parameters.len() != values.len() {
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
        Ok((
            self.blocks[&target],
            values.into_iter().map(|v| v.value).collect(),
        ))
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
                if self.finish_outputs && self.compiler.data.state.materialized.contains_key(id) {
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
                let id = self.compiler.function(*function, args.iter().map(|v| v.ty.clone()).collect())?;
                let ty = self.compiler.functions[id.0 as usize].body.return_ty.clone();
                if self.compiler.data.state.blocks[*function]
                    .interface
                    .as_ref()
                    .is_none_or(|f| f.results != results.len())
                {
                    return Err(error("call result arity"));
                }
                let value = self.op(OpTag::Call(id), args, ty)?;
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
                let (place, ty) = self.indexed_destination(buffer, i)?;
                let v = self.cast(v, &ty)?;
                self.builder
                    .push_void_inst(InstKind::Store {
                        place,
                        value: v.value,
                    })
                    .map_err(builder_error)?;
            }
            Instruction::Allocate(id) => {
                let buffer = &self.compiler.data.state.buffers[*id];
                if buffer.storage == Storage::Function {
                    let Some(&count) = self.compiler.data.state.abi.local_lengths.get(id) else {
                        return Err(error("TODO: dynamic invocation-local allocation"));
                    };
                    let count = count.max(1) as usize;
                    let ty = sized_array(count, concrete(&buffer.element)?);
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
            Instruction::Barrier => {
                self.builder.push_void_inst(InstKind::ControlBarrier).map_err(builder_error)?;
            }
            Instruction::Atomic {
                result,
                buffer,
                index,
                op,
                values,
            } => {
                let index = self.value(index)?;
                let (place, ty) = self.indexed_destination(buffer, index)?;
                let values = self
                    .values(values)?
                    .into_iter()
                    .map(|v| self.cast(v, &ty).map(|v| v.value))
                    .collect::<Result<_, _>>()?;
                let ty = if *op == AtomicOp::CompareExchange {
                    Type::Constructed(TypeName::Tuple(2), vec![ty, bool_type()])
                } else {
                    ty
                };
                let value = self
                    .builder
                    .push_inst(
                        InstKind::Atomic {
                            place,
                            op: *op,
                            values,
                        },
                        ty.clone(),
                    )
                    .map_err(builder_error)?;
                self.environment.locals.insert(
                    result.clone(),
                    Typed {
                        value: value.into(),
                        ty,
                    },
                );
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
        [] => unit(),
        [ty] => ty.clone(),
        _ => types::tuple(types.to_vec()),
    }
}
fn uint(n: u32) -> ValueRef {
    ValueRef::Const(ConstantValue::U32(n))
}

#[cfg(test)]
#[path = "to_ssa_tests.rs"]
mod tests;
