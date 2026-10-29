//! Direct instruction emission from selected scalar terms and source boundaries.
use super::{builder_error, error, Compiler, OptimizeError, Typed};
use crate::builtins::catalog;
use crate::egglog::bindings::Bindings;
use crate::op::{OpTag, PureViewSource};
use crate::ssa::builder::BuilderError;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::{BlockId, FuncBody, InstKind, Terminator};
use crate::ssa::types::{PlaceId, ValueRef};
use crate::types::{self, Type, TypeName};
use crate::BindingRef;
use crate::FunctionId;
use crate::{LookupMap, LookupSet};
use egglog_engine::{RawValues, TermId, Value};
use std::cell::RefCell;
use wyn_graph::DominatorTree;

mod collective;
mod control;
mod materialize;
mod readout;
mod resources;
mod subgroups;
mod values;

pub(super) struct Body<'a, 'p, 'source> {
    pub compiler: &'a mut Compiler<'p, 'source>,
    pub builder: FuncBuilder,
    pub context: Value,
    pub values: Bindings<Value, Typed>,
    // Keep emitted values at their actual blocks; a dominating instance can
    // serve uses even when its requested lexical placement was unavailable.
    scalar_values: LookupMap<TermId, Vec<(BlockId, Typed)>>,
    pub scopes: Bindings<Value, BlockId>,
    loading: LookupSet<Value>,
    dominators: RefCell<Option<DominatorTree<BlockId>>>,
    pub grid: Option<(u32, u32, u32)>,
    local_resources: LookupMap<Value, Typed>,
    local_arrays: LookupMap<ValueRef, PlaceId>,
    next_workgroup_array: u32,
    // Kernel storage ownership survives invocation-local element callbacks.
    pub workgroup_width: Option<u32>,
    pub invocation_execution: bool,
    pub subgroup: Option<subgroups::Subgroup>,
}

impl<'a, 'p, 'source> Body<'a, 'p, 'source> {
    pub fn new(
        compiler: &'a mut Compiler<'p, 'source>,
        scope: Value,
        parameters: Vec<Type>,
        result: Type,
    ) -> Result<Self, OptimizeError> {
        let Some(context) = compiler.facts.context(scope) else {
            return Err(error("function has no scalar context"));
        };
        let builder = FuncBuilder::new(parameters.clone(), result);
        let mut values = Bindings::default();
        for (i, ty) in parameters.into_iter().enumerate() {
            let Some(source) = compiler.facts.parameter(scope, i as i64) else {
                return Err(error("missing function parameter identity"));
            };
            values.insert(
                source,
                Typed {
                    value: builder.get_param(i).into(),
                    ty,
                },
            );
        }
        let mut scopes = Bindings::default();
        scopes.insert(scope, builder.entry());
        Ok(Self {
            compiler,
            builder,
            context,
            values,
            scopes,
            // Keep emitted values at their actual blocks; a dominating instance can
            // serve uses even when its requested lexical placement was unavailable.
            scalar_values: LookupMap::default(),
            loading: LookupSet::default(),
            dominators: RefCell::new(None),
            grid: None,
            local_resources: LookupMap::default(),
            local_arrays: LookupMap::default(),
            next_workgroup_array: 0,
            workgroup_width: None,
            invocation_execution: true,
            subgroup: None,
        })
    }

    pub fn finish(mut self, result: Typed) -> Result<FuncBody, OptimizeError> {
        self.terminate(Terminator::Return(
            (result.ty != types::unit()).then_some(result.value),
        ))
        .map_err(builder_error)?;
        let mut body = self.builder.finish().map_err(builder_error)?;
        body.return_ty = result.ty;
        Ok(body)
    }

    pub fn current(&self) -> Result<BlockId, OptimizeError> {
        let Some(block) = self.builder.current_block() else {
            return Err(error("no current SSA block"));
        };
        Ok(block)
    }

    pub fn value(&mut self, scope: Value, source: Value) -> Result<Typed, OptimizeError> {
        if let Some(value) = self.values.get(&source) {
            return Ok(value.clone());
        }
        if !self.loading.insert(source) {
            return Err(error("cyclic source evaluation during SSA emission"));
        }
        let value = if self.compiler.facts.contains("HostLoopAggregate", (source,)) {
            self.local(scope, source)?
        } else if let Some(&term) = self.compiler.program.stage.selected.roots.get(&(self.context, source))
        {
            self.scalar_body(scope, term)?
        } else {
            self.boundary(scope, source)?
        };
        self.loading.remove(&source);
        self.values.insert(source, value.clone());
        Ok(value)
    }

    fn scalar(&mut self, scope: Value, term: TermId) -> Result<Typed, OptimizeError> {
        self.scalar_at(scope, term, self.current()?)
    }

    fn scalar_at(&mut self, scope: Value, term: TermId, target: BlockId) -> Result<Typed, OptimizeError> {
        if let Some(&binding) = self.compiler.bindings.captures.get(&term) {
            let (_, fields) = self.compiler.program.stage.selected.app(term)?;
            let Some(ty) =
                self.compiler.facts.ty(self.compiler.program.stage.selected.values[fields[1]]).cloned()
            else {
                return Err(error("capture type missing"));
            };
            let element = crate::ssa::layout::storage_value_type(&ty);
            let zero = self.literal("0", &types::i32())?;
            let one = self.literal("1", &types::i32())?;
            let view = self.op(
                OpTag::StorageView(PureViewSource::Storage(binding)),
                vec![zero.clone(), one],
                Body::view_type(&element, types::buffer_tag(binding)),
            )?;
            let value = self.index(view, zero)?;
            return self.cast(value, &ty);
        }
        if let Some(value) = self.available_scalar(term)? {
            return Ok(value);
        }
        let program = self.compiler.program;
        let selected = &program.stage.selected;
        let (name, fields) = selected.app(term)?;
        let Some(ty) = self.compiler.facts.ty(selected.values[fields[1]]).cloned() else {
            return Err(error("selected scalar type missing"));
        };
        // Placement is applied to the resulting pure instruction. Operand
        // evaluation still occurs at the current control/effect position.
        let value = {
            match name {
                "ExprLength" => self.source_length(scope, selected.values[fields[2]])?,
                "ExprLeaf" => {
                    let source = selected.values[fields[2]];
                    // A source boundary can have its own optimized root in this
                    // context. Follow it unless it leads straight back to this leaf.
                    if let Some(value) = self.values.get(&source) {
                        value.clone()
                    } else if selected.roots.get(&(self.context, source)).is_some_and(|&root| root != term)
                    {
                        self.value(scope, source)?
                    } else {
                        self.boundary(scope, source)?
                    }
                }
                "ExprExecute" => self.local(scope, selected.values[fields[2]])?,
                "ExprInstruction" if selected.text(fields[3])? == "index" => {
                    let [mut array, mut index]: [TermId; 2] = selected
                        .arguments(fields[4])?
                        .try_into()
                        .map_err(|_| error("index needs two operands"))?;
                    // Keep a nested read as one access path. Materializing each
                    // intermediate fixed-size row turns a scalar lookup into a
                    // whole-row copy per invocation. Respect already evaluated
                    // values, which may be snapshots across an update.
                    let mut steps = Vec::new();
                    loop {
                        let (base, path) = selected.projected_array(array)?;
                        array = base;
                        steps.push((path, index));
                        if self.available_scalar(array)?.is_some()
                            || self.compiler.bindings.captures.contains_key(&array)
                        {
                            break;
                        }
                        let Some((parent, next_index)) = selected.index_operands(array)? else {
                            break;
                        };
                        array = parent;
                        index = next_index;
                    }
                    let array = self.scalar(scope, array)?;
                    let steps = steps
                        .into_iter()
                        .rev()
                        .map(|(fields, index)| Ok((fields, self.scalar(scope, index)?)))
                        .collect::<Result<Vec<_>, OptimizeError>>()?;
                    let value = self.index_path(array, &steps)?;
                    self.cast(value, &ty)?
                }
                "ExprInstruction" => {
                    let mut arguments = self.arguments(scope, fields[4])?;
                    let tag = selected.operator(fields[3], arguments.len())?;
                    if matches!(tag, OpTag::Intrinsic {id, ..} if id == catalog().known().length) {
                        let [array] = arguments.as_slice() else {
                            return Err(error("length requires one array"));
                        };
                        return self.length(array.clone());
                    }
                    if matches!(tag, OpTag::Intrinsic { id, .. }
                        if id == catalog().known().array_with
                            || id == catalog().known().array_with_in_place)
                    {
                        // A scheduled producer may reside in a read-only storage
                        // view. Fixed-array updates operate on owned contents,
                        // not on that descriptor or its shared backing buffer.
                        let layout = self.compiler.facts.value_layout(selected.values[fields[2]])?;
                        if self.compiler.facts.enode("ArrayLayout", layout).is_some() {
                            arguments[0] = self.materialize(arguments[0].clone(), layout)?;
                        }
                    }
                    let Some(representation) =
                        self.compiler.facts.lookup("InstructionResult", (selected.values[fields[2]],))
                    else {
                        return Err(error("instruction result representation missing"));
                    };
                    let ty = if let Some(fields) =
                        self.compiler.facts.enode("OperandResult", representation)
                    {
                        let index = usize::try_from(self.compiler.facts.integer(fields[0]))
                            .map_err(|_| error("invalid result operand"))?;
                        let Some(argument) = arguments.get(index) else {
                            return Err(error("result operand missing"));
                        };
                        argument.ty.clone()
                    } else if let Some(fields) = self.compiler.facts.enode("SemanticResult", representation)
                    {
                        let Some(ty) = self.compiler.facts.ty(fields[0]) else {
                            return Err(error("instruction result type missing"));
                        };
                        ty.clone()
                    } else {
                        return Err(error("unknown instruction representation"));
                    };
                    self.op(tag, arguments, ty)?
                }
                "ExprCall" => {
                    let args = self.arguments(scope, fields[4])?;
                    self.call(selected.values[fields[3]], args)?
                }
                "ExprParameter" => {
                    let region = selected.values[fields[2]];
                    let index = selected.integer(fields[3])?;
                    let Some(source) = self.compiler.facts.parameter(region, index) else {
                        return Err(error("selected parameter missing"));
                    };
                    let Some(value) = self.values.get(&source).cloned() else {
                        return Err(error(format!(
                            "selected parameter {source:?} in {region:?} is unbound in {:?}, bound {:?}",
                            self.context,
                            self.values.keys().collect::<Vec<_>>()
                        )));
                    };
                    value
                }
                "ExprLiteral" => {
                    let text = selected.text(fields[2])?;
                    self.literal(text, &ty)?
                }
                "ExprUnary" | "ExprBinary" | "ExprOp" => {
                    let arguments = selected
                        .operation_arguments(term)?
                        .into_iter()
                        .map(|arg| self.scalar(scope, arg))
                        .collect::<Result<Vec<_>, _>>()?;
                    let tag = selected.operator(fields[2], arguments.len())?;
                    self.op_at(target, tag, arguments, ty)?
                }
                "ExprInvoke" => {
                    let callee = selected.values[fields[2]];
                    let args = self.arguments(scope, fields[3])?;
                    self.call(callee, args)?
                }
                "ExprTuple" | "ExprVector" => {
                    let args = self.arguments(scope, fields[2])?;
                    let tag = if name == "ExprTuple" { OpTag::Tuple } else { OpTag::Vector };
                    let ty =
                        if let Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), _) =
                            types::strip_existentials(&ty)
                        {
                            Type::Constructed(name.clone(), args.iter().map(|v| v.ty.clone()).collect())
                        } else if name == "ExprTuple" {
                            types::tuple(args.iter().map(|v| v.ty.clone()).collect())
                        } else {
                            ty
                        };
                    self.op_at(target, tag, args, ty)?
                }
                "ExprProject" => {
                    let index = selected.integer(fields[3])? as usize;
                    let base = self.scalar(scope, fields[2])?;
                    self.field(base, index)?
                }
                "ExprCoerce" => {
                    let value = self.scalar(scope, fields[2])?;
                    self.cast(value, &ty)?
                }
                "ExprChoice" => self.scalar_body(scope, term)?,
                _ => return Err(error(format!("unresolved selected constructor {name}"))),
            }
        };
        let block = match value.value {
            ValueRef::Ssa(id) => {
                let Some(block) = self.builder.func().block_of_value(id) else {
                    return Err(error("selected expression produced an unplaced SSA value"));
                };
                block
            }
            _ => target,
        };
        self.scalar_values.entry(term).or_default().push((block, value.clone()));
        Ok(value)
    }

    fn available_scalar(&self, term: TermId) -> Result<Option<Typed>, OptimizeError> {
        let current = self.current()?;
        Ok(self.scalar_values.get(&term).and_then(|values| {
            values
                .iter()
                .rev()
                .find(|(block, _)| self.dominates(*block, current))
                .map(|(_, value)| value.clone())
        }))
    }

    fn dominates(&self, definition: BlockId, use_block: BlockId) -> bool {
        if definition == use_block {
            return true;
        }
        let mut cached = self.dominators.borrow_mut();
        let tree = cached.get_or_insert_with(|| {
            let function = self.builder.func();
            DominatorTree::build(function.entry, |block, out| {
                out.extend(function.blocks[block].term.successors())
            })
        });
        tree.dominates(definition, use_block)
    }

    fn terminate(&mut self, term: Terminator) -> Result<(), BuilderError> {
        *self.dominators.get_mut() = None;
        self.builder.terminate(term)
    }

    fn arguments(&mut self, scope: Value, term: TermId) -> Result<Vec<Typed>, OptimizeError> {
        self.compiler
            .program
            .stage
            .selected
            .arguments(term)?
            .into_iter()
            .map(|arg| self.scalar(scope, arg))
            .collect()
    }

    pub fn callback(
        &mut self,
        owner: Value,
        operation: Value,
        arguments: Vec<Typed>,
    ) -> Result<Typed, OptimizeError> {
        let Some(scope) = self.compiler.facts.callback(operation) else {
            return Err(error("collective callback is missing"));
        };
        self.invocation_local(|body| {
            body.in_region(owner, scope, arguments, |body, source| {
                let Some(&term) = body.compiler.program.stage.selected.roots.get(&(body.context, source))
                else {
                    return Err(error("callback has no selected scalar body"));
                };
                body.scalar_body(scope, term)
            })
        })
    }

    pub fn invocation_local<T>(
        &mut self,
        emit: impl FnOnce(&mut Self) -> Result<T, OptimizeError>,
    ) -> Result<T, OptimizeError> {
        let invocation = std::mem::replace(&mut self.invocation_execution, true);
        let result = emit(self);
        self.invocation_execution = invocation;
        result
    }

    pub fn in_region<T>(
        &mut self,
        owner: Value,
        scope: Value,
        arguments: Vec<Typed>,
        emit: impl FnOnce(&mut Self, Value) -> Result<T, OptimizeError>,
    ) -> Result<T, OptimizeError> {
        let Some(context) = self.compiler.facts.context(scope) else {
            return Err(error("callback scalar context is missing"));
        };
        let Some(source) = self.compiler.facts.result(scope) else {
            return Err(error("callback result is missing"));
        };
        let mut bindings = Vec::new();
        for (formal, actual) in self.compiler.facts.captures(scope)? {
            bindings.push((formal, self.value(owner, actual)?));
        }
        if self.compiler.facts.parameter(scope, arguments.len() as i64).is_some() {
            return Err(error("callback has too few arguments"));
        }
        for (i, argument) in arguments.into_iter().enumerate() {
            let Some(formal) = self.compiler.facts.parameter(scope, i as i64) else {
                return Err(error("callback has too many arguments"));
            };
            let Some(ty) = self.compiler.facts.source_type(formal).cloned() else {
                return Err(error("callback parameter type is missing"));
            };
            bindings.push((formal, self.cast(argument, &ty)?));
        }
        let block = self.current()?;
        let old_context = std::mem::replace(&mut self.context, context);
        let old_values = self.values.checkpoint();
        let old_scopes = std::mem::take(&mut self.scopes);
        // Parameters change between invocations, even in the same SSA block.
        let old_scalar = std::mem::take(&mut self.scalar_values);
        let old_resources = std::mem::take(&mut self.local_resources);
        self.scopes.insert(scope, block);
        for (formal, value) in bindings {
            self.values.insert(formal, value);
        }
        let result = emit(self, source);
        self.context = old_context;
        self.values.restore(old_values);
        self.scopes = old_scopes;
        self.scalar_values = old_scalar;
        self.local_resources = old_resources;
        result
    }

    pub fn execution_scope(&self) -> Result<Value, OptimizeError> {
        let name = if self.invocation_execution { "InvocationScope" } else { "WorkgroupScope" };
        let Some(scope) = self.compiler.facts.constructor(name, RawValues(Vec::new())) else {
            return Err(error("execution scope is missing"));
        };
        Ok(scope)
    }

    /// Read a selected execution scope; the active width belongs to its kernel.
    pub fn has_workgroup_scope(&self, fact: &str, owner: Value) -> bool {
        self.compiler
            .facts
            .lookup(fact, (owner,))
            .is_some_and(|scope| self.compiler.facts.enode("WorkgroupScope", scope).is_some())
    }

    pub fn call(&mut self, scope: Value, args: Vec<Typed>) -> Result<Typed, OptimizeError> {
        let domain = if self.workgroup_width.is_some() { "WorkgroupScope" } else { "InvocationScope" };
        let Some(enclosing) = self.compiler.facts.constructor(domain, RawValues(Vec::new())) else {
            return Err(error("call storage domain missing"));
        };
        let Some(inline) = self.compiler.facts.lookup("InlineCall", (scope, enclosing)) else {
            return Err(error("call has no selected execution context"));
        };
        if self.compiler.program.graph.value_to_base::<bool>(inline) {
            return self.in_region(scope, scope, args, |body, source| body.value(scope, source));
        }
        let id = self.compiler.function(scope, &args)?;
        let Some(function) = self.compiler.functions.iter().find(|function| function.id == id) else {
            return Err(error(
                "recursive helper requires a declared return representation",
            ));
        };
        let ty = function.body.return_ty.clone();
        self.op(OpTag::Call(id), args, ty)
    }

    pub fn op(
        &mut self,
        tag: OpTag<BindingRef, FunctionId>,
        args: Vec<Typed>,
        ty: Type,
    ) -> Result<Typed, OptimizeError> {
        self.op_at(self.current()?, tag, args, ty)
    }
    fn op_at(
        &mut self,
        block: BlockId,
        tag: OpTag<BindingRef, FunctionId>,
        mut args: Vec<Typed>,
        mut ty: Type,
    ) -> Result<Typed, OptimizeError> {
        if matches!(tag, OpTag::Intrinsic { id, .. } if id == catalog().known().record_with) {
            let [record, index, replacement] = args.as_slice() else {
                return Err(error("record update requires three operands"));
            };
            let index =
                match index.value {
                    ValueRef::Const(crate::ssa::types::ConstantValue::I32(index)) => Some(index),
                    ValueRef::Ssa(value) => self.builder.func().inst_of_value(value).and_then(|inst| {
                        match &self.builder.func().insts[inst].data {
                            InstKind::Op {
                                tag: OpTag::Int(text),
                                ..
                            } => text.parse().ok(),
                            _ => None,
                        }
                    }),
                    _ => None,
                }
                .ok_or_else(|| error("record update requires a constant field index"))?;
            let Type::Constructed(name @ TypeName::Record(_), fields) = &record.ty else {
                return Err(error("record update requires a record"));
            };
            let mut fields = fields.clone();
            let field = usize::try_from(index)
                .ok()
                .and_then(|index| fields.get_mut(index))
                .ok_or_else(|| error("record update field index out of range"))?;
            // Preserve concrete field representations, including bound array views.
            *field = replacement.ty.clone();
            ty = Type::Constructed(name.clone(), fields);
            args[1].value = ValueRef::Const(crate::ssa::types::ConstantValue::I32(index));
        }
        if matches!(tag,OpTag::Intrinsic{id,..} if id==catalog().known().scratch_annotation) {
            let [array] = args.as_slice() else {
                return Err(error("scratch annotation needs one array"));
            };
            // The planner has consumed this permission to omit initialization.
            // Any initializer it retained has already been emitted above.
            return Ok(array.clone());
        }
        // A selected lexical scope may precede a merge introduced while
        // emitting an operand. Keep the instruction after that operand.
        let mut available = true;
        for arg in &args {
            if let ValueRef::Ssa(value) = arg.value {
                let Some(definition) = self.builder.func().block_of_value(value) else {
                    return Err(error("selected instruction has an unplaced operand"));
                };
                available &= self.dominates(definition, block);
            }
        }
        let block = if available { block } else { self.current()? };
        let instruction = InstKind::Op {
            tag,
            operands: args.into_iter().map(|arg| arg.value).collect(),
        };
        let value = self.builder.func_mut().append_inst(block, instruction, ty.clone());
        Ok(Typed {
            value: value.into(),
            ty,
        })
    }
}
