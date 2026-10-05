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
use egglog_engine::{TermId, Value};
use std::cell::RefCell;
use wyn_graph::DominatorTree;

mod collective;
mod control;
mod materialize;
mod readout;
mod resources;
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
        let builder = FuncBuilder::new(
            parameters.iter().enumerate().map(|(i, ty)| (ty.clone(), format!("arg{i}"))).collect(),
            result,
        );
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
        let value =
            if let Some(&term) = self.compiler.program.stage.selected.roots.get(&(self.context, source)) {
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
        if let Some(&binding) = self.compiler.plan.captures.get(&term) {
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
                super::interface::view_type(&element, types::buffer_tag(binding)),
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
                "ScalarLeaf" => {
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
                "ScalarExecute" => self.local(scope, selected.values[fields[2]])?,
                "ScalarInstruction" => {
                    let arguments = self.arguments(scope, fields[4])?;
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
                    let tag = selected.operator(fields[3], arguments.len())?;
                    if matches!(tag, OpTag::Index) {
                        let [array, index]: [Typed; 2] =
                            arguments.try_into().map_err(|_| error("index needs two operands"))?;
                        let value = self.index(array, index)?;
                        self.cast(value, &ty)?
                    } else {
                        self.op(tag, arguments, ty)?
                    }
                }
                "ScalarCall" => {
                    let args = self.arguments(scope, fields[4])?;
                    self.call(selected.values[fields[3]], args)?
                }
                "ScalarParameter" => {
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
                "ScalarLiteral" => {
                    let text = selected.text(fields[2])?;
                    self.literal(text, &ty)?
                }
                "ScalarUnary" | "ScalarBinary" | "ScalarOp" => {
                    let arguments = selected
                        .operation_arguments(term)?
                        .into_iter()
                        .map(|arg| self.scalar(scope, arg))
                        .collect::<Result<Vec<_>, _>>()?;
                    let tag = selected.operator(fields[2], arguments.len())?;
                    self.op_at(target, tag, arguments, ty)?
                }
                "ScalarInvoke" => {
                    let callee = selected.values[fields[2]];
                    let args = self.arguments(scope, fields[3])?;
                    self.call(callee, args)?
                }
                "ScalarTuple" | "ScalarVector" => {
                    let args = self.arguments(scope, fields[2])?;
                    let tag = if name == "ScalarTuple" {
                        OpTag::Tuple(args.len())
                    } else {
                        OpTag::Vector(args.len())
                    };
                    let ty =
                        if let Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), _) =
                            types::strip_existentials(&ty)
                        {
                            Type::Constructed(name.clone(), args.iter().map(|v| v.ty.clone()).collect())
                        } else if name == "ScalarTuple" {
                            types::tuple(args.iter().map(|v| v.ty.clone()).collect())
                        } else {
                            ty
                        };
                    self.op_at(target, tag, args, ty)?
                }
                "ScalarProject" => {
                    let index = selected.integer(fields[3])? as usize;
                    let base = self.scalar(scope, fields[2])?;
                    self.field(base, index)?
                }
                "ScalarCoerce" => {
                    let value = self.scalar(scope, fields[2])?;
                    self.cast(value, &ty)?
                }
                "ScalarChoice" => self.scalar_body(scope, term)?,
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
        let Some(context) = self.compiler.facts.context(scope) else {
            return Err(error("callback scalar context is missing"));
        };
        let Some(source) = self.compiler.facts.result(scope) else {
            return Err(error("callback result is missing"));
        };
        let Some(&term) = self.compiler.program.stage.selected.roots.get(&(context, source)) else {
            return Err(error("callback has no selected scalar body"));
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
        let result = self.scalar_body(scope, term);
        self.context = old_context;
        self.values.restore(old_values);
        self.scopes = old_scopes;
        self.scalar_values = old_scalar;
        self.local_resources = old_resources;
        result
    }

    pub fn call(&mut self, scope: Value, args: Vec<Typed>) -> Result<Typed, OptimizeError> {
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
        args: Vec<Typed>,
        ty: Type,
    ) -> Result<Typed, OptimizeError> {
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
