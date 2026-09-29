//! Direct instruction emission from selected scalar terms and source boundaries.
use super::{builder_error, error, Compiler, OptimizeError, Typed};
use crate::builtins::catalog;
use crate::egglog::bindings::Bindings;
use crate::egglog::source::Term;
use crate::egglog::to_ssa::host;
use crate::host::ScalarExpr;
use crate::interface::StorageBindingDecl;
use crate::op::OpTag;
use crate::ssa::builder::BuilderError;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::PlaceId;
use crate::ssa::types::ValueRef;
use crate::ssa::types::{BlockId, FuncBody, InstKind, Terminator};
use crate::tlc::ArrayExpr;
use crate::tlc::{TermKind, VarRef};
use crate::types::{self, Type, TypeExt, TypeName};
use crate::BindingRef;
use crate::FunctionId;
use crate::{LookupMap, LookupSet};
use egglog_engine::{TermId, Value};
use std::cell::RefCell;
use wyn_graph::DominatorTree;

mod collective;
mod control;
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
    pub active_operations: LookupSet<Value>,
    pub host_arguments: Bindings<Value, ScalarExpr>,
    pub capture_bindings: Vec<StorageBindingDecl>,
    pub grid: Option<(u32, u32, u32)>,
    pub host_stage: Option<String>,
    pub host_scalar_bindings: LookupMap<TermId, BindingRef>,
    pub resource_uses: LookupMap<Value, i64>,
    pub local_arrays: LookupMap<ValueRef, PlaceId>,
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
            resource_uses: LookupMap::default(),
            local_arrays: LookupMap::default(),
            host_arguments: Bindings::default(),
            capture_bindings: Vec::new(),
            host_stage: None,
            grid: None,
            host_scalar_bindings: LookupMap::default(),
            active_operations: LookupSet::default(),
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

    pub fn identity(&self, scope: Value, term: &Term) -> Result<Value, OptimizeError> {
        let Some(&value) = self.compiler.program.identities.occurrences.get(&(scope, term.id)) else {
            return Err(error("source occurrence missing during SSA emission"));
        };
        Ok(value)
    }

    pub fn source(&mut self, scope: Value, term: &'source Term) -> Result<Typed, OptimizeError> {
        // Let syntax has no graph node, but evaluating a binding preserves its
        // effects even if the resulting value has no live scalar uses.
        if let TermKind::Let { rhs: value, body, .. } = &term.kind {
            let identity = self.identity(scope, value)?;
            if !self.compiler.plan.pure(identity) {
                self.source(scope, value)?;
            }
            return self.source(scope, body);
        }
        let source = self.identity(scope, term)?;
        self.value(scope, source)
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
                self.scalar(scope, term)?
            } else {
                self.original(scope, source)?
            };
        self.loading.remove(&source);
        self.values.insert(source, value.clone());
        Ok(value)
    }

    fn original(&mut self, scope: Value, source: Value) -> Result<Typed, OptimizeError> {
        if let Some(value) = self.values.get(&source) {
            return Ok(value.clone());
        }
        if let Some(resource) = self.compiler.plan.value_ref(source) {
            if !self.compiler.facts.operation(source).is_some_and(|op| self.active_operations.contains(&op))
                && self.compiler.plan.backing(resource).is_some()
                && !self.compiler.plan.external(resource).is_some()
            {
                let view = self.resource(scope, resource, 1)?;
                if self
                    .compiler
                    .facts
                    .source_type(source)
                    .is_some_and(|ty| !TypeExt::is_array(types::strip_existentials(ty)))
                {
                    let zero = self.literal("0", &types::i32())?;
                    let value = self.index(view, zero)?;
                    let Some(ty) = self.compiler.facts.source_type(source).cloned() else {
                        return Err(error("missing source_type"));
                    };
                    return self.cast(value, &ty);
                }
                return Ok(view);
            }
        }
        if let Some(actual) = self.compiler.facts.alias(source) {
            return self.value(scope, actual);
        }
        if let Some(symbol) = self.compiler.facts.global_symbol(source) {
            let Some(region) = self.compiler.facts.definition(symbol) else {
                return Err(error("global definition missing"));
            };
            return self.call(region, Vec::new());
        }
        if let Some((parent, index)) = self.compiler.facts.projection(source) {
            let parent = self.value(scope, parent)?;
            return self.field(parent, index);
        }
        if let Some(&(array, owner)) = self.compiler.program.identities.arrays.get(&source) {
            if !matches!(array, ArrayExpr::Var(_, _)) {
                return self.array(owner, array, &array.array_type());
            }
        }
        let Some(&(term, owner)) = self.compiler.program.identities.origins.get(&source) else {
            return Err(error(format!("unbound source value {source:?}")));
        };
        if let TermKind::App { func, args } = &term.kind {
            if matches!(func.kind,TermKind::Var(VarRef::Builtin{id,..}) if id==catalog().known().length) {
                if let [array] = args.as_slice() {
                    let array = self.identity(owner, array)?;
                    let value = self.source_length(scope, array)?;
                    return self.cast(value, &term.ty);
                }
            }
        }
        self.term(owner, scope, source, term)
    }

    fn term(
        &mut self,
        owner: Value,
        scope: Value,
        source: Value,
        term: &'source Term,
    ) -> Result<Typed, OptimizeError> {
        match &term.kind {
            TermKind::IntLit(text) => self.literal(text, &term.ty),
            TermKind::FloatLit(x) => self.literal(&x.to_bits().to_string(), &term.ty),
            TermKind::BoolLit(x) => self.op(OpTag::Bool(*x), vec![], types::bool_type()),
            TermKind::UnitLit => self.op(OpTag::Unit, vec![], types::unit()),
            TermKind::App { func, args } => {
                let mut arguments = Vec::new();
                for argument in args {
                    arguments.push(self.source(owner, argument)?);
                }
                let tag = match &func.kind {
                    TermKind::BinOp(op) => OpTag::BinOp(op.op),
                    TermKind::UnOp(op) => OpTag::UnaryOp(op.op),
                    TermKind::Var(VarRef::Builtin { id, overload_idx }) => OpTag::Intrinsic {
                        id: *id,
                        overload_idx: *overload_idx,
                    },
                    _ => {
                        let function = self.identity(owner, func)?;
                        return self.call_value(owner, function, arguments);
                    }
                };
                self.op(tag, arguments, term.ty.clone())
            }
            TermKind::Tuple(fields) | TermKind::VecLit(fields) => {
                let mut values = Vec::new();
                for field in fields {
                    values.push(self.source(owner, field)?);
                }
                let tag = if matches!(term.kind, TermKind::Tuple(_)) {
                    OpTag::Tuple(values.len())
                } else {
                    OpTag::Vector(values.len())
                };
                let ty = if let Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), _) =
                    types::strip_existentials(&term.ty)
                {
                    Type::Constructed(name.clone(), values.iter().map(|v| v.ty.clone()).collect())
                } else {
                    term.ty.clone()
                };
                self.op(tag, values, ty)
            }
            TermKind::TupleProj { tuple, idx } => {
                let tuple = self.source(owner, tuple)?;
                self.field(tuple, *idx)
            }
            TermKind::Coerce { inner, target_ty } => {
                let value = self.source(owner, inner)?;
                self.cast(value, target_ty)
            }
            TermKind::Index { array, index } => {
                let array = self.source(owner, array)?;
                let index = self.source(owner, index)?;
                self.index(array, index)
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let condition = self.source(owner, cond)?;
                let Some((yes, no)) = self.compiler.facts.branches(source) else {
                    return Err(error("missing branch scopes"));
                };
                self.branch(
                    scope,
                    condition,
                    |body| body.source(yes, then_branch),
                    |body| body.source(no, else_branch),
                    Some((yes, no)),
                )
            }
            TermKind::Loop { .. } => self.loop_(owner, source, term),
            TermKind::ArrayExpr(array) => self.array(owner, array, &term.ty),
            TermKind::Let { .. } => self.source(owner, term),
            TermKind::Soac(soac) => self.collective(owner, source, soac, &term.ty),
            TermKind::Var(_)
            | TermKind::Lambda(_)
            | TermKind::Closure(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Extern(_) => Err(error("callable used as a runtime scalar")),
        }
    }

    fn scalar(&mut self, scope: Value, term: TermId) -> Result<Typed, OptimizeError> {
        if let Some(value) = host::capture_scalar(self, term)? {
            return Ok(value);
        }
        let current = self.current()?;
        if let Some(values) = self.scalar_values.get(&term) {
            if let Some((_, value)) = values.iter().rev().find(|(block, _)| self.dominates(*block, current))
            {
                return Ok(value.clone());
            }
        }
        let program = self.compiler.program;
        let selected = &program.stage.selected;
        let expression = selected.values[term];
        let target = self.scalar_target(expression)?;
        let (name, fields) = selected.app(term)?;
        let Some(ty) = self.compiler.facts.ty(selected.values[fields[1]]).cloned() else {
            return Err(error("selected scalar type missing"));
        };
        // Placement is applied to the resulting pure instruction. Operand
        // evaluation still occurs at the current control/effect position.
        let value = match name {
            "ScalarLeaf" => {
                let source = selected.values[fields[2]];
                // A source boundary can have its own optimized root in this
                // context. Follow it unless it leads straight back to this leaf.
                if selected.roots.get(&(self.context, source)).is_some_and(|&root| root != term) {
                    self.value(scope, source)?
                } else {
                    self.original(scope, source)?
                }
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
                self.op_at(self.scalar_target(expression)?, tag, arguments, ty)?
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
                let ty = if let Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), _) =
                    types::strip_existentials(&ty)
                {
                    Type::Constructed(name.clone(), args.iter().map(|v| v.ty.clone()).collect())
                } else {
                    ty
                };
                self.op_at(self.scalar_target(expression)?, tag, args, ty)?
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
            "ScalarChoice" => {
                let condition = self.scalar(scope, fields[2])?;
                self.branch(
                    scope,
                    condition,
                    |body| body.scalar(scope, fields[3]),
                    |body| body.scalar(scope, fields[4]),
                    None,
                )?
            }
            _ => return Err(error(format!("unresolved selected constructor {name}"))),
        };
        let block = match value.value {
            ValueRef::Ssa(id) => self.builder.func().block_of_value(id).unwrap_or(target),
            _ => target,
        };
        self.scalar_values.entry(term).or_default().push((block, value.clone()));
        Ok(value)
    }

    fn scalar_target(&self, expression: Value) -> Result<BlockId, OptimizeError> {
        let current = self.current()?;
        let mut target = None;
        for (&scope, &block) in self.scopes.iter() {
            if self.compiler.facts.placement(self.context, expression, scope)
                && self.dominates(block, current)
                && target.is_none_or(|old| self.dominates(old, block))
            {
                target = Some(block);
            }
        }
        Ok(target.unwrap_or(current))
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

    pub fn call_value(
        &mut self,
        scope: Value,
        function: Value,
        mut arguments: Vec<Typed>,
    ) -> Result<Typed, OptimizeError> {
        let mut function = function;
        while let Some(actual) = self.compiler.facts.lookup("SsaCaptured", (function,)) {
            function = actual;
        }
        if let Some(&(term, owner)) = self.compiler.program.identities.origins.get(&function) {
            if let TermKind::Closure(closure) = &term.kind {
                for capture in &closure.captures {
                    arguments.push(self.source(owner, capture)?);
                }
            }
        }
        let Some(callee) = self.compiler.facts.callable(function) else {
            return Err(error(format!("unresolved call target in {scope:?}")));
        };
        self.call(callee, arguments)
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
        mut args: Vec<Typed>,
        ty: Type,
    ) -> Result<Typed, OptimizeError> {
        if matches!(tag,OpTag::Intrinsic{id,..} if id==catalog().known().array_with) {
            let Some(array) = args.first().cloned() else {
                return Err(error("array update has no destination"));
            };
            if array.ty.array_variant().is_some_and(types::is_array_variant_view) {
                let local = control::local_state_type(&array.ty);
                if local == array.ty {
                    return Err(error("functional storage update needs a bounded local value"));
                }
                args[0] = self.stored(array, &local)?;
            }
        }
        let ty = if matches!(tag,OpTag::Intrinsic{id,..} if id==catalog().known().storage_store || id==catalog().known().array_with || id==catalog().known().array_with_in_place)
        {
            let Some(destination) = args.first() else {
                return Err(error("array update has no destination"));
            };
            destination.ty.clone()
        } else {
            ty
        };
        // A selected lexical scope may precede a merge introduced while
        // emitting an operand. Keep the instruction after that operand.
        let available = args.iter().all(|arg| match arg.value {
            ValueRef::Ssa(value) => self
                .builder
                .func()
                .block_of_value(value)
                .is_none_or(|definition| self.dominates(definition, block)),
            _ => true,
        });
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
