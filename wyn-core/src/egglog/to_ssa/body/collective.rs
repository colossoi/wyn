//! Local collective loops and callback emission use the selected scalar context.
use super::{builder_error, error, Body, OptimizeError, Typed, Value};
use crate::egglog::to_ssa::host;
use crate::egglog::to_ssa::kernels;
use crate::egglog::to_ssa::kernels::element;
use crate::egglog::to_ssa::sizes;
use crate::host::SizeExpr;
use crate::op::BinaryOperator;
use crate::op::OpTag;
use crate::ssa::types::{InstKind, PlaceId};
use crate::tlc::data::{ExplicitCapturesPayload, ExplicitClosurePayload};
use crate::tlc::SoacOp;
use crate::tlc::TermKind;
use crate::types::{self, sized_array, Type, TypeExt, TypeName};
use crate::LookupMap;

type Soac = SoacOp<ExplicitClosurePayload, ExplicitCapturesPayload>;

impl<'source> Body<'_, '_, 'source> {
    pub(in crate::egglog::to_ssa) fn callback(
        &mut self,
        owner: Value,
        operation: Value,
        mut arguments: Vec<Typed>,
    ) -> Result<Typed, OptimizeError> {
        let Some(mut scope) = self.compiler.facts.callback(operation) else {
            return Err(error("collective callback is missing"));
        };
        let Some(&(_, Some(mut source))) = self.compiler.program.identities.scopes.get(&scope) else {
            return Err(error("callback source is missing"));
        };
        let identity = self.identity(scope, source)?;
        let callee = self.compiler.facts.callable(identity);
        let mut bindings = Vec::new();
        let mut host_arguments = vec![None; arguments.len()];
        if let Some(callee) = callee {
            let Some(value) = self.compiler.plan.source(operation) else {
                return Err(error("callback owner missing"));
            };
            let Some(&(term, parent)) = self.compiler.program.identities.origins.get(&value) else {
                return Err(error("callback source missing"));
            };
            let TermKind::Soac(soac) = &term.kind else {
                return Err(error("callback owner is not a collective"));
            };
            let callback = match soac {
                SoacOp::Map { lam, .. }
                | SoacOp::Scatter { lam, .. }
                | SoacOp::BucketScatter { lam, .. } => lam,
                SoacOp::Reduce { op, .. } | SoacOp::Scan { op, .. } | SoacOp::ReduceByIndex { op, .. } => {
                    op
                }
                SoacOp::Filter { pred, .. } => pred,
            };
            for (_, _, capture) in &callback.data.captures {
                let source = self.identity(parent, capture)?;
                host_arguments.push(host::expression(self.compiler, source, &self.host_arguments));
                arguments.push(self.source(parent, capture)?);
            }
            if arguments.iter().any(|argument| argument.ty.is_array()) {
                return self.call_value(owner, identity, arguments);
            }
            scope = callee;
            let Some(&(_, Some(body))) = self.compiler.program.identities.scopes.get(&scope) else {
                return Err(error("callback body missing"));
            };
            source = body;
        } else {
            for (formal, argument) in self.compiler.facts.captures(scope) {
                if !self.compiler.facts.callable(argument).is_some() {
                    bindings.push((formal, self.value(owner, argument)?));
                }
            }
        }
        for (i, argument) in arguments.into_iter().enumerate() {
            let Some(formal) = self.compiler.facts.parameter(scope, i as i64) else {
                return Err(error("callback parameter missing"));
            };
            let argument = if let Some(ty) = self.compiler.facts.source_type(formal).cloned() {
                self.cast(argument, &ty)?
            } else {
                argument
            };
            bindings.push((formal, argument));
        }
        let Some(context) = self.compiler.facts.context(scope) else {
            return Err(error("callback scalar context is missing"));
        };
        let old_context = std::mem::replace(&mut self.context, context);
        let old_values = self.values.checkpoint();
        let old_scopes = std::mem::take(&mut self.scopes);
        let old_scalar = std::mem::take(&mut self.scalar_values);
        let old_host = self.host_arguments.checkpoint();
        let old_bindings = callee.map(|_| std::mem::take(&mut self.host_scalar_bindings));
        self.scopes.insert(scope, self.current()?);
        for (i, (formal, value)) in bindings.into_iter().enumerate() {
            self.values.insert(formal, value);
            if callee.is_some() {
                self.host_arguments.remove(&formal);
                if let Some(host) = host_arguments[i].take() {
                    self.host_arguments.insert(formal, host);
                }
            }
        }
        let result = self.source(scope, source);
        self.context = old_context;
        self.values.restore(old_values);
        self.scopes = old_scopes;
        self.scalar_values = old_scalar;
        self.host_arguments.restore(old_host);
        if let Some(bindings) = old_bindings {
            self.host_scalar_bindings = bindings;
        }
        result
    }

    pub(super) fn collective(
        &mut self,
        scope: Value,
        source: Value,
        soac: &'source Soac,
        ty: &Type,
    ) -> Result<Typed, OptimizeError> {
        let Some(operation) = self.compiler.facts.operation(source) else {
            return Err(error("collective operation missing"));
        };
        if !self.compiler.facts.device_operation(operation) {
            return Err(error(format!(
                "host collective {operation:?} source {source:?} requires its scheduled kernel; stages {:?}; resource {:?}",
                self.compiler.plan.stages.iter().map(|s| (&s.phase, s.operation)).collect::<Vec<_>>(),
                self.compiler.plan.value_ref(source)
            )));
        }
        let inputs = self.compiler.facts.inputs(operation);
        if let SoacOp::Reduce { ne, .. } = soac {
            let Some(plan) = self.compiler.plan.group(operation) else {
                return Err(error("local reduction plan missing"));
            };
            let Some(owner) = self.compiler.plan.owner(plan) else {
                return Err(error("missing owner"));
            };
            let Some(domain) = self.compiler.plan.domain(owner) else {
                return Err(error("local reduction domain missing"));
            };
            let n = self.extent(scope, domain)?;
            let initial = self.source(scope, ne)?;
            let zero = self.literal("0", &types::i32())?;
            let one = self.literal("1", &types::i32())?;
            let Some(&(_, input)) = inputs.first() else {
                return Err(error("reduction input missing"));
            };
            let mut result = self.counted(zero, n, one, vec![initial], |body, index, mut state| {
                let value = element(body, scope, plan, input, index, &mut LookupMap::default())?;
                state.push(value);
                Ok(vec![body.callback(scope, operation, state)?])
            })?;
            return Ok(result.remove(0));
        }
        let mut arrays = Vec::new();
        for (_, input) in inputs {
            arrays.push(self.value(scope, input)?);
        }
        let Some(first) = arrays.first() else {
            return Err(error("collective has no input domain"));
        };
        let n = self.length(first.clone())?;
        let zero = self.literal("0", &types::i32())?;
        let one = self.literal("1", &types::i32())?;
        match soac {
            SoacOp::Reduce { .. } => unreachable!("local reductions return before array materialization"),
            SoacOp::Map { .. } | SoacOp::Scan { .. } => {
                let Some(element) = ty.elem_type() else {
                    return Err(error("local collective output is not an array"));
                };
                let count = self.local_capacity(source, first)?;
                let output_ty = sized_array(count.max(1), element.clone());
                let place = self.builder.new_place(output_ty.clone());
                self.builder
                    .push_void_inst(InstKind::Alloca {
                        elem_ty: output_ty.clone(),
                        result: place,
                    })
                    .map_err(builder_error)?;
                let initial =
                    if let SoacOp::Scan { ne, .. } = soac { vec![self.source(scope, ne)?] } else { vec![] };
                self.counted(zero, n, one, initial, |body, index, mut state| {
                    let mut arguments = Vec::new();
                    for array in &arrays {
                        arguments.push(body.index(array.clone(), index.clone())?);
                    }
                    let scan = !state.is_empty();
                    state.extend(arguments);
                    let value = body.callback(scope, operation, state)?;
                    body.local_store(place, element, index, value.clone())?;
                    Ok(if scan { vec![value] } else { vec![] })
                })?;
                let value = self
                    .builder
                    .push_inst(InstKind::Load { place }, output_ty.clone())
                    .map_err(builder_error)?;
                Ok(Typed {
                    value: value.into(),
                    ty: output_ty,
                })
            }
            SoacOp::Filter { .. } => {
                let Some(element) = first.ty.elem_type().cloned() else {
                    return Err(error("local filter input has no element"));
                };
                let capacity = self.local_capacity(source, first)?.max(1);
                let array_ty = sized_array(capacity, element.clone());
                let place = self.builder.new_place(array_ty.clone());
                self.builder
                    .push_void_inst(InstKind::Alloca {
                        elem_ty: array_ty.clone(),
                        result: place,
                    })
                    .map_err(builder_error)?;
                let count =
                    self.counted(zero.clone(), n, one.clone(), vec![zero], |body, index, state| {
                        let value = body.index(arrays[0].clone(), index)?;
                        let keep = body.callback(scope, operation, vec![value.clone()])?;
                        let next = body.branch(
                            scope,
                            keep,
                            |body| {
                                body.local_store(place, &element, state[0].clone(), value)?;
                                body.binary(BinaryOperator::Add, state[0].clone(), one.clone())
                            },
                            |_| Ok(state[0].clone()),
                            None,
                        )?;
                        Ok(vec![next])
                    })?;
                let value = self
                    .builder
                    .push_inst(InstKind::Load { place }, array_ty.clone())
                    .map_err(builder_error)?;
                let data = Typed {
                    value: value.into(),
                    ty: array_ty,
                };
                let bounded = types::make_array1(
                    element,
                    types::array_variant_bounded(),
                    Type::Constructed(TypeName::Size(capacity), vec![]),
                    types::no_buffer(),
                );
                self.op(OpTag::Tuple(2), vec![data, count[0].clone()], bounded)
            }
            SoacOp::Scatter { .. } | SoacOp::ReduceByIndex { .. } => {
                let Some(source) = self.compiler.facts.destination(operation) else {
                    return Err(error("missing destination"));
                };
                let output = self.value(scope, source)?;
                self.index_place(output.clone(), zero.clone())?;
                self.counted(zero, n, one, vec![], |body, index, _| {
                    let mut args = Vec::new();
                    for array in &arrays {
                        args.push(body.index(array.clone(), index.clone())?);
                    }
                    let (key, value) = if matches!(soac, SoacOp::Scatter { .. }) {
                        let pair = body.callback(scope, operation, args)?;
                        (body.field(pair.clone(), 0)?, body.field(pair, 1)?)
                    } else {
                        (args.remove(0), args.remove(0))
                    };
                    let zero = body.literal("0", &key.ty)?;
                    let count = body.length(output.clone())?;
                    let positive = body.binary(BinaryOperator::GreaterEqual, key.clone(), zero)?;
                    let below = body.binary(BinaryOperator::Less, key.clone(), count)?;
                    let valid = body.binary(BinaryOperator::LogicalAnd, positive, below)?;
                    body.when(valid, |body| {
                        let value = if matches!(soac, SoacOp::ReduceByIndex { .. }) {
                            let old = body.index(output.clone(), key.clone())?;
                            body.callback(scope, operation, vec![old, value])?
                        } else {
                            value
                        };
                        let (place, ty) = body.index_place(output.clone(), key)?;
                        let value = body.cast(value, &ty)?;
                        body.builder
                            .push_void_inst(InstKind::Store {
                                place,
                                value: value.value,
                            })
                            .map_err(builder_error)?;
                        Ok(())
                    })?;
                    Ok(vec![])
                })?;
                self.updated(output)
            }
            SoacOp::BucketScatter {
                input_dimensions,
                domain_rank,
                ..
            } => {
                let Some(destination) = self.compiler.facts.destination(operation) else {
                    return Err(error("missing destination"));
                };
                let output = self.value(scope, destination)?;
                let Some(Type::Constructed(TypeName::Size(count), _)) = output.ty.array_size() else {
                    return Err(error("local bucket count must have a static capacity"));
                };
                let count = *count;
                let uint = Type::Constructed(TypeName::UInt(32), vec![]);
                let zero = self.literal("0", &uint)?;
                let counts = self.op(
                    OpTag::ArrayLit(count),
                    vec![zero.clone(); count],
                    sized_array(count, uint.clone()),
                )?;
                let overflow = self.op(OpTag::ArrayLit(1), vec![zero], sized_array(1, uint))?;
                let zero = self.literal("0", &types::i32())?;
                for array in [&output, &counts, &overflow] {
                    self.index_place(array.clone(), zero.clone())?;
                }
                let Some(plan) = self.compiler.plan.group(operation) else {
                    return Err(error("missing group"));
                };
                kernels::bucket_updates(
                    self,
                    scope,
                    operation,
                    plan,
                    output.clone(),
                    counts.clone(),
                    overflow.clone(),
                    input_dimensions,
                    *domain_rank,
                    true,
                    1,
                )?;
                let output = self.updated(output)?;
                let counts = self.updated(counts)?;
                let index = self.literal("0", &types::i32())?;
                let overflow = self.index(overflow, index)?;
                self.tuple(vec![output, counts, overflow])
            }
        }
    }

    fn local_capacity(&self, source: Value, input: &Typed) -> Result<usize, OptimizeError> {
        for ty in self.compiler.facts.source_type(source).into_iter().chain(std::iter::once(&input.ty)) {
            if let Some(Type::Constructed(TypeName::Size(n), _)) = ty.array_size() {
                return Ok(*n);
            }
        }
        if let Some(operation) = self.compiler.facts.operation(source) {
            if let Some(plan) = self.compiler.plan.group(operation) {
                let Some(owner) = self.compiler.plan.owner(plan) else {
                    return Err(error("missing owner"));
                };
                if let Some(extent) = self.compiler.plan.domain(owner) {
                    if let Ok(SizeExpr::Integer(n)) = sizes::extent(self.compiler, extent) {
                        return usize::try_from(n).map_err(|_| error("negative local array capacity"));
                    }
                }
            }
        }
        Err(error("local collective has no static capacity"))
    }

    fn local_store(
        &mut self,
        array: PlaceId,
        ty: &Type,
        index: Typed,
        value: Typed,
    ) -> Result<(), OptimizeError> {
        let place = self.builder.new_place(ty.clone());
        self.builder
            .push_void_inst(InstKind::PlaceIndex {
                place: array,
                index: index.value,
                result: place,
            })
            .map_err(builder_error)?;
        let value = self.cast(value, ty)?;
        self.builder
            .push_void_inst(InstKind::Store {
                place,
                value: value.value,
            })
            .map_err(builder_error)?;
        Ok(())
    }
}
