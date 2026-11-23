//! Emit statements in lexical order; descriptor arithmetic comes from bindings.
use super::{builder_error, error, Body, Collection, CollectionShape, OptimizeError, Typed, Value};
use crate::builtins::catalog;
use crate::egglog::dataflow::ir::{Bounds, Code};
use crate::egglog::execution::StorageReuse;
use crate::egglog::graph::CollectiveInput;
use crate::egglog::graph::Formula;
use crate::egglog::payload::Payload;
use crate::egglog::query::Query;
use crate::egglog::to_ssa::{host, runtime_binding, same_value_structure};
use crate::host::Expr;
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::ssa::layout::{storage_elem_stride, storage_value_type};
use crate::ssa::types::InstKind;
use crate::types::{self, Type, TypeExt, TypeName};
use crate::ResourceAccess;

fn collection_type(declared: &Type, element: &Type, axes: usize) -> Result<Type, OptimizeError> {
    if axes == 0 {
        if let (Some(Type::Constructed(TypeName::Size(size), _)), Some(declared_item), Some(actual_item)) =
            (declared.array_size(), declared.elem_type(), element.elem_type())
        {
            return Ok(types::sized_array(
                *size,
                collection_type(declared_item, actual_item, 0)?,
            ));
        }
        return Ok(element.clone());
    }
    if let (Type::Constructed(TypeName::Tuple(_), fields), Type::Constructed(TypeName::Tuple(_), values)) =
        (declared, element)
    {
        if fields.len() != values.len() {
            return Err(error("collection product arity mismatch"));
        }
        return Ok(types::tuple(
            fields
                .iter()
                .zip(values)
                .map(|(field, value)| collection_type(field, value, axes))
                .collect::<Result<Vec<_>, _>>()?,
        ));
    }
    let Some(child) = declared.elem_type() else {
        return Err(error("collection rank exceeds declared array rank"));
    };
    let Some(size) = declared.array_size() else {
        return Err(error("collection extent missing"));
    };
    Ok(types::make_array1(
        collection_type(child, element, axes - 1)?,
        types::array_variant_composite(),
        size.clone(),
        types::no_buffer(),
    ))
}

impl Body<'_, '_, '_> {
    pub fn named(&self, name: Value) -> Result<Typed, OptimizeError> {
        let Some(value) =
            self.values.get(&name).or_else(|| self.collections.get(&name).and_then(Collection::value))
        else {
            return Err(error(format!("unbound emitted value {name:?}")));
        };
        Ok(value.clone())
    }

    pub fn atom(&mut self, name: Value) -> Result<Typed, OptimizeError> {
        self.named(name)
    }

    pub fn index_expression(&mut self, expression: Value) -> Result<Typed, OptimizeError> {
        let (kind, fields) = self.query().application(
            expression,
            &[
                "IndexConstant",
                "IndexValue",
                "IndexAdd",
                "IndexSub",
                "IndexMul",
                "IndexDiv",
                "IndexMod",
                "IndexMin",
                "IndexMax",
            ],
        )?;
        let ty = types::i32();
        if kind == "IndexConstant" {
            let n = self.compiler.program.graph.value_to_base::<i64>(fields[0]);
            i32::try_from(n).map_err(|_| error("memory index exceeds the shader address range"))?;
            return self.literal(n, &ty);
        }
        if kind == "IndexValue" {
            let value = self.named(fields[0])?;
            return self.cast(value, &ty);
        }
        let a = self.index_expression(fields[0])?;
        let b = self.index_expression(fields[1])?;
        if matches!(kind, "IndexMin" | "IndexMax") {
            if let (Some(a), Some(b)) = (self.integer_literal(&a), self.integer_literal(&b)) {
                return self.literal(if kind == "IndexMin" { a.min(b) } else { a.max(b) }, &ty);
            }
            let compare = if kind == "IndexMin" { BinaryOperator::Less } else { BinaryOperator::Greater };
            let condition = self.binary(compare, a.clone(), b.clone())?;
            return self.select(condition, a, b);
        }
        let operator = match kind {
            "IndexAdd" => BinaryOperator::Add,
            "IndexSub" => BinaryOperator::Subtract,
            "IndexMul" => BinaryOperator::Multiply,
            "IndexDiv" => BinaryOperator::Divide,
            "IndexMod" => BinaryOperator::Remainder,
            _ => unreachable!(),
        };
        self.binary(operator, a, b)
    }

    pub fn operand(&mut self, operand: Value) -> Result<Option<Typed>, OptimizeError> {
        let Some(name) = self.query().value_name(operand)? else {
            return Ok(None);
        };
        self.named(name).map(Some)
    }

    pub fn yield_values(&mut self, names: &[Value]) -> Result<Vec<Typed>, OptimizeError> {
        let mut values = Vec::new();
        for &name in names {
            if let Some(value) = self.operand(name)? {
                values.push(self.snapshot(value)?);
            }
        }
        Ok(values)
    }

    pub fn results(&mut self, region: Value) -> Result<Vec<Typed>, OptimizeError> {
        self.yield_values(&self.query().0.yield_values(region))
    }

    pub fn block(&mut self, region: Value) -> Result<Vec<Typed>, OptimizeError> {
        self.statements(&self.query().operations(region)?)?;
        self.results(region)
    }

    pub fn statements(&mut self, statements: &[Value]) -> Result<(), OptimizeError> {
        for &statement in statements {
            self.statement(statement)?;
        }
        Ok(())
    }

    pub(in crate::egglog::to_ssa) fn bind(
        &mut self,
        pattern: &[Value],
        values: Vec<Typed>,
    ) -> Result<(), OptimizeError> {
        let mut values = values.into_iter();
        for &binding in pattern {
            if let Some((name, ty)) = runtime_binding(self.query(), binding)? {
                let Some(value) = values.next() else {
                    return Err(error("statement result arity is too small"));
                };
                let ty = self.collection_output_type(name, ty);
                let value =
                    if same_value_structure(&value.ty, &ty) { value } else { self.cast(value, &ty)? };
                self.values.insert(name, value);
            }
        }
        if values.next().is_some() {
            return Err(error("statement result arity is too large"));
        }
        Ok(())
    }

    pub fn lambda(&mut self, region: Value, arguments: Vec<Typed>) -> Result<Vec<Typed>, OptimizeError> {
        let checkpoint = self.values.checkpoint();
        self.bind(&self.query().0.parameters(region), arguments)?;
        let result = self.block(region);
        self.values.restore(checkpoint);
        result
    }

    pub(in crate::egglog::to_ssa) fn unpack(&mut self, value: Typed) -> Result<Vec<Typed>, OptimizeError> {
        if value.ty == types::unit() {
            return Ok(Vec::new());
        }
        let Type::Constructed(TypeName::Tuple(_), types) = &value.ty else {
            return Ok(vec![value]);
        };
        (0..types.len()).map(|i| self.field(value.clone(), i)).collect()
    }

    pub(in crate::egglog::to_ssa) fn pack_results(
        &mut self,
        mut values: Vec<Typed>,
    ) -> Result<Typed, OptimizeError> {
        if values.len() == 1 {
            return Ok(values.remove(0));
        }
        self.tuple(values)
    }

    pub(in crate::egglog::to_ssa) fn unpack_results(
        &mut self,
        value: Typed,
        count: usize,
    ) -> Result<Vec<Typed>, OptimizeError> {
        if count == 1 {
            return Ok(vec![value]);
        }
        self.unpack(value)
    }

    pub(in crate::egglog::to_ssa) fn statement(&mut self, statement: Value) -> Result<(), OptimizeError> {
        let span = self.query().0.span(statement);
        let previous = std::mem::replace(&mut self.span, span);
        let result = self.emit_statement(statement);
        self.span = previous;
        result
    }

    fn emit_statement(&mut self, statement: Value) -> Result<(), OptimizeError> {
        host::remember(
            Query(&self.compiler.program.graph),
            self.function,
            &mut self.host_values,
            statement,
        )?;
        let query = self.query();
        let pattern = query.0.outputs(statement);
        let operands = query.0.inputs(statement);
        let inputs = operands.iter().map(|&name| query.name(name)).collect::<Result<Vec<_>, _>>()?;
        let code = query.0.code(statement)?;
        if matches!(code, Code::Parallel)
            && self.compiler.execution.is_device(statement)
            && self.active_dispatch != Some(statement)
        {
            return Err(error("device parallel region requires a dispatch entry"));
        }
        let values = match code {
            Code::Constant { ty, bits } => vec![if ty == types::unit() {
                self.op(OpTag::Unit, vec![], ty)?
            } else {
                self.literal(bits, &ty)?
            }],
            Code::Product => {
                let values = inputs.iter().map(|&name| self.named(name)).collect::<Result<Vec<_>, _>>()?;
                vec![self.tuple(values)?]
            }
            Code::Project { index } => {
                let value = self.named(inputs[0])?;
                vec![self.field(value, index)?]
            }
            Code::IndexValue => {
                vec![self.index_expression(self.query().0.formula(statement, Formula::Index)?)?]
            }
            Code::Copy => {
                let source = self.named(inputs[0])?;
                if source.ty.array_variant().is_some_and(types::is_array_variant_composite) {
                    if let [binding] = pattern.as_slice() {
                        if let Some((output, _)) = runtime_binding(self.query(), *binding)? {
                            if self.compiler.execution.can_snapshot(output) {
                                let value = self.snapshot(source)?;
                                return self.bind(&pattern, vec![value]);
                            }
                        }
                    }
                }
                if let [binding] = pattern.as_slice() {
                    if let Some((output, ty)) = runtime_binding(self.query(), *binding)? {
                        if let Some(storage) = self.reuse_storage(output, &ty)? {
                            return self.bind(&pattern, vec![storage]);
                        }
                    }
                }
                let Some(element) = source.ty.elem_type().cloned() else {
                    return Err(error("copy source has no element"));
                };
                let known_count = self.compiler.execution.storage.static_count.get(&inputs[0]);
                let count = if let Some(&count) = known_count {
                    self.literal(count, &types::i32())?
                } else {
                    self.array_length(source.clone())?
                };
                let inline = source.ty.array_variant().is_some_and(types::is_array_variant_composite)
                    && matches!(
                        source.ty.array_size(),
                        Some(Type::Constructed(TypeName::Size(_), _))
                    );
                let destination = if self.active_dispatch.is_none() && !inline {
                    let length = self.host_array_length(&source)?;
                    let Some(stride) = storage_elem_stride(&storage_value_type(&element)) else {
                        return Err(error("copy element has no stride"));
                    };
                    let bytes =
                        Expr::Multiply(Box::new(length.clone()), Box::new(Expr::Integer(stride.into())));
                    let (_, destination) = self.device_buffer(&element, bytes, count.clone())?;
                    self.set_shape(&destination, vec![length])?;
                    destination
                } else {
                    let capacity = self
                        .integer_literal(&count)
                        .and_then(|n| usize::try_from(n).ok())
                        .or_else(|| match source.ty.array_size() {
                            Some(Type::Constructed(TypeName::Size(size), _)) => Some(*size),
                            _ => None,
                        });
                    let capacity = capacity.or(self
                        .compiler
                        .execution
                        .storage
                        .outer_capacity
                        .get(&inputs[0])
                        .and_then(|&value| usize::try_from(value).ok()));
                    let Some(capacity) = capacity else {
                        return Err(error("local copy needs a capacity bound"));
                    };
                    self.local_array_with_length(types::sized_array(capacity, element), count.clone())?
                };
                self.counted(
                    Self::number(0),
                    count,
                    Self::number(1),
                    vec![],
                    |body, index, _| {
                        let value = body.index(source, index.clone())?;
                        body.store(destination.clone(), index, value)?;
                        Ok(vec![])
                    },
                )?;
                vec![destination]
            }
            Code::Scratch => vec![self.scratch(statement, &inputs, &pattern)?],
            Code::Length { axis } => {
                let array = self.named(inputs[0])?;
                let host_extent = if axis == 0 {
                    self.host_array_length(&array).ok()
                } else {
                    self.host_shape(&array).and_then(|shape| shape.into_iter().nth(axis))
                };
                if let [binding] = pattern.as_slice() {
                    if let Some((name, _)) = runtime_binding(self.query(), *binding)? {
                        if let Some(length) = host_extent.as_ref().and_then(host::scalar_extent) {
                            self.host_values.insert(name, length);
                        }
                    }
                }
                if axis == 0 {
                    vec![self.array_length(array)?]
                } else {
                    let Some(dimension) = self.view_dimensions(array)?.into_iter().nth(axis) else {
                        return Err(error("logical length axis is outside the array rank"));
                    };
                    vec![dimension]
                }
            }
            Code::Index | Code::TraversalInput => {
                let mut array = self.named(inputs[0])?;
                for &name in &inputs[1..] {
                    let index = self.named(name)?;
                    array = self.index(array, index)?;
                }
                vec![array]
            }
            Code::Range => {
                let extent = self.query().0.formula(statement, Formula::Index)?;
                let start = self.atom(inputs[0])?;
                let step = self.atom(inputs[1])?;
                let length = self.index_expression(extent)?;
                let length = if let Some(count) = self.integer_literal(&length) {
                    self.literal(count.max(0), &length.ty)?
                } else {
                    let zero = self.literal(0, &length.ty)?;
                    let positive = self.binary(BinaryOperator::Greater, length.clone(), zero.clone())?;
                    self.select(positive, length, zero)?
                };
                let length = self.cast(length, &start.ty)?;
                let ty = types::make_array1(
                    start.ty.clone(),
                    Type::Constructed(TypeName::ArrayVariantVirtual, vec![]),
                    Type::Constructed(TypeName::SizePlaceholder, vec![]),
                    types::no_buffer(),
                );
                let value = self.op(OpTag::Tuple, vec![start, step, length], ty)?;
                if let Some(length) = host::scalar_index(self.query(), &self.host_values, extent)? {
                    self.set_shape(
                        &value,
                        vec![Expr::Max(
                            Box::new(Expr::Scalar(length)),
                            Box::new(Expr::Integer(0)),
                        )],
                    )?;
                }
                vec![value]
            }
            Code::ArrayLiteral => {
                let mut elements =
                    inputs.iter().copied().map(|atom| self.atom(atom)).collect::<Result<Vec<_>, _>>()?;
                let [binding] = pattern.as_slice() else {
                    return Err(error("array literal must bind one value"));
                };
                let Some((_, declared)) = runtime_binding(self.query(), *binding)? else {
                    return Err(error("array literal has no value binding"));
                };
                if let Some(first) = elements.first() {
                    if first.ty.is_array() && elements.iter().any(|value| value.ty != first.ty) {
                        elements = self.coalesce_arrays(elements)?;
                    }
                }
                let element = elements
                    .first()
                    .map(|element| element.ty.clone())
                    .or_else(|| declared.elem_type().cloned());
                let Some(element) = element else {
                    return Err(error("array literal has no element type"));
                };
                let ty = types::sized_array(elements.len(), element);
                vec![self.op(OpTag::ArrayLit, elements, ty)?]
            }
            Code::Scalar(payload) => {
                let args = inputs.iter().map(|&name| self.named(name)).collect::<Result<Vec<_>, _>>()?;
                let [binding] = pattern.as_slice() else {
                    return Err(error("scalar instruction must bind one result"));
                };
                let Some((_, ty)) = runtime_binding(self.query(), *binding)? else {
                    return Err(error("scalar result is a capability"));
                };
                vec![self.primitive(payload, args, ty)?]
            }
            Code::Branch => {
                let condition = self.named(inputs[0])?;
                let yes = self.query().0.control_region(statement, 0_i64)?;
                let no = self.query().0.control_region(statement, 1_i64)?;
                let types = pattern
                    .iter()
                    .map(|&name| runtime_binding(self.query(), name))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .map(|(_, ty)| ty)
                    .collect::<Vec<_>>();
                if self.compiler.execution.speculation.contains(&statement) {
                    let a = self.block(yes)?;
                    let b = self.block(no)?;
                    if a.len() != types.len() || b.len() != types.len() {
                        return Err(error("selected branch result arity mismatch"));
                    }
                    let mut values = Vec::new();
                    for ((a, b), ty) in a.into_iter().zip(b).zip(&types) {
                        let a = self.canonical_value(a, ty)?;
                        let b = self.canonical_value(b, ty)?;
                        values.push(self.select(condition.clone(), a, b)?);
                    }
                    return self.bind(&pattern, values);
                }
                let ty = types::tuple(types);
                let result = self.branch(
                    condition,
                    |body| {
                        let values = body.block(yes)?;
                        let value = body.tuple(values)?;
                        body.canonical_value(value, &ty)
                    },
                    |body| {
                        let values = body.block(no)?;
                        let value = body.tuple(values)?;
                        body.canonical_value(value, &ty)
                    },
                )?;
                self.unpack(result)?
            }
            Code::CountedLoop | Code::WhileLoop => {
                let region = self.query().0.control_region(statement, 0_i64)?;
                let mut params = Vec::new();
                let mut initial = Vec::new();
                for slot in 0..self.query().0.output_count(statement) {
                    let parameter = self.query().0.parameter(region, slot)?;
                    if let Some((name, ty)) = runtime_binding(self.query(), parameter)? {
                        let argument = self.query().0.argument(statement, 0_i64, slot)?;
                        let Some(value) = self.operand(argument)? else {
                            return Err(error("loop carry has no runtime value"));
                        };
                        let mut value = self.canonical_value(value, &ty)?;
                        if self.compiler.execution.storage.fixed_loop.contains(&(statement, slot)) {
                            value = self.fixed_loop_state(value, self.query().value_name(argument)?)?;
                        }
                        params.push(name);
                        initial.push(value);
                    }
                }
                let install = |body: &mut Self, state: &[Typed]| {
                    for (&name, value) in params.iter().zip(state) {
                        body.values.insert(name, value.clone());
                    }
                };
                if matches!(code, Code::CountedLoop) {
                    let index = self.query().loop_parameter(statement)?;
                    let count =
                        self.index_expression(self.query().0.formula(statement, Formula::LoopCount)?)?;
                    self.counted(
                        Self::number(0),
                        count,
                        Self::number(1),
                        initial,
                        |body, ordinal, state| {
                            install(body, &state);
                            let value = body.cast(ordinal, &types::i32())?;
                            body.values.insert(index, value);
                            body.block(region)
                        },
                    )?
                } else if self.compiler.execution.workgroup(statement).uniform_loop {
                    let condition = self.query().loop_parameter(statement)?;
                    self.workgroup_while(region, condition, &params, initial)?
                } else {
                    let condition = self.query().loop_parameter(statement)?;
                    self.while_loop(
                        initial,
                        |body, state| {
                            install(body, state);
                            body.named(condition)
                        },
                        |body, state| {
                            install(body, &state);
                            body.block(region)
                        },
                    )?
                }
            }
            Code::Call => {
                let mut args = Vec::new();
                for &name in &operands {
                    if let Some(value) = self.operand(name)? {
                        args.push(value);
                    }
                }
                let target = self.query().0.call_target(statement)?;
                let (id, parameters, ty) = self.compiler.function(target)?;
                if args.len() != parameters.len() {
                    return Err(error("shader call arity differs from its declared signature"));
                }
                let args = args
                    .into_iter()
                    .zip(parameters)
                    .map(|(argument, parameter)| self.cast(argument, &parameter))
                    .collect::<Result<Vec<_>, _>>()?;
                let result = self.op(OpTag::Call(id), args, ty)?;
                if pattern.len() == 1 {
                    vec![result]
                } else {
                    self.unpack(result)?
                }
            }
            Code::WithAcc => {
                let region = self.query().0.control_region(statement, 0_i64)?;
                if self.compiler.execution.workgroup(statement).accumulator {
                    let values = self.workgroup_accumulator(statement, region)?;
                    let values = self.scope_outputs(&pattern, values)?;
                    return self.bind(&pattern, values);
                }
                self.statements(&self.query().operations(region)?)?;
                let mut values = Vec::new();
                for group in 0..self.query().0.accumulator_group_count(statement) {
                    let count =
                        self.query().0.collective_input_count(statement, group, CollectiveInput::Backing);
                    for slot in 0..count {
                        let name = self.query().0.collective_input(
                            statement,
                            group,
                            CollectiveInput::Backing,
                            slot,
                        )?;
                        values.push(self.named(name)?);
                    }
                }
                values.extend(self.results(region)?);
                self.scope_outputs(&pattern, values)?
            }
            Code::BucketScatter { coordinates, depth } => {
                vec![self.bucket_scatter(statement, &coordinates, depth)?]
            }
            Code::View => {
                let mut array = self.named(inputs[0])?;
                for slice in self.query().vector(self.query().0.formula(statement, Formula::Slices)?)? {
                    if let Some(at) = self.query().enode("At", slice)? {
                        let index = self.index_expression(at[0])?;
                        array = self.index(array, index)?;
                    } else {
                        let fields = self.query().fields("Slice", slice)?;
                        let start = self.index_expression(fields[0])?;
                        let count = self.index_expression(fields[1])?;
                        let stride = self.index_expression(fields[2])?;
                        let mut shape = self.host_shape(&array).unwrap_or_default();
                        array = self.slice_array(array, start, count, stride)?;
                        if let Some(length) =
                            host::scalar_index(self.query(), &self.host_values, fields[1])?
                        {
                            let length = Expr::Scalar(length);
                            if shape.is_empty() {
                                shape.push(length);
                            } else {
                                shape[0] = length;
                            }
                            self.set_shape(&array, shape)?;
                        }
                    }
                }
                vec![array]
            }
            Code::Update { bounds } => {
                let slices = self.query().vector(self.query().0.formula(statement, Formula::Slices)?)?;
                let value = self.atom(inputs[1])?;
                let source = self.named(inputs[0])?;
                if source.ty.array_variant().is_some_and(types::is_array_variant_composite) {
                    if let [slice] = slices.as_slice() {
                        if let Some(at) = self.query().enode("At", *slice)? {
                            let index = self.index_expression(at[0])?;
                            let source = self.snapshot(source)?;
                            let ty = source.ty.clone();
                            // Delay updates until use without speculating them out of their branch.
                            let scope = self.current()?;
                            let result = self.builder.func_mut().append_floating_inst(
                                scope,
                                InstKind::Op {
                                    tag: OpTag::Intrinsic {
                                        id: catalog().known().array_with,
                                        overload_idx: 0,
                                    },
                                    operands: vec![source.value, index.value, value.value],
                                },
                                ty.clone(),
                            );
                            return self.bind(
                                &pattern,
                                vec![Typed {
                                    value: result.into(),
                                    ty,
                                }],
                            );
                        }
                    }
                }
                self.update_slices(inputs[0], &slices, vec![], value, bounds == Bounds::Proven)?;
                let value = self.named(inputs[0])?;
                vec![self.snapshot(value)?]
            }
            Code::Accumulate { .. } => self.accumulate(statement)?,
            Code::Parallel => self.parallel(statement, &pattern)?,
        };
        self.bind(&pattern, values)
    }

    pub(in crate::egglog::to_ssa) fn scope_outputs(
        &mut self,
        pattern: &[Value],
        values: Vec<Typed>,
    ) -> Result<Vec<Typed>, OptimizeError> {
        let mut values = values.into_iter().peekable();
        let mut outputs = Vec::new();
        for &binding in pattern {
            if let Some((_, ty)) = runtime_binding(self.query(), binding)? {
                outputs.push(self.scope_value(&ty, &mut values)?);
            }
        }
        if values.next().is_some() {
            return Err(error("accumulator scope returned too many values"));
        }
        Ok(outputs)
    }

    fn scope_value(
        &mut self,
        ty: &Type,
        values: &mut std::iter::Peekable<std::vec::IntoIter<Typed>>,
    ) -> Result<Typed, OptimizeError> {
        if values.peek().is_some_and(|value| same_value_structure(&value.ty, ty)) {
            let Some(value) = values.next() else {
                return Err(error("accumulator result disappeared"));
            };
            return Ok(value);
        }
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = ty {
            let values = fields
                .iter()
                .map(|field| self.scope_value(field, values))
                .collect::<Result<Vec<_>, _>>()?;
            return self.tuple(values);
        }
        let Some(value) = values.next() else {
            return Err(error("accumulator scope returned too few values"));
        };
        self.canonical_value(value, ty)
    }

    fn primitive(
        &mut self,
        payload: Payload,
        arguments: Vec<Typed>,
        ty: Type,
    ) -> Result<Typed, OptimizeError> {
        if payload == Payload::IndexCast {
            let [value]: [Typed; 1] = arguments.try_into().map_err(|_| error("index conversion arity"))?;
            return self.cast(value, &ty);
        }
        if payload == Payload::Select {
            let [no, yes, condition] = arguments.as_slice() else {
                return Err(error("select requires a condition and two values"));
            };
            return self.select(condition.clone(), yes.clone(), no.clone());
        }
        let Some(tag) = payload.instruction() else {
            return Err(error("scalar payload has no ordinary instruction"));
        };
        if matches!(tag, OpTag::Intrinsic { id, .. } if id == catalog().known().record_with) {
            let [base, index, update]: [Typed; 3] =
                arguments.try_into().map_err(|_| error("record update arity"))?;
            let field = self
                .integer_literal(&index)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| error("record update requires a constant field"))?;
            let Type::Constructed(_, logical_fields) = &ty else {
                return Err(error("record update requires a product"));
            };
            let logical =
                logical_fields.get(field).ok_or_else(|| error("record update field out of bounds"))?;
            let update = self.canonical_value(update, logical)?;
            let base = self.canonical_value(base, &ty)?;
            let Type::Constructed(name, mut fields) = base.ty.clone() else {
                return Err(error("record update requires a physical product"));
            };
            let slot =
                fields.get_mut(field).ok_or_else(|| error("physical record update field out of bounds"))?;
            *slot = update.ty.clone();
            return self.op(tag, vec![base, index, update], Type::Constructed(name, fields));
        }
        self.op(tag, arguments, ty)
    }
}

impl Body<'_, '_, '_> {
    pub fn read_array(&mut self, name: Value, indices: &[Typed]) -> Result<Typed, OptimizeError> {
        if let Some(mut array) = self
            .values
            .get(&name)
            .filter(|value| {
                value.ty.is_array()
                    || matches!(
                        value.ty,
                        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), _)
                    )
            })
            .cloned()
        {
            for index in indices {
                array = self.index(array, index.clone())?;
            }
            return Ok(array);
        }
        Err(error("array value has no SSA representation"))
    }

    pub fn write_array(
        &mut self,
        name: Value,
        indices: &[Typed],
        value: Typed,
    ) -> Result<(), OptimizeError> {
        if matches!(self.collections.get(&name), Some(Collection::Pending { .. })) {
            let Some(Collection::Pending { ty, shape, block }) = self.collections.remove(&name) else {
                return Err(error("pending collection missing"));
            };
            let ty = collection_type(&ty, &value.ty, indices.len())?;
            let storage = if let Some(storage) = self.reuse_storage(name, &ty)? {
                storage
            } else {
                let capacity = self.compiler.execution.storage.outer_capacity.get(&name).copied();
                self.collection_storage(&ty, &shape, capacity, block)?
            };
            self.collections.insert(name, Collection::Allocated(storage));
        }
        if let Some(array) = self
            .values
            .get(&name)
            .or_else(|| self.collections.get(&name).and_then(Collection::value))
            .filter(|value| {
                value.ty.is_array()
                    || matches!(
                        value.ty,
                        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), _)
                    )
            })
            .cloned()
        {
            return self.write_array_value(array, indices, value);
        }
        Err(error("array destination has no SSA representation"))
    }

    fn scratch(
        &mut self,
        statement: Value,
        inputs: &[Value],
        pattern: &[Value],
    ) -> Result<Typed, OptimizeError> {
        let (ty, count, length, capacity) = if let Some(shape) =
            self.query().0.formula_at(statement, Formula::Extents)
        {
            let [extent] = self.query().vector(shape)?[..] else {
                return Err(error("scratch allocation requires one outer extent"));
            };
            let [binding] = pattern else {
                return Err(error("scratch allocation must bind one array"));
            };
            let Some((name, ty)) = runtime_binding(self.query(), *binding)? else {
                return Err(error("scratch allocation has no value binding"));
            };
            let count = self.index_expression(extent)?;
            let nonnegative = self.binary(BinaryOperator::GreaterEqual, count.clone(), Self::number(0))?;
            self.require_condition(nonnegative)?;
            let length = if self.active_dispatch.is_none() {
                host::scalar_index(self.query(), &self.host_values, extent)?.map(Expr::Scalar)
            } else {
                None
            };
            let capacity =
                self.compiler.execution.storage.static_count.get(&name).copied().filter(|size| *size >= 0);
            (ty, count, length, capacity)
        } else {
            let [source] = inputs else {
                return Err(error("scratch allocation needs an array shape"));
            };
            let source = self.named(*source)?;
            let count = self.array_length(source.clone())?;
            let length =
                if self.active_dispatch.is_none() { Some(self.host_array_length(&source)?) } else { None };
            (source.ty, count, length, None)
        };
        let block = self.current()?;
        self.collection_storage(&ty, &CollectionShape::outer(count, length), capacity, block)
    }

    pub(in crate::egglog::to_ssa) fn reuse_storage(
        &mut self,
        output: Value,
        ty: &Type,
    ) -> Result<Option<Typed>, OptimizeError> {
        let element = ty.elem_type().map(storage_value_type);
        match self.compiler.execution.storage_reuse(output) {
            Some(StorageReuse::Input { name, binding }) => {
                let Some(array) = self.values.get(name).cloned() else {
                    return Err(error("planned input backing is unavailable"));
                };
                if types::array_view_buffer(&array.ty) != Some(*binding)
                    || array.ty.elem_type().map(storage_value_type) != element
                {
                    return Err(error(
                        "planned input backing has a different storage type or binding",
                    ));
                }
                if !self.reused_buffers.insert(*binding) {
                    return Err(error("planned input backing was already claimed"));
                }
                Ok(Some(array))
            }
            Some(StorageReuse::Candidates(candidates)) => {
                for input in candidates {
                    let Some(array) = self.values.get(input).cloned() else {
                        continue;
                    };
                    let Some(binding) = types::array_view_buffer(&array.ty) else {
                        continue;
                    };
                    if array.ty.elem_type().map(storage_value_type) == element
                        && self.reused_buffers.insert(binding)
                    {
                        return Ok(Some(array));
                    }
                }
                Ok(None)
            }
            Some(StorageReuse::Fresh) | None => Ok(None),
        }
    }

    fn write_array_value(
        &mut self,
        array: Typed,
        indices: &[Typed],
        value: Typed,
    ) -> Result<(), OptimizeError> {
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = &array.ty {
            for i in 0..fields.len() {
                let destination = self.field(array.clone(), i)?;
                let value = self.field(value.clone(), i)?;
                self.write_array_value(destination, indices, value)?;
            }
            return Ok(());
        }
        let Some((first, rest)) = indices.split_first() else {
            return Err(error("array write needs an index"));
        };
        if array.ty.array_rank().is_some_and(|rank| rank > 1) {
            let row = self.index(array, first.clone())?;
            if rest.is_empty() {
                let count = self.array_length(value.clone())?;
                self.copy_elements(value, row, count)?;
                return Ok(());
            }
            return self.write_array_value(row, rest, value);
        }
        self.storage_access(&array.ty, ResourceAccess::Write);
        let (mut place, mut ty) = self.index_place(array, first.clone())?;
        for index in rest {
            let Some(element) = ty.elem_type().cloned() else {
                return Err(error("array write index rank exceeds destination rank"));
            };
            let next = self.builder.new_place(element.clone());
            self.builder
                .push_void_inst(InstKind::PlaceIndex {
                    place,
                    index: index.value.clone(),
                    result: next,
                })
                .map_err(builder_error)?;
            place = next;
            ty = element;
        }
        let value = self.cast(value, &ty)?;
        self.builder
            .push_void_inst(InstKind::Store {
                place,
                value: value.value,
            })
            .map(|_| ())
            .map_err(builder_error)
    }

    fn update_slices(
        &mut self,
        destination: Value,
        slices: &[Value],
        prefix: Vec<Typed>,
        value: Typed,
        proven: bool,
    ) -> Result<(), OptimizeError> {
        let Some((&slice, rest)) = slices.split_first() else {
            return self.write_array(destination, &prefix, value);
        };
        if let Some(at) = self.query().enode("At", slice)? {
            let index = self.index_expression(at[0])?;
            let mut prefix = prefix;
            prefix.push(index);
            if proven {
                return self.update_slices(destination, rest, prefix, value, proven);
            }
            let array = self.named(destination)?;
            let valid = self.valid_indices(array, &prefix)?;
            return self.when(valid, |body| {
                body.update_slices(destination, rest, prefix, value, proven)
            });
        }
        let fields = self.query().fields("Slice", slice)?;
        let start = self.index_expression(fields[0])?;
        let count = self.index_expression(fields[1])?;
        let stride = self.index_expression(fields[2])?;
        let length = self.array_length(value.clone())?;
        let matches = self.binary(BinaryOperator::Equal, count.clone(), length)?;
        self.require_condition(matches)?;
        self.counted(
            Self::number(0),
            count,
            Self::number(1),
            vec![],
            |body, ordinal, _| {
                let item = body.index(value, ordinal.clone())?;
                let delta = body.binary(BinaryOperator::Multiply, stride.clone(), ordinal)?;
                let index = body.binary(BinaryOperator::Add, start.clone(), delta)?;
                let mut prefix = prefix.clone();
                prefix.push(index);
                if proven {
                    body.update_slices(destination, rest, prefix, item, proven)?;
                } else {
                    let array = body.named(destination)?;
                    let valid = body.valid_indices(array, &prefix)?;
                    body.when(valid, |body| {
                        body.update_slices(destination, rest, prefix, item, proven)
                    })?;
                }
                Ok(vec![])
            },
        )?;
        Ok(())
    }

    pub(in crate::egglog::to_ssa) fn slice_array(
        &mut self,
        array: Typed,
        start: Typed,
        count: Typed,
        stride: Typed,
    ) -> Result<Typed, OptimizeError> {
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = &array.ty {
            let mut components = Vec::new();
            for i in 0..fields.len() {
                let component = self.field(array.clone(), i)?;
                components.push(self.slice_array(
                    component,
                    start.clone(),
                    count.clone(),
                    stride.clone(),
                )?);
            }
            return self.tuple(components);
        }
        let zero = self.literal(0, &count.ty)?;
        let valid = self.binary(BinaryOperator::GreaterEqual, count.clone(), zero)?;
        self.require_condition(valid)?;
        if array.ty.array_variant().is_some_and(types::is_array_variant_virtual) {
            let Some(element) = array.ty.elem_type().cloned() else {
                return Err(error("range element missing"));
            };
            let first = self.index(array.clone(), start)?;
            let step = self.op(OpTag::Project { index: 1 }, vec![array.clone()], element.clone())?;
            let stride = self.cast(stride, &element)?;
            let step = self.binary(BinaryOperator::Multiply, step, stride)?;
            let count = self.cast(count, &element)?;
            return self.op(OpTag::Tuple, vec![first, step, count], array.ty);
        }
        if array.ty.array_variant().is_some_and(types::is_array_variant_view) {
            let one = self.literal(1, &stride.ty)?;
            let contiguous = self.binary(BinaryOperator::Equal, stride, one)?;
            self.require_condition(contiguous)?;
            if array.ty.array_rank().is_some_and(|rank| rank > 1) {
                let mut shape = self.view_dimensions(array.clone())?;
                let mut offset = start;
                for dimension in &shape[1..] {
                    offset = self.binary(BinaryOperator::Multiply, offset, dimension.clone())?;
                }
                shape[0] = count;
                let mut operands = vec![offset];
                operands.extend(shape);
                let ty = array.ty.clone();
                operands.push(array);
                return self.op(OpTag::StorageView(PureViewSource::Inherited), operands, ty);
            }
            let Some(element) = array.ty.elem_type() else {
                return Err(error("slice element missing"));
            };
            let Some(buffer) = array.ty.array_buffer() else {
                return Err(error("slice buffer missing"));
            };
            let ty = types::view_array_with_size(
                element,
                Type::Constructed(TypeName::SizePlaceholder, vec![]),
                buffer.clone(),
            );
            return self.op(
                OpTag::StorageView(PureViewSource::Inherited),
                vec![start, count, array],
                ty,
            );
        }
        let Some(Type::Constructed(TypeName::Size(capacity), _)) = array.ty.array_size() else {
            return Err(error("local slice requires a capacity bound"));
        };
        let capacity = *capacity;
        let Some(element) = array.ty.elem_type().cloned() else {
            return Err(error("slice element missing"));
        };
        let capacity_value = self.literal(
            i64::try_from(capacity).map_err(|_| error("slice capacity exceeds i64"))?,
            &count.ty,
        )?;
        let fits = self.binary(BinaryOperator::LessEqual, count.clone(), capacity_value)?;
        self.require_condition(fits)?;
        let storage = self.local_array(types::sized_array(capacity, element.clone()))?;
        self.counted(
            Self::number(0),
            count.clone(),
            Self::number(1),
            vec![],
            |body, ordinal, _| {
                let delta = body.binary(BinaryOperator::Multiply, ordinal.clone(), stride)?;
                let index = body.binary(BinaryOperator::Add, start, delta)?;
                let item = body.index(array, index)?;
                body.store(storage.clone(), ordinal, item)?;
                Ok(vec![])
            },
        )?;
        let storage = self.snapshot(storage)?;
        let count = self.cast(count, &types::i32())?;
        let ty = types::make_array1(
            element,
            types::array_variant_bounded(),
            Type::Constructed(TypeName::Size(capacity), vec![]),
            types::no_buffer(),
        );
        self.op(OpTag::Tuple, vec![storage, count], ty)
    }

    pub fn shape_count(&mut self, shape: &[Typed]) -> Result<Typed, OptimizeError> {
        let mut count = Self::number(1);
        for length in shape {
            count = self.binary(BinaryOperator::Multiply, count, length.clone())?;
        }
        Ok(count)
    }

    pub fn coordinates(
        &mut self,
        mut ordinal: Typed,
        shape: &[Typed],
    ) -> Result<Vec<Typed>, OptimizeError> {
        let mut coordinates = Vec::new();
        for (axis, length) in shape.iter().enumerate().rev() {
            // Traversal guarantees ordinal < product(shape); the leading coordinate needs no wrap.
            if axis == 0 {
                coordinates.push(ordinal);
                break;
            }
            let positive = self.binary(BinaryOperator::Greater, length.clone(), Self::number(0))?;
            let one = self.literal(1, &length.ty)?;
            let divisor = self.select(positive, length.clone(), one)?;
            coordinates.push(self.binary(BinaryOperator::Remainder, ordinal.clone(), divisor.clone())?);
            ordinal = self.binary(BinaryOperator::Divide, ordinal, divisor)?;
        }
        coordinates.reverse();
        Ok(coordinates)
    }

    fn accumulate(&mut self, operation: Value) -> Result<Vec<Typed>, OptimizeError> {
        let query = self.query();
        let operands = query.0.inputs(operation);
        let token = query.fields("AccNameRef", operands[0])?[0];
        let (owner, group) = self.compiler.execution.token_binding(token)?;
        let arrays = query.0.collective_inputs(owner, group, CollectiveInput::Backing);
        let policy = query.0.accumulator_policy(owner, group)?;
        let combine = query.0.combiner_at(owner, group);
        let indices = query
            .vector(query.0.formula(operation, Formula::Extents)?)?
            .into_iter()
            .map(|index| self.index_expression(index))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(&first) = arrays.first() else {
            return Err(error("empty accumulator destination"));
        };
        if self.values.get(&first).is_some_and(|value| value.ty.is_array()) {
            let array = self.named(first)?;
            let valid = self.valid_indices(array, &indices)?;
            let values = self.yield_values(&operands[1..])?;
            if let Some(update) = self.atomic_policy(&arrays, policy, combine)? {
                let [value] = values.as_slice() else {
                    return Err(error("atomic accumulator must receive one scalar"));
                };
                self.when(valid, |body| {
                    body.atomic_accumulate(first, &indices, value.clone(), update)
                })?;
                return Ok(vec![]);
            }
            self.when(valid, |body| {
                let values = if let Some(combine) = combine {
                    let mut operands = arrays
                        .iter()
                        .map(|&array| body.read_array(array, &indices))
                        .collect::<Result<Vec<_>, _>>()?;
                    operands.extend(values);
                    body.lambda(combine, operands)?
                } else {
                    values
                };
                if values.len() != arrays.len() {
                    return Err(error("accumulator value count differs from destinations"));
                }
                for (&array, value) in arrays.iter().zip(values) {
                    body.write_array(array, &indices, value)?;
                }
                Ok(())
            })?;
            return Ok(vec![]);
        }
        Err(error("accumulator destination has no SSA array representation"))
    }
}
