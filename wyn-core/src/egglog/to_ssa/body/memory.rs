//! Emit statements in lexical order; descriptor arithmetic comes from bindings.
use super::{builder_error, error, Body, OptimizeError, Typed, Value};
use crate::builtins::{catalog, select};
use crate::egglog::query::Query;
use crate::egglog::to_ssa::{host, primitive_type, runtime_binding, same_value_structure};
use crate::host::Expr;
use crate::op::{BinaryOperator, OpTag, PureViewSource, UnaryOperator};
use crate::ssa::layout::{storage_elem_stride, storage_value_type};
use crate::ssa::types::InstKind;
use crate::types::{self, Type, TypeExt, TypeName};
use crate::ResourceAccess;
use egglog_engine::sort::{SetContainer, S, Z};

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
        let Some(value) = self.values.get(&name).or_else(|| self.collection_values.get(&name)) else {
            return Err(error(format!("unbound emitted value {name:?}")));
        };
        Ok(value.clone())
    }

    pub fn atom(&mut self, atom: Value) -> Result<Typed, OptimizeError> {
        let (kind, fields) = self.query().application(atom, &["Var", "LiteralBits"])?;
        if kind == "Var" {
            return self.named(fields[0]);
        }
        let ty = primitive_type(self.query(), fields[0])?;
        let bits = self.compiler.program.graph.value_to_base::<i64>(fields[1]);
        if ty == types::unit() {
            return self.op(OpTag::Unit, vec![], ty);
        }
        self.literal(bits, &ty)
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
        let (kind, fields) =
            self.query().application(operand, &["ValueOperand", "IndexOperand", "AccOperand"])?;
        match kind {
            "AccOperand" => Ok(None),
            "IndexOperand" => self.index_expression(fields[0]).map(Some),
            "ValueOperand" => self.atom(fields[0]).map(Some),
            _ => unreachable!(),
        }
    }

    pub fn results(&mut self, results: Value) -> Result<Vec<Typed>, OptimizeError> {
        let mut values = Vec::new();
        for result in self.query().vector(results)? {
            let fields = self.query().fields("Yield", result)?;
            if let Some(value) = self.operand(fields[1])? {
                values.push(self.snapshot(value)?);
            }
        }
        Ok(values)
    }

    pub fn block(&mut self, block: Value) -> Result<Vec<Typed>, OptimizeError> {
        let fields = self.query().fields("Block", block)?;
        self.statements(fields[0])?;
        self.results(fields[1])
    }

    pub fn statements(&mut self, statements: Value) -> Result<(), OptimizeError> {
        for statement in self.query().vector(statements)? {
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

    pub fn lambda(&mut self, lambda: Value, arguments: Vec<Typed>) -> Result<Vec<Typed>, OptimizeError> {
        let fields = self.query().fields("FunctionBody", lambda)?;
        let checkpoint = self.values.checkpoint();
        let pattern = self
            .query()
            .vector(fields[0])?
            .into_iter()
            .map(|param| Ok(self.query().fields("Parameter", param)?[0]))
            .collect::<Result<Vec<_>, OptimizeError>>()?;
        self.bind(&pattern, arguments)?;
        let result = self.block(fields[2]);
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
        host::remember(
            Query(&self.compiler.program.graph),
            self.function,
            &mut self.host_values,
            statement,
        )?;
        let fields = self.query().fields("Let", statement)?;
        let pattern = self.query().vector(fields[1])?;
        let expression = fields[3];
        let (kind, arguments) = self.query().application(
            expression,
            &[
                "Use",
                "Product",
                "Project",
                "Length",
                "Index",
                "PrimOp",
                "IndexOp",
                "Assert",
                "Call",
                "If",
                "Loop",
                "Parallel",
                "WithAcc",
                "BucketScatter",
                "View",
                "GenerateArray",
                "Copy",
                "Scratch",
                "Update",
                "Accumulate",
                "Reserve",
            ],
        )?;
        if kind == "Parallel"
            && self.query().enode("DeviceKernel", arguments[0])?.is_some()
            && self.active_dispatch != Some(statement)
        {
            return Err(error("device parallel region requires a dispatch entry"));
        }
        let values = match kind {
            "Use" => self.operand(arguments[0])?.into_iter().collect(),
            "Product" => {
                let values = self
                    .query()
                    .vector(arguments[0])?
                    .into_iter()
                    .map(|atom| self.atom(atom))
                    .collect::<Result<Vec<_>, _>>()?;
                vec![self.tuple(values)?]
            }
            "Project" => {
                let value = self.atom(arguments[0])?;
                let index = usize::try_from(self.query().0.value_to_base::<i64>(arguments[1]))
                    .map_err(|_| error("negative product field"))?;
                vec![self.field(value, index)?]
            }
            "IndexOp" => vec![self.index_expression(arguments[0])?],
            "Copy" if self.values.get(&arguments[0]).is_some_and(|value| value.ty.is_array()) => {
                let source = self.named(arguments[0])?;
                if source.ty.array_variant().is_some_and(types::is_array_variant_composite) {
                    if let [binding] = pattern.as_slice() {
                        if let Some((output, _)) = runtime_binding(self.query(), *binding)? {
                            if self.value_updates_only(output)? {
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
                let known_count = self
                    .query()
                    .lookup("LogicalOuterCount", (self.function, arguments[0]))?
                    .map(|count| self.query().lookup("StaticIndex", (self.function, count)))
                    .transpose()?
                    .flatten();
                let count = if let Some(count) = known_count {
                    self.literal(self.query().0.value_to_base::<i64>(count), &types::i32())?
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
                    self.host_lengths.insert(destination.value.clone(), length);
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
                        .query()
                        .lookup("OuterCapacity", (self.function, arguments[0]))?
                        .and_then(|value| {
                            usize::try_from(self.query().0.value_to_base::<i64>(value)).ok()
                        }));
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
            "Scratch" => {
                let source = self.named(arguments[0])?;
                let count = self.array_length(source.clone())?;
                let length = if self.active_dispatch.is_none() {
                    Some(self.host_array_length(&source)?)
                } else {
                    None
                };
                let block = self.current()?;
                vec![self.collection_storage(&source.ty, count, length.as_ref(), None, block)?]
            }
            "GenerateArray" if self.query().enode("UninitializedShape", arguments[0])?.is_some() => {
                let fields = self.query().fields("UninitializedShape", arguments[0])?;
                let shape = self.query().vector(fields[0])?;
                let [extent] = shape.as_slice() else {
                    return Err(error("scratch allocation requires one outer extent"));
                };
                let [binding] = pattern.as_slice() else {
                    return Err(error("scratch allocation must bind one array"));
                };
                let Some((_, ty)) = runtime_binding(self.query(), *binding)? else {
                    return Err(error("scratch allocation has no value binding"));
                };
                let count = self.index_expression(*extent)?;
                let nonnegative =
                    self.binary(BinaryOperator::GreaterEqual, count.clone(), Self::number(0))?;
                self.require_condition(nonnegative)?;
                let length = if self.active_dispatch.is_none() {
                    host::scalar_index(self.query(), &self.host_values, *extent)?.map(Expr::Scalar)
                } else {
                    None
                };
                let block = self.current()?;
                vec![self.collection_storage(&ty, count, length.as_ref(), None, block)?]
            }
            "Length" => {
                let array = self.named(arguments[0])?;
                let axis = self.query().0.value_to_base::<i64>(arguments[1]);
                if axis != 0 {
                    return Err(error("logical length requires an outer array axis"));
                }
                if let [binding] = pattern.as_slice() {
                    if let Some((name, _)) = runtime_binding(self.query(), *binding)? {
                        if let Ok(length) = self.host_array_length(&array) {
                            let scalar = host::scalar_extent(&length);
                            if let Some(scalar) = scalar {
                                self.host_values.insert(name, scalar);
                            }
                        }
                    }
                }
                vec![self.array_length(array)?]
            }
            "Index" => {
                let mut array = self.named(arguments[0])?;
                for atom in self.query().vector(arguments[1])? {
                    let index = self.atom(atom)?;
                    array = self.index(array, index)?;
                }
                vec![array]
            }
            "GenerateArray" if self.query().enode("RangeValues", arguments[0])?.is_some() => {
                let fields = self.query().fields("RangeValues", arguments[0])?;
                let start = self.atom(fields[0])?;
                let step = self.atom(fields[2])?;
                let length = self.index_expression(fields[1])?;
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
                if let Some(length) = host::scalar_index(self.query(), &self.host_values, fields[1])? {
                    self.host_lengths.insert(
                        value.value,
                        Expr::Max(Box::new(Expr::Scalar(length)), Box::new(Expr::Integer(0))),
                    );
                }
                vec![value]
            }
            "GenerateArray" if self.query().enode("ArrayLiteral", arguments[0])?.is_some() => {
                let literal = self.query().fields("ArrayLiteral", arguments[0])?;
                let mut elements = self
                    .query()
                    .vector(literal[0])?
                    .into_iter()
                    .map(|atom| self.atom(atom))
                    .collect::<Result<Vec<_>, _>>()?;
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
            "PrimOp" => {
                let opcode = self.compiler.program.graph.value_to_base::<S>(arguments[0]).to_string();
                let args = self
                    .query()
                    .vector(arguments[1])?
                    .into_iter()
                    .map(|atom| self.atom(atom))
                    .collect::<Result<Vec<_>, _>>()?;
                let [binding] = pattern.as_slice() else {
                    return Err(error("scalar instruction must bind one result"));
                };
                let Some((_, ty)) = runtime_binding(self.query(), *binding)? else {
                    return Err(error("scalar result is not a value"));
                };
                vec![self.primitive(&opcode, args, ty)?]
            }
            "If" => {
                let condition = self.atom(arguments[0])?;
                let types: Vec<_> = pattern
                    .iter()
                    .map(|&binding| runtime_binding(self.query(), binding))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .map(|(_, ty)| ty)
                    .collect();
                let mut speculate = types.iter().all(select::supported_type);
                for &branch in &arguments[2..4] {
                    let effects = self.query().required("EffectsBody", (self.function, branch))?;
                    let Some(effects) = self.query().0.value_to_container::<SetContainer>(effects) else {
                        return Err(error("invalid branch effects"));
                    };
                    speculate &= effects.data.is_empty();
                    speculate &=
                        self.query().lookup("BodyWork", (self.function, branch))?.is_some_and(|work| {
                            u64::try_from(&*self.query().0.value_to_base::<Z>(work))
                                .is_ok_and(|work| work <= 4)
                        });
                }
                if speculate {
                    let yes = self.block(arguments[2])?;
                    let no = self.block(arguments[3])?;
                    if yes.len() != types.len() || no.len() != types.len() {
                        return Err(error("selected branch result arity mismatch"));
                    }
                    let mut values = Vec::new();
                    for ((yes, no), ty) in yes.into_iter().zip(no).zip(&types) {
                        let yes = self.canonical_value(yes, ty)?;
                        let no = self.canonical_value(no, ty)?;
                        values.push(self.select(condition.clone(), yes, no)?);
                    }
                    return self.bind(&pattern, values);
                }
                let ty = types::tuple(types);
                let result = self.branch(
                    condition,
                    |body| {
                        let values = body.block(arguments[2])?;
                        let value = body.tuple(values)?;
                        body.canonical_value(value, &ty)
                    },
                    |body| {
                        let values = body.block(arguments[3])?;
                        let value = body.tuple(values)?;
                        body.canonical_value(value, &ty)
                    },
                )?;
                self.unpack(result)?
            }
            "Loop" => {
                let mut params = Vec::new();
                let mut initial = Vec::new();
                for (slot, merge) in self.query().vector(arguments[0])?.into_iter().enumerate() {
                    let fields = self.query().fields("Carry", merge)?;
                    let param = self.query().fields("Parameter", fields[0])?;
                    if let Some((name, ty)) = runtime_binding(self.query(), param[0])? {
                        let Some(value) = self.operand(fields[1])? else {
                            return Err(error("loop carry has no runtime value"));
                        };
                        params.push(name);
                        let mut value = self.canonical_value(value, &ty)?;
                        let mut components = 0;
                        let mut preserved = true;
                        self.query().for_each("LoopArrayComponent", |row| {
                            if row[0] == self.function
                                && row[1] == expression
                                && self.query().0.value_to_base::<i64>(row[2]) == slot as i64
                            {
                                components += 1;
                                preserved &= self.query().contains(
                                    "LoopPreservesComponentLength",
                                    (row[0], row[1], row[2], row[3]),
                                )?;
                            }
                            Ok(())
                        })?;
                        if self.query().contains(
                            "LoopPreservesOuterLength",
                            (self.function, expression, slot as i64),
                        )? || (components > 0 && preserved)
                        {
                            let initial_name = self
                                .query()
                                .enode("ValueOperand", fields[1])?
                                .map(|operand| self.query().enode("Var", operand[0]))
                                .transpose()?
                                .flatten()
                                .map(|atom| atom[0]);
                            value = self.fixed_loop_state(value, initial_name)?;
                        }
                        initial.push(value);
                    }
                }
                let install = |body: &mut Self, state: &[Typed]| {
                    for (&name, value) in params.iter().zip(state) {
                        body.values.insert(name, value.clone());
                    }
                };
                if let Some(form) = self.query().enode("Counted", arguments[1])? {
                    let start = self.index_expression(form[1])?;
                    let count = self.index_expression(form[2])?;
                    let step = self.index_expression(form[3])?;
                    self.counted(
                        Self::number(0),
                        count,
                        Self::number(1),
                        initial,
                        |body, ordinal, state| {
                            install(body, &state);
                            let ordinal = body.cast(ordinal, &start.ty)?;
                            let offset = body.binary(BinaryOperator::Multiply, ordinal, step)?;
                            let index = body.binary(BinaryOperator::Add, start, offset)?;
                            body.values.insert(form[0], index);
                            body.block(arguments[3])
                        },
                    )?
                } else {
                    let form = self.query().fields("While", arguments[1])?;
                    self.while_loop(
                        initial,
                        |body, state| {
                            install(body, state);
                            body.named(form[0])
                        },
                        |body, state| {
                            install(body, &state);
                            body.block(arguments[3])
                        },
                    )?
                }
            }
            "Call" => {
                let mut args = Vec::new();
                for arg in self.query().vector(arguments[1])? {
                    let arg = self.query().fields("Pass", arg)?;
                    if let Some(value) = self.operand(arg[0])? {
                        args.push(value);
                    }
                }
                let (id, ty) = self.compiler.function(arguments[0])?;
                let result = self.op(OpTag::Call(id), args, ty)?;
                if pattern.len() == 1 {
                    vec![result]
                } else {
                    self.unpack(result)?
                }
            }
            "WithAcc" => {
                let lambda = self.query().fields("FunctionBody", arguments[1])?;
                let block = self.query().fields("Block", lambda[2])?;
                self.statements(block[0])?;
                let mut values = Vec::new();
                for input in self.query().vector(arguments[0])? {
                    let input = self.query().fields("AccumulatorInput", input)?;
                    for name in self.query().vector(input[0])? {
                        let value = self.named(name)?;
                        values.push(value);
                    }
                }
                values.extend(self.results(block[1])?);
                if pattern.len() == 1 && values.len() > 1 {
                    vec![self.tuple(values)?]
                } else {
                    values
                }
            }
            "BucketScatter" => vec![self.bucket_scatter(&arguments)?],
            "View" => {
                let mut array = self.named(arguments[0])?;
                for slice in self.query().vector(arguments[1])? {
                    if let Some(at) = self.query().enode("At", slice)? {
                        let index = self.index_expression(at[0])?;
                        array = self.index(array, index)?;
                    } else {
                        let fields = self.query().fields("Slice", slice)?;
                        let start = self.index_expression(fields[0])?;
                        let count = self.index_expression(fields[1])?;
                        let stride = self.index_expression(fields[2])?;
                        let mut shape = self.host_shapes.get(&array.value).cloned();
                        array = self.slice_array(array, start, count, stride)?;
                        if let Some(length) =
                            host::scalar_index(self.query(), &self.host_values, fields[1])?
                        {
                            let length = Expr::Scalar(length);
                            self.host_lengths.insert(array.value, length.clone());
                            if let Some(shape) = shape.as_mut() {
                                shape[0] = length;
                                self.host_shapes.insert(array.value, shape.clone());
                            }
                        }
                    }
                }
                vec![array]
            }
            "Update" => {
                let slices = self.query().vector(arguments[2])?;
                let value = self.atom(arguments[3])?;
                let source = self.named(arguments[1])?;
                if source.ty.array_variant().is_some_and(types::is_array_variant_composite) {
                    if let [slice] = slices.as_slice() {
                        if let Some(at) = self.query().enode("At", *slice)? {
                            let index = self.index_expression(at[0])?;
                            let source = self.snapshot(source)?;
                            let ty = source.ty.clone();
                            // Value updates are total and immutable. Place them at
                            // their use so old element reads precede optional reuse.
                            let result = self.builder.func_mut().append_floating_inst(
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
                self.update_slices(
                    arguments[1],
                    &slices,
                    vec![],
                    value,
                    self.query().enode("ProvenInBounds", arguments[0])?.is_some(),
                )?;
                let value = self.named(arguments[1])?;
                vec![self.snapshot(value)?]
            }
            "Assert" => {
                let condition = self.atom(arguments[0])?;
                self.require_condition(condition)?;
                Vec::new()
            }
            "Accumulate" | "Reserve" => self.accumulate(kind, &arguments)?,
            "Parallel" => self.parallel(&arguments, &[], &pattern)?,
            _ => return Err(error(format!("unimplemented logical operation {kind}"))),
        };
        self.bind(&pattern, values)
    }

    fn primitive(&mut self, opcode: &str, arguments: Vec<Typed>, ty: Type) -> Result<Typed, OptimizeError> {
        if opcode == "vector-index" {
            let [vector, index]: [Typed; 2] =
                arguments.try_into().map_err(|_| error("vector indexing arity"))?;
            return self.op(OpTag::Index, vec![vector, index], ty);
        }
        if opcode == "index-cast" {
            let [value]: [Typed; 1] = arguments.try_into().map_err(|_| error("index conversion arity"))?;
            return self.cast(value, &ty);
        }
        if opcode == "select" {
            let [no, yes, condition] = arguments.as_slice() else {
                return Err(error("select requires a condition and two values"));
            };
            return self.select(condition.clone(), yes.clone(), no.clone());
        }
        let tag = match opcode {
            "unit" => OpTag::Unit,
            "vector" => OpTag::Vector,
            "tuple" => OpTag::Tuple,
            _ if opcode.starts_with("project:") => OpTag::Project {
                index: opcode[8..].parse().map_err(|_| error("invalid vector projection"))?,
            },
            _ => {
                if let Some((name, overload)) =
                    opcode.strip_prefix("builtin:").and_then(|name| name.rsplit_once(':'))
                {
                    let Some(builtin) = catalog().lookup_by_any_name(name) else {
                        return Err(error(format!("unknown primitive {opcode}")));
                    };
                    OpTag::Intrinsic {
                        id: builtin.id,
                        overload_idx: overload.parse().map_err(|_| error("invalid builtin overload"))?,
                    }
                } else if let Some(builtin) = catalog().lookup_by_any_name(opcode) {
                    OpTag::Intrinsic {
                        id: builtin.id,
                        overload_idx: 0,
                    }
                } else if arguments.len() == 1 {
                    OpTag::UnaryOp(
                        UnaryOperator::try_from(opcode)
                            .map_err(|_| error(format!("unknown unary primitive {opcode}")))?,
                    )
                } else {
                    OpTag::BinOp(
                        BinaryOperator::try_from(opcode)
                            .map_err(|_| error(format!("unknown binary primitive {opcode}")))?,
                    )
                }
            }
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
        if let Some((ty, count, length, shape, block)) = self.collection_requests.remove(&name) {
            let ty = collection_type(&ty, &value.ty, indices.len())?;
            let storage = if let Some(storage) = self.reuse_storage(name, &ty)? {
                storage
            } else {
                let capacity = self
                    .query()
                    .lookup("OuterCapacity", (self.function, name))?
                    .map(|value| self.query().0.value_to_base::<i64>(value));
                if let Some(shape) = shape {
                    self.ranked_collection_storage(&ty, &shape, block)?
                } else {
                    self.collection_storage(&ty, count, length.as_ref(), capacity, block)?
                }
            };
            self.collection_values.insert(name, storage);
        }
        if let Some(array) = self
            .values
            .get(&name)
            .or_else(|| self.collection_values.get(&name))
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

    pub(in crate::egglog::to_ssa) fn reuse_storage(
        &mut self,
        output: Value,
        ty: &Type,
    ) -> Result<Option<Typed>, OptimizeError> {
        let mut candidates = Vec::new();
        self.query().for_each("CanReuse", |row| {
            if row[0] == self.function && row[1] == output {
                candidates.push(row[2]);
            }
            Ok(())
        })?;
        for input in candidates {
            let Some(array) = self.values.get(&input).cloned() else {
                continue;
            };
            let Some(binding) = types::array_view_buffer(&array.ty) else {
                continue;
            };
            if array.ty.elem_type().map(storage_value_type) == ty.elem_type().map(storage_value_type)
                && self.reused_buffers.insert(binding)
            {
                return Ok(Some(array));
            }
        }
        Ok(None)
    }

    fn value_updates_only(&self, name: Value) -> Result<bool, OptimizeError> {
        let root = self
            .query()
            .constructor("FunctionRoot", (self.function,))?
            .ok_or_else(|| error("function root missing for value update"))?;
        let reference = self
            .query()
            .constructor("ValueNameRef", (name,))?
            .ok_or_else(|| error("array name missing for value update"))?;
        let mut updates = false;
        let mut safe = true;
        self.query().for_each("ContainsExp", |row| {
            if row[0] != root {
                return Ok(());
            }
            let names = self.query().required("DirectNamesExp", (row[1],))?;
            let Some(names) = self.query().0.value_to_container::<egglog_engine::sort::SetContainer>(names)
            else {
                return Err(error("invalid direct name set"));
            };
            if !names.data.contains(&reference) {
                return Ok(());
            }
            if let Some(update) = self.query().enode("Update", row[1])? {
                let slices = self.query().vector(update[2])?;
                let simple = update[1] == name
                    && slices.len() == 1
                    && self.query().enode("At", slices[0])?.is_some();
                updates |= simple;
                safe &= simple;
            } else {
                safe &= self.query().enode("Index", row[1])?.is_some()
                    || self.query().enode("Length", row[1])?.is_some();
            }
            Ok(())
        })?;
        Ok(safe && updates)
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

    pub(in crate::egglog::to_ssa) fn write_atom(
        &mut self,
        destination: Value,
        prefix: &[Typed],
        atom: Value,
    ) -> Result<(), OptimizeError> {
        let value = self.atom(atom)?;
        self.write_array(destination, prefix, value)
    }

    fn accumulate(&mut self, kind: &str, arguments: &[Value]) -> Result<Vec<Typed>, OptimizeError> {
        let capability = self.query().required(
            "TokenCapability",
            (self.function, arguments[usize::from(kind != "Reserve")]),
        )?;
        let capability = self.query().fields("AccumulatorInput", capability)?;
        let arrays = self.query().vector(capability[0])?;
        let policy = capability[1];
        let token = usize::from(kind != "Reserve");
        let indices = self
            .query()
            .vector(arguments[token + 1])?
            .into_iter()
            .map(|index| self.index_expression(index))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(&first) = arrays.first() else {
            return Err(error("empty accumulator destination"));
        };
        if self.values.get(&first).is_some_and(|value| value.ty.is_array()) {
            let array = self.named(first)?;
            let valid = self.valid_indices(array, &indices)?;
            if kind == "Reserve" {
                let delta = self.atom(arguments[2])?;
                let ty = delta.ty.clone();
                let value = self.branch(
                    valid,
                    |body| {
                        let previous = body.read_array(first, &indices)?;
                        let next = body.binary(BinaryOperator::Add, previous.clone(), delta)?;
                        body.write_array(first, &indices, next)?;
                        Ok(previous)
                    },
                    |body| body.literal(0, &ty),
                )?;
                return Ok(vec![value]);
            }
            let values = self
                .query()
                .vector(arguments[3])?
                .into_iter()
                .map(|atom| self.atom(atom))
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(update) = self.atomic_policy(&arrays, policy)? {
                let [value] = values.as_slice() else {
                    return Err(error("atomic accumulator must receive one scalar"));
                };
                self.when(valid, |body| {
                    body.atomic_accumulate(first, &indices, value.clone(), update)
                })?;
                return Ok(vec![]);
            }
            self.when(valid, |body| {
                let values = if let Some(combine) = body.query().enode("CombineWrites", policy)? {
                    let mut operands = arrays
                        .iter()
                        .map(|&array| body.read_array(array, &indices))
                        .collect::<Result<Vec<_>, _>>()?;
                    operands.extend(values);
                    body.lambda(combine[0], operands)?
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
