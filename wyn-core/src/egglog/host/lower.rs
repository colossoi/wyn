//! Translate selected scalar terms and explicit control facts to the host IR.
use crate::binding_layout::{extract_storage_binding, extract_uniform_binding};
use crate::builtins::lowering::PrimOp;
use crate::builtins::{catalog, BuiltinLowering};
use crate::egglog::query::Query;
use crate::egglog::{bindings::Bindings, OptimizeError, Optimized, Program};
use crate::egglog::{facts::Facts, output_error as error};
use crate::host::{ScalarExpr, ScalarSource, ScalarType};
use crate::interface::StorageLayout;
use crate::op::OpTag;
use crate::ssa::layout::{block_layout, storage_value_type, type_byte_size};
use crate::types::{Type, TypeExt, TypeName};
use crate::{BindingRef, FunctionId};
use egglog_engine::{TermId, Value};

/// Unsupported target operations are distinct from incomplete compiler facts.
pub(in crate::egglog) enum Error {
    Unsupported,
    Invalid(OptimizeError),
}
impl From<OptimizeError> for Error {
    fn from(error: OptimizeError) -> Self {
        Self::Invalid(error)
    }
}
impl Error {
    pub(in crate::egglog) fn required(self, value: &str) -> OptimizeError {
        match self {
            Self::Unsupported => error(format!("{value} has no supported host expression")),
            Self::Invalid(error) => error,
        }
    }
}
type Result<T> = std::result::Result<T, Error>;

pub(in crate::egglog) fn scalar_type(ty: &Type) -> Option<ScalarType> {
    match ty {
        Type::Constructed(TypeName::Int(32), _) => Some(ScalarType::I32),
        Type::Constructed(TypeName::UInt(32), _) => Some(ScalarType::U32),
        Type::Constructed(TypeName::Float(32), _) => Some(ScalarType::F32),
        Type::Constructed(TypeName::Bool, _) => Some(ScalarType::Bool),
        _ => None,
    }
}
pub(in crate::egglog) fn expression(
    program: &Program<'_, Optimized>,
    context: Value,
    source: Value,
) -> Result<ScalarExpr> {
    let mut lower = Lower::new(program);
    let value = lower.source(context, source)?;
    Ok(lower.finish(value))
}
/// Allocation capacities may use proven bounds for device-produced logical lengths.
pub(in crate::egglog) fn allocation(
    program: &Program<'_, Optimized>,
    context: Value,
    source: Value,
) -> Result<ScalarExpr> {
    let mut lower = Lower::new(program);
    lower.capacity_bounds = true;
    let value = lower.source(context, source)?;
    Ok(lower.finish(value))
}
pub(in crate::egglog) fn selected(program: &Program<'_, Optimized>, term: TermId) -> Result<ScalarExpr> {
    let mut lower = Lower::new(program);
    let value = lower.term(term)?;
    Ok(lower.finish(value))
}
pub(in crate::egglog) fn length(program: &Program<'_, Optimized>, source: Value) -> Result<ScalarExpr> {
    let mut lower = Lower::new(program);
    let value = lower.length(source)?;
    Ok(lower.finish(value))
}
struct Lower<'a, 'source> {
    facts: Facts<'a, 'source>,
    values: Bindings<Value, ScalarExpr>,
    terms: Bindings<TermId, ScalarExpr>,
    bindings: Vec<(String, ScalarExpr)>,
    next: usize,
    capacity_bounds: bool,
}
impl<'a, 'source> Lower<'a, 'source> {
    fn new(program: &'a Program<'source, Optimized>) -> Self {
        Self {
            facts: Facts { program },
            values: Bindings::default(),
            terms: Bindings::default(),
            bindings: vec![],
            next: 0,
            capacity_bounds: false,
        }
    }
    fn length(&mut self, source: Value) -> Result<ScalarExpr> {
        let Some(extent) = self.facts.lookup("LogicalExtent", (source,)) else {
            return Err(error(format!("host array {source:?} has no selected logical extent")).into());
        };
        self.extent(extent)
    }
    fn extent(&mut self, extent: Value) -> Result<ScalarExpr> {
        if self.capacity_bounds {
            if let Some(bound) = self.facts.lookup("HostBound", (extent,)) {
                return self.extent(bound);
            }
        }
        if let Some(fields) = self.facts.enode("Fixed", extent) {
            return Ok(ScalarExpr::I32(
                i32::try_from(self.facts.integer(fields[0]))
                    .map_err(|_| error("logical length exceeds i32"))?,
            ));
        }
        if let Some(fields) = self.facts.enode("Length", extent) {
            let source = fields[0];
            if let Some((binding, stride)) = self.facts.input_storage(source)? {
                return Ok(ScalarExpr::BufferLength {
                    source: ScalarSource::Binding {
                        set: binding.set,
                        binding: binding.binding,
                    },
                    stride,
                });
            }
            if self.facts.lookup("LogicalExtent", (source,)) == Some(extent) {
                // Invocation-local views have no host-readable descriptor.
                return Err(Error::Unsupported);
            }
            return self.length(source);
        }
        if let Some(fields) = self.facts.enode("Scalar", extent) {
            let source = fields[0];
            let Some(context) = self.facts.lookup("ScalarSourceContext", (source,)) else {
                return Err(error("host extent has no selected scalar context").into());
            };
            return self.source(context, source);
        }
        for (name, op) in [("Difference", "sub"), ("Product", "mul")] {
            if let Some(fields) = self.facts.enode(name, extent) {
                return Ok(apply(
                    op,
                    ScalarType::I32,
                    vec![self.extent(fields[0])?, self.extent(fields[1])?],
                ));
            }
        }
        // Device counts require readback. Capacity bounds are not logical lengths.
        if self.facts.enode("Stored", extent).is_some() {
            return Err(Error::Unsupported);
        }
        Err(error("unsupported selected logical extent").into())
    }
    fn name(&mut self) -> String {
        let name = format!("value{}", self.next);
        self.next += 1;
        name
    }
    fn finish(&mut self, mut value: ScalarExpr) -> ScalarExpr {
        // The final binding has no later uses: return its expression directly.
        while let ScalarExpr::Local(name) = &value {
            if !self.bindings.last().is_some_and(|(last, _)| last == name) {
                break;
            }
            value = self.bindings.pop().unwrap().1;
        }
        if self.bindings.is_empty() {
            return value;
        }
        ScalarExpr::Let {
            bindings: std::mem::take(&mut self.bindings),
            result: Box::new(value),
        }
    }
    fn scope(&mut self, emit: impl FnOnce(&mut Self) -> Result<ScalarExpr>) -> Result<ScalarExpr> {
        let outer = std::mem::take(&mut self.bindings);
        let values = self.values.checkpoint();
        let terms = self.terms.checkpoint();
        let result = emit(self).map(|value| self.finish(value));
        self.values.restore(values);
        self.terms.restore(terms);
        self.bindings = outer;
        result
    }
    fn source(&mut self, context: Value, source: Value) -> Result<ScalarExpr> {
        if let Some(value) = self.values.get(&source) {
            return Ok(value.clone());
        }
        let Some(&term) = self.facts.program.stage.selected.roots.get(&(context, source)) else {
            return self.boundary(context, source);
        };
        self.term(term)
    }
    fn region(&mut self, region: Value) -> Result<ScalarExpr> {
        let Some(context) = self.facts.context(region) else {
            return Err(error("host region context missing").into());
        };
        let Some(source) = self.facts.result(region) else {
            return Err(error("host region result missing").into());
        };
        self.source(context, source)
    }
    fn boundary(&mut self, context: Value, source: Value) -> Result<ScalarExpr> {
        if let Some(value) = self.values.get(&source) {
            return Ok(value.clone());
        }
        if let Some(actual) = self.facts.alias(source) {
            return self.source(context, actual);
        }
        if let Some(array) = self.facts.lookup("SourceLength", (source,)) {
            // Fusion may replace the query with a count-reduction result.
            if !self.capacity_bounds && self.stored(source) {
                return Err(Error::Unsupported);
            }
            return self.length(array);
        }
        if let Some((base, index)) = self.facts.projection(source) {
            return Ok(field(self.source(context, base)?, index));
        }
        if let Some(formal) = self.facts.enode("SourceFormal", source) {
            if let Some(actual) = self.facts.lookup("SsaCaptureAt", (formal[0], source)) {
                let Some(context) = self.facts.lookup("ScalarSourceContext", (actual,)) else {
                    return Err(error("host capture context missing").into());
                };
                return self.source(context, actual);
            }
        }
        if let Some(region) = self.facts.lookup("SourceParameterRegion", (source,)) {
            let Some(index) = self.facts.lookup("SourceParameterIndex", (source,)) else {
                return Err(error("host parameter index missing").into());
            };
            let index = self.facts.integer(index);
            if !self.facts.entry_region(region) {
                // A callback element or unbound function argument is device-local.
                return Err(Error::Unsupported);
            }
            return self.parameter(region, index, source);
        }
        if let Some((yes, no)) = self.facts.branches(source) {
            let Some(condition) = self.facts.lookup("SsaBranchCondition", (source,)) else {
                return Err(error("host branch condition missing").into());
            };
            return Ok(ScalarExpr::If {
                condition: Box::new(self.source(context, condition)?),
                yes: Box::new(self.scope(|lower| lower.region(yes))?),
                no: Box::new(self.scope(|lower| lower.region(no))?),
            });
        }
        if let Some((header, iteration)) = self.facts.loops(source) {
            let Some(state) = self.facts.loop_state(header) else {
                return Err(error("host loop state missing").into());
            };
            let initial = self.facts.loop_initial(header)?;
            let initial = self.source(context, initial)?;
            let Some(form) = self.facts.lookup("SourceLoopForm", (source,)) else {
                return Err(error("host loop form missing").into());
            };
            let name = self.name();
            if let Some(fields) = self.facts.enode("WhileCondition", form) {
                let Some(context) = self.facts.context(header) else {
                    return Err(error("host loop context missing").into());
                };
                let condition = self.scope(|lower| {
                    lower.values.insert(state, ScalarExpr::Local(name.clone()));
                    lower.source(context, fields[0])
                })?;
                let step = self.scope(|lower| {
                    lower.values.insert(state, ScalarExpr::Local(name.clone()));
                    lower.region(iteration)
                })?;
                return Ok(ScalarExpr::Loop {
                    name,
                    initial: Box::new(initial),
                    condition: Box::new(condition),
                    step: Box::new(step),
                });
            }
            if let Some(fields) = self.facts.enode("ForCount", form) {
                let Some(index) = self.facts.iteration(iteration) else {
                    return Err(error("host loop index missing").into());
                };
                let Some(ty) = self.facts.ty(fields[1]) else {
                    return Err(error("host loop count type missing").into());
                };
                let Some(ty) = scalar_type(ty) else {
                    return Err(Error::Unsupported);
                };
                let (zero, one) = match ty {
                    ScalarType::I32 => (ScalarExpr::I32(0), ScalarExpr::I32(1)),
                    ScalarType::U32 => (ScalarExpr::U32(0), ScalarExpr::U32(1)),
                    _ => return Err(error("host loop counter is not an integer").into()),
                };
                let limit = self.source(context, fields[0])?;
                let local = ScalarExpr::Local(name.clone());
                let step = self.scope(|lower| {
                    lower.values.insert(state, field(local.clone(), 0));
                    lower.values.insert(index, field(local.clone(), 1));
                    Ok(ScalarExpr::Tuple(vec![
                        lower.region(iteration)?,
                        apply("add", ty, vec![field(local.clone(), 1), one]),
                    ]))
                })?;
                return Ok(field(
                    ScalarExpr::Loop {
                        name,
                        initial: Box::new(ScalarExpr::Tuple(vec![initial, zero])),
                        condition: Box::new(apply("lt", ty, vec![field(local, 1), limit])),
                        step: Box::new(step),
                    },
                    0,
                ));
            }
            return Err(Error::Unsupported);
        }
        // Stored GPU values and collective results cannot be read back implicitly.
        if self.stored(source) {
            return Err(Error::Unsupported);
        }
        if self.facts.operation(source).is_some() {
            return Err(Error::Unsupported);
        }
        Err(error(format!(
            "host expression has no selected root or boundary: {source:?}"
        ))
        .into())
    }
    /// Emit a host read directly from the selected ABI and source declaration.
    fn parameter(&self, region: Value, index: i64, source: Value) -> Result<ScalarExpr> {
        let owner =
            self.facts.definition_name(region).ok_or_else(|| error("host parameter owner missing"))?;
        let definition = self
            .facts
            .program
            .source
            .defs
            .iter()
            .find(|d| d.name == owner)
            .ok_or_else(|| error("host parameter definition missing"))?;
        let crate::tlc::DefMeta::EntryPoint(entry) = &definition.meta else {
            return Err(error("host parameter is not an entry input").into());
        };
        let index_usize = usize::try_from(index).map_err(|_| error("invalid host parameter index"))?;
        let parameter = entry
            .declaration
            .params
            .get(index_usize)
            .ok_or_else(|| error("host parameter declaration missing"))?;
        let ty = self.facts.source_type(source).ok_or_else(|| error("host parameter type missing"))?;
        let ty = storage_value_type(&crate::types::canonical_storage_buffer_ty(ty));
        let abi = Query(&self.facts.program.graph).required("ParameterAbi", (region, index))?;
        if entry.data.param_bindings.get(index_usize).and_then(Option::as_ref).is_some()
            || extract_storage_binding(parameter).is_some()
        {
            return Err(Error::Unsupported);
        }
        if let Some(binding) = extract_uniform_binding(parameter) {
            return read(
                ScalarSource::Binding {
                    set: binding.set,
                    binding: binding.binding,
                },
                0,
                &ty,
                StorageLayout::Std140,
            );
        }
        if let Some(fields) = self.facts.enode("PushInput", abi) {
            let offset = self.facts.unsigned(fields[0], "push constant offset")?;
            return read(
                ScalarSource::PushConstant {
                    name: parameter.name.clone(),
                    offset,
                },
                0,
                &ty,
                StorageLayout::Std430,
            );
        }
        Err(Error::Unsupported)
    }

    fn stored(&self, source: Value) -> bool {
        self.facts
            .lookup("SelectedAccess", (source,))
            .is_some_and(|access| self.facts.enode("StoredRead", access).is_some())
    }
    fn term(&mut self, term: TermId) -> Result<ScalarExpr> {
        if let Some(value) = self.terms.get(&term) {
            return Ok(value.clone());
        }
        let selected = &self.facts.program.stage.selected;
        let (name, f) = selected.app(term)?;
        let context = selected.values[f[0]];
        let Some(ty) = self.facts.ty(selected.values[f[1]]) else {
            return Err(error("host expression type missing").into());
        };
        let value = match name {
            "ScalarLeaf" | "ScalarExecute" => self.boundary(context, selected.values[f[2]])?,
            "ScalarParameter" => {
                let Some(source) = self.facts.parameter(selected.values[f[2]], selected.integer(f[3])?)
                else {
                    return Err(error("selected host parameter missing").into());
                };
                self.boundary(context, source)?
            }
            "ScalarLiteral" => {
                let text = selected.text(f[2])?;
                let malformed = || error("invalid selected host literal");
                match scalar_type(ty) {
                    Some(ScalarType::I32) => ScalarExpr::I32(text.parse().map_err(|_| malformed())?),
                    Some(ScalarType::U32) => ScalarExpr::U32(text.parse().map_err(|_| malformed())?),
                    Some(ScalarType::F32) => ScalarExpr::F32(text.parse().map_err(|_| malformed())?),
                    Some(ScalarType::Bool) => ScalarExpr::Bool(text.parse().map_err(|_| malformed())?),
                    None => return Err(Error::Unsupported),
                }
            }
            "ScalarUnary" | "ScalarBinary" | "ScalarOp" => {
                let terms = selected.operation_arguments(term)?;
                let Some(&first) = terms.first() else {
                    return Err(error("host operation has no operands").into());
                };
                let (_, operand) = selected.app(first)?;
                let Some(argument) = self.facts.ty(selected.values[operand[1]]) else {
                    return Err(error("host operand type missing").into());
                };
                let (Some(argument), Some(result)) = (scalar_type(argument), scalar_type(ty)) else {
                    return Err(Error::Unsupported);
                };
                let op = operation(selected.operator(f[2], terms.len())?, argument, result)?;
                apply(
                    op,
                    argument,
                    terms.into_iter().map(|term| self.term(term)).collect::<Result<_>>()?,
                )
            }
            "ScalarTuple" | "ScalarVector" => ScalarExpr::Tuple(
                selected.arguments(f[2])?.into_iter().map(|term| self.term(term)).collect::<Result<_>>()?,
            ),
            "ScalarProject" => field(
                self.term(f[2])?,
                usize::try_from(selected.integer(f[3])?).map_err(|_| error("invalid host field index"))?,
            ),
            "ScalarCoerce" => {
                let (_, operand) = selected.app(f[2])?;
                let Some(from) = self.facts.ty(selected.values[operand[1]]) else {
                    return Err(error("host conversion type missing").into());
                };
                if from == ty {
                    self.term(f[2])?
                } else {
                    let (Some(from), Some(to)) = (scalar_type(from), scalar_type(ty)) else {
                        return Err(Error::Unsupported);
                    };
                    apply(&format!("to-{}", to.name()), from, vec![self.term(f[2])?])
                }
            }
            "ScalarChoice" => ScalarExpr::If {
                condition: Box::new(self.term(f[2])?),
                yes: Box::new(self.scope(|lower| lower.term(f[3]))?),
                no: Box::new(self.scope(|lower| lower.term(f[4]))?),
            },
            // The host IR has no callable-function representation. Leave an
            // extracted call on the GPU instead of changing the inline decision.
            "ScalarCall" | "ScalarInvoke" => return Err(Error::Unsupported),
            "ScalarInstruction" => {
                let Some(array) = self.facts.lookup("SourceLength", (selected.values[f[2]],)) else {
                    return Err(Error::Unsupported);
                };
                self.length(array)?
            }
            _ => return Err(error(format!("unresolved selected host constructor {name}")).into()),
        };
        // Keep aggregate structure visible to projections. Parameter aggregates
        // contain reads for every field; binding the whole tuple eagerly would
        // read unrelated fields even when only one component is needed.
        // Computed scalar children already have their own ordered bindings.
        if matches!(value, ScalarExpr::Tuple(_)) {
            self.terms.insert(term, value.clone());
            return Ok(value);
        }
        let name = self.name();
        self.bindings.push((name.clone(), value));
        let value = ScalarExpr::Local(name);
        self.terms.insert(term, value.clone());
        Ok(value)
    }
}
fn read(source: ScalarSource, offset: u32, ty: &Type, rules: StorageLayout) -> Result<ScalarExpr> {
    if let Some(ty) = scalar_type(ty) {
        return Ok(ScalarExpr::Parameter { source, offset, ty });
    }
    if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = ty {
        let Some(layout) = block_layout(ty, rules) else {
            return Err(error("host parameter layout missing").into());
        };
        return Ok(ScalarExpr::Tuple(
            fields
                .iter()
                .enumerate()
                .map(|(index, ty)| read(source.clone(), offset + layout.member_offsets[index], ty, rules))
                .collect::<Result<_>>()?,
        ));
    }
    if let Some(count) = ty.vec_size() {
        let Some(ty) = ty.elem_type() else {
            return Err(error("host vector element type missing").into());
        };
        let Some(stride) = type_byte_size(ty) else {
            return Err(Error::Unsupported);
        };
        return Ok(ScalarExpr::Tuple(
            (0..count)
                .map(|index| read(source.clone(), offset + index as u32 * stride, ty, rules))
                .collect::<Result<_>>()?,
        ));
    }
    Err(Error::Unsupported)
}
fn field(tuple: ScalarExpr, index: usize) -> ScalarExpr {
    if let ScalarExpr::Tuple(mut fields) = tuple {
        return fields.swap_remove(index);
    }
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
fn operation(
    tag: OpTag<BindingRef, FunctionId>,
    argument: ScalarType,
    result: ScalarType,
) -> Result<&'static str> {
    let op = match tag {
        OpTag::UnaryOp(op) => operator(op.symbol(), 1),
        OpTag::BinOp(op) => operator(op.symbol(), 2),
        OpTag::Intrinsic { id, overload_idx } => {
            let Some(overload) = catalog().get(id).overloads().get(overload_idx) else {
                return Err(error("host builtin overload missing").into());
            };
            builtin_op(&overload.lowering, argument, result)
        }
        _ => None,
    };
    let Some(op) = op else {
        return Err(Error::Unsupported);
    };
    Ok(op)
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
fn builtin_op(
    lowering: &BuiltinLowering,
    argument: ScalarType,
    result: ScalarType,
) -> Option<&'static str> {
    Some(match lowering {
        // Same-width signed/unsigned conversions preserve the 32 source bits.
        BuiltinLowering::PrimOp(PrimOp::Bitcast)
            if matches!(argument, ScalarType::I32 | ScalarType::U32)
                && matches!(result, ScalarType::I32 | ScalarType::U32) =>
        {
            if result == ScalarType::I32 {
                "to-i32"
            } else {
                "to-u32"
            }
        }
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
