//! Immutable instruction details. Operands and evaluation occurrences live in egglog.
use super::{output_error, OptimizeError};
use crate::builtins::{by_id, catalog, BuiltinId};
use crate::op::{BinaryOperator, OpTag, UnaryOperator};
use crate::{BindingRef, FunctionId};
use egglog_engine::ast::{Literal, Span};
use egglog_engine::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog_engine::prelude::{add_base_sort, BaseSort};
use egglog_engine::sort::{BaseValues, Boxed, I64Sort, StringSort, S};
use egglog_engine::{Core, EGraph, Primitive, PurePrim, PureState, TermDag, TermId, Value, Write};
use std::borrow::Cow;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Payload {
    Binary(BinaryOperator),
    Unary(UnaryOperator),
    Intrinsic {
        id: BuiltinId,
        overload: usize,
    },
    Unit,
    Vector,
    Tuple,
    Project(u32),
    Index,
    IndexCast,
    Select,
}

impl Payload {
    // Decode textual fixtures and diagnostic dumps at the boundary, using the
    // same operator and builtin identities as TLC and SSA.
    pub fn parse(name: &str, arity: usize) -> Option<Self> {
        Some(match name {
            "unit" => Self::Unit,
            "vector" => Self::Vector,
            "tuple" => Self::Tuple,
            "vector-index" => Self::Index,
            "index-cast" => Self::IndexCast,
            "select" => Self::Select,
            _ if name.starts_with("project:") => Self::Project(name[8..].parse().ok()?),
            _ => {
                if let Some((name, overload)) =
                    name.strip_prefix("builtin:").and_then(|name| name.rsplit_once(':'))
                {
                    let builtin = catalog().lookup_by_any_name(name)?;
                    let overload = overload.parse().ok()?;
                    builtin.overloads().get(overload)?;
                    Self::Intrinsic {
                        id: builtin.id,
                        overload,
                    }
                } else if let Some(builtin) = catalog().lookup_by_any_name(name) {
                    builtin.overloads().first()?;
                    Self::Intrinsic {
                        id: builtin.id,
                        overload: 0,
                    }
                } else if arity == 1 {
                    Self::Unary(UnaryOperator::try_from(name).ok()?)
                } else {
                    Self::Binary(BinaryOperator::try_from(name).ok()?)
                }
            }
        })
    }

    pub fn spelling(self) -> Cow<'static, str> {
        match self {
            Self::Binary(op) => op.symbol().into(),
            Self::Unary(op) => op.symbol().into(),
            Self::Intrinsic { id, overload } => {
                format!("builtin:{}:{overload}", by_id(id).dispatch_name()).into()
            }
            Self::Unit => "unit".into(),
            Self::Vector => "vector".into(),
            Self::Tuple => "tuple".into(),
            Self::Project(index) => format!("project:{index}").into(),
            Self::Index => "vector-index".into(),
            Self::IndexCast => "index-cast".into(),
            Self::Select => "select".into(),
        }
    }

    pub fn instruction(self) -> Option<OpTag<BindingRef, FunctionId>> {
        Some(match self {
            Self::Binary(op) => OpTag::BinOp(op),
            Self::Unary(op) => OpTag::UnaryOp(op),
            Self::Intrinsic { id, overload } => OpTag::Intrinsic {
                id,
                overload_idx: overload,
            },
            Self::Unit => OpTag::Unit,
            Self::Vector => OpTag::Vector,
            Self::Tuple => OpTag::Tuple,
            Self::Project(index) => OpTag::Project { index },
            Self::Index => OpTag::Index,
            Self::IndexCast | Self::Select => return None,
        })
    }
}

pub(super) fn read(graph: &EGraph, value: Value) -> Payload {
    graph.value_to_base::<Boxed<Payload>>(value).0
}

pub(super) fn register(graph: &mut EGraph) -> Result<(), OptimizeError> {
    add_base_sort(graph, PayloadSort, Span::Panic).map_err(|error| output_error(error.to_string()))?;
    graph.add_pure_primitive(ParsePayload, None);
    Ok(())
}

pub(super) fn publish_properties(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.update(|mut sink| {
        for (op, identity, left_identity) in [
            (BinaryOperator::Add, 0_i64, true),
            (BinaryOperator::Subtract, 0, false),
            (BinaryOperator::Multiply, 1, true),
            (BinaryOperator::BitwiseOr, 0, true),
            (BinaryOperator::BitwiseXor, 0, true),
        ] {
            let payload = sink.base_to_value(Boxed(Payload::Binary(op)));
            sink.add("IntegerIdentity", (payload, identity))?;
            if left_identity {
                sink.add("LeftIntegerIdentity", (payload, identity))?;
            }
        }
        for op in [
            BinaryOperator::Add,
            BinaryOperator::BitwiseAnd,
            BinaryOperator::BitwiseOr,
            BinaryOperator::BitwiseXor,
        ] {
            let payload = sink.base_to_value(Boxed(Payload::Binary(op)));
            sink.add("NativeIntegerOpcode", payload)?;
        }
        for (property, payload) in [
            ("AdditionPayload", Payload::Binary(BinaryOperator::Add)),
            ("SubtractionPayload", Payload::Binary(BinaryOperator::Subtract)),
            ("MultiplicationPayload", Payload::Binary(BinaryOperator::Multiply)),
            ("SelectPayload", Payload::Select),
            ("IndexCastPayload", Payload::IndexCast),
        ] {
            let payload = sink.base_to_value(Boxed(payload));
            sink.add(property, payload)?;
        }
        Ok(())
    })?;
    Ok(())
}

#[derive(Debug)]
pub(super) struct PayloadSort;

impl BaseSort for PayloadSort {
    type Base = Boxed<Payload>;

    fn name(&self) -> &str {
        "InstructionPayload"
    }

    fn reconstruct_termdag(&self, values: &BaseValues, value: Value, dag: &mut TermDag) -> TermId {
        let payload = values.unwrap::<Boxed<Payload>>(value).0;
        let name = dag.lit(Literal::String(payload.spelling().into_owned()));
        let arity = dag.lit(Literal::Int(if matches!(payload, Payload::Unary(_)) {
            1
        } else {
            2
        }));
        dag.app("wyn-payload".into(), vec![name, arity])
    }
}

#[derive(Clone)]
struct ParsePayload;

impl Primitive for ParsePayload {
    fn name(&self) -> &str {
        "wyn-payload"
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![
                StringSort.to_arcsort(),
                I64Sort.to_arcsort(),
                PayloadSort.to_arcsort(),
            ],
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for ParsePayload {
    fn apply<'a, 'db>(&self, state: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let [name, arity] = *args else {
            unreachable!("payload decoder requires a name and arity")
        };
        let name = state.value_to_base::<S>(name);
        let arity = usize::try_from(state.value_to_base::<i64>(arity)).ok()?;
        Some(state.base_to_value(Boxed(Payload::parse(name.as_str(), arity)?)))
    }
}

#[cfg(test)]
#[path = "payload_tests.rs"]
mod tests;
