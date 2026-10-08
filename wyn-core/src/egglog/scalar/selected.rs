//! Decode the native extracted DAG for host and shader emission.
use crate::builtins::catalog;
use crate::egglog::{output_error as error, OptimizeError};
use crate::op::{BinaryOperator, OpTag, UnaryOperator};
use crate::{BindingRef, FunctionId, LookupMap};
use egglog_engine::{ast::Literal, Term, TermDag, TermId, Value};

/// Egglog's own extracted DAG, retained unchanged for direct SSA emission.
pub(in crate::egglog) struct Selected {
    pub dag: TermDag,
    pub values: Vec<Value>,
    pub roots: LookupMap<(Value, Value), TermId>,
}

// Borrowed decoding of egglog's extracted DAG, shared by host and SSA emission.
impl Selected {
    pub fn app(&self, term: TermId) -> Result<(&str, &[TermId]), OptimizeError> {
        let Term::App(name, fields) = self.dag.get(term) else {
            return Err(error("selected scalar is not a constructor"));
        };
        Ok((name, fields))
    }
    pub fn text(&self, term: TermId) -> Result<&str, OptimizeError> {
        let Term::Lit(Literal::String(text)) = self.dag.get(term) else {
            return Err(error("expected selected scalar text"));
        };
        Ok(text)
    }
    pub fn integer(&self, term: TermId) -> Result<i64, OptimizeError> {
        let Term::Lit(Literal::Int(value)) = self.dag.get(term) else {
            return Err(error("expected selected scalar integer"));
        };
        Ok(*value)
    }
    pub fn arguments(&self, mut term: TermId) -> Result<Vec<TermId>, OptimizeError> {
        let mut args = Vec::new();
        loop {
            let (name, fields) = self.app(term)?;
            match name {
                "ScalarNil" => return Ok(args),
                "ScalarCons" => {
                    args.push(fields[1]);
                    term = fields[2];
                }
                _ => return Err(error("unresolved scalar substitution")),
            }
        }
    }
    pub fn projected_array(&self, mut term: TermId) -> Result<(TermId, Vec<usize>), OptimizeError> {
        let mut path = Vec::new();
        loop {
            let (name, fields) = self.app(term)?;
            if name != "ScalarProject" {
                path.reverse();
                return Ok((term, path));
            }
            path.push(usize::try_from(self.integer(fields[3])?).map_err(|_| error("invalid projection"))?);
            term = fields[2];
        }
    }
    pub fn operation_arguments(&self, term: TermId) -> Result<Vec<TermId>, OptimizeError> {
        let (name, fields) = self.app(term)?;
        match name {
            "ScalarUnary" => Ok(vec![fields[3]]),
            "ScalarBinary" => Ok(fields[3..5].to_vec()),
            "ScalarOp" => self.arguments(fields[3]),
            _ => Err(error("expected scalar operation")),
        }
    }
    pub fn operator(
        &self,
        term: TermId,
        arity: usize,
    ) -> Result<OpTag<BindingRef, FunctionId>, OptimizeError> {
        let name = self.text(term)?;
        match name {
            "unit" => return Ok(OpTag::Unit),
            "array" => return Ok(OpTag::ArrayLit(arity)),
            "range" => return Ok(OpTag::ArrayRange { has_step: arity == 3 }),
            "index" => return Ok(OpTag::Index),
            _ => {}
        }
        if let Some(name) = name.strip_prefix("builtin:") {
            let Some((name, index)) = name.rsplit_once(':') else {
                return Err(error("invalid builtin identity"));
            };
            let Some(builtin) = catalog().lookup_by_any_name(name) else {
                return Err(error("missing selected builtin"));
            };
            Ok(OpTag::Intrinsic {
                id: builtin.id,
                overload_idx: index.parse().map_err(|_| error("invalid builtin overload"))?,
            })
        } else if arity == 1 {
            Ok(OpTag::UnaryOp(
                UnaryOperator::try_from(name).map_err(|_| error("invalid unary operator"))?,
            ))
        } else {
            Ok(OpTag::BinOp(
                BinaryOperator::try_from(name).map_err(|_| error("invalid binary operator"))?,
            ))
        }
    }
}
