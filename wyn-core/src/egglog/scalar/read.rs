//! Cost-based extraction of typed terms back into the interned sidecar DAG.
use super::term::{app, key};
use super::{error, intern_expr, OptimizeError};
use crate::egglog::data::{Array, ExprId, ExprKind, Ir};
use egglog_engine::ast::Literal;
use egglog_engine::extract::{CostModel, Extractor, TreeAdditiveCostModel};
use egglog_engine::sort::S;
use egglog_engine::{ArcSort, EGraph, Enode, Function, Term, TermDag, Value};
use std::collections::BTreeMap;

struct Cost;
impl CostModel<u64> for Cost {
    fn fold(&self, _head: &str, children: &[u64], head: u64) -> u64 {
        children.iter().fold(head, |a, b| a.saturating_add(*b))
    }
    fn enode_cost(&self, graph: &EGraph, f: &Function, e: &Enode<'_>) -> u64 {
        TreeAdditiveCostModel::default().enode_cost(graph, f, e)
    }
    fn base_value_cost(&self, graph: &EGraph, sort: &ArcSort, value: Value) -> u64 {
        if sort.name() == "String" {
            let text = graph.value_to_base::<S>(value);
            if matches!(text.as_str(), "/" | "%" | "//" | "%%" | "**") {
                return 8;
            }
        }
        1
    }
}

pub(super) fn extract(graph: &EGraph, data: &mut Ir) -> Result<BTreeMap<ExprId, ExprId>, OptimizeError> {
    let Some(sort) = graph.get_sort_by_name("Expr") else {
        return Err(error("missing expression sort"));
    };
    let extractor = Extractor::compute_costs_from_rootsorts(None, graph, Cost);
    let mut dag = TermDag::default();
    let mut reader = Reader {
        data,
        memo: BTreeMap::new(),
    };
    let mut replacements = BTreeMap::new();
    let mut result = Ok(());
    graph.function_entries_while("SourceExpression", |entry| {
        result = (|| {
            let id = u32::try_from(graph.value_to_base::<i64>(entry.inputs[0]))
                .map(ExprId::from)
                .map_err(|_| error("invalid source expression ID"))?;
            let Some((_, node)) =
                extractor.extract_best_with_sort(graph, &mut dag, entry.output, sort.clone())
            else {
                return Err(error(&format!("no finite extraction for {id:?}")));
            };
            let value = reader.value(&dag, node)?;
            if id != value {
                replacements.insert(id, value);
            }
            Ok(())
        })();
        result.is_ok()
    })?;
    result?;
    Ok(replacements)
}

struct Reader<'a> {
    data: &'a mut Ir,
    memo: BTreeMap<usize, ExprId>,
}
impl Reader<'_> {
    fn value(&mut self, dag: &TermDag, id: usize) -> Result<ExprId, OptimizeError> {
        if let Some(&v) = self.memo.get(&id) {
            return Ok(v);
        }
        let args = app(dag, id, "Typed", 2)?;
        let ty = key(dag, args[0], "TypeId")?;
        let Term::App(tag, a) = dag.get(args[1]) else {
            return Err(error("expected expression node"));
        };
        let kind = match (tag.as_str(), a.as_slice()) {
            ("Global", &[x]) => ExprKind::Global(key(dag, x, "SymbolId")?),
            ("Parameter", &[x]) => ExprKind::Parameter(key(dag, x, "ParameterId")?),
            ("Builtin", &[x]) => ExprKind::Builtin(key(dag, x, "BuiltinId")?),
            ("Extern", &[x]) => ExprKind::Extern(key(dag, x, "ExternId")?),
            ("OperationResult", &[x]) => ExprKind::OperationResult(key(dag, x, "OperationId")?),
            ("Lambda", &[x]) => ExprKind::Lambda(key(dag, x, "RegionId")?),
            ("BinOp", &[x]) => ExprKind::BinOp(string(dag, x)?),
            ("UnOp", &[x]) => ExprKind::UnOp(string(dag, x)?),
            ("Int", &[x]) => ExprKind::Int(string(dag, x)?),
            ("FloatBits", &[x]) => {
                ExprKind::FloatBits(u32::try_from(integer(dag, x)?).map_err(|_| error("float bits"))?)
            }
            ("Bool", &[x]) => match dag.get(x) {
                Term::Lit(Literal::Bool(b)) => ExprKind::Bool(*b),
                _ => return Err(error("boolean literal")),
            },
            ("UnitValue", &[]) => ExprKind::Unit,
            ("PureApp", &[f, xs]) => ExprKind::PureApp {
                function: self.value(dag, f)?,
                args: self.values(dag, xs)?,
            },
            ("Tuple", &[xs]) => ExprKind::Tuple(self.values(dag, xs)?),
            ("Vector", &[xs]) => ExprKind::Vector(self.values(dag, xs)?),
            ("Coerce", &[x]) => ExprKind::Coerce(self.value(dag, x)?),
            ("Project", &[x, i]) => ExprKind::Project {
                tuple: self.value(dag, x)?,
                index: usize::try_from(integer(dag, i)?).map_err(|_| error("field index"))?,
            },
            ("Select", &[c, a, b]) => ExprKind::If {
                condition: self.value(dag, c)?,
                then_value: self.value(dag, a)?,
                else_value: self.value(dag, b)?,
            },
            ("Closure", &[code, count, xs]) => ExprKind::Closure {
                code: key(dag, code, "SymbolId")?,
                param_count: usize::try_from(integer(dag, count)?).map_err(|_| error("closure arity"))?,
                captures: self.values(dag, xs)?,
            },
            ("ArrayValue", &[x]) => ExprKind::Array(self.array(dag, x)?),
            _ => return Err(error(&format!("unknown extracted node {tag}"))),
        };
        let value = intern_expr(self.data, ty, kind);
        self.memo.insert(id, value);
        Ok(value)
    }
    fn values(&mut self, dag: &TermDag, id: usize) -> Result<Vec<ExprId>, OptimizeError> {
        vector(dag, id)?.iter().map(|&id| self.value(dag, id)).collect()
    }
    fn array(&mut self, dag: &TermDag, id: usize) -> Result<Array, OptimizeError> {
        let Term::App(tag, a) = dag.get(id) else {
            return Err(error("array node"));
        };
        Ok(match (tag.as_str(), a.as_slice()) {
            ("ArrayInput", &[x]) => Array::Value(self.value(dag, x)?),
            ("ArrayLiteral", &[xs]) => Array::Literal(self.values(dag, xs)?),
            ("Zip", &[xs]) => {
                Array::Zip(vector(dag, xs)?.iter().map(|&x| self.array(dag, x)).collect::<Result<_, _>>()?)
            }
            ("Range", &[a, n]) => Array::Range {
                start: self.value(dag, a)?,
                len: self.value(dag, n)?,
                step: None,
            },
            ("StridedRange", &[a, n, s]) => Array::Range {
                start: self.value(dag, a)?,
                len: self.value(dag, n)?,
                step: Some(self.value(dag, s)?),
            },
            _ => return Err(error("unknown extracted array")),
        })
    }
}
fn vector(dag: &TermDag, id: usize) -> Result<&[usize], OptimizeError> {
    match dag.get(id) {
        Term::App(tag, args) if tag == "vec-of" || tag == "vec-empty" => Ok(args),
        _ => Err(error("expected extracted vector")),
    }
}
fn integer(dag: &TermDag, id: usize) -> Result<i64, OptimizeError> {
    match dag.get(id) {
        Term::Lit(Literal::Int(v)) => Ok(*v),
        _ => Err(error("expected integer")),
    }
}
fn string(dag: &TermDag, id: usize) -> Result<String, OptimizeError> {
    match dag.get(id) {
        Term::Lit(Literal::String(v)) => Ok(v.clone()),
        _ => Err(error("expected string")),
    }
}
