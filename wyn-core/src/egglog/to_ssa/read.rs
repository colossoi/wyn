//! Borrowed queries into egglog; this view owns no source or planning facts.
use super::{error, OptimizeError, Optimized, Program};
use crate::builtins::catalog;
use crate::egglog::scalar::Selected;
use crate::op::{BinaryOperator, OpTag, UnaryOperator};
use crate::types::Type;
use crate::SymbolId;
use crate::{BindingRef, FunctionId};
use egglog_engine::sort::SetContainer;
use egglog_engine::{ast::Literal, Term, TermId};
use egglog_engine::{IntoValues, Read, Value};

pub(super) struct Facts<'a, 'source> {
    pub program: &'a Program<'source, Optimized>,
}
impl<'a, 'source> Facts<'a, 'source> {
    // Access declarations are loaded with this compiler module. Wrong names or
    // arities here are compiler programming errors, never missing user facts.
    pub fn lookup(&self, table: &str, keys: impl IntoValues) -> Option<Value> {
        match self.program.graph.read(|r| r.lookup(table, keys)) {
            Ok(value) => value,
            Err(error) => panic!("invalid native fact accessor {table}: {error}"),
        }
    }
    pub fn contains(&self, table: &str, keys: impl IntoValues) -> bool {
        match self.program.graph.read(|r| r.contains(table, keys)) {
            Ok(value) => value,
            Err(error) => panic!("invalid native fact predicate {table}: {error}"),
        }
    }
    pub fn constructor(&self, table: &str, keys: impl IntoValues) -> Option<Value> {
        match self.program.graph.read(|r| r.eclass_of(table, keys)) {
            Ok(value) => value,
            Err(error) => panic!("invalid native constructor accessor {table}: {error}"),
        }
    }
    pub fn enode(&self, name: &str, value: Value) -> Option<Vec<Value>> {
        let mut fields = None;
        match self
            .program
            .graph
            .read(|r| r.enodes_for_eclass(name, value, |row| fields = Some(row.children.to_vec())))
        {
            Ok(()) => fields,
            Err(error) => panic!("invalid native enode accessor {name}: {error}"),
        }
    }
    pub fn integer(&self, value: Value) -> i64 {
        self.program.graph.value_to_base::<i64>(value)
    }
    pub fn flag(&self, table: &str, value: Value) -> bool {
        self.lookup(table, (value,)).is_some_and(|v| self.program.graph.value_to_base::<bool>(v))
    }
    pub fn set(&self, table: &str, keys: impl IntoValues) -> Vec<Value> {
        self.lookup(table, keys)
            .and_then(|v| {
                self.program
                    .graph
                    .value_to_container::<SetContainer>(v)
                    .map(|set| set.data.iter().copied().collect())
            })
            .unwrap_or_default()
    }
    pub fn alias(&self, value: Value) -> Option<Value> {
        self.lookup("SsaAlias", (value,))
    }
    pub fn context(&self, value: Value) -> Option<Value> {
        self.lookup("SsaContext", (value,)).or_else(|| self.constructor("ScalarFunction", (value,)))
    }
    pub fn result(&self, value: Value) -> Option<Value> {
        self.lookup("SsaResult", (value,))
    }
    pub fn callable(&self, value: Value) -> Option<Value> {
        self.lookup("SsaCallable", (value,))
    }
    pub fn loop_state(&self, value: Value) -> Option<Value> {
        self.lookup("SsaLoopState", (value,))
    }
    pub fn iteration(&self, value: Value) -> Option<Value> {
        self.lookup("SsaIteration", (value,))
    }
    pub fn destination(&self, value: Value) -> Option<Value> {
        self.lookup("SsaDestination", (value,))
    }
    pub fn operation(&self, value: Value) -> Option<Value> {
        self.lookup("SsaOperation", (value,))
    }
    pub fn callback(&self, value: Value) -> Option<Value> {
        self.lookup("SsaCallback", (value,))
    }
    pub fn array_part(&self, value: Value) -> Option<Value> {
        self.lookup("SsaArrayPart", (value,))
    }
    pub fn definition_name(&self, scope: Value) -> Option<SymbolId> {
        let token = self.lookup("SsaDefinitionName", (scope,))?;
        Some(*self.program.identities.symbols.resolve(self.program.graph.value_to_base::<i64>(token)))
    }
    pub fn definition(&self, symbol: SymbolId) -> Option<Value> {
        self.lookup("SsaDefinition", (self.program.identities.symbols.get(&symbol)?,))
    }
    pub fn global_symbol(&self, value: Value) -> Option<SymbolId> {
        Some(*self.program.identities.symbols.resolve(self.integer(self.enode("SourceGlobal", value)?[0])))
    }
    pub fn formal_name(&self, value: Value) -> Option<SymbolId> {
        Some(*self.program.identities.symbols.resolve(self.integer(self.enode("SourceFormal", value)?[1])))
    }
    pub fn ty(&self, value: Value) -> Option<&'a Type> {
        Some(self.program.identities.types.resolve(self.integer(self.enode("TypeId", value)?[0])))
    }
    pub fn source_type(&self, value: Value) -> Option<&'a Type> {
        self.ty(self.lookup("SourceType", (value,))?)
    }
    pub fn parameter(&self, scope: Value, index: i64) -> Option<Value> {
        self.lookup("SsaParameter", (scope, index))
    }
    pub fn branches(&self, value: Value) -> Option<(Value, Value)> {
        Some((
            self.lookup("SsaBranchYes", (value,))?,
            self.lookup("SsaBranchNo", (value,))?,
        ))
    }
    pub fn loops(&self, value: Value) -> Option<(Value, Value)> {
        Some((
            self.lookup("SsaLoopHeader", (value,))?,
            self.lookup("SsaLoopIteration", (value,))?,
        ))
    }
    pub fn projection(&self, value: Value) -> Option<(Value, usize)> {
        Some((
            self.lookup("SsaProjectionBase", (value,))?,
            self.lookup("SsaProjectionIndex", (value,)).map(|v| self.integer(v))? as usize,
        ))
    }
    pub fn slice(&self, value: Value) -> Option<(Value, Value, Value)> {
        Some((
            self.lookup("SsaSliceBase", (value,))?,
            self.lookup("SsaSliceStart", (value,))?,
            self.lookup("SsaSliceEnd", (value,))?,
        ))
    }
    pub fn input(&self, operation: Value, index: i64) -> Option<Value> {
        self.lookup("SsaInputAt", (operation, index))
    }
    pub fn inputs(&self, operation: Value) -> Vec<(i64, Value)> {
        let count = self.lookup("SsaInputCount", (operation,)).map(|v| self.integer(v)).unwrap_or(0);
        (0..count).filter_map(|i| self.input(operation, i).map(|v| (i, v))).collect()
    }
    pub fn captures(&self, scope: Value) -> Vec<(Value, Value)> {
        self.set("SsaCaptures", (scope,))
            .into_iter()
            .filter_map(|formal| {
                self.lookup("SsaCaptureAt", (scope, formal)).map(|actual| (formal, actual))
            })
            .collect()
    }
    pub fn device_region(&self, scope: Value) -> bool {
        self.flag("SourceRegionDevice", scope)
    }
    pub fn device_operation(&self, op: Value) -> bool {
        self.contains("SsaDeviceOperation", (op,))
    }
    pub fn total(&self, context: Value, expression: Value) -> bool {
        self.contains("ScalarTotal", (context, expression))
    }
    pub fn placement(&self, context: Value, expression: Value, scope: Value) -> bool {
        self.contains("ScalarSelectedPlacement", (context, expression, scope))
    }
    pub fn dispatch_context(&self, operation: Value) -> Option<Value> {
        self.constructor("ScalarDispatch", (operation,))
    }
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
    pub fn operator(
        &self,
        term: TermId,
        arity: usize,
    ) -> Result<OpTag<BindingRef, FunctionId>, OptimizeError> {
        let name = self.text(term)?;
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
