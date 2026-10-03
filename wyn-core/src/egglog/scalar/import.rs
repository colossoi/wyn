//! Admit source identities, never cloned TLC expressions. Lexical substitutions
//! were resolved by the single source walk; helper substitution belongs to egglog.
use crate::builtins::{by_id, catalog, Purity};
use crate::egglog::query::Query;
use crate::egglog::source::{Identities, Term};
use crate::egglog::{OptimizeError, ScalarOptimization};
use crate::tlc::data::{ExplicitCapturesPayload, ExplicitClosurePayload};
use crate::tlc::{ArrayExpr, TermKind, VarRef};
use crate::types::{Type, TypeName};
use crate::{LookupMap, LookupSet};
use egglog_engine::{FullState, Value, Write};

type Cache = LookupMap<(Value, Value, bool), Value>;

pub(super) struct Importer<'graph, 'db, 'a, 'source> {
    sink: FullState<'graph, 'db>,
    identities: &'a Identities<'source>,
    facts: Query<'a>,
    cache: &'a mut Cache,
    templates: &'a mut LookupMap<Value, Value>,
    loading: LookupSet<(Value, Value, bool)>,
    policy: ScalarOptimization,
}

impl<'graph, 'db, 'a, 'source> Importer<'graph, 'db, 'a, 'source> {
    pub(super) fn new(
        sink: FullState<'graph, 'db>,
        identities: &'a Identities<'source>,
        facts: Query<'a>,
        cache: &'a mut Cache,
        templates: &'a mut LookupMap<Value, Value>,
        policy: ScalarOptimization,
    ) -> Self {
        Self {
            sink,
            identities,
            facts,
            cache,
            templates,
            loading: LookupSet::default(),
            policy,
        }
    }

    pub(super) fn root(
        &mut self,
        context: Value,
        region: Value,
        source: Value,
        expand: bool,
    ) -> Result<(), OptimizeError> {
        self.sink.add("ScalarActive", context)?;
        let value = self.value(context, source, expand)?;
        self.sink.add("ScalarRoot", (context, region, source, value))?;
        Ok(())
    }

    fn template_region(&self, context: Value) -> Option<Value> {
        self.templates.iter().find_map(|(&region, &ctx)| (ctx == context).then_some(region))
    }

    fn ty(&mut self, source: Value) -> Result<Value, OptimizeError> {
        let Some(ty) = self.facts.lookup("SourceType", (source,))? else {
            return Err(OptimizeError::Output("scalar source has no type".into()));
        };
        Ok(ty)
    }

    fn value(&mut self, context: Value, source: Value, expand: bool) -> Result<Value, OptimizeError> {
        let expand = expand || self.facts.rematerialized(source)?;
        let key = (context, source, expand);
        if let Some(&value) = self.cache.get(&key) {
            return Ok(value);
        }
        let ty = self.ty(source)?;
        if !self.loading.insert(key) {
            return Err(OptimizeError::Output(
                "cycle in admitted scalar source DAG".into(),
            ));
        }
        let value = if let Some((owner, index)) = self.facts.parameter(source)? {
            let value = self.sink.add("ScalarParameter", (context, ty, owner, index))?;
            value
        } else if let Some(actual) = self.facts.alias(source)? {
            self.value(
                context,
                actual,
                expand && !self.facts.flag("ScalarSourceDispatched", (actual,))?,
            )?
        } else if let Some((callee, arguments)) = self.facts.call(source)? {
            let arguments = arguments
                .iter()
                .map(|&argument| self.value(context, argument, false))
                .collect::<Result<Vec<_>, _>>()?;
            let arguments = self.args(context, &arguments)?;
            self.template(callee)?;
            self.sink.add("ScalarCall", (context, ty, source, callee, arguments))?
        } else if self.facts.operation(source)?.is_some() && !expand {
            self.opaque(source, ty, context, false)?
        } else if let Some(&(array, scope)) =
            self.identities.arrays.get(&source).filter(|(array, _)| !matches!(array, ArrayExpr::Var(_, _)))
        {
            self.array(context, source, scope, array, ty)?
        } else if let Some(&(term, scope)) = self.identities.origins.get(&source) {
            self.type_facts(ty, &term.ty)?;
            self.expression(context, source, scope, term, ty)?
        } else {
            self.opaque(source, ty, context, false)?
        };
        self.loading.remove(&key);
        self.cache.insert(key, value);
        Ok(value)
    }

    fn opaque(
        &mut self,
        source: Value,
        ty: Value,
        context: Value,
        execute: bool,
    ) -> Result<Value, OptimizeError> {
        let name = if execute { "ScalarExecute" } else { "ScalarLeaf" };
        let value = self.sink.add(name, (context, ty, source))?;
        Ok(value)
    }

    fn source(&self, scope: Value, term: &Term) -> Result<Value, OptimizeError> {
        let Some(&value) = self.identities.occurrences.get(&(scope, term.id)) else {
            return Err(OptimizeError::Output("missing scalar source occurrence".into()));
        };
        Ok(value)
    }

    fn child(&mut self, context: Value, scope: Value, term: &Term) -> Result<Value, OptimizeError> {
        let source = self.source(scope, term)?;
        // An admitted helper is a substitution template. Its own reproducible calls
        // may expand; stored results outside that helper remain opaque inputs.
        let expand = match self.template_region(context) {
            Some(region) => self.facts.contains("ScalarTemplateExpands", (region, source))?,
            None => false,
        };
        self.value(context, source, expand)
    }

    fn args(&mut self, context: Value, values: &[Value]) -> Result<Value, OptimizeError> {
        let mut args = self.sink.add("ScalarNil", context)?;
        for &value in values.iter().rev() {
            args = self.sink.add("ScalarCons", (context, value, args))?;
        }
        Ok(args)
    }

    fn type_facts(&mut self, key: Value, ty: &Type) -> Result<(), OptimizeError> {
        let name = match ty {
            Type::Constructed(TypeName::Int(bits), _) => {
                self.sink.add("ScalarInteger", key)?;
                format!("i{bits}")
            }
            Type::Constructed(TypeName::UInt(bits), _) => {
                self.sink.add("ScalarInteger", key)?;
                format!("u{bits}")
            }
            Type::Constructed(TypeName::Float(32), _) => "f32".into(),
            Type::Constructed(TypeName::Bool, _) => "bool".into(),
            _ => return Ok(()),
        };
        self.sink.add("ScalarType", (key, name.as_str()))?;
        Ok(())
    }

    fn template(&mut self, region: Value) -> Result<(), OptimizeError> {
        if self.policy == ScalarOptimization::Basic {
            return Ok(());
        }
        if !self.facts.contains("SourceInlineEligible", (region,))? || self.templates.contains_key(&region)
        {
            return Ok(());
        }
        let Some(source) = self.facts.lookup("SourceResult", (region,))? else {
            return Err(OptimizeError::Output("inline helper has no result".into()));
        };
        // Templates must not reuse cached stored results from the separately
        // scheduled function body. Only explicit substitution activates them.
        let context = self.sink.add("ScalarTemplate", region)?;
        self.templates.insert(region, context);
        self.sink.set("ScalarTemplateSafe", region, true)?;
        let body = self.value(context, source, true)?;
        self.sink.add("ScalarInlineBody", (region, body))?;
        Ok(())
    }

    fn expression(
        &mut self,
        context: Value,
        source: Value,
        scope: Value,
        term: &'source Term,
        ty: Value,
    ) -> Result<Value, OptimizeError> {
        match &term.kind {
            TermKind::IntLit(text) => Ok(self.sink.add("ScalarLiteral", (context, ty, text.as_str()))?),
            TermKind::FloatLit(value) => Ok(self.sink.add(
                "ScalarLiteral",
                (context, ty, value.to_bits().to_string().as_str()),
            )?),
            TermKind::BoolLit(value) => Ok(self.sink.add(
                "ScalarLiteral",
                (context, ty, if *value { "true" } else { "false" }),
            )?),
            TermKind::UnitLit => {
                let args = self.args(context, &[])?;
                self.sink.add("ScalarOperatorSafe", (ty, "unit"))?;
                Ok(self.sink.add("ScalarOp", (context, ty, "unit", args))?)
            }
            TermKind::ArrayExpr(_) => {
                let Some(array) = self.facts.alias(source)? else {
                    return Err(OptimizeError::Output("array expression has no atom".into()));
                };
                self.value(context, array, false)
            }
            TermKind::Index { array, index } => {
                let array = self.child(context, scope, array)?;
                let index = self.child(context, scope, index)?;
                let args = self.args(context, &[array, index])?;
                Ok(self.sink.add("ScalarInstruction", (context, ty, source, "index", args))?)
            }
            TermKind::App { func, args } => {
                if matches!(&func.kind, TermKind::Var(VarRef::Builtin { id, .. }) if *id == catalog().known().length || *id == catalog().known().slice)
                {
                    return self.opaque(source, ty, context, true);
                }
                let mut retained = false;
                let operator = match &func.kind {
                    TermKind::BinOp(op) => Some((op.op.symbol().to_string(), op.op.is_speculatable())),
                    TermKind::UnOp(op) => Some((op.op.symbol().to_string(), true)),
                    TermKind::Var(VarRef::Builtin { id, overload_idx }) => {
                        let builtin = by_id(*id);
                        let Some(overload) = builtin.overloads().get(*overload_idx) else {
                            return Err(OptimizeError::Output("invalid scalar builtin overload".into()));
                        };
                        if builtin.raw.purity != Purity::Pure || !overload.lowering.is_reusable() {
                            retained = true;
                        }
                        Some((
                            format!("builtin:{}:{overload_idx}", builtin.dispatch_name()),
                            overload.lowering.is_speculatable(),
                        ))
                    }
                    _ => None,
                };
                let function = self.source(scope, func)?;
                let callee = self.facts.callable(function)?;
                if operator.is_none() && callee.is_none() {
                    return self.opaque(source, ty, context, true);
                }
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    values.push(self.child(context, scope, arg)?);
                }
                if let Some((op, safe)) = operator {
                    if retained {
                        let arguments = self.args(context, &values)?;
                        return Ok(self
                            .sink
                            .add("ScalarInstruction", (context, ty, source, op.as_str(), arguments))?);
                    }
                    if safe {
                        self.sink.add("ScalarOperatorSafe", (ty, op.as_str()))?;
                    }
                    return Ok(match values.as_slice() {
                        &[x] => self.sink.add("ScalarUnary", (context, ty, op.as_str(), x))?,
                        &[x, y] => self.sink.add("ScalarBinary", (context, ty, op.as_str(), x, y))?,
                        _ => {
                            let arguments = self.args(context, &values)?;
                            self.sink.add("ScalarOp", (context, ty, op.as_str(), arguments))?
                        }
                    });
                }
                let Some(callee) = callee else {
                    return Err(OptimizeError::Output("missing admitted call target".into()));
                };
                let arguments = self.args(context, &values)?;
                if !self.facts.duplicable(source)? {
                    return Ok(self.sink.add("ScalarCall", (context, ty, source, callee, arguments))?);
                }
                self.template(callee)?;
                Ok(self.sink.add("ScalarInvoke", (context, ty, callee, arguments))?)
            }
            TermKind::Tuple(fields) | TermKind::VecLit(fields) => {
                let mut values = Vec::with_capacity(fields.len());
                for field in fields {
                    values.push(self.child(context, scope, field)?);
                }
                let args = self.args(context, &values)?;
                let name =
                    if matches!(term.kind, TermKind::Tuple(_)) { "ScalarTuple" } else { "ScalarVector" };
                Ok(self.sink.add(name, (context, ty, args))?)
            }
            TermKind::TupleProj { tuple, idx } => {
                let tuple = self.child(context, scope, tuple)?;
                Ok(self.sink.add("ScalarProject", (context, ty, tuple, *idx as i64))?)
            }
            TermKind::Coerce { inner, .. } => {
                let inner = self.child(context, scope, inner)?;
                Ok(self.sink.add("ScalarCoerce", (context, ty, inner))?)
            }
            TermKind::If { cond, .. }
                if !self.facts.operation(source)?.is_some() || self.facts.duplicable(source)? =>
            {
                let Some(branch) = self.facts.row("SourceBranch", |r| r[0] == source)? else {
                    return Err(OptimizeError::Output("scalar conditional has no branches".into()));
                };
                let Some(yes_result) = self.facts.lookup("SourceResult", (branch[2],))? else {
                    return Err(OptimizeError::Output("missing then result".into()));
                };
                let Some(no_result) = self.facts.lookup("SourceResult", (branch[3],))? else {
                    return Err(OptimizeError::Output("missing else result".into()));
                };
                let c = self.child(context, scope, cond)?;
                let a = self.value(context, yes_result, self.template_region(context).is_some())?;
                let b = self.value(context, no_result, self.template_region(context).is_some())?;
                Ok(self.sink.add("ScalarChoice", (context, ty, c, a, b))?)
            }
            TermKind::Var(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Lambda(_)
            | TermKind::Closure(_)
            | TermKind::Let { .. }
            | TermKind::Extern(_)
            | TermKind::If { .. }
            | TermKind::Loop { .. }
            | TermKind::Soac(_) => self.opaque(source, ty, context, true),
        }
    }

    fn array(
        &mut self,
        context: Value,
        source: Value,
        scope: Value,
        array: &'source ArrayExpr<ExplicitClosurePayload, ExplicitCapturesPayload>,
        ty: Value,
    ) -> Result<Value, OptimizeError> {
        let (operator, values) = match array {
            ArrayExpr::Literal(elements) => (
                "array",
                elements.iter().map(|e| self.child(context, scope, e)).collect::<Result<Vec<_>, _>>()?,
            ),
            ArrayExpr::Range { start, len, step } => {
                let mut values = vec![
                    self.child(context, scope, start)?,
                    self.child(context, scope, len)?,
                ];
                if let Some(step) = step {
                    values.push(self.child(context, scope, step)?);
                }
                ("range", values)
            }
            ArrayExpr::Zip(parts) => {
                let mut values = Vec::new();
                for i in 0..parts.len() {
                    let Some(part) = self
                        .facts
                        .row("SourceArrayPart", |r| {
                            r[0] == source && self.facts.0.value_to_base::<i64>(r[1]) == i as i64
                        })?
                        .map(|r| r[2])
                    else {
                        return Err(OptimizeError::Output("zip component is missing".into()));
                    };
                    values.push(self.value(context, part, false)?);
                }
                let args = self.args(context, &values)?;
                return Ok(self.sink.add("ScalarTuple", (context, ty, args))?);
            }
            ArrayExpr::Var(_, _) => return self.opaque(source, ty, context, false),
        };
        let args = self.args(context, &values)?;
        self.sink.add("ScalarOperatorSafe", (ty, operator))?;
        Ok(self.sink.add("ScalarOp", (context, ty, operator, args))?)
    }
}
