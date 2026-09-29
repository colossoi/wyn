//! Admit source identities, never cloned TLC expressions. Lexical substitutions
//! were resolved by the single source walk; helper substitution belongs to egglog.
use super::Facts;
use crate::builtins::{by_id, Purity};
use crate::egglog::source::{Identities, Term};
use crate::egglog::OptimizeError;
use crate::tlc::{TermKind, VarRef};
use crate::types::{Type, TypeName};
use crate::{LookupMap, LookupSet};
use egglog_engine::{FullState, Value, Write};

type Cache = LookupMap<(Value, Value, bool), Value>;

pub(super) struct Importer<'graph, 'db, 'a, 'source> {
    sink: FullState<'graph, 'db>,
    identities: &'a Identities<'source>,
    facts: &'a Facts,
    cache: &'a mut Cache,
    templates: &'a mut LookupMap<Value, Value>,
    loading: LookupSet<(Value, Value, bool)>,
}

impl<'graph, 'db, 'a, 'source> Importer<'graph, 'db, 'a, 'source> {
    pub(super) fn new(
        sink: FullState<'graph, 'db>,
        identities: &'a Identities<'source>,
        facts: &'a Facts,
        cache: &'a mut Cache,
        templates: &'a mut LookupMap<Value, Value>,
    ) -> Self {
        Self {
            sink,
            identities,
            facts,
            cache,
            templates,
            loading: LookupSet::default(),
        }
    }

    pub(super) fn region(&mut self, context: Value, region: Value) -> Result<(), OptimizeError> {
        if let Some(roots) = self.facts.roots.get(&region) {
            for &source in roots {
                let value = self.value(context, source, !self.facts.dispatched.contains(&source))?;
                self.sink.add("ScalarRoot", (context, region, source, value))?;
            }
        }
        Ok(())
    }

    pub(super) fn root(
        &mut self,
        context: Value,
        region: Value,
        source: Value,
        expand: bool,
    ) -> Result<(), OptimizeError> {
        let value = self.value(context, source, expand)?;
        self.sink.add("ScalarRoot", (context, region, source, value))?;
        Ok(())
    }

    fn template_region(&self, context: Value) -> Option<Value> {
        self.templates.iter().find_map(|(&region, &ctx)| (ctx == context).then_some(region))
    }

    fn ty(&mut self, source: Value) -> Result<Value, OptimizeError> {
        let Some(&ty) = self.facts.types.get(&source) else {
            return Err(OptimizeError::Output("scalar source has no type".into()));
        };
        Ok(ty)
    }

    fn value(&mut self, context: Value, source: Value, expand: bool) -> Result<Value, OptimizeError> {
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
        let value = if let Some(&(owner, index)) = self.facts.parameters.get(&source) {
            let value = self.sink.add("ScalarParameter", (context, ty, owner, index))?;
            value
        } else if self.facts.operations.contains(&source) && !expand {
            self.leaf(source, ty, context)?
        } else if let Some(&(term, scope)) = self.identities.origins.get(&source) {
            self.type_facts(ty, &term.ty)?;
            self.expression(context, source, scope, term, ty)?
        } else {
            self.leaf(source, ty, context)?
        };
        self.loading.remove(&key);
        self.cache.insert(key, value);
        Ok(value)
    }

    fn leaf(&mut self, source: Value, ty: Value, context: Value) -> Result<Value, OptimizeError> {
        let value = self.sink.add("ScalarLeaf", (context, ty, source))?;
        let owner = self.facts.owners.get(&source).copied();
        if let Some(owner) = owner {
            if let Some(region) = self.template_region(context) {
                if super::home(owner, self.identities, self.facts) == region {
                    self.sink.set("ScalarTemplateSafe", region, false)?;
                }
            }
        }
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
        let expand = self.template_region(context).is_some_and(|region| {
            self.facts
                .owners
                .get(&source)
                .is_some_and(|&owner| super::home(owner, self.identities, self.facts) == region)
        }) && self.facts.duplicable.contains(&source);
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
        if !self.facts.eligible.contains(&region) || self.templates.contains_key(&region) {
            return Ok(());
        }
        let Some(&source) = self.facts.results.get(&region) else {
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
            TermKind::App { func, args } => {
                let operator = match &func.kind {
                    TermKind::BinOp(op) => Some((op.op.symbol().to_string(), op.op.is_speculatable())),
                    TermKind::UnOp(op) => Some((op.op.symbol().to_string(), true)),
                    TermKind::Var(VarRef::Builtin { id, overload_idx }) => {
                        let builtin = by_id(*id);
                        let Some(overload) = builtin.overloads().get(*overload_idx) else {
                            return Err(OptimizeError::Output("invalid scalar builtin overload".into()));
                        };
                        if builtin.raw.purity != Purity::Pure || !overload.lowering.is_reusable() {
                            return self.leaf(source, ty, context);
                        }
                        Some((
                            format!("builtin:{}:{overload_idx}", builtin.dispatch_name()),
                            overload.lowering.is_speculatable(),
                        ))
                    }
                    _ => None,
                };
                let function = self.source(scope, func)?;
                let callee = self.facts.callable.get(&function).copied();
                if operator.is_none() && (!self.facts.duplicable.contains(&source) || callee.is_none()) {
                    return self.leaf(source, ty, context);
                }
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    values.push(self.child(context, scope, arg)?);
                }
                let arguments = self.args(context, &values)?;
                if let Some((op, safe)) = operator {
                    if safe {
                        self.sink.add("ScalarOperatorSafe", (ty, op.as_str()))?;
                    }
                    return Ok(self.sink.add("ScalarOp", (context, ty, op.as_str(), arguments))?);
                }
                let Some(callee) = callee else {
                    return Err(OptimizeError::Output("missing admitted call target".into()));
                };
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
                if !self.facts.operations.contains(&source) || self.facts.duplicable.contains(&source) =>
            {
                let Some(&(yes, no)) = self.facts.branches.get(&source) else {
                    return Err(OptimizeError::Output("scalar conditional has no branches".into()));
                };
                let Some(&yes_result) = self.facts.results.get(&yes) else {
                    return Err(OptimizeError::Output("missing then result".into()));
                };
                let Some(&no_result) = self.facts.results.get(&no) else {
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
            | TermKind::UnitLit
            | TermKind::Extern(_)
            | TermKind::If { .. }
            | TermKind::Loop { .. }
            | TermKind::Soac(_)
            | TermKind::ArrayExpr(_)
            | TermKind::Index { .. } => self.leaf(source, ty, context),
        }
    }
}
