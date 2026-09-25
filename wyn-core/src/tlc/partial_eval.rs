//! Stack-based partial evaluator for TLC.
//!
//! Simpler than NBE-style: collect application spines, evaluate args,
//! apply when we have enough arguments (using arity metadata).

use super::data::Empty;
use super::ownership::OwnershipValidated;
use super::run::UnpinnedPolymorphic;
use super::{
    Def, Lambda, LoopKind, Program, RewriteDecision, Term, TermId, TermIdSource, TermKind, TermRewriter,
    VarRef,
};
use crate::ast::{BinaryOp, Span, TypeName, UnaryOp};
use crate::builtins;
use crate::builtins::{by_id, Purity};
use crate::constant_eval::{self, Constant};
use crate::op::BinaryOperator;
use crate::scalar_eval::{self, Scalar};
use crate::types::TypeExt;
use crate::LookupMap;
use crate::LookupSet;
use crate::SymbolId;
use polytype::Type;

/// TLC after partial evaluation.
#[derive(Debug, Clone, Copy)]
pub enum PartialEvaledTag {}
pub type PartialEvaled =
    super::Program<PartialEvaledTag, UnpinnedPolymorphic, super::context::TransformedGlobal>;

/// Consume a validated TLC program and rebuild its definitions from the
/// evaluator's residual terms.
pub fn partial_eval(program: OwnershipValidated) -> PartialEvaled {
    program.assert_flat_apps();
    let Program {
        defs,
        symbols,
        mut term_ids,
        global_context,
        state: _,
    } = program;
    let definitions = defs
        .iter()
        .map(|def| {
            (
                def.name,
                DefinitionTemplate {
                    arity: def.arity,
                    body: def.body.clone(),
                    constant_candidate: matches!(&def.meta, super::DefMeta::Function) && def.arity == 0,
                },
            )
        })
        .collect();
    let mut evaluator = PartialEvaluator::new(definitions, &mut term_ids);
    evaluator.discover_global_constants();
    let defs = defs.into_iter().map(|def| evaluator.evaluate_definition(def)).collect();
    drop(evaluator);

    let program = Program::from_parts(defs, symbols, term_ids, global_context);
    program.assert_flat_apps();
    program
}

// =============================================================================
// Values
// =============================================================================

/// A compile-time value.
#[derive(Debug, Clone)]
enum Value {
    /// Known scalar: int, float, bool
    Int(i64),
    Float(f64),
    Bool(bool),
    /// A fixed-size vector whose components are all known scalars.
    Vector(Vec<Scalar>),

    /// Partial application: function waiting for more args. Each
    /// accumulated arg carries its source type so the reifier can
    /// rebuild the App term without re-deriving types from the def.
    Partial {
        sym: SymbolId,
        args: Vec<(Value, Type<TypeName>)>,
    },

    /// Unknown at compile time - residual code
    Unknown(Term<Empty, Empty>),
}

fn parse_integer_value(spelling: &str, ty: &Type<TypeName>) -> Result<i64, String> {
    if matches!(ty, Type::Constructed(TypeName::UInt(_), _)) {
        return spelling
            .parse::<u64>()
            .map(|value| scalar_eval::wrap_int(value as i128, ty))
            .or_else(|unsigned_error| {
                spelling.parse::<i64>().map_err(|signed_error| {
                    format!(
                        "unsigned parse failed: {unsigned_error}; signed residual parse failed: {signed_error}"
                    )
                })
            });
    }
    spelling.parse::<i64>().map_err(|error| error.to_string())
}

impl Value {
    fn as_constant(&self) -> Option<Constant> {
        match self {
            Self::Vector(values) => Some(Constant::Vector(values.clone())),
            _ => self.as_scalar().map(Constant::from_scalar),
        }
    }
    fn from_constant(value: Constant) -> Self {
        match value {
            Constant::Vector(values) => Self::Vector(values),
            Constant::Int(value) => Self::Int(value),
            Constant::Float(value) => Self::Float(value),
            Constant::Bool(value) => Self::Bool(value),
        }
    }
    fn is_constant(&self) -> bool {
        matches!(
            self,
            Self::Int(_) | Self::Float(_) | Self::Bool(_) | Self::Vector(_)
        )
    }

    fn is_known(&self) -> bool {
        !matches!(self, Value::Unknown(_))
    }

    fn as_scalar(&self) -> Option<Scalar> {
        match self {
            Value::Int(value) => Some(Scalar::Int(*value)),
            Value::Float(value) => Some(Scalar::Float(*value)),
            Value::Bool(value) => Some(Scalar::Bool(*value)),
            _ => None,
        }
    }

    fn from_scalar(value: Scalar) -> Self {
        match value {
            Scalar::Int(value) => Value::Int(value),
            Scalar::Float(value) => Value::Float(value),
            Scalar::Bool(value) => Value::Bool(value),
        }
    }
}

// =============================================================================
// Evaluator
// =============================================================================

#[derive(Clone)]
struct DefinitionTemplate {
    arity: usize,
    body: Term<Empty, Empty>,
    constant_candidate: bool,
}

struct PartialEvaluator<'a> {
    /// Function bodies retained for compile-time call expansion.
    definitions: LookupMap<SymbolId, DefinitionTemplate>,
    /// Term ID source for generating new terms
    term_ids: &'a mut TermIdSource,
    /// Environment: symbol -> Value
    env: LookupMap<SymbolId, Value>,
    /// Definitions currently being evaluated. Re-entry means recursion, which
    /// is deliberately left residual instead of recursing in the compiler.
    active_defs: LookupSet<SymbolId>,
    /// Zero-arity globals proven to reduce to scalars or small constant vectors.
    global_constants: LookupMap<SymbolId, Value>,
    /// Globals already considered for `global_constants`, including failures.
    resolved_global_constants: LookupSet<SymbolId>,
}

impl<'a> PartialEvaluator<'a> {
    fn new(definitions: LookupMap<SymbolId, DefinitionTemplate>, term_ids: &'a mut TermIdSource) -> Self {
        Self {
            definitions,
            term_ids,
            env: LookupMap::new(),
            active_defs: LookupSet::new(),
            global_constants: LookupMap::new(),
            resolved_global_constants: LookupSet::new(),
        }
    }

    /// Discover the constant environment once, before residualizing any
    /// definition bodies. Recursive resolution handles constants defined in
    /// terms of other constants; `active_defs` leaves cycles residual.
    fn discover_global_constants(&mut self) {
        let candidates = self
            .definitions
            .iter()
            .filter_map(|(symbol, definition)| definition.constant_candidate.then_some(*symbol))
            .collect::<Vec<_>>();
        for symbol in candidates {
            self.resolve_global_constant(symbol);
        }
    }

    fn evaluate_definition(&mut self, def: Def<UnpinnedPolymorphic>) -> Def<UnpinnedPolymorphic> {
        let body_val =
            self.global_constants.get(&def.name).cloned().unwrap_or_else(|| self.eval(&def.body));
        let body = self.reify(body_val, &def.body.ty, def.body.span);
        let body = body.rewrite(&mut ResidualConstantFolder { evaluator: self });
        Def { body, ..def }
    }

    /// Evaluate a term to a Value.
    fn eval(&mut self, term: &Term<Empty, Empty>) -> Value {
        match &term.kind {
            // Literals → known values
            TermKind::IntLit(s) => {
                Value::Int(parse_integer_value(s, &term.ty).unwrap_or_else(|error| {
                    panic!("BUG: invalid integer literal from lexer: {s}: {error}")
                }))
            }
            TermKind::FloatLit(f) => Value::Float(*f as f64),
            TermKind::BoolLit(b) => Value::Bool(*b),
            TermKind::UnitLit => Value::Unknown(term.clone()),
            TermKind::Closure(_) => {
                unreachable!("closure terms do not exist before defunctionalization")
            }
            TermKind::Coerce { inner, target_ty } => {
                let inner_val = self.eval(inner);
                let inner_term = self.reify(inner_val, &inner.ty, inner.span);
                Value::Unknown(self.mk_term(
                    term.ty.clone(),
                    term.span,
                    TermKind::Coerce {
                        inner: Box::new(inner_term),
                        target_ty: target_ty.clone(),
                    },
                ))
            }

            // Variable lookup
            TermKind::Var(VarRef::Symbol(sym)) => {
                let sym = *sym;
                if let Some(val) = self.env.get(&sym) {
                    val.clone()
                } else if let Some(def) = self.definitions.get(&sym).cloned() {
                    if def.arity == 0 {
                        if let Some(value) = self.resolve_global_constant(sym) {
                            value
                        } else if !self.active_defs.insert(sym) {
                            Value::Unknown(term.clone())
                        } else {
                            let value = self.eval(&def.body);
                            self.active_defs.remove(&sym);
                            value
                        }
                    } else {
                        // Function - create partial application with 0 args applied
                        Value::Partial { sym, args: vec![] }
                    }
                } else {
                    // Unknown variable (intrinsic or undefined)
                    Value::Unknown(term.clone())
                }
            }

            // Builtin reference: not constant-foldable on its own.
            TermKind::Var(VarRef::Builtin { .. }) => Value::Unknown(term.clone()),

            // Let binding. Inline only *duplicable* rhs values (literals, bare
            // vars, function values); for a non-trivial residual rhs, keep the
            // `let` in the residual and reference it by name. Inlining a
            // residual at every use site duplicates it — exponential for chains
            // like `let (x0, x1) = mix(..)` repeated, since each step uses the
            // previous twice — and dissolving the sole binding is what leaves a
            // var dangling ("Unknown global: <v>") when a residual fails to
            // substitute it. Keeping the binding fixes both.
            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                let rhs_val = self.eval(rhs);
                if is_duplicable(&rhs_val) {
                    self.env.insert(*name, rhs_val);
                    self.eval(body)
                } else {
                    let rhs_term = self.reify(rhs_val, &rhs.ty, rhs.span);
                    let name_var = self.mk_term(
                        rhs_term.ty.clone(),
                        rhs.span,
                        TermKind::Var(VarRef::Symbol(*name)),
                    );
                    self.env.insert(*name, Value::Unknown(name_var));
                    let body_val = self.eval(body);
                    let body_term = self.reify(body_val, &body.ty, body.span);
                    Value::Unknown(self.mk_term(
                        term.ty.clone(),
                        term.span,
                        TermKind::Let {
                            name: *name,
                            name_ty: name_ty.clone(),
                            rhs: Box::new(rhs_term),
                            body: Box::new(body_term),
                        },
                    ))
                }
            }

            // If expression
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond_val = self.eval(cond);
                match cond_val {
                    Value::Bool(true) => self.eval(then_branch),
                    Value::Bool(false) => self.eval(else_branch),
                    _ => {
                        // Unknown condition - residualize
                        let then_val = self.eval(then_branch);
                        let else_val = self.eval(else_branch);
                        self.reify_if(cond_val, then_val, else_val, term)
                    }
                }
            }

            // Application - evaluate args (pairing each with its source
            // term type) and apply. Carrying types alongside values lets
            // downstream reification rebuild the App without re-deriving
            // types from the def.
            TermKind::App { func, args } => {
                let arg_vals: Vec<(Value, Type<TypeName>)> =
                    args.iter().map(|a| (self.eval(a), a.ty.clone())).collect();
                self.apply(func, arg_vals, term)
            }

            // Residual lambda. Its body may reference env-bound captures —
            // the arguments of a function being inlined around it — so
            // substitute those in, leaving the lambda closed over reified
            // values rather than vars that depend on a binder being in
            // scope. Binder-aware: the lambda's own params shadow env. Then
            // fold constant sub-expressions in the result.
            TermKind::Lambda(lam) => {
                let mut bound: LookupSet<SymbolId> = lam.params.iter().map(|(p, _)| *p).collect();
                let body = self.substitute_residual_vars((*lam.body).clone(), &mut bound);
                Value::Unknown(self.mk_term(
                    term.ty.clone(),
                    term.span,
                    TermKind::Lambda(Lambda {
                        params: lam.params.clone(),
                        body: Box::new(body),
                        ret_ty: lam.ret_ty.clone(),
                    }),
                ))
            }

            // Loops are not evaluated at compile time, but an enclosing
            // dissolved binding may still occur in the initializer, domain,
            // condition, or body. Residualize through the binder-aware
            // substitution path so those values do not become free.
            TermKind::Loop { .. } => self.residualize_unreduced(term),

            // Operators as values - residualize
            TermKind::BinOp(_) | TermKind::UnOp(_) => Value::Unknown(term.clone()),

            // Extern declarations - residualize (linked at SPIR-V level)
            TermKind::Extern(_) => Value::Unknown(term.clone()),

            // SOAC nodes are opaque to partial evaluation, but their sub-
            // terms may reference env-bound `Var`s (e.g. `let m = lit in
            // map(f, m)` — the SOAC consumes `m`). Substitute those refs
            // through the SOAC's children so the dissolved `Let` doesn't
            // leave dangling free vars downstream.
            TermKind::Soac(_) | TermKind::ArrayExpr(_) => self.residualize_unreduced(term),

            // Structural ops: evaluate children so let-bound `Var`s
            // get substituted through, then rebuild the variant. Without
            // this an enclosing `Let` whose body references one of these
            // would leave a dangling `Var(name)` after the let is
            // dissolved by the eval pass.
            TermKind::Tuple(parts) => {
                let part_vals: Vec<Value> = parts.iter().map(|p| self.eval(p)).collect();
                let part_terms: Vec<Term<Empty, Empty>> =
                    parts.iter().zip(part_vals).map(|(p, v)| self.reify(v, &p.ty, p.span)).collect();
                Value::Unknown(self.mk_term(term.ty.clone(), term.span, TermKind::Tuple(part_terms)))
            }
            TermKind::TupleProj { tuple, idx } => {
                let tuple_val = self.eval(tuple);
                if let Value::Vector(values) = &tuple_val {
                    if let Some(value) = values.get(*idx) {
                        return Value::from_scalar(*value);
                    }
                }
                let tuple_term = self.reify(tuple_val, &tuple.ty, tuple.span);
                Value::Unknown(self.mk_term(
                    term.ty.clone(),
                    term.span,
                    TermKind::TupleProj {
                        tuple: Box::new(tuple_term),
                        idx: *idx,
                    },
                ))
            }
            TermKind::Index { array, index } => {
                let array_val = self.eval(array);
                let index_val = self.eval(index);
                let array_term = self.reify(array_val, &array.ty, array.span);
                let index_term = self.reify(index_val, &index.ty, index.span);
                Value::Unknown(self.mk_term(
                    term.ty.clone(),
                    term.span,
                    TermKind::Index {
                        array: Box::new(array_term),
                        index: Box::new(index_term),
                    },
                ))
            }
            TermKind::VecLit(parts) => {
                let part_vals: Vec<Value> = parts.iter().map(|p| self.eval(p)).collect();
                if term.ty.vec_size() == Some(parts.len()) && (2..=4).contains(&parts.len()) {
                    if let Some(values) = part_vals.iter().map(Value::as_scalar).collect() {
                        return Value::Vector(values);
                    }
                }
                let part_terms: Vec<Term<Empty, Empty>> =
                    parts.iter().zip(part_vals).map(|(p, v)| self.reify(v, &p.ty, p.span)).collect();
                Value::Unknown(self.mk_term(term.ty.clone(), term.span, TermKind::VecLit(part_terms)))
            }
        }
    }

    /// Apply a function to arguments based on the base term kind.
    fn apply(
        &mut self,
        base: &Term<Empty, Empty>,
        args: Vec<(Value, Type<TypeName>)>,
        original: &Term<Empty, Empty>,
    ) -> Value {
        match &base.kind {
            // `eval_binop`/`eval_unop` return `Some` only for a genuine fold
            // or simplification (including identities like `x + 0 → x`).
            // `None` means "couldn't fold" — and we must rebuild the residual
            // from the *evaluated* operands via `residualize_call`, never from
            // `original`: an operand may be a let-bound `Var` that an enclosing
            // dissolved `Let` substituted, which `original` still names by hand
            // (the source of "Unknown global: <local>" at codegen).
            TermKind::BinOp(op) => {
                let folded = if args.len() >= 2 {
                    self.eval_binop(op, &args[0].0, &args[1].0, &args[0].1)
                } else {
                    None
                };
                folded.unwrap_or_else(|| self.residualize_unreduced(original))
            }

            TermKind::UnOp(op) => {
                let folded =
                    if !args.is_empty() { self.eval_unop(op, &args[0].0, &args[0].1) } else { None };
                folded.unwrap_or_else(|| self.residualize_unreduced(original))
            }

            TermKind::Var(VarRef::Symbol(sym)) => self.apply_var(*sym, args, original),

            TermKind::Var(VarRef::Builtin { id, overload_idx }) => self
                .eval_builtin(*id, *overload_idx, &args, &original.ty)
                .unwrap_or_else(|| self.residualize_call(base.clone(), args, original)),

            _ => {
                // Higher-order or computed function - can't evaluate.
                // Residualize through the eval'd args, not the original
                // term, so substitutions in args don't get clobbered.
                self.residualize_call(base.clone(), args, original)
            }
        }
    }

    /// Residualize a term the evaluator can't reduce, patching any free
    /// variable an enclosing *dissolved* `Let` bound into the env. This is a
    /// binder-aware substitution (`substitute_residual_vars`): it preserves
    /// the original syntax, node ids, and any nested binders, replacing only
    /// dangling free vars. Use this for every unreduced residual (binops,
    /// unops, SOACs, ...) — never `original.clone()`, which would keep naming
    /// a dissolved let's variable and surface as "Unknown global: <name>" at
    /// codegen; and not a `reify`-based rebuild, which reconstructs operands
    /// (e.g. partial applications) and can corrupt closure-call arities.
    fn residualize_unreduced(&mut self, term: &Term<Empty, Empty>) -> Value {
        let mut bound = LookupSet::new();
        Value::Unknown(self.substitute_residual_vars(term.clone(), &mut bound))
    }

    /// Rebuild an App from the (already-evaluated) `func` and `args`,
    /// reifying each arg back to a Term. Used when partial_eval can't
    /// reduce the call further but the args may have been substituted
    /// (e.g. a let-bound `Var` resolved to its rhs term). Cloning the
    /// original term in this position would discard those substitutions.
    fn residualize_call(
        &mut self,
        func: Term<Empty, Empty>,
        args: Vec<(Value, Type<TypeName>)>,
        original: &Term<Empty, Empty>,
    ) -> Value {
        let arg_terms: Vec<Term<Empty, Empty>> =
            args.into_iter().map(|(arg, ty)| self.reify(arg, &ty, original.span)).collect();
        let result = self.mk_term(
            original.ty.clone(),
            original.span,
            TermKind::App {
                func: Box::new(func),
                args: arg_terms,
            },
        );
        Value::Unknown(result)
    }

    /// Apply a named function to arguments.
    fn apply_var(
        &mut self,
        sym: SymbolId,
        args: Vec<(Value, Type<TypeName>)>,
        original: &Term<Empty, Empty>,
    ) -> Value {
        // Check if this is a let-bound variable aliasing a function.
        // This handles cases like `let f = g in f x` where g is a known function.
        if let Some(Value::Partial {
            sym: real_sym,
            args: partial_args,
            ..
        }) = self.env.get(&sym).cloned()
        {
            // Combine the partial application's args with the new args
            let mut combined_args = partial_args;
            combined_args.extend(args);
            // Apply to the real function (recursive to handle chains like let h = f in let g = h in g x)
            return self.apply_var(real_sym, combined_args, original);
        }

        // Check if this is a let-bound variable aliasing a function name
        // (intrinsic, builtin, or top-level def). Handles `let f = f32.sin in f x`.
        if let Some(Value::Unknown(Term {
            kind: TermKind::Var(VarRef::Symbol(real_sym)),
            ..
        })) = self.env.get(&sym)
        {
            let real_sym = *real_sym;
            return self.apply_var(real_sym, args, original);
        }

        // A let-bound variable whose residual is a non-Var term — most
        // importantly a *lambda* (`let g = |x| ... in g a`). Apply the args to
        // that residual directly. Otherwise, since the `let` is dissolved and
        // `g` is not a top-level def, we'd fall through to `reify_call(sym)`
        // and leave `Var(g)` dangling in `g a` — surfacing as "Unknown
        // function" at codegen, or a mis-threaded closure call.
        if let Some(Value::Unknown(t)) = self.env.get(&sym).cloned() {
            if !matches!(t.kind, TermKind::Var(_)) {
                return self.apply(&t, args, original);
            }
        }

        // Check for known function
        if let Some(def) = self.definitions.get(&sym).cloned() {
            let args_len = args.len();
            // Vector constants are usable by builtins, but expanding arbitrary
            // functions with them also requires instantiating polymorphic body
            // types. Leave those calls for the existing monomorphization pass.
            let all_known = args.iter().all(|(v, _)| v.is_known() && !matches!(v, Value::Vector(_)));
            if args_len >= def.arity && def.arity > 0 && all_known {
                if !self.active_defs.insert(sym) {
                    self.reify_call(sym, args, original)
                } else {
                    let value = self.inline(&def, args);
                    self.active_defs.remove(&sym);
                    value
                }
            } else if args_len < def.arity {
                // Partial application
                Value::Partial { sym, args }
            } else {
                // Some unknown args or zero-arity - residualize
                self.reify_call(sym, args, original)
            }
        } else {
            // Unknown function - residualize
            self.reify_call(sym, args, original)
        }
    }

    /// Inline a function call.
    fn inline(&mut self, def: &DefinitionTemplate, args: Vec<(Value, Type<TypeName>)>) -> Value {
        let mut body = &def.body;
        let mut args_iter = args.into_iter();

        loop {
            if let TermKind::Lambda(Lambda {
                params, body: inner, ..
            }) = &body.kind
            {
                // Bind as many args as this lambda has params
                let mut consumed = 0;
                for (param, _) in params {
                    if let Some((arg, _arg_ty)) = args_iter.next() {
                        self.env.insert(*param, arg);
                        consumed += 1;
                    } else {
                        break;
                    }
                }
                body = inner;
                if consumed == 0 || args_iter.len() == 0 {
                    break;
                }
            } else {
                break;
            }
        }

        self.eval(body)
    }

    /// Evaluate a binary operation. `Some` means a genuine fold or
    /// simplification was performed (a literal result, or an identity like
    /// `x + 0 → x` that returns a residual operand); `None` means it could
    /// not be reduced and the caller must rebuild residual syntax from the
    /// evaluated operands. Crucially `None` is *not* the same as "returned a
    /// residual" — conflating them makes the caller either drop a valid
    /// simplification or leave a dissolved let's variable dangling.
    ///
    /// `ty` is the operand type; integer folds wrap to its bit width so the
    /// result matches runtime semantics (e.g. u32 multiply is mod 2^32). The
    /// fold is done in `i128` to avoid overflowing before the wrap.
    fn eval_binop(&self, op: &BinaryOp, lhs: &Value, rhs: &Value, ty: &Type<TypeName>) -> Option<Value> {
        if let (Some(lhs), Some(rhs)) = (lhs.as_constant(), rhs.as_constant()) {
            if let Some(result) = constant_eval::binary(op.op, &lhs, &rhs, ty) {
                return Some(Value::from_constant(result));
            }
        }

        Some(match (op.op, lhs, rhs) {
            (BinaryOperator::Add, Value::Int(0), _) => rhs.clone(),
            (BinaryOperator::Add, _, Value::Int(0)) => lhs.clone(),
            (BinaryOperator::Multiply, Value::Int(1), _) => rhs.clone(),
            (BinaryOperator::Multiply, _, Value::Int(1)) => lhs.clone(),
            (BinaryOperator::Multiply, Value::Int(0), _) | (BinaryOperator::Multiply, _, Value::Int(0)) => {
                Value::Int(0)
            }
            _ => return None,
        })
    }

    /// Evaluate a unary operation. `Some`/`None` as in `eval_binop`.
    fn eval_unop(&self, op: &UnaryOp, arg: &Value, ty: &Type<TypeName>) -> Option<Value> {
        scalar_eval::unary(op.op, arg.as_scalar()?, ty).map(Value::from_scalar)
    }

    /// Evaluate supported pure scalar builtins and their componentwise vector
    /// forms. Unsupported operations and domains remain residual calls.
    fn eval_builtin(
        &self,
        id: builtins::BuiltinId,
        overload_idx: usize,
        args: &[(Value, Type<TypeName>)],
        result_ty: &Type<TypeName>,
    ) -> Option<Value> {
        let def = by_id(id);
        if def.raw.purity != Purity::Pure || args.iter().any(|(v, _)| !v.is_known()) {
            return None;
        }
        let constants =
            args.iter().map(|(v, ty)| Some((v.as_constant()?, ty.clone()))).collect::<Option<Vec<_>>>()?;
        let lowering = &def.overloads().get(overload_idx)?.lowering;
        // Partial evaluation precedes intrinsic specialization. Resolve generic
        // numeric operations here as well, without losing broadcast metadata.
        let scalar_ty = if result_ty.is_vec() { result_ty.elem_type()? } else { result_ty };
        if let Type::Constructed(scalar, _) = scalar_ty {
            if let Some(specialized) =
                builtins::catalog().specialized_numeric_lowering(id, overload_idx, scalar)
            {
                return constant_eval::builtin(&specialized, &constants, result_ty)
                    .map(Value::from_constant);
            }
        }
        constant_eval::builtin(lowering, &constants, result_ty).map(Value::from_constant)
    }

    fn literal_value(&self, term: &Term<Empty, Empty>) -> Option<Value> {
        match &term.kind {
            TermKind::IntLit(s) => Some(Value::Int(parse_integer_value(s, &term.ty).unwrap_or_else(
                |error| {
                    panic!(
                        "lexer-produced IntLit `{s}` failed to parse for {:?}: {error}",
                        term.ty
                    )
                },
            ))),
            TermKind::FloatLit(f) => Some(Value::Float(*f as f64)),
            TermKind::BoolLit(b) => Some(Value::Bool(*b)),
            TermKind::VecLit(parts)
                if term.ty.vec_size() == Some(parts.len()) && (2..=4).contains(&parts.len()) =>
            {
                parts
                    .iter()
                    .map(|part| self.literal_value(part)?.as_scalar())
                    .collect::<Option<Vec<_>>>()
                    .map(Value::Vector)
            }
            _ => None,
        }
    }

    // =========================================================================
    // Reification (Value → Term)
    // =========================================================================

    fn reify(&mut self, value: Value, ty: &Type<TypeName>, span: Span) -> Term<Empty, Empty> {
        match value {
            Value::Int(n) => self.mk_term(ty.clone(), span, TermKind::IntLit(n.to_string())),
            Value::Float(f) => self.mk_term(ty.clone(), span, TermKind::FloatLit(f as f32)),
            Value::Bool(b) => self.mk_term(ty.clone(), span, TermKind::BoolLit(b)),
            Value::Vector(values) => {
                let elem_ty = ty.elem_type().expect("constant vector must have a vector type");
                let parts = values
                    .into_iter()
                    .map(|value| self.reify(Value::from_scalar(value), elem_ty, span))
                    .collect();
                self.mk_term(ty.clone(), span, TermKind::VecLit(parts))
            }
            Value::Unknown(t) => t,
            Value::Partial { sym, args, .. } => self.reify_partial(sym, args, ty, span),
        }
    }

    /// Binder-aware walk that replaces every free `Var(VarRef::Symbol(sym))`
    /// with the reified `env[sym]` value, if any. Used when residualizing
    /// nodes like `Soac` and `ArrayExpr` that the evaluator would otherwise
    /// clone wholesale: without it, an enclosing `let m = lit in soac(...,
    /// m)` dissolves the let and leaves a dangling `Var(m)` inside the SOAC.
    ///
    /// `bound` tracks symbols currently in scope (shadowing env). It's
    /// mutated in place as we descend into binders and restored on exit.
    fn substitute_residual_vars(
        &mut self,
        term: Term<Empty, Empty>,
        bound: &mut LookupSet<SymbolId>,
    ) -> Term<Empty, Empty> {
        let mut term = term;
        self.substitute_residual_vars_tracked(&mut term, bound);
        term
    }

    /// Mutate an exclusively owned residual tree in place and report whether
    /// this subtree changed. Boxes and vectors survive unchanged; only a
    /// replaced variable and its ancestor path receive fresh IDs.
    fn substitute_residual_vars_tracked(
        &mut self,
        term: &mut Term<Empty, Empty>,
        bound: &mut LookupSet<SymbolId>,
    ) -> bool {
        let replacement = match &term.kind {
            TermKind::Var(VarRef::Symbol(name)) if !bound.contains(name) => {
                self.env.get(name).cloned().or_else(|| self.global_constants.get(name).cloned())
            }
            _ => None,
        };
        if let Some(value) = replacement {
            *term = self.reify(value, &term.ty.clone(), term.span);
            return true;
        }

        let changed = match &mut term.kind {
            TermKind::Let { name, rhs, body, .. } => {
                let rhs_changed = self.substitute_residual_vars_tracked(rhs, bound);
                let added = bound.insert(*name);
                let body_changed = self.substitute_residual_vars_tracked(body, bound);
                if added {
                    bound.remove(name);
                }
                rhs_changed || body_changed
            }
            TermKind::Lambda(lam) => {
                let added = add_bound_symbols(bound, lam.params.iter().map(|(name, _)| *name));
                let changed = self.substitute_residual_vars_tracked(&mut lam.body, bound);
                remove_bound_symbols(bound, added);
                changed
            }
            TermKind::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body,
                ..
            } => {
                let mut changed = self.substitute_residual_vars_tracked(init, bound);

                let loop_var_added = bound.insert(*loop_var);
                for (_, _, extraction) in init_bindings.iter_mut() {
                    changed |= self.substitute_residual_vars_tracked(extraction, bound);
                }
                if loop_var_added {
                    bound.remove(loop_var);
                }

                let iteration_var = match kind {
                    LoopKind::For { var, iter, .. } => {
                        changed |= self.substitute_residual_vars_tracked(iter, bound);
                        Some(*var)
                    }
                    LoopKind::ForRange {
                        var, bound: limit, ..
                    } => {
                        changed |= self.substitute_residual_vars_tracked(limit, bound);
                        Some(*var)
                    }
                    LoopKind::While { cond } => {
                        changed |= self.substitute_residual_vars_tracked(cond, bound);
                        None
                    }
                };

                let added = add_bound_symbols(
                    bound,
                    std::iter::once(*loop_var)
                        .chain(init_bindings.iter().map(|(name, _, _)| *name))
                        .chain(iteration_var),
                );
                changed |= self.substitute_residual_vars_tracked(body, bound);
                remove_bound_symbols(bound, added);
                changed
            }
            _ => {
                let mut changed = false;
                term.for_each_child_mut(&mut |child| {
                    changed |= self.substitute_residual_vars_tracked(child, bound);
                });
                changed
            }
        };

        if changed {
            term.id = self.term_ids.next_id();
        }
        changed
    }

    /// Memoize a zero-arity global only when it reduces to a scalar or a small
    /// constant vector. This is intentionally narrower than general inlining:
    /// the environment can be copied into residual lambdas without duplicating
    /// runtime work or unbounded aggregate construction.
    fn resolve_global_constant(&mut self, symbol: SymbolId) -> Option<Value> {
        if let Some(value) = self.global_constants.get(&symbol) {
            return Some(value.clone());
        }
        if self.resolved_global_constants.contains(&symbol) {
            return None;
        }
        let definition = self.definitions.get(&symbol)?.clone();
        if !definition.constant_candidate || !self.active_defs.insert(symbol) {
            return None;
        }
        let value = self.eval(&definition.body);
        self.active_defs.remove(&symbol);
        self.resolved_global_constants.insert(symbol);
        if !value.is_constant() {
            return None;
        }
        self.global_constants.insert(symbol, value.clone());
        Some(value)
    }

    fn reify_partial(
        &mut self,
        sym: SymbolId,
        args: Vec<(Value, Type<TypeName>)>,
        ty: &Type<TypeName>,
        span: Span,
    ) -> Term<Empty, Empty> {
        let func_term = self.mk_term(ty.clone(), span, TermKind::Var(VarRef::Symbol(sym)));
        let arg_terms: Vec<Term<Empty, Empty>> =
            args.into_iter().map(|(arg, arg_ty)| self.reify(arg, &arg_ty, span)).collect();
        self.mk_term(
            ty.clone(),
            span,
            TermKind::App {
                func: Box::new(func_term),
                args: arg_terms,
            },
        )
    }

    fn reify_call(
        &mut self,
        sym: SymbolId,
        args: Vec<(Value, Type<TypeName>)>,
        original: &Term<Empty, Empty>,
    ) -> Value {
        let func_term = self.mk_term(
            original.ty.clone(),
            original.span,
            TermKind::Var(VarRef::Symbol(sym)),
        );
        let arg_terms: Vec<Term<Empty, Empty>> =
            args.into_iter().map(|(arg, arg_ty)| self.reify(arg, &arg_ty, original.span)).collect();
        let result = self.mk_term(
            original.ty.clone(),
            original.span,
            TermKind::App {
                func: Box::new(func_term),
                args: arg_terms,
            },
        );
        Value::Unknown(result)
    }

    fn reify_if(
        &mut self,
        cond: Value,
        then_val: Value,
        else_val: Value,
        original: &Term<Empty, Empty>,
    ) -> Value {
        let cond_term = self.reify(cond, &Type::Constructed(TypeName::Bool, vec![]), original.span);
        let then_term = self.reify(then_val, &original.ty, original.span);
        let else_term = self.reify(else_val, &original.ty, original.span);

        Value::Unknown(self.mk_term(
            original.ty.clone(),
            original.span,
            TermKind::If {
                cond: Box::new(cond_term),
                then_branch: Box::new(then_term),
                else_branch: Box::new(else_term),
            },
        ))
    }

    fn mk_term(
        &mut self,
        ty: Type<TypeName>,
        span: Span,
        kind: TermKind<Empty, Empty>,
    ) -> Term<Empty, Empty> {
        Term::fresh(self.term_ids, ty, span, kind)
    }
}

fn add_bound_symbols(
    bound: &mut LookupSet<SymbolId>,
    symbols: impl IntoIterator<Item = SymbolId>,
) -> Vec<SymbolId> {
    symbols.into_iter().filter(|symbol| bound.insert(*symbol)).collect()
}

fn remove_bound_symbols(bound: &mut LookupSet<SymbolId>, symbols: Vec<SymbolId>) {
    for symbol in symbols {
        bound.remove(&symbol);
    }
}

/// Constant folding is a node-local residual-tree rewrite. The generic
/// rewriter supplies the bottom-up traversal, preserves unchanged storage,
/// and refreshes IDs only along changed paths.
struct ResidualConstantFolder<'e, 'ids> {
    evaluator: &'e mut PartialEvaluator<'ids>,
}

impl TermRewriter<Empty, Empty> for ResidualConstantFolder<'_, '_> {
    fn next_term_id(&mut self) -> TermId {
        self.evaluator.term_ids.next_id()
    }

    fn rewrite_node(&mut self, term: &mut Term<Empty, Empty>) -> RewriteDecision {
        if let TermKind::TupleProj { tuple, idx } = &term.kind {
            if let Some(Value::Vector(values)) = self.evaluator.literal_value(tuple) {
                if let Some(value) = values.get(*idx) {
                    term.kind = self.evaluator.reify(Value::from_scalar(*value), &term.ty, term.span).kind;
                    return RewriteDecision::Changed;
                }
            }
        }
        // Retained lambdas are not interpreted, but their literal let bindings
        // must still expose constants to builtin calls (including vector calls).
        if let TermKind::Let { name, rhs, body, .. } = &mut term.kind {
            if let Some(value) = self.evaluator.literal_value(rhs) {
                let saved_env = std::mem::take(&mut self.evaluator.env);
                self.evaluator.env.insert(*name, value);
                self.evaluator.substitute_residual_vars_tracked(body, &mut LookupSet::new());
                self.evaluator.env = saved_env;
                self.rewrite_tracked(body);
                term.kind = std::mem::replace(&mut body.kind, TermKind::UnitLit);
                return RewriteDecision::Changed;
            }
        }
        let ty = term.ty.clone();
        let folded = {
            let TermKind::App { func, args } = &term.kind else {
                return RewriteDecision::Unchanged;
            };
            let Some(args) = args
                .iter()
                .map(|arg| self.evaluator.literal_value(arg).map(|value| (value, arg.ty.clone())))
                .collect::<Option<Vec<_>>>()
            else {
                return RewriteDecision::Unchanged;
            };

            match &func.kind {
                TermKind::BinOp(op) if args.len() >= 2 => {
                    self.evaluator.eval_binop(op, &args[0].0, &args[1].0, &args[0].1)
                }
                TermKind::UnOp(op) if !args.is_empty() => {
                    self.evaluator.eval_unop(op, &args[0].0, &args[0].1)
                }
                TermKind::Var(VarRef::Builtin { id, overload_idx }) => {
                    self.evaluator.eval_builtin(*id, *overload_idx, &args, &ty)
                }
                _ => None,
            }
        };

        let Some(value) = folded else {
            return RewriteDecision::Unchanged;
        };
        term.kind = self.evaluator.reify(value, &ty, term.span).kind;
        RewriteDecision::Changed
    }
}

/// Whether a partial-eval value is cheap to duplicate at every use site.
/// Literals, bare-variable residuals, unit, and function values are; a
/// non-trivial residual term (a computation, or an aggregate containing one)
/// is not — duplicating those at each use is what makes `let`-chains blow up
/// exponentially, so they are kept as `let`-bindings instead of inlined.
fn is_duplicable(v: &Value) -> bool {
    match v {
        Value::Int(_) | Value::Float(_) | Value::Bool(_) | Value::Vector(_) | Value::Partial { .. } => true,
        // Lambdas stay inlined: they aren't the source of the duplication
        // blowup (that's self-referential value chains), and `apply_var` must
        // see the lambda value in the env to apply it — binding the name to
        // `Var(name)` instead would make `apply_var` self-alias and recurse.
        Value::Unknown(t) => matches!(t.kind, TermKind::Var(_) | TermKind::UnitLit | TermKind::Lambda(_)),
    }
}

#[cfg(test)]
#[path = "partial_eval_tests.rs"]
mod partial_eval_tests;
