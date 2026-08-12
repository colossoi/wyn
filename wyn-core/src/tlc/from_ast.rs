//! AST-to-TLC node transformation.
//!
//! The top-level driver lives in `run`; this module owns recursive AST
//! lowering and the pattern-binding helpers that extend `Transformer`.

use super::{
    count_function_arity, data, peel_lets, run, ArrayExpr, Def, DefMeta, EntryPoint, Lambda, LoopKind,
    Place, ProgramParts, SoacBody, SoacOp, Term, TermIdSource, TermKind, VarRef,
};
use crate::ast::{self, Span, TypeName};
use crate::builtins::{catalog, BuiltinId};
use crate::op::BinaryOperator;
use crate::types::{SoacOwnership, TypeExt};
use crate::{interface, LookupMap, SymbolId, SymbolTable};
use polytype::Type;

// =============================================================================
// AST to TLC Transformation
// =============================================================================

/// A pending let-binding to be applied after all lambdas are created.
#[derive(Debug, Clone)]
pub(crate) struct PendingBinding {
    pub(super) name: SymbolId,
    pub(super) ty: Type<TypeName>,
    pub(super) expr: Term,
}

/// Flattened-no-sharing layout for a structural sum type. Computed
/// once per sum and then consulted by the Constructor and Match
/// transforms for tag values and per-payload slot offsets.
pub(super) struct SumLayout {
    /// All slot types of the lowered tuple, lowered. Index 0 is the
    /// u32 tag; indices 1.. are the variant payloads concatenated
    /// in source order.
    pub(super) slot_types: Vec<Type<TypeName>>,
    /// For each constructor name: its tag value (source-order index)
    /// and the starting slot index of its payload in `slot_types`.
    pub(super) constructor_info: LookupMap<String, (u32, usize)>,
}

/// Context for transforming AST to TLC.
pub(crate) struct Transformer<'a> {
    pub(super) term_ids: &'a mut TermIdSource,
    /// Source-level identities arrive from name resolution. This arena is used
    /// only to allocate compiler-synthesized TLC temporaries.
    symbols: &'a mut SymbolTable,
    /// Shared placeholder symbol for pattern-matching scrutinees.
    placeholder_sym: SymbolId,
}

impl<'a> Transformer<'a> {
    pub fn new(symbols: &'a mut SymbolTable, term_ids: &'a mut TermIdSource) -> Self {
        let placeholder_sym = symbols.alloc("_w_placeholder".to_string());
        Self {
            term_ids,
            symbols,
            placeholder_sym,
        }
    }

    /// Allocate an identity for a compiler-synthesized TLC value. Source
    /// binders must use the `ResolvedBinding::symbol` already stored in-tree.
    pub(super) fn fresh(&mut self, diagnostic_name: impl Into<String>) -> SymbolId {
        self.symbols.alloc(diagnostic_name.into())
    }

    /// Transform an AST program to TLC.
    /// Returns program parts without the symbol table - caller must combine with
    /// their owned symbol table using `ProgramParts::with_symbols`.
    pub fn transform_program(
        &mut self,
        program: &crate::ast_type_holes::HolesResolved,
    ) -> ProgramParts<run::UnpinnedPolymorphic> {
        let mut defs = Vec::new();

        for decl in &program.declarations {
            match decl {
                ast::Declaration::Decl(d) => {
                    if let Some(def) = self.transform_decl(d) {
                        defs.push(def);
                    }
                }
                ast::Declaration::Entry(e) => {
                    if let Some(def) = self.transform_entry(e) {
                        defs.push(def);
                    }
                }
                ast::Declaration::Extern(e) => {
                    let syntax = &e.data.source.syntax;
                    let ty = Self::lower_type(Self::extract_monotype(&e.data.scheme));
                    let body = self.mk_term(
                        ty.clone(),
                        syntax.span,
                        TermKind::Extern(syntax.linkage_name.clone()),
                    );
                    let arity = count_function_arity(&ty);
                    defs.push(Def {
                        data: data::PolymorphicDefinition {
                            scheme: Some(e.data.scheme.clone()),
                        },
                        name: e.data.source.symbol,
                        ty,
                        body,
                        meta: DefMeta::Function,
                        arity,
                        param_diets: syntax.param_diets.clone(),
                        return_diet: syntax.return_diet.clone(),
                    });
                }
                ast::Declaration::Frontend(never) => match *never {},
            }
        }

        ProgramParts { defs }
    }

    pub fn transform_decl(
        &mut self,
        decl: &ast::Decl<ast::TypedDefinition, ast::HolesResolvedTree>,
    ) -> Option<Def<run::UnpinnedPolymorphic>> {
        let body_ty = Self::type_of(&decl.body.h);
        let full_ty = self.build_function_type(&decl.params, &body_ty);
        let body = self.transform_with_params(&decl.params, &decl.body, full_ty.clone());

        Some(Def {
            data: data::PolymorphicDefinition {
                scheme: Some(decl.data.scheme.clone()),
            },
            name: decl.data.source.symbol,
            ty: full_ty,
            body,
            meta: DefMeta::Function,
            arity: decl.params.len(),
            param_diets: decl.param_diets.clone(),
            return_diet: decl.return_diet.clone(),
        })
    }

    fn transform_entry(
        &mut self,
        entry: &ast::EntryDecl<ast::TypedEntry, ast::HolesResolvedTree, interface::ResolvedAttribute>,
    ) -> Option<Def<run::UnpinnedPolymorphic>> {
        let body_ty = Self::type_of(&entry.body.h);
        let full_ty = self.build_function_type(&entry.params, &body_ty);
        let body = self.transform_with_params(&entry.params, &entry.body, full_ty.clone());

        Some(Def {
            data: data::PolymorphicDefinition {
                scheme: Some(entry.data.scheme.clone()),
            },
            name: entry.data.source.symbol,
            ty: full_ty,
            body,
            meta: DefMeta::EntryPoint(EntryPoint {
                declaration: Box::new(interface::EntryDecl {
                    entry_kind: entry.data.source.source.syntax.entry_kind,
                    compute_dispatch: entry.data.source.source.syntax.compute_dispatch.clone(),
                    graphics_group: None,
                    name: entry.name.clone(),
                    name_span: entry.name_span,
                    size_params: entry.size_params.clone(),
                    type_params: entry.type_params.clone(),
                    params: entry.params.iter().map(Self::lower_entry_param).collect(),
                    outputs: entry.data.source.source.syntax.outputs.clone(),
                    feedback: entry.data.source.source.feedback.clone(),
                    param_diets: entry.data.source.source.syntax.param_diets.clone(),
                    return_diet: entry.data.source.source.syntax.return_diet.clone(),
                }),
                data: (),
            }),
            arity: entry.params.len(),
            param_diets: entry.data.source.source.syntax.param_diets.clone(),
            return_diet: entry.data.source.source.syntax.return_diet.clone(),
        })
    }

    fn build_function_type<A>(
        &self,
        params: &[ast::Pattern<ast::HolesResolvedTree, A>],
        ret_ty: &Type<TypeName>,
    ) -> Type<TypeName> {
        let mut ty = ret_ty.clone();

        for param in params.iter().rev() {
            let param_ty = self.pattern_type(param);
            ty = Type::Constructed(TypeName::Arrow, vec![param_ty, ty]);
        }

        ty
    }

    fn pattern_type<A>(&self, pattern: &ast::Pattern<ast::HolesResolvedTree, A>) -> Type<TypeName> {
        match &pattern.kind {
            // For attributed patterns, recurse into the inner pattern
            ast::PatternKind::Attributed(_, inner) => self.pattern_type(inner),
            _ => Self::type_of(&pattern.h),
        }
    }

    fn lower_entry_param(
        pattern: &ast::Pattern<ast::HolesResolvedTree, interface::ResolvedAttribute>,
    ) -> interface::EntryParamDecl {
        fn metadata(
            pattern: &ast::Pattern<ast::HolesResolvedTree, interface::ResolvedAttribute>,
            attributes: &mut Vec<interface::ResolvedAttribute>,
        ) -> Option<String> {
            match &pattern.kind {
                ast::PatternKind::Name(binding) => Some(binding.source.clone()),
                ast::PatternKind::Attributed(found, inner) => {
                    attributes.extend(found.iter().cloned());
                    metadata(inner, attributes)
                }
                ast::PatternKind::Typed(inner, _) => metadata(inner, attributes),
                _ => None,
            }
        }

        let mut attributes = Vec::new();
        let name = metadata(pattern, &mut attributes).unwrap_or_else(|| "_".to_string());
        interface::EntryParamDecl {
            name,
            span: pattern.h.span,
            ty: Self::type_of(&pattern.h),
            attributes,
        }
    }

    fn transform_with_params<A>(
        &mut self,
        params: &[ast::Pattern<ast::HolesResolvedTree, A>],
        body: &ast::Expression<ast::HolesResolvedTree>,
        full_ty: Type<TypeName>,
    ) -> Term {
        let span = params.first().map(|p| p.h.span).unwrap_or(body.h.span);
        self.build_lambda_chain(params, body, full_ty, span)
    }

    /// Build a chain of nested lambdas from patterns, deferring all let-bindings
    /// until after all lambdas are created. This ensures no let-bindings appear
    /// between nested lambdas, which is important for consistent capture analysis.
    fn build_lambda_chain<A>(
        &mut self,
        params: &[ast::Pattern<ast::HolesResolvedTree, A>],
        body: &ast::Expression<ast::HolesResolvedTree>,
        full_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        if params.is_empty() {
            return self.transform_expr(body);
        }

        // Collect all lambda parameters and their pending bindings
        // compute_pattern_bindings already creates SymbolIds via define()
        let mut lambda_info: Vec<(SymbolId, Type<TypeName>, Vec<PendingBinding>)> = Vec::new();
        let mut current_ty = full_ty;

        // Use the shared placeholder symbol for the scrutinee in compute_pattern_bindings
        let placeholder_sym = self.placeholder_sym;

        for param in params {
            let param_ty = self.get_param_type(&current_ty);

            // Use a placeholder scrutinee - we need to call compute_pattern_bindings to get
            // the param name and projection bindings, but the actual lambda param value
            // won't exist until runtime
            let placeholder = self.mk_term(
                param_ty.clone(),
                span,
                TermKind::Var(VarRef::Symbol(placeholder_sym)),
            );
            let (param_sym, mut bindings) = self.compute_pattern_bindings(param, placeholder, span);

            // For complex patterns (Tuple/Record), compute_pattern_bindings returns bindings that
            // include the top-level binding (fresh = scrutinee). For lambdas, we don't want this
            // since the lambda param IS the fresh name. Skip the first binding if it matches.
            if !bindings.is_empty() && bindings[0].name == param_sym {
                bindings.remove(0);
            }

            lambda_info.push((param_sym, param_ty.clone(), bindings));
            current_ty = self.get_body_type(&current_ty);
        }

        // Transform the body expression
        let mut result = self.transform_expr(body);

        // Apply all bindings in reverse order (innermost first, so outermost ends up innermost)
        for (_, _, bindings) in lambda_info.iter().rev() {
            for binding in bindings.iter().rev() {
                result = self.mk_term(
                    result.ty.clone(),
                    span,
                    TermKind::Let {
                        name: binding.name,
                        name_ty: binding.ty.clone(),
                        rhs: Box::new(binding.expr.clone()),
                        body: Box::new(result),
                    },
                );
            }
        }

        // Build a single flat lambda with all params
        let all_params: Vec<(SymbolId, Type<TypeName>)> =
            lambda_info.into_iter().map(|(sym, ty, _)| (sym, ty)).collect();
        let ret_ty = result.ty.clone();
        let lam_ty = {
            let mut ty = ret_ty.clone();
            for (_, param_ty) in all_params.iter().rev() {
                ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), ty]);
            }
            ty
        };
        result = self.mk_term(
            lam_ty,
            span,
            TermKind::Lambda(Lambda {
                params: all_params,
                ret_ty,
                body: Box::new(result),
            }),
        );

        result
    }

    /// Compute bindings for a pattern matched against a scrutinee variable.
    /// Returns (bound_symbol, list_of_pending_bindings).
    ///
    /// The bound_symbol is either:
    /// - A symbol for the pattern's name (for simple Name patterns)
    /// - A fresh symbol (for complex patterns like Tuple/Record)
    ///
    /// For Name/Wildcard patterns, no bindings are returned - the caller is responsible
    /// for creating the top-level binding if needed (e.g., for let-in).
    ///
    // compute_pattern_bindings + compute_pattern_bindings_inner +
    // build_tuple_projection live in tlc/patterns/bindings.rs (extends
    // this type via an `impl` block there).

    /// Apply a list of bindings around a body term, creating nested let expressions.
    /// Bindings are applied in reverse order so the first binding is outermost.
    pub(super) fn apply_bindings_around(
        &mut self,
        bindings: Vec<PendingBinding>,
        body: Term,
        span: Span,
    ) -> Term {
        bindings.into_iter().rev().fold(body, |acc, b| {
            self.mk_term(
                acc.ty.clone(),
                span,
                TermKind::Let {
                    name: b.name,
                    name_ty: b.ty,
                    rhs: Box::new(b.expr),
                    body: Box::new(acc),
                },
            )
        })
    }

    // simple_pattern_name, extract_tuple_types, resolve_field_index,
    // and extract_record_types live in tlc/patterns/bindings.rs.

    pub(super) fn transform_expr(&mut self, expr: &ast::Expression<ast::HolesResolvedTree>) -> Term {
        let ty = Self::type_of(&expr.h);
        let span = expr.h.span;

        match &expr.kind {
            ast::ExprKind::IntLiteral(s) => self.mk_term(ty, span, TermKind::IntLit(s.0.clone())),

            ast::ExprKind::FloatLiteral(f) => self.mk_term(ty, span, TermKind::FloatLit(*f)),

            ast::ExprKind::BoolLiteral(b) => self.mk_term(ty, span, TermKind::BoolLit(*b)),

            ast::ExprKind::Unit => self.mk_term(ty, span, TermKind::UnitLit),

            ast::ExprKind::Identifier(identifier) => match identifier.resolution {
                ast::IdentifierResolution::Symbol(symbol) => {
                    self.mk_term(ty, span, TermKind::Var(VarRef::Symbol(symbol)))
                }
                ast::IdentifierResolution::Builtin { id, overload_idx } => {
                    self.mk_term(ty, span, TermKind::Var(VarRef::Builtin { id, overload_idx }))
                }
                ast::IdentifierResolution::VecConstructor { .. } | ast::IdentifierResolution::Soac(_) => {
                    panic!(
                        "BUG: constructor/SOAC identifier reached TLC outside application at {:?}",
                        expr.h.id
                    )
                }
            },

            ast::ExprKind::ArrayLiteral(elements) => {
                log::debug!("ArrayLiteral with {} elements", elements.len());
                let terms: Vec<Term> = elements.iter().map(|e| self.transform_expr(e)).collect();
                self.mk_array_lit(terms, ty, span)
            }

            ast::ExprKind::VecMatLiteral(elements) => {
                // For matrices, columns are vectors not arrays
                // Check if result type is Mat and transform columns accordingly
                if ty.is_mat() {
                    // Mat[elem, cols, rows] - column type is Vec[elem, rows]
                    if let (Some(elem), Some(rows_ty)) = (ty.elem_type(), ty.mat_rows_type()) {
                        let col_ty = Type::Constructed(TypeName::Vec, vec![elem.clone(), rows_ty.clone()]);
                        // Transform elements, treating ArrayLiterals as vectors
                        let col_terms: Vec<Term> =
                            elements.iter().map(|e| self.transform_as_vector(e, col_ty.clone())).collect();
                        return self.build_vec_lit_from_terms(&col_terms, ty, span);
                    }
                }
                let terms: Vec<Term> = elements.iter().map(|e| self.transform_expr(e)).collect();
                self.mk_vec_lit(terms, ty, span)
            }

            ast::ExprKind::ArrayIndex(array, index) => {
                let arr = self.transform_expr(array);
                let idx = self.transform_expr(index);
                self.mk_index(arr, idx, ty, span)
            }

            ast::ExprKind::ArrayWith {
                array, index, value, ..
            } => {
                let arr = self.transform_expr(array);
                let idx = self.transform_expr(index);
                let val = self.transform_expr(value);
                let aw_id = if matches!(ty, Type::Constructed(TypeName::StorageTexture, _)) {
                    catalog().known().image_with
                } else {
                    catalog().known().array_with
                };
                self.build_call_by_id(aw_id, &[arr, idx, val], ty, span)
            }

            ast::ExprKind::VecWith {
                target,
                components,
                op,
                value,
            } => self.transform_vec_with(target, components, *op, value, ty, span),

            ast::ExprKind::RecordWith { record, path, value } => {
                self.transform_record_with(record, path, value, ty, span)
            }

            ast::ExprKind::BinaryOp(op, lhs, rhs) => {
                let l = self.transform_expr(lhs);
                let r = self.transform_expr(rhs);
                self.build_binop(op.clone(), l, r, ty, span)
            }

            ast::ExprKind::UnaryOp(op, operand) => {
                let arg = self.transform_expr(operand);
                self.build_unop(op.clone(), arg, ty, span)
            }

            ast::ExprKind::Tuple(elements) => {
                let terms: Vec<Term> = elements.iter().map(|e| self.transform_expr(e)).collect();
                self.mk_tuple(terms, ty, span)
            }

            ast::ExprKind::RecordLiteral(fields) => {
                // Records are tuples - reorder fields to match type's field order
                let field_map: LookupMap<&str, &ast::Expression<ast::HolesResolvedTree>> =
                    fields.iter().map(|(name, expr)| (name.as_str(), expr)).collect();

                let ordered_exprs: Vec<ast::Expression<ast::HolesResolvedTree>> = match &ty {
                    Type::Constructed(TypeName::Record(type_fields), _) => type_fields
                        .iter()
                        .filter_map(|f| field_map.get(f.as_str()).map(|e| (*e).clone()))
                        .collect(),
                    _ => fields.iter().map(|(_, e)| e.clone()).collect(),
                };

                let terms: Vec<Term> = ordered_exprs.iter().map(|e| self.transform_expr(e)).collect();
                self.mk_tuple(terms, ty, span)
            }

            ast::ExprKind::Lambda(lam) => self.transform_lambda(&lam.params, &lam.body, ty, span),

            ast::ExprKind::Application(func, args) => self.transform_application(func, args, ty, span),

            ast::ExprKind::LetIn(let_in) => {
                if let Some(name) = self.simple_pattern_symbol(&let_in.pattern) {
                    let rhs = self.transform_expr(&let_in.value);
                    let body = self.transform_expr(&let_in.body);
                    self.mk_term(
                        body.ty.clone(),
                        span,
                        TermKind::Let {
                            name,
                            name_ty: rhs.ty.clone(),
                            rhs: Box::new(rhs),
                            body: Box::new(body),
                        },
                    )
                } else {
                    let rhs = self.transform_expr(&let_in.value);
                    let (_, bindings) = self.compute_pattern_bindings(&let_in.pattern, rhs, span);
                    let body = self.transform_expr(&let_in.body);
                    self.apply_bindings_around(bindings, body, span)
                }
            }

            ast::ExprKind::FieldAccess(record, field) => {
                let rec = self.transform_expr(record);
                // Vec swizzle (1–4 letters from a single swizzle set —
                // `xyzw` or `rgba`): build per-letter projections;
                // single letter → scalar, multi → _w_vec_lit.
                if rec.ty.is_vec() && crate::types::is_swizzle_field(field) {
                    let elem_ty = rec
                        .ty
                        .elem_type()
                        .cloned()
                        .expect("rec.ty.is_vec() above guarantees a vec elem type");
                    let n_components = field.chars().count();

                    // Single-letter swizzle is one projection — no
                    // duplication concern; project the rec term directly.
                    if n_components == 1 {
                        let idx = crate::types::swizzle_component_index(field.chars().next().unwrap())
                            .expect("is_swizzle_field already accepted this letter");
                        return self.mk_tuple_proj(rec, idx as usize, elem_ty, span);
                    }

                    // Multi-letter swizzle desugars to one
                    // `mk_tuple_proj` per component. If `rec` is a
                    // non-trivial producer (App, Soac, If, Loop, …),
                    // cloning it once per component leaves downstream
                    // passes with several independent copies of the
                    // same producer — egregious when the producer is a
                    // `reduce(...)`: the SoA / CSE layers don't share
                    // them, and the compiled output runs the reduce
                    // once per swizzle slot. Let-bind first so each
                    // projection reads the same evaluated value;
                    // mirrors what `transform_vec_with` does for `with`
                    // updates (`_w_vw_t_…`).
                    let needs_share = !matches!(
                        &rec.kind,
                        TermKind::Var(_)
                            | TermKind::IntLit(_)
                            | TermKind::FloatLit(_)
                            | TermKind::BoolLit(_)
                            | TermKind::UnitLit
                    );

                    let (base, wrap_let): (Term, Option<(SymbolId, Type<TypeName>, Term)>) = if needs_share
                    {
                        let t_id = self.term_ids.next_id();
                        let t_sym = self.fresh(&format!("_w_swz_t_{}", t_id));
                        let t_ty = rec.ty.clone();
                        let var = self.mk_term(t_ty.clone(), span, TermKind::Var(VarRef::Symbol(t_sym)));
                        (var, Some((t_sym, t_ty, rec)))
                    } else {
                        (rec, None)
                    };

                    let components: Vec<Term> = field
                        .chars()
                        .map(|c| {
                            let idx = crate::types::swizzle_component_index(c)
                                .expect("is_swizzle_field already accepted this letter");
                            self.mk_tuple_proj(base.clone(), idx as usize, elem_ty.clone(), span)
                        })
                        .collect();

                    let body = self.build_vec_lit_from_terms(&components, ty.clone(), span);
                    return match wrap_let {
                        Some((name, name_ty, rhs)) => self.mk_term(
                            ty,
                            span,
                            TermKind::Let {
                                name,
                                name_ty,
                                rhs: Box::new(rhs),
                                body: Box::new(body),
                            },
                        ),
                        None => body,
                    };
                }
                // Resolve field name to index, treat record as tuple
                let field_idx = self
                    .resolve_field_index(&rec.ty, field)
                    .unwrap_or_else(|| panic!("BUG: field '{}' not in record type", field));
                self.mk_tuple_proj(rec, field_idx, ty, span)
            }

            ast::ExprKind::If(if_expr) => {
                let cond = self.transform_expr(&if_expr.condition);
                let then_branch = self.transform_expr(&if_expr.then_branch);
                let else_branch = self.transform_expr(&if_expr.else_branch);
                self.mk_term(
                    ty,
                    span,
                    TermKind::If {
                        cond: Box::new(cond),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    },
                )
            }

            ast::ExprKind::Loop(loop_expr) => self.transform_loop(loop_expr, ty, span),

            ast::ExprKind::Match(match_expr) => self.transform_match(match_expr, ty, span),

            ast::ExprKind::Constructor(name, args) => {
                // Lower `#ck(a1..am)` to a flat tuple
                // `(tag=k, slot_1, ..., slot_total-1)` where the active
                // constructor's payload occupies slots [offset_k, offset_k+m)
                // and dead slots get blank-filled.
                let raw_sum_ty = Self::raw_type(&expr.h);
                let variants = match &raw_sum_ty {
                    Type::Constructed(TypeName::Sum(v), _) => v.clone(),
                    Type::Constructed(TypeName::FragmentOutput, args) if args.len() == 1 => {
                        crate::types::fragment_output_variants(args[0].clone())
                    }
                    _ => panic!("BUG: Constructor `#{}` has non-sum type {:?}", name, raw_sum_ty),
                };
                let layout = Self::sum_layout(&variants);
                let &(tag_value, payload_offset) = layout
                    .constructor_info
                    .get(name)
                    .expect("BUG: Phase B should have validated constructor name");

                let arg_terms: Vec<Term> = args.iter().map(|a| self.transform_expr(a)).collect();

                let tag_term = self.mk_term(
                    Type::Constructed(TypeName::UInt(32), vec![]),
                    span,
                    TermKind::IntLit(tag_value.to_string()),
                );
                let mut slot_terms: Vec<Term> = Vec::with_capacity(layout.slot_types.len());
                slot_terms.push(tag_term);
                for slot_idx in 1..layout.slot_types.len() {
                    let slot_ty = &layout.slot_types[slot_idx];
                    if slot_idx >= payload_offset && slot_idx < payload_offset + arg_terms.len() {
                        slot_terms.push(arg_terms[slot_idx - payload_offset].clone());
                    } else {
                        slot_terms.push(self.build_blank(slot_ty, span));
                    }
                }
                self.mk_tuple(slot_terms, ty, span)
            }

            ast::ExprKind::Range(range) => {
                let start = self.transform_expr(&range.start);
                let end = self.transform_expr(&range.end);
                let step = range.step.as_ref().map(|s| self.transform_expr(s));
                let elem_ty = end.ty.clone();
                let minus = ast::BinaryOp {
                    op: BinaryOperator::Subtract,
                };
                let plus = ast::BinaryOp {
                    op: BinaryOperator::Add,
                };
                let div = ast::BinaryOp {
                    op: BinaryOperator::Divide,
                };

                // Element count per range kind:
                //   `a..b`   (Exclusive)   → b - a
                //   `a..<b`  (ExclusiveLt) → b - a
                //   `a..>b`  (ExclusiveGt) → a - b   (descending half-open)
                //   `a...b`  (Inclusive)   → b - a + 1
                let mut len = match range.kind {
                    ast::RangeKind::Exclusive | ast::RangeKind::ExclusiveLt => {
                        self.build_binop(minus.clone(), end.clone(), start.clone(), elem_ty.clone(), span)
                    }
                    ast::RangeKind::ExclusiveGt => {
                        self.build_binop(minus.clone(), start.clone(), end.clone(), elem_ty.clone(), span)
                    }
                    ast::RangeKind::Inclusive => {
                        let one = self.mk_term(elem_ty.clone(), span, TermKind::IntLit("1".to_string()));
                        let diff =
                            self.build_binop(minus, end.clone(), start.clone(), elem_ty.clone(), span);
                        self.build_binop(plus, diff, one, elem_ty.clone(), span)
                    }
                };
                if let Some(ref step_term) = step {
                    len = self.build_binop(div, len, step_term.clone(), elem_ty.clone(), span);
                }

                let range_ae = ArrayExpr::Range {
                    start: Box::new(start),
                    len: Box::new(len),
                    step: step.map(Box::new),
                };
                self.mk_term(ty, span, TermKind::ArrayExpr(range_ae))
            }

            ast::ExprKind::Slice(slice) => {
                // Transform slice to _w_intrinsic_slice(arr, start, end).
                // The slice aliases the source — it's a view, not a copy.
                let arr = self.transform_expr(&slice.array);

                // Omitted start defaults to 0.
                let start = slice
                    .start
                    .as_ref()
                    .map(|e| self.transform_expr(e))
                    .unwrap_or_else(|| self.mk_i32(0, span));

                // Omitted end defaults to `length(arr)`. `_w_intrinsic_length`
                // is registered as returning i32 and works on every array
                // flavor (composite / view / virtual) — the subsequent
                // `buffer_specialize` / SPIR-V passes rewrite it into the
                // right per-flavor lowering.
                let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
                let known = catalog().known();
                let end =
                    slice.end.as_ref().map(|e| self.transform_expr(e)).unwrap_or_else(|| {
                        self.build_call_by_id(known.length, &[arr.clone()], i32_ty, span)
                    });

                self.build_call_by_id(known.slice, &[arr, start, end], ty, span)
            }

            ast::ExprKind::TypeAscription(inner, _) => self.transform_expr(inner),

            ast::ExprKind::TypeCoercion(inner, _) => {
                let term = self.transform_expr(inner);
                let target_ty = ty.clone();
                self.mk_term(
                    ty,
                    span,
                    TermKind::Coerce {
                        inner: Box::new(term),
                        target_ty,
                    },
                )
            }

            ast::ExprKind::TypeHole(never) => match *never {},
        }
    }

    fn transform_lambda(
        &mut self,
        params: &[ast::Pattern<ast::HolesResolvedTree>],
        body: &ast::Expression<ast::HolesResolvedTree>,
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        self.build_lambda_chain(params, body, ty, span)
    }

    fn get_param_type(&self, ty: &Type<TypeName>) -> Type<TypeName> {
        match ty {
            Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => args[0].clone(),
            _ => panic!("BUG: Expected arrow type for function param, got {:?}", ty),
        }
    }

    fn get_body_type(&self, ty: &Type<TypeName>) -> Type<TypeName> {
        match ty {
            Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => args[1].clone(),
            _ => ty.clone(),
        }
    }

    fn transform_application(
        &mut self,
        func: &ast::Expression<ast::HolesResolvedTree>,
        args: &[ast::Expression<ast::HolesResolvedTree>],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        // Lower as a SOAC iff the resolver tagged the callee as one
        // (so a user `def map` shadowing the builtin is a normal call).
        if let Some(kind) = self.resolve_soac(func) {
            return self.transform_soac_call(kind, args, ty, span);
        }

        // Constructor-style vec conversion (`vec2i32(v)`, …).
        // The type checker recorded a `VecConstructor` ResolvedValueRef
        // for the callee. Desugar to a `VecLit` of componentwise scalar
        // conversion calls — `vec2i32(v)` ⟶ `@[i32(v.x), i32(v.y)]`
        // with each `i32(…)` resolved to its concrete per-type catalog
        // entry by the source-component type.
        if let ast::ExprKind::Identifier(identifier) = &func.kind {
            if let ast::IdentifierResolution::VecConstructor {
                arity,
                component_conversion,
            } = &identifier.resolution
            {
                debug_assert_eq!(
                    args.len(),
                    1,
                    "BUG: vec constructor expected 1 arg, got {}",
                    args.len()
                );
                return self.transform_vec_constructor(&args[0], *component_conversion, *arity, ty, span);
            }
        }

        let func_term = self.transform_expr(func);

        if args.is_empty() {
            return func_term;
        }

        let arg_terms: Vec<Term> = args.iter().map(|a| self.transform_expr(a)).collect();

        // If func_term is already an App, flatten by merging args.
        // The AST represents chained calls as nested Application nodes.
        if let TermKind::App { .. } = &func_term.kind {
            let TermKind::App {
                func: inner_func,
                args: inner_args,
            } = func_term.kind
            else {
                unreachable!()
            };
            let mut all_args = inner_args;
            all_args.extend(arg_terms);
            return self.mk_term(
                ty,
                span,
                TermKind::App {
                    func: inner_func,
                    args: all_args,
                },
            );
        }

        self.mk_term(
            ty,
            span,
            TermKind::App {
                func: Box::new(func_term),
                args: arg_terms,
            },
        )
    }

    /// Synthesise `vec<N><target_elem>(v)` as a `VecLit` of N
    /// componentwise scalar conversion calls. Each component is
    /// `<target_elem>.<source_elem>(v.<i>)`, where `<source_elem>` is
    /// read from `v`'s converted Term type.
    fn transform_vec_constructor(
        &mut self,
        arg: &ast::Expression<ast::HolesResolvedTree>,
        component_conversion: BuiltinId,
        arity: usize,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let arg_term = self.transform_expr(arg);
        // The arg is a vec whose elem type tells us the source-elem
        // module name. `i32`-to-`f32` keys map to the catalog entry
        // `f32.i32` (target.source); we form that surface name and
        // dispatch via the builtin catalog.
        let source_elem_ty = arg_term
            .ty
            .elem_type()
            .expect("vec constructor arg must be a vec — type checker enforces this")
            .clone();
        let target_elem_ty =
            result_ty.elem_type().expect("vec constructor result type is always a vec").clone();

        // Bind the arg once to a synthetic let so each component
        // projection reuses the same evaluation. `SymbolId(0)` is the
        // shared "sequence" sentinel — fine for unit-typed bindings
        // but here we want the value preserved, so allocate a fresh
        // name.
        let arg_sym = self.fresh("_w_vec_conv_arg");
        let arg_ref = self.mk_term(arg_term.ty.clone(), span, TermKind::Var(VarRef::Symbol(arg_sym)));

        // Build N per-component conversion calls.
        let mut components: Vec<Term> = Vec::with_capacity(arity);
        for i in 0..arity {
            let proj = self.mk_tuple_proj(arg_ref.clone(), i, source_elem_ty.clone(), span);
            let conv_func = self.mk_term(
                Type::Constructed(
                    TypeName::Arrow,
                    vec![source_elem_ty.clone(), target_elem_ty.clone()],
                ),
                span,
                TermKind::Var(VarRef::Builtin {
                    id: component_conversion,
                    overload_idx: 0,
                }),
            );
            let conv_call = self.mk_term(
                target_elem_ty.clone(),
                span,
                TermKind::App {
                    func: Box::new(conv_func),
                    args: vec![proj],
                },
            );
            components.push(conv_call);
        }

        let vec_lit = self.build_vec_lit_from_terms(&components, result_ty.clone(), span);

        // Wrap in `let _w_vec_conv_arg = <arg_term> in <vec_lit>`.
        self.mk_term(
            result_ty.clone(),
            span,
            TermKind::Let {
                name: arg_sym,
                name_ty: arg_term.ty.clone(),
                rhs: Box::new(arg_term),
                body: Box::new(vec_lit),
            },
        )
    }

    /// The SOAC this call's callee denotes, per the frontend resolver —
    /// `None` for everything else, including a user `def` (top-level or
    /// local) that shadows a SOAC name. Structural: no surface-name match
    /// or scope re-derivation here; the resolver already decided.
    fn resolve_soac(&self, func: &ast::Expression<ast::HolesResolvedTree>) -> Option<ast::SoacKind> {
        match &func.kind {
            ast::ExprKind::Identifier(identifier) => match identifier.resolution {
                ast::IdentifierResolution::Soac(kind) => Some(kind),
                _ => None,
            },
            _ => None,
        }
    }

    /// Dispatch SOAC call by structural kind.
    fn transform_soac_call(
        &mut self,
        kind: ast::SoacKind,
        args: &[ast::Expression<ast::HolesResolvedTree>],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        match kind {
            ast::SoacKind::Replicate => self.transform_replicate(args, ty, span),
            ast::SoacKind::Map => self.transform_soac_map(args, ty, span),
            ast::SoacKind::Reduce => self.transform_soac_reduce(args, ty, span),
            ast::SoacKind::Scan => self.transform_soac_scan(args, ty, span),
            ast::SoacKind::Filter => self.transform_soac_filter(args, ty, span),
            ast::SoacKind::Zip => self.transform_soac_zip(args, ty, span),
            ast::SoacKind::ReduceByIndex => self.transform_soac_reduce_by_index(args, ty, span),
            ast::SoacKind::Scatter => self.transform_soac_scatter(args, ty, span),
            ast::SoacKind::BucketScatter(rank) => self.transform_soac_bucket_scatter(args, ty, span, rank),
        }
    }

    /// Transform `replicate(n, value)` into a map over the index range
    /// `[0, n)`. Binding `value` outside the map preserves ordinary argument
    /// evaluation and makes nested-array replication an explicit capture
    /// instead of recomputing its producer in every logical work item.
    fn transform_replicate(
        &mut self,
        args: &[ast::Expression<ast::HolesResolvedTree>],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        assert_eq!(args.len(), 2, "replicate requires 2 arguments");
        let size = self.transform_expr(&args[0]);
        let value = self.transform_expr(&args[1]);
        let value_ty = value.ty.clone();
        let value_name = self.fresh("_w_replicate_value");
        let value_ref = self.mk_term(
            value_ty.clone(),
            value.span,
            TermKind::Var(VarRef::Symbol(value_name)),
        );
        let index_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let index = self.fresh("_w_replicate_index");
        let zero = self.mk_term(index_ty.clone(), span, TermKind::IntLit("0".into()));
        let map = self.mk_term(
            ty.clone(),
            span,
            TermKind::Soac(SoacOp::Map {
                lam: SoacBody {
                    lam: Lambda {
                        params: vec![(index, index_ty)],
                        body: Box::new(value_ref),
                        ret_ty: value_ty.clone(),
                    },
                    data: (),
                },
                inputs: vec![ArrayExpr::Range {
                    start: Box::new(zero),
                    len: Box::new(size),
                    step: None,
                }],
                destination: SoacOwnership::Fresh,
            }),
        );
        self.mk_term(
            ty,
            span,
            TermKind::Let {
                name: value_name,
                name_ty: value_ty,
                rhs: Box::new(value),
                body: Box::new(map),
            },
        )
    }

    /// Convert a transformed array-argument term into an ANF SOAC input. A bare
    /// variable passes through as `Var`; any other term (a producer SOAC, a
    /// call, …) is let-bound to a fresh `_anf` name, with the binding pushed to
    /// `binds` for the caller to wrap around the SOAC via [`Self::wrap_binds`].
    fn soac_input(
        &mut self,
        arr_term: Term,
        binds: &mut Vec<(SymbolId, Type<TypeName>, Term)>,
    ) -> ArrayExpr {
        // Lift any binding lets above the SOAC (e.g. `iota(N)` desugars to
        // `let arg = N in Range{…}`), keeping the input itself atomic.
        let (mut peeled, core) = peel_lets(arr_term);
        binds.append(&mut peeled);
        match core.kind {
            TermKind::Var(vr) => ArrayExpr::Var(vr, core.ty),
            // An array expression (Range / Literal / Zip) is
            // itself an atomic SOAC input; consume it directly rather than
            // let-binding a name to it.
            TermKind::ArrayExpr(ae) => ae,
            _ => {
                let ty = core.ty.clone();
                let sym = self.fresh("_anf");
                binds.push((sym, ty.clone(), core));
                ArrayExpr::Var(VarRef::Symbol(sym), ty)
            }
        }
    }

    /// Wrap `binds` as nested `let`s (outermost first) around `body`.
    fn wrap_binds(&mut self, binds: Vec<(SymbolId, Type<TypeName>, Term)>, body: Term, span: Span) -> Term {
        let mut result = body;
        for (name, name_ty, rhs) in binds.into_iter().rev() {
            let body_ty = result.ty.clone();
            result = self.mk_term(
                body_ty,
                span,
                TermKind::Let {
                    name,
                    name_ty,
                    rhs: Box::new(rhs),
                    body: Box::new(result),
                },
            );
        }
        result
    }

    /// Transform `map(f, arr)` → `Soac(Map { lam, inputs })`.
    fn transform_soac_map(
        &mut self,
        args: &[ast::Expression<ast::HolesResolvedTree>],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        assert!(args.len() >= 2, "map requires at least 2 arguments");
        let func_term = self.transform_expr(&args[0]);
        let arr_term = self.transform_expr(&args[1]);

        let lam = self.term_to_lambda(func_term);

        // Absorb zip: if arr_term is ArrayExpr(Zip(...)), flatten into inputs.
        // The lambda still takes a single tuple param — the soa::normalize pass
        // will rewrite it to take separate params. A zip whose children needed
        // let-binding arrives wrapped in those lets (from `transform_soac_zip`),
        // so peel them off and re-wrap around the whole map.
        let (mut binds, core) = peel_lets(arr_term);
        let inputs = match core.kind {
            TermKind::ArrayExpr(ArrayExpr::Zip(exprs)) => exprs,
            _ => vec![self.soac_input(core, &mut binds)],
        };

        let soac = self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::Map {
                lam,
                inputs,
                destination: SoacOwnership::Fresh,
            }),
        );
        self.wrap_binds(binds, soac, span)
    }

    /// Transform `reduce(op, ne, arr)` → `Soac(Reduce { op, ne, input })`.
    fn transform_soac_reduce(
        &mut self,
        args: &[ast::Expression<ast::HolesResolvedTree>],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        assert!(args.len() >= 3, "reduce requires 3 arguments");
        let op_term = self.transform_expr(&args[0]);
        let ne_term = self.transform_expr(&args[1]);
        let arr_term = self.transform_expr(&args[2]);

        let op = self.term_to_lambda(op_term);

        let mut binds = Vec::new();
        let input = self.soac_input(arr_term, &mut binds);
        let soac = self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::Reduce {
                op,
                ne: Box::new(ne_term),
                input,
            }),
        );
        self.wrap_binds(binds, soac, span)
    }

    /// Transform `scan(op, ne, arr)` → `Soac(Scan { op, ne, input })`.
    fn transform_soac_scan(
        &mut self,
        args: &[ast::Expression<ast::HolesResolvedTree>],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        assert!(args.len() >= 3, "scan requires 3 arguments");
        let op_term = self.transform_expr(&args[0]);
        let ne_term = self.transform_expr(&args[1]);
        let arr_term = self.transform_expr(&args[2]);

        let op = self.term_to_lambda(op_term);

        let mut binds = Vec::new();
        let input = self.soac_input(arr_term, &mut binds);
        let soac = self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::Scan {
                op,
                ne: Box::new(ne_term),
                input,
                // Initial construction; apply_ownership may flip later.
                destination: SoacOwnership::Fresh,
            }),
        );
        self.wrap_binds(binds, soac, span)
    }

    /// Transform `filter(pred, arr)` → `Soac(Filter { pred, input })`.
    fn transform_soac_filter(
        &mut self,
        args: &[ast::Expression<ast::HolesResolvedTree>],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        assert!(args.len() >= 2, "filter requires 2 arguments");
        let pred_term = self.transform_expr(&args[0]);
        let arr_term = self.transform_expr(&args[1]);

        let pred = self.term_to_lambda(pred_term);

        let mut binds = Vec::new();
        let input = self.soac_input(arr_term, &mut binds);
        let soac = self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::Filter {
                pred,
                input,
                // Initial construction; apply_ownership may flip later.
                destination: SoacOwnership::Fresh,
            }),
        );
        self.wrap_binds(binds, soac, span)
    }

    /// Transform `zip(a, b, ...)` → `ArrayExpr(Zip(...))`. Each child becomes an
    /// ANF atom; any producer child is let-bound, the bindings wrapping the zip
    /// term (a consuming `map` peels them back off — see `transform_soac_map`).
    fn transform_soac_zip(
        &mut self,
        args: &[ast::Expression<ast::HolesResolvedTree>],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let mut binds = Vec::new();
        let mut exprs = Vec::with_capacity(args.len());
        for a in args {
            let t = self.transform_expr(a);
            exprs.push(self.soac_input(t, &mut binds));
        }
        let zip = self.mk_term(ty, span, TermKind::ArrayExpr(ArrayExpr::Zip(exprs)));
        self.wrap_binds(binds, zip, span)
    }

    /// Transform `reduce_by_index(dest, op, ne, indices, values)`.
    fn transform_soac_reduce_by_index(
        &mut self,
        args: &[ast::Expression<ast::HolesResolvedTree>],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        assert!(args.len() >= 5, "reduce_by_index requires 5 arguments");
        let dest_term = self.transform_expr(&args[0]);
        let op_term = self.transform_expr(&args[1]);
        let ne_term = self.transform_expr(&args[2]);
        let indices_term = self.transform_expr(&args[3]);
        let values_term = self.transform_expr(&args[4]);

        let op = self.term_to_lambda(op_term);

        // Build a Place from dest_term
        let dest_elem_ty = self.get_array_element_type(&dest_term.ty);
        let dest = Place {
            id: match &dest_term.kind {
                TermKind::Var(VarRef::Symbol(sym)) => sym.clone(),
                _ => {
                    // Bind dest to a fresh name
                    let fresh = self.fresh("_w_rbi_dest");
                    fresh
                }
            },
            elem_ty: dest_elem_ty,
        };

        let mut binds = Vec::new();
        let indices = self.soac_input(indices_term, &mut binds);
        let values = self.soac_input(values_term, &mut binds);
        let soac = self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::ReduceByIndex {
                dest,
                op,
                ne: Box::new(ne_term),
                indices,
                values,
            }),
        );
        self.wrap_binds(binds, soac, span)
    }

    /// `scatter(dest, indices, values)` → `SoacOp::Scatter`. Writes
    /// `values[i]` into `dest[indices[i]]` for each `i`; out-of-bounds indices
    /// are ignored (Futhark semantics). The `dest` must be a Var (a `#[storage]`
    /// buffer param in the rasterizer use case) — its `Place`
    /// carries the symbol the EGIR conversion resolves to the dest's view.
    fn transform_soac_scatter(
        &mut self,
        args: &[ast::Expression<ast::HolesResolvedTree>],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        assert!(args.len() >= 3, "scatter requires 3 arguments");
        let dest_term = self.transform_expr(&args[0]);
        let indices_term = self.transform_expr(&args[1]);
        let values_term = self.transform_expr(&args[2]);

        let dest_elem_ty = self.get_array_element_type(&dest_term.ty);
        let idx_elem_ty = self.get_array_element_type(&indices_term.ty);
        let val_elem_ty = self.get_array_element_type(&values_term.ty);
        let dest = Place {
            id: match &dest_term.kind {
                TermKind::Var(VarRef::Symbol(sym)) => sym.clone(),
                _ => self.fresh("_w_scatter_dest"),
            },
            elem_ty: dest_elem_ty,
        };

        // Identity envelope `λ(i, v) → (i, v)`. Fusion composes producer
        // lambdas into this and splices their inputs in place of `is`/`vs`.
        let i_sym = self.fresh("_w_scatter_i");
        let v_sym = self.fresh("_w_scatter_v");
        let i_var = self.mk_term(idx_elem_ty.clone(), span, TermKind::Var(VarRef::Symbol(i_sym)));
        let v_var = self.mk_term(val_elem_ty.clone(), span, TermKind::Var(VarRef::Symbol(v_sym)));
        let tuple_ty =
            Type::Constructed(TypeName::Tuple(2), vec![idx_elem_ty.clone(), val_elem_ty.clone()]);
        let body = self.mk_tuple(vec![i_var, v_var], tuple_ty.clone(), span);
        let lam = SoacBody {
            lam: Lambda {
                params: vec![(i_sym, idx_elem_ty), (v_sym, val_elem_ty)],
                body: Box::new(body),
                ret_ty: tuple_ty,
            },
            data: (),
        };

        let mut binds = Vec::new();
        let indices = self.soac_input(indices_term, &mut binds);
        let values = self.soac_input(values_term, &mut binds);
        let soac = self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::Scatter {
                dest,
                lam,
                inputs: vec![indices, values],
            }),
        );
        self.wrap_binds(binds, soac, span)
    }

    /// Transform `bucket_scatter_Nd(dest, items)`, where every ranked item
    /// leaf is a `(bucket_key, value)` pair.
    fn transform_soac_bucket_scatter(
        &mut self,
        args: &[ast::Expression<ast::HolesResolvedTree>],
        ty: Type<TypeName>,
        span: Span,
        input_rank: u8,
    ) -> Term {
        assert_eq!(args.len(), 2, "bucket_scatter_Nd requires 2 arguments");
        let dest_term = self.transform_expr(&args[0]);
        let items_term = self.transform_expr(&args[1]);

        let dest_row_ty = self.get_array_element_type(&dest_term.ty);
        let dest_elem_ty = self.get_array_element_type(&dest_row_ty);
        let mut item_ty = items_term.ty.clone();
        for _ in 0..input_rank {
            item_ty = self.get_array_element_type(&item_ty);
        }
        let Type::Constructed(TypeName::Tuple(2), pair_types) = &item_ty else {
            panic!("BUG: bucket_scatter_Nd item must be a (key, value) pair, got {item_ty:?}");
        };
        let key_ty = pair_types[0].clone();
        let value_ty = pair_types[1].clone();
        let pair_ty = Type::Constructed(TypeName::Tuple(2), vec![key_ty.clone(), value_ty.clone()]);
        let dest = Place {
            id: match &dest_term.kind {
                TermKind::Var(VarRef::Symbol(symbol)) => *symbol,
                _ => self.fresh("_w_bucket_scatter_dest"),
            },
            elem_ty: dest_elem_ty,
        };

        let mut binds = Vec::new();
        let items = self.soac_input(items_term, &mut binds);
        let item = self.fresh("_w_bucket_scatter_item");
        let item_value = self.mk_term(pair_ty.clone(), span, TermKind::Var(VarRef::Symbol(item)));
        let key = self.mk_tuple_proj(item_value.clone(), 0, key_ty.clone(), span);
        let value = self.mk_tuple_proj(item_value, 1, value_ty.clone(), span);
        let zero = self.mk_term(key_ty.clone(), span, TermKind::IntLit("0".into()));
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
        let active = self.build_binop(
            ast::BinaryOp {
                op: crate::op::BinaryOperator::GreaterEqual,
            },
            key.clone(),
            zero,
            bool_ty.clone(),
            span,
        );
        let emission_ty = Type::Constructed(TypeName::Tuple(3), vec![bool_ty, key_ty, value_ty]);
        let body = self.mk_tuple(vec![active, key, value], emission_ty.clone(), span);
        let lam = SoacBody {
            lam: Lambda {
                params: vec![(item, pair_ty.clone())],
                body: Box::new(body),
                ret_ty: emission_ty,
            },
            data: (),
        };
        let soac = self.mk_term(
            ty,
            span,
            TermKind::Soac(SoacOp::BucketScatter {
                dest,
                lam,
                inputs: vec![items],
                input_dimensions: vec![(0..input_rank).collect()],
                domain_rank: input_rank,
            }),
        );
        self.wrap_binds(binds, soac, span)
    }

    /// Convert a term to a SoacBody. If it's already a Lambda, wrap it.
    /// Otherwise, eta-expand all parameters: `f : A -> B -> C` → `|a, b| f(a)(b)`.
    /// Captures are always empty here — this runs pre-defunctionalization.
    fn term_to_lambda(&mut self, term: Term) -> SoacBody {
        match term.kind {
            TermKind::Lambda(lam) => SoacBody { lam, data: () },
            _ => {
                // Decompose the full arrow chain: A -> B -> C gives ([A, B], C)
                let mut param_tys = Vec::new();
                let mut current = term.ty.clone();
                while let Type::Constructed(TypeName::Arrow, ref args) = current {
                    if args.len() == 2 {
                        param_tys.push(args[0].clone());
                        current = args[1].clone();
                    } else {
                        break;
                    }
                }
                assert!(
                    !param_tys.is_empty(),
                    "BUG: Expected arrow type for SOAC function arg, got {:?}",
                    term.ty
                );
                let ret_ty = current;

                // Create parameter symbols. Display names must be distinct
                // per-parameter — SPIR-V keys off numeric parameter ids so
                // it survives duplicate names, but WGSL inherits the display
                // name verbatim and rejects a function whose parameter list
                // repeats a name.
                let params: Vec<(SymbolId, Type<TypeName>)> = param_tys
                    .iter()
                    .enumerate()
                    .map(|(i, ty)| (self.fresh(&format!("_soac_arg_{}", i)), ty.clone()))
                    .collect();

                // Build flat App(f, [a, b, ...])
                let span = term.span;
                let arg_terms: Vec<Term> = params
                    .iter()
                    .map(|(sym, ty)| self.mk_term(ty.clone(), span, TermKind::Var(VarRef::Symbol(*sym))))
                    .collect();
                let body = self.mk_term(
                    ret_ty.clone(),
                    span,
                    TermKind::App {
                        func: Box::new(term),
                        args: arg_terms,
                    },
                );

                SoacBody {
                    lam: Lambda {
                        params,
                        body: Box::new(body),
                        ret_ty,
                    },
                    data: (),
                }
            }
        }
    }

    fn transform_loop(
        &mut self,
        loop_expr: &ast::LoopExpr<ast::HolesResolvedTree>,
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        // Get the init expression and accumulator type
        let init_term = loop_expr.init.as_ref().map(|e| self.transform_expr(e)).unwrap_or_else(|| {
            // No accumulator - use unit
            self.mk_term(Type::Constructed(TypeName::Unit, vec![]), span, TermKind::UnitLit)
        });
        let acc_ty = init_term.ty.clone();

        // Build loop_var and init_bindings from the pattern
        let (loop_var, loop_var_ty, init_bindings) =
            self.build_loop_var_and_bindings(&loop_expr.pattern, &acc_ty, span);

        match &loop_expr.form {
            ast::LoopForm::For(idx_pattern, bound) => {
                let bound_term = self.transform_expr(bound);
                let index_ty = Type::Constructed(TypeName::Int(32), vec![]);
                let idx_var_sym = self
                    .simple_pattern_symbol(idx_pattern)
                    .expect("BUG: range-loop binder must be a simple pattern");

                // Transform body after defining the index variable
                let body = self.transform_expr(&loop_expr.body);

                self.mk_term(
                    ty,
                    span,
                    TermKind::Loop {
                        loop_var,
                        loop_var_ty,
                        init: Box::new(init_term),
                        init_bindings,
                        kind: LoopKind::ForRange {
                            var: idx_var_sym,
                            var_ty: index_ty,
                            bound: Box::new(bound_term),
                        },
                        body: Box::new(body),
                    },
                )
            }

            ast::LoopForm::ForIn(elem_pattern, iter) => {
                let iter_term = self.transform_expr(iter);
                let elem_ty = self.get_array_element_type(&iter_term.ty);
                let elem_var_sym =
                    self.simple_pattern_symbol(elem_pattern).unwrap_or_else(|| self.fresh("_w_elem"));

                // Transform body after defining the element variable
                let body = self.transform_expr(&loop_expr.body);

                self.mk_term(
                    ty,
                    span,
                    TermKind::Loop {
                        loop_var,
                        loop_var_ty,
                        init: Box::new(init_term),
                        init_bindings,
                        kind: LoopKind::For {
                            var: elem_var_sym,
                            var_ty: elem_ty,
                            iter: Box::new(iter_term),
                        },
                        body: Box::new(body),
                    },
                )
            }

            ast::LoopForm::While(cond) => {
                let body = self.transform_expr(&loop_expr.body);
                let cond_term = self.transform_expr(cond);

                self.mk_term(
                    ty,
                    span,
                    TermKind::Loop {
                        loop_var,
                        loop_var_ty,
                        init: Box::new(init_term),
                        init_bindings,
                        kind: LoopKind::While {
                            cond: Box::new(cond_term),
                        },
                        body: Box::new(body),
                    },
                )
            }
        }
    }

    /// Build loop variable name and init_bindings from a pattern.
    fn build_loop_var_and_bindings(
        &mut self,
        pattern: &ast::Pattern<ast::HolesResolvedTree>,
        acc_ty: &Type<TypeName>,
        span: Span,
    ) -> (SymbolId, Type<TypeName>, Vec<(SymbolId, Type<TypeName>, Term)>) {
        use crate::pattern::binding_paths;

        // For a simple name pattern, use it directly
        if let ast::PatternKind::Name(binding) = &pattern.kind {
            return (binding.symbol, acc_ty.clone(), vec![]);
        }

        // For complex patterns, create a fresh loop_var and build projections
        let loop_var_name = format!("_w_loop_{}", self.term_ids.next_id());
        let loop_var_sym = self.fresh(&loop_var_name);
        let paths = binding_paths(pattern);

        let init_bindings = paths
            .into_iter()
            .filter_map(|bp| {
                if bp.path.is_empty() {
                    // This is the root binding - shouldn't happen for complex patterns
                    None
                } else {
                    let binding_ty = self.type_at_path(acc_ty, &bp.path);
                    let proj_term = self.build_projection_chain(loop_var_sym, acc_ty, &bp.path, span);
                    let binding_sym = bp.symbol.unwrap_or_else(|| {
                        panic!("BUG: loop binding '{}' lacks its resolved identity", bp.name)
                    });
                    Some((binding_sym, binding_ty, proj_term))
                }
            })
            .collect();

        (loop_var_sym, acc_ty.clone(), init_bindings)
    }

    /// Get the type at a given projection path within a tuple/record type.
    fn type_at_path(&self, ty: &Type<TypeName>, path: &[usize]) -> Type<TypeName> {
        let mut current = ty.clone();
        for &idx in path {
            current = match &current {
                Type::Constructed(TypeName::Tuple(_), args) => {
                    args.get(idx).cloned().unwrap_or_else(|| {
                        panic!(
                            "BUG: tuple projection index {} out of bounds for {:?}",
                            idx, current
                        )
                    })
                }
                Type::Constructed(TypeName::Record(fields), args) => {
                    args.get(idx).cloned().unwrap_or_else(|| {
                        panic!(
                            "BUG: record projection index {} out of bounds for {:?} (fields: {:?})",
                            idx, current, fields
                        )
                    })
                }
                _ => panic!("BUG: projection on non-tuple/record type: {:?}", current),
            };
        }
        current
    }

    /// Build a chain of tuple projections: proj[path[n-1]](...proj[path[0]](var))
    fn build_projection_chain(
        &mut self,
        var_sym: SymbolId,
        var_ty: &Type<TypeName>,
        path: &[usize],
        span: Span,
    ) -> Term {
        let mut current_ty = var_ty.clone();
        let mut current = self.mk_term(current_ty.clone(), span, TermKind::Var(VarRef::Symbol(var_sym)));

        for &idx in path {
            let elem_ty = self.type_at_path(&current_ty, &[idx]);
            current = self.mk_tuple_proj(current, idx, elem_ty.clone(), span);
            current_ty = elem_ty;
        }

        current
    }

    fn get_array_element_type(&self, ty: &Type<TypeName>) -> Type<TypeName> {
        ty.elem_type()
            .filter(|_| ty.is_array())
            .cloned()
            .unwrap_or_else(|| panic!("BUG: Expected array type, got {:?}", ty))
    }

    /// Lower `target with .swizzle [op]= value` into a let-bound
    /// vec-build:
    ///
    /// ```text
    ///   let _t = target in
    ///   let _r = value (or _t.swizzle <op> value, for compound) in
    ///   _w_vec_lit(_t.0, ..., _r.0 or _t.i, ..., _t.{N-1})
    /// ```
    ///
    /// `_t` and `_r` are bound to fresh symbols so the inputs evaluate
    /// once even when they're arbitrary expressions.
    fn transform_vec_with(
        &mut self,
        target: &ast::Expression<ast::HolesResolvedTree>,
        components: &[u8],
        op: Option<BinaryOperator>,
        value: &ast::Expression<ast::HolesResolvedTree>,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let t_term = self.transform_expr(target);
        let target_ty = t_term.ty.clone();
        let elem_ty = target_ty.elem_type().cloned().expect("VecWith target must be a vec by type-check");
        let vec_size = target_ty.vec_size().expect("VecWith target must have known size");

        // Bind `_t = target` so each per-slot projection reads the
        // same evaluated value.
        let t_id = self.term_ids.next_id();
        let t_sym = self.fresh(&format!("_w_vw_t_{}", t_id));
        let t_var = self.mk_term(target_ty.clone(), span, TermKind::Var(VarRef::Symbol(t_sym)));

        // Compute the RHS term. For plain `=`, that's just `value`.
        // For compound `op=`, build `_t.swizzle <op> value` so the
        // existing binary-op machinery handles vec-vec / vec-mat /
        // vec-scalar dispatch identically to a hand-written
        // `t.swizzle op rhs`.
        let v_term_raw = self.transform_expr(value);
        let rhs_term = match op {
            None => v_term_raw,
            Some(op) => {
                let swizzle_read = self.build_swizzle_read(&t_var, components, &elem_ty, span);
                let result_slot_ty = swizzle_read.ty.clone();
                self.build_binop(
                    ast::BinaryOp { op },
                    swizzle_read,
                    v_term_raw,
                    result_slot_ty,
                    span,
                )
            }
        };

        // Bind `_r = <rhs>` so per-slot reads share one evaluation.
        let r_id = self.term_ids.next_id();
        let r_sym = self.fresh(&format!("_w_vw_r_{}", r_id));
        let r_var = self.mk_term(rhs_term.ty.clone(), span, TermKind::Var(VarRef::Symbol(r_sym)));

        // Locate each component's position in `components` so we know
        // which RHS slot supplies each target slot.
        let component_pos: Vec<Option<usize>> =
            (0..vec_size).map(|slot| components.iter().position(|&c| c as usize == slot)).collect();

        // Build per-slot terms: RHS slot for swizzle positions,
        // original target slot otherwise.
        let single_component = components.len() == 1;
        let slot_terms: Vec<Term> = component_pos
            .iter()
            .enumerate()
            .map(|(slot, found)| match found {
                Some(rhs_pos) => {
                    if single_component {
                        // RHS is the elem type itself, not a vec.
                        r_var.clone()
                    } else {
                        self.build_proj(&r_var, *rhs_pos, &elem_ty, span)
                    }
                }
                None => self.build_proj(&t_var, slot, &elem_ty, span),
            })
            .collect();

        let body = self.build_vec_lit_from_terms(&slot_terms, result_ty.clone(), span);

        // Wrap: let _t = target in let _r = rhs in body.
        let inner = self.mk_term(
            result_ty.clone(),
            span,
            TermKind::Let {
                name: r_sym,
                name_ty: rhs_term.ty.clone(),
                rhs: Box::new(rhs_term),
                body: Box::new(body),
            },
        );
        self.mk_term(
            result_ty,
            span,
            TermKind::Let {
                name: t_sym,
                name_ty: target_ty,
                rhs: Box::new(t_term),
                body: Box::new(inner),
            },
        )
    }

    /// Lower `r with field = e` (single-level) and `r with a.x = e`
    /// (nested) by binding the record to a fresh symbol and rebuilding
    /// it via `_w_tuple` with the path target replaced. Each level of
    /// the path produces its own bind-and-rebuild; nested paths chain
    /// inside the outer rebuild's replacement slot.
    fn transform_record_with(
        &mut self,
        record: &ast::Expression<ast::HolesResolvedTree>,
        path: &[String],
        value: &ast::Expression<ast::HolesResolvedTree>,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let r_term = self.transform_expr(record);
        let record_ty = r_term.ty.clone();
        let new_value = self.transform_expr(value);

        let r_id = self.term_ids.next_id();
        let r_sym = self.fresh(&format!("_w_rw_r_{}", r_id));
        let r_var = self.mk_term(record_ty.clone(), span, TermKind::Var(VarRef::Symbol(r_sym)));

        let body = self.build_record_with_body(&r_var, &record_ty, path, new_value, span);

        self.mk_term(
            result_ty,
            span,
            TermKind::Let {
                name: r_sym,
                name_ty: record_ty,
                rhs: Box::new(r_term),
                body: Box::new(body),
            },
        )
    }

    /// Recursive builder for `transform_record_with`. `target` is a
    /// Var term referring to the record at this level of the path.
    fn build_record_with_body(
        &mut self,
        target: &Term,
        record_ty: &Type<TypeName>,
        path: &[String],
        new_value: Term,
        span: Span,
    ) -> Term {
        let (fields, field_types) = match record_ty {
            Type::Constructed(TypeName::Record(fs), tys) => (fs, tys),
            _ => panic!("BUG: record-with target must be a record type at lowering"),
        };
        let head = &path[0];
        let idx = fields.get_index(head).expect("BUG: typeck verified record field exists");

        let replacement = if path.len() == 1 {
            new_value
        } else {
            let inner_ty = field_types[idx].clone();
            let inner_proj = self.build_proj(target, idx, &inner_ty, span);
            let inner_id = self.term_ids.next_id();
            let inner_sym = self.fresh(&format!("_w_rw_inner_{}", inner_id));
            let inner_var = self.mk_term(inner_ty.clone(), span, TermKind::Var(VarRef::Symbol(inner_sym)));
            let inner_body =
                self.build_record_with_body(&inner_var, &inner_ty, &path[1..], new_value, span);
            self.mk_term(
                inner_ty.clone(),
                span,
                TermKind::Let {
                    name: inner_sym,
                    name_ty: inner_ty,
                    rhs: Box::new(inner_proj),
                    body: Box::new(inner_body),
                },
            )
        };

        let field_terms: Vec<Term> = (0..fields.len())
            .map(|i| {
                if i == idx {
                    replacement.clone()
                } else {
                    self.build_proj(target, i, &field_types[i], span)
                }
            })
            .collect();

        self.mk_tuple(field_terms, record_ty.clone(), span)
    }

    /// Build a swizzle read on a Var term: emits per-letter
    /// `_w_tuple_proj` calls and assembles them with
    /// `_w_vec_lit_from_terms` (or returns the single term when
    /// `components.len() == 1`).
    fn build_swizzle_read(
        &mut self,
        target_var: &Term,
        components: &[u8],
        elem_ty: &Type<TypeName>,
        span: Span,
    ) -> Term {
        let projs: Vec<Term> =
            components.iter().map(|&c| self.build_proj(target_var, c as usize, elem_ty, span)).collect();
        if projs.len() == 1 {
            projs.into_iter().next().unwrap()
        } else {
            let result_ty = Type::Constructed(
                TypeName::Vec,
                vec![
                    elem_ty.clone(),
                    Type::Constructed(TypeName::Size(components.len()), vec![]),
                ],
            );
            self.build_vec_lit_from_terms(&projs, result_ty, span)
        }
    }

    /// Build `TermKind::TupleProj` returning `result_ty`.
    fn build_proj(&mut self, target: &Term, idx: usize, result_ty: &Type<TypeName>, span: Span) -> Term {
        self.mk_tuple_proj(target.clone(), idx, result_ty.clone(), span)
    }

    fn transform_match(
        &mut self,
        match_expr: &ast::MatchExpr<ast::HolesResolvedTree>,
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        debug_assert!(
            !match_expr.cases.is_empty(),
            "checker rejects empty match upstream"
        );
        self.compile_match(match_expr, ty, span)
    }

    /// Produce a typed-blank Term for `ty`. Fills dead
    /// constructor-payload slots in a flattened sum-type tuple.
    fn build_blank(&mut self, ty: &Type<TypeName>, span: Span) -> Term {
        match ty {
            Type::Constructed(TypeName::Int(_), _) | Type::Constructed(TypeName::UInt(_), _) => {
                self.mk_term(ty.clone(), span, TermKind::IntLit("0".to_string()))
            }
            Type::Constructed(TypeName::Float(_), _) => {
                self.mk_term(ty.clone(), span, TermKind::FloatLit(0.0))
            }
            Type::Constructed(TypeName::Bool, _) => {
                self.mk_term(ty.clone(), span, TermKind::BoolLit(false))
            }
            Type::Constructed(TypeName::Unit, _) => self.mk_term(ty.clone(), span, TermKind::UnitLit),
            Type::Constructed(TypeName::Tuple(_), elems)
            | Type::Constructed(TypeName::Record(_), elems) => {
                let blank_terms: Vec<Term> = elems.iter().map(|t| self.build_blank(t, span)).collect();
                self.mk_tuple(blank_terms, ty.clone(), span)
            }
            Type::Constructed(TypeName::Array, args) => {
                // Rank-1 invariant: sum payloads can only hold rank-1
                // arrays with a constant size.
                debug_assert_eq!(
                    args.len(),
                    4,
                    "Array sum payload must have [elem, variant, size, region] args"
                );
                let elem_ty = &args[0];
                let n = match &args[2] {
                    Type::Constructed(TypeName::Size(n), _) => *n,
                    other => panic!(
                        "BUG: array-typed sum payload must have constant size (got {:?}); \
                         the type checker should reject symbolic-size sum payloads upstream",
                        other
                    ),
                };
                let elem_blank = self.build_blank(elem_ty, span);
                let elems: Vec<Term> = std::iter::repeat(elem_blank).take(n).collect();
                self.mk_term(ty.clone(), span, TermKind::ArrayExpr(ArrayExpr::Literal(elems)))
            }
            Type::Constructed(TypeName::Vec, args) => {
                debug_assert_eq!(args.len(), 2, "Vec type must have [elem, size] args");
                let elem_ty = &args[0];
                let n = match &args[1] {
                    Type::Constructed(TypeName::Size(n), _) => *n,
                    other => panic!("BUG: Vec sum payload must have constant size (got {:?})", other),
                };
                let elem_blank = self.build_blank(elem_ty, span);
                let elems: Vec<Term> = std::iter::repeat(elem_blank).take(n).collect();
                self.mk_term(ty.clone(), span, TermKind::VecLit(elems))
            }
            Type::Constructed(TypeName::Arrow, _) => {
                panic!(
                    "BUG: function-typed sum payloads are not supported, but reached \
                     build_blank. The type checker should reject this at the Constructor \
                     or Match site."
                );
            }
            Type::Variable(_)
            | Type::Constructed(TypeName::Size(_), _)
            | Type::Constructed(TypeName::SizeVar(_), _)
            | Type::Constructed(TypeName::SizePlaceholder, _)
            | Type::Constructed(TypeName::AddressPlaceholder, _)
            | Type::Constructed(TypeName::ArrayVariantView, _)
            | Type::Constructed(TypeName::ArrayVariantComposite, _)
            | Type::Constructed(TypeName::ArrayVariantVirtual, _)
            | Type::Constructed(TypeName::Skolem(_), _) => {
                panic!(
                    "BUG: build_blank reached a non-value-level type {:?}; \
                     these shouldn't appear in sum payload slot positions",
                    ty
                );
            }
            _ => panic!("blank for sum-payload type {:?} is not yet implemented", ty),
        }
    }

    // Helper: build binary op application
    pub(super) fn build_binop(
        &mut self,
        op: ast::BinaryOp,
        lhs: Term,
        rhs: Term,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        // Build the binop type: lhs.ty -> rhs.ty -> result_ty
        let binop_ty = Type::Constructed(
            TypeName::Arrow,
            vec![
                lhs.ty.clone(),
                Type::Constructed(TypeName::Arrow, vec![rhs.ty.clone(), result_ty.clone()]),
            ],
        );
        let binop_term = self.mk_term(binop_ty, span, TermKind::BinOp(op));
        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(binop_term),
                args: vec![lhs, rhs],
            },
        )
    }

    // Helper: build unary op application
    fn build_unop(&mut self, op: ast::UnaryOp, arg: Term, result_ty: Type<TypeName>, span: Span) -> Term {
        let unop_ty = Type::Constructed(TypeName::Arrow, vec![arg.ty.clone(), result_ty.clone()]);
        let unop_term = self.mk_term(unop_ty, span, TermKind::UnOp(op));
        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(unop_term),
                args: vec![arg],
            },
        )
    }

    /// Build a flat call against a catalog `BuiltinId`.
    fn build_call_by_id(
        &mut self,
        id: BuiltinId,
        args: &[Term],
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let func_var = VarRef::Builtin { id, overload_idx: 0 };
        if args.is_empty() {
            return self.mk_term(result_ty, span, TermKind::Var(func_var));
        }
        let mut func_ty = result_ty.clone();
        for arg in args.iter().rev() {
            func_ty = Type::Constructed(TypeName::Arrow, vec![arg.ty.clone(), func_ty]);
        }
        let func_term = self.mk_term(func_ty, span, TermKind::Var(func_var));
        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(func_term),
                args: args.to_vec(),
            },
        )
    }

    /// Construct a `TermKind::Tuple` directly.
    pub(super) fn mk_tuple(&mut self, parts: Vec<Term>, result_ty: Type<TypeName>, span: Span) -> Term {
        self.mk_term(result_ty, span, TermKind::Tuple(parts))
    }

    /// Construct a `TermKind::TupleProj` directly.
    pub(super) fn mk_tuple_proj(
        &mut self,
        tuple: Term,
        idx: usize,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        self.mk_term(
            result_ty,
            span,
            TermKind::TupleProj {
                tuple: Box::new(tuple),
                idx,
            },
        )
    }

    /// Construct a `TermKind::Index` directly.
    fn mk_index(&mut self, array: Term, index: Term, result_ty: Type<TypeName>, span: Span) -> Term {
        self.mk_term(
            result_ty,
            span,
            TermKind::Index {
                array: Box::new(array),
                index: Box::new(index),
            },
        )
    }

    /// Construct a `TermKind::VecLit` directly.
    pub(super) fn mk_vec_lit(&mut self, parts: Vec<Term>, result_ty: Type<TypeName>, span: Span) -> Term {
        self.mk_term(result_ty, span, TermKind::VecLit(parts))
    }

    /// Construct an array literal `[a, b, c]` as
    /// `TermKind::ArrayExpr(ArrayExpr::Literal(parts))`.
    pub(super) fn mk_array_lit(&mut self, parts: Vec<Term>, result_ty: Type<TypeName>, span: Span) -> Term {
        self.mk_term(result_ty, span, TermKind::ArrayExpr(ArrayExpr::Literal(parts)))
    }

    fn type_of(header: &ast::TypedHeader) -> Type<TypeName> {
        Self::lower_type(Self::raw_type(header))
    }

    /// Like `lookup_type`, but returns the type *before* sum-type
    /// lowering — used by Constructor and Match transforms that need
    /// to inspect the original `Sum` variants for layout computation.
    pub(super) fn raw_type(header: &ast::TypedHeader) -> Type<TypeName> {
        Self::extract_monotype(&header.ty)
    }

    /// Recursively rewrite `Sum(variants)` types into a flattened tuple
    /// `(tag: u32, ...all_variant_payload_slots)`. Sum types do not
    /// survive past AST→TLC; downstream passes only see tuples.
    pub(super) fn lower_type(ty: Type<TypeName>) -> Type<TypeName> {
        match ty {
            Type::Constructed(TypeName::Sum(variants), _) => {
                let layout = Self::sum_layout(&variants);
                Type::Constructed(TypeName::Tuple(layout.slot_types.len()), layout.slot_types)
            }
            Type::Constructed(TypeName::FragmentOutput, args) if args.len() == 1 => {
                let variants = crate::types::fragment_output_variants(args[0].clone());
                let layout = Self::sum_layout(&variants);
                Type::Constructed(TypeName::Tuple(layout.slot_types.len()), layout.slot_types)
            }
            Type::Constructed(name, args) => {
                let lowered_args: Vec<_> = args.into_iter().map(Self::lower_type).collect();
                Type::Constructed(name, lowered_args)
            }
            Type::Variable(_) => ty,
        }
    }

    /// Compute the flattened-no-sharing layout for a sum type.
    /// Slot 0 is always the u32 tag; slots 1..end are each
    /// constructor's payloads laid out in source order with no
    /// sharing between variants. The dead slots for an inactive
    /// variant are blank-filled at construction.
    pub(super) fn sum_layout(variants: &[(String, Vec<Type<TypeName>>)]) -> SumLayout {
        let tag_ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let mut slot_types = vec![tag_ty];
        let mut constructor_info = LookupMap::new();
        for (i, (name, payload)) in variants.iter().enumerate() {
            constructor_info.insert(name.clone(), (i as u32, slot_types.len()));
            for p in payload {
                slot_types.push(Self::lower_type(p.clone()));
            }
        }
        SumLayout {
            slot_types,
            constructor_info,
        }
    }

    fn extract_monotype(scheme: &polytype::TypeScheme<TypeName>) -> Type<TypeName> {
        match scheme {
            polytype::TypeScheme::Monotype(ty) => ty.clone(),
            polytype::TypeScheme::Polytype { body, .. } => Self::extract_monotype(body),
        }
    }

    pub(super) fn mk_term(&mut self, ty: Type<TypeName>, span: Span, kind: TermKind) -> Term {
        Term::fresh(self.term_ids, ty, span, kind)
    }

    fn mk_i32(&mut self, value: i32, span: Span) -> Term {
        self.mk_term(
            Type::Constructed(TypeName::Int(32), vec![]),
            span,
            TermKind::IntLit(value.to_string()),
        )
    }

    /// Transform an expression as a vector, converting ArrayLiteral to a VecLit term.
    fn transform_as_vector(
        &mut self,
        expr: &ast::Expression<ast::HolesResolvedTree>,
        vec_ty: Type<TypeName>,
    ) -> Term {
        let span = expr.h.span;
        match &expr.kind {
            ast::ExprKind::ArrayLiteral(elements) => {
                let terms: Vec<Term> = elements.iter().map(|e| self.transform_expr(e)).collect();
                self.mk_vec_lit(terms, vec_ty, span)
            }
            _ => self.transform_expr(expr),
        }
    }

    /// Build a `TermKind::VecLit` from already-transformed terms.
    fn build_vec_lit_from_terms(&mut self, terms: &[Term], result_ty: Type<TypeName>, span: Span) -> Term {
        self.mk_vec_lit(terms.to_vec(), result_ty, span)
    }
}
