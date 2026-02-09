//! Structure-of-Arrays (SoA) transform for TLC.
//!
//! Transforms `[n](A,B)` (array of tuples) into `([n]A, [n]B)` (tuple of arrays).
//! This runs after monomorphize (all types concrete) as a TLC-to-TLC pass.
//!
//! After this pass, arrays never contain tuples — they only contain scalars or
//! other arrays. Standalone tuples may still exist and are handled separately.
//!
//! The transform also rewrites standalone `zip(a, b)` calls into `_w_tuple(a, b)`,
//! making zip free.

use super::{ArrayExpr, Def, Lambda, LoopKind, Place, Program, SoacOp, Term, TermIdSource, TermKind};
use crate::SymbolTable;
use crate::ast::{Span, TypeName};
use polytype::Type;

// =============================================================================
// Type rewriting
// =============================================================================

/// Recursively rewrite types so that `Array[Tuple(n)(T1..Tn), v, s]`
/// becomes `Tuple(n)(Array[soa(T1), v, s], ..., Array[soa(Tn), v, s])`.
///
/// Other types are rewritten recursively but their top-level structure is preserved.
pub fn soa_type(ty: &Type<TypeName>) -> Type<TypeName> {
    match ty {
        Type::Constructed(TypeName::Array, args) if args.len() == 3 => {
            let elem = soa_type(&args[0]);
            let variant = args[1].clone();
            let size = args[2].clone();

            // If the element type is a tuple, distribute the array into each component.
            // Recursively apply soa_type to each distributed array, so nested
            // array-of-tuples like [n](int, vec3) get further distributed.
            if let Type::Constructed(TypeName::Tuple(n), ref component_types) = elem {
                let distributed: Vec<Type<TypeName>> = component_types
                    .iter()
                    .map(|ct| {
                        soa_type(&Type::Constructed(
                            TypeName::Array,
                            vec![ct.clone(), variant.clone(), size.clone()],
                        ))
                    })
                    .collect();
                Type::Constructed(TypeName::Tuple(n), distributed)
            } else {
                Type::Constructed(TypeName::Array, vec![elem, variant, size])
            }
        }
        Type::Constructed(TypeName::Tuple(n), args) => {
            let rewritten: Vec<Type<TypeName>> = args.iter().map(soa_type).collect();
            Type::Constructed(TypeName::Tuple(*n), rewritten)
        }
        Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => {
            Type::Constructed(TypeName::Arrow, vec![soa_type(&args[0]), soa_type(&args[1])])
        }
        Type::Constructed(name, args) => {
            // For other constructed types (Vec, Mat, scalars, etc.), recurse into args
            let rewritten: Vec<Type<TypeName>> = args.iter().map(soa_type).collect();
            Type::Constructed(name.clone(), rewritten)
        }
        Type::Variable(_) => ty.clone(),
    }
}

/// Check if a type was an array-of-tuple before SoA transformation.
/// Returns Some(n) where n is the tuple arity if it was.
fn is_array_of_tuple(ty: &Type<TypeName>) -> Option<usize> {
    match ty {
        Type::Constructed(TypeName::Array, args) if args.len() == 3 => match &args[0] {
            Type::Constructed(TypeName::Tuple(n), _) => Some(*n),
            // Check recursively: the element might become a tuple after soa_type
            _ => {
                let elem_soa = soa_type(&args[0]);
                match elem_soa {
                    Type::Constructed(TypeName::Tuple(n), _) => Some(n),
                    _ => None,
                }
            }
        },
        _ => None,
    }
}

/// Extract the tuple component types from an array-of-tuple type.
/// Returns (component_types, variant, size).
fn array_of_tuple_parts(
    ty: &Type<TypeName>,
) -> Option<(Vec<Type<TypeName>>, Type<TypeName>, Type<TypeName>)> {
    match ty {
        Type::Constructed(TypeName::Array, args) if args.len() == 3 => match &args[0] {
            Type::Constructed(TypeName::Tuple(_), components) => {
                Some((components.clone(), args[1].clone(), args[2].clone()))
            }
            _ => None,
        },
        _ => None,
    }
}

// =============================================================================
// Term rewriting
// =============================================================================

/// SoA transformer state.
struct SoaTransformer {
    term_ids: TermIdSource,
    symbols: SymbolTable,
}

impl SoaTransformer {
    fn new(symbols: SymbolTable) -> Self {
        SoaTransformer {
            term_ids: TermIdSource::new(),
            symbols,
        }
    }

    /// Main entry: transform a complete program.
    fn transform_program(mut self, program: Program) -> Program {
        let defs = program.defs.into_iter().map(|def| self.transform_def(def)).collect();
        Program {
            defs,
            uniforms: program.uniforms,
            storage: program.storage,
            symbols: self.symbols,
        }
    }

    fn transform_def(&mut self, def: Def) -> Def {
        let new_ty = soa_type(&def.ty);
        let new_body = self.transform_term(&def.body);
        Def {
            name: def.name,
            ty: new_ty,
            body: new_body,
            meta: def.meta,
            arity: def.arity,
        }
    }

    /// Transform a term, rewriting types via soa_type and transforming
    /// operations that touch array-of-tuple types.
    fn transform_term(&mut self, term: &Term) -> Term {
        let orig_ty = &term.ty;
        let new_ty = soa_type(orig_ty);
        let span = term.span;

        match &term.kind {
            TermKind::Var(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::StringLit(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Extern(_) => self.mk_term(new_ty, span, term.kind.clone()),

            TermKind::Lambda(lam) => {
                let new_lam = self.transform_lambda(lam);
                self.mk_term(new_ty, span, TermKind::Lambda(new_lam))
            }

            TermKind::App { func, arg } => self.transform_app(func, arg, orig_ty, new_ty, span),

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                let new_rhs = self.transform_term(rhs);
                let new_body = self.transform_term(body);
                self.mk_term(
                    new_ty,
                    span,
                    TermKind::Let {
                        name: *name,
                        name_ty: soa_type(name_ty),
                        rhs: Box::new(new_rhs),
                        body: Box::new(new_body),
                    },
                )
            }

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let new_cond = self.transform_term(cond);
                let new_then = self.transform_term(then_branch);
                let new_else = self.transform_term(else_branch);
                self.mk_term(
                    new_ty,
                    span,
                    TermKind::If {
                        cond: Box::new(new_cond),
                        then_branch: Box::new(new_then),
                        else_branch: Box::new(new_else),
                    },
                )
            }

            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                let new_init = self.transform_term(init);
                let new_loop_var_ty = soa_type(loop_var_ty);
                let new_init_bindings = init_bindings
                    .iter()
                    .map(|(sym, ty, expr)| (*sym, soa_type(ty), self.transform_term(expr)))
                    .collect();
                let new_kind = self.transform_loop_kind(kind);
                let new_body = self.transform_term(body);
                self.mk_term(
                    new_ty,
                    span,
                    TermKind::Loop {
                        loop_var: *loop_var,
                        loop_var_ty: new_loop_var_ty,
                        init: Box::new(new_init),
                        init_bindings: new_init_bindings,
                        kind: new_kind,
                        body: Box::new(new_body),
                    },
                )
            }

            TermKind::Soac(soac) => {
                let new_soac = self.transform_soac(soac);
                self.mk_term(new_ty, span, TermKind::Soac(new_soac))
            }

            TermKind::ArrayExpr(ae) => self.transform_array_expr_term(ae, orig_ty, new_ty, span),

            TermKind::Force(inner) => {
                let new_inner = self.transform_term(inner);
                self.mk_term(new_ty, span, TermKind::Force(Box::new(new_inner)))
            }

            TermKind::Pack {
                exists_ty,
                dims,
                value,
            } => {
                let new_value = self.transform_term(value);
                self.mk_term(
                    new_ty,
                    span,
                    TermKind::Pack {
                        exists_ty: soa_type(exists_ty),
                        dims: dims.clone(),
                        value: Box::new(new_value),
                    },
                )
            }

            TermKind::Unpack {
                scrut,
                dim_binders,
                value_binder,
                body,
            } => {
                let new_scrut = self.transform_term(scrut);
                let new_body = self.transform_term(body);
                self.mk_term(
                    new_ty,
                    span,
                    TermKind::Unpack {
                        scrut: Box::new(new_scrut),
                        dim_binders: dim_binders.clone(),
                        value_binder: *value_binder,
                        body: Box::new(new_body),
                    },
                )
            }
        }
    }

    /// Transform a function application. This is where we intercept intrinsics
    /// that operate on array-of-tuple types and rewrite them.
    fn transform_app(
        &mut self,
        func: &Term,
        arg: &Term,
        orig_result_ty: &Type<TypeName>,
        new_result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        // Collect application spine to detect intrinsic patterns
        let (base, args) = collect_app_spine(func, arg);

        match &base.kind {
            TermKind::Var(sym) => {
                let name = self.symbols.get(*sym).cloned().unwrap_or_default();
                match name.as_str() {
                    // _w_index(arr, i) where arr was [n](A,B)
                    "_w_index" if args.len() == 2 => {
                        let arr_orig_ty = &args[0].ty;
                        if let Some(n) = is_array_of_tuple(arr_orig_ty) {
                            let (comp_tys, variant, size) = array_of_tuple_parts(arr_orig_ty).unwrap();
                            let new_arr = self.transform_term(args[0]);
                            let new_idx = self.transform_term(args[1]);
                            return self.rewrite_index_aot(
                                &new_arr, &new_idx, &comp_tys, &variant, &size, n, span,
                            );
                        }
                    }

                    // _w_array_with(arr, i, val) where arr was [n](A,B)
                    "_w_array_with" if args.len() == 3 => {
                        let arr_orig_ty = &args[0].ty;
                        if let Some(n) = is_array_of_tuple(arr_orig_ty) {
                            let (comp_tys, variant, size) = array_of_tuple_parts(arr_orig_ty).unwrap();
                            let new_arr = self.transform_term(args[0]);
                            let new_idx = self.transform_term(args[1]);
                            let new_val = self.transform_term(args[2]);
                            return self.rewrite_array_with_aot(
                                &new_arr, &new_idx, &new_val, &comp_tys, &variant, &size, n, span,
                            );
                        }
                    }

                    // _w_array_lit(e1, e2, ...) where result was [n](A,B)
                    "_w_array_lit" if !args.is_empty() => {
                        if is_array_of_tuple(orig_result_ty).is_some() {
                            let (comp_tys, variant, size) = array_of_tuple_parts(orig_result_ty).unwrap();
                            let new_elems: Vec<Term> =
                                args.iter().map(|a| self.transform_term(a)).collect();
                            return self
                                .rewrite_array_lit_aot(&new_elems, &comp_tys, &variant, &size, span);
                        }
                    }

                    // _w_intrinsic_uninit() where result was [n](A,B)
                    "_w_intrinsic_uninit" if args.is_empty() => {
                        if is_array_of_tuple(orig_result_ty).is_some() {
                            let soa_ty = soa_type(orig_result_ty);
                            return self.rewrite_uninit_aot(&soa_ty, *sym, span);
                        }
                    }

                    // _w_intrinsic_length(arr) where arr was [n](A,B)
                    "_w_intrinsic_length" if args.len() == 1 => {
                        let arr_orig_ty = &args[0].ty;
                        if is_array_of_tuple(arr_orig_ty).is_some() {
                            let new_arr = self.transform_term(args[0]);
                            return self.rewrite_length_aot(&new_arr, *sym, new_result_ty, span);
                        }
                    }

                    _ => {}
                }
            }
            _ => {}
        }

        // Default: recursively transform all parts
        let new_func = self.transform_term(func);
        let new_arg = self.transform_term(arg);
        self.mk_term(
            new_result_ty,
            span,
            TermKind::App {
                func: Box::new(new_func),
                arg: Box::new(new_arg),
            },
        )
    }

    // =========================================================================
    // Array-of-Tuple rewrite helpers
    // =========================================================================

    /// `_w_index(arr, i)` where arr was `[n](A,B)`, now `([n]A, [n]B)`:
    /// → `_w_tuple(_w_index(proj(arr,0), i), _w_index(proj(arr,1), i))`
    fn rewrite_index_aot(
        &mut self,
        arr: &Term,
        idx: &Term,
        comp_tys: &[Type<TypeName>],
        variant: &Type<TypeName>,
        size: &Type<TypeName>,
        n: usize,
        span: Span,
    ) -> Term {
        let components: Vec<Term> = (0..n)
            .map(|i| {
                // Recursively soa_type the constructed array so nested array-of-tuples
                // like [8](int, vec3) are further distributed to ([8]int, [8]vec3).
                let comp_arr_ty = soa_type(&Type::Constructed(
                    TypeName::Array,
                    vec![soa_type(&comp_tys[i]), variant.clone(), size.clone()],
                ));
                let proj = self.mk_tuple_proj(arr.clone(), i, comp_arr_ty, span);
                let elem_ty = soa_type(&comp_tys[i]);
                self.mk_index(proj, idx.clone(), elem_ty, span)
            })
            .collect();

        let result_ty =
            Type::Constructed(TypeName::Tuple(n), comp_tys.iter().map(|t| soa_type(t)).collect());
        self.mk_tuple(components, result_ty, span)
    }

    /// `_w_array_with(arr, i, val)` where arr was `[n](A,B)`:
    /// → `_w_tuple(array_with(proj(arr,0), i, proj(val,0)), ...)`
    fn rewrite_array_with_aot(
        &mut self,
        arr: &Term,
        idx: &Term,
        val: &Term,
        comp_tys: &[Type<TypeName>],
        variant: &Type<TypeName>,
        size: &Type<TypeName>,
        n: usize,
        span: Span,
    ) -> Term {
        let components: Vec<Term> = (0..n)
            .map(|i| {
                let soa_comp_ty = soa_type(&comp_tys[i]);
                let comp_arr_ty = soa_type(&Type::Constructed(
                    TypeName::Array,
                    vec![soa_comp_ty.clone(), variant.clone(), size.clone()],
                ));
                let arr_proj = self.mk_tuple_proj(arr.clone(), i, comp_arr_ty.clone(), span);
                let val_proj = self.mk_tuple_proj(val.clone(), i, soa_comp_ty, span);
                self.mk_array_with(arr_proj, idx.clone(), val_proj, comp_arr_ty, span)
            })
            .collect();

        let result_ty = Type::Constructed(
            TypeName::Tuple(n),
            (0..n)
                .map(|i| {
                    soa_type(&Type::Constructed(
                        TypeName::Array,
                        vec![soa_type(&comp_tys[i]), variant.clone(), size.clone()],
                    ))
                })
                .collect(),
        );
        self.mk_tuple(components, result_ty, span)
    }

    /// `_w_array_lit(e1, e2, ...)` where result was `[n](A,B)`:
    /// → `_w_tuple(_w_array_lit(proj(e1,0), proj(e2,0), ...), ...)`
    fn rewrite_array_lit_aot(
        &mut self,
        elems: &[Term],
        comp_tys: &[Type<TypeName>],
        variant: &Type<TypeName>,
        size: &Type<TypeName>,
        span: Span,
    ) -> Term {
        let n = comp_tys.len();
        let components: Vec<Term> = (0..n)
            .map(|i| {
                let soa_comp_ty = soa_type(&comp_tys[i]);
                let projected_elems: Vec<Term> = elems
                    .iter()
                    .map(|e| self.mk_tuple_proj(e.clone(), i, soa_comp_ty.clone(), span))
                    .collect();
                let arr_ty = soa_type(&Type::Constructed(
                    TypeName::Array,
                    vec![soa_comp_ty, variant.clone(), size.clone()],
                ));
                self.mk_array_lit(projected_elems, arr_ty, span)
            })
            .collect();

        let result_ty = Type::Constructed(
            TypeName::Tuple(n),
            (0..n)
                .map(|i| {
                    soa_type(&Type::Constructed(
                        TypeName::Array,
                        vec![soa_type(&comp_tys[i]), variant.clone(), size.clone()],
                    ))
                })
                .collect(),
        );
        self.mk_tuple(components, result_ty, span)
    }

    /// `_w_intrinsic_uninit()` where result was `[n](A,B)`:
    /// → `_w_tuple(_w_intrinsic_uninit(), _w_intrinsic_uninit())`
    fn rewrite_uninit_aot(
        &mut self,
        soa_ty: &Type<TypeName>,
        uninit_sym: crate::SymbolId,
        span: Span,
    ) -> Term {
        match soa_ty {
            Type::Constructed(TypeName::Tuple(_), comp_tys) => {
                let components: Vec<Term> = comp_tys
                    .iter()
                    .map(|ct| {
                        // Each component is a call to _w_intrinsic_uninit with the component type
                        self.mk_term(ct.clone(), span, TermKind::Var(uninit_sym))
                    })
                    .collect();
                self.mk_tuple(components, soa_ty.clone(), span)
            }
            _ => self.mk_term(soa_ty.clone(), span, TermKind::Var(uninit_sym)),
        }
    }

    /// `_w_intrinsic_length(arr)` where arr was `[n](A,B)`:
    /// → `_w_intrinsic_length(proj(arr, 0))`
    fn rewrite_length_aot(
        &mut self,
        arr: &Term,
        length_sym: crate::SymbolId,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        // arr is now a tuple of arrays. Project the first component and take its length.
        let first_comp_ty = match &arr.ty {
            Type::Constructed(TypeName::Tuple(_), comp_tys) => comp_tys[0].clone(),
            _ => arr.ty.clone(),
        };
        let first_arr = self.mk_tuple_proj(arr.clone(), 0, first_comp_ty, span);

        let func_ty = Type::Constructed(TypeName::Arrow, vec![first_arr.ty.clone(), result_ty.clone()]);
        let func = self.mk_term(func_ty, span, TermKind::Var(length_sym));
        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(func),
                arg: Box::new(first_arr),
            },
        )
    }

    // =========================================================================
    // SOAC rewriting
    // =========================================================================

    fn transform_soac(&mut self, soac: &SoacOp) -> SoacOp {
        match soac {
            SoacOp::Map { lam, inputs } => {
                let new_lam = self.transform_lambda(lam);
                let new_inputs: Vec<ArrayExpr> =
                    inputs.iter().map(|ae| self.transform_array_expr(ae)).collect();
                SoacOp::Map {
                    lam: new_lam,
                    inputs: new_inputs,
                }
            }
            SoacOp::Reduce { op, ne, input, props } => {
                let new_op = self.transform_lambda(op);
                let new_ne = self.transform_term(ne);
                let new_input = self.transform_array_expr(input);
                SoacOp::Reduce {
                    op: new_op,
                    ne: Box::new(new_ne),
                    input: new_input,
                    props: props.clone(),
                }
            }
            SoacOp::Scan { op, ne, input } => {
                let new_op = self.transform_lambda(op);
                let new_ne = self.transform_term(ne);
                let new_input = self.transform_array_expr(input);
                SoacOp::Scan {
                    op: new_op,
                    ne: Box::new(new_ne),
                    input: new_input,
                }
            }
            SoacOp::Filter { pred, input } => {
                let new_pred = self.transform_lambda(pred);
                let new_input = self.transform_array_expr(input);
                SoacOp::Filter {
                    pred: new_pred,
                    input: new_input,
                }
            }
            SoacOp::Scatter {
                dest,
                indices,
                values,
            } => {
                let new_dest = self.transform_place(dest);
                let new_indices = self.transform_array_expr(indices);
                let new_values = self.transform_array_expr(values);
                SoacOp::Scatter {
                    dest: new_dest,
                    indices: new_indices,
                    values: new_values,
                }
            }
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
                props,
            } => {
                let new_dest = self.transform_place(dest);
                let new_op = self.transform_lambda(op);
                let new_ne = self.transform_term(ne);
                let new_indices = self.transform_array_expr(indices);
                let new_values = self.transform_array_expr(values);
                SoacOp::ReduceByIndex {
                    dest: new_dest,
                    op: new_op,
                    ne: Box::new(new_ne),
                    indices: new_indices,
                    values: new_values,
                    props: props.clone(),
                }
            }
        }
    }

    fn transform_place(&mut self, place: &Place) -> Place {
        match place {
            Place::BufferSlice {
                base,
                offset,
                shape,
                elem_ty,
            } => Place::BufferSlice {
                base: Box::new(self.transform_term(base)),
                offset: Box::new(self.transform_term(offset)),
                shape: shape.clone(),
                elem_ty: soa_type(elem_ty),
            },
            Place::LocalArray { id, shape, elem_ty } => Place::LocalArray {
                id: *id,
                shape: shape.clone(),
                elem_ty: soa_type(elem_ty),
            },
        }
    }

    // =========================================================================
    // ArrayExpr rewriting
    // =========================================================================

    fn transform_array_expr(&mut self, ae: &ArrayExpr) -> ArrayExpr {
        match ae {
            ArrayExpr::Ref(term) => {
                let new_term = self.transform_term(term);
                ArrayExpr::Ref(Box::new(new_term))
            }
            ArrayExpr::Zip(exprs) => {
                // Zip is preserved as-is in ArrayExpr (it gets absorbed by Map).
                // If standalone, the surrounding term will be an ArrayExpr(Zip(...))
                // and we'll handle that in transform_array_expr_term.
                let new_exprs: Vec<ArrayExpr> =
                    exprs.iter().map(|e| self.transform_array_expr(e)).collect();
                ArrayExpr::Zip(new_exprs)
            }
            ArrayExpr::Soac(op) => {
                let new_op = self.transform_soac(op);
                ArrayExpr::Soac(Box::new(new_op))
            }
            ArrayExpr::Generate {
                shape,
                index_fn,
                elem_ty,
            } => ArrayExpr::Generate {
                shape: shape.clone(),
                index_fn: self.transform_lambda(index_fn),
                elem_ty: soa_type(elem_ty),
            },
            ArrayExpr::Literal(terms) => {
                let new_terms: Vec<Term> = terms.iter().map(|t| self.transform_term(t)).collect();
                ArrayExpr::Literal(new_terms)
            }
            ArrayExpr::Range { start, len } => ArrayExpr::Range {
                start: Box::new(self.transform_term(start)),
                len: Box::new(self.transform_term(len)),
            },
        }
    }

    /// Transform an ArrayExpr appearing as a standalone term.
    /// This is where standalone Zip gets rewritten to _w_tuple.
    fn transform_array_expr_term(
        &mut self,
        ae: &ArrayExpr,
        _orig_ty: &Type<TypeName>,
        new_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        match ae {
            ArrayExpr::Zip(exprs) => {
                // Standalone zip: `zip(a, b)` where result was `[n](A,B)`.
                // After SoA, this is just `_w_tuple(a, b)` — zip is free!
                let components: Vec<Term> = exprs
                    .iter()
                    .map(|e| match e {
                        ArrayExpr::Ref(t) => self.transform_term(t),
                        _ => {
                            // Nested non-Ref array exprs — transform them
                            let inner_ty = self.array_expr_type(e);
                            let new_inner_ty = soa_type(&inner_ty);
                            self.transform_array_expr_term(e, &inner_ty, new_inner_ty, span)
                        }
                    })
                    .collect();
                self.mk_tuple(components, new_ty, span)
            }
            _ => {
                let new_ae = self.transform_array_expr(ae);
                self.mk_term(new_ty, span, TermKind::ArrayExpr(new_ae))
            }
        }
    }

    fn array_expr_type(&self, ae: &ArrayExpr) -> Type<TypeName> {
        match ae {
            ArrayExpr::Ref(t) => t.ty.clone(),
            ArrayExpr::Zip(_) => Type::Constructed(TypeName::Unit, vec![]),
            ArrayExpr::Soac(_) => Type::Constructed(TypeName::Unit, vec![]),
            ArrayExpr::Generate { elem_ty, .. } => elem_ty.clone(),
            ArrayExpr::Literal(terms) => {
                if let Some(first) = terms.first() {
                    first.ty.clone()
                } else {
                    Type::Constructed(TypeName::Unit, vec![])
                }
            }
            ArrayExpr::Range { start, .. } => start.ty.clone(),
        }
    }

    // =========================================================================
    // Lambda rewriting
    // =========================================================================

    fn transform_lambda(&mut self, lam: &Lambda) -> Lambda {
        let new_params: Vec<(crate::SymbolId, Type<TypeName>)> =
            lam.params.iter().map(|(sym, ty)| (*sym, soa_type(ty))).collect();
        let new_body = self.transform_term(&lam.body);
        let new_captures: Vec<(crate::SymbolId, Type<TypeName>, Term)> = lam
            .captures
            .iter()
            .map(|(sym, ty, term)| (*sym, soa_type(ty), self.transform_term(term)))
            .collect();
        Lambda {
            params: new_params,
            body: Box::new(new_body),
            ret_ty: soa_type(&lam.ret_ty),
            captures: new_captures,
        }
    }

    fn transform_loop_kind(&mut self, kind: &LoopKind) -> LoopKind {
        match kind {
            LoopKind::For { var, var_ty, iter } => LoopKind::For {
                var: *var,
                var_ty: soa_type(var_ty),
                iter: Box::new(self.transform_term(iter)),
            },
            LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                var: *var,
                var_ty: soa_type(var_ty),
                bound: Box::new(self.transform_term(bound)),
            },
            LoopKind::While { cond } => LoopKind::While {
                cond: Box::new(self.transform_term(cond)),
            },
        }
    }

    // =========================================================================
    // Term construction helpers
    // =========================================================================

    fn mk_term(&mut self, ty: Type<TypeName>, span: Span, kind: TermKind) -> Term {
        Term {
            id: self.term_ids.next_id(),
            ty,
            span,
            kind,
        }
    }

    /// Build `_w_tuple(c0, c1, ...)`.
    fn mk_tuple(&mut self, components: Vec<Term>, result_ty: Type<TypeName>, span: Span) -> Term {
        let tuple_sym = self.resolve_or_alloc("_w_tuple");
        if components.is_empty() {
            return self.mk_term(result_ty, span, TermKind::Var(tuple_sym));
        }

        // Build curried application: _w_tuple(c0)(c1)...
        let n = components.len();

        // Compute intermediate arrow types working backwards
        let mut intermediate_tys = vec![result_ty.clone()];
        for comp in components.iter().rev().skip(1) {
            let prev = intermediate_tys.last().unwrap().clone();
            intermediate_tys.push(Type::Constructed(TypeName::Arrow, vec![comp.ty.clone(), prev]));
        }
        intermediate_tys.reverse();

        let func_ty = Type::Constructed(
            TypeName::Arrow,
            vec![components[0].ty.clone(), intermediate_tys[0].clone()],
        );
        let func = self.mk_term(func_ty, span, TermKind::Var(tuple_sym));

        let mut result = self.mk_term(
            intermediate_tys[0].clone(),
            span,
            TermKind::App {
                func: Box::new(func),
                arg: Box::new(components[0].clone()),
            },
        );

        for i in 1..n {
            let app_ty = if i < n - 1 { intermediate_tys[i].clone() } else { result_ty.clone() };
            result = self.mk_term(
                app_ty,
                span,
                TermKind::App {
                    func: Box::new(result),
                    arg: Box::new(components[i].clone()),
                },
            );
        }

        result
    }

    /// Build `_w_tuple_proj(term, index)`.
    fn mk_tuple_proj(&mut self, term: Term, index: usize, result_ty: Type<TypeName>, span: Span) -> Term {
        let proj_sym = self.resolve_or_alloc("_w_tuple_proj");
        let idx_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let idx_term = self.mk_term(idx_ty.clone(), span, TermKind::IntLit(index.to_string()));

        // _w_tuple_proj : term.ty -> i32 -> result_ty
        let inner_ty = Type::Constructed(TypeName::Arrow, vec![idx_ty, result_ty.clone()]);
        let func_ty = Type::Constructed(TypeName::Arrow, vec![term.ty.clone(), inner_ty.clone()]);
        let func = self.mk_term(func_ty, span, TermKind::Var(proj_sym));

        let app1 = self.mk_term(
            inner_ty,
            span,
            TermKind::App {
                func: Box::new(func),
                arg: Box::new(term),
            },
        );

        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(app1),
                arg: Box::new(idx_term),
            },
        )
    }

    /// Build `_w_index(arr, idx)`.
    fn mk_index(&mut self, arr: Term, idx: Term, result_ty: Type<TypeName>, span: Span) -> Term {
        let index_sym = self.resolve_or_alloc("_w_index");
        let inner_ty = Type::Constructed(TypeName::Arrow, vec![idx.ty.clone(), result_ty.clone()]);
        let func_ty = Type::Constructed(TypeName::Arrow, vec![arr.ty.clone(), inner_ty.clone()]);
        let func = self.mk_term(func_ty, span, TermKind::Var(index_sym));

        let app1 = self.mk_term(
            inner_ty,
            span,
            TermKind::App {
                func: Box::new(func),
                arg: Box::new(arr),
            },
        );

        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(app1),
                arg: Box::new(idx),
            },
        )
    }

    /// Build `_w_array_with(arr, idx, val)`.
    fn mk_array_with(
        &mut self,
        arr: Term,
        idx: Term,
        val: Term,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let aw_sym = self.resolve_or_alloc("_w_array_with");
        let t3 = Type::Constructed(TypeName::Arrow, vec![val.ty.clone(), result_ty.clone()]);
        let t2 = Type::Constructed(TypeName::Arrow, vec![idx.ty.clone(), t3.clone()]);
        let t1 = Type::Constructed(TypeName::Arrow, vec![arr.ty.clone(), t2.clone()]);
        let func = self.mk_term(t1, span, TermKind::Var(aw_sym));

        let app1 = self.mk_term(
            t2,
            span,
            TermKind::App {
                func: Box::new(func),
                arg: Box::new(arr),
            },
        );
        let app2 = self.mk_term(
            t3,
            span,
            TermKind::App {
                func: Box::new(app1),
                arg: Box::new(idx),
            },
        );
        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(app2),
                arg: Box::new(val),
            },
        )
    }

    /// Build `_w_array_lit(e0, e1, ...)`.
    fn mk_array_lit(&mut self, elems: Vec<Term>, result_ty: Type<TypeName>, span: Span) -> Term {
        let al_sym = self.resolve_or_alloc("_w_array_lit");
        if elems.is_empty() {
            return self.mk_term(result_ty, span, TermKind::Var(al_sym));
        }

        let n = elems.len();
        let mut intermediate_tys = vec![result_ty.clone()];
        for elem in elems.iter().rev().skip(1) {
            let prev = intermediate_tys.last().unwrap().clone();
            intermediate_tys.push(Type::Constructed(TypeName::Arrow, vec![elem.ty.clone(), prev]));
        }
        intermediate_tys.reverse();

        let func_ty = Type::Constructed(
            TypeName::Arrow,
            vec![elems[0].ty.clone(), intermediate_tys[0].clone()],
        );
        let func = self.mk_term(func_ty, span, TermKind::Var(al_sym));

        let mut result = self.mk_term(
            intermediate_tys[0].clone(),
            span,
            TermKind::App {
                func: Box::new(func),
                arg: Box::new(elems[0].clone()),
            },
        );

        for i in 1..n {
            let app_ty = if i < n - 1 { intermediate_tys[i].clone() } else { result_ty.clone() };
            result = self.mk_term(
                app_ty,
                span,
                TermKind::App {
                    func: Box::new(result),
                    arg: Box::new(elems[i].clone()),
                },
            );
        }

        result
    }

    /// Resolve a symbol name or allocate a new one.
    fn resolve_or_alloc(&mut self, name: &str) -> crate::SymbolId {
        // Search existing symbols
        for (id, existing_name) in self.symbols.iter() {
            if existing_name == name {
                return *id;
            }
        }
        self.symbols.alloc(name.to_string())
    }
}

/// Collect the spine of a curried application chain.
/// `f(a)(b)(c)` → `(f, [a, b, c])`.
fn collect_app_spine<'a>(func: &'a Term, arg: &'a Term) -> (&'a Term, Vec<&'a Term>) {
    let mut args = vec![arg];
    let mut current = func;
    loop {
        match &current.kind {
            TermKind::App {
                func: inner_func,
                arg: inner_arg,
            } => {
                args.push(inner_arg.as_ref());
                current = inner_func.as_ref();
            }
            _ => {
                args.reverse();
                return (current, args);
            }
        }
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Run the SoA transform on a TLC program.
/// This rewrites `[n](A,B)` types to `([n]A, [n]B)` and adjusts all operations
/// that touch array-of-tuple types.
pub fn soa_transform(program: Program) -> Program {
    let transformer = SoaTransformer::new(program.symbols.clone());
    transformer.transform_program(program)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn i32_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::Int(32), vec![])
    }

    fn f32_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::Float(32), vec![])
    }

    fn size_ty(n: usize) -> Type<TypeName> {
        Type::Constructed(TypeName::Size(n), vec![])
    }

    fn composite_variant() -> Type<TypeName> {
        Type::Constructed(TypeName::ArrayVariantComposite, vec![])
    }

    fn array_ty(elem: Type<TypeName>, size: usize) -> Type<TypeName> {
        Type::Constructed(TypeName::Array, vec![elem, composite_variant(), size_ty(size)])
    }

    fn tuple_ty(args: Vec<Type<TypeName>>) -> Type<TypeName> {
        Type::Constructed(TypeName::Tuple(args.len()), args)
    }

    #[test]
    fn test_soa_type_scalar() {
        assert_eq!(soa_type(&i32_ty()), i32_ty());
        assert_eq!(soa_type(&f32_ty()), f32_ty());
    }

    #[test]
    fn test_soa_type_plain_array() {
        let arr = array_ty(f32_ty(), 4);
        assert_eq!(soa_type(&arr), arr);
    }

    #[test]
    fn test_soa_type_array_of_tuple() {
        // [4](i32, f32) → ([4]i32, [4]f32)
        let arr = array_ty(tuple_ty(vec![i32_ty(), f32_ty()]), 4);
        let expected = tuple_ty(vec![array_ty(i32_ty(), 4), array_ty(f32_ty(), 4)]);
        assert_eq!(soa_type(&arr), expected);
    }

    #[test]
    fn test_soa_type_nested_array() {
        // [n][m](A,B) → ([n][m]A, [n][m]B)
        let inner = array_ty(tuple_ty(vec![i32_ty(), f32_ty()]), 3);
        let outer = array_ty(inner, 5);
        let result = soa_type(&outer);

        // First soa_type on outer: elem is [3](i32,f32), which soa_type transforms to ([3]i32, [3]f32)
        // That's a tuple, so the outer array distributes:
        // ([5]([3]i32), [5]([3]f32)) — but wait, [5]([3]i32) is [5][3]i32 which is fine (no tuple elem)
        // Actually: soa_type([5][3](i32,f32)):
        //   elem = [3](i32,f32), soa_type → ([3]i32, [3]f32), which is Tuple
        //   So outer distributes: ([5]([3]i32), [5]([3]f32))
        //   But [5]([3]i32) has elem = ([3]i32) which is NOT a tuple, so it stays.
        // Actually ([3]i32) is Array, not Tuple. So the elem of the outer after soa
        // is ([3]i32, [3]f32) which IS a tuple. So we get:
        // (Array[([3]i32), composite, 5], Array[([3]f32), composite, 5])
        // = ([5][3]i32 via nesting... no, it's [5]([3]i32))
        // Hmm, [5](something) where something = Tuple. So distribute:
        // The soa_type of the inner element is ([3]i32, [3]f32).
        // Distributing array over this tuple: ([5]([3]i32), [5]([3]f32))
        // These are arrays whose elements are arrays (not tuples), so no further transformation.
        let expected = tuple_ty(vec![
            array_ty(array_ty(i32_ty(), 3), 5),
            array_ty(array_ty(f32_ty(), 3), 5),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_soa_type_standalone_tuple() {
        // (A, [n](B,C)) → (A, ([n]B, [n]C))
        let inner = array_ty(tuple_ty(vec![i32_ty(), f32_ty()]), 4);
        let standalone = tuple_ty(vec![f32_ty(), inner]);
        let result = soa_type(&standalone);
        let expected = tuple_ty(vec![
            f32_ty(),
            tuple_ty(vec![array_ty(i32_ty(), 4), array_ty(f32_ty(), 4)]),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_soa_type_arrow() {
        // ([4](i32,f32)) -> i32  →  ([4]i32, [4]f32) -> i32
        let param = array_ty(tuple_ty(vec![i32_ty(), f32_ty()]), 4);
        let arrow = Type::Constructed(TypeName::Arrow, vec![param, i32_ty()]);
        let result = soa_type(&arrow);
        let expected = Type::Constructed(
            TypeName::Arrow,
            vec![
                tuple_ty(vec![array_ty(i32_ty(), 4), array_ty(f32_ty(), 4)]),
                i32_ty(),
            ],
        );
        assert_eq!(result, expected);
    }
}
