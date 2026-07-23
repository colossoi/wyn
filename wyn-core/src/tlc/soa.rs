//! Structure-of-Arrays (SoA) transform and SOAC normalization for TLC.
//!
//! This pass does two things in a single recursive walk:
//!
//! 1. **SoA transform**: Rewrites `[n](A,B)` (array of tuples) into `([n]A, [n]B)`
//!    (tuple of arrays). After this, arrays never contain tuples — they only contain
//!    scalars or other arrays. Operations on array-of-tuple types (index, array_with,
//!    array_lit, uninit, length) are rewritten to operate on the distributed components.
//!
//! 2. **SOAC normalization**: When a Map has multiple inputs (from an absorbed zip) but
//!    the lambda takes a single tuple parameter, rewrites the lambda to take N separate
//!    parameters. Also converts standalone `zip(a, b)` into `_w_tuple(a, b)`.
//!
//! Runs before fusion and defunctionalization.

use super::{
    ArrayExpr, Def, Lambda, LoopKind, Place, Program, SoacOp, Term, TermIdSource, TermKind, VarRef,
};
use crate::ast::{Span, TypeName};
use crate::builtins::{by_id, catalog};
use crate::types::TypeExt;
use crate::SymbolTable;
use polytype::Type;

// =============================================================================
// Type rewriting
// =============================================================================

/// Recursively rewrite types so that `Array[Tuple(n)(T1..Tn), v, s]`
/// becomes `Tuple(n)(Array[soa(T1), v, s], …, Array[soa(Tn), v, s])`.
///
/// Other types are rewritten recursively but their top-level structure is preserved.
pub fn soa_type(ty: &Type<TypeName>) -> Type<TypeName> {
    match ty {
        _ if ty.is_array() => {
            let elem = soa_type(ty.elem_type().expect("Array has elem"));
            let size = ty.array_size().expect("Array has size").clone();
            let region = ty.array_buffer().expect("Array has region").clone();
            let variant = match ty.array_variant().expect("Array has variant") {
                // Resolve unresolved variant variables to Composite when distributing.
                Type::Variable(_) => Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
                v => v.clone(),
            };

            // If the element type is a tuple, distribute the array into each component.
            // Recursively apply soa_type to each distributed array, so nested
            // array-of-tuples like [n](int, vec3) get further distributed.
            if let Type::Constructed(TypeName::Tuple(n), ref component_types) = elem {
                let distributed: Vec<Type<TypeName>> = component_types
                    .iter()
                    .map(|ct| {
                        soa_type(&Type::Constructed(
                            TypeName::Array,
                            vec![ct.clone(), variant.clone(), size.clone(), region.clone()],
                        ))
                    })
                    .collect();
                Type::Constructed(TypeName::Tuple(n), distributed)
            } else {
                Type::Constructed(TypeName::Array, vec![elem, variant, size, region])
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

/// The BROAD predicate: will SoA expand this array's element into a tuple (so
/// the array as a whole becomes a tuple-of-arrays)? True when the element is
/// directly a tuple, OR becomes one after `soa_type` (a nested array-of-tuple
/// element). Returns the resulting tuple arity.
///
/// Use this ONLY where you need the yes/no shape question and then call
/// `soa_type` yourself (e.g. `uninit`/`length` rewrites). Do NOT use it to guard
/// `array_of_tuple_parts`: this predicate is strictly broader than that
/// extractor's domain (it also accepts the nested case, for which no flat tuple
/// components exist), so guarding with it and then unwrapping the parts panics.
/// Guard with `array_of_tuple_parts` directly instead.
fn soa_yields_tuple_arrays(ty: &Type<TypeName>) -> Option<usize> {
    let elem = ty.elem_type()?;
    if !ty.is_array() {
        return None;
    }
    match elem {
        Type::Constructed(TypeName::Tuple(n), _) => Some(*n),
        // The element might become a tuple after soa_type (nested array-of-tuple).
        _ => match soa_type(elem) {
            Type::Constructed(TypeName::Tuple(n), _) => Some(n),
            _ => None,
        },
    }
}

/// Extract the parts of an array whose element is DIRECTLY a tuple:
/// `(arity, component_types, variant, size, region)`. Returns `None` for any
/// other type — including the nested array-of-tuple case that
/// `soa_yields_tuple_arrays` accepts but for which no flat components exist.
///
/// This IS its own guard: `if let Some(parts) = array_of_tuple_parts(ty)`. The
/// invariant "the thing I checked has extractable parts" holds by construction
/// because the check and the extraction are the same function.
fn array_of_tuple_parts(
    ty: &Type<TypeName>,
) -> Option<(
    usize,
    Vec<Type<TypeName>>,
    Type<TypeName>,
    Type<TypeName>,
    Type<TypeName>,
)> {
    if !ty.is_array() {
        return None;
    }
    match ty.elem_type()? {
        Type::Constructed(TypeName::Tuple(n), components) => {
            let variant = ty.array_variant().expect("Array has variant").clone();
            let size = ty.array_size().expect("Array has size").clone();
            let region = ty.array_buffer().expect("Array has region").clone();
            Some((*n, components.clone(), variant, size, region))
        }
        _ => None,
    }
}

// =============================================================================
// SOAC normalization helpers (standalone, don't need self)
// =============================================================================

/// Count how many flat (non-tuple) types a type expands to.
fn flat_type_count(ty: &Type<TypeName>) -> usize {
    match ty {
        Type::Constructed(TypeName::Tuple(_), children) if !children.is_empty() => {
            children.iter().map(flat_type_count).sum()
        }
        _ => 1,
    }
}

/// Recursively flatten nested tuple types: ((A, B), C) -> [A, B, C]
fn flatten_tuple_types(types: &[Type<TypeName>]) -> Vec<Type<TypeName>> {
    let mut flat = Vec::new();
    for ty in types {
        match ty {
            Type::Constructed(TypeName::Tuple(_), children) if !children.is_empty() => {
                flat.extend(flatten_tuple_types(children));
            }
            _ => flat.push(ty.clone()),
        }
    }
    flat
}

fn has_type_variables(ty: &Type<TypeName>) -> bool {
    match ty {
        Type::Variable(_) => true,
        Type::Constructed(_, args) => args.iter().any(has_type_variables),
    }
}

// =============================================================================
// Term rewriting
// =============================================================================

/// SoA transformer state.
struct SoaTransformer<'a, 'ids> {
    term_ids: &'ids mut TermIdSource,
    symbols: &'a mut SymbolTable,
}

impl<'a, 'ids> SoaTransformer<'a, 'ids> {
    fn new(symbols: &'a mut SymbolTable, term_ids: &'ids mut TermIdSource) -> Self {
        SoaTransformer { term_ids, symbols }
    }

    fn transform_def(&mut self, def: &mut Def) {
        def.ty = soa_type(&def.ty);
        def.body = self.transform_term(&def.body);
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
            | TermKind::UnitLit
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Extern(_) => self.mk_term(new_ty, span, term.kind.clone()),

            TermKind::Coerce { inner, target_ty } => {
                let new_inner = self.transform_term(inner);
                self.mk_term(
                    new_ty,
                    span,
                    TermKind::Coerce {
                        inner: Box::new(new_inner),
                        target_ty: soa_type(target_ty),
                    },
                )
            }

            TermKind::Lambda(lam) => {
                let new_lam = self.transform_lambda(lam);
                self.mk_term(new_ty, span, TermKind::Lambda(new_lam))
            }

            TermKind::App { func, args } => self.transform_app(func, args, orig_ty, new_ty, span),

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
                // Try to normalize Map+Zip after SoA transform
                let new_soac = match new_soac {
                    SoacOp::Map {
                        lam,
                        inputs,
                        destination,
                    } => self.try_normalize_map(lam, inputs, destination),
                    other => other,
                };
                self.mk_term(new_ty, span, TermKind::Soac(new_soac))
            }

            TermKind::ArrayExpr(ae) => self.transform_array_expr_term(ae, orig_ty, new_ty, span),

            TermKind::Tuple(parts) => {
                let new_parts: Vec<Term> = parts.iter().map(|p| self.transform_term(p)).collect();
                self.mk_term(new_ty, span, TermKind::Tuple(new_parts))
            }
            TermKind::TupleProj { tuple, idx } => {
                let idx = *idx;
                let new_tuple = self.transform_term(tuple);
                self.mk_term(
                    new_ty,
                    span,
                    TermKind::TupleProj {
                        tuple: Box::new(new_tuple),
                        idx,
                    },
                )
            }
            TermKind::Index { array, index } => {
                // Array-of-tuple index: distribute over per-component arrays.
                let arr_orig_ty = &array.ty;
                if let Some((n, comp_tys, variant, size, region)) = array_of_tuple_parts(arr_orig_ty) {
                    let new_arr = self.transform_term(array);
                    let new_idx = self.transform_term(index);
                    return self.rewrite_index_aot(
                        &new_arr, &new_idx, &comp_tys, &variant, &size, &region, n, span,
                    );
                }
                let new_array = self.transform_term(array);
                let new_index = self.transform_term(index);
                self.mk_term(
                    new_ty,
                    span,
                    TermKind::Index {
                        array: Box::new(new_array),
                        index: Box::new(new_index),
                    },
                )
            }
            TermKind::VecLit(parts) => {
                let new_parts: Vec<Term> = parts.iter().map(|p| self.transform_term(p)).collect();
                self.mk_term(new_ty, span, TermKind::VecLit(new_parts))
            }
        }
    }

    /// Transform a function application. This is where we intercept intrinsics
    /// that operate on array-of-tuple types and rewrite them.
    fn transform_app(
        &mut self,
        func: &Term,
        args: &[Term],
        orig_result_ty: &Type<TypeName>,
        new_result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let known = catalog().known();
        if let Some(id) = crate::tlc::var_term_builtin_id(func, &self.symbols) {
            // array_with(arr, i, val) where arr was [n](A,B)
            if (id == known.array_with || id == known.array_with_in_place) && args.len() == 3 {
                let arr_orig_ty = &args[0].ty;
                if let Some((n, comp_tys, variant, size, region)) = array_of_tuple_parts(arr_orig_ty) {
                    let new_arr = self.transform_term(&args[0]);
                    let new_idx = self.transform_term(&args[1]);
                    let new_val = self.transform_term(&args[2]);
                    return self.rewrite_array_with_aot(
                        &new_arr, &new_idx, &new_val, &comp_tys, &variant, &size, &region, n, span,
                    );
                }
            }

            // _w_intrinsic_uninit() where result was [n](A,B)
            if id == known.uninit && args.is_empty() {
                if soa_yields_tuple_arrays(orig_result_ty).is_some() {
                    let sym = match &func.kind {
                        TermKind::Var(VarRef::Symbol(s)) => *s,
                        // For Builtin form, we need a Symbol for `rewrite_uninit_aot`.
                        // Look up the surface name and intern it.
                        _ => self.symbols.alloc(by_id(id).raw.surface_name.to_string()),
                    };
                    let soa_ty = soa_type(orig_result_ty);
                    return self.rewrite_uninit_aot(&soa_ty, sym, span);
                }
            }

            // _w_intrinsic_length(arr) where arr was [n](A,B)
            if id == known.length && args.len() == 1 {
                let arr_orig_ty = &args[0].ty;
                if soa_yields_tuple_arrays(arr_orig_ty).is_some() {
                    let sym = match &func.kind {
                        TermKind::Var(VarRef::Symbol(s)) => *s,
                        _ => self.symbols.alloc(by_id(id).raw.surface_name.to_string()),
                    };
                    let new_arr = self.transform_term(&args[0]);
                    return self.rewrite_length_aot(&new_arr, sym, new_result_ty, span);
                }
            }
        }

        // Default: recursively transform all parts.
        let _ = orig_result_ty;
        let new_func = self.transform_term(func);
        let new_args: Vec<Term> = args.iter().map(|a| self.transform_term(a)).collect();
        self.mk_term(
            new_result_ty,
            span,
            TermKind::App {
                func: Box::new(new_func),
                args: new_args,
            },
        )
    }

    // =========================================================================
    // Array-of-Tuple rewrite helpers
    // =========================================================================

    /// Distribute an index over an array-of-tuple input. For
    /// `arr: [n](A,B)` distributed to `([n]A, [n]B)`, rewrites
    /// `arr[i]` to `(proj(arr,0)[i], proj(arr,1)[i])`.
    fn rewrite_index_aot(
        &mut self,
        arr: &Term,
        idx: &Term,
        comp_tys: &[Type<TypeName>],
        variant: &Type<TypeName>,
        size: &Type<TypeName>,
        region: &Type<TypeName>,
        n: usize,
        span: Span,
    ) -> Term {
        let components: Vec<Term> = (0..n)
            .map(|i| {
                // Recursively soa_type the constructed array so nested array-of-tuples
                // like [8](int, vec3) are further distributed to ([8]int, [8]vec3).
                let comp_arr_ty = soa_type(&Type::Constructed(
                    TypeName::Array,
                    vec![
                        soa_type(&comp_tys[i]),
                        variant.clone(),
                        size.clone(),
                        region.clone(),
                    ],
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
    /// -> `_w_tuple(array_with(proj(arr,0), i, proj(val,0)), ...)`
    fn rewrite_array_with_aot(
        &mut self,
        arr: &Term,
        idx: &Term,
        val: &Term,
        comp_tys: &[Type<TypeName>],
        variant: &Type<TypeName>,
        size: &Type<TypeName>,
        region: &Type<TypeName>,
        n: usize,
        span: Span,
    ) -> Term {
        let components: Vec<Term> = (0..n)
            .map(|i| {
                let soa_comp_ty = soa_type(&comp_tys[i]);
                let comp_arr_ty = soa_type(&Type::Constructed(
                    TypeName::Array,
                    vec![soa_comp_ty.clone(), variant.clone(), size.clone(), region.clone()],
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
                        vec![
                            soa_type(&comp_tys[i]),
                            variant.clone(),
                            size.clone(),
                            region.clone(),
                        ],
                    ))
                })
                .collect(),
        );
        self.mk_tuple(components, result_ty, span)
    }

    /// `_w_array_lit(e1, e2, ...)` where result was `[n](A,B)`:
    /// -> `_w_tuple(_w_array_lit(proj(e1,0), proj(e2,0), ...), ...)`
    fn rewrite_array_lit_aot(
        &mut self,
        elems: &[Term],
        comp_tys: &[Type<TypeName>],
        variant: &Type<TypeName>,
        size: &Type<TypeName>,
        region: &Type<TypeName>,
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
                    vec![soa_comp_ty, variant.clone(), size.clone(), region.clone()],
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
                        vec![
                            soa_type(&comp_tys[i]),
                            variant.clone(),
                            size.clone(),
                            region.clone(),
                        ],
                    ))
                })
                .collect(),
        );
        self.mk_tuple(components, result_ty, span)
    }

    /// `_w_intrinsic_uninit()` where result was `[n](A,B)`:
    /// -> `_w_tuple(_w_intrinsic_uninit(), _w_intrinsic_uninit())`
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
                        self.mk_term(ct.clone(), span, TermKind::Var(VarRef::Symbol(uninit_sym)))
                    })
                    .collect();
                self.mk_tuple(components, soa_ty.clone(), span)
            }
            _ => self.mk_term(soa_ty.clone(), span, TermKind::Var(VarRef::Symbol(uninit_sym))),
        }
    }

    /// `_w_intrinsic_length(arr)` where arr was `[n](A,B)`:
    /// -> `_w_intrinsic_length(proj(arr, 0))`
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
        let func = self.mk_term(func_ty, span, TermKind::Var(VarRef::Symbol(length_sym)));
        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(func),
                args: vec![first_arr],
            },
        )
    }

    // =========================================================================
    // SOAC rewriting
    // =========================================================================

    fn transform_soac(&mut self, soac: &SoacOp) -> SoacOp {
        match soac {
            SoacOp::Map {
                lam,
                inputs,
                destination,
            } => {
                let new_lam = self.transform_soac_body(lam);
                let new_inputs: Vec<ArrayExpr> =
                    inputs.iter().map(|ae| self.transform_array_expr(ae)).collect();
                SoacOp::Map {
                    lam: new_lam,
                    inputs: new_inputs,
                    destination: *destination,
                }
            }
            SoacOp::Reduce { op, ne, input } => {
                let new_op = self.transform_soac_body(op);
                let new_ne = self.transform_term(ne);
                let new_input = self.transform_array_expr(input);
                SoacOp::Reduce {
                    op: new_op,
                    ne: Box::new(new_ne),
                    input: new_input,
                }
            }
            SoacOp::Scan {
                op,
                ne,
                input,
                destination,
            } => {
                let new_op = self.transform_soac_body(op);
                let new_ne = self.transform_term(ne);
                let new_input = self.transform_array_expr(input);
                SoacOp::Scan {
                    op: new_op,
                    ne: Box::new(new_ne),
                    input: new_input,
                    destination: *destination,
                }
            }
            SoacOp::Filter {
                pred,
                input,
                destination,
            } => {
                let new_pred = self.transform_soac_body(pred);
                let new_input = self.transform_array_expr(input);
                SoacOp::Filter {
                    pred: new_pred,
                    input: new_input,
                    destination: *destination,
                }
            }
            SoacOp::Scatter { dest, lam, inputs } => {
                let new_dest = self.transform_place(dest);
                let new_lam = self.transform_soac_body(lam);
                let new_inputs: Vec<ArrayExpr> =
                    inputs.iter().map(|ae| self.transform_array_expr(ae)).collect();
                SoacOp::Scatter {
                    dest: new_dest,
                    lam: new_lam,
                    inputs: new_inputs,
                }
            }
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
            } => {
                let new_dest = self.transform_place(dest);
                let new_op = self.transform_soac_body(op);
                let new_ne = self.transform_term(ne);
                let new_indices = self.transform_array_expr(indices);
                let new_values = self.transform_array_expr(values);
                SoacOp::ReduceByIndex {
                    dest: new_dest,
                    op: new_op,
                    ne: Box::new(new_ne),
                    indices: new_indices,
                    values: new_values,
                }
            }
        }
    }

    fn transform_place(&mut self, place: &Place) -> Place {
        Place {
            id: place.id,
            elem_ty: soa_type(&place.elem_ty),
        }
    }

    // =========================================================================
    // ArrayExpr rewriting
    // =========================================================================

    fn transform_array_expr(&mut self, ae: &ArrayExpr) -> ArrayExpr {
        match ae {
            ArrayExpr::Var(vr, ty) => {
                let t = crate::tlc::atom_var_term(*vr, ty.clone(), &mut self.term_ids);
                let new = self.transform_term(&t);
                match new.kind {
                    TermKind::Var(new_vr) => ArrayExpr::Var(new_vr, new.ty),
                    _ => unreachable!("SoA transform of a named input yields a named input"),
                }
            }
            ArrayExpr::Zip(exprs) => {
                let new_exprs: Vec<ArrayExpr> =
                    exprs.iter().map(|e| self.transform_array_expr(e)).collect();
                ArrayExpr::Zip(new_exprs)
            }
            ArrayExpr::Literal(terms) => {
                let new_terms: Vec<Term> = terms.iter().map(|t| self.transform_term(t)).collect();
                ArrayExpr::Literal(new_terms)
            }
            ArrayExpr::Range { start, len, step } => ArrayExpr::Range {
                start: Box::new(self.transform_term(start)),
                len: Box::new(self.transform_term(len)),
                step: step.as_ref().map(|s| Box::new(self.transform_term(s))),
            },
        }
    }

    /// Transform an ArrayExpr appearing as a standalone term.
    /// Standalone Zip is converted to tuple construction here.
    fn transform_array_expr_term(
        &mut self,
        ae: &ArrayExpr,
        orig_ty: &Type<TypeName>,
        new_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        // Standalone Zip -> tuple construction: zip(a, b) becomes a Tuple term.
        if let ArrayExpr::Zip(exprs) = ae {
            if !exprs.is_empty() {
                let components: Vec<Term> = exprs
                    .iter()
                    .map(|inner_ae| match inner_ae {
                        ArrayExpr::Var(vr, ty) => {
                            let t = crate::tlc::atom_var_term(*vr, ty.clone(), &mut self.term_ids);
                            self.transform_term(&t)
                        }
                        _ => {
                            let new_inner = self.transform_array_expr(inner_ae);
                            self.mk_term(new_ty.clone(), span, TermKind::ArrayExpr(new_inner))
                        }
                    })
                    .collect();
                return self.mk_tuple(components, new_ty, span);
            }
        }

        // Array-of-tuple literal: distribute into per-component arrays.
        if let ArrayExpr::Literal(elems) = ae {
            if !elems.is_empty() {
                if let Some((_n, comp_tys, variant, size, region)) = array_of_tuple_parts(orig_ty) {
                    let new_elems: Vec<Term> = elems.iter().map(|t| self.transform_term(t)).collect();
                    return self
                        .rewrite_array_lit_aot(&new_elems, &comp_tys, &variant, &size, &region, span);
                }
            }
        }

        let new_ae = self.transform_array_expr(ae);
        self.mk_term(new_ty, span, TermKind::ArrayExpr(new_ae))
    }

    // =========================================================================
    // Lambda rewriting
    // =========================================================================

    fn transform_lambda(&mut self, lam: &Lambda) -> Lambda {
        let new_params: Vec<(crate::SymbolId, Type<TypeName>)> =
            lam.params.iter().map(|(sym, ty)| (*sym, soa_type(ty))).collect();
        let new_body = self.transform_term(&lam.body);
        Lambda {
            params: new_params,
            body: Box::new(new_body),
            ret_ty: soa_type(&lam.ret_ty),
        }
    }

    fn transform_soac_body(&mut self, sb: &super::SoacBody) -> super::SoacBody {
        let new_lam = self.transform_lambda(&sb.lam);
        let new_captures: Vec<(crate::SymbolId, Type<TypeName>, Term)> = sb
            .captures
            .iter()
            .map(|(sym, ty, term)| (*sym, soa_type(ty), self.transform_term(term)))
            .collect();
        super::SoacBody {
            lam: new_lam,
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
    // SOAC normalization: Map+Zip flattening
    // =========================================================================

    /// If the Map has multiple inputs but a single tuple-typed lambda param,
    /// split the param into N separate params and substitute.
    fn try_normalize_map(
        &mut self,
        sb: super::SoacBody,
        inputs: Vec<ArrayExpr>,
        destination: crate::types::SoacOwnership,
    ) -> SoacOp {
        if inputs.len() <= 1 || sb.lam.params.len() != 1 {
            return SoacOp::Map {
                lam: sb,
                inputs,
                destination,
            };
        }

        let (old_param, param_ty) = (sb.lam.params[0].0, sb.lam.params[0].1.clone());

        // Must be a concrete tuple type matching the input count.
        let flat_types = match &param_ty {
            Type::Constructed(TypeName::Tuple(_), types) if !types.is_empty() => flatten_tuple_types(types),
            _ => {
                return SoacOp::Map {
                    lam: sb,
                    inputs,
                    destination,
                };
            }
        };

        if flat_types.len() != inputs.len() || has_type_variables(&param_ty) {
            return SoacOp::Map {
                lam: sb,
                inputs,
                destination,
            };
        }

        let super::SoacBody { lam, captures } = sb;

        // Create N fresh params.
        let new_params: Vec<(crate::SymbolId, Type<TypeName>)> = flat_types
            .iter()
            .enumerate()
            .map(|(i, ty)| (self.symbols.alloc(format!("_sn_{}", i)), ty.clone()))
            .collect();

        // Substitute: every `Var(old_param)` -> `_w_tuple(Var(p0), ..., Var(pN))`
        // reconstructed with the original tuple type. Downstream simplification
        // (partial eval / project folding) will reduce proj(tuple(...), i) -> pi.
        let span = lam.body.span;
        let rewritten_body = self.substitute_param(*lam.body, old_param, &new_params, &param_ty, span);

        SoacOp::Map {
            lam: super::SoacBody {
                lam: Lambda {
                    params: new_params,
                    body: Box::new(rewritten_body),
                    ret_ty: lam.ret_ty,
                },
                captures,
            },
            inputs,
            destination,
        }
    }

    /// Replace every occurrence of `Var(old_sym)` with a tuple reconstruction
    /// from the new params. Respects shadowing.
    fn substitute_param(
        &mut self,
        term: Term,
        old_sym: crate::SymbolId,
        new_params: &[(crate::SymbolId, Type<TypeName>)],
        tuple_ty: &Type<TypeName>,
        span: Span,
    ) -> Term {
        if let TermKind::Var(VarRef::Symbol(sym)) = &term.kind {
            if *sym == old_sym {
                return self.build_tuple_reconstruction(new_params, tuple_ty, span);
            }
        }

        // Stop at shadowing.
        match &term.kind {
            TermKind::Let { name, .. } if *name == old_sym => return term,
            TermKind::Lambda(lam) if lam.params.iter().any(|(s, _)| *s == old_sym) => return term,
            _ => {}
        }

        let fresh_id = self.term_ids.next_id();
        term.map_children(fresh_id, &mut |child| {
            self.substitute_param(child, old_sym, new_params, tuple_ty, span)
        })
    }

    /// Build `Tuple(Var(p0), Var(p1), ..., Var(pN))` matching the original tuple type.
    ///
    /// For nested tuple types like `((A, B), C)` with flat params `[p0, p1, p2]`,
    /// builds `Tuple(Tuple(Var(p0), Var(p1)), Var(p2))` to match the nesting.
    fn build_tuple_reconstruction(
        &mut self,
        new_params: &[(crate::SymbolId, Type<TypeName>)],
        tuple_ty: &Type<TypeName>,
        span: Span,
    ) -> Term {
        match tuple_ty {
            Type::Constructed(TypeName::Tuple(_), component_types) if !component_types.is_empty() => {
                let mut offset = 0;
                let mut elements = Vec::with_capacity(component_types.len());
                for comp_ty in component_types {
                    let count = flat_type_count(comp_ty);
                    let sub_params = &new_params[offset..offset + count];
                    let elem = self.build_tuple_reconstruction(sub_params, comp_ty, span);
                    elements.push(elem);
                    offset += count;
                }
                self.mk_tuple(elements, tuple_ty.clone(), span)
            }
            _ => {
                // Leaf -- single param.
                assert_eq!(new_params.len(), 1);
                let (sym, ty) = &new_params[0];
                self.mk_term(ty.clone(), span, TermKind::Var(VarRef::Symbol(*sym)))
            }
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

    /// Build a `TermKind::Tuple` term.
    fn mk_tuple(&mut self, components: Vec<Term>, result_ty: Type<TypeName>, span: Span) -> Term {
        self.mk_term(result_ty, span, TermKind::Tuple(components))
    }

    /// Build a `TermKind::TupleProj` term.
    fn mk_tuple_proj(&mut self, term: Term, index: usize, result_ty: Type<TypeName>, span: Span) -> Term {
        self.mk_term(
            result_ty,
            span,
            TermKind::TupleProj {
                tuple: Box::new(term),
                idx: index,
            },
        )
    }

    /// Build a `TermKind::Index` term.
    fn mk_index(&mut self, arr: Term, idx: Term, result_ty: Type<TypeName>, span: Span) -> Term {
        self.mk_term(
            result_ty,
            span,
            TermKind::Index {
                array: Box::new(arr),
                index: Box::new(idx),
            },
        )
    }

    /// Build `_w_intrinsic_array_with(arr, idx, val)` as a
    /// `VarRef::Builtin` call so downstream passes dispatch by id.
    fn mk_array_with(
        &mut self,
        arr: Term,
        idx: Term,
        val: Term,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let aw_id = catalog().known().array_with;
        let t3 = Type::Constructed(TypeName::Arrow, vec![val.ty.clone(), result_ty.clone()]);
        let t2 = Type::Constructed(TypeName::Arrow, vec![idx.ty.clone(), t3.clone()]);
        let t1 = Type::Constructed(TypeName::Arrow, vec![arr.ty.clone(), t2.clone()]);
        let func = self.mk_term(
            t1,
            span,
            TermKind::Var(VarRef::Builtin {
                id: aw_id,
                overload_idx: 0,
            }),
        );

        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(func),
                args: vec![arr, idx, val],
            },
        )
    }

    /// Build a `TermKind::ArrayExpr(ArrayExpr::Literal(elems))` term.
    fn mk_array_lit(&mut self, elems: Vec<Term>, result_ty: Type<TypeName>, span: Span) -> Term {
        self.mk_term(result_ty, span, TermKind::ArrayExpr(ArrayExpr::Literal(elems)))
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Run the combined SoA transform and SOAC normalization on a TLC program.
///
/// 1. Rewrites `[n](A,B)` types to `([n]A, [n]B)` and adjusts all operations
///    that touch array-of-tuple types.
/// 2. Flattens Map+Zip into multi-input Map with split lambda params.
/// 3. Converts standalone Zip to tuple construction.
pub fn run(program: &mut Program) {
    let mut transformer = SoaTransformer::new(&mut program.symbols, &mut program.term_ids);
    for def in &mut program.defs {
        transformer.transform_def(def);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[path = "soa_transform_tests.rs"]
mod soa_transform_tests;
