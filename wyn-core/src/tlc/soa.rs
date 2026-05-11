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

use super::{ArrayExpr, Def, Lambda, LoopKind, Place, Program, SoacOp, Term, TermIdSource, TermKind};
use crate::SymbolTable;
use crate::ast::{Span, TypeName};
use crate::types::TypeExt;
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
        _ if ty.is_array() => {
            let elem = soa_type(ty.elem_type().expect("Array has elem"));
            let size = ty.array_size().expect("Array has size").clone();
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
                            vec![ct.clone(), size.clone(), variant.clone()],
                        ))
                    })
                    .collect();
                Type::Constructed(TypeName::Tuple(n), distributed)
            } else {
                Type::Constructed(TypeName::Array, vec![elem, size, variant])
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
    let elem = ty.elem_type()?;
    if !ty.is_array() {
        return None;
    }
    match elem {
        Type::Constructed(TypeName::Tuple(n), _) => Some(*n),
        // Check recursively: the element might become a tuple after soa_type
        _ => {
            let elem_soa = soa_type(elem);
            match elem_soa {
                Type::Constructed(TypeName::Tuple(n), _) => Some(n),
                _ => None,
            }
        }
    }
}

/// Extract the tuple component types from an array-of-tuple type.
/// Returns (component_types, variant, size).
fn array_of_tuple_parts(
    ty: &Type<TypeName>,
) -> Option<(Vec<Type<TypeName>>, Type<TypeName>, Type<TypeName>)> {
    if !ty.is_array() {
        return None;
    }
    match ty.elem_type()? {
        Type::Constructed(TypeName::Tuple(_), components) => {
            let variant = ty.array_variant().expect("Array has variant").clone();
            let size = ty.array_size().expect("Array has size").clone();
            Some((components.clone(), variant, size))
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
            def_syms: program.def_syms,
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
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Extern(_) => self.mk_term(new_ty, span, term.kind.clone()),

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
                        consumes_input,
                    } => self.try_normalize_map(lam, inputs, consumes_input),
                    other => other,
                };
                self.mk_term(new_ty, span, TermKind::Soac(new_soac))
            }

            TermKind::ArrayExpr(ae) => self.transform_array_expr_term(ae, orig_ty, new_ty, span),

            TermKind::Force(inner) => {
                let new_inner = self.transform_term(inner);
                self.mk_term(new_ty, span, TermKind::Force(Box::new(new_inner)))
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
        // Catalog-entry dispatch via BuiltinId (works for both
        // `VarRef::Symbol` legacy paths and `VarRef::Builtin` modern
        // user code).
        let known = crate::builtins::catalog().known();
        if let Some(id) = crate::tlc::var_term_builtin_id(func, &self.symbols) {
            // _w_intrinsic_array_with(arr, i, val) where arr was [n](A,B)
            if (id == known.array_with || id == known.array_with_in_place) && args.len() == 3 {
                let arr_orig_ty = &args[0].ty;
                if let Some(n) = is_array_of_tuple(arr_orig_ty) {
                    let (comp_tys, variant, size) = array_of_tuple_parts(arr_orig_ty).unwrap();
                    let new_arr = self.transform_term(&args[0]);
                    let new_idx = self.transform_term(&args[1]);
                    let new_val = self.transform_term(&args[2]);
                    return self.rewrite_array_with_aot(
                        &new_arr, &new_idx, &new_val, &comp_tys, &variant, &size, n, span,
                    );
                }
            }

            // _w_intrinsic_uninit() where result was [n](A,B)
            if id == known.uninit && args.is_empty() {
                if is_array_of_tuple(orig_result_ty).is_some() {
                    let sym = match &func.kind {
                        TermKind::Var(crate::tlc::VarRef::Symbol(s)) => *s,
                        // For Builtin form, we need a Symbol for `rewrite_uninit_aot`.
                        // Look up the surface name and intern it.
                        _ => self.symbols.alloc(crate::builtins::by_id(id).raw.surface_name.to_string()),
                    };
                    let soa_ty = soa_type(orig_result_ty);
                    return self.rewrite_uninit_aot(&soa_ty, sym, span);
                }
            }

            // _w_intrinsic_length(arr) where arr was [n](A,B)
            if id == known.length && args.len() == 1 {
                let arr_orig_ty = &args[0].ty;
                if is_array_of_tuple(arr_orig_ty).is_some() {
                    let sym = match &func.kind {
                        TermKind::Var(crate::tlc::VarRef::Symbol(s)) => *s,
                        _ => self.symbols.alloc(crate::builtins::by_id(id).raw.surface_name.to_string()),
                    };
                    let new_arr = self.transform_term(&args[0]);
                    return self.rewrite_length_aot(&new_arr, sym, new_result_ty, span);
                }
            }
        }

        // Compiler-generated `_w_*` operators (not catalog entries) —
        // match on `VarRef::Symbol` name directly.
        if let TermKind::Var(crate::tlc::VarRef::Symbol(sym)) = &func.kind {
            let name = self.symbols.get(*sym).cloned().unwrap_or_default();
            match name.as_str() {
                "_w_index" if args.len() == 2 => {
                    let arr_orig_ty = &args[0].ty;
                    if let Some(n) = is_array_of_tuple(arr_orig_ty) {
                        let (comp_tys, variant, size) = array_of_tuple_parts(arr_orig_ty).unwrap();
                        let new_arr = self.transform_term(&args[0]);
                        let new_idx = self.transform_term(&args[1]);
                        return self
                            .rewrite_index_aot(&new_arr, &new_idx, &comp_tys, &variant, &size, n, span);
                    }
                }
                "_w_array_lit" if !args.is_empty() => {
                    if is_array_of_tuple(orig_result_ty).is_some() {
                        let (comp_tys, variant, size) = array_of_tuple_parts(orig_result_ty).unwrap();
                        let new_elems: Vec<Term> = args.iter().map(|a| self.transform_term(a)).collect();
                        return self.rewrite_array_lit_aot(&new_elems, &comp_tys, &variant, &size, span);
                    }
                }
                _ => {}
            }
        }

        // Default: recursively transform all parts
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

    /// `_w_index(arr, i)` where arr was `[n](A,B)`, now `([n]A, [n]B)`:
    /// -> `_w_tuple(_w_index(proj(arr,0), i), _w_index(proj(arr,1), i))`
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
                    vec![soa_type(&comp_tys[i]), size.clone(), variant.clone()],
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
        n: usize,
        span: Span,
    ) -> Term {
        let components: Vec<Term> = (0..n)
            .map(|i| {
                let soa_comp_ty = soa_type(&comp_tys[i]);
                let comp_arr_ty = soa_type(&Type::Constructed(
                    TypeName::Array,
                    vec![soa_comp_ty.clone(), size.clone(), variant.clone()],
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
                        vec![soa_type(&comp_tys[i]), size.clone(), variant.clone()],
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
                    vec![soa_comp_ty, size.clone(), variant.clone()],
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
                        vec![soa_type(&comp_tys[i]), size.clone(), variant.clone()],
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
                        self.mk_term(
                            ct.clone(),
                            span,
                            TermKind::Var(crate::tlc::VarRef::Symbol(uninit_sym)),
                        )
                    })
                    .collect();
                self.mk_tuple(components, soa_ty.clone(), span)
            }
            _ => self.mk_term(
                soa_ty.clone(),
                span,
                TermKind::Var(crate::tlc::VarRef::Symbol(uninit_sym)),
            ),
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
        let func = self.mk_term(
            func_ty,
            span,
            TermKind::Var(crate::tlc::VarRef::Symbol(length_sym)),
        );
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
                consumes_input,
            } => {
                let new_lam = self.transform_soac_body(lam);
                let new_inputs: Vec<ArrayExpr> =
                    inputs.iter().map(|ae| self.transform_array_expr(ae)).collect();
                SoacOp::Map {
                    lam: new_lam,
                    inputs: new_inputs,
                    consumes_input: *consumes_input,
                }
            }
            SoacOp::Reduce { op, ne, input, props } => {
                let new_op = self.transform_soac_body(op);
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
                let new_op = self.transform_soac_body(op);
                let new_ne = self.transform_term(ne);
                let new_input = self.transform_array_expr(input);
                SoacOp::Scan {
                    op: new_op,
                    ne: Box::new(new_ne),
                    input: new_input,
                }
            }
            SoacOp::Filter { pred, input } => {
                let new_pred = self.transform_soac_body(pred);
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
                    props: props.clone(),
                }
            }
            SoacOp::Redomap {
                op,
                reduce_op,
                ne,
                inputs,
                props,
            } => {
                let new_op = self.transform_soac_body(op);
                let new_reduce_op = self.transform_soac_body(reduce_op);
                let new_ne = self.transform_term(ne);
                let new_inputs: Vec<ArrayExpr> =
                    inputs.iter().map(|ae| self.transform_array_expr(ae)).collect();
                SoacOp::Redomap {
                    op: new_op,
                    reduce_op: new_reduce_op,
                    ne: Box::new(new_ne),
                    inputs: new_inputs,
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
                index_fn: self.transform_soac_body(index_fn),
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
            ArrayExpr::StorageBuffer {
                set,
                binding,
                offset,
                len,
                elem_ty,
            } => ArrayExpr::StorageBuffer {
                set: *set,
                binding: *binding,
                offset: Box::new(self.transform_term(offset)),
                len: Box::new(self.transform_term(len)),
                elem_ty: soa_type(elem_ty),
            },
        }
    }

    /// Transform an ArrayExpr appearing as a standalone term.
    /// Standalone Zip is converted to tuple construction here.
    fn transform_array_expr_term(
        &mut self,
        ae: &ArrayExpr,
        _orig_ty: &Type<TypeName>,
        new_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        // Standalone Zip -> tuple construction: zip(a, b) becomes _w_tuple(a, b).
        if let ArrayExpr::Zip(exprs) = ae {
            if !exprs.is_empty() {
                let components: Vec<Term> = exprs
                    .iter()
                    .map(|inner_ae| match inner_ae {
                        ArrayExpr::Ref(t) => self.transform_term(t),
                        _ => {
                            let new_inner = self.transform_array_expr(inner_ae);
                            self.mk_term(new_ty.clone(), span, TermKind::ArrayExpr(new_inner))
                        }
                    })
                    .collect();
                return self.mk_tuple(components, new_ty, span);
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
        consumes_input: bool,
    ) -> SoacOp {
        if inputs.len() <= 1 || sb.lam.params.len() != 1 {
            return SoacOp::Map {
                lam: sb,
                inputs,
                consumes_input,
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
                    consumes_input,
                };
            }
        };

        if flat_types.len() != inputs.len() || has_type_variables(&param_ty) {
            return SoacOp::Map {
                lam: sb,
                inputs,
                consumes_input,
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
            consumes_input,
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
        if let TermKind::Var(crate::tlc::VarRef::Symbol(sym)) = &term.kind {
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

        term.map_children(&mut |child| self.substitute_param(child, old_sym, new_params, tuple_ty, span))
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
                self.mk_term(ty.clone(), span, TermKind::Var(crate::tlc::VarRef::Symbol(*sym)))
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

    /// Build `_w_tuple(c0, c1, ...)`.
    fn mk_tuple(&mut self, components: Vec<Term>, result_ty: Type<TypeName>, span: Span) -> Term {
        let tuple_sym = self.resolve_or_alloc("_w_tuple");
        if components.is_empty() {
            return self.mk_term(
                result_ty,
                span,
                TermKind::Var(crate::tlc::VarRef::Symbol(tuple_sym)),
            );
        }

        // Build flat application: _w_tuple(c0, c1, ...)
        let mut func_ty = result_ty.clone();
        for comp in components.iter().rev() {
            func_ty = Type::Constructed(TypeName::Arrow, vec![comp.ty.clone(), func_ty]);
        }
        let func = self.mk_term(
            func_ty,
            span,
            TermKind::Var(crate::tlc::VarRef::Symbol(tuple_sym)),
        );

        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(func),
                args: components,
            },
        )
    }

    /// Build `_w_tuple_proj(term, index)`.
    fn mk_tuple_proj(&mut self, term: Term, index: usize, result_ty: Type<TypeName>, span: Span) -> Term {
        let proj_sym = self.resolve_or_alloc("_w_tuple_proj");
        let idx_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let idx_term = self.mk_term(idx_ty.clone(), span, TermKind::IntLit(index.to_string()));

        // _w_tuple_proj : term.ty -> i32 -> result_ty
        let inner_ty = Type::Constructed(TypeName::Arrow, vec![idx_ty, result_ty.clone()]);
        let func_ty = Type::Constructed(TypeName::Arrow, vec![term.ty.clone(), inner_ty.clone()]);
        let func = self.mk_term(func_ty, span, TermKind::Var(crate::tlc::VarRef::Symbol(proj_sym)));

        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(func),
                args: vec![term, idx_term],
            },
        )
    }

    /// Build `_w_index(arr, idx)`.
    fn mk_index(&mut self, arr: Term, idx: Term, result_ty: Type<TypeName>, span: Span) -> Term {
        let index_sym = self.resolve_or_alloc("_w_index");
        let inner_ty = Type::Constructed(TypeName::Arrow, vec![idx.ty.clone(), result_ty.clone()]);
        let func_ty = Type::Constructed(TypeName::Arrow, vec![arr.ty.clone(), inner_ty.clone()]);
        let func = self.mk_term(
            func_ty,
            span,
            TermKind::Var(crate::tlc::VarRef::Symbol(index_sym)),
        );

        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(func),
                args: vec![arr, idx],
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
        let aw_id = crate::builtins::catalog().known().array_with;
        let t3 = Type::Constructed(TypeName::Arrow, vec![val.ty.clone(), result_ty.clone()]);
        let t2 = Type::Constructed(TypeName::Arrow, vec![idx.ty.clone(), t3.clone()]);
        let t1 = Type::Constructed(TypeName::Arrow, vec![arr.ty.clone(), t2.clone()]);
        let func = self.mk_term(
            t1,
            span,
            TermKind::Var(crate::tlc::VarRef::Builtin {
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

    /// Build `_w_array_lit(e0, e1, ...)`.
    fn mk_array_lit(&mut self, elems: Vec<Term>, result_ty: Type<TypeName>, span: Span) -> Term {
        let al_sym = self.resolve_or_alloc("_w_array_lit");
        if elems.is_empty() {
            return self.mk_term(result_ty, span, TermKind::Var(crate::tlc::VarRef::Symbol(al_sym)));
        }

        let func_ty = Type::Constructed(TypeName::Arrow, vec![elems[0].ty.clone(), result_ty.clone()]);
        let func = self.mk_term(func_ty, span, TermKind::Var(crate::tlc::VarRef::Symbol(al_sym)));

        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(func),
                args: elems,
            },
        )
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

// =============================================================================
// Public API
// =============================================================================

/// Run the combined SoA transform and SOAC normalization on a TLC program.
///
/// 1. Rewrites `[n](A,B)` types to `([n]A, [n]B)` and adjusts all operations
///    that touch array-of-tuple types.
/// 2. Flattens Map+Zip into multi-input Map with split lambda params.
/// 3. Converts standalone Zip to tuple construction.
pub fn run(program: Program) -> Program {
    let transformer = SoaTransformer::new(program.symbols.clone());
    transformer.transform_program(program)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[path = "soa_transform_tests.rs"]
mod soa_transform_tests;
