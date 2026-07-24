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

use super::data::Empty;
use super::inline::SoacHelpersInlined;
use super::partial_eval::PartialEvaled;
use super::{
    ArrayExpr, Family, Lambda, Program, RewriteDecision, SoacBody, SoacOp, Stage, Term, TermId,
    TermIdSource, TermKind, TermRewriter, VarRef,
};
use crate::ast::{Span, TypeName};
use crate::builtins::{by_id, catalog};
use crate::types::TypeExt;
use crate::SymbolTable;
use polytype::Type;

/// TLC after its first SoA normalization.
#[derive(Debug, Clone, Copy, Default)]
pub struct SoaNormalized;

impl super::Stage for SoaNormalized {
    type Family = super::run::Polymorphic;
    type GlobalContext = super::context::RewriteGlobal;
}

/// TLC after normalization of array structure exposed by inlining.
#[derive(Debug, Clone, Copy, Default)]
pub struct InlinedSoaNormalized;

impl super::Stage for InlinedSoaNormalized {
    type Family = super::monomorphize::Monomorphic;
    type GlobalContext = super::context::RewriteGlobal;
}

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

    fn transform_term(&mut self, term: Term<Empty, Empty>) -> Term<Empty, Empty> {
        let mut term = term.rewrite(self);
        term.rewrite_types(self.term_ids, &mut soa_type);
        term.rewrite(&mut MapNormalizer {
            term_ids: &mut *self.term_ids,
            symbols: &mut *self.symbols,
        })
    }

    fn structural_replacement(&mut self, term: &Term<Empty, Empty>) -> Option<Term<Empty, Empty>> {
        match &term.kind {
            TermKind::App { func, args } => self.rewrite_special_app(func, args, &term.ty, term.span),
            TermKind::Index { array, index } => {
                let (n, component_types, variant, size, region) = array_of_tuple_parts(&array.ty)?;
                Some(self.rewrite_index_aot(
                    array,
                    index,
                    &component_types,
                    &variant,
                    &size,
                    &region,
                    n,
                    term.span,
                ))
            }
            TermKind::ArrayExpr(array) => self.rewrite_special_array_expr(array, &term.ty, term.span),
            _ => None,
        }
    }

    /// Transform a function application. This is where we intercept intrinsics
    /// that operate on array-of-tuple types and rewrite them.
    fn rewrite_special_app(
        &mut self,
        func: &Term<Empty, Empty>,
        args: &[Term<Empty, Empty>],
        orig_result_ty: &Type<TypeName>,
        span: Span,
    ) -> Option<Term<Empty, Empty>> {
        let known = catalog().known();
        if let Some(id) = crate::tlc::var_term_builtin_id(func, &self.symbols) {
            // array_with(arr, i, val) where arr was [n](A,B)
            if (id == known.array_with || id == known.array_with_in_place) && args.len() == 3 {
                let arr_orig_ty = &args[0].ty;
                if let Some((n, comp_tys, variant, size, region)) = array_of_tuple_parts(arr_orig_ty) {
                    return Some(self.rewrite_array_with_aot(
                        &args[0], &args[1], &args[2], &comp_tys, &variant, &size, &region, n, span,
                    ));
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
                    return Some(self.rewrite_uninit_aot(&soa_ty, sym, span));
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
                    return Some(self.rewrite_length_aot(&args[0], sym, soa_type(orig_result_ty), span));
                }
            }
        }
        None
    }

    // =========================================================================
    // Array-of-Tuple rewrite helpers
    // =========================================================================

    /// Distribute an index over an array-of-tuple input. For
    /// `arr: [n](A,B)` distributed to `([n]A, [n]B)`, rewrites
    /// `arr[i]` to `(proj(arr,0)[i], proj(arr,1)[i])`.
    fn rewrite_index_aot(
        &mut self,
        arr: &Term<Empty, Empty>,
        idx: &Term<Empty, Empty>,
        comp_tys: &[Type<TypeName>],
        variant: &Type<TypeName>,
        size: &Type<TypeName>,
        region: &Type<TypeName>,
        n: usize,
        span: Span,
    ) -> Term<Empty, Empty> {
        let components: Vec<Term<Empty, Empty>> = (0..n)
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
        arr: &Term<Empty, Empty>,
        idx: &Term<Empty, Empty>,
        val: &Term<Empty, Empty>,
        comp_tys: &[Type<TypeName>],
        variant: &Type<TypeName>,
        size: &Type<TypeName>,
        region: &Type<TypeName>,
        n: usize,
        span: Span,
    ) -> Term<Empty, Empty> {
        let components: Vec<Term<Empty, Empty>> = (0..n)
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
        elems: &[Term<Empty, Empty>],
        comp_tys: &[Type<TypeName>],
        variant: &Type<TypeName>,
        size: &Type<TypeName>,
        region: &Type<TypeName>,
        span: Span,
    ) -> Term<Empty, Empty> {
        let n = comp_tys.len();
        let components: Vec<Term<Empty, Empty>> = (0..n)
            .map(|i| {
                let soa_comp_ty = soa_type(&comp_tys[i]);
                let projected_elems: Vec<Term<Empty, Empty>> = elems
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
    ) -> Term<Empty, Empty> {
        match soa_ty {
            Type::Constructed(TypeName::Tuple(_), comp_tys) => {
                let components: Vec<Term<Empty, Empty>> = comp_tys
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
        arr: &Term<Empty, Empty>,
        length_sym: crate::SymbolId,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term<Empty, Empty> {
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

    /// Rewrite a standalone array expression whose representation changes.
    fn rewrite_special_array_expr(
        &mut self,
        ae: &ArrayExpr<Empty, Empty>,
        orig_ty: &Type<TypeName>,
        span: Span,
    ) -> Option<Term<Empty, Empty>> {
        let new_ty = soa_type(orig_ty);

        // Standalone Zip -> tuple construction: zip(a, b) becomes a Tuple term.
        if let ArrayExpr::Zip(exprs) = ae {
            if !exprs.is_empty() {
                let components: Vec<Term<Empty, Empty>> = exprs
                    .iter()
                    .map(|inner_ae| match inner_ae {
                        ArrayExpr::Var(vr, ty) => {
                            crate::tlc::atom_var_term(*vr, ty.clone(), &mut self.term_ids)
                        }
                        _ => self.mk_term(new_ty.clone(), span, TermKind::ArrayExpr(inner_ae.clone())),
                    })
                    .collect();
                return Some(self.mk_tuple(components, new_ty, span));
            }
        }

        // Array-of-tuple literal: distribute into per-component arrays.
        if let ArrayExpr::Literal(elems) = ae {
            if !elems.is_empty() {
                if let Some((_n, comp_tys, variant, size, region)) = array_of_tuple_parts(orig_ty) {
                    return Some(
                        self.rewrite_array_lit_aot(elems, &comp_tys, &variant, &size, &region, span),
                    );
                }
            }
        }
        None
    }

    // =========================================================================
    // Term construction helpers
    // =========================================================================

    fn mk_term(
        &mut self,
        ty: Type<TypeName>,
        span: Span,
        kind: TermKind<Empty, Empty>,
    ) -> Term<Empty, Empty> {
        Term {
            id: self.term_ids.next_id(),
            ty,
            span,
            kind,
        }
    }

    /// Build a `TermKind::Tuple` term.
    fn mk_tuple(
        &mut self,
        components: Vec<Term<Empty, Empty>>,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term<Empty, Empty> {
        self.mk_term(result_ty, span, TermKind::Tuple(components))
    }

    /// Build a `TermKind::TupleProj` term.
    fn mk_tuple_proj(
        &mut self,
        term: Term<Empty, Empty>,
        index: usize,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term<Empty, Empty> {
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
    fn mk_index(
        &mut self,
        arr: Term<Empty, Empty>,
        idx: Term<Empty, Empty>,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term<Empty, Empty> {
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
        arr: Term<Empty, Empty>,
        idx: Term<Empty, Empty>,
        val: Term<Empty, Empty>,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term<Empty, Empty> {
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
    fn mk_array_lit(
        &mut self,
        elems: Vec<Term<Empty, Empty>>,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term<Empty, Empty> {
        self.mk_term(result_ty, span, TermKind::ArrayExpr(ArrayExpr::Literal(elems)))
    }
}

impl TermRewriter<Empty, Empty> for SoaTransformer<'_, '_> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_node_before_children(&mut self, term: &mut Term<Empty, Empty>) -> RewriteDecision {
        let Some(mut replacement) = self.structural_replacement(term) else {
            return RewriteDecision::Unchanged;
        };
        replacement.id = term.id;
        *term = replacement;
        RewriteDecision::Changed
    }
}

/// Normalize multi-input maps only after the uniform SoA type rewrite, so the
/// lambda parameter shape and the input shapes are compared in the same type
/// representation.
struct MapNormalizer<'a, 'ids> {
    term_ids: &'ids mut TermIdSource,
    symbols: &'a mut SymbolTable,
}

impl MapNormalizer<'_, '_> {
    fn normalize_map(
        &mut self,
        body: SoacBody<Empty, Empty>,
        inputs: Vec<ArrayExpr<Empty, Empty>>,
        destination: crate::types::SoacOwnership,
    ) -> Option<SoacOp<Empty, Empty>> {
        if inputs.len() <= 1 || body.lam.params.len() != 1 {
            return None;
        }

        let (old_param, param_ty) = (body.lam.params[0].0, body.lam.params[0].1.clone());
        let flat_types = match &param_ty {
            Type::Constructed(TypeName::Tuple(_), types) if !types.is_empty() => flatten_tuple_types(types),
            _ => return None,
        };
        if flat_types.len() != inputs.len() || has_type_variables(&param_ty) {
            return None;
        }

        let SoacBody { lam, data } = body;
        let new_params: Vec<(crate::SymbolId, Type<TypeName>)> = flat_types
            .into_iter()
            .enumerate()
            .map(|(index, ty)| (self.symbols.alloc(format!("_sn_{index}")), ty))
            .collect();
        let span = lam.body.span;
        let rewritten_body = super::subst::substitute_with(
            *lam.body,
            old_param,
            &mut |_occurrence, term_ids| build_tuple_reconstruction(&new_params, &param_ty, span, term_ids),
            self.term_ids,
        );

        Some(SoacOp::Map {
            lam: SoacBody {
                lam: Lambda {
                    params: new_params,
                    body: Box::new(rewritten_body),
                    ret_ty: lam.ret_ty,
                },
                data,
            },
            inputs,
            destination,
        })
    }
}

impl TermRewriter<Empty, Empty> for MapNormalizer<'_, '_> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_node(&mut self, term: &mut Term<Empty, Empty>) -> RewriteDecision {
        let replacement = match &term.kind {
            TermKind::Soac(SoacOp::Map {
                lam,
                inputs,
                destination,
            }) => self.normalize_map(lam.clone(), inputs.clone(), *destination).map(TermKind::Soac),
            _ => None,
        };
        let Some(kind) = replacement else {
            return RewriteDecision::Unchanged;
        };
        term.kind = kind;
        RewriteDecision::Changed
    }
}

/// Build a tuple reconstruction from flattened map parameters. Each call is
/// made for one substituted occurrence, so every inserted term receives IDs
/// from the owning program.
fn build_tuple_reconstruction(
    new_params: &[(crate::SymbolId, Type<TypeName>)],
    tuple_ty: &Type<TypeName>,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Term<Empty, Empty> {
    let kind = match tuple_ty {
        Type::Constructed(TypeName::Tuple(_), component_types) if !component_types.is_empty() => {
            let mut offset = 0;
            let mut elements = Vec::with_capacity(component_types.len());
            for component_type in component_types {
                let count = flat_type_count(component_type);
                elements.push(build_tuple_reconstruction(
                    &new_params[offset..offset + count],
                    component_type,
                    span,
                    term_ids,
                ));
                offset += count;
            }
            TermKind::Tuple(elements)
        }
        _ => {
            assert_eq!(new_params.len(), 1);
            TermKind::Var(VarRef::Symbol(new_params[0].0))
        }
    };
    Term {
        id: term_ids.next_id(),
        ty: tuple_ty.clone(),
        span,
        kind,
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Run the first combined SoA transform and SOAC normalization.
///
/// 1. Rewrites `[n](A,B)` types to `([n]A, [n]B)` and adjusts all operations
///    that touch array-of-tuple types.
/// 2. Flattens Map+Zip into multi-input Map with split lambda params.
/// 3. Converts standalone Zip to tuple construction.
pub fn run(program: Program<PartialEvaled>) -> Program<SoaNormalized> {
    transform_program(program)
}

/// Re-run the same normalization after inlining exposes new array structure.
pub fn rerun(program: Program<SoacHelpersInlined>) -> Program<InlinedSoaNormalized> {
    transform_program(program)
}

fn transform_program<S, T>(program: Program<S>) -> Program<T>
where
    S: Stage,
    T: Stage<Family = S::Family, GlobalContext = S::GlobalContext>,
    S::Family: Family<ClosureData = Empty, SoacBodyData = Empty>,
{
    let Program {
        defs,
        mut symbols,
        def_syms,
        mut term_ids,
        global_context,
    } = program;
    let mut transformer = SoaTransformer::new(&mut symbols, &mut term_ids);
    let defs = defs
        .into_iter()
        .map(|mut def| {
            def.ty = soa_type(&def.ty);
            def.body = transformer.transform_term(def.body);
            def
        })
        .collect();
    drop(transformer);
    Program::from_parts(defs, symbols, def_syms, term_ids, global_context)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[path = "soa_transform_tests.rs"]
mod soa_transform_tests;
