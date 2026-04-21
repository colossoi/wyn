//! Buffer specialization pass for TLC.
//!
//! Runs after monomorphize, before inline. Rewrites functions that take
//! view-array parameters so that the buffer identity (set, binding) is
//! baked into the function body. After this pass, no `DefMeta::Function`
//! should have view-array parameters — only entry points retain them.
//!
//! When `first(data)` is called where `data` is bound to storage buffer
//! (set=0, binding=0), this creates a specialized copy:
//!
//! ```text
//! def first_buf0_0(offset: u32, len: u32) =
//!     _w_intrinsic_storage_index(0, 0, offset + 0)
//! ```

use super::{
    ArrayExpr, Def, DefMeta, Lambda, LoopKind, Place, Program, SoacOp, Term, TermIdSource, TermKind,
};
use crate::ast::{self, Span, TypeName};
use crate::interface;
use crate::types::TypeExt;
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use std::collections::HashMap;

/// Buffer binding info for a view-array parameter.
///
/// `scalar_params` distinguishes the two contexts a view can appear in:
/// * `None` — an entry-point view param. The storage is accessed through
///   synthesized terms: `offset = IntLit(0)`, `len = _w_intrinsic_storage_len(set, binding)`.
/// * `Some((offset_sym, len_sym))` — a view param that was replaced in a
///   specialized helper function by two scalar u32 params. `offset` and
///   `len` are materialized as `Var` refs to these symbols.
///
/// `try_resolve_view_expr` + `materialize_offset_len` are the single place
/// that turns either shape into concrete TLC terms, so every downstream
/// use (indexing, length, slice, StorageBuffer conversion) is uniform.
#[derive(Debug, Clone)]
struct BufferBinding {
    set: u32,
    binding: u32,
    elem_ty: Type<TypeName>,
    scalar_params: Option<(SymbolId, SymbolId)>,
}

/// Resolved view info: an expression that represents a view into a buffer.
/// `offset` and `len` are TLC terms (may be Var refs to param symbols, or
/// computed expressions like `offset + start`).
struct ViewInfo {
    offset: Term,
    len: Term,
    set: u32,
    binding: u32,
    elem_ty: Type<TypeName>,
}

/// A specialization key: for each parameter position, `Some(binding)` if it's
/// a buffer-backed view, `None` otherwise.
type SpecKey = Vec<Option<(u32, u32)>>;

/// Buffer specializer state.
struct BufferSpecializer {
    symbols: SymbolTable,
    term_ids: TermIdSource,
    /// Known buffer params in current scope: symbol → BufferBinding
    buffer_map: HashMap<SymbolId, BufferBinding>,
    /// Cache: (original_def_sym, spec_key) → specialized_def_sym
    specializations: HashMap<(SymbolId, SpecKey), SymbolId>,
    /// Newly generated defs
    new_defs: Vec<Def>,
    /// All defs by symbol for lookup
    def_map: HashMap<SymbolId, Def>,
}

/// Check if a type is a view array (unsized, ArrayVariantView).
fn is_view_array(ty: &Type<TypeName>) -> bool {
    if !ty.is_array() {
        return false;
    }
    let is_view = ty
        .array_variant()
        .map(|v| matches!(v, Type::Constructed(TypeName::ArrayVariantView, _)))
        .unwrap_or(false);
    let is_unsized = ty.array_size().map(|s| matches!(s, Type::Variable(_))).unwrap_or(false);
    is_view && is_unsized
}

/// Extract element type from an array type.
fn array_elem_type(ty: &Type<TypeName>) -> Type<TypeName> {
    ty.elem_type().expect("array has elem type").clone()
}

/// Extract the parameters of a function by walking nested lambdas.
fn extract_params(term: &Term) -> (Vec<(SymbolId, Type<TypeName>)>, &Term) {
    match &term.kind {
        TermKind::Lambda(lam) => {
            let (mut inner_params, inner_body) = extract_params(&lam.body);
            let mut params: Vec<(SymbolId, Type<TypeName>)> = lam.params.clone();
            params.append(&mut inner_params);
            (params, inner_body)
        }
        _ => (vec![], term),
    }
}

/// Run buffer specialization on a TLC program.
pub fn run(program: Program) -> Program {
    let mut specializer = BufferSpecializer {
        symbols: program.symbols,
        term_ids: TermIdSource::new(),
        buffer_map: HashMap::new(),
        specializations: HashMap::new(),
        new_defs: Vec::new(),
        def_map: HashMap::new(),
    };

    // Build def_map
    for def in &program.defs {
        specializer.def_map.insert(def.name, def.clone());
    }

    // Process each entry point
    let mut processed_defs: Vec<Def> = Vec::new();
    for def in &program.defs {
        match &def.meta {
            DefMeta::EntryPoint(entry) => {
                if matches!(entry.entry_type, interface::Attribute::Compute) {
                    let processed = specializer.process_entry_point(def, entry);
                    processed_defs.push(processed);
                } else {
                    processed_defs.push(def.clone());
                }
            }
            DefMeta::Function => {
                processed_defs.push(def.clone());
            }
        }
    }

    // Append newly generated specialized defs
    processed_defs.extend(specializer.new_defs.drain(..));

    Program {
        defs: processed_defs,
        uniforms: program.uniforms,
        storage: program.storage,
        symbols: specializer.symbols,
        def_syms: program.def_syms,
    }
}

impl BufferSpecializer {
    /// Process a compute entry point: compute buffer bindings for its params,
    /// then walk the body rewriting calls to functions that receive buffer args.
    fn process_entry_point(&mut self, def: &Def, _entry: &interface::EntryDecl) -> Def {
        // Compute buffer bindings for entry params (same logic as to_ssa.rs)
        let (params, _inner_body) = extract_params(&def.body);
        let mut binding_num = 0u32;
        let old_buffer_map = self.buffer_map.clone();
        self.buffer_map.clear();

        for (sym, ty) in &params {
            if is_view_array(ty) {
                let elem_ty = array_elem_type(ty);
                self.buffer_map.insert(
                    *sym,
                    BufferBinding {
                        set: 0,
                        binding: binding_num,
                        elem_ty,
                        scalar_params: None,
                    },
                );
                binding_num += 1;
            }
        }

        // Rewrite the body
        let new_body = self.rewrite_term(&def.body);

        self.buffer_map = old_buffer_map;

        Def {
            body: new_body,
            ..def.clone()
        }
    }

    /// Rewrite a term, specializing function calls that pass buffer-backed args.
    fn rewrite_term(&mut self, term: &Term) -> Term {
        match &term.kind {
            TermKind::Lambda(lam) => {
                let new_body = self.rewrite_term(&lam.body);
                let new_captures: Vec<_> =
                    lam.captures.iter().map(|(s, t, e)| (*s, t.clone(), self.rewrite_term(e))).collect();
                Term {
                    kind: TermKind::Lambda(Lambda {
                        params: lam.params.clone(),
                        body: Box::new(new_body),
                        ret_ty: lam.ret_ty.clone(),
                        captures: new_captures,
                    }),
                    ..term.clone()
                }
            }

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                // Three cases, checked in order:
                //
                // 1. RHS is a view expression (slice or bare view ref) — bind
                //    fresh offset/len symbols, register `name` in buffer_map
                //    with `scalar_params: Some((offset_sym, len_sym))`, and
                //    wrap the let-in chain: `let offset_sym = … in let len_sym = … in <body>`.
                //    The original `let name = rhs in body` is *dropped* because
                //    `name`'s TLC type (a view array) no longer corresponds to
                //    any representable value after specialization.
                // 2. RHS is a bare `Var` naming an already-tracked buffer
                //    param — pure alias propagation into `buffer_map`. (Subset
                //    of case 1, handled by it, but kept visible for clarity.)
                // 3. Otherwise — ordinary let, recurse on both sides.
                if let Some(view) = self.try_resolve_view_expr(rhs) {
                    let span = term.span;
                    let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);
                    let name_str = self.symbols.get(*name).cloned().unwrap_or_default();
                    let offset_sym = self.symbols.alloc(format!("{}_offset", name_str));
                    let len_sym = self.symbols.alloc(format!("{}_len", name_str));

                    self.buffer_map.insert(
                        *name,
                        BufferBinding {
                            set: view.set,
                            binding: view.binding,
                            elem_ty: view.elem_ty.clone(),
                            scalar_params: Some((offset_sym, len_sym)),
                        },
                    );
                    let new_body = self.rewrite_term(body);
                    self.buffer_map.remove(name);

                    // Wrap new_body as: let offset_sym = view.offset in
                    //                   let len_sym = view.len in new_body
                    let inner = Term {
                        id: self.term_ids.next_id(),
                        ty: term.ty.clone(),
                        span,
                        kind: TermKind::Let {
                            name: len_sym,
                            name_ty: u32_ty.clone(),
                            rhs: Box::new(view.len),
                            body: Box::new(new_body),
                        },
                    };
                    return Term {
                        id: self.term_ids.next_id(),
                        ty: term.ty.clone(),
                        span,
                        kind: TermKind::Let {
                            name: offset_sym,
                            name_ty: u32_ty,
                            rhs: Box::new(view.offset),
                            body: Box::new(inner),
                        },
                    };
                }

                let new_rhs = self.rewrite_term(rhs);
                let new_body = self.rewrite_term(body);

                Term {
                    kind: TermKind::Let {
                        name: *name,
                        name_ty: name_ty.clone(),
                        rhs: Box::new(new_rhs),
                        body: Box::new(new_body),
                    },
                    ..term.clone()
                }
            }

            TermKind::App { func, args } => {
                match &func.kind {
                    TermKind::Var(sym) => {
                        let name = self.symbols.get(*sym).cloned().unwrap_or_default();

                        // `_w_index(arr_expr, i)` where arr_expr resolves to a
                        // view (bare Var of a buffer-backed param, or a
                        // `_w_intrinsic_slice(...)` chain): rewrite to
                        // `_w_intrinsic_storage_index(set, binding, offset + i)`
                        // so the indexed load goes through the storage view
                        // path rather than being lowered later as
                        // `materialize + dynamic_extract` on a runtime-sized
                        // buffer (which is invalid — runtime-sized storage
                        // arrays aren't copyable into a local composite).
                        if name == "_w_index" && args.len() == 2 {
                            if let Some(view) = self.try_resolve_view_expr(&args[0]) {
                                let span = term.span;
                                let u32_ty: Type<TypeName> =
                                    Type::Constructed(TypeName::UInt(32), vec![]);
                                let idx = self.rewrite_term(&args[1]);
                                let set_lit =
                                    self.make_int_lit(&view.set.to_string(), u32_ty.clone(), span);
                                let binding_lit = self.make_int_lit(
                                    &view.binding.to_string(),
                                    u32_ty.clone(),
                                    span,
                                );
                                let offset_plus_idx = self.make_binop_app(
                                    ast::BinaryOp { op: "+".to_string() },
                                    view.offset,
                                    idx,
                                    u32_ty.clone(),
                                    span,
                                );
                                return self.make_app(
                                    "_w_intrinsic_storage_index",
                                    vec![set_lit, binding_lit, offset_plus_idx],
                                    view.elem_ty,
                                    span,
                                );
                            }
                        }

                        // `_w_intrinsic_length(arr_expr)` on anything that
                        // resolves to a view — rewrite to the view's `len`
                        // (either `_w_intrinsic_storage_len(set, binding)` for
                        // entry params, or the scalar `Var(len_sym)` for
                        // specialized-helper params, or `end - start` for
                        // slices).
                        if name == "_w_intrinsic_length" && args.len() == 1 {
                            if let Some(view) = self.try_resolve_view_expr(&args[0]) {
                                return view.len;
                            }
                        }

                        // Check if any arguments are buffer-backed (clone to avoid borrow)
                        let arg_refs: Vec<&Term> = args.iter().collect();
                        let buffer_args: Vec<Option<BufferBinding>> = args
                            .iter()
                            .map(|a| {
                                if let TermKind::Var(s) = &a.kind {
                                    self.buffer_map.get(s).cloned()
                                } else {
                                    None
                                }
                            })
                            .collect();

                        let has_buffer_args = buffer_args.iter().any(|b| b.is_some());
                        if has_buffer_args {
                            if let Some(target_def) = self.def_map.get(sym).cloned() {
                                if matches!(target_def.meta, DefMeta::Function) {
                                    return self.specialize_call(
                                        *sym,
                                        &target_def,
                                        &arg_refs,
                                        &buffer_args,
                                        term,
                                    );
                                }
                            }
                        }

                        // No buffer args or not a known function — recurse normally
                        let new_func = self.rewrite_term(func);
                        let new_args: Vec<Term> = args.iter().map(|a| self.rewrite_term(a)).collect();
                        Term {
                            kind: TermKind::App {
                                func: Box::new(new_func),
                                args: new_args,
                            },
                            ..term.clone()
                        }
                    }
                    _ => {
                        let new_func = self.rewrite_term(func);
                        let new_args: Vec<Term> = args.iter().map(|a| self.rewrite_term(a)).collect();
                        Term {
                            kind: TermKind::App {
                                func: Box::new(new_func),
                                args: new_args,
                            },
                            ..term.clone()
                        }
                    }
                }
            }

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let new_cond = self.rewrite_term(cond);
                let new_then = self.rewrite_term(then_branch);
                let new_else = self.rewrite_term(else_branch);
                Term {
                    kind: TermKind::If {
                        cond: Box::new(new_cond),
                        then_branch: Box::new(new_then),
                        else_branch: Box::new(new_else),
                    },
                    ..term.clone()
                }
            }

            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                let new_init = self.rewrite_term(init);
                let new_bindings: Vec<_> =
                    init_bindings.iter().map(|(s, t, e)| (*s, t.clone(), self.rewrite_term(e))).collect();
                let new_kind = self.rewrite_loop_kind(kind);
                let new_body = self.rewrite_term(body);
                Term {
                    kind: TermKind::Loop {
                        loop_var: *loop_var,
                        loop_var_ty: loop_var_ty.clone(),
                        init: Box::new(new_init),
                        init_bindings: new_bindings,
                        kind: new_kind,
                        body: Box::new(new_body),
                    },
                    ..term.clone()
                }
            }

            TermKind::Soac(soac) => {
                let new_soac = self.rewrite_soac(soac);
                Term {
                    kind: TermKind::Soac(new_soac),
                    ..term.clone()
                }
            }

            TermKind::ArrayExpr(ae) => {
                let new_ae = self.rewrite_array_expr(ae);
                Term {
                    kind: TermKind::ArrayExpr(new_ae),
                    ..term.clone()
                }
            }

            TermKind::Force(inner) => {
                let new_inner = self.rewrite_term(inner);
                Term {
                    kind: TermKind::Force(Box::new(new_inner)),
                    ..term.clone()
                }
            }

            TermKind::Pack {
                exists_ty,
                dims,
                value,
            } => {
                let new_value = self.rewrite_term(value);
                Term {
                    kind: TermKind::Pack {
                        exists_ty: exists_ty.clone(),
                        dims: dims.clone(),
                        value: Box::new(new_value),
                    },
                    ..term.clone()
                }
            }

            TermKind::Unpack {
                scrut,
                dim_binders,
                value_binder,
                body,
            } => {
                let new_scrut = self.rewrite_term(scrut);
                let new_body = self.rewrite_term(body);
                Term {
                    kind: TermKind::Unpack {
                        scrut: Box::new(new_scrut),
                        dim_binders: dim_binders.clone(),
                        value_binder: *value_binder,
                        body: Box::new(new_body),
                    },
                    ..term.clone()
                }
            }

            // Leaves — no rewriting needed
            TermKind::Var(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::StringLit(_)
            | TermKind::Extern(_) => term.clone(),
        }
    }

    fn rewrite_loop_kind(&mut self, kind: &LoopKind) -> LoopKind {
        match kind {
            LoopKind::For { var, var_ty, iter } => LoopKind::For {
                var: *var,
                var_ty: var_ty.clone(),
                iter: Box::new(self.rewrite_term(iter)),
            },
            LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                var: *var,
                var_ty: var_ty.clone(),
                bound: Box::new(self.rewrite_term(bound)),
            },
            LoopKind::While { cond } => LoopKind::While {
                cond: Box::new(self.rewrite_term(cond)),
            },
        }
    }

    fn rewrite_soac(&mut self, soac: &SoacOp) -> SoacOp {
        match soac {
            SoacOp::Map { lam, inputs } => SoacOp::Map {
                lam: self.rewrite_lambda(lam),
                inputs: inputs.iter().map(|ae| self.rewrite_array_expr(ae)).collect(),
            },
            SoacOp::Reduce { op, ne, input, props } => SoacOp::Reduce {
                op: self.rewrite_lambda(op),
                ne: Box::new(self.rewrite_term(ne)),
                input: self.rewrite_array_expr(input),
                props: props.clone(),
            },
            SoacOp::Scan { op, ne, input } => SoacOp::Scan {
                op: self.rewrite_lambda(op),
                ne: Box::new(self.rewrite_term(ne)),
                input: self.rewrite_array_expr(input),
            },
            SoacOp::Filter { pred, input } => SoacOp::Filter {
                pred: self.rewrite_lambda(pred),
                input: self.rewrite_array_expr(input),
            },
            SoacOp::Scatter {
                dest,
                indices,
                values,
            } => SoacOp::Scatter {
                dest: self.rewrite_place(dest),
                indices: self.rewrite_array_expr(indices),
                values: self.rewrite_array_expr(values),
            },
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
                props,
            } => SoacOp::ReduceByIndex {
                dest: self.rewrite_place(dest),
                op: self.rewrite_lambda(op),
                ne: Box::new(self.rewrite_term(ne)),
                indices: self.rewrite_array_expr(indices),
                values: self.rewrite_array_expr(values),
                props: props.clone(),
            },
            SoacOp::Redomap {
                op,
                reduce_op,
                ne,
                inputs,
                props,
            } => SoacOp::Redomap {
                op: self.rewrite_lambda(op),
                reduce_op: self.rewrite_lambda(reduce_op),
                ne: Box::new(self.rewrite_term(ne)),
                inputs: inputs.iter().map(|ae| self.rewrite_array_expr(ae)).collect(),
                props: props.clone(),
            },
        }
    }

    fn rewrite_lambda(&mut self, lam: &Lambda) -> Lambda {
        Lambda {
            params: lam.params.clone(),
            body: Box::new(self.rewrite_term(&lam.body)),
            ret_ty: lam.ret_ty.clone(),
            captures: lam.captures.iter().map(|(s, t, e)| (*s, t.clone(), self.rewrite_term(e))).collect(),
        }
    }

    fn rewrite_array_expr(&mut self, ae: &ArrayExpr) -> ArrayExpr {
        match ae {
            ArrayExpr::Ref(t) => {
                // If this Ref resolves to a view (entry param, specialized
                // helper param, or `_w_intrinsic_slice(...)` chain), emit the
                // canonical `StorageBuffer` form so downstream passes (in
                // particular parallelize's Stage A) see a SOAC input shape
                // they can analyze.
                if let Some(view) = self.try_resolve_view_expr(t) {
                    return ArrayExpr::StorageBuffer {
                        set: view.set,
                        binding: view.binding,
                        offset: Box::new(view.offset),
                        len: Box::new(view.len),
                        elem_ty: view.elem_ty,
                    };
                }
                ArrayExpr::Ref(Box::new(self.rewrite_term(t)))
            }
            ArrayExpr::Zip(aes) => ArrayExpr::Zip(aes.iter().map(|a| self.rewrite_array_expr(a)).collect()),
            ArrayExpr::Soac(op) => ArrayExpr::Soac(Box::new(self.rewrite_soac(op))),
            ArrayExpr::Generate {
                shape,
                index_fn,
                elem_ty,
            } => ArrayExpr::Generate {
                shape: shape.clone(),
                index_fn: self.rewrite_lambda(index_fn),
                elem_ty: elem_ty.clone(),
            },
            ArrayExpr::Literal(terms) => {
                ArrayExpr::Literal(terms.iter().map(|t| self.rewrite_term(t)).collect())
            }
            ArrayExpr::Range { start, len } => ArrayExpr::Range {
                start: Box::new(self.rewrite_term(start)),
                len: Box::new(self.rewrite_term(len)),
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
                offset: Box::new(self.rewrite_term(offset)),
                len: Box::new(self.rewrite_term(len)),
                elem_ty: elem_ty.clone(),
            },
        }
    }

    fn rewrite_place(&mut self, place: &Place) -> Place {
        match place {
            Place::BufferSlice {
                base,
                offset,
                shape,
                elem_ty,
            } => Place::BufferSlice {
                base: Box::new(self.rewrite_term(base)),
                offset: Box::new(self.rewrite_term(offset)),
                shape: shape.clone(),
                elem_ty: elem_ty.clone(),
            },
            Place::LocalArray { id, shape, elem_ty } => Place::LocalArray {
                id: *id,
                shape: shape.clone(),
                elem_ty: elem_ty.clone(),
            },
        }
    }

    /// Create a specialized version of a function for specific buffer bindings.
    /// At the call site, replace `f(buf)` with `f_specialized(0, storage_len(set, binding))`.
    fn specialize_call(
        &mut self,
        func_sym: SymbolId,
        target_def: &Def,
        args: &[&Term],
        buffer_args: &[Option<BufferBinding>],
        original_term: &Term,
    ) -> Term {
        let span = original_term.span;

        // Build specialization key
        let spec_key: SpecKey =
            buffer_args.iter().map(|b| b.as_ref().map(|bb| (bb.set, bb.binding))).collect();

        // Check if we already have this specialization
        let spec_sym = if let Some(&sym) = self.specializations.get(&(func_sym, spec_key.clone())) {
            sym
        } else {
            // Create the specialized function
            let spec_sym = self.create_specialized_def(func_sym, target_def, buffer_args, span);
            self.specializations.insert((func_sym, spec_key), spec_sym);
            spec_sym
        };

        // Build the call site: replace buffer args with (offset=0, len=storage_len(set,binding))
        let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);
        let mut new_args: Vec<Term> = Vec::new();

        for (arg, buf) in args.iter().zip(buffer_args.iter()) {
            if let Some(binding) = buf {
                // Replace with (0, _w_intrinsic_storage_len(set, binding))
                let zero = self.make_int_lit("0", u32_ty.clone(), span);
                let set_lit = self.make_int_lit(&binding.set.to_string(), u32_ty.clone(), span);
                let binding_lit = self.make_int_lit(&binding.binding.to_string(), u32_ty.clone(), span);
                let storage_len = self.make_app(
                    "_w_intrinsic_storage_len",
                    vec![set_lit, binding_lit],
                    u32_ty.clone(),
                    span,
                );
                new_args.push(zero);
                new_args.push(storage_len);
            } else {
                new_args.push(self.rewrite_term(arg));
            }
        }

        // Build the application: spec_sym(new_args...)
        let func_ref = Term {
            id: self.term_ids.next_id(),
            ty: target_def.ty.clone(), // approximate — not critical
            span,
            kind: TermKind::Var(spec_sym),
        };

        if new_args.is_empty() {
            func_ref
        } else {
            Term {
                id: self.term_ids.next_id(),
                ty: original_term.ty.clone(),
                span,
                kind: TermKind::App {
                    func: Box::new(func_ref),
                    args: new_args,
                },
            }
        }
    }

    /// Create a specialized copy of a function where view params are replaced
    /// with (offset: u32, len: u32) and array ops use storage intrinsics.
    fn create_specialized_def(
        &mut self,
        _func_sym: SymbolId,
        target_def: &Def,
        buffer_args: &[Option<BufferBinding>],
        _span: Span,
    ) -> SymbolId {
        let orig_name = self.symbols.get(target_def.name).expect("BUG: symbol not in table").clone();

        // Build suffix from buffer bindings
        let suffix: String = buffer_args
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.as_ref().map(|bb| format!("_b{}s{}b{}", i, bb.set, bb.binding)))
            .collect();
        let spec_name = format!("{}{}", orig_name, suffix);
        let spec_sym = self.symbols.alloc(spec_name);

        // Extract the original function params and body
        let (orig_params, orig_body) = extract_params(&target_def.body);

        let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);

        // Build new params, and temporarily swap in a buffer_map that
        // describes the specialized function's view params (via
        // `scalar_params: Some(...)`). The unified `rewrite_term` uses the
        // same `self.buffer_map` for both direct-entry and specialized-body
        // contexts — the `scalar_params` field decides how offset/len
        // materialize.
        let mut new_params: Vec<(SymbolId, Type<TypeName>)> = Vec::new();
        let saved_buffer_map = std::mem::take(&mut self.buffer_map);

        for (i, (sym, ty)) in orig_params.iter().enumerate() {
            if let Some(Some(binding)) = buffer_args.get(i) {
                // Replace view param with offset + len
                let offset_sym = self.symbols.alloc(format!("{}_offset", self.symbols.get(*sym).unwrap()));
                let len_sym = self.symbols.alloc(format!("{}_len", self.symbols.get(*sym).unwrap()));
                new_params.push((offset_sym, u32_ty.clone()));
                new_params.push((len_sym, u32_ty.clone()));
                self.buffer_map.insert(
                    *sym,
                    BufferBinding {
                        set: binding.set,
                        binding: binding.binding,
                        elem_ty: binding.elem_ty.clone(),
                        scalar_params: Some((offset_sym, len_sym)),
                    },
                );
            } else {
                new_params.push((*sym, ty.clone()));
            }
        }

        // Rewrite the body. `rewrite_term` handles view lookups uniformly.
        let new_body = self.rewrite_term(orig_body);

        self.buffer_map = saved_buffer_map;

        // Wrap body in nested lambdas for the new params
        let final_body = wrap_in_lambdas(new_body, &new_params, &mut self.term_ids);

        let spec_def = Def {
            name: spec_sym,
            ty: target_def.ty.clone(), // TODO: could refine this
            body: final_body,
            meta: DefMeta::Function,
            arity: new_params.len(),
        };

        self.new_defs.push(spec_def);
        spec_sym
    }


    // =========================================================================
    // View expression resolution
    // =========================================================================

    /// Try to resolve an expression to a `ViewInfo` — i.e. determine whether
    /// it refers (directly or via slicing) to a known buffer-backed view.
    ///
    /// Returns `Some(ViewInfo)` for:
    /// - `Var(sym)` where sym is in `view_params`
    /// - `_w_intrinsic_slice(expr, start, end)` where expr resolves to a view
    fn try_resolve_view_expr(&mut self, term: &Term) -> Option<ViewInfo> {
        match &term.kind {
            TermKind::Var(sym) => {
                let binding = self.buffer_map.get(sym).cloned()?;
                let (offset, len) = self.materialize_offset_len(&binding, term.span);
                Some(ViewInfo {
                    offset,
                    len,
                    set: binding.set,
                    binding: binding.binding,
                    elem_ty: binding.elem_ty,
                })
            }
            TermKind::App { func, args } => {
                // Check for _w_intrinsic_slice(expr, start, end)
                if let TermKind::Var(sym) = &func.kind {
                    let name = self.symbols.get(*sym).cloned().unwrap_or_default();
                    if name == "_w_intrinsic_slice" && args.len() == 3 {
                        let parent = self.try_resolve_view_expr(&args[0])?;
                        let span = term.span;
                        let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);

                        // Rewrite start and end so any nested view refs are resolved
                        let start = self.rewrite_term(&args[1]);
                        let end = self.rewrite_term(&args[2]);

                        // new_offset = parent.offset + start
                        let new_offset = self.make_binop_app(
                            ast::BinaryOp { op: "+".to_string() },
                            parent.offset,
                            start.clone(),
                            u32_ty.clone(),
                            span,
                        );

                        // new_len = end - start
                        let new_len = self.make_binop_app(
                            ast::BinaryOp { op: "-".to_string() },
                            end,
                            start,
                            u32_ty,
                            span,
                        );

                        return Some(ViewInfo {
                            offset: new_offset,
                            len: new_len,
                            set: parent.set,
                            binding: parent.binding,
                            elem_ty: parent.elem_ty,
                        });
                    }
                }
                None
            }
            _ => None,
        }
    }

    // =========================================================================
    // Helper term constructors
    // =========================================================================

    /// Materialize the (offset, len) TLC terms for a view binding. Dispatches
    /// on `BufferBinding::scalar_params`: direct-entry bindings synthesize
    /// literal-0 and a `_w_intrinsic_storage_len(set, binding)` call;
    /// specialized-helper bindings materialize as `Var` refs to the scalar
    /// params the specializer allocated.
    fn materialize_offset_len(&mut self, b: &BufferBinding, span: Span) -> (Term, Term) {
        let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);
        match b.scalar_params {
            Some((offset_sym, len_sym)) => (
                Term {
                    id: self.term_ids.next_id(),
                    ty: u32_ty.clone(),
                    span,
                    kind: TermKind::Var(offset_sym),
                },
                Term {
                    id: self.term_ids.next_id(),
                    ty: u32_ty,
                    span,
                    kind: TermKind::Var(len_sym),
                },
            ),
            None => {
                let set_lit = self.make_int_lit(&b.set.to_string(), u32_ty.clone(), span);
                let binding_lit = self.make_int_lit(&b.binding.to_string(), u32_ty.clone(), span);
                let zero = self.make_int_lit("0", u32_ty.clone(), span);
                let len = self.make_app(
                    "_w_intrinsic_storage_len",
                    vec![set_lit, binding_lit],
                    u32_ty,
                    span,
                );
                (zero, len)
            }
        }
    }

    fn make_int_lit(&mut self, value: &str, ty: Type<TypeName>, span: Span) -> Term {
        Term {
            id: self.term_ids.next_id(),
            ty,
            span,
            kind: TermKind::IntLit(value.to_string()),
        }
    }

    fn make_app(
        &mut self,
        intrinsic_name: &str,
        args: Vec<Term>,
        ret_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let func_sym = self.symbols.alloc(intrinsic_name.to_string());
        let func_term = Term {
            id: self.term_ids.next_id(),
            ty: ret_ty.clone(), // approximate
            span,
            kind: TermKind::Var(func_sym),
        };

        if args.is_empty() {
            func_term
        } else {
            Term {
                id: self.term_ids.next_id(),
                ty: ret_ty,
                span,
                kind: TermKind::App {
                    func: Box::new(func_term),
                    args,
                },
            }
        }
    }

    fn make_binop_app(
        &mut self,
        op: ast::BinaryOp,
        lhs: Term,
        rhs: Term,
        ret_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let op_term = Term {
            id: self.term_ids.next_id(),
            ty: ret_ty.clone(),
            span,
            kind: TermKind::BinOp(op),
        };
        Term {
            id: self.term_ids.next_id(),
            ty: ret_ty,
            span,
            kind: TermKind::App {
                func: Box::new(op_term),
                args: vec![lhs, rhs],
            },
        }
    }
}

/// Wrap a body term in a single flat lambda for the given parameter list.
fn wrap_in_lambdas(body: Term, params: &[(SymbolId, Type<TypeName>)], term_ids: &mut TermIdSource) -> Term {
    if params.is_empty() {
        return body;
    }

    let ret_ty = body.ty.clone();
    let mut lam_ty = ret_ty.clone();
    for (_, ty) in params.iter().rev() {
        lam_ty = Type::Constructed(TypeName::Arrow, vec![ty.clone(), lam_ty]);
    }
    Term {
        id: term_ids.next_id(),
        ty: lam_ty,
        span: body.span,
        kind: TermKind::Lambda(Lambda {
            params: params.to_vec(),
            body: Box::new(body),
            ret_ty,
            captures: vec![],
        }),
    }
}
