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
    ArrayExpr, Def, DefMeta, Lambda, LoopKind, Place, Program, SoacOp, Term, TermIdSource,
    TermKind,
};
use crate::ast::{self, Span, TypeName};
use crate::types::TypeExt;
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use std::collections::HashMap;

/// Buffer binding info for a view-array parameter.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct BufferBinding {
    set: u32,
    binding: u32,
    elem_ty: Type<TypeName>,
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
    let is_unsized = ty
        .array_size()
        .map(|s| matches!(s, Type::Variable(_)))
        .unwrap_or(false);
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
pub fn buffer_specialize(program: Program) -> Program {
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
                if matches!(entry.entry_type, ast::Attribute::Compute) {
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
    }
}

impl BufferSpecializer {
    /// Process a compute entry point: compute buffer bindings for its params,
    /// then walk the body rewriting calls to functions that receive buffer args.
    fn process_entry_point(&mut self, def: &Def, _entry: &ast::EntryDecl) -> Def {
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
                let new_captures: Vec<_> = lam
                    .captures
                    .iter()
                    .map(|(s, t, e)| (*s, t.clone(), self.rewrite_term(e)))
                    .collect();
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
                let new_rhs = self.rewrite_term(rhs);

                // If the let-binding aliases a buffer-backed variable, propagate.
                let was_buffer = if let TermKind::Var(sym) = &rhs.kind {
                    if let Some(binding) = self.buffer_map.get(sym) {
                        self.buffer_map.insert(*name, binding.clone());
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                let new_body = self.rewrite_term(body);

                if was_buffer {
                    self.buffer_map.remove(name);
                }

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

            TermKind::App { func, arg } => {
                // Collect the full application spine to see if we're calling a
                // function with buffer-backed args.
                let (base, args) = collect_app_spine(term);
                match &base.kind {
                    TermKind::Var(sym) => {
                        // Check if any arguments are buffer-backed (clone to avoid borrow)
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
                                        &args,
                                        &buffer_args,
                                        term,
                                    );
                                }
                            }
                        }

                        // No buffer args or not a known function — recurse normally
                        let new_func = self.rewrite_term(func);
                        let new_arg = self.rewrite_term(arg);
                        Term {
                            kind: TermKind::App {
                                func: Box::new(new_func),
                                arg: Box::new(new_arg),
                            },
                            ..term.clone()
                        }
                    }
                    _ => {
                        let new_func = self.rewrite_term(func);
                        let new_arg = self.rewrite_term(arg);
                        Term {
                            kind: TermKind::App {
                                func: Box::new(new_func),
                                arg: Box::new(new_arg),
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
                let new_bindings: Vec<_> = init_bindings
                    .iter()
                    .map(|(s, t, e)| (*s, t.clone(), self.rewrite_term(e)))
                    .collect();
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

            TermKind::Pack { exists_ty, dims, value } => {
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
            SoacOp::Reduce {
                op,
                ne,
                input,
                props,
            } => SoacOp::Reduce {
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
        }
    }

    fn rewrite_lambda(&mut self, lam: &Lambda) -> Lambda {
        Lambda {
            params: lam.params.clone(),
            body: Box::new(self.rewrite_term(&lam.body)),
            ret_ty: lam.ret_ty.clone(),
            captures: lam
                .captures
                .iter()
                .map(|(s, t, e)| (*s, t.clone(), self.rewrite_term(e)))
                .collect(),
        }
    }

    fn rewrite_array_expr(&mut self, ae: &ArrayExpr) -> ArrayExpr {
        match ae {
            ArrayExpr::Ref(t) => ArrayExpr::Ref(Box::new(self.rewrite_term(t))),
            ArrayExpr::Zip(aes) => {
                ArrayExpr::Zip(aes.iter().map(|a| self.rewrite_array_expr(a)).collect())
            }
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
        let spec_key: SpecKey = buffer_args
            .iter()
            .map(|b| b.as_ref().map(|bb| (bb.set, bb.binding)))
            .collect();

        // Check if we already have this specialization
        let spec_sym = if let Some(&sym) = self.specializations.get(&(func_sym, spec_key.clone())) {
            sym
        } else {
            // Create the specialized function
            let spec_sym = self.create_specialized_def(func_sym, target_def, buffer_args, span);
            self.specializations
                .insert((func_sym, spec_key), spec_sym);
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
                let binding_lit =
                    self.make_int_lit(&binding.binding.to_string(), u32_ty.clone(), span);
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

        let mut result = func_ref;
        for arg in new_args {
            let app_ty = original_term.ty.clone(); // final app has return type
            result = Term {
                id: self.term_ids.next_id(),
                ty: app_ty,
                span,
                kind: TermKind::App {
                    func: Box::new(result),
                    arg: Box::new(arg),
                },
            };
        }

        result
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
        let orig_name = self
            .symbols
            .get(target_def.name)
            .expect("BUG: symbol not in table")
            .clone();

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

        // Build new params and a mapping from old view params to their buffer info
        let mut new_params: Vec<(SymbolId, Type<TypeName>)> = Vec::new();
        // Maps old param sym → (offset_sym, len_sym, set, binding, elem_ty)
        let mut view_param_map: HashMap<SymbolId, (SymbolId, SymbolId, u32, u32, Type<TypeName>)> =
            HashMap::new();

        for (i, (sym, ty)) in orig_params.iter().enumerate() {
            if let Some(Some(binding)) = buffer_args.get(i) {
                // Replace view param with offset + len
                let offset_sym = self.symbols.alloc(format!(
                    "{}_offset",
                    self.symbols.get(*sym).unwrap()
                ));
                let len_sym = self.symbols.alloc(format!(
                    "{}_len",
                    self.symbols.get(*sym).unwrap()
                ));
                new_params.push((offset_sym, u32_ty.clone()));
                new_params.push((len_sym, u32_ty.clone()));
                view_param_map.insert(
                    *sym,
                    (
                        offset_sym,
                        len_sym,
                        binding.set,
                        binding.binding,
                        binding.elem_ty.clone(),
                    ),
                );
            } else {
                new_params.push((*sym, ty.clone()));
            }
        }

        // Rewrite the body: replace _w_index(arr, i) and _w_intrinsic_length(arr)
        // where arr maps to a buffer param.
        let new_body = self.rewrite_specialized_body(orig_body, &view_param_map);

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

    /// Rewrite the body of a specialized function, replacing operations on
    /// view params with storage intrinsics.
    fn rewrite_specialized_body(
        &mut self,
        term: &Term,
        view_params: &HashMap<SymbolId, (SymbolId, SymbolId, u32, u32, Type<TypeName>)>,
    ) -> Term {
        match &term.kind {
            TermKind::App { func, arg } => {
                // Check if this is a full application spine we need to rewrite
                let (base, args) = collect_app_spine(term);
                match &base.kind {
                    TermKind::Var(sym) => {
                        let name = self.symbols.get(*sym).cloned().unwrap_or_default();

                        // _w_index(arr_expr, i) where arr_expr resolves to a view
                        if name == "_w_index" && args.len() == 2 {
                            if let Some(view) =
                                self.try_resolve_view_expr(args[0], view_params)
                            {
                                let span = term.span;
                                let u32_ty: Type<TypeName> =
                                    Type::Constructed(TypeName::UInt(32), vec![]);
                                let idx =
                                    self.rewrite_specialized_body(&args[1], view_params);
                                let set_lit =
                                    self.make_int_lit(&view.set.to_string(), u32_ty.clone(), span);
                                let binding_lit = self
                                    .make_int_lit(&view.binding.to_string(), u32_ty.clone(), span);
                                let add_result = self.make_binop_app(
                                    ast::BinaryOp { op: "+".to_string() },
                                    view.offset,
                                    idx,
                                    u32_ty.clone(),
                                    span,
                                );
                                return self.make_app(
                                    "_w_intrinsic_storage_index",
                                    vec![set_lit, binding_lit, add_result],
                                    view.elem_ty,
                                    span,
                                );
                            }
                        }

                        // _w_intrinsic_length(arr_expr) where arr_expr resolves to a view
                        if name == "_w_intrinsic_length" && args.len() == 1 {
                            if let Some(view) =
                                self.try_resolve_view_expr(args[0], view_params)
                            {
                                return view.len;
                            }
                        }

                        // _w_intrinsic_slice(arr_expr, start, end) where arr_expr resolves to a view
                        // The slice itself produces a view — it can only be consumed by
                        // _w_index or _w_intrinsic_length above. If it appears bare
                        // (e.g. passed to a function), we fall through to default recursion.
                        // But we should NOT recurse into the arr_expr of the slice, because
                        // that would hit the bare Var case and fail. Instead we leave it for
                        // the outer consumer (_w_index, _w_intrinsic_length, or Let) to resolve.

                        // Check for calls to functions that themselves take view args
                        // (recursive specialization)
                        let inner_buffer_args: Vec<Option<BufferBinding>> = args
                            .iter()
                            .map(|a| {
                                if let TermKind::Var(s) = &a.kind {
                                    if view_params.contains_key(s) {
                                        // This is a view param — build a BufferBinding for it
                                        let (_, _, set, binding, elem_ty) = &view_params[s];
                                        Some(BufferBinding {
                                            set: *set,
                                            binding: *binding,
                                            elem_ty: elem_ty.clone(),
                                        })
                                    } else {
                                        self.buffer_map.get(s).cloned()
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect();

                        let has_inner_buf = inner_buffer_args.iter().any(|b| b.is_some());
                        if has_inner_buf {
                            if let Some(target_def) = self.def_map.get(sym).cloned() {
                                if matches!(target_def.meta, DefMeta::Function) {
                                    let rewritten_args: Vec<Term> = args
                                        .iter()
                                        .map(|a| self.rewrite_specialized_body(a, view_params))
                                        .collect();
                                    let rewritten_refs: Vec<&Term> =
                                        rewritten_args.iter().collect();
                                    return self.specialize_call(
                                        *sym,
                                        &target_def,
                                        &rewritten_refs,
                                        &inner_buffer_args,
                                        term,
                                    );
                                }
                            }
                        }

                        // Default: recurse into func and arg
                        let new_func =
                            self.rewrite_specialized_body(func, view_params);
                        let new_arg =
                            self.rewrite_specialized_body(arg, view_params);
                        Term {
                            kind: TermKind::App {
                                func: Box::new(new_func),
                                arg: Box::new(new_arg),
                            },
                            ..term.clone()
                        }
                    }
                    _ => {
                        let new_func =
                            self.rewrite_specialized_body(func, view_params);
                        let new_arg =
                            self.rewrite_specialized_body(arg, view_params);
                        Term {
                            kind: TermKind::App {
                                func: Box::new(new_func),
                                arg: Box::new(new_arg),
                            },
                            ..term.clone()
                        }
                    }
                }
            }

            TermKind::Lambda(lam) => {
                let new_body = self.rewrite_specialized_body(&lam.body, view_params);
                let new_captures: Vec<_> = lam
                    .captures
                    .iter()
                    .map(|(s, t, e)| (*s, t.clone(), self.rewrite_specialized_body(e, view_params)))
                    .collect();
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
                // Check if the RHS is a view expression (e.g. a slice of a buffer param).
                // If so, bind fresh offset/len symbols and extend view_params for the body.
                if let Some(view) = self.try_resolve_view_expr(rhs, view_params) {
                    let span = term.span;
                    let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);

                    // Fresh symbols for the let-bound view's offset and len
                    let offset_sym = self.symbols.alloc(format!(
                        "{}_offset",
                        self.symbols.get(*name).cloned().unwrap_or_default()
                    ));
                    let len_sym = self.symbols.alloc(format!(
                        "{}_len",
                        self.symbols.get(*name).cloned().unwrap_or_default()
                    ));

                    // Extend view_params so the body knows `name` is a view
                    let mut extended = view_params.clone();
                    extended.insert(
                        *name,
                        (offset_sym, len_sym, view.set, view.binding, view.elem_ty.clone()),
                    );

                    let new_body = self.rewrite_specialized_body(body, &extended);

                    // Wrap body in let-bindings:
                    //   let offset_sym = <view.offset> in
                    //   let len_sym = <view.len> in
                    //   <new_body>
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

                let new_rhs = self.rewrite_specialized_body(rhs, view_params);
                let new_body = self.rewrite_specialized_body(body, view_params);
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

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let new_cond = self.rewrite_specialized_body(cond, view_params);
                let new_then = self.rewrite_specialized_body(then_branch, view_params);
                let new_else = self.rewrite_specialized_body(else_branch, view_params);
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
                let new_init = self.rewrite_specialized_body(init, view_params);
                let new_bindings: Vec<_> = init_bindings
                    .iter()
                    .map(|(s, t, e)| (*s, t.clone(), self.rewrite_specialized_body(e, view_params)))
                    .collect();
                let new_kind = self.rewrite_specialized_loop_kind(kind, view_params);
                let new_body = self.rewrite_specialized_body(body, view_params);
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
                let new_soac = self.rewrite_specialized_soac(soac, view_params);
                Term {
                    kind: TermKind::Soac(new_soac),
                    ..term.clone()
                }
            }

            TermKind::ArrayExpr(ae) => {
                let new_ae = self.rewrite_specialized_array_expr(ae, view_params);
                Term {
                    kind: TermKind::ArrayExpr(new_ae),
                    ..term.clone()
                }
            }

            TermKind::Force(inner) => {
                let new_inner = self.rewrite_specialized_body(inner, view_params);
                Term {
                    kind: TermKind::Force(Box::new(new_inner)),
                    ..term.clone()
                }
            }

            TermKind::Pack { exists_ty, dims, value } => {
                let new_value = self.rewrite_specialized_body(value, view_params);
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
                let new_scrut = self.rewrite_specialized_body(scrut, view_params);
                let new_body = self.rewrite_specialized_body(body, view_params);
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

            // Leaves
            TermKind::Var(_) => {
                // If this var refers to a view param that was replaced,
                // this is a bare reference (not in _w_index or _w_intrinsic_length).
                // This shouldn't happen in well-formed code since view arrays
                // are only used via index/length, but return it as-is.
                term.clone()
            }
            TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::StringLit(_)
            | TermKind::Extern(_) => term.clone(),
        }
    }

    fn rewrite_specialized_loop_kind(
        &mut self,
        kind: &LoopKind,
        view_params: &HashMap<SymbolId, (SymbolId, SymbolId, u32, u32, Type<TypeName>)>,
    ) -> LoopKind {
        match kind {
            LoopKind::For { var, var_ty, iter } => LoopKind::For {
                var: *var,
                var_ty: var_ty.clone(),
                iter: Box::new(self.rewrite_specialized_body(iter, view_params)),
            },
            LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                var: *var,
                var_ty: var_ty.clone(),
                bound: Box::new(self.rewrite_specialized_body(bound, view_params)),
            },
            LoopKind::While { cond } => LoopKind::While {
                cond: Box::new(self.rewrite_specialized_body(cond, view_params)),
            },
        }
    }

    fn rewrite_specialized_soac(
        &mut self,
        soac: &SoacOp,
        view_params: &HashMap<SymbolId, (SymbolId, SymbolId, u32, u32, Type<TypeName>)>,
    ) -> SoacOp {
        match soac {
            SoacOp::Map { lam, inputs } => SoacOp::Map {
                lam: self.rewrite_specialized_lambda(lam, view_params),
                inputs: inputs
                    .iter()
                    .map(|ae| self.rewrite_specialized_array_expr(ae, view_params))
                    .collect(),
            },
            SoacOp::Reduce {
                op,
                ne,
                input,
                props,
            } => SoacOp::Reduce {
                op: self.rewrite_specialized_lambda(op, view_params),
                ne: Box::new(self.rewrite_specialized_body(ne, view_params)),
                input: self.rewrite_specialized_array_expr(input, view_params),
                props: props.clone(),
            },
            SoacOp::Scan { op, ne, input } => SoacOp::Scan {
                op: self.rewrite_specialized_lambda(op, view_params),
                ne: Box::new(self.rewrite_specialized_body(ne, view_params)),
                input: self.rewrite_specialized_array_expr(input, view_params),
            },
            SoacOp::Filter { pred, input } => SoacOp::Filter {
                pred: self.rewrite_specialized_lambda(pred, view_params),
                input: self.rewrite_specialized_array_expr(input, view_params),
            },
            SoacOp::Scatter {
                dest,
                indices,
                values,
            } => SoacOp::Scatter {
                dest: self.rewrite_specialized_place(dest, view_params),
                indices: self.rewrite_specialized_array_expr(indices, view_params),
                values: self.rewrite_specialized_array_expr(values, view_params),
            },
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
                props,
            } => SoacOp::ReduceByIndex {
                dest: self.rewrite_specialized_place(dest, view_params),
                op: self.rewrite_specialized_lambda(op, view_params),
                ne: Box::new(self.rewrite_specialized_body(ne, view_params)),
                indices: self.rewrite_specialized_array_expr(indices, view_params),
                values: self.rewrite_specialized_array_expr(values, view_params),
                props: props.clone(),
            },
        }
    }

    fn rewrite_specialized_lambda(
        &mut self,
        lam: &Lambda,
        view_params: &HashMap<SymbolId, (SymbolId, SymbolId, u32, u32, Type<TypeName>)>,
    ) -> Lambda {
        Lambda {
            params: lam.params.clone(),
            body: Box::new(self.rewrite_specialized_body(&lam.body, view_params)),
            ret_ty: lam.ret_ty.clone(),
            captures: lam
                .captures
                .iter()
                .map(|(s, t, e)| {
                    (
                        *s,
                        t.clone(),
                        self.rewrite_specialized_body(e, view_params),
                    )
                })
                .collect(),
        }
    }

    fn rewrite_specialized_array_expr(
        &mut self,
        ae: &ArrayExpr,
        view_params: &HashMap<SymbolId, (SymbolId, SymbolId, u32, u32, Type<TypeName>)>,
    ) -> ArrayExpr {
        match ae {
            ArrayExpr::Ref(t) => {
                ArrayExpr::Ref(Box::new(self.rewrite_specialized_body(t, view_params)))
            }
            ArrayExpr::Zip(aes) => ArrayExpr::Zip(
                aes.iter()
                    .map(|a| self.rewrite_specialized_array_expr(a, view_params))
                    .collect(),
            ),
            ArrayExpr::Soac(op) => {
                ArrayExpr::Soac(Box::new(self.rewrite_specialized_soac(op, view_params)))
            }
            ArrayExpr::Generate {
                shape,
                index_fn,
                elem_ty,
            } => ArrayExpr::Generate {
                shape: shape.clone(),
                index_fn: self.rewrite_specialized_lambda(index_fn, view_params),
                elem_ty: elem_ty.clone(),
            },
            ArrayExpr::Literal(terms) => ArrayExpr::Literal(
                terms
                    .iter()
                    .map(|t| self.rewrite_specialized_body(t, view_params))
                    .collect(),
            ),
            ArrayExpr::Range { start, len } => ArrayExpr::Range {
                start: Box::new(self.rewrite_specialized_body(start, view_params)),
                len: Box::new(self.rewrite_specialized_body(len, view_params)),
            },
        }
    }

    fn rewrite_specialized_place(
        &mut self,
        place: &Place,
        view_params: &HashMap<SymbolId, (SymbolId, SymbolId, u32, u32, Type<TypeName>)>,
    ) -> Place {
        match place {
            Place::BufferSlice {
                base,
                offset,
                shape,
                elem_ty,
            } => Place::BufferSlice {
                base: Box::new(self.rewrite_specialized_body(base, view_params)),
                offset: Box::new(self.rewrite_specialized_body(offset, view_params)),
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

    // =========================================================================
    // View expression resolution
    // =========================================================================

    /// Try to resolve an expression to a `ViewInfo` — i.e. determine whether
    /// it refers (directly or via slicing) to a known buffer-backed view.
    ///
    /// Returns `Some(ViewInfo)` for:
    /// - `Var(sym)` where sym is in `view_params`
    /// - `_w_intrinsic_slice(expr, start, end)` where expr resolves to a view
    fn try_resolve_view_expr(
        &mut self,
        term: &Term,
        view_params: &HashMap<SymbolId, (SymbolId, SymbolId, u32, u32, Type<TypeName>)>,
    ) -> Option<ViewInfo> {
        match &term.kind {
            TermKind::Var(sym) => {
                let (offset_sym, len_sym, set, binding, elem_ty) = view_params.get(sym)?;
                let u32_ty: Type<TypeName> = Type::Constructed(TypeName::UInt(32), vec![]);
                Some(ViewInfo {
                    offset: Term {
                        id: self.term_ids.next_id(),
                        ty: u32_ty.clone(),
                        span: term.span,
                        kind: TermKind::Var(*offset_sym),
                    },
                    len: Term {
                        id: self.term_ids.next_id(),
                        ty: u32_ty,
                        span: term.span,
                        kind: TermKind::Var(*len_sym),
                    },
                    set: *set,
                    binding: *binding,
                    elem_ty: elem_ty.clone(),
                })
            }
            TermKind::App { .. } => {
                // Check for _w_intrinsic_slice(expr, start, end)
                let (base, args) = collect_app_spine(term);
                if let TermKind::Var(sym) = &base.kind {
                    let name = self.symbols.get(*sym).cloned().unwrap_or_default();
                    if name == "_w_intrinsic_slice" && args.len() == 3 {
                        let parent = self.try_resolve_view_expr(args[0], view_params)?;
                        let span = term.span;
                        let u32_ty: Type<TypeName> =
                            Type::Constructed(TypeName::UInt(32), vec![]);

                        // Rewrite start and end so any nested view refs are resolved
                        let start = self.rewrite_specialized_body(args[1], view_params);
                        let end = self.rewrite_specialized_body(args[2], view_params);

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
        let mut result = Term {
            id: self.term_ids.next_id(),
            ty: ret_ty.clone(), // approximate
            span,
            kind: TermKind::Var(func_sym),
        };

        for (i, arg) in args.into_iter().enumerate() {
            let app_ty = if i == 0 { ret_ty.clone() } else { ret_ty.clone() }; // approximate
            result = Term {
                id: self.term_ids.next_id(),
                ty: app_ty,
                span,
                kind: TermKind::App {
                    func: Box::new(result),
                    arg: Box::new(arg),
                },
            };
        }

        result
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
        let app1 = Term {
            id: self.term_ids.next_id(),
            ty: ret_ty.clone(),
            span,
            kind: TermKind::App {
                func: Box::new(op_term),
                arg: Box::new(lhs),
            },
        };
        Term {
            id: self.term_ids.next_id(),
            ty: ret_ty,
            span,
            kind: TermKind::App {
                func: Box::new(app1),
                arg: Box::new(rhs),
            },
        }
    }
}

/// Collect the full application spine: `f(a)(b)(c)` → `(f, [a, b, c])`.
fn collect_app_spine(term: &Term) -> (&Term, Vec<&Term>) {
    let mut args = Vec::new();
    let mut current = term;
    while let TermKind::App { func, arg } = &current.kind {
        args.push(arg.as_ref());
        current = func.as_ref();
    }
    args.reverse();
    (current, args)
}

/// Wrap a body term in nested lambdas for the given parameter list.
fn wrap_in_lambdas(body: Term, params: &[(SymbolId, Type<TypeName>)], term_ids: &mut TermIdSource) -> Term {
    if params.is_empty() {
        return body;
    }

    // Build from outermost to innermost (curried style: each lambda takes one param)
    let mut result = body;
    for (sym, ty) in params.iter().rev() {
        result = Term {
            id: term_ids.next_id(),
            ty: Type::Constructed(
                TypeName::Arrow,
                vec![ty.clone(), result.ty.clone()],
            ),
            span: result.span,
            kind: TermKind::Lambda(Lambda {
                params: vec![(*sym, ty.clone())],
                body: Box::new(result),
                ret_ty: Type::Constructed(TypeName::Unit, vec![]), // will be overridden
                captures: vec![],
            }),
        };
    }

    result
}
