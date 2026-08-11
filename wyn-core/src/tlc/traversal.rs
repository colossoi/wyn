//! Generic traversal and reconstruction for TLC terms.

use super::*;

// =============================================================================
// Generic child traversal
// =============================================================================

impl<C: Payload, S: Payload> Term<C, S> {
    /// Walk this term and its descendants in preorder.
    pub fn walk<V>(&self, visitor: &mut V)
    where
        V: TermVisitor<C, S>,
    {
        visitor.walk(self);
    }

    /// Rewrite this owned term while retaining its existing child allocations.
    pub fn rewrite<R>(self, rewriter: &mut R) -> Self
    where
        R: TermRewriter<C, S>,
    {
        rewriter.rewrite(self)
    }

    /// Consume and rebuild this term with a rewriter's owned-node hooks.
    pub fn rewrite_owned<R>(self, rewriter: &mut R) -> Self
    where
        R: TermRewriter<C, S>,
    {
        rewriter.rewrite_owned(self)
    }

    /// Rewrite every type stored in this term tree.
    ///
    /// This includes the term result types and the types stored in binders,
    /// coercions, loop metadata, SOAC capture ABIs, destinations, and named
    /// array-input atoms. Passes provide only the type-local operation.
    ///
    /// Every visited term receives a fresh ID because its stored type may
    /// change. This deliberately errs on the side of invalidating IDs rather
    /// than asking each caller to determine which type rewrites were no-ops.
    pub fn rewrite_types<M>(&mut self, term_ids: &mut TermIdSource, map: &mut M)
    where
        M: FnMut(&Type<TypeName>) -> Type<TypeName>,
    {
        self.ty = map(&self.ty);
        match &mut self.kind {
            TermKind::Lambda(lambda) => rewrite_lambda_types(lambda, map),
            TermKind::Let { name_ty, .. } => *name_ty = map(name_ty),
            TermKind::Coerce { target_ty, .. } => *target_ty = map(target_ty),
            TermKind::Loop {
                loop_var_ty,
                init_bindings,
                kind,
                ..
            } => {
                *loop_var_ty = map(loop_var_ty);
                for (_, ty, _) in init_bindings {
                    *ty = map(ty);
                }
                match kind {
                    LoopKind::For { var_ty, .. } | LoopKind::ForRange { var_ty, .. } => {
                        *var_ty = map(var_ty);
                    }
                    LoopKind::While { .. } => {}
                }
            }
            TermKind::Soac(soac) => rewrite_soac_types(soac, map),
            TermKind::ArrayExpr(array) => rewrite_array_expr_types(array, map),
            _ => {}
        }
        self.for_each_child_mut(&mut |child| child.rewrite_types(term_ids, map));
        self.id = term_ids.next_id();
    }

    /// If this term is `App { func: Var(sym), args }` — a direct named
    /// call — return `Some((sym, args))`. Returns `None` for operator
    /// dispatch (`App { BinOp/UnOp/Extern, .. }`), partial applications,
    /// non-call terms, etc.
    ///
    /// Use this in post-defunctionalize passes and backends to one-step
    /// destructure named calls instead of nesting two `match`es.
    pub fn as_direct_call(&self) -> Option<(SymbolId, &[Term<C, S>])> {
        match &self.kind {
            TermKind::App { func, args } => match &func.kind {
                TermKind::Var(VarRef::Symbol(sym)) => Some((*sym, args.as_slice())),
                _ => None,
            },
            _ => None,
        }
    }

    /// Assert that no App node in this tree has a func that is itself an App.
    pub fn assert_flat_apps(&self) {
        self.assert_flat_apps_in("<unknown>");
    }

    pub(super) fn assert_flat_apps_in(&self, def_name: &str) {
        if let TermKind::App { func, args } = &self.kind {
            if let TermKind::App { args: inner_args, .. } = &func.kind {
                panic!(
                    "Nested App detected in def '{}': outer has {} args, inner func has {} args. \
                     Inner func kind: {:?}",
                    def_name,
                    args.len(),
                    inner_args.len(),
                    if let TermKind::App { func: f, .. } = &func.kind {
                        format!("App(func={:?})", std::mem::discriminant(&f.kind))
                    } else {
                        format!("{:?}", std::mem::discriminant(&func.kind))
                    }
                );
            }
        }
        self.for_each_child(&mut |child| child.assert_flat_apps_in(def_name));
    }

    /// Apply `f` to every immediate `Term` child, returning a rebuilt `Term`
    /// with the caller-provided fresh ID. Recurses into Lambda, SoacOp,
    /// ArrayExpr, LoopKind, and Place sub-structures.
    ///
    /// This is the single place that knows the shape of TermKind — passes
    /// that need a uniform bottom-up or top-down walk can use this instead
    /// of hand-rolling a match over every variant.
    pub fn map_children<F>(self, fresh_id: TermId, f: &mut F) -> Self
    where
        F: FnMut(Term<C, S>) -> Term<C, S>,
    {
        let kind = match self.kind {
            // Leaves — no Term children
            TermKind::Var(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::Extern(_) => self.kind,

            TermKind::Closure(data) => TermKind::Closure(C::map(data, f)),

            TermKind::Coerce { inner, target_ty } => TermKind::Coerce {
                inner: Box::new(f(*inner)),
                target_ty,
            },

            TermKind::App { func, args } => TermKind::App {
                func: Box::new(f(*func)),
                args: args.into_iter().map(&mut *f).collect(),
            },

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => TermKind::Let {
                name,
                name_ty,
                rhs: Box::new(f(*rhs)),
                body: Box::new(f(*body)),
            },

            TermKind::Lambda(lam) => TermKind::Lambda(map_lambda_children(lam, f)),

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => TermKind::If {
                cond: Box::new(f(*cond)),
                then_branch: Box::new(f(*then_branch)),
                else_branch: Box::new(f(*else_branch)),
            },

            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => TermKind::Loop {
                loop_var,
                loop_var_ty,
                init: Box::new(f(*init)),
                init_bindings: init_bindings.into_iter().map(|(s, t, e)| (s, t, f(e))).collect(),
                kind: map_loop_kind_children(kind, f),
                body: Box::new(f(*body)),
            },

            TermKind::Soac(soac) => TermKind::Soac(map_soac_children(soac, f)),

            TermKind::ArrayExpr(ae) => TermKind::ArrayExpr(map_array_expr_children(ae, f)),

            TermKind::Tuple(parts) => TermKind::Tuple(parts.into_iter().map(&mut *f).collect()),

            TermKind::TupleProj { tuple, idx } => TermKind::TupleProj {
                tuple: Box::new(f(*tuple)),
                idx,
            },

            TermKind::Index { array, index } => TermKind::Index {
                array: Box::new(f(*array)),
                index: Box::new(f(*index)),
            },

            TermKind::VecLit(parts) => TermKind::VecLit(parts.into_iter().map(&mut *f).collect()),
        };

        Term {
            id: fresh_id,
            kind,
            ..self
        }
    }

    /// Visit every immediate `Term` child by reference. This is the by-ref
    /// counterpart to `map_children` — use it for analysis passes that
    /// inspect without transforming.
    pub fn for_each_child<F>(&self, f: &mut F)
    where
        F: FnMut(&Term<C, S>),
    {
        match &self.kind {
            TermKind::Var(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::Extern(_) => {}

            TermKind::Closure(data) => C::for_each(data, f),

            TermKind::Coerce { inner, .. } => f(inner),

            TermKind::App { func, args } => {
                f(func);
                for a in args {
                    f(a);
                }
            }

            TermKind::Let { rhs, body, .. } => {
                f(rhs);
                f(body);
            }

            TermKind::Lambda(lam) => visit_lambda_children(lam, f),

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                f(cond);
                f(then_branch);
                f(else_branch);
            }

            TermKind::Loop {
                init,
                init_bindings,
                kind,
                body,
                ..
            } => {
                f(init);
                for (_, _, e) in init_bindings {
                    f(e);
                }
                visit_loop_kind_children(kind, f);
                f(body);
            }

            TermKind::Soac(soac) => visit_soac_children(soac, f),
            TermKind::ArrayExpr(ae) => visit_array_expr_children(ae, f),

            TermKind::Tuple(parts) | TermKind::VecLit(parts) => {
                for p in parts {
                    f(p);
                }
            }
            TermKind::TupleProj { tuple, .. } => f(tuple),
            TermKind::Index { array, index } => {
                f(array);
                f(index);
            }
        }
    }

    /// Visit every immediate `Term` child by mutable reference — the in-place
    /// counterpart to `map_children`. The method itself writes nothing; it
    /// hands each child out as `&mut Term` so the callback can rewrite (or
    /// wholesale replace) children without rebuilding the tree.
    pub fn for_each_child_mut<F>(&mut self, f: &mut F)
    where
        F: FnMut(&mut Term<C, S>),
    {
        match &mut self.kind {
            TermKind::Var(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::Extern(_) => {}

            TermKind::Closure(data) => C::for_each_mut(data, f),

            TermKind::Coerce { inner, .. } => f(inner),

            TermKind::App { func, args } => {
                f(func);
                for a in args {
                    f(a);
                }
            }

            TermKind::Let { rhs, body, .. } => {
                f(rhs);
                f(body);
            }

            TermKind::Lambda(lam) => visit_lambda_children_mut(lam, f),

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                f(cond);
                f(then_branch);
                f(else_branch);
            }

            TermKind::Loop {
                init,
                init_bindings,
                kind,
                body,
                ..
            } => {
                f(init);
                for (_, _, e) in init_bindings {
                    f(e);
                }
                visit_loop_kind_children_mut(kind, f);
                f(body);
            }

            TermKind::Soac(soac) => visit_soac_children_mut(soac, f),
            TermKind::ArrayExpr(ae) => visit_array_expr_children_mut(ae, f),

            TermKind::Tuple(parts) | TermKind::VecLit(parts) => {
                for p in parts {
                    f(p);
                }
            }
            TermKind::TupleProj { tuple, .. } => f(tuple),
            TermKind::Index { array, index } => {
                f(array);
                f(index);
            }
        }
    }
}

fn rewrite_lambda_types<C, S, M>(lambda: &mut Lambda<C, S>, map: &mut M)
where
    C: Payload,
    S: Payload,
    M: FnMut(&Type<TypeName>) -> Type<TypeName>,
{
    for (_, ty) in &mut lambda.params {
        *ty = map(ty);
    }
    lambda.ret_ty = map(&lambda.ret_ty);
}

fn rewrite_soac_body_types<C, S, M>(body: &mut SoacBody<C, S>, map: &mut M)
where
    C: Payload,
    S: Payload,
    M: FnMut(&Type<TypeName>) -> Type<TypeName>,
{
    rewrite_lambda_types(&mut body.lam, map);
    S::for_each_mut(&mut body.data, &mut |(_, ty, _)| *ty = map(ty));
}

fn rewrite_soac_types<C, S, M>(soac: &mut SoacOp<C, S>, map: &mut M)
where
    C: Payload,
    S: Payload,
    M: FnMut(&Type<TypeName>) -> Type<TypeName>,
{
    match soac {
        SoacOp::Map { lam, inputs, .. } => {
            rewrite_soac_body_types(lam, map);
            for input in inputs {
                rewrite_array_expr_types(input, map);
            }
        }
        SoacOp::Reduce { op, input, .. } | SoacOp::Scan { op, input, .. } => {
            rewrite_soac_body_types(op, map);
            rewrite_array_expr_types(input, map);
        }
        SoacOp::Filter { pred, input, .. } => {
            rewrite_soac_body_types(pred, map);
            rewrite_array_expr_types(input, map);
        }
        SoacOp::Scatter {
            dest, lam, inputs, ..
        } => {
            dest.elem_ty = map(&dest.elem_ty);
            rewrite_soac_body_types(lam, map);
            for input in inputs {
                rewrite_array_expr_types(input, map);
            }
        }
        SoacOp::BucketScatter {
            dest, lam, inputs, ..
        } => {
            dest.elem_ty = map(&dest.elem_ty);
            rewrite_soac_body_types(lam, map);
            for input in inputs {
                rewrite_array_expr_types(input, map);
            }
        }
        SoacOp::ReduceByIndex {
            dest,
            op,
            indices,
            values,
            ..
        } => {
            dest.elem_ty = map(&dest.elem_ty);
            rewrite_soac_body_types(op, map);
            rewrite_array_expr_types(indices, map);
            rewrite_array_expr_types(values, map);
        }
    }
}

fn rewrite_array_expr_types<C, S, M>(array: &mut ArrayExpr<C, S>, map: &mut M)
where
    C: Payload,
    S: Payload,
    M: FnMut(&Type<TypeName>) -> Type<TypeName>,
{
    match array {
        ArrayExpr::Var(_, ty) => *ty = map(ty),
        ArrayExpr::Zip(parts) => {
            for part in parts {
                rewrite_array_expr_types(part, map);
            }
        }
        ArrayExpr::Literal(_) | ArrayExpr::Range { .. } => {}
    }
}

fn visit_lambda_children<C, S, V>(lam: &Lambda<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&Term<C, S>),
{
    f(&lam.body);
}

fn visit_soac_body_children<C, S, V>(sb: &SoacBody<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&Term<C, S>),
{
    visit_lambda_children(&sb.lam, f);
    S::for_each(&sb.data, &mut |capture| f(&capture.2));
}

fn visit_soac_children<C, S, V>(soac: &SoacOp<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&Term<C, S>),
{
    match soac {
        SoacOp::Map { lam, inputs, .. } => {
            visit_soac_body_children(lam, f);
            for ae in inputs {
                visit_array_expr_children(ae, f);
            }
        }
        SoacOp::Reduce { op, ne, input, .. } => {
            visit_soac_body_children(op, f);
            f(ne);
            visit_array_expr_children(input, f);
        }
        SoacOp::Scan { op, ne, input, .. } => {
            visit_soac_body_children(op, f);
            f(ne);
            visit_array_expr_children(input, f);
        }
        SoacOp::Filter { pred, input, .. } => {
            visit_soac_body_children(pred, f);
            visit_array_expr_children(input, f);
        }
        SoacOp::Scatter { lam, inputs, .. } => {
            visit_soac_body_children(lam, f);
            for input in inputs {
                visit_array_expr_children(input, f);
            }
        }
        SoacOp::BucketScatter { lam, inputs, .. } => {
            visit_soac_body_children(lam, f);
            for input in inputs {
                visit_array_expr_children(input, f);
            }
        }
        SoacOp::ReduceByIndex {
            op,
            ne,
            indices,
            values,
            ..
        } => {
            visit_soac_body_children(op, f);
            f(ne);
            visit_array_expr_children(indices, f);
            visit_array_expr_children(values, f);
        }
    }
}

fn visit_array_expr_children<C, S, V>(ae: &ArrayExpr<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&Term<C, S>),
{
    match ae {
        // Visit the named input as a var term, so analyses (free-var / capture
        // collection, etc.) see the reference.
        ArrayExpr::Var(vr, ty) => f(&synthetic_atom_var_term(*vr, ty.clone())),
        ArrayExpr::Zip(aes) => {
            for ae in aes {
                visit_array_expr_children(ae, f);
            }
        }
        ArrayExpr::Literal(terms) => {
            for t in terms {
                f(t);
            }
        }
        ArrayExpr::Range { start, len, step } => {
            f(start);
            f(len);
            if let Some(s) = step {
                f(s);
            }
        }
    }
}

fn visit_loop_kind_children<C, S, V>(kind: &LoopKind<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&Term<C, S>),
{
    match kind {
        LoopKind::For { iter, .. } => f(iter),
        LoopKind::ForRange { bound, .. } => f(bound),
        LoopKind::While { cond } => f(cond),
    }
}

fn visit_lambda_children_mut<C, S, V>(lam: &mut Lambda<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&mut Term<C, S>),
{
    f(&mut lam.body);
}

fn visit_soac_body_children_mut<C, S, V>(sb: &mut SoacBody<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&mut Term<C, S>),
{
    visit_lambda_children_mut(&mut sb.lam, f);
    S::for_each_mut(&mut sb.data, &mut |capture| f(&mut capture.2));
}

fn visit_soac_children_mut<C, S, V>(soac: &mut SoacOp<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&mut Term<C, S>),
{
    match soac {
        SoacOp::Map { lam, inputs, .. } => {
            visit_soac_body_children_mut(lam, f);
            for ae in inputs {
                visit_array_expr_children_mut(ae, f);
            }
        }
        SoacOp::Reduce { op, ne, input, .. } => {
            visit_soac_body_children_mut(op, f);
            f(ne);
            visit_array_expr_children_mut(input, f);
        }
        SoacOp::Scan { op, ne, input, .. } => {
            visit_soac_body_children_mut(op, f);
            f(ne);
            visit_array_expr_children_mut(input, f);
        }
        SoacOp::Filter { pred, input, .. } => {
            visit_soac_body_children_mut(pred, f);
            visit_array_expr_children_mut(input, f);
        }
        SoacOp::Scatter { lam, inputs, .. } => {
            visit_soac_body_children_mut(lam, f);
            for input in inputs {
                visit_array_expr_children_mut(input, f);
            }
        }
        SoacOp::BucketScatter { lam, inputs, .. } => {
            visit_soac_body_children_mut(lam, f);
            for input in inputs {
                visit_array_expr_children_mut(input, f);
            }
        }
        SoacOp::ReduceByIndex {
            op,
            ne,
            indices,
            values,
            ..
        } => {
            visit_soac_body_children_mut(op, f);
            f(ne);
            visit_array_expr_children_mut(indices, f);
            visit_array_expr_children_mut(values, f);
        }
    }
}

fn visit_array_expr_children_mut<C, S, V>(ae: &mut ArrayExpr<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&mut Term<C, S>),
{
    match ae {
        // Feed the named input through a reconstructed var term (as
        // `map_array_expr_children` does), so rewrites that rename or replace
        // a variable reach SOAC inputs, then re-atomize the result.
        ArrayExpr::Var(vr, ty) => {
            let mut tmp = synthetic_atom_var_term(*vr, ty.clone());
            f(&mut tmp);
            *ae = term_as_input_atom(tmp);
        }
        ArrayExpr::Zip(aes) => {
            for ae in aes {
                visit_array_expr_children_mut(ae, f);
            }
        }
        ArrayExpr::Literal(terms) => {
            for t in terms {
                f(t);
            }
        }
        ArrayExpr::Range { start, len, step } => {
            f(start);
            f(len);
            if let Some(s) = step {
                f(s);
            }
        }
    }
}

fn visit_loop_kind_children_mut<C, S, V>(kind: &mut LoopKind<C, S>, f: &mut V)
where
    C: Payload,
    S: Payload,
    V: FnMut(&mut Term<C, S>),
{
    match kind {
        LoopKind::For { iter, .. } => f(iter),
        LoopKind::ForRange { bound, .. } => f(bound),
        LoopKind::While { cond } => f(cond),
    }
}

fn map_lambda_children<C, S, M>(lam: Lambda<C, S>, f: &mut M) -> Lambda<C, S>
where
    C: Payload,
    S: Payload,
    M: FnMut(Term<C, S>) -> Term<C, S>,
{
    Lambda {
        body: Box::new(f(*lam.body)),
        ..lam
    }
}

fn map_soac_body_children<C, S, M>(sb: SoacBody<C, S>, f: &mut M) -> SoacBody<C, S>
where
    C: Payload,
    S: Payload,
    M: FnMut(Term<C, S>) -> Term<C, S>,
{
    SoacBody {
        lam: map_lambda_children(sb.lam, f),
        data: S::map(sb.data, &mut |(symbol, ty, term)| (symbol, ty, f(term))),
    }
}

fn map_soac_children<C, S, M>(soac: SoacOp<C, S>, f: &mut M) -> SoacOp<C, S>
where
    C: Payload,
    S: Payload,
    M: FnMut(Term<C, S>) -> Term<C, S>,
{
    match soac {
        SoacOp::Map {
            lam,
            inputs,
            destination,
        } => SoacOp::Map {
            lam: map_soac_body_children(lam, f),
            inputs: inputs.into_iter().map(|ae| map_array_expr_children(ae, f)).collect(),
            destination,
        },
        SoacOp::Reduce { op, ne, input } => SoacOp::Reduce {
            op: map_soac_body_children(op, f),
            ne: Box::new(f(*ne)),
            input: map_array_expr_children(input, f),
        },
        SoacOp::Scan {
            op,
            ne,
            input,
            destination,
        } => SoacOp::Scan {
            op: map_soac_body_children(op, f),
            ne: Box::new(f(*ne)),
            input: map_array_expr_children(input, f),
            destination,
        },
        SoacOp::Filter {
            pred,
            input,
            destination,
        } => SoacOp::Filter {
            pred: map_soac_body_children(pred, f),
            input: map_array_expr_children(input, f),
            destination,
        },
        SoacOp::Scatter { dest, lam, inputs } => SoacOp::Scatter {
            dest,
            lam: map_soac_body_children(lam, f),
            inputs: inputs.into_iter().map(|ae| map_array_expr_children(ae, f)).collect(),
        },
        SoacOp::BucketScatter {
            dest,
            lam,
            inputs,
            input_dimensions,
            domain_rank,
        } => SoacOp::BucketScatter {
            dest,
            lam: map_soac_body_children(lam, f),
            inputs: inputs.into_iter().map(|input| map_array_expr_children(input, f)).collect(),
            input_dimensions,
            domain_rank,
        },
        SoacOp::ReduceByIndex {
            dest,
            op,
            ne,
            indices,
            values,
        } => SoacOp::ReduceByIndex {
            dest,
            op: map_soac_body_children(op, f),
            ne: Box::new(f(*ne)),
            indices: map_array_expr_children(indices, f),
            values: map_array_expr_children(values, f),
        },
    }
}

fn map_array_expr_children<C, S, M>(ae: ArrayExpr<C, S>, f: &mut M) -> ArrayExpr<C, S>
where
    C: Payload,
    S: Payload,
    M: FnMut(Term<C, S>) -> Term<C, S>,
{
    match ae {
        // Apply `f` to the named input through a reconstructed var term, so
        // substitutions that rename (or inline) a variable reach SOAC inputs,
        // then re-atomize the result.
        ArrayExpr::Var(vr, ty) => term_as_input_atom(f(synthetic_atom_var_term(vr, ty))),
        ArrayExpr::Zip(aes) => {
            ArrayExpr::Zip(aes.into_iter().map(|ae| map_array_expr_children(ae, f)).collect())
        }
        ArrayExpr::Literal(terms) => ArrayExpr::Literal(terms.into_iter().map(f).collect()),
        ArrayExpr::Range { start, len, step } => ArrayExpr::Range {
            start: Box::new(f(*start)),
            len: Box::new(f(*len)),
            step: step.map(|s| Box::new(f(*s))),
        },
    }
}

fn map_loop_kind_children<C, S, M>(kind: LoopKind<C, S>, f: &mut M) -> LoopKind<C, S>
where
    C: Payload,
    S: Payload,
    M: FnMut(Term<C, S>) -> Term<C, S>,
{
    match kind {
        LoopKind::For { var, var_ty, iter } => LoopKind::For {
            var,
            var_ty,
            iter: Box::new(f(*iter)),
        },
        LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
            var,
            var_ty,
            bound: Box::new(f(*bound)),
        },
        LoopKind::While { cond } => LoopKind::While {
            cond: Box::new(f(*cond)),
        },
    }
}
