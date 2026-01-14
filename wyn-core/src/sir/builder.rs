//! Builder for ergonomic SIR construction.
//!
//! Provides a fluent API for building SIR programs with automatic
//! ID allocation and type tracking.

use crate::ast::Span;

use super::{
    types::SizeVar, Body, Exp, Lambda, LambdaId, Map, Param, Pat, PatElem, Prim, Reduce, Scan,
    SirType, Size, Soac, Stm, StmId, VarId,
};

/// Builder for constructing SIR programs.
#[derive(Debug, Default)]
pub struct SirBuilder {
    next_var: u32,
    next_stm: u32,
    next_lam: u32,
    next_size: u32,
}

impl SirBuilder {
    pub fn new() -> Self {
        SirBuilder::default()
    }

    /// Allocate a fresh variable ID.
    pub fn fresh_var(&mut self) -> VarId {
        let v = VarId(self.next_var);
        self.next_var += 1;
        v
    }

    /// Allocate a fresh statement ID.
    pub fn fresh_stm(&mut self) -> StmId {
        let s = StmId(self.next_stm);
        self.next_stm += 1;
        s
    }

    /// Allocate a fresh lambda ID.
    pub fn fresh_lambda(&mut self) -> LambdaId {
        let l = LambdaId(self.next_lam);
        self.next_lam += 1;
        l
    }

    /// Allocate a fresh size variable.
    pub fn fresh_size_var(&mut self) -> SizeVar {
        let s = SizeVar(self.next_size);
        self.next_size += 1;
        s
    }

    /// Create a single-variable pattern.
    pub fn pat1(&mut self, name: impl Into<String>, ty: SirType) -> Pat {
        let v = self.fresh_var();
        Pat {
            binds: vec![PatElem {
                var: v,
                ty,
                name_hint: name.into(),
            }],
        }
    }

    /// Create a pattern with a specific variable (doesn't allocate new var).
    pub fn pat_with_var(&self, var: VarId, name: impl Into<String>, ty: SirType) -> Pat {
        Pat {
            binds: vec![PatElem {
                var,
                ty,
                name_hint: name.into(),
            }],
        }
    }

    /// Create a tuple pattern.
    pub fn pat_tuple(&mut self, elems: Vec<(String, SirType)>) -> Pat {
        Pat {
            binds: elems
                .into_iter()
                .map(|(name, ty)| PatElem {
                    var: self.fresh_var(),
                    ty,
                    name_hint: name,
                })
                .collect(),
        }
    }

    /// Create a statement binding a single expression.
    pub fn stm(&mut self, pat: Pat, exp: Exp, ty: SirType, span: Span) -> Stm {
        Stm {
            id: self.fresh_stm(),
            pat,
            exp,
            ty,
            span,
        }
    }

    /// Create a parameter.
    pub fn param(&mut self, name: impl Into<String>, ty: SirType, span: Span) -> Param {
        Param {
            name_hint: name.into(),
            var: self.fresh_var(),
            ty,
            span,
        }
    }

    /// Create a parameter with a specific variable.
    pub fn param_with_var(
        &self,
        var: VarId,
        name: impl Into<String>,
        ty: SirType,
        span: Span,
    ) -> Param {
        Param {
            name_hint: name.into(),
            var,
            ty,
            span,
        }
    }

    /// Create a lambda.
    pub fn lambda(
        &mut self,
        params: Vec<Param>,
        captures: Vec<VarId>,
        body: Body,
        ret_tys: Vec<SirType>,
        span: Span,
    ) -> Lambda {
        Lambda {
            id: self.fresh_lambda(),
            params,
            captures,
            body,
            ret_tys,
            span,
        }
    }

    /// Create a lambda with fresh parameters (allocates vars automatically).
    pub fn lambda_with_fresh_params(
        &mut self,
        param_specs: Vec<(impl Into<String>, SirType)>,
        captures: Vec<VarId>,
        body_fn: impl FnOnce(&[VarId]) -> Body,
        ret_tys: Vec<SirType>,
        span: Span,
    ) -> Lambda {
        let params: Vec<_> = param_specs
            .into_iter()
            .map(|(name, ty)| self.param(name, ty, span))
            .collect();
        let param_vars: Vec<_> = params.iter().map(|p| p.var).collect();
        let body = body_fn(&param_vars);
        Lambda {
            id: self.fresh_lambda(),
            params,
            captures,
            body,
            ret_tys,
            span,
        }
    }

    // =========================================================================
    // Expression Builders
    // =========================================================================

    /// Create a constant i32 expression.
    pub fn const_i32(&self, n: i32) -> Exp {
        Exp::Prim(Prim::ConstI32(n))
    }

    /// Create a constant u32 expression.
    pub fn const_u32(&self, n: u32) -> Exp {
        Exp::Prim(Prim::ConstU32(n))
    }

    /// Create a constant f32 expression.
    pub fn const_f32(&self, n: f32) -> Exp {
        Exp::Prim(Prim::ConstF32(n))
    }

    /// Create an add expression.
    pub fn add(&self, a: VarId, b: VarId) -> Exp {
        Exp::Prim(Prim::Add(a, b))
    }

    /// Create a subtract expression.
    pub fn sub(&self, a: VarId, b: VarId) -> Exp {
        Exp::Prim(Prim::Sub(a, b))
    }

    /// Create a multiply expression.
    pub fn mul(&self, a: VarId, b: VarId) -> Exp {
        Exp::Prim(Prim::Mul(a, b))
    }

    /// Create a variable reference expression.
    pub fn var(&self, v: VarId) -> Exp {
        Exp::Var(v)
    }

    /// Create an array index expression.
    pub fn index(&self, arr: VarId, idx: VarId) -> Exp {
        Exp::Prim(Prim::Index { arr, idx })
    }

    /// Create a function application expression.
    pub fn apply(&self, func: impl Into<String>, args: Vec<VarId>) -> Exp {
        Exp::Apply {
            func: func.into(),
            args,
        }
    }

    // Note: No tuple/tuple_proj - tuples are eliminated via scalar replacement

    /// Create an if expression.
    pub fn if_then_else(&self, cond: VarId, then_body: Body, else_body: Body) -> Exp {
        Exp::If {
            cond,
            then_body,
            else_body,
        }
    }

    // =========================================================================
    // SOAC Builders
    // =========================================================================

    /// Create a map SOAC.
    pub fn map(&self, w: Size, f: Lambda, arrs: Vec<VarId>) -> Exp {
        Exp::Op(super::Op::Soac(Soac::Map(Map { w, f, arrs })))
    }

    /// Create a reduce SOAC.
    pub fn reduce(
        &self,
        w: Size,
        f: Lambda,
        neutral: VarId,
        arr: VarId,
        assoc: super::types::AssocInfo,
    ) -> Exp {
        Exp::Op(super::Op::Soac(Soac::Reduce(Reduce {
            w,
            f,
            neutral,
            arr,
            assoc,
        })))
    }

    /// Create a scan SOAC.
    pub fn scan(
        &self,
        w: Size,
        f: Lambda,
        neutral: VarId,
        arr: VarId,
        assoc: super::types::AssocInfo,
    ) -> Exp {
        Exp::Op(super::Op::Soac(Soac::Scan(Scan {
            w,
            f,
            neutral,
            arr,
            assoc,
        })))
    }

    /// Create an iota SOAC.
    pub fn iota(&self, n: Size, elem_ty: super::types::ScalarTy) -> Exp {
        Exp::Op(super::Op::Soac(Soac::Iota { n, elem_ty }))
    }

    /// Create a replicate SOAC.
    pub fn replicate(&self, n: Size, value: VarId) -> Exp {
        Exp::Op(super::Op::Soac(Soac::Replicate { n, value }))
    }

    // =========================================================================
    // Body Builders
    // =========================================================================

    /// Create an empty body.
    pub fn empty_body(&self) -> Body {
        Body::empty()
    }

    /// Create a body with a single result.
    pub fn body_just(&self, var: VarId) -> Body {
        Body::just(var)
    }

    /// Create a body with statements and results.
    pub fn body(&self, stms: Vec<Stm>, result: Vec<VarId>) -> Body {
        Body { stms, result }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TypeName;
    use polytype::Type;

    fn dummy_span() -> Span {
        Span::new(0, 0, 0, 0)
    }

    fn i32_type() -> SirType {
        Type::Constructed(TypeName::Int(32), vec![])
    }

    #[test]
    fn test_builder_fresh_ids() {
        let mut b = SirBuilder::new();

        let v1 = b.fresh_var();
        let v2 = b.fresh_var();
        assert_ne!(v1, v2);
        assert_eq!(v1.0, 0);
        assert_eq!(v2.0, 1);

        let s1 = b.fresh_stm();
        let s2 = b.fresh_stm();
        assert_ne!(s1, s2);

        let l1 = b.fresh_lambda();
        let l2 = b.fresh_lambda();
        assert_ne!(l1, l2);
    }

    #[test]
    fn test_builder_pat1() {
        let mut b = SirBuilder::new();
        let pat = b.pat1("x", i32_type());

        assert_eq!(pat.binds.len(), 1);
        assert_eq!(pat.binds[0].name_hint, "x");
        assert_eq!(pat.binds[0].var.0, 0);
    }

    #[test]
    fn test_builder_simple_body() {
        let mut b = SirBuilder::new();

        // let x = 42 in x
        let pat = b.pat1("x", i32_type());
        let x_var = pat.binds[0].var;
        let exp = b.const_i32(42);
        let stm = b.stm(pat, exp, i32_type(), dummy_span());

        let body = b.body(vec![stm], vec![x_var]);

        assert_eq!(body.stms.len(), 1);
        assert_eq!(body.result.len(), 1);
        assert_eq!(body.result[0], x_var);
    }
}
