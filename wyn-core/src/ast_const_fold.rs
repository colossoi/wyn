//! Early integer constant folding and static-size staticization.
//!
//! Its original consumer is array-size inference: the type checker derives a
//! static `Size(N)` for a range / slice / array expression *only* when its
//! bounds are literal integers (`try_extract_const_int`, which matches bare
//! `IntLiteral` and nothing else). A named constant or unfolded arithmetic
//! leaves the size a fresh variable — i.e. runtime / unsized. The pass exposes
//! literal bounds:
//! - inline i32 constants: `def N = 256; 0..<N` → `0..<256` → `Size(256)`
//! - fold integer arithmetic: `0..<(2 + 4)` → `0..<6` → `Size(6)`
//! - resolve named type dimensions: `[N]u32` → `[256]u32`
//!
//! Non-i32 integer constants cannot become array sizes, but they are still
//! useful compile-time values (for example, fixed graphics draw counts). They
//! are inlined with an explicit type ascription so replacing `N: u32` never
//! silently changes a use of `N` into an i32 literal.

use crate::ast;
use crate::ast::UnaryOp;
use crate::ast::{
    Decl, Declaration, EntryDecl, ExprKind, Expression, IfExpr, LetInExpr, LoopExpr, LoopForm, MatchExpr,
    Pattern, PatternKind, Program, RangeExpr, Type, TypeName,
};
use crate::interface;
use crate::op::{BinaryOperator, UnaryOperator};
use crate::resolve_resources;
use crate::semantic_modules;
use crate::{LookupMap, NodeCounter};

/// AST after early integer constants have been exposed.
#[derive(Debug, Clone, Copy)]
pub enum ConstantsFoldedTag {}
pub type ConstantsFolded = Program<
    ConstantsFoldedTag,
    resolve_resources::ResourcesResolvedFamily,
    semantic_modules::SemanticModules,
>;

/// AST-level constant folder for integer constants.
pub struct AstConstFolder {
    /// Known integer constants: name → value
    constants: LookupMap<String, IntegerConstant>,
    /// Allocator borrowed from the program while folding. Typed literals add
    /// one inner AST node for their explicit type ascription.
    node_ids: NodeCounter,
}

#[derive(Clone)]
struct IntegerConstant {
    value: i64,
    ty: Type,
}

impl AstConstFolder {
    /// Add a constant for testing purposes
    #[cfg(test)]
    pub fn add_constant(&mut self, name: &str, value: i64) {
        self.constants.insert(
            name.to_string(),
            IntegerConstant {
                value,
                ty: Self::i32_type(),
            },
        );
    }
}

impl Default for AstConstFolder {
    fn default() -> Self {
        Self::new()
    }
}

impl AstConstFolder {
    pub fn new() -> Self {
        Self {
            constants: LookupMap::new(),
            node_ids: NodeCounter::new(),
        }
    }

    /// Fold constants in an entire program.
    ///
    /// Two passes:
    /// 1. Collect top-level constant definitions (parameterless defs with integer values)
    /// 2. Fold and inline in expressions and type dimensions
    pub fn fold_program(&mut self, program: &mut resolve_resources::ResourcesResolved) {
        std::mem::swap(&mut self.node_ids, &mut program.node_ids);

        // First pass: collect top-level constant definitions
        for decl in &program.declarations {
            if let Declaration::Decl(d) = decl {
                if d.params.is_empty() && d.size_params.is_empty() && d.type_params.is_empty() {
                    if let Some(ty) = Self::integer_constant_type(d.ty.as_ref(), &d.body) {
                        if let Some(value) = self.try_eval_any_integer_const(&d.body) {
                            self.constants.insert(d.name.clone(), IntegerConstant { value, ty });
                        }
                    }
                }
            }
        }

        // Second pass: fold and inline in all expressions
        for decl in &mut program.declarations {
            self.fold_declaration(decl);
        }

        std::mem::swap(&mut self.node_ids, &mut program.node_ids);
    }

    fn fold_declaration(&mut self, decl: &mut Declaration<resolve_resources::ResourcesResolvedFamily>) {
        match decl {
            Declaration::Decl(d) => self.fold_decl(d),
            Declaration::Entry(e) => self.fold_entry_decl(e),
            Declaration::Extern(e) => {
                self.fold_type(&mut e.data.ty, &e.data.size_params);
            }
            Declaration::Frontend(frontend) => match frontend {
                ast::ResourcesResolvedFrontend::Sig(sig) => {
                    self.fold_type(&mut sig.ty, &sig.size_params);
                }
                ast::ResourcesResolvedFrontend::TypeBind(bind) => {
                    let bound_sizes = bind
                        .type_params
                        .iter()
                        .filter_map(|param| match param {
                            ast::TypeParam::Size(name) => Some(name.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    self.fold_type(&mut bind.definition, &bound_sizes);
                }
                ast::ResourcesResolvedFrontend::Open(_) => {}
            },
        }
    }

    fn fold_decl(&mut self, d: &mut Decl) {
        for param in &mut d.params {
            self.fold_pattern(param, &d.size_params);
        }
        if let Some(ty) = &mut d.ty {
            self.fold_type(ty, &d.size_params);
        }
        self.fold_expr_scoped(&mut d.body, &d.size_params);
    }

    fn fold_entry_decl(
        &mut self,
        e: &mut EntryDecl<ast::ResolvedEntry, ast::SourceTree, interface::ResolvedAttribute>,
    ) {
        for param in &mut e.params {
            self.fold_pattern(param, &e.size_params);
        }
        for output in &mut e.data.syntax.outputs {
            self.fold_type(&mut output.ty, &e.size_params);
        }
        self.fold_expr_scoped(&mut e.body, &e.size_params);
    }

    /// Replace named type dimensions with static sizes when they refer to a
    /// known top-level i32 constant. Declaration size parameters and
    /// existential binders take precedence over constants with the same name.
    fn fold_type(&self, ty: &mut Type, bound_sizes: &[String]) {
        let Type::Constructed(name, args) = ty else {
            return;
        };

        if let TypeName::SizeVar(size_name) = name {
            if !bound_sizes.contains(size_name) {
                if let Some(size) = self.constants.get(size_name).and_then(|constant| {
                    Self::is_i32_type(&constant.ty)
                        .then(|| constant.value)
                        .and_then(|value| usize::try_from(value).ok())
                }) {
                    *name = TypeName::Size(size);
                }
            }
        }

        match name {
            TypeName::Sum(variants) => {
                for (_, payload) in variants {
                    for payload_ty in payload {
                        self.fold_type(payload_ty, bound_sizes);
                    }
                }
                for arg in args {
                    self.fold_type(arg, bound_sizes);
                }
            }
            TypeName::Existential(vars) => {
                let mut nested_bound_sizes = bound_sizes.to_vec();
                nested_bound_sizes.extend(vars.iter().cloned());
                for arg in args {
                    self.fold_type(arg, &nested_bound_sizes);
                }
            }
            _ => {
                for arg in args {
                    self.fold_type(arg, bound_sizes);
                }
            }
        }
    }

    fn fold_pattern<A>(&self, pattern: &mut Pattern<ast::SourceTree, A>, bound_sizes: &[String]) {
        match &mut pattern.kind {
            PatternKind::Tuple(patterns)
            | PatternKind::Vec(patterns)
            | PatternKind::Constructor(_, patterns) => {
                for pattern in patterns {
                    self.fold_pattern(pattern, bound_sizes);
                }
            }
            PatternKind::Record(fields) => {
                for field in fields {
                    if let ast::RecordPatternTarget::Pattern(pattern) = &mut field.target {
                        self.fold_pattern(pattern, bound_sizes);
                    }
                }
            }
            PatternKind::Typed(inner, ty) => {
                self.fold_pattern(inner, bound_sizes);
                self.fold_type(ty, bound_sizes);
            }
            PatternKind::Attributed(_, inner) => self.fold_pattern(inner, bound_sizes),
            PatternKind::Name(_) | PatternKind::Wildcard | PatternKind::Literal(_) | PatternKind::Unit => {}
        }
    }

    /// Recursively fold constants in an expression.
    /// Modifies the expression in place.
    #[cfg(test)]
    pub fn fold_expr(&mut self, expr: &mut Expression) {
        self.fold_expr_scoped(expr, &[]);
    }

    fn fold_expr_scoped(&mut self, expr: &mut Expression, bound_sizes: &[String]) {
        match &mut expr.kind {
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::Unit
            | ExprKind::TypeHole(_) => {
                // Leaf nodes, nothing to fold
            }

            ExprKind::Identifier(identifier) => {
                // Inline known constants (only for unqualified names)
                if identifier.qualifiers.is_empty() {
                    if let Some(constant) = self.constants.get(&identifier.name).cloned() {
                        expr.kind = self.constant_expr_kind(&constant, &expr.h);
                    }
                }
            }

            ExprKind::BinaryOp(ref op, ref mut lhs, ref mut rhs) => {
                self.fold_expr_scoped(lhs, bound_sizes);
                self.fold_expr_scoped(rhs, bound_sizes);
                // Try to fold after children are folded
                if let Some(val) = self.try_fold_binop(&op.op, lhs, rhs) {
                    expr.kind = ExprKind::IntLiteral(val.to_string().into());
                } else {
                    // Try algebraic identity rewrites
                    Self::try_algebraic_simplify(expr);
                }
            }

            ExprKind::UnaryOp(op, operand) => {
                self.fold_expr_scoped(operand, bound_sizes);
                // Try to fold after child is folded
                if let Some(val) = self.try_fold_unaryop(&op.op, operand) {
                    expr.kind = ExprKind::IntLiteral(val.to_string().into());
                }
            }

            ExprKind::ArrayLiteral(elements) | ExprKind::VecMatLiteral(elements) => {
                for elem in elements {
                    self.fold_expr_scoped(elem, bound_sizes);
                }
            }

            ExprKind::ArrayIndex(arr, idx) => {
                self.fold_expr_scoped(arr, bound_sizes);
                self.fold_expr_scoped(idx, bound_sizes);
            }

            ExprKind::ArrayWith {
                array, index, value, ..
            } => {
                self.fold_expr_scoped(array, bound_sizes);
                self.fold_expr_scoped(index, bound_sizes);
                self.fold_expr_scoped(value, bound_sizes);
            }

            ExprKind::VecWith { target, value, .. } => {
                self.fold_expr_scoped(target, bound_sizes);
                self.fold_expr_scoped(value, bound_sizes);
            }

            ExprKind::RecordWith { record, value, .. } => {
                self.fold_expr_scoped(record, bound_sizes);
                self.fold_expr_scoped(value, bound_sizes);
            }

            ExprKind::Tuple(elements) => {
                for elem in elements {
                    self.fold_expr_scoped(elem, bound_sizes);
                }
            }

            ExprKind::Constructor(_, args) => {
                for arg in args {
                    self.fold_expr_scoped(arg, bound_sizes);
                }
            }

            ExprKind::RecordLiteral(fields) => {
                for (_name, value) in fields {
                    self.fold_expr_scoped(value, bound_sizes);
                }
            }

            ExprKind::Lambda(lambda) => {
                for param in &mut lambda.params {
                    self.fold_pattern(param, bound_sizes);
                }
                self.fold_expr_scoped(&mut lambda.body, bound_sizes);
            }

            ExprKind::Application(func, args) => {
                self.fold_expr_scoped(func, bound_sizes);
                for arg in args {
                    self.fold_expr_scoped(arg, bound_sizes);
                }
            }

            ExprKind::LetIn(let_in) => {
                self.fold_let_in(let_in, bound_sizes);
            }

            ExprKind::FieldAccess(obj, _field) => {
                self.fold_expr_scoped(obj, bound_sizes);
            }

            ExprKind::If(if_expr) => {
                self.fold_if(if_expr, bound_sizes);
            }

            ExprKind::Loop(loop_expr) => {
                self.fold_loop(loop_expr, bound_sizes);
            }

            ExprKind::Match(match_expr) => {
                self.fold_match(match_expr, bound_sizes);
            }

            ExprKind::Range(range) => {
                self.fold_range(range, bound_sizes);
            }

            ExprKind::Slice(slice) => {
                self.fold_expr_scoped(&mut slice.array, bound_sizes);
                if let Some(start) = &mut slice.start {
                    self.fold_expr_scoped(start, bound_sizes);
                }
                if let Some(end) = &mut slice.end {
                    self.fold_expr_scoped(end, bound_sizes);
                }
            }

            ExprKind::TypeAscription(inner, ty) | ExprKind::TypeCoercion(inner, ty) => {
                self.fold_expr_scoped(inner, bound_sizes);
                self.fold_type(ty, bound_sizes);
            }
        }
    }

    fn fold_let_in(&mut self, let_in: &mut LetInExpr, bound_sizes: &[String]) {
        self.fold_pattern(&mut let_in.pattern, bound_sizes);
        if let Some(ty) = &mut let_in.ty {
            self.fold_type(ty, bound_sizes);
        }

        // Fold the value first
        self.fold_expr_scoped(&mut let_in.value, bound_sizes);

        // Check if this introduces a constant
        // For simplicity, only handle simple name patterns
        let const_binding = if let ast::PatternKind::Name(name) = &let_in.pattern.kind {
            Self::integer_constant_type(let_in.ty.as_ref(), &let_in.value).and_then(|ty| {
                self.try_eval_any_integer_const(&let_in.value)
                    .map(|value| (name.clone(), IntegerConstant { value, ty }))
            })
        } else {
            None
        };

        // If we found a constant, temporarily add it to scope
        let shadowed = const_binding
            .as_ref()
            .and_then(|(name, constant)| self.constants.insert(name.clone(), constant.clone()));

        // Fold the body
        self.fold_expr_scoped(&mut let_in.body, bound_sizes);

        // Remove the temporary binding (it's scoped to this let)
        if let Some((name, _)) = const_binding {
            if let Some(constant) = shadowed {
                self.constants.insert(name, constant);
            } else {
                self.constants.remove(&name);
            }
        }
    }

    fn fold_if(&mut self, if_expr: &mut IfExpr, bound_sizes: &[String]) {
        self.fold_expr_scoped(&mut if_expr.condition, bound_sizes);
        self.fold_expr_scoped(&mut if_expr.then_branch, bound_sizes);
        self.fold_expr_scoped(&mut if_expr.else_branch, bound_sizes);
    }

    fn fold_loop(&mut self, loop_expr: &mut LoopExpr, bound_sizes: &[String]) {
        self.fold_pattern(&mut loop_expr.pattern, bound_sizes);
        if let Some(init) = &mut loop_expr.init {
            self.fold_expr_scoped(init, bound_sizes);
        }
        match &mut loop_expr.form {
            LoopForm::For(var, bound) => {
                self.fold_pattern(var, bound_sizes);
                self.fold_expr_scoped(bound, bound_sizes);
            }
            LoopForm::ForIn(pattern, iter) => {
                self.fold_pattern(pattern, bound_sizes);
                self.fold_expr_scoped(iter, bound_sizes);
            }
            LoopForm::While(cond) => {
                self.fold_expr_scoped(cond, bound_sizes);
            }
        }
        self.fold_expr_scoped(&mut loop_expr.body, bound_sizes);
    }

    fn fold_match(&mut self, match_expr: &mut MatchExpr, bound_sizes: &[String]) {
        self.fold_expr_scoped(&mut match_expr.scrutinee, bound_sizes);
        for case in &mut match_expr.cases {
            self.fold_pattern(&mut case.pattern, bound_sizes);
            self.fold_expr_scoped(&mut case.body, bound_sizes);
        }
    }

    fn fold_range(&mut self, range: &mut RangeExpr, bound_sizes: &[String]) {
        self.fold_expr_scoped(&mut range.start, bound_sizes);
        if let Some(step) = &mut range.step {
            self.fold_expr_scoped(step, bound_sizes);
        }
        self.fold_expr_scoped(&mut range.end, bound_sizes);
    }

    fn i32_type() -> Type {
        Type::Constructed(TypeName::Int(32), vec![])
    }

    fn is_i32_type(ty: &Type) -> bool {
        matches!(ty, Type::Constructed(TypeName::Int(32), _))
    }

    /// Determine the integer type carried by a constant declaration. An
    /// unannotated integer defaults to i32; a suffixed literal reaches the AST
    /// as a top-level type ascription.
    fn integer_constant_type(annotation: Option<&Type>, body: &Expression) -> Option<Type> {
        let ty = annotation
            .cloned()
            .or_else(|| match &body.kind {
                ExprKind::TypeAscription(_, ty) => Some(ty.clone()),
                _ => None,
            })
            .unwrap_or_else(Self::i32_type);
        matches!(ty, Type::Constructed(TypeName::Int(_) | TypeName::UInt(_), _)).then_some(ty)
    }

    /// Rebuild a reference as a literal while retaining non-i32 integer types.
    /// i32 deliberately stays bare because static-size inference recognizes a
    /// bare integer literal.
    fn constant_expr_kind(&mut self, constant: &IntegerConstant, header: &ast::Header) -> ExprKind {
        let literal = ExprKind::IntLiteral(constant.value.to_string().into());
        if Self::is_i32_type(&constant.ty) {
            literal
        } else {
            ExprKind::TypeAscription(
                Box::new(Expression {
                    h: ast::Header {
                        id: self.node_ids.next_id(),
                        span: header.span,
                    },
                    kind: literal,
                }),
                constant.ty.clone(),
            )
        }
    }

    /// Try to evaluate an expression as a constant integer.
    /// Returns None if the expression is not a constant integer.
    fn try_eval_const(&self, expr: &Expression) -> Option<i64> {
        match &expr.kind {
            ExprKind::IntLiteral(n) => i64::try_from(n).ok(),
            ExprKind::Identifier(identifier) if identifier.qualifiers.is_empty() => self
                .constants
                .get(&identifier.name)
                .filter(|constant| Self::is_i32_type(&constant.ty))
                .map(|constant| constant.value),
            ExprKind::BinaryOp(op, lhs, rhs) => {
                let l = self.try_eval_const(lhs)?;
                let r = self.try_eval_const(rhs)?;
                self.eval_binop(&op.op, l, r)
            }
            ExprKind::UnaryOp(op, operand) => {
                let v = self.try_eval_const(operand)?;
                self.eval_unaryop(&op.op, v)
            }
            ExprKind::TypeAscription(inner, ty) if Self::is_i32_type(ty) => self.try_eval_const(inner),
            _ => None,
        }
    }

    /// Evaluate an integer declaration while collecting constants, regardless
    /// of its explicit integer type. Ordinary expression folding continues to
    /// use `try_eval_const`, which is i32-only and therefore cannot erase a
    /// non-i32 type ascription around an arithmetic expression.
    fn try_eval_any_integer_const(&self, expr: &Expression) -> Option<i64> {
        match &expr.kind {
            ExprKind::IntLiteral(n) => i64::try_from(n).ok(),
            ExprKind::Identifier(identifier) if identifier.qualifiers.is_empty() => {
                self.constants.get(&identifier.name).map(|constant| constant.value)
            }
            ExprKind::BinaryOp(op, lhs, rhs) => {
                let lhs = self.try_eval_any_integer_const(lhs)?;
                let rhs = self.try_eval_any_integer_const(rhs)?;
                self.eval_binop(&op.op, lhs, rhs)
            }
            ExprKind::UnaryOp(op, operand) => {
                let operand = self.try_eval_any_integer_const(operand)?;
                self.eval_unaryop(&op.op, operand)
            }
            ExprKind::TypeAscription(inner, ty)
                if Self::integer_constant_type(Some(ty), inner).is_some() =>
            {
                self.try_eval_any_integer_const(inner)
            }
            _ => None,
        }
    }

    fn try_fold_binop(&self, op: &BinaryOperator, lhs: &Expression, rhs: &Expression) -> Option<i64> {
        let l = self.try_eval_const(lhs)?;
        let r = self.try_eval_const(rhs)?;
        self.eval_binop(op, l, r)
    }

    fn try_fold_unaryop(&self, op: &UnaryOperator, operand: &Expression) -> Option<i64> {
        let v = self.try_eval_const(operand)?;
        self.eval_unaryop(op, v)
    }

    fn eval_binop(&self, op: &BinaryOperator, l: i64, r: i64) -> Option<i64> {
        match op {
            BinaryOperator::Add => Some(l.wrapping_add(r)),
            BinaryOperator::Subtract => Some(l.wrapping_sub(r)),
            BinaryOperator::Multiply => Some(l.wrapping_mul(r)),
            BinaryOperator::Divide if r != 0 => Some(l / r),
            BinaryOperator::Remainder if r != 0 => Some(l % r),
            _ => None,
        }
    }

    fn eval_unaryop(&self, op: &UnaryOperator, v: i64) -> Option<i64> {
        match op {
            UnaryOperator::Negate => Some(-v),
            _ => None,
        }
    }

    /// Check if an expression is a zero literal (int or float)
    fn is_zero(expr: &Expression) -> bool {
        match &expr.kind {
            ExprKind::IntLiteral(n) => n.as_str() == "0",
            ExprKind::FloatLiteral(v) => *v == 0.0,
            _ => false,
        }
    }

    /// Check if an expression is a one literal (int or float)
    fn is_one(expr: &Expression) -> bool {
        match &expr.kind {
            ExprKind::IntLiteral(n) => n.as_str() == "1",
            ExprKind::FloatLiteral(v) => *v == 1.0,
            _ => false,
        }
    }

    /// Check if an expression is a negative one literal (int or float)
    fn is_neg_one(expr: &Expression) -> bool {
        match &expr.kind {
            ExprKind::IntLiteral(n) => n.as_str() == "-1",
            ExprKind::FloatLiteral(v) => *v == -1.0,
            _ => false,
        }
    }

    /// Try to apply algebraic identity simplifications to a binary op expression.
    /// Modifies expr in place if a simplification applies.
    ///
    /// Simplifications:
    /// - 0 - x → -x
    /// - 0 + x, x + 0 → x
    /// - 0 * x, x * 0 → 0
    /// - 1 * x, x * 1 → x
    /// - -1 * x, x * -1 → -x
    /// - x - 0, x / 1 → x
    fn try_algebraic_simplify(expr: &mut Expression) {
        // Check if any simplification applies
        let dominated_by_zero = if let ExprKind::BinaryOp(ref op, ref lhs, ref rhs) = expr.kind {
            match op.op {
                BinaryOperator::Subtract if Self::is_zero(lhs) => Some(()),
                BinaryOperator::Add if Self::is_zero(lhs) || Self::is_zero(rhs) => Some(()),
                BinaryOperator::Subtract if Self::is_zero(rhs) => Some(()),
                BinaryOperator::Multiply if Self::is_zero(lhs) || Self::is_zero(rhs) => Some(()),
                BinaryOperator::Multiply if Self::is_one(lhs) || Self::is_one(rhs) => Some(()),
                BinaryOperator::Multiply if Self::is_neg_one(lhs) || Self::is_neg_one(rhs) => Some(()),
                BinaryOperator::Divide if Self::is_one(rhs) => Some(()),
                _ => None,
            }
        } else {
            None
        };

        if dominated_by_zero.is_none() {
            return;
        }

        // Take ownership and apply the rewrite
        let old_kind = std::mem::replace(&mut expr.kind, ExprKind::Unit);
        if let ExprKind::BinaryOp(binop, lhs, rhs) = old_kind {
            expr.kind = match binop.op {
                // 0 - x → -x
                BinaryOperator::Subtract if Self::is_zero(&lhs) => {
                    let unop = UnaryOp {
                        op: UnaryOperator::Negate,
                    };
                    ExprKind::UnaryOp(unop, rhs)
                }
                // 0 + x → x
                BinaryOperator::Add if Self::is_zero(&lhs) => rhs.kind,
                // x + 0, x - 0 → x
                BinaryOperator::Add | BinaryOperator::Subtract if Self::is_zero(&rhs) => lhs.kind,
                // 0 * x → 0, x * 0 → 0
                BinaryOperator::Multiply if Self::is_zero(&lhs) => lhs.kind,
                BinaryOperator::Multiply if Self::is_zero(&rhs) => rhs.kind,
                // 1 * x → x
                BinaryOperator::Multiply if Self::is_one(&lhs) => rhs.kind,
                // x * 1, x / 1 → x
                BinaryOperator::Multiply | BinaryOperator::Divide if Self::is_one(&rhs) => lhs.kind,
                // -1 * x → -x
                BinaryOperator::Multiply if Self::is_neg_one(&lhs) => {
                    let unop = UnaryOp {
                        op: UnaryOperator::Negate,
                    };
                    ExprKind::UnaryOp(unop, rhs)
                }
                // x * -1 → -x
                BinaryOperator::Multiply if Self::is_neg_one(&rhs) => {
                    let unop = UnaryOp {
                        op: UnaryOperator::Negate,
                    };
                    ExprKind::UnaryOp(unop, lhs)
                }
                // Shouldn't reach here, but restore original if we do
                _ => ExprKind::BinaryOp(binop, lhs, rhs),
            };
        }
    }
}

/// Expose typed integer constants and literal bounds needed by static-size inference.
pub fn fold_constants(mut program: resolve_resources::ResourcesResolved) -> ConstantsFolded {
    let mut folder = AstConstFolder::new();
    folder.fold_program(&mut program);
    program.retag()
}

#[cfg(test)]
#[path = "ast_const_fold_tests.rs"]
mod ast_const_fold_tests;
