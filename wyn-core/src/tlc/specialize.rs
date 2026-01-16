//! Specialization pass for TLC.
//!
//! Specializes polymorphic intrinsic names based on argument types.
//! For example: `sign(x)` where `x: f32` becomes `f32.sign(x)`.

use super::{Def, FunctionName, Program, Term, TermIdSource, TermKind};
use crate::ast::TypeName;
use polytype::Type;

/// Shape classification for matrix multiplication dispatch
#[derive(Debug, Clone, Copy, PartialEq)]
enum ArgShape {
    Matrix,
    Vector,
    Other,
}

/// Specialize polymorphic intrinsics in a TLC program.
pub fn specialize(program: Program) -> Program {
    let mut specializer = Specializer {
        term_ids: TermIdSource::new(),
    };

    Program {
        defs: program.defs.into_iter().map(|d| specializer.specialize_def(d)).collect(),
        uniforms: program.uniforms,
        storage: program.storage,
    }
}

struct Specializer {
    term_ids: TermIdSource,
}

impl Specializer {
    fn specialize_def(&mut self, def: Def) -> Def {
        Def {
            body: self.specialize_term(def.body),
            ..def
        }
    }

    fn specialize_term(&mut self, term: Term) -> Term {
        let kind = match term.kind {
            TermKind::App { func, arg } => {
                let specialized_arg = self.specialize_term(*arg);
                let specialized_func = self.specialize_func(*func, &specialized_arg);
                TermKind::App {
                    func: Box::new(specialized_func),
                    arg: Box::new(specialized_arg),
                }
            }

            TermKind::Lam {
                param,
                param_ty,
                body,
            } => TermKind::Lam {
                param,
                param_ty,
                body: Box::new(self.specialize_term(*body)),
            },

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => TermKind::Let {
                name,
                name_ty,
                rhs: Box::new(self.specialize_term(*rhs)),
                body: Box::new(self.specialize_term(*body)),
            },

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => TermKind::If {
                cond: Box::new(self.specialize_term(*cond)),
                then_branch: Box::new(self.specialize_term(*then_branch)),
                else_branch: Box::new(self.specialize_term(*else_branch)),
            },

            // Leaves unchanged
            k @ (TermKind::Var(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::StringLit(_)) => k,
        };

        Term {
            id: self.term_ids.next_id(),
            ty: term.ty,
            span: term.span,
            kind,
        }
    }

    fn specialize_func(&mut self, func: FunctionName, arg: &Term) -> FunctionName {
        match func {
            FunctionName::Var(name) => {
                let specialized = self.specialize_name(&name, &arg.ty);
                FunctionName::Var(specialized)
            }
            FunctionName::Term(t) => {
                // Check if the Term is a direct Var reference (sign, abs, etc.)
                // This is the common case from transform_application
                if let TermKind::Var(name) = &t.kind {
                    let specialized_name = self.specialize_name(name, &arg.ty);
                    if specialized_name != *name {
                        // Specialize the type too: polymorphic A -> A becomes concrete arg_ty -> arg_ty
                        let specialized_ty = self.specialize_unary_type(&t.ty, &arg.ty);
                        // Return specialized var wrapped in Term
                        let specialized_term = Term {
                            id: self.term_ids.next_id(),
                            ty: specialized_ty,
                            span: t.span,
                            kind: TermKind::Var(specialized_name),
                        };
                        return FunctionName::Term(Box::new(specialized_term));
                    }
                }
                // Check for 2-arg functions like mul: App(App(mul, arg1), arg2)
                // Here t is App(mul, arg1) and arg is arg2
                if let TermKind::App {
                    func: inner_func,
                    arg: inner_arg,
                } = &t.kind
                {
                    if let Some(func_name) = Self::extract_func_name(inner_func) {
                        if let Some(specialized_name) =
                            self.specialize_two_arg(&func_name, &inner_arg.ty, &arg.ty)
                        {
                            // Specialize the inner arg recursively
                            let specialized_inner_arg = self.specialize_term(*inner_arg.clone());
                            // Specialize the type: partial app type becomes (arg2_ty -> result_ty)
                            let specialized_partial_ty =
                                self.specialize_mul_partial_type(&func_name, &inner_arg.ty, &arg.ty);
                            // Rebuild: App(specialized_name, inner_arg)
                            let new_inner_app = Term {
                                id: self.term_ids.next_id(),
                                ty: specialized_partial_ty,
                                span: t.span,
                                kind: TermKind::App {
                                    func: Box::new(FunctionName::Var(specialized_name)),
                                    arg: Box::new(specialized_inner_arg),
                                },
                            };
                            return FunctionName::Term(Box::new(new_inner_app));
                        }
                    }
                }
                // For nested applications or other terms, recursively specialize
                let specialized = self.specialize_term(*t);
                FunctionName::Term(Box::new(specialized))
            }
            // Intrinsic, BinOp, and UnOp don't need specialization
            other => other,
        }
    }

    /// Extract function name from a FunctionName if it's a simple Var or Intrinsic
    fn extract_func_name(func: &FunctionName) -> Option<String> {
        match func {
            FunctionName::Var(name) | FunctionName::Intrinsic(name) => Some(name.clone()),
            FunctionName::Term(t) => {
                if let TermKind::Var(name) = &t.kind {
                    Some(name.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Specialize two-argument functions based on both argument types.
    /// Returns None if no specialization needed.
    fn specialize_two_arg(
        &self,
        name: &str,
        arg1_ty: &Type<TypeName>,
        arg2_ty: &Type<TypeName>,
    ) -> Option<String> {
        use crate::intrinsics::IntrinsicSource;

        // First resolve any alias to its intrinsic name
        let resolved = IntrinsicSource::resolve_alias(name).unwrap_or(name);

        match resolved {
            // mul alias resolves to _w_intrinsic_mul, match both
            "mul" | "_w_intrinsic_mul" => {
                let shape1 = self.classify_shape(arg1_ty);
                let shape2 = self.classify_shape(arg2_ty);
                match (shape1, shape2) {
                    (ArgShape::Matrix, ArgShape::Matrix) => Some("_w_intrinsic_mul_mat_mat".to_string()),
                    (ArgShape::Matrix, ArgShape::Vector) => Some("_w_intrinsic_mul_mat_vec".to_string()),
                    (ArgShape::Vector, ArgShape::Matrix) => Some("_w_intrinsic_mul_vec_mat".to_string()),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Classify a type as Matrix, Vector, or Other
    fn classify_shape(&self, ty: &Type<TypeName>) -> ArgShape {
        match ty {
            Type::Constructed(TypeName::Mat, _) => ArgShape::Matrix,
            Type::Constructed(TypeName::Vec, _) => ArgShape::Vector,
            _ => ArgShape::Other,
        }
    }

    /// Specialize a function name based on argument type.
    /// Transforms: abs, sign, floor, ceil, fract, min, max, clamp → f32.abs, i32.sign, etc.
    fn specialize_name(&self, name: &str, arg_ty: &Type<TypeName>) -> String {
        use crate::intrinsics::IntrinsicSource;

        // First resolve any alias to its intrinsic name
        let resolved = IntrinsicSource::resolve_alias(name).unwrap_or(name);

        match resolved {
            "abs" | "sign" | "floor" | "ceil" | "fract" | "min" | "max" | "clamp" => {
                if let Some(prefix) = self.type_prefix(arg_ty) {
                    format!("{}.{}", prefix, resolved)
                } else {
                    resolved.to_string()
                }
            }
            _ => resolved.to_string(),
        }
    }

    /// Get the type prefix for specialization (f32, i32, u32, etc.)
    fn type_prefix(&self, ty: &Type<TypeName>) -> Option<String> {
        // Extract element type for vectors
        let elem_ty = match ty {
            Type::Constructed(TypeName::Vec, args) if args.len() >= 2 => &args[1],
            _ => ty,
        };

        match elem_ty {
            Type::Constructed(TypeName::Float(bits), _) => Some(format!("f{}", bits)),
            Type::Constructed(TypeName::Int(bits), _) => Some(format!("i{}", bits)),
            Type::Constructed(TypeName::UInt(bits), _) => Some(format!("u{}", bits)),
            _ => None,
        }
    }

    /// Specialize a unary function type (A -> A) to concrete (arg_ty -> arg_ty).
    /// For functions like abs, sign, floor, ceil, fract that have signature A -> A.
    fn specialize_unary_type(&self, _func_ty: &Type<TypeName>, arg_ty: &Type<TypeName>) -> Type<TypeName> {
        // Unary intrinsics like abs, sign, floor, ceil, fract have signature: A -> A
        // When specialized, they become: concrete -> concrete
        Type::Constructed(TypeName::Arrow, vec![arg_ty.clone(), arg_ty.clone()])
    }

    /// Specialize the partial application type for mul operations.
    /// Returns the type after applying the first argument (arg2_ty -> result_ty).
    fn specialize_mul_partial_type(
        &self,
        func_name: &str,
        arg1_ty: &Type<TypeName>,
        arg2_ty: &Type<TypeName>,
    ) -> Type<TypeName> {
        // Compute the result type based on the shapes
        let result_ty = match func_name {
            "mul" => {
                let shape1 = self.classify_shape(arg1_ty);
                let shape2 = self.classify_shape(arg2_ty);
                match (shape1, shape2) {
                    // mat * mat = mat
                    (ArgShape::Matrix, ArgShape::Matrix) => arg1_ty.clone(),
                    // mat * vec = vec (columns of matrix)
                    (ArgShape::Matrix, ArgShape::Vector) => arg2_ty.clone(),
                    // vec * mat = vec (rows of result)
                    (ArgShape::Vector, ArgShape::Matrix) => {
                        // Result is a vector with size = number of columns in matrix
                        // Extract column count from mat type
                        if let Type::Constructed(TypeName::Mat, args) = arg2_ty {
                            if args.len() >= 3 {
                                // mat(rows, cols, elem) -> vec(cols, elem)
                                return Type::Constructed(
                                    TypeName::Vec,
                                    vec![args[1].clone(), args[2].clone()],
                                );
                            }
                        }
                        arg1_ty.clone() // fallback
                    }
                    _ => arg1_ty.clone(), // scalar or unknown
                }
            }
            _ => arg1_ty.clone(), // unknown function
        };

        // Return partial application type: arg2_ty -> result_ty
        Type::Constructed(TypeName::Arrow, vec![arg2_ty.clone(), result_ty])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::tlc::DefMeta;

    fn dummy_span() -> Span {
        Span {
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
        }
    }

    #[test]
    fn test_specialize_sign_f32() {
        let mut ids = TermIdSource::new();
        let f32_ty = Type::Constructed(TypeName::Float(32), vec![]);

        // Build: sign(x) where x: f32
        let x_var = Term {
            id: ids.next_id(),
            ty: f32_ty.clone(),
            span: dummy_span(),
            kind: TermKind::Var("x".to_string()),
        };

        let sign_call = Term {
            id: ids.next_id(),
            ty: f32_ty.clone(),
            span: dummy_span(),
            kind: TermKind::App {
                func: Box::new(FunctionName::Var("sign".to_string())),
                arg: Box::new(x_var),
            },
        };

        let program = Program {
            defs: vec![Def {
                name: "test".to_string(),
                ty: f32_ty.clone(),
                body: sign_call,
                meta: DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
        };

        let specialized = specialize(program);

        // Check that sign became f32.sign
        match &specialized.defs[0].body.kind {
            TermKind::App { func, .. } => match func.as_ref() {
                FunctionName::Var(name) => assert_eq!(name, "f32.sign"),
                _ => panic!("Expected Var"),
            },
            _ => panic!("Expected App"),
        }
    }

    #[test]
    fn test_specialize_sign_f32_term_var() {
        // Test the case where sign comes through FunctionName::Term(Var)
        // This is how transform_application produces it
        let mut ids = TermIdSource::new();
        let f32_ty = Type::Constructed(TypeName::Float(32), vec![]);

        let x_var = Term {
            id: ids.next_id(),
            ty: f32_ty.clone(),
            span: dummy_span(),
            kind: TermKind::Var("x".to_string()),
        };

        let sign_var = Term {
            id: ids.next_id(),
            ty: Type::Constructed(TypeName::Arrow, vec![f32_ty.clone(), f32_ty.clone()]),
            span: dummy_span(),
            kind: TermKind::Var("sign".to_string()),
        };

        // App { func: Term(Var("sign")), arg: x }
        let sign_call = Term {
            id: ids.next_id(),
            ty: f32_ty.clone(),
            span: dummy_span(),
            kind: TermKind::App {
                func: Box::new(FunctionName::Term(Box::new(sign_var))),
                arg: Box::new(x_var),
            },
        };

        let program = Program {
            defs: vec![Def {
                name: "test".to_string(),
                ty: f32_ty.clone(),
                body: sign_call,
                meta: DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
        };

        let specialized = specialize(program);

        // Check that sign became f32.sign
        match &specialized.defs[0].body.kind {
            TermKind::App { func, .. } => match func.as_ref() {
                FunctionName::Term(inner) => match &inner.kind {
                    TermKind::Var(name) => assert_eq!(name, "f32.sign"),
                    _ => panic!("Expected Var inside Term, got {:?}", inner.kind),
                },
                _ => panic!("Expected Term, got {:?}", func),
            },
            _ => panic!("Expected App"),
        }
    }

    #[test]
    fn test_specialize_min_i32() {
        let mut ids = TermIdSource::new();
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

        // Build: min(a, b) where a, b: i32
        // In curried form: App(Term(App(Var("min"), a)), b)
        let a_var = Term {
            id: ids.next_id(),
            ty: i32_ty.clone(),
            span: dummy_span(),
            kind: TermKind::Var("a".to_string()),
        };

        let b_var = Term {
            id: ids.next_id(),
            ty: i32_ty.clone(),
            span: dummy_span(),
            kind: TermKind::Var("b".to_string()),
        };

        let partial_ty = Type::Constructed(TypeName::Arrow, vec![i32_ty.clone(), i32_ty.clone()]);

        let min_a = Term {
            id: ids.next_id(),
            ty: partial_ty,
            span: dummy_span(),
            kind: TermKind::App {
                func: Box::new(FunctionName::Var("min".to_string())),
                arg: Box::new(a_var),
            },
        };

        let min_a_b = Term {
            id: ids.next_id(),
            ty: i32_ty.clone(),
            span: dummy_span(),
            kind: TermKind::App {
                func: Box::new(FunctionName::Term(Box::new(min_a))),
                arg: Box::new(b_var),
            },
        };

        let program = Program {
            defs: vec![Def {
                name: "test".to_string(),
                ty: i32_ty.clone(),
                body: min_a_b,
                meta: DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
        };

        let specialized = specialize(program);

        // Check that min became i32.min in the inner application
        match &specialized.defs[0].body.kind {
            TermKind::App { func, .. } => match func.as_ref() {
                FunctionName::Term(inner) => match &inner.kind {
                    TermKind::App { func: inner_func, .. } => match inner_func.as_ref() {
                        FunctionName::Var(name) => assert_eq!(name, "i32.min"),
                        _ => panic!("Expected Var"),
                    },
                    _ => panic!("Expected inner App"),
                },
                _ => panic!("Expected Term"),
            },
            _ => panic!("Expected App"),
        }
    }
}
