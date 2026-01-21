//! Specialization pass for TLC.
//!
//! Specializes polymorphic intrinsic names based on argument types.
//! For example: `sign(x)` where `x: f32` becomes `f32.sign(x)`.

use super::{Def, FunctionName, Program, Term, TermIdSource, TermKind};
use crate::ast::TypeName;
use polytype::Type;

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
                        // Return specialized var wrapped in Term
                        let specialized_term = Term {
                            id: self.term_ids.next_id(),
                            ty: t.ty.clone(),
                            span: t.span,
                            kind: TermKind::Var(specialized_name),
                        };
                        return FunctionName::Term(Box::new(specialized_term));
                    }
                }

                // Check for fully-applied binary functions like mul
                // Pattern: App { func: Term(App { func: Term(Var("mul")), arg: first_arg }), arg: second_arg }
                // We're in specialize_func for the outer App, so:
                // - t is the inner App { func: Term(Var("mul")), arg: first_arg }
                // - arg is the second_arg
                if let TermKind::App {
                    func: inner_func,
                    arg: first_arg,
                } = &t.kind
                {
                    // Check if inner_func is Var("mul") (direct) or Term(Var("mul")) (wrapped)
                    let maybe_mul_name = match inner_func.as_ref() {
                        FunctionName::Var(name) => Some(name.as_str()),
                        FunctionName::Term(term) => {
                            if let TermKind::Var(name) = &term.kind {
                                Some(name.as_str())
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };

                    if maybe_mul_name == Some("mul") {
                        // Specialize mul based on both arg types
                        if let Some(specialized_name) =
                            self.specialize_mul(&first_arg.ty, &arg.ty)
                        {
                            // Build: App { func: Var(specialized_name), arg: first_arg }
                            // This becomes the new inner term
                            let specialized_first_arg = self.specialize_term(*first_arg.clone());
                            let new_inner = Term {
                                id: self.term_ids.next_id(),
                                ty: t.ty.clone(),
                                span: t.span,
                                kind: TermKind::App {
                                    func: Box::new(FunctionName::Var(specialized_name)),
                                    arg: Box::new(specialized_first_arg),
                                },
                            };
                            return FunctionName::Term(Box::new(new_inner));
                        }
                    }
                }

                // For nested applications or other terms, recursively specialize
                let specialized = self.specialize_term(*t);
                FunctionName::Term(Box::new(specialized))
            }
            // BinOp and UnOp don't need specialization
            other => other,
        }
    }

    /// Specialize a function name based on argument type.
    /// Transforms: abs, sign, min, max, clamp → f32.abs, i32.sign, etc.
    fn specialize_name(&self, name: &str, arg_ty: &Type<TypeName>) -> String {
        match name {
            "abs" | "sign" | "min" | "max" | "clamp" => {
                if let Some(prefix) = self.type_prefix(arg_ty) {
                    format!("{}.{}", prefix, name)
                } else {
                    name.to_string()
                }
            }
            _ => name.to_string(),
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

    /// Specialize mul based on argument shapes.
    /// Returns the specialized name: mul_mat_mat, mul_mat_vec, or mul_vec_mat
    fn specialize_mul(&self, first_ty: &Type<TypeName>, second_ty: &Type<TypeName>) -> Option<String> {
        let shape1 = Self::classify_shape(first_ty);
        let shape2 = Self::classify_shape(second_ty);

        match (shape1, shape2) {
            (ArgShape::Matrix, ArgShape::Matrix) => Some("mul_mat_mat".to_string()),
            (ArgShape::Matrix, ArgShape::Vector) => Some("mul_mat_vec".to_string()),
            (ArgShape::Vector, ArgShape::Matrix) => Some("mul_vec_mat".to_string()),
            _ => None, // Fall back to original mul (will error later if truly invalid)
        }
    }

    /// Classify a type as Matrix, Vector, or Other for mul specialization
    fn classify_shape(ty: &Type<TypeName>) -> ArgShape {
        match ty {
            Type::Constructed(TypeName::Mat, _) => ArgShape::Matrix,
            Type::Constructed(TypeName::Vec, _) => ArgShape::Vector,
            _ => ArgShape::Other,
        }
    }
}

/// Argument shape classification for mul specialization
#[derive(Debug, Clone, Copy, PartialEq)]
enum ArgShape {
    Matrix,
    Vector,
    Other,
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

        let partial_ty = Type::Constructed(
            TypeName::Arrow,
            vec![i32_ty.clone(), i32_ty.clone()],
        );

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
