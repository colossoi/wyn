//! Specialization pass for TLC.
//!
//! Specializes polymorphic intrinsic names based on argument types.
//! For example: `sign(x)` where `x: f32` becomes `f32.sign(x)`.

use super::{Def, LoopKind, Program, Term, TermIdSource, TermKind};
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

            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                let new_init_bindings = init_bindings
                    .into_iter()
                    .map(|(name, ty, expr)| (name, ty, self.specialize_term(expr)))
                    .collect();
                let new_kind = match kind {
                    LoopKind::For { var, var_ty, iter } => LoopKind::For {
                        var,
                        var_ty,
                        iter: Box::new(self.specialize_term(*iter)),
                    },
                    LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                        var,
                        var_ty,
                        bound: Box::new(self.specialize_term(*bound)),
                    },
                    LoopKind::While { cond } => LoopKind::While {
                        cond: Box::new(self.specialize_term(*cond)),
                    },
                };
                TermKind::Loop {
                    loop_var,
                    loop_var_ty,
                    init: Box::new(self.specialize_term(*init)),
                    init_bindings: new_init_bindings,
                    kind: new_kind,
                    body: Box::new(self.specialize_term(*body)),
                }
            }

            // Leaves unchanged
            k @ (TermKind::Var(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::StringLit(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Extern(_)) => k,
        };

        Term {
            id: self.term_ids.next_id(),
            ty: term.ty,
            span: term.span,
            kind,
        }
    }

    fn specialize_func(&mut self, func: Term, arg: &Term) -> Term {
        match &func.kind {
            TermKind::Var(name) => {
                let specialized = self.specialize_name(name, &arg.ty);
                if specialized != *name {
                    Term {
                        id: self.term_ids.next_id(),
                        ty: func.ty.clone(),
                        span: func.span,
                        kind: TermKind::Var(specialized),
                    }
                } else {
                    func
                }
            }

            // Check for fully-applied binary functions like mul
            // Pattern: App { func: Var("mul"), arg: first_arg }
            TermKind::App {
                func: inner_func,
                arg: first_arg,
            } => {
                // Check if inner_func is Var("mul")
                let maybe_mul_name =
                    if let TermKind::Var(name) = &inner_func.kind { Some(name.as_str()) } else { None };

                if maybe_mul_name == Some("mul") {
                    // Specialize mul based on both arg types
                    if let Some(specialized_name) = self.specialize_mul(&first_arg.ty, &arg.ty) {
                        // Build: App { func: Var(specialized_name), arg: first_arg }
                        // This becomes the new inner term
                        let specialized_first_arg = self.specialize_term(*first_arg.clone());
                        let specialized_var = Term {
                            id: self.term_ids.next_id(),
                            ty: inner_func.ty.clone(),
                            span: inner_func.span,
                            kind: TermKind::Var(specialized_name),
                        };
                        return Term {
                            id: self.term_ids.next_id(),
                            ty: func.ty.clone(),
                            span: func.span,
                            kind: TermKind::App {
                                func: Box::new(specialized_var),
                                arg: Box::new(specialized_first_arg),
                            },
                        };
                    }
                }

                // For nested applications or other terms, recursively specialize
                self.specialize_term(func)
            }

            // BinOp and UnOp don't need specialization - return as-is
            TermKind::BinOp(_) | TermKind::UnOp(_) => func,

            // For other term kinds, recursively specialize
            _ => self.specialize_term(func),
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
