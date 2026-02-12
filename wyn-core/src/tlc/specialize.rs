//! Specialization pass for TLC.
//!
//! Specializes polymorphic intrinsic names based on argument types.
//! For example: `sign(x)` where `x: f32` becomes `f32.sign(x)`.

use super::{ArrayExpr, Def, Lambda, LoopKind, Program, SoacOp, Term, TermIdSource, TermKind};
use crate::SymbolTable;
use crate::ast::TypeName;
use polytype::Type;

/// Specialize polymorphic intrinsics in a TLC program.
pub fn specialize(program: Program) -> Program {
    let mut specializer = Specializer {
        symbols: program.symbols,
        term_ids: TermIdSource::new(),
    };

    let defs = program.defs.into_iter().map(|d| specializer.specialize_def(d)).collect();

    Program {
        defs,
        uniforms: program.uniforms,
        storage: program.storage,
        symbols: specializer.symbols,
    }
}

struct Specializer {
    symbols: SymbolTable,
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

            TermKind::Lambda(Lambda {
                params,
                body,
                ret_ty,
                captures,
            }) => TermKind::Lambda(Lambda {
                params,
                body: Box::new(self.specialize_term(*body)),
                ret_ty,
                captures: captures.into_iter().map(|(s, ty, t)| (s, ty, self.specialize_term(t))).collect(),
            }),

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

            TermKind::Soac(soac) => TermKind::Soac(self.specialize_soac(soac)),

            TermKind::ArrayExpr(ae) => TermKind::ArrayExpr(self.specialize_array_expr(ae)),

            TermKind::Force(inner) => TermKind::Force(Box::new(self.specialize_term(*inner))),

            TermKind::Pack { .. } | TermKind::Unpack { .. } => {
                unreachable!("Pack/Unpack nodes not yet produced at this phase")
            }
        };

        Term {
            id: self.term_ids.next_id(),
            ty: term.ty,
            span: term.span,
            kind,
        }
    }

    fn specialize_lambda(&mut self, lam: Lambda) -> Lambda {
        Lambda {
            params: lam.params,
            body: Box::new(self.specialize_term(*lam.body)),
            ret_ty: lam.ret_ty,
            captures: lam.captures.into_iter().map(|(s, ty, t)| (s, ty, self.specialize_term(t))).collect(),
        }
    }

    fn specialize_soac(&mut self, soac: SoacOp) -> SoacOp {
        match soac {
            SoacOp::Map { lam, inputs } => SoacOp::Map {
                lam: self.specialize_lambda(lam),
                inputs: inputs.into_iter().map(|ae| self.specialize_array_expr(ae)).collect(),
            },
            SoacOp::Reduce { op, ne, input, props } => SoacOp::Reduce {
                op: self.specialize_lambda(op),
                ne: Box::new(self.specialize_term(*ne)),
                input: self.specialize_array_expr(input),
                props,
            },
            SoacOp::Scan { op, ne, input } => SoacOp::Scan {
                op: self.specialize_lambda(op),
                ne: Box::new(self.specialize_term(*ne)),
                input: self.specialize_array_expr(input),
            },
            SoacOp::Filter { pred, input } => SoacOp::Filter {
                pred: self.specialize_lambda(pred),
                input: self.specialize_array_expr(input),
            },
            SoacOp::Scatter {
                dest,
                indices,
                values,
            } => SoacOp::Scatter {
                dest,
                indices: self.specialize_array_expr(indices),
                values: self.specialize_array_expr(values),
            },
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
                props,
            } => SoacOp::ReduceByIndex {
                dest,
                op: self.specialize_lambda(op),
                ne: Box::new(self.specialize_term(*ne)),
                indices: self.specialize_array_expr(indices),
                values: self.specialize_array_expr(values),
                props,
            },
        }
    }

    fn specialize_array_expr(&mut self, ae: ArrayExpr) -> ArrayExpr {
        match ae {
            ArrayExpr::Ref(t) => ArrayExpr::Ref(Box::new(self.specialize_term(*t))),
            ArrayExpr::Zip(exprs) => {
                ArrayExpr::Zip(exprs.into_iter().map(|e| self.specialize_array_expr(e)).collect())
            }
            ArrayExpr::Soac(op) => ArrayExpr::Soac(Box::new(self.specialize_soac(*op))),
            ArrayExpr::Generate {
                shape,
                index_fn,
                elem_ty,
            } => ArrayExpr::Generate {
                shape,
                index_fn: self.specialize_lambda(index_fn),
                elem_ty,
            },
            ArrayExpr::Literal(terms) => {
                ArrayExpr::Literal(terms.into_iter().map(|t| self.specialize_term(t)).collect())
            }
            ArrayExpr::Range { start, len } => ArrayExpr::Range {
                start: Box::new(self.specialize_term(*start)),
                len: Box::new(self.specialize_term(*len)),
            },
        }
    }

    fn specialize_func(&mut self, func: Term, arg: &Term) -> Term {
        match &func.kind {
            TermKind::Var(sym) => {
                let sym = *sym;
                let name = self.symbols.get(sym).expect("BUG: symbol not in table");
                if let Some(specialized_name) = self.specialize_name(name, &arg.ty) {
                    let specialized_sym = self.symbols.alloc(specialized_name);
                    Term {
                        id: self.term_ids.next_id(),
                        ty: func.ty.clone(),
                        span: func.span,
                        kind: TermKind::Var(specialized_sym),
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
                let maybe_mul_name = if let TermKind::Var(sym) = &inner_func.kind {
                    Some(self.symbols.get(*sym).expect("BUG: symbol not in table").as_str())
                } else {
                    None
                };

                if maybe_mul_name == Some("mul") {
                    // Convert mul(a, b) → BinOp("*")(a, b)
                    let specialized_first_arg = self.specialize_term(*first_arg.clone());
                    let binop = Term {
                        id: self.term_ids.next_id(),
                        ty: inner_func.ty.clone(),
                        span: inner_func.span,
                        kind: TermKind::BinOp(crate::ast::BinaryOp { op: "*".to_string() }),
                    };
                    return Term {
                        id: self.term_ids.next_id(),
                        ty: func.ty.clone(),
                        span: func.span,
                        kind: TermKind::App {
                            func: Box::new(binop),
                            arg: Box::new(specialized_first_arg),
                        },
                    };
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
    /// Returns Some(specialized_name) if specialization is needed, None otherwise.
    fn specialize_name(&self, name: &str, arg_ty: &Type<TypeName>) -> Option<String> {
        match name {
            "abs" | "sign" | "min" | "max" | "clamp" => {
                if let Some(prefix) = self.type_prefix(arg_ty) {
                    Some(format!("{}.{}", prefix, name))
                } else {
                    None
                }
            }
            _ => None,
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
}
