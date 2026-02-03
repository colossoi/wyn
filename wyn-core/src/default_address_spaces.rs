//! Default unconstrained address space variables to Function.
//!
//! This pass runs after monomorphization. At this point, address spaces that
//! depend on call sites have been propagated through specialization. Any
//! remaining unconstrained address space variables are local arrays that don't
//! escape, so they default to Function (stack-allocated).

use crate::ast::TypeName;
use crate::mir::{Body, Def, Program};
use polytype::Type;
use std::collections::HashSet;

/// Default unconstrained address space variables to Function in a program.
pub fn default_address_spaces(mut program: Program) -> Program {
    // First pass: collect variables that appear in address space position
    let mut addr_vars = HashSet::new();
    for def in &program.defs {
        collect_in_def(def, &mut addr_vars);
    }

    // Second pass: replace those variables with AddressFunction
    for def in &mut program.defs {
        default_in_def(def, &addr_vars);
    }
    program
}

fn collect_in_def(def: &Def, vars: &mut HashSet<usize>) {
    match def {
        Def::Function { ret_type, body, .. } => {
            collect_addr_vars(ret_type, vars);
            collect_in_body(body, vars);
        }
        Def::EntryPoint {
            inputs,
            outputs,
            body,
            ..
        } => {
            for input in inputs {
                collect_addr_vars(&input.ty, vars);
            }
            for output in outputs {
                collect_addr_vars(&output.ty, vars);
            }
            collect_in_body(body, vars);
        }
        Def::Constant { body, .. } => {
            collect_in_body(body, vars);
        }
        Def::Uniform { ty, .. } => {
            collect_addr_vars(ty, vars);
        }
        Def::Storage { ty, .. } => {
            collect_addr_vars(ty, vars);
        }
    }
}

fn collect_in_body(body: &Body, vars: &mut HashSet<usize>) {
    for local in &body.locals {
        collect_addr_vars(&local.ty, vars);
    }
    for ty in &body.types {
        collect_addr_vars(ty, vars);
    }
}

/// Collect variable IDs that appear in address space position (args[1] of Array).
fn collect_addr_vars(ty: &Type<TypeName>, vars: &mut HashSet<usize>) {
    match ty {
        Type::Constructed(TypeName::Array, args) if args.len() == 3 => {
            // args[1] is the address space
            if let Type::Variable(id) = &args[1] {
                vars.insert(*id);
            }
            // Recurse into all args
            for arg in args {
                collect_addr_vars(arg, vars);
            }
        }
        Type::Constructed(_, args) => {
            for arg in args {
                collect_addr_vars(arg, vars);
            }
        }
        Type::Variable(_) => {}
    }
}

fn default_in_def(def: &mut Def, addr_vars: &HashSet<usize>) {
    match def {
        Def::Function { ret_type, body, .. } => {
            default_in_type(ret_type, addr_vars);
            default_in_body(body, addr_vars);
        }
        Def::EntryPoint {
            inputs,
            outputs,
            body,
            ..
        } => {
            for input in inputs {
                default_in_type(&mut input.ty, addr_vars);
            }
            for output in outputs {
                default_in_type(&mut output.ty, addr_vars);
            }
            default_in_body(body, addr_vars);
        }
        Def::Constant { body, .. } => {
            default_in_body(body, addr_vars);
        }
        Def::Uniform { ty, .. } => {
            default_in_type(ty, addr_vars);
        }
        Def::Storage { ty, .. } => {
            default_in_type(ty, addr_vars);
        }
    }
}

fn default_in_body(body: &mut Body, addr_vars: &HashSet<usize>) {
    for local in &mut body.locals {
        default_in_type(&mut local.ty, addr_vars);
    }
    for ty in &mut body.types {
        default_in_type(ty, addr_vars);
    }
}

fn default_in_type(ty: &mut Type<TypeName>, addr_vars: &HashSet<usize>) {
    match ty {
        Type::Constructed(TypeName::Array, args) if args.len() == 3 => {
            // Array[elem, variant, size] - handle specially
            // Recurse into element type first
            default_in_type(&mut args[0], addr_vars);

            let is_runtime_sized = matches!(&args[2], Type::Variable(_));

            // Check if variant is an unconstrained variable
            if let Type::Variable(id) = &args[1] {
                if addr_vars.contains(id) {
                    // Default variant based on whether size is runtime or fixed
                    args[1] = if is_runtime_sized {
                        // Runtime-sized arrays must use View variant
                        Type::Constructed(TypeName::ArrayVariantView, vec![])
                    } else {
                        // Fixed-size arrays can use Composite variant
                        Type::Constructed(TypeName::ArrayVariantComposite, vec![])
                    };
                }
            } else if is_runtime_sized {
                // Variant is already set, but if size is runtime and variant is Composite,
                // we must correct it to View (runtime-sized arrays can't be Composite)
                if matches!(&args[1], Type::Constructed(TypeName::ArrayVariantComposite, _)) {
                    args[1] = Type::Constructed(TypeName::ArrayVariantView, vec![]);
                } else {
                    default_in_type(&mut args[1], addr_vars);
                }
            } else {
                // Fixed-size, just recurse
                default_in_type(&mut args[1], addr_vars);
            }

            // Recurse into size (though size variables aren't in addr_vars)
            default_in_type(&mut args[2], addr_vars);
        }
        Type::Variable(id) if addr_vars.contains(id) => {
            // Non-array variable in addr position - should not happen
            // but default to Composite for safety
            *ty = Type::Constructed(TypeName::ArrayVariantComposite, vec![]);
        }
        Type::Variable(_) => {}
        Type::Constructed(_, args) => {
            for arg in args {
                default_in_type(arg, addr_vars);
            }
        }
    }
}
