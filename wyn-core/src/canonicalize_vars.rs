//! Pass to align body type variables with scheme type variables.
//!
//! After this pass, for any Def::Function with scheme: Some(s),
//! every Type::Variable(k) in the body uses the same IDs as the scheme.
//!
//! This simplifies monomorphization by establishing the invariant that
//! body types and scheme types share the same variable namespace.

use crate::ast::TypeName;
use crate::mir::{Body, Def, Program};
use crate::types::TypeScheme;
use polytype::Type;
use std::collections::HashMap;

/// Renaming map from body variable ID to scheme variable ID
type VarRenaming = HashMap<usize, usize>;

/// Canonicalize all type variables in a program.
/// After this pass, body type variables match scheme type variables.
pub fn canonicalize_program(program: Program) -> Program {
    Program {
        defs: program.defs.into_iter().map(canonicalize_def).collect(),
        lambda_registry: program.lambda_registry,
    }
}

/// Canonicalize type variables in a single definition.
fn canonicalize_def(def: Def) -> Def {
    match def {
        Def::Function {
            id,
            name,
            params,
            ret_type,
            scheme: Some(ref scheme),
            attributes,
            body,
            span,
        } => {
            // 1. Extract scheme function type
            let scheme_ty = unwrap_scheme(scheme);
            let (scheme_params, scheme_ret) = split_function_type(scheme_ty);

            // 2. Extract body function type from params and root
            let body_params: Vec<_> = params.iter().map(|&p| body.get_local(p).ty.clone()).collect();
            let body_ret = body.get_type(body.root).clone();

            // 3. Compute renaming: body_var_id -> scheme_var_id
            let mut renaming = VarRenaming::new();
            for (body_ty, scheme_ty) in body_params.iter().zip(scheme_params.iter()) {
                compute_renaming(body_ty, scheme_ty, &mut renaming);
            }
            compute_renaming(&body_ret, &scheme_ret, &mut renaming);

            // If no renaming needed, verify and return unchanged
            if renaming.is_empty() {
                debug_verify_canonicalization(&name, scheme, &body);
                return Def::Function {
                    id,
                    name,
                    params,
                    ret_type,
                    scheme: Some(scheme.clone()),
                    attributes,
                    body,
                    span,
                };
            }

            // 4. Apply renaming to entire body and ret_type
            let body = apply_renaming_to_body(body, &renaming);
            let ret_type = apply_renaming(&ret_type, &renaming);

            // Verify the invariant holds after canonicalization
            debug_verify_canonicalization(&name, scheme, &body);

            Def::Function {
                id,
                name,
                params,
                ret_type,
                scheme: Some(scheme.clone()),
                attributes,
                body,
                span,
            }
        }
        // Non-function defs and functions without schemes pass through unchanged
        other => other,
    }
}

/// Unwrap a TypeScheme to get the inner monotype.
fn unwrap_scheme(scheme: &TypeScheme) -> &Type<TypeName> {
    match scheme {
        TypeScheme::Monotype(ty) => ty,
        TypeScheme::Polytype { body, .. } => unwrap_scheme(body),
    }
}

/// Split a function type into (param_types, return_type).
/// Handles curried function types: (A -> B -> C) becomes ([A, B], C)
fn split_function_type(ty: &Type<TypeName>) -> (Vec<Type<TypeName>>, Type<TypeName>) {
    let mut params = Vec::new();
    let mut current = ty.clone();

    while let Type::Constructed(TypeName::Arrow, ref args) = current {
        if args.len() == 2 {
            params.push(args[0].clone());
            current = args[1].clone();
        } else {
            break;
        }
    }

    (params, current)
}

/// Compute variable renaming by structural alignment.
/// When we see (Variable(body_id), Variable(scheme_id)), record body_id -> scheme_id.
fn compute_renaming(body_ty: &Type<TypeName>, scheme_ty: &Type<TypeName>, renaming: &mut VarRenaming) {
    match (body_ty, scheme_ty) {
        (Type::Variable(body_id), Type::Variable(scheme_id)) => {
            // Only add mapping if they differ
            if body_id != scheme_id {
                renaming.insert(*body_id, *scheme_id);
            }
        }
        (Type::Constructed(name1, args1), Type::Constructed(name2, args2))
            if name1 == name2 && args1.len() == args2.len() =>
        {
            for (b, s) in args1.iter().zip(args2.iter()) {
                compute_renaming(b, s, renaming);
            }
        }
        _ => {} // Mismatched structure - no renaming possible
    }
}

/// Apply renaming to a type, replacing Variable(old) with Variable(new).
fn apply_renaming(ty: &Type<TypeName>, renaming: &VarRenaming) -> Type<TypeName> {
    match ty {
        Type::Variable(id) => {
            if let Some(&new_id) = renaming.get(id) {
                Type::Variable(new_id)
            } else {
                ty.clone()
            }
        }
        Type::Constructed(name, args) => {
            let new_args = args.iter().map(|a| apply_renaming(a, renaming)).collect();
            Type::Constructed(name.clone(), new_args)
        }
    }
}

/// Apply renaming to all types in a body.
fn apply_renaming_to_body(mut body: Body, renaming: &VarRenaming) -> Body {
    // Rename local types
    for local in body.locals.iter_mut() {
        local.ty = apply_renaming(&local.ty, renaming);
    }

    // Rename expression types
    for i in 0..body.types.len() {
        body.types[i] = apply_renaming(&body.types[i], renaming);
    }

    body
}

/// Collect all type variable IDs from a scheme.
#[cfg(debug_assertions)]
fn collect_scheme_vars(scheme: &TypeScheme) -> std::collections::HashSet<usize> {
    let mut vars = std::collections::HashSet::new();
    collect_scheme_vars_inner(scheme, &mut vars);
    vars
}

#[cfg(debug_assertions)]
fn collect_scheme_vars_inner(scheme: &TypeScheme, vars: &mut std::collections::HashSet<usize>) {
    match scheme {
        TypeScheme::Monotype(ty) => collect_type_vars(ty, vars),
        TypeScheme::Polytype { variable, body, .. } => {
            vars.insert(*variable);
            collect_scheme_vars_inner(body, vars);
        }
    }
}

/// Collect all type variable IDs from a type.
#[cfg(debug_assertions)]
fn collect_type_vars(ty: &Type<TypeName>, vars: &mut std::collections::HashSet<usize>) {
    match ty {
        Type::Variable(id) => {
            vars.insert(*id);
        }
        Type::Constructed(_, args) => {
            for arg in args {
                collect_type_vars(arg, vars);
            }
        }
    }
}

/// Debug assertion: verify all body variables are in the scheme's variable set.
/// Call this after canonicalization to ensure the invariant holds.
#[cfg(debug_assertions)]
fn debug_verify_canonicalization(name: &str, scheme: &TypeScheme, body: &Body) {
    let scheme_vars = collect_scheme_vars(scheme);

    // Check all local types
    for (idx, local) in body.locals.iter().enumerate() {
        let mut local_vars = std::collections::HashSet::new();
        collect_type_vars(&local.ty, &mut local_vars);
        for var in &local_vars {
            if !scheme_vars.contains(var) {
                panic!(
                    "canonicalize_vars BUG in '{}': local[{}] '{}' has Variable({}) \
                     which is not in scheme vars {:?}. \
                     This suggests incomplete renaming from body namespace to scheme namespace.",
                    name, idx, local.name, var, scheme_vars
                );
            }
        }
    }

    // Check all expression types
    for (idx, ty) in body.types.iter().enumerate() {
        let mut expr_vars = std::collections::HashSet::new();
        collect_type_vars(ty, &mut expr_vars);
        for var in &expr_vars {
            if !scheme_vars.contains(var) {
                panic!(
                    "canonicalize_vars BUG in '{}': expr[{}] has Variable({}) \
                     which is not in scheme vars {:?}. \
                     This suggests incomplete renaming from body namespace to scheme namespace.",
                    name, idx, var, scheme_vars
                );
            }
        }
    }
}

#[cfg(not(debug_assertions))]
fn debug_verify_canonicalization(_name: &str, _scheme: &TypeScheme, _body: &Body) {
    // No-op in release builds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_renaming_simple() {
        // body: Variable(100), scheme: Variable(1)
        // Should produce: 100 -> 1
        let body_ty = Type::Variable(100);
        let scheme_ty = Type::Variable(1);
        let mut renaming = VarRenaming::new();
        compute_renaming(&body_ty, &scheme_ty, &mut renaming);
        assert_eq!(renaming.get(&100), Some(&1));
    }

    #[test]
    fn test_compute_renaming_nested() {
        // body: Array[Variable(100), Size(3), AddressFunction]
        // scheme: Array[Variable(1), Size(3), AddressFunction]
        // Should produce: 100 -> 1
        let body_ty = Type::Constructed(
            TypeName::Array,
            vec![
                Type::Variable(100),
                Type::Constructed(TypeName::Size(3), vec![]),
                Type::Constructed(TypeName::AddressFunction, vec![]),
            ],
        );
        let scheme_ty = Type::Constructed(
            TypeName::Array,
            vec![
                Type::Variable(1),
                Type::Constructed(TypeName::Size(3), vec![]),
                Type::Constructed(TypeName::AddressFunction, vec![]),
            ],
        );
        let mut renaming = VarRenaming::new();
        compute_renaming(&body_ty, &scheme_ty, &mut renaming);
        assert_eq!(renaming.get(&100), Some(&1));
    }

    #[test]
    fn test_apply_renaming() {
        let mut renaming = VarRenaming::new();
        renaming.insert(100, 1);

        let ty = Type::Constructed(
            TypeName::Arrow,
            vec![
                Type::Variable(100),
                Type::Constructed(TypeName::Float(32), vec![]),
            ],
        );

        let renamed = apply_renaming(&ty, &renaming);
        match renamed {
            Type::Constructed(TypeName::Arrow, args) => {
                assert_eq!(args[0], Type::Variable(1));
            }
            _ => panic!("Expected Arrow type"),
        }
    }

    #[test]
    fn test_no_renaming_when_same() {
        // body: Variable(1), scheme: Variable(1)
        // Should produce empty renaming
        let body_ty = Type::Variable(1);
        let scheme_ty = Type::Variable(1);
        let mut renaming = VarRenaming::new();
        compute_renaming(&body_ty, &scheme_ty, &mut renaming);
        assert!(renaming.is_empty());
    }
}
