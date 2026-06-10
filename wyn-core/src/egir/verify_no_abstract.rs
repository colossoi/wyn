//! Backend-boundary invariant: `Array[_, ArrayVariantAbstract, _, _]`
//! must not reach SPIR-V or WGSL emission.
//!
//! `ArrayVariantAbstract` is a TLC-level representation-polymorphic
//! variant — see `types::TypeName::ArrayVariantAbstract`. Producers
//! (e.g. `filter`) leave their result as Abstract in the TLC type and
//! the EGIR producer lowering picks a concrete variant (Bounded for
//! static-capacity inputs, View for runtime-sized). By the time we
//! reach SSA, every value must carry a concrete representation the
//! backend can lower.
//!
//! When this verifier fires, it means a call edge passed an
//! Abstract-typed value into a non-inlined consumer without per-rep
//! call-edge specialization. The fix is to add producer-rep
//! specialization at the offending Call edge (mirroring
//! `buffer_specialize` for views), or to inline the consumer. The
//! error message points at the function/value carrying the offending
//! type so users have something concrete to act on.
//!
//! Plumbed in `lib.rs` between SSA construction and backend lowering
//! (`SsaConverted::lower` / `SsaConverted::lower_wgsl`).

use crate::ast::TypeName;
use crate::error::CompilerError;
use crate::ssa::types::{FuncBody, Program};
use crate::types;
use polytype::Type;

pub type Result<T> = std::result::Result<T, CompilerError>;

/// Walk every value type in every function body, every entry body, and
/// every program-level constant body. Reject any `Array[_, Abstract,
/// _, _]`. The first offending value is returned as a `TypeError`; the
/// message names the function/entry/constant and the value index so the
/// user can locate the source span via `--output-mir`.
pub fn run(program: &Program) -> Result<()> {
    for func in &program.functions {
        check_body(&format!("function `{}`", func.name), &func.body)?;
    }
    for entry in &program.entry_points {
        check_body(&format!("entry `{}`", entry.name), &entry.body)?;
    }
    for constant in &program.constants {
        check_body(&format!("constant `{}`", constant.name), &constant.body)?;
    }
    Ok(())
}

fn check_body(scope: &str, body: &FuncBody) -> Result<()> {
    // Params first — the most common failure surface (a non-inlined
    // size-poly helper carrying an Abstract param).
    for (i, (_, ty, name)) in body.params.iter().enumerate() {
        check_ty(ty, &format!("{} param #{} `{}`", scope, i, name))?;
    }
    check_ty(&body.return_ty, &format!("{} return type", scope))?;
    // Every SSA value's type.
    for (vid, info) in body.inner.values.iter() {
        check_ty(&info.ty, &format!("{} value {:?}", scope, vid))?;
    }
    // Place element types (Alloca, ViewIndex, OutputSlot destinations).
    for (pid, place) in body.places.iter() {
        check_ty(&place.elem_ty, &format!("{} place {:?} elem type", scope, pid))?;
    }
    Ok(())
}

fn check_ty(ty: &Type<TypeName>, location: &str) -> Result<()> {
    // Walk every subtree. An Abstract may appear at any depth inside
    // tuples, records, sums, or nested arrays.
    if contains_abstract(ty) {
        return Err(CompilerError::TypeError(
            format!(
                "internal: representation-polymorphic array variant `abstract` reached backend \
                 lowering at {}: {:?}. This means a call edge passed a filter-like SOAC's \
                 result to a non-inlined size-polymorphic consumer; the consumer needs to be \
                 either inlined or per-representation specialized at the call edge (see \
                 `egir/verify_no_abstract.rs` for context).",
                location, ty
            ),
            None,
        ));
    }
    Ok(())
}

fn contains_abstract(ty: &Type<TypeName>) -> bool {
    if types::is_array_variant_abstract(ty) {
        return true;
    }
    match ty {
        Type::Variable(_) => false,
        Type::Constructed(_, args) => args.iter().any(contains_abstract),
    }
}

#[cfg(test)]
#[path = "verify_no_abstract_tests.rs"]
mod tests;
