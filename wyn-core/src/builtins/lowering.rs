use crate::impl_source::{BuiltinImpl, Intrinsic, PrimOp};

/// How a builtin lowers to backend operations.
///
/// `BuiltinLowering` lives in the catalog and is the source of truth for
/// codegen dispatch. Phase 4 of the unification migration replaces
/// backend `impl_source.get(&name)` with `catalog.lowering(id)` returning
/// one of these variants.
#[derive(Debug, Clone)]
pub enum BuiltinLowering {
    /// Direct mapping to a `PrimOp` (GLSL.std.450 extended instruction
    /// or core SPIR-V op).
    PrimOp(PrimOp),
    /// Genuine intrinsic that needs backend-specific lowering (e.g.
    /// uninit, length, array_with).
    Intrinsic(Intrinsic),
    /// Function imported from a pre-compiled SPIR-V module — the string
    /// is the linkage name in `OpDecorate LinkageAttributes`.
    LinkedSpirv(&'static str),
}

impl BuiltinLowering {
    /// Convert to the `BuiltinImpl` shape `ImplSource` consumes.
    pub fn to_builtin_impl(&self) -> BuiltinImpl {
        match self {
            BuiltinLowering::PrimOp(op) => BuiltinImpl::PrimOp(op.clone()),
            BuiltinLowering::Intrinsic(i) => BuiltinImpl::Intrinsic(i.clone()),
            BuiltinLowering::LinkedSpirv(name) => BuiltinImpl::LinkedSpirv((*name).to_string()),
        }
    }
}
