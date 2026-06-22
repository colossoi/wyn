// Most of this module is a target for an in-progress refactor —
// subsequent commits move call sites here and these items become
// referenced. Suppressing the dead-code warning at module scope
// keeps the build clean during the transition.
#![allow(dead_code)]

//! Compiler-agnostic wrapper around `rspirv`'s `Builder`.
//!
//! This module is the target for an ongoing refactor of `spirv/mod.rs`.
//! Goal: every SPIR-V emission path goes through this builder, and the
//! builder itself knows nothing about Wyn's compiler types
//! (`PolyType<TypeName>`, `BindingRef`, etc.). The compiler-aware
//! lowering layer in `spirv/mod.rs` calls into the builder with plain
//! SPIR-V primitives (`spirv::Word`, sizes, decorations) and owns its
//! own compiler-type → SPIR-V mappings on top.
//!
//! What the builder will own (incrementally moved here):
//! - rspirv `Builder` lifecycle + capability/memory-model setup.
//! - Structurally-deduped type emission (`OpTypeVector`,
//!   `OpTypeStruct`, `OpTypePointer`, `OpTypeRuntimeArray`, etc.).
//! - Value-keyed constant emission (`OpConstant*`).
//! - Once-only decoration tracking (`Block`, `NonWritable`,
//!   `ArrayStride`, …).
//! - Function/block/instruction emission.
//!
//! What stays in `spirv/mod.rs`:
//! - `PolyType<TypeName> → Id<TypeKind>` mappings.
//! - `BindingRef → buffer/var` mappings.
//! - Per-function lowering state (current block, env, output vars).
//! - Compiler-symbol → function-id maps.
//! - The `lower_*` family that walks SSA and calls into the builder.

use rspirv::dr::Builder;
use rspirv::spirv::{self, AddressingModel, Capability, MemoryModel};
use std::hash::Hash;
use std::marker::PhantomData;

/// Phantom-typed wrapper around `spirv::Word` that prevents call-site
/// confusion between distinct kinds of IDs (type vs. value vs.
/// variable, etc.). `Deref<Target = spirv::Word>` lets `*id` extract
/// the raw word for the few rspirv interactions (entry-point interface
/// lists) that need a heterogeneous `Vec<Word>` — but function
/// arguments typed `spirv::Word` will *not* auto-coerce, so passing a
/// `TypeId` where a `ValueId` is expected is still a build error.
pub struct Id<K>(spirv::Word, PhantomData<fn() -> K>);

impl<K> Id<K> {
    pub fn new(raw: spirv::Word) -> Self {
        Id(raw, PhantomData)
    }
}

impl<K> std::ops::Deref for Id<K> {
    type Target = spirv::Word;
    fn deref(&self) -> &spirv::Word {
        &self.0
    }
}

// Manual impls — derive(Clone, Copy, …) would require `K: Clone`,
// which is unnecessary for a phantom type parameter.
impl<K> Clone for Id<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K> Copy for Id<K> {}
impl<K> PartialEq for Id<K> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<K> Eq for Id<K> {}
impl<K> Hash for Id<K> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}
impl<K> std::fmt::Debug for Id<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Id<{}>({})",
            std::any::type_name::<K>().rsplit("::").next().unwrap_or("?"),
            self.0
        )
    }
}

// Kind markers. Each is an empty type used only as a phantom parameter
// to distinguish what an `Id` refers to.
pub struct TypeKind;
pub struct ConstKind;
pub struct ValueKind;
pub struct VarKind;
pub struct FuncKind;
pub struct BlockKind;

pub type TypeId = Id<TypeKind>;
pub type ConstId = Id<ConstKind>;
pub type ValueId = Id<ValueKind>;
pub type VarId = Id<VarKind>;
pub type FuncId = Id<FuncKind>;
pub type BlockId = Id<BlockKind>;

/// Wrapper around `rspirv::dr::Builder`. Methods that emit dedupable
/// constructs (types, constants, certain decorations) maintain
/// internal caches keyed only on SPIR-V primitives — no compiler
/// types reach this struct.
///
/// At this stage of the refactor only the lifecycle / capability
/// setup is delegated through here; the rest is still in
/// `spirv::mod`'s `Constructor`. Subsequent commits will move
/// type-emission, constant-emission, and decoration tracking here.
pub struct SpirvBuilder {
    inner: Builder,
}

impl SpirvBuilder {
    /// Create a builder pre-configured for a Wyn-style shader module:
    /// SPIR-V 1.5, `Shader` capability, `Logical` addressing model,
    /// `GLSL450` memory model.
    pub fn new() -> Self {
        let mut inner = Builder::new();
        inner.set_version(1, 5);
        inner.capability(Capability::Shader);
        inner.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
        SpirvBuilder { inner }
    }

    /// Consume the builder and produce the finished SPIR-V module.
    pub fn into_module(self) -> rspirv::dr::Module {
        self.inner.module()
    }
}

/// Temporary bridge: existing call sites can keep writing
/// `builder.type_int(32, 1)` etc. directly during the refactor.
/// Each subsequent stage adds typed `SpirvBuilder` methods and the
/// matching call sites switch to them; the final commit removes
/// these `Deref` impls and the build flags any remaining raw rspirv
/// access. **Do not add new `Deref`-using call sites** — write
/// against the typed `SpirvBuilder` API instead.
impl std::ops::Deref for SpirvBuilder {
    type Target = Builder;
    fn deref(&self) -> &Builder {
        &self.inner
    }
}

impl std::ops::DerefMut for SpirvBuilder {
    fn deref_mut(&mut self) -> &mut Builder {
        &mut self.inner
    }
}

impl Default for SpirvBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "builder_tests.rs"]
mod tests;
