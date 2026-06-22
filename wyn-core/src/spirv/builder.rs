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
use std::collections::{HashMap, HashSet};
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
    // Well-known types eagerly created at construction so call sites
    // can look them up by name instead of asking rspirv each time.
    void_type: TypeId,
    bool_type: TypeId,
    i32_type: TypeId,
    u32_type: TypeId,
    f32_type: TypeId,
    glsl_ext_inst_id: spirv::Word,
    // Constant deduping. Each cache is keyed by the value; the value
    // (or the bit pattern, for floats) maps to the SPIR-V id, with
    // a reverse map for integers so consumers that need the literal
    // back (array sizes, loop bounds…) can recover it from the id.
    int_const_cache: HashMap<i32, ConstId>,
    int_const_reverse: HashMap<ConstId, i32>,
    uint_const_cache: HashMap<u32, ConstId>,
    uint_const_reverse: HashMap<ConstId, u32>,
    float_const_cache: HashMap<u32, ConstId>, // bits as u32
    bool_const_cache: HashMap<bool, ConstId>,
    // Every constant id emitted via this builder, so consumers can
    // ask `is_constant(id)` to decide whether a composite can be
    // `OpConstantComposite` vs `OpCompositeConstruct`.
    constant_ids: HashSet<ConstId>,
}

impl SpirvBuilder {
    /// Create a builder pre-configured for a Wyn-style shader module:
    /// SPIR-V 1.5, `Shader` capability, `Logical` addressing model,
    /// `GLSL450` memory model. Eagerly emits the well-known scalar
    /// types and imports the GLSL extended instruction set.
    pub fn new() -> Self {
        let mut inner = Builder::new();
        inner.set_version(1, 5);
        inner.capability(Capability::Shader);
        inner.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
        let void_type = TypeId::new(inner.type_void());
        let bool_type = TypeId::new(inner.type_bool());
        let i32_type = TypeId::new(inner.type_int(32, 1));
        let u32_type = TypeId::new(inner.type_int(32, 0));
        let f32_type = TypeId::new(inner.type_float(32));
        let glsl_ext_inst_id = inner.ext_inst_import("GLSL.std.450");
        SpirvBuilder {
            inner,
            void_type,
            bool_type,
            i32_type,
            u32_type,
            f32_type,
            glsl_ext_inst_id,
            int_const_cache: HashMap::new(),
            int_const_reverse: HashMap::new(),
            uint_const_cache: HashMap::new(),
            uint_const_reverse: HashMap::new(),
            float_const_cache: HashMap::new(),
            bool_const_cache: HashMap::new(),
            constant_ids: HashSet::new(),
        }
    }

    /// Consume the builder and produce the finished SPIR-V module.
    pub fn into_module(self) -> rspirv::dr::Module {
        self.inner.module()
    }

    pub fn void_type(&self) -> TypeId {
        self.void_type
    }
    pub fn bool_type(&self) -> TypeId {
        self.bool_type
    }
    pub fn i32_type(&self) -> TypeId {
        self.i32_type
    }
    pub fn u32_type(&self) -> TypeId {
        self.u32_type
    }
    pub fn f32_type(&self) -> TypeId {
        self.f32_type
    }
    pub fn glsl_ext_inst_id(&self) -> spirv::Word {
        self.glsl_ext_inst_id
    }

    /// Get or create an `OpConstant` for an `i32` value.
    pub fn const_i32(&mut self, value: i32) -> ConstId {
        if let Some(&id) = self.int_const_cache.get(&value) {
            return id;
        }
        let id = ConstId::new(self.inner.constant_bit32(*self.i32_type, value as u32));
        self.int_const_cache.insert(value, id);
        self.int_const_reverse.insert(id, value);
        self.constant_ids.insert(id);
        id
    }

    /// Get or create an `OpConstant` for a `u32` value.
    pub fn const_u32(&mut self, value: u32) -> ConstId {
        if let Some(&id) = self.uint_const_cache.get(&value) {
            return id;
        }
        let id = ConstId::new(self.inner.constant_bit32(*self.u32_type, value));
        self.uint_const_cache.insert(value, id);
        self.uint_const_reverse.insert(id, value);
        self.constant_ids.insert(id);
        id
    }

    /// Get or create an `OpConstant` for an `f32` value.
    pub fn const_f32(&mut self, value: f32) -> ConstId {
        let bits = value.to_bits();
        if let Some(&id) = self.float_const_cache.get(&bits) {
            return id;
        }
        let id = ConstId::new(self.inner.constant_bit32(*self.f32_type, bits));
        self.float_const_cache.insert(bits, id);
        self.constant_ids.insert(id);
        id
    }

    /// Get or create an `OpConstantTrue` / `OpConstantFalse`.
    pub fn const_bool(&mut self, value: bool) -> ConstId {
        if let Some(&id) = self.bool_const_cache.get(&value) {
            return id;
        }
        let id = ConstId::new(if value {
            self.inner.constant_true(*self.bool_type)
        } else {
            self.inner.constant_false(*self.bool_type)
        });
        self.bool_const_cache.insert(value, id);
        self.constant_ids.insert(id);
        id
    }

    /// Reverse lookup: literal `i32` for a constant id created via
    /// `const_i32`. `None` if the id wasn't an i32 constant.
    pub fn get_const_i32_value(&self, id: ConstId) -> Option<i32> {
        self.int_const_reverse.get(&id).copied()
    }

    /// Reverse lookup: literal `u32` for a constant id created via
    /// `const_u32`. `None` if the id wasn't a u32 constant.
    pub fn get_const_u32_value(&self, id: ConstId) -> Option<u32> {
        self.uint_const_reverse.get(&id).copied()
    }

    /// True iff `id` is a module-level constant created via the
    /// builder. Use to decide between `OpConstantComposite`
    /// (all-constant operands) and `OpCompositeConstruct`.
    pub fn is_constant(&self, id: ConstId) -> bool {
        self.constant_ids.contains(&id)
    }

    /// Register a constant id minted outside this struct's typed
    /// methods (composite constants, null constants). Callers emit
    /// the rspirv instruction themselves and report the id here so
    /// `is_constant` returns true for it.
    pub fn register_constant(&mut self, id: ConstId) {
        self.constant_ids.insert(id);
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
