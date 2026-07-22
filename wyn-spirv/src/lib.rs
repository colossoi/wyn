//! Compiler-agnostic typed wrapper around `rspirv`'s `Builder`. Owns
//! the typed `Id<K>` phantoms, the type / constant / decoration dedup
//! caches, and re-exports `rspirv::{binary, dr, spirv}` so consumers
//! don't need a direct rspirv dep.

pub use rspirv::{binary, dr, spirv};

use rspirv::dr::Builder;
use rspirv::spirv::{AddressingModel, Capability, MemoryModel};
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

/// Wrapper around `rspirv::dr::Builder` that maintains the SPIR-V
/// dedup caches (types, constants, decorations) and the function-
/// lifecycle state, all keyed purely on SPIR-V primitives — no
/// compiler types reach this struct.
///
/// `Deref<Target = rspirv::dr::Builder>` is provided so callers can
/// reach raw rspirv methods that this wrapper hasn't typed yet.
/// Dedup key for `OpTypeImage`: its full operand tuple
/// `(sampled_type, dim, depth, arrayed, ms, sampled, format, access)`.
type ImageTypeKey = (
    TypeId,
    spirv::Dim,
    u32,
    u32,
    u32,
    u32,
    spirv::ImageFormat,
    Option<spirv::AccessQualifier>,
);

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
    // Structural type dedup. Each cache is keyed purely on SPIR-V
    // primitives so the wrapper doesn't need to know how the caller
    // organizes types upstream — a `PolyType` → `TypeId` map lives
    // on the lowering layer and calls these helpers with the
    // already-resolved SPIR-V components.
    vec_type_cache: HashMap<(TypeId, u32), TypeId>,
    struct_type_cache: HashMap<Vec<TypeId>, TypeId>,
    ptr_type_cache: HashMap<(spirv::StorageClass, TypeId), TypeId>,
    runtime_array_cache: HashMap<(TypeId, u32), TypeId>, // (elem_type, stride) -> decorated type
    array_type_cache: HashMap<(TypeId, spirv::Word), TypeId>, // (elem_type, length-constant) -> sized array
    image_type_cache: HashMap<ImageTypeKey, TypeId>,
    sampled_image_cache: HashMap<TypeId, TypeId>, // image type -> OpTypeSampledImage
    sampler_type: Option<TypeId>,                 // the single OpTypeSampler
    matrix_type_cache: HashMap<(TypeId, u32), TypeId>, // (column vector, column count) -> OpTypeMatrix
    int_type_cache: HashMap<(u32, u32), TypeId>,  // (width, signedness) -> OpTypeInt
    float_type_cache: HashMap<u32, TypeId>,       // width -> OpTypeFloat
    // Block-decorated struct wrappers around a runtime array (storage
    // buffer) or a single value (uniform / push-constant). Cached per
    // wrapped-type so two `#[uniform]` params of the same shape don't
    // produce double `Block`/`Offset` decorations that spirv-val
    // rejects.
    buffer_block_cache: HashMap<TypeId, TypeId>, // runtime_array_type -> Block-decorated struct
    uniform_block_cache: HashMap<TypeId, TypeId>, // value_type -> Block-decorated struct
    // Struct ids already decorated `Block` + member offsets (once
    // per id). Distinct from the caches above because `rspirv`
    // structurally dedups `OpTypeStruct`s — different wrappers can
    // land on the same id and would re-decorate without this guard.
    block_decorated: HashSet<TypeId>,
    // Composite-constant dedup: emitting the same `OpConstantComposite`
    // twice would produce two distinct ids that wgpu treats as
    // separate constants. Keyed on `(result_type, constituent_ids)`.
    composite_const_cache: HashMap<(TypeId, Vec<ConstId>), ConstId>,
    // OpConstantNull dedup, one id per type.
    null_const_cache: HashMap<TypeId, ConstId>,
    // Compile-time-constant arrays promoted to a module-scope `Private`
    // OpVariable (initializer = the OpConstantComposite), so a runtime
    // index becomes an OpAccessChain into a shared global instead of a
    // per-occurrence Function var + whole-array OpStore. Keyed on the
    // constant id, which `composite_const_cache` already interns by
    // value — so equal arrays (inline literal, local `let`, module
    // `def`, or many inlined copies) collapse to one global.
    private_global_cache: HashMap<ConstId, VarId>,
    // Storage-buffer variables already decorated `NonWritable`. A
    // binding shared across entries (multi-entry modules) reaches
    // the decoration site once per entry with the same var id;
    // spirv-val rejects the repeated decoration.
    nonwritable_decorated: HashSet<VarId>,
    // Types whose buffer-layout decorations (ArrayStride on array
    // types, member offsets on struct-element types) have already
    // been emitted. A single set across both decoration kinds
    // because both fire from the buffer-layout pass and share the
    // "once per type per buffer-layout context" semantics.
    buffer_layout_decorated: HashSet<TypeId>,
    // SPIR-V array → element-type lookup. Populated by callers when
    // they create array types (fixed-size or runtime) so later passes
    // (buffer-layout walks, `with []` lowering) can recover the elem
    // type id without re-deriving from the compiler-side PolyType.
    array_elem: HashMap<TypeId, TypeId>,
    // Forward-declared function ids keyed by name, so a body emission
    // can land on the id its callers already reference.
    functions: HashMap<String, FuncId>,
    // Open-function block layout. Local variables hoist into the
    // variables block; the body emits into the code block; at function
    // close, an unconditional branch wires the former to the latter.
    variables_block: Option<BlockId>,
    first_code_block: Option<BlockId>,
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
            vec_type_cache: HashMap::new(),
            struct_type_cache: HashMap::new(),
            ptr_type_cache: HashMap::new(),
            runtime_array_cache: HashMap::new(),
            array_type_cache: HashMap::new(),
            image_type_cache: HashMap::new(),
            sampled_image_cache: HashMap::new(),
            sampler_type: None,
            matrix_type_cache: HashMap::new(),
            // Seed with the eagerly-created scalar types so `type_int`/`type_float`
            // never re-declare them.
            int_type_cache: HashMap::from([((32, 1), i32_type), ((32, 0), u32_type)]),
            float_type_cache: HashMap::from([(32, f32_type)]),
            buffer_block_cache: HashMap::new(),
            uniform_block_cache: HashMap::new(),
            block_decorated: HashSet::new(),
            composite_const_cache: HashMap::new(),
            null_const_cache: HashMap::new(),
            private_global_cache: HashMap::new(),
            nonwritable_decorated: HashSet::new(),
            buffer_layout_decorated: HashSet::new(),
            array_elem: HashMap::new(),
            functions: HashMap::new(),
            variables_block: None,
            first_code_block: None,
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

    /// Get or create a module-scope `Private` `OpVariable` whose
    /// initializer is `const_id` (an `OpConstantComposite` / `OpConstant`),
    /// with pointee `value_type`. Deduped by `const_id`, so equal
    /// constants — already interned to one id by `composite_const_cache` —
    /// share one global. Lets a dynamic index of a constant array become
    /// an `OpAccessChain` into a single materialization instead of a
    /// per-occurrence Function var + whole-array `OpStore`.
    ///
    /// The variable must land at module scope (`types_global_values`), so
    /// the current block is deselected around the `variable` call — rspirv
    /// routes `OpVariable` there only when no block is selected (the same
    /// reason workgroup-shared globals are emitted between functions).
    pub fn hoist_constant_global(&mut self, const_id: ConstId, value_type: TypeId) -> VarId {
        if let Some(&var) = self.private_global_cache.get(&const_id) {
            return var;
        }
        let ptr_type = self.type_pointer(spirv::StorageClass::Private, value_type);
        let saved = self.inner.selected_block();
        self.inner.select_block(None).expect("deselect block for module-scope variable");
        let var =
            VarId::new(self.inner.variable(*ptr_type, None, spirv::StorageClass::Private, Some(*const_id)));
        self.inner.select_block(saved).expect("restore selected block");
        self.private_global_cache.insert(const_id, var);
        var
    }

    /// All hoisted `Private` constant globals, for `OpEntryPoint`
    /// interface registration (SPIR-V ≥1.4 lists every referenced global).
    pub fn private_globals(&self) -> impl Iterator<Item = VarId> + '_ {
        self.private_global_cache.values().copied()
    }

    /// True iff `var` is a hoisted `Private` constant global — so an access
    /// chain into it must use `StorageClass::Private`, not `Function`.
    pub fn is_private_global(&self, var: VarId) -> bool {
        self.private_global_cache.values().any(|&v| v == var)
    }

    /// Get or create `OpTypeVector elem size`.
    pub fn type_vec(&mut self, elem: TypeId, size: u32) -> TypeId {
        let key = (elem, size);
        if let Some(&ty) = self.vec_type_cache.get(&key) {
            return ty;
        }
        let ty = TypeId::new(self.inner.type_vector(*elem, size));
        self.vec_type_cache.insert(key, ty);
        ty
    }

    /// Get or create `OpTypeStruct field0 field1 …`. Pure structural
    /// dedup — interface blocks (push-constant / uniform / storage
    /// wrappers) need their own dedup keyed on `(kind, members)` so
    /// they never share with a plain struct of the same shape.
    pub fn type_struct(&mut self, fields: Vec<TypeId>) -> TypeId {
        if let Some(&ty) = self.struct_type_cache.get(&fields) {
            return ty;
        }
        let raw_fields: Vec<spirv::Word> = fields.iter().map(|t| **t).collect();
        let ty = TypeId::new(self.inner.type_struct(raw_fields));
        self.struct_type_cache.insert(fields, ty);
        ty
    }

    /// Get or create `OpTypePointer storage_class pointee`.
    pub fn type_pointer(&mut self, storage_class: spirv::StorageClass, pointee: TypeId) -> TypeId {
        let key = (storage_class, pointee);
        if let Some(&ty) = self.ptr_type_cache.get(&key) {
            return ty;
        }
        let ty = TypeId::new(self.inner.type_pointer(None, storage_class, *pointee));
        self.ptr_type_cache.insert(key, ty);
        ty
    }

    /// Get or create `OpTypeRuntimeArray elem`, decorated once with
    /// `ArrayStride stride`.
    pub fn type_runtime_array(&mut self, elem: TypeId, stride: u32) -> TypeId {
        let key = (elem, stride);
        if let Some(&ty) = self.runtime_array_cache.get(&key) {
            return ty;
        }
        let ty = TypeId::new(self.inner.type_runtime_array(*elem));
        self.inner.decorate(
            *ty,
            spirv::Decoration::ArrayStride,
            [rspirv::dr::Operand::LiteralBit32(stride)],
        );
        self.runtime_array_cache.insert(key, ty);
        ty
    }

    /// Get or create a sized `OpTypeArray elem length`, where `length` is a
    /// constant id.
    pub fn type_array(&mut self, elem: TypeId, length: spirv::Word) -> TypeId {
        let key = (elem, length);
        if let Some(&ty) = self.array_type_cache.get(&key) {
            return ty;
        }
        let ty = TypeId::new(self.inner.type_array(*elem, length));
        self.array_type_cache.insert(key, ty);
        ty
    }

    /// Get or create an `OpTypeImage` from its full operand tuple.
    #[allow(clippy::too_many_arguments)]
    pub fn type_image(
        &mut self,
        sampled_type: TypeId,
        dim: spirv::Dim,
        depth: u32,
        arrayed: u32,
        ms: u32,
        sampled: u32,
        format: spirv::ImageFormat,
        access: Option<spirv::AccessQualifier>,
    ) -> TypeId {
        let key = (sampled_type, dim, depth, arrayed, ms, sampled, format, access);
        if let Some(&ty) = self.image_type_cache.get(&key) {
            return ty;
        }
        let ty = TypeId::new(self.inner.type_image(
            *sampled_type,
            dim,
            depth,
            arrayed,
            ms,
            sampled,
            format,
            access,
        ));
        self.image_type_cache.insert(key, ty);
        ty
    }

    /// Get or create `OpTypeSampledImage image`.
    pub fn type_sampled_image(&mut self, image: TypeId) -> TypeId {
        if let Some(&ty) = self.sampled_image_cache.get(&image) {
            return ty;
        }
        let ty = TypeId::new(self.inner.type_sampled_image(*image));
        self.sampled_image_cache.insert(image, ty);
        ty
    }

    /// Get or create the single `OpTypeSampler`.
    pub fn type_sampler(&mut self) -> TypeId {
        if let Some(ty) = self.sampler_type {
            return ty;
        }
        let ty = TypeId::new(self.inner.type_sampler());
        self.sampler_type = Some(ty);
        ty
    }

    /// Get or create `OpTypeMatrix column_type column_count`.
    pub fn type_matrix(&mut self, column_type: TypeId, column_count: u32) -> TypeId {
        let key = (column_type, column_count);
        if let Some(&ty) = self.matrix_type_cache.get(&key) {
            return ty;
        }
        let ty = TypeId::new(self.inner.type_matrix(*column_type, column_count));
        self.matrix_type_cache.insert(key, ty);
        ty
    }

    /// Get or create `OpTypeInt width signedness`.
    pub fn type_int(&mut self, width: u32, signedness: u32) -> TypeId {
        let key = (width, signedness);
        if let Some(&ty) = self.int_type_cache.get(&key) {
            return ty;
        }
        let ty = TypeId::new(self.inner.type_int(width, signedness));
        self.int_type_cache.insert(key, ty);
        ty
    }

    /// Get or create `OpTypeFloat width`.
    pub fn type_float(&mut self, width: u32) -> TypeId {
        if let Some(&ty) = self.float_type_cache.get(&width) {
            return ty;
        }
        let ty = TypeId::new(self.inner.type_float(width));
        self.float_type_cache.insert(width, ty);
        ty
    }

    /// Decorate `ty` as `Block` with the given member byte offsets,
    /// exactly once per struct id. Multiple wrappers (uniform /
    /// push-constant / storage buffer) over the same inner layout
    /// can share an id due to rspirv's structural dedup; this guard
    /// keeps spirv-val happy by not re-decorating.
    pub fn decorate_block_once(&mut self, ty: TypeId, member_offsets: &[u32]) {
        if !self.block_decorated.insert(ty) {
            return;
        }
        self.inner.decorate(*ty, spirv::Decoration::Block, []);
        for (i, &offset) in member_offsets.iter().enumerate() {
            self.inner.member_decorate(
                *ty,
                i as u32,
                spirv::Decoration::Offset,
                [rspirv::dr::Operand::LiteralBit32(offset)],
            );
        }
    }

    /// `{runtime_array}` struct decorated as `Block` with member
    /// offset 0 — the Vulkan storage-buffer block shape. Cached per
    /// `runtime_array` so two storage buffers of the same shape share
    /// one block-wrapped struct id.
    pub fn buffer_block_type(&mut self, runtime_array: TypeId, matrix_stride: Option<u32>) -> TypeId {
        if let Some(&ty) = self.buffer_block_cache.get(&runtime_array) {
            return ty;
        }
        let ty = self.type_struct(vec![runtime_array]);
        self.decorate_block_once(ty, &[0]);
        if let Some(stride) = matrix_stride {
            self.inner.member_decorate(
                *ty,
                0,
                spirv::Decoration::MatrixStride,
                [rspirv::dr::Operand::LiteralBit32(stride)],
            );
            self.inner.member_decorate(
                *ty,
                0,
                spirv::Decoration::ColMajor,
                std::iter::empty::<rspirv::dr::Operand>(),
            );
        }
        self.buffer_block_cache.insert(runtime_array, ty);
        ty
    }

    /// `{value}` struct decorated as `Block` with member offset 0 —
    /// the Vulkan uniform-buffer block shape. Cached per `value` so
    /// two `#[uniform]` params of the same shape share one block.
    pub fn uniform_block_type(&mut self, value: TypeId) -> TypeId {
        if let Some(&ty) = self.uniform_block_cache.get(&value) {
            return ty;
        }
        let ty = self.type_struct(vec![value]);
        self.decorate_block_once(ty, &[0]);
        self.uniform_block_cache.insert(value, ty);
        ty
    }

    /// Emit `OpConstantComposite ty elems…` if all `elems` are
    /// constants minted through this builder, else
    /// `OpCompositeConstruct ty elems…`. Both forms are cached so
    /// repeated builds of the same shape collapse to one id.
    pub fn composite_or_construct(
        &mut self,
        ty: TypeId,
        elems: Vec<spirv::Word>,
    ) -> Result<spirv::Word, rspirv::dr::Error> {
        let elem_const_ids: Option<Vec<ConstId>> = elems
            .iter()
            .map(|&w| {
                let c = ConstId::new(w);
                self.constant_ids.contains(&c).then_some(c)
            })
            .collect();
        if let Some(const_ids) = elem_const_ids {
            let key = (ty, const_ids.clone());
            if let Some(&cached) = self.composite_const_cache.get(&key) {
                return Ok(*cached);
            }
            let id = ConstId::new(
                self.inner.constant_composite(*ty, const_ids.iter().map(|c| **c).collect::<Vec<_>>()),
            );
            self.constant_ids.insert(id);
            self.composite_const_cache.insert(key, id);
            Ok(*id)
        } else {
            self.inner.composite_construct(*ty, None, elems)
        }
    }

    /// Get or create `OpConstantNull ty`.
    pub fn const_null(&mut self, ty: TypeId) -> ConstId {
        if let Some(&id) = self.null_const_cache.get(&ty) {
            return id;
        }
        let id = ConstId::new(self.inner.constant_null(*ty));
        self.null_const_cache.insert(ty, id);
        self.constant_ids.insert(id);
        id
    }

    /// Decorate a storage-buffer variable `NonWritable` exactly once.
    /// Returns true if the decoration was emitted this call (false if
    /// the var was already decorated).
    pub fn decorate_nonwritable_once(&mut self, var: VarId) -> bool {
        if !self.nonwritable_decorated.insert(var) {
            return false;
        }
        self.inner.decorate(
            *var,
            spirv::Decoration::NonWritable,
            std::iter::empty::<rspirv::dr::Operand>(),
        );
        true
    }

    /// Decorate `ty` as `ArrayStride stride` exactly once per buffer-
    /// layout context. Returns true on first decoration, false if `ty`
    /// already saw a buffer-layout decoration (caller treats nested
    /// types in the same chain as already-decorated).
    pub fn decorate_array_stride_once(&mut self, ty: TypeId, stride: u32) -> bool {
        if !self.buffer_layout_decorated.insert(ty) {
            return false;
        }
        self.inner.decorate(
            *ty,
            spirv::Decoration::ArrayStride,
            [rspirv::dr::Operand::LiteralBit32(stride)],
        );
        true
    }

    /// Mark `ty` as having had its buffer-layout decorations applied,
    /// returning true on first mark. Lets callers that don't go
    /// through `decorate_array_stride_once` (member-offset paths)
    /// still participate in the same "once-per-type" dedup.
    pub fn mark_buffer_layout_decorated_once(&mut self, ty: TypeId) -> bool {
        self.buffer_layout_decorated.insert(ty)
    }

    /// Register that SPIR-V array type `arr` has element type `elem`.
    /// Caller emits the array type via rspirv (or via `type_runtime_array`)
    /// and reports the relationship here so subsequent
    /// `array_element_type` lookups can recover the elem id without
    /// re-deriving from the compiler-side PolyType.
    pub fn register_array_element(&mut self, arr: TypeId, elem: TypeId) {
        self.array_elem.insert(arr, elem);
    }

    /// Element type of a previously-registered SPIR-V array type.
    /// `None` if no array of that id has been registered.
    pub fn array_element_type(&self, arr: TypeId) -> Option<TypeId> {
        self.array_elem.get(&arr).copied()
    }

    /// Look up the SPIR-V id of a function declared via
    /// `forward_declare_function` or already opened by `begin_function`.
    pub fn get_function(&self, name: &str) -> Option<FuncId> {
        self.functions.get(name).copied()
    }

    /// Reserve a SPIR-V id for a function whose body comes later.
    /// Idempotent on `name` — repeat calls return the same id, so
    /// callers can resolve cross-function references without ordering
    /// constraints.
    pub fn forward_declare_function(&mut self, name: &str) -> FuncId {
        if let Some(&id) = self.functions.get(name) {
            return id;
        }
        let func_id = FuncId::new(self.inner.id());
        self.functions.insert(name.to_string(), func_id);
        func_id
    }

    /// Emit an extern function stub with `Import` linkage — body is
    /// supplied at link time. Adds the `Linkage` capability on first
    /// use. The id is registered under `name` so subsequent calls
    /// resolve to it.
    pub fn forward_declare_linked_function(
        &mut self,
        name: &str,
        linkage_name: &str,
        param_types: &[TypeId],
        return_type: TypeId,
    ) -> FuncId {
        self.inner.capability(Capability::Linkage);
        let func_type =
            self.inner.type_function(*return_type, param_types.iter().map(|t| **t).collect::<Vec<_>>());
        let func_id = self
            .inner
            .begin_function(*return_type, None, spirv::FunctionControl::NONE, func_type)
            .expect("rspirv begin_function failed for linked function");
        for &param_ty in param_types {
            self.inner.function_parameter(*param_ty).expect("rspirv function_parameter failed");
        }
        self.inner.end_function().expect("rspirv end_function failed for linked function");
        self.inner.decorate(
            func_id,
            spirv::Decoration::LinkageAttributes,
            [
                dr::Operand::LiteralString(linkage_name.to_string()),
                dr::Operand::LinkageType(spirv::LinkageType::Import),
            ],
        );
        let func_id = FuncId::new(func_id);
        self.functions.insert(name.to_string(), func_id);
        func_id
    }

    /// Open a new function. Picks up the forward-declared id under
    /// `name` if one exists, otherwise allocates fresh. Opens both
    /// the variables block (where `declare_variable` hoists locals)
    /// and the first code block (selected as the current emission
    /// target on return).
    ///
    /// Returns `(func_id, param_ids, first_code_block)`.
    pub fn begin_function(
        &mut self,
        name: &str,
        param_types: &[TypeId],
        return_type: TypeId,
    ) -> Result<(FuncId, Vec<spirv::Word>, BlockId), dr::Error> {
        let func_type =
            self.inner.type_function(*return_type, param_types.iter().map(|t| **t).collect::<Vec<_>>());
        let func_id = if let Some(&pre_id) = self.functions.get(name) {
            FuncId::new(self.inner.begin_function(
                *return_type,
                Some(*pre_id),
                spirv::FunctionControl::NONE,
                func_type,
            )?)
        } else {
            let id =
                self.inner.begin_function(*return_type, None, spirv::FunctionControl::NONE, func_type)?;
            let fid = FuncId::new(id);
            self.functions.insert(name.to_string(), fid);
            fid
        };

        let mut param_ids = Vec::with_capacity(param_types.len());
        for &param_ty in param_types {
            param_ids.push(self.inner.function_parameter(*param_ty)?);
        }

        let vars_block_id = BlockId::new(self.inner.id());
        let code_block_id = BlockId::new(self.inner.id());
        self.variables_block = Some(vars_block_id);
        self.first_code_block = Some(code_block_id);

        self.inner.begin_block(Some(*vars_block_id))?;
        self.inner.select_block(None)?;
        self.inner.begin_block(Some(*code_block_id))?;

        Ok((func_id, param_ids, code_block_id))
    }

    /// Close the current function: branch the variables block to the
    /// first code block, emit `OpFunctionEnd`, and clear the
    /// open-function layout state.
    pub fn end_function(&mut self) -> Result<(), dr::Error> {
        if let (Some(vars_block), Some(code_block)) = (self.variables_block, self.first_code_block) {
            let func = self.inner.module_ref().functions.last().expect("end_function: no open function");
            let vars_idx = func
                .blocks
                .iter()
                .position(|b| b.label.as_ref().map(|l| l.result_id) == Some(Some(*vars_block)));
            if let Some(idx) = vars_idx {
                self.inner.select_block(Some(idx))?;
                self.inner.branch(*code_block)?;
            }
        }
        self.inner.end_function()?;
        self.variables_block = None;
        self.first_code_block = None;
        Ok(())
    }

    /// Emit an `OpVariable` of `value_type` into the open function's
    /// variables block, then restore the previously-selected block so
    /// subsequent emission lands where the caller expects.
    pub fn declare_variable(&mut self, value_type: TypeId) -> Result<VarId, dr::Error> {
        let ptr_type = self.type_pointer(spirv::StorageClass::Function, value_type);
        let current_idx = self.inner.selected_block();
        let vars_block = self.variables_block.expect("declare_variable called outside an open function");
        let func = self.inner.module_ref().functions.last().expect("declare_variable: no open function");
        let vars_idx = func
            .blocks
            .iter()
            .position(|b| b.label.as_ref().map(|l| l.result_id) == Some(Some(*vars_block)))
            .expect("declare_variable: variables block not in module");
        self.inner.select_block(Some(vars_idx))?;
        let var_id = self.inner.variable(*ptr_type, None, spirv::StorageClass::Function, None);
        self.inner.select_block(current_idx)?;
        Ok(VarId::new(var_id))
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
#[path = "lib_tests.rs"]
mod tests;
