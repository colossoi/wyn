//! Per-function SSA-body lowering. `LowerCtx::new` → `lower()` is
//! the entry surface; everything else is internal.

use super::*;

/// Per-function SSA → SPIR-V lowering state. Fields are
/// `pub(super)` so sibling-file `impl LowerCtx` blocks
/// (`lower_ops.rs`, `lower_index.rs`, `lower_builtin.rs`) can reach
/// them.
pub(super) struct LowerCtx<'a, 'b> {
    pub(super) constructor: &'a mut Constructor,
    pub(super) body: &'b FuncBody,
    /// True when lowering an entry point (void function — OpReturnValue is invalid).
    pub(super) is_entry_point: bool,
    /// Map from SSA ValueId to SPIR-V Word.
    pub(super) value_map: LookupMap<ValueId, spirv::Word>,
    /// Map from SSA BlockId to SPIR-V block label.
    pub(super) block_map: LookupMap<BlockId, spirv::Word>,
    /// Map from block to its SPIR-V block index (for phi insertion).
    pub(super) block_indices: LookupMap<BlockId, usize>,
    /// Phi node info: (target_block, param_idx, value, source_block)
    /// Collected during terminator lowering, inserted after all blocks processed.
    pub(super) phi_inputs: Vec<(BlockId, usize, spirv::Word, spirv::Word)>,
    /// Map from a `StorageView(Workgroup)` result ValueId to its workgroup
    /// array id (key into `Constructor::workgroup_vars`), so `ViewIndex` can
    /// access-chain into the workgroup variable instead of a storage buffer.
    pub(super) workgroup_view: LookupMap<ValueId, u32>,
    /// Map from a `PlaceId` to the SPIR-V pointer word that addresses it.
    /// Populated by place-producing instructions (`OutputSlot`,
    /// `ViewIndex`, `Alloca`) and read by `Load` / `Store`.
    pub(super) place_ptr_id: LookupMap<crate::ssa::types::PlaceId, spirv::Word>,
    /// Span of the instruction currently being lowered (set by `lower_inst`).
    /// Consumed via `blame_span()` so backend errors blame the source line of
    /// the originating expression.
    pub(super) current_span: Option<Span>,
    /// Function-level span fallback when an instruction has no span.
    pub(super) func_span: Span,
    /// SPIR-V function-parameter ids in declaration order. Empty for
    /// entry points (which take I/O via variables, not params).
    pub(super) param_ids: Vec<spirv::Word>,
    /// SPIR-V label of the block currently being emitted into. Updated
    /// by `begin_block()` as the lowering walk advances; read by phi-
    /// tracking in `lower_terminator`.
    pub(super) current_block: spirv::Word,
}

impl<'a, 'b> LowerCtx<'a, 'b> {
    pub(super) fn new(
        constructor: &'a mut Constructor,
        body: &'b FuncBody,
        is_entry_point: bool,
        func_span: Span,
        param_ids: Vec<spirv::Word>,
        first_code_block: spirv::Word,
    ) -> Self {
        LowerCtx {
            constructor,
            body,
            is_entry_point,
            value_map: LookupMap::new(),
            block_map: LookupMap::new(),
            block_indices: LookupMap::new(),
            phi_inputs: Vec::new(),
            workgroup_view: LookupMap::new(),
            place_ptr_id: LookupMap::new(),
            current_span: None,
            func_span,
            param_ids,
            current_block: first_code_block,
        }
    }

    /// Begin a SPIR-V block and track it as the current emission target.
    pub(super) fn begin_block(&mut self, block_id: spirv::Word) -> Result<()> {
        self.constructor.builder.begin_block(Some(block_id))?;
        self.current_block = block_id;
        Ok(())
    }

    /// SPIR-V pointer word for a `PlaceId` — set by the defining instruction
    /// (`OutputSlot`, `ViewIndex`, `Alloca`), consumed by `Load` / `Store`.
    pub(super) fn place_ptr(&self, place: crate::ssa::types::PlaceId) -> Result<spirv::Word> {
        self.place_ptr_id.get(&place).copied().ok_or_else(|| {
            err_spirv_at!(
                self.blame_span(),
                "SPIR-V: place {:?} has no pointer — its defining instruction \
                 was not lowered (or ran after a consumer)",
                place
            )
        })
    }

    /// Source span that blames an instruction's lowering errors. Falls back
    /// to the function span when the instruction has no span of its own.
    pub(super) fn blame_span(&self) -> Span {
        self.current_span.unwrap_or(self.func_span)
    }

    pub(super) fn lower(&mut self) -> Result<spirv::Word> {
        // Map function parameters to their SPIR-V values.
        // For regular functions, use positional mapping (param_ids) to avoid
        // name collisions when two params share a string name.
        // For entry points (no param_ids), fall back to name-based env lookup.
        if self.param_ids.len() == self.body.params().len() && !self.param_ids.is_empty() {
            for (i, (value_id, _, _)) in self.body.params().enumerate() {
                self.value_map.insert(value_id, self.param_ids[i]);
            }
        } else {
            for (value_id, _, name) in self.body.params() {
                if let Some(&spirv_id) = self.constructor.env.get(name) {
                    self.value_map.insert(value_id, spirv_id);
                }
            }
        }

        // Create all SPIR-V blocks and pre-allocate phi IDs for all block params.
        // Phi IDs must be allocated up front because SSA values from one block's
        // params may be referenced by instructions in other blocks (e.g., an if/else
        // result used in an array literal that spans multiple merge blocks).
        let entry_block = self.body.entry_block();
        for (block_id, block) in &self.body.inner.blocks {
            if block.insts.is_empty() && matches!(block.term, Terminator::Unreachable) {
                continue;
            }
            if block_id == entry_block {
                self.block_map.insert(block_id, self.current_block);
            } else {
                let spirv_block = self.constructor.builder.id();
                self.block_map.insert(block_id, spirv_block);
            }

            for &param in &block.params {
                let phi_id = self.constructor.builder.id();
                self.value_map.insert(param, phi_id);
            }
        }

        let rpo = self.compute_rpo();

        for &block_id in &rpo {
            let block = &self.body.inner.blocks[block_id];

            if block_id != entry_block {
                let spirv_block = self.block_map[&block_id];
                self.begin_block(spirv_block)?;
            }

            // Record block index for phi insertion
            if let Some(idx) = self.constructor.builder.selected_block() {
                self.block_indices.insert(block_id, idx);
            }

            // Lower instructions
            for &inst_id in &block.insts {
                let inst = self.body.get_inst(inst_id);
                self.lower_inst(inst).map_err(|e| err_spirv!("Block({:?}): {}", block_id, e))?;
            }

            // Lower terminator
            self.lower_terminator(block_id, block, &block.term)?;
        }

        // Insert phi nodes for all block parameters
        self.insert_phi_nodes()?;

        // Return placeholder - actual return handled by terminators in SSA
        Ok(self.constructor.const_i32(0))
    }

    /// Compute a structured block ordering for SPIR-V emission.
    ///
    /// SPIR-V requires that all blocks in a loop/selection construct appear
    /// between the header and the merge block. A plain RPO can violate this
    /// (e.g. placing a loop's merge before its continue block). This traversal
    /// defers merge blocks until after all construct-interior blocks are visited.
    pub(super) fn compute_rpo(&self) -> Vec<BlockId> {
        let mut visited: LookupSet<BlockId> = LookupSet::new();
        let mut order = Vec::with_capacity(self.body.inner.blocks.len());

        fn visit(
            body: &FuncBody,
            bid: BlockId,
            visited: &mut LookupSet<BlockId>,
            order: &mut Vec<BlockId>,
        ) {
            if visited.contains(&bid) {
                return;
            }
            let block = &body.inner.blocks[bid];
            if block.insts.is_empty() && matches!(block.term, Terminator::Unreachable) {
                return;
            }
            visited.insert(bid);
            order.push(bid);

            let merge_bid = block.control_header.as_ref().map(|ctrl| match ctrl {
                ControlHeader::Loop { merge, .. } => *merge,
                ControlHeader::Selection { merge } => *merge,
            });

            match &block.term {
                Terminator::Branch { target, .. } => {
                    if Some(*target) != merge_bid {
                        visit(body, *target, visited, order);
                    }
                }
                Terminator::CondBranch {
                    then_target,
                    else_target,
                    ..
                } => {
                    if Some(*then_target) != merge_bid {
                        visit(body, *then_target, visited, order);
                    }
                    if Some(*else_target) != merge_bid {
                        visit(body, *else_target, visited, order);
                    }
                }
                _ => {}
            }

            if let Some(m) = merge_bid {
                visit(body, m, visited, order);
            }
        }

        visit(self.body, self.body.entry_block(), &mut visited, &mut order);
        order
    }

    pub(super) fn lower_inst(&mut self, inst: &WynInstNode) -> Result<()> {
        let ssa_result_ty = inst.result.map(|r| self.body.inner.value_type(r).clone());
        let result_ty = ssa_result_ty.as_ref().map(|t| self.constructor.polytype_to_spirv(t)).unwrap_or(0);
        self.current_span = inst.span;

        let spirv_result = match &inst.data {
            InstKind::Op { tag, operands } => match tag {
                crate::op::OpTag::ResourceLen(_) => {
                    panic!("logical resource length reached SPIR-V lowering")
                }
                crate::op::OpTag::Int(s) | crate::op::OpTag::Uint(s) => match ssa_result_ty.as_ref() {
                    Some(PolyType::Constructed(TypeName::UInt(32), _)) => {
                        let val: u32 = s
                            .parse()
                            .map_err(|_| err_spirv_at!(self.blame_span(), "Invalid u32: {}", s))?;
                        self.constructor.const_u32(val)
                    }
                    _ => {
                        let val: i32 = s
                            .parse()
                            .map_err(|_| err_spirv_at!(self.blame_span(), "Invalid i32: {}", s))?;
                        self.constructor.const_i32(val)
                    }
                },

                crate::op::OpTag::Float(s) => {
                    let val: f32 =
                        s.parse().map_err(|_| err_spirv_at!(self.blame_span(), "Invalid f32: {}", s))?;
                    self.constructor.const_f32(val)
                }

                crate::op::OpTag::Bool(b) => self.constructor.const_bool(*b),

                crate::op::OpTag::Unit => {
                    unreachable!(
                        "OpTag::Unit should never reach SPIR-V codegen; unit values are not materializable"
                    )
                }

                crate::op::OpTag::BinOp(op) => {
                    let lhs = operands[0];
                    let rhs = operands[1];
                    let lhs_id = self.get_value_ref(lhs)?;
                    let rhs_id = self.get_value_ref(rhs)?;
                    let lhs_ty = self.get_value_type_ref(lhs);
                    let rhs_ty = self.get_value_type_ref(rhs);
                    self.lower_binop(op, lhs_id, rhs_id, &lhs_ty, &rhs_ty, result_ty)?
                }

                crate::op::OpTag::UnaryOp(op) => {
                    let operand = operands[0];
                    let operand_id = self.get_value_ref(operand)?;
                    let operand_ty = self.get_value_type_ref(operand);
                    self.lower_unaryop(op, operand_id, &operand_ty, result_ty)?
                }

                crate::op::OpTag::Tuple(_) => {
                    let elem_ids: Vec<_> =
                        operands.iter().map(|v| self.get_value_ref(*v)).collect::<Result<_>>()?;
                    self.constructor.composite_or_constant(result_ty, elem_ids)?
                }

                crate::op::OpTag::ArrayLit(_) => {
                    let elem_ids: Vec<_> =
                        operands.iter().map(|v| self.get_value_ref(*v)).collect::<Result<_>>()?;
                    self.constructor.composite_or_constant(result_ty, elem_ids)?
                }

                crate::op::OpTag::ArrayRange { has_step } => {
                    // Virtual array represented as {start, step, len} struct
                    // This matches the layout expected by lower_virtual_index
                    let start_id = self.get_value_ref(operands[0])?;
                    let len_id = self.get_value_ref(operands[1])?;
                    let step_id = if *has_step {
                        self.get_value_ref(operands[2])?
                    } else {
                        // Default step = 1, matching the element type of the range.
                        let elem_ty = ssa_result_ty.as_ref().and_then(|t| t.elem_type());
                        if matches!(elem_ty, Some(PolyType::Constructed(TypeName::UInt(_), _))) {
                            self.constructor.const_u32(1)
                        } else {
                            self.constructor.const_i32(1)
                        }
                    };

                    // Construct the struct: {start, step, len}
                    self.constructor.builder.composite_construct(
                        result_ty,
                        None,
                        vec![start_id, step_id, len_id],
                    )?
                }

                crate::op::OpTag::Vector(_) => {
                    let elem_ids: Vec<_> =
                        operands.iter().map(|v| self.get_value_ref(*v)).collect::<Result<_>>()?;
                    self.constructor.composite_or_constant(result_ty, elem_ids)?
                }

                crate::op::OpTag::Matrix { .. } => {
                    // Matrix is constructed as an array of vectors (columns)
                    // For now, flatten and construct
                    let all_elems: Vec<_> =
                        operands.iter().map(|v| self.get_value_ref(*v)).collect::<Result<_>>()?;
                    self.constructor.composite_or_constant(result_ty, all_elems)?
                }

                crate::op::OpTag::Project { index } => {
                    let base = operands[0];
                    let base_ty = self.get_value_type_ref(base);
                    let base_id = self.get_value_ref(base)?;

                    // If base is a pointer, load it first
                    let composite_id = if types::is_pointer(&base_ty) {
                        let pointee_ty = types::pointee(&base_ty).expect("Pointer should have pointee");
                        let value_type = self.constructor.polytype_to_spirv(pointee_ty);
                        self.constructor.builder.load(value_type, None, base_id, None, [])?
                    } else {
                        base_id
                    };

                    self.constructor.builder.composite_extract(result_ty, None, composite_id, [*index])?
                }

                crate::op::OpTag::Index => self.lower_index(operands[0], operands[1], result_ty)?,

                crate::op::OpTag::Call(func) => {
                    let args: Vec<ValueRef> = operands.clone();
                    let arg_ids: Vec<_> =
                        args.iter().map(|v| self.get_value_ref(*v)).collect::<Result<_>>()?;
                    // User-defined functions shadow same-named catalog
                    // builtins. Most catalog names that flow through here
                    // are specialized per-type ops (`f32.clamp`,
                    // `_w_intrinsic_*` etc.) that can't collide with user
                    // identifiers — but plain names like `step` can, and
                    // checking the user table first lets the user's `def
                    // step` win the resolution.
                    let emitted = self.constructor.emitted_function_name(func).to_string();
                    if let Some(func_id) = self.constructor.builder.get_function(&emitted) {
                        self.constructor.builder.function_call(result_ty, None, *func_id, arg_ids)?
                    } else if let Some(def) = catalog().lookup_by_any_name(func) {
                        let builtin_impl = &def.overloads()[0].lowering;
                        self.lower_builtin_call(
                            def.id,
                            builtin_impl,
                            func,
                            &args,
                            &arg_ids,
                            result_ty,
                            inst,
                        )?
                    } else {
                        bail_spirv_at!(self.blame_span(), "Unknown function: {}", func)
                    }
                }

                crate::op::OpTag::Global(name) => {
                    if let Some(func_id) = self.constructor.builder.get_function(name) {
                        // Global constant function - call it with no args to get the value.
                        // This handles `def verts: [3]vec4f32 = [...]` referenced as just `verts`.
                        self.constructor.builder.function_call(result_ty, None, *func_id, [])?
                    } else {
                        bail_spirv_at!(self.blame_span(), "Unknown global: {}", name)
                    }
                }

                crate::op::OpTag::Extern(linkage_name) => {
                    self.constructor.linked_functions.get(linkage_name).copied().ok_or_else(|| {
                        err_spirv_at!(self.blame_span(), "Unknown extern: {}", linkage_name)
                    })?
                }

                crate::op::OpTag::Intrinsic { id, overload_idx } => {
                    let args: Vec<ValueRef> = operands.clone();
                    let arg_ids: Vec<_> =
                        args.iter().map(|v| self.get_value_ref(*v)).collect::<Result<_>>()?;
                    let def = crate::builtins::by_id(*id);
                    let lowering = &def.overloads()[*overload_idx].lowering;
                    // Variants with a structural arm in `lower_builtin_call`
                    // dispatch via the BuiltinLowering value or the entry
                    // id; the rest still fall through to the name-keyed
                    // `lower_intrinsic` until they're promoted.
                    let known = catalog().known();
                    let typed_dispatch = matches!(
                        lowering,
                        BuiltinLowering::PrimOp(_)
                            | BuiltinLowering::LinkedSpirv(_)
                            | BuiltinLowering::ExtInstSplat { .. }
                    ) || (matches!(lowering, BuiltinLowering::ByBuiltinId)
                        && (*id == known.slice
                            || *id == known.storage_len
                            || *id == known.thread_id
                            || *id == known.local_id
                            || *id == known.num_workgroups
                            || *id == known.length
                            || *id == known.uninit
                            || *id == known.array_with
                            || *id == known.array_with_in_place
                            || *id == known.texture_load
                            || *id == known.texture_sample));
                    if typed_dispatch {
                        self.lower_builtin_call(
                            *id,
                            lowering,
                            def.dispatch_name(),
                            &args,
                            &arg_ids,
                            result_ty,
                            inst,
                        )?
                    } else {
                        bail_spirv!(
                            "OpTag::Intrinsic with no SPIR-V backend dispatch: '{}' \
                             (id={:?}, lowering={:?}). HOF / SOAC intrinsics should be \
                             lowered at EGIR; everything else needs an arm in \
                             lower_builtin_call and an entry in the typed_dispatch list.",
                            def.dispatch_name(),
                            id,
                            lowering
                        )
                    }
                }

                crate::op::OpTag::StorageImageLoad(binding) => {
                    let coord = self.get_value_ref(operands[0])?;
                    let image = self.constructor.load_storage_image(*binding)?;
                    self.constructor.builder.image_read(result_ty, None, image, coord, None, [])?
                }

                crate::op::OpTag::StorageImageStore(binding) => {
                    let coord = self.get_value_ref(operands[0])?;
                    let texel = self.get_value_ref(operands[1])?;
                    let image = self.constructor.load_storage_image(*binding)?;
                    self.constructor.builder.image_write(image, coord, texel, None, [])?;
                    self.constructor.const_i32(0)
                }

                crate::op::OpTag::StorageView(src) => {
                    let offset = operands[0];
                    let len = operands[1];
                    let offset_id = self.get_value_ref(offset)?;
                    let len_id = self.get_value_ref(len)?;

                    match src {
                        crate::op::PureViewSource::Storage(br) => {
                            let (set, binding) = (&br.set, &br.binding);
                            // The descriptor rides in the result value's type
                            // (`Buffer(set, binding)`); consumers recover the
                            // buffer var from there. Validate the binding is a
                            // declared storage buffer and build the {offset,len}
                            // struct.
                            if self.constructor.storage_buffer(BindingRef::new(*set, *binding)).is_some() {
                                let u32_ty = self.constructor.u32_type;
                                let view_struct_type =
                                    self.constructor.get_or_create_struct_type(vec![u32_ty, u32_ty]);
                                self.constructor.builder.composite_construct(
                                    view_struct_type,
                                    None,
                                    [offset_id, len_id],
                                )?
                            } else {
                                bail_spirv_at!(
                                    self.blame_span(),
                                    "Unknown storage buffer: set={}, binding={}",
                                    set,
                                    binding
                                )
                            }
                        }
                        crate::op::PureViewSource::Inherited => {
                            let parent =
                                operands[2].as_ssa().expect("StorageView Inherited parent must be SSA");
                            let parent_id = self.get_value(parent)?;
                            let u32_ty = self.constructor.u32_type;
                            let view_struct_type =
                                self.constructor.get_or_create_struct_type(vec![u32_ty, u32_ty]);

                            // Extract parent_offset (field 0) from parent view
                            let parent_offset =
                                self.constructor.builder.composite_extract(u32_ty, None, parent_id, [0])?;

                            // new_offset = parent_offset + offset
                            let new_offset =
                                self.constructor.builder.i_add(u32_ty, None, parent_offset, offset_id)?;

                            self.constructor.builder.composite_construct(
                                view_struct_type,
                                None,
                                [new_offset, len_id],
                            )?
                        }
                        crate::op::PureViewSource::Workgroup { id, .. } => {
                            // The workgroup var was created in entry setup.
                            // Record the view→id mapping so ViewIndex chains
                            // into it; the {offset, len} struct is built the
                            // same way as a storage view.
                            if let Some(result) = inst.result {
                                self.workgroup_view.insert(result, *id);
                            }
                            let u32_ty = self.constructor.u32_type;
                            let view_struct_type =
                                self.constructor.get_or_create_struct_type(vec![u32_ty, u32_ty]);
                            self.constructor.builder.composite_construct(
                                view_struct_type,
                                None,
                                [offset_id, len_id],
                            )?
                        }
                    }
                }

                crate::op::OpTag::StorageViewLen => {
                    let view = operands[0];
                    let view_id = self.get_value_ref(view)?;
                    // Extract len from view struct (field 1 in {offset, len})
                    self.constructor.builder.composite_extract(result_ty, None, view_id, [1u32])?
                }

                crate::op::OpTag::Materialize => {
                    let value = operands[0];
                    let value_id = self.get_value_ref(value)?;
                    let value_ty = self.get_value_type_ref(value);
                    let spirv_type = self.constructor.polytype_to_spirv(&value_ty);
                    if self.constructor.builder.is_constant(builder::ConstId::new(value_id)) {
                        // A compile-time-constant array (e.g. a literal table)
                        // materialized only so a runtime index can address it:
                        // hoist it once to a module-scope `Private` global
                        // (initializer = the `OpConstantComposite`, deduped by
                        // constant id) instead of an `OpStore` of the whole
                        // array into a per-occurrence `Function` variable. The
                        // same constant — inlined across many sites — collapses
                        // to one global. `DynamicExtract` chains through it with
                        // `StorageClass::Private`.
                        *self.constructor.builder.hoist_constant_global(
                            builder::ConstId::new(value_id),
                            builder::TypeId::new(spirv_type),
                        )
                    } else {
                        let var = self.constructor.declare_variable("_materialize", spirv_type)?;
                        self.constructor.builder.store(var, value_id, None, [])?;
                        var
                    }
                }

                crate::op::OpTag::DynamicExtract => {
                    let base = operands[0];
                    let index = operands[1];
                    let base_var = self.get_value_ref(base)?;
                    let index_id = self.get_value_ref(index)?;
                    let base_ty = self.get_value_type_ref(base);
                    // For a Bounded base, the underlying value is a struct
                    // `{buffer: [N]T, len: i32}`, so the access chain has to
                    // index member 0 first (constant) before the dynamic
                    // index reaches the array element. Other variants
                    // (Composite/View/Virtual) chain directly to the
                    // element.
                    // A constant base hoisted by `Materialize` lives in a
                    // `Private` global; everything else is a `Function` var.
                    // The access-chain pointer's storage class must match.
                    let storage_class =
                        if self.constructor.builder.is_private_global(builder::VarId::new(base_var)) {
                            spirv::StorageClass::Private
                        } else {
                            spirv::StorageClass::Function
                        };
                    let elem_ptr_type = *self
                        .constructor
                        .builder
                        .type_pointer(storage_class, builder::TypeId::new(result_ty));
                    let elem_ptr = if matches!(
                        base_ty.array_variant(),
                        Some(PolyType::Constructed(TypeName::ArrayVariantBounded, _))
                    ) {
                        let zero = self.constructor.const_u32(0);
                        self.constructor.builder.access_chain(
                            elem_ptr_type,
                            None,
                            base_var,
                            [zero, index_id],
                        )?
                    } else {
                        self.constructor.builder.access_chain(elem_ptr_type, None, base_var, [index_id])?
                    };
                    self.constructor.builder.load(result_ty, None, elem_ptr, None, [])?
                }

                crate::op::OpTag::ViewIndex
                | crate::op::OpTag::PlaceIndex
                | crate::op::OpTag::OutputSlot { .. } => {
                    unreachable!("OpTag::{:?} is EGIR-only and must not reach SSA backend", tag)
                }
            },

            InstKind::Alloca { elem_ty, result } => {
                let elem_spirv_ty = self.constructor.polytype_to_spirv(elem_ty);
                let ptr = self.constructor.declare_variable("_alloca", elem_spirv_ty)?;
                self.place_ptr_id.insert(*result, ptr);
                // Void instruction — no value result; return a harmless dummy.
                self.constructor.const_i32(0)
            }

            InstKind::Load { place } => {
                let ptr_id = self.place_ptr(*place)?;
                self.constructor.builder.load(result_ty, None, ptr_id, None, [])?
            }

            InstKind::Store { place, value } => {
                let ptr_id = self.place_ptr(*place)?;
                let val_id = self.get_value_ref(*value)?;
                self.constructor.builder.store(ptr_id, val_id, None, [])?;
                // Store doesn't produce a value, but we return dummy
                self.constructor.const_i32(0)
            }

            InstKind::ViewIndex { view, index, result } => {
                let view_id = self.get_value_ref(*view)?;
                let index_id = self.get_value_ref(*index)?;
                let u32_ty = self.constructor.u32_type;

                // Extract offset (field 0) from view struct {offset, len}.
                // The backing buffer comes from the view type's region.
                let base_offset =
                    self.constructor.builder.composite_extract(u32_ty, None, view_id, [0u32])?;

                let view_ssa = view.as_ssa().ok_or_else(|| {
                    err_spirv_at!(self.blame_span(), "ViewIndex view operand must be SSA")
                })?;

                // Workgroup-shared view: access-chain into the module-scope
                // `array<T, count>` in Workgroup storage (no wrapping block
                // struct, so the chain is just `[offset + index]`).
                if let Some(&wg_id) = self.workgroup_view.get(&view_ssa) {
                    let (wg_var, elem_ty_id) =
                        *self.constructor.workgroup_vars.get(&wg_id).ok_or_else(|| {
                            err_spirv_at!(self.blame_span(), "ViewIndex: unknown workgroup id {}", wg_id)
                        })?;
                    let actual_index =
                        self.constructor.builder.i_add(u32_ty, None, base_offset, index_id)?;
                    let elem_ptr_type =
                        self.constructor.get_or_create_ptr_type(spirv::StorageClass::Workgroup, elem_ty_id);
                    let ptr = self.constructor.builder.access_chain(
                        elem_ptr_type,
                        None,
                        wg_var,
                        [actual_index],
                    )?;
                    self.place_ptr_id.insert(*result, ptr);
                    return Ok(());
                }

                let buffer_var = self.view_buffer_var(view_ssa)?;

                let actual_index = self.constructor.builder.i_add(u32_ty, None, base_offset, index_id)?;
                let zero = self.constructor.const_i32(0);
                // Infer element SPIR-V type from the place's elem_ty — the
                // place's type is what `Load` will return / `Store` writes.
                let place_elem = self.body.place_elem_ty(*result).clone();
                let elem_ty_id = self.constructor.polytype_to_spirv(&place_elem);
                let elem_ptr_type =
                    self.constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, elem_ty_id);
                let ptr = self.constructor.builder.access_chain(
                    elem_ptr_type,
                    None,
                    buffer_var,
                    [zero, actual_index],
                )?;
                self.place_ptr_id.insert(*result, ptr);
                // Void instruction.
                self.constructor.const_i32(0)
            }

            InstKind::PlaceIndex { place, index, result } => {
                // Take an existing place (an `Alloca`'d Function-scope array,
                // or any other indexable place) and index into it to produce
                // a sub-place addressing one element. Lowers to OpAccessChain
                // on the base variable.
                let base_ptr = self.place_ptr(*place)?;
                let index_id = self.get_value_ref(*index)?;
                let place_elem = self.body.place_elem_ty(*result).clone();
                let elem_ty_id = self.constructor.polytype_to_spirv(&place_elem);
                let elem_ptr_type =
                    self.constructor.get_or_create_ptr_type(spirv::StorageClass::Function, elem_ty_id);
                let ptr =
                    self.constructor.builder.access_chain(elem_ptr_type, None, base_ptr, [index_id])?;
                self.place_ptr_id.insert(*result, ptr);
                // Void instruction.
                self.constructor.const_i32(0)
            }

            InstKind::OutputSlot { index, result } => {
                // Each output was wired up in `lower_ssa_entry_point`; bind
                // the place to its output variable pointer.
                if *index >= self.constructor.current_entry_outputs.len() {
                    bail_spirv_at!(
                        self.blame_span(),
                        "Output index {} out of bounds (have {} outputs)",
                        index,
                        self.constructor.current_entry_outputs.len()
                    );
                }
                let ptr = self.constructor.current_entry_outputs[*index];
                self.place_ptr_id.insert(*result, ptr);
                // Void instruction.
                self.constructor.const_i32(0)
            }
            InstKind::ControlBarrier => {
                // Workgroup execution + memory barrier: synchronize the
                // workgroup and make workgroup-shared writes visible.
                let wg_scope = self.constructor.const_u32(spirv::Scope::Workgroup as u32);
                let semantics = self.constructor.const_u32(
                    (spirv::MemorySemantics::WORKGROUP_MEMORY | spirv::MemorySemantics::ACQUIRE_RELEASE)
                        .bits(),
                );
                self.constructor.builder.control_barrier(wg_scope, wg_scope, semantics)?;
                // Void instruction.
                self.constructor.const_i32(0)
            }
        };

        if let Some(result_value) = inst.result {
            self.value_map.insert(result_value, spirv_result);
        }

        Ok(())
    }

    pub(super) fn lower_terminator(
        &mut self,
        _block_id: BlockId,
        _block: &crate::ssa::framework::BasicBlock,
        term: &Terminator,
    ) -> Result<()> {
        let current_block = self.current_block;

        match term {
            Terminator::Branch { target, args } => {
                for (param_idx, &arg) in args.iter().enumerate() {
                    let arg_id = self.get_value(arg)?;
                    self.phi_inputs.push((*target, param_idx, arg_id, current_block));
                }

                let target_label = self.block_map[target];
                self.constructor.builder.branch(target_label)?;
            }

            Terminator::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => {
                let cond_id = self.get_value(*cond)?;
                let then_label = self.block_map[then_target];
                let else_label = self.block_map[else_target];

                for (param_idx, &arg) in then_args.iter().enumerate() {
                    let arg_id = self.get_value(arg)?;
                    self.phi_inputs.push((*then_target, param_idx, arg_id, current_block));
                }
                for (param_idx, &arg) in else_args.iter().enumerate() {
                    let arg_id = self.get_value(arg)?;
                    self.phi_inputs.push((*else_target, param_idx, arg_id, current_block));
                }

                // Emit structured control flow merge instructions if this is a header block
                if let Some(control) = &self.body.inner.blocks[_block_id].control_header {
                    match control {
                        ControlHeader::Loop {
                            merge,
                            continue_block,
                        } => {
                            let merge_label = self.block_map[merge];
                            let continue_label = self.block_map[continue_block];
                            self.constructor.builder.loop_merge(
                                merge_label,
                                continue_label,
                                spirv::LoopControl::NONE,
                                [],
                            )?;
                        }
                        ControlHeader::Selection { merge } => {
                            let merge_label = self.block_map[merge];
                            self.constructor
                                .builder
                                .selection_merge(merge_label, spirv::SelectionControl::NONE)?;
                        }
                    }
                }

                self.constructor.builder.branch_conditional(cond_id, then_label, else_label, [])?;
            }

            Terminator::Return(Some(value)) => {
                if self.is_entry_point {
                    bail_spirv!(
                        "Return(value) in entry point body — entry points are void functions \
                         and must use OutputPtr+Store then ReturnUnit"
                    );
                }
                let value_id = self.get_value(*value)?;
                self.constructor.builder.ret_value(value_id)?;
            }

            Terminator::Return(None) => {
                self.constructor.builder.ret()?;
            }

            Terminator::Unreachable => {
                self.constructor.builder.unreachable()?;
            }
        }

        Ok(())
    }

    pub(super) fn insert_phi_nodes(&mut self) -> Result<()> {
        // Group phi inputs by (target_block, param_idx)
        let mut phi_map: LookupMap<(BlockId, usize), Vec<(spirv::Word, spirv::Word)>> = LookupMap::new();

        for (target_block, param_idx, value, source_block) in &self.phi_inputs {
            phi_map.entry((*target_block, *param_idx)).or_default().push((*value, *source_block));
        }

        // Insert phi nodes
        for ((block_id, param_idx), incoming) in phi_map {
            let block = &self.body.inner.blocks[block_id];
            let param = block.params[param_idx];
            let param_ty = self.constructor.polytype_to_spirv(self.body.inner.value_type(param));

            let phi_id = self.value_map[&param];

            // Get block index for insertion
            if let Some(&block_idx) = self.block_indices.get(&block_id) {
                self.constructor.builder.select_block(Some(block_idx))?;
                self.constructor.builder.insert_phi(
                    InsertPoint::Begin,
                    param_ty,
                    Some(phi_id),
                    incoming,
                )?;
                self.constructor.builder.select_block(None)?;
            }
        }

        Ok(())
    }

    pub(super) fn get_value(&self, value: ValueId) -> Result<spirv::Word> {
        self.value_map.get(&value).copied().ok_or_else(|| {
            // Build diagnostic info to help debug SSA/lowering issues
            let producer_block = self
                .body
                .inner
                .insts
                .values()
                .find(|i| i.result == Some(value))
                .map(|i| format!("produced in Block({:?})", i.parent));
            let block_param = self.body.inner.blocks.iter().find_map(|(bid, b)| {
                b.params.contains(&value).then(|| format!("block param of Block({:?})", bid))
            });
            let origin = producer_block.or(block_param).unwrap_or_else(|| "not found in body".to_string());
            err_spirv!("Unknown SSA value: {:?} ({})", value, origin)
        })
    }

    pub(super) fn get_value_ref(&mut self, vr: ValueRef) -> Result<spirv::Word> {
        match vr {
            ValueRef::Ssa(id) => self.get_value(id),
            ValueRef::Const(c) => match c {
                ConstantValue::I32(v) => Ok(self.constructor.const_i32(v)),
                ConstantValue::U32(v) => Ok(self.constructor.const_u32(v)),
                ConstantValue::F32(bits) => Ok(self.constructor.const_f32(f32::from_bits(bits))),
                ConstantValue::Bool(b) => Ok(self.constructor.const_bool(b)),
            },
        }
    }

    pub(super) fn get_value_type_ref(&self, vr: ValueRef) -> PolyType<TypeName> {
        match vr {
            ValueRef::Ssa(id) => self.body.get_value_type(id).clone(),
            ValueRef::Const(c) => match c {
                ConstantValue::I32(_) => PolyType::Constructed(TypeName::Int(32), vec![]),
                ConstantValue::U32(_) => PolyType::Constructed(TypeName::UInt(32), vec![]),
                ConstantValue::F32(_) => PolyType::Constructed(TypeName::Float(32), vec![]),
                ConstantValue::Bool(_) => PolyType::Constructed(TypeName::Bool, vec![]),
            },
        }
    }
}
