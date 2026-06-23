//! Per-function SSA-body lowering — `LowerCtx` and its huge `impl`
//! that walks instructions and emits SPIR-V.
//!
//! Held together: the per-instruction `lower_inst()` dispatch
//! (`OpTag::*` arms), terminator + phi-node insertion, array /
//! slice / view indexing, binary / unary / primop arithmetic,
//! the builtin call surface (reduce / scatter / gather / texture
//! / extern intrinsics), and the value / block-id mapping helpers.
//!
//! Public API surface (everything else is private to this file):
//! - `LowerCtx::new(constructor, body, is_entry, span)`
//! - `LowerCtx::lower(self)` — drains the body, returns the SPIR-V
//!   function id.

use super::*;

/// Context for lowering SSA to SPIR-V.
///
/// Fields are `pub(super)` so sibling files (`lower_builtin.rs`,
/// `lower_ops.rs`, `lower_index.rs`) hosting `impl LowerCtx` blocks
/// can read and mutate them. Treat the struct as private to
/// `mod spirv` — outside callers reach it only through
/// `LowerCtx::new` / `LowerCtx::lower`.
pub(super) struct LowerCtx<'a, 'b> {
    pub(super) constructor: &'a mut Constructor,
    pub(super) body: &'b FuncBody,
    /// True when lowering an entry point (void function — OpReturnValue is invalid).
    pub(super) is_entry_point: bool,
    /// Map from SSA ValueId to SPIR-V Word.
    pub(super) value_map: HashMap<ValueId, spirv::Word>,
    /// Map from SSA BlockId to SPIR-V block label.
    pub(super) block_map: HashMap<BlockId, spirv::Word>,
    /// Map from block to its SPIR-V block index (for phi insertion).
    pub(super) block_indices: HashMap<BlockId, usize>,
    /// Phi node info: (target_block, param_idx, value, source_block)
    /// Collected during terminator lowering, inserted after all blocks processed.
    pub(super) phi_inputs: Vec<(BlockId, usize, spirv::Word, spirv::Word)>,
    /// Map from a `StorageView(Workgroup)` result ValueId to its workgroup
    /// array id (key into `Constructor::workgroup_vars`), so `ViewIndex` can
    /// access-chain into the workgroup variable instead of a storage buffer.
    pub(super) workgroup_view: HashMap<ValueId, u32>,
    /// Map from a `PlaceId` to the SPIR-V pointer word that addresses it.
    /// Populated by place-producing instructions (`OutputSlot`,
    /// `ViewIndex`, `Alloca`) and read by `Load` / `Store`.
    pub(super) place_ptr_id: HashMap<crate::ssa::types::PlaceId, spirv::Word>,
    /// Span of the instruction currently being lowered (set by `lower_inst`).
    /// Consumed via `blame_span()` so backend errors blame the source line of
    /// the originating expression.
    pub(super) current_span: Option<Span>,
    /// Function-level span fallback when an instruction has no span.
    pub(super) func_span: Span,
}

impl<'a, 'b> LowerCtx<'a, 'b> {
    pub(super) fn new(
        constructor: &'a mut Constructor,
        body: &'b FuncBody,
        is_entry_point: bool,
        func_span: Span,
    ) -> Self {
        LowerCtx {
            constructor,
            body,
            is_entry_point,
            value_map: HashMap::new(),
            block_map: HashMap::new(),
            block_indices: HashMap::new(),
            phi_inputs: Vec::new(),
            workgroup_view: HashMap::new(),
            place_ptr_id: HashMap::new(),
            current_span: None,
            func_span,
        }
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
        if self.constructor.param_ids.len() == self.body.params.len()
            && !self.constructor.param_ids.is_empty()
        {
            for (i, (value_id, _, _)) in self.body.params.iter().enumerate() {
                self.value_map.insert(*value_id, self.constructor.param_ids[i]);
            }
        } else {
            for (value_id, _, name) in &self.body.params {
                if let Some(&spirv_id) = self.constructor.env.get(name) {
                    self.value_map.insert(*value_id, spirv_id);
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
                let current = self.constructor.current_block.unwrap();
                self.block_map.insert(block_id, current);
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
                self.constructor.begin_block(spirv_block)?;
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
        let mut visited: HashSet<BlockId> = HashSet::new();
        let mut order = Vec::with_capacity(self.body.inner.blocks.len());

        fn visit(body: &FuncBody, bid: BlockId, visited: &mut HashSet<BlockId>, order: &mut Vec<BlockId>) {
            if visited.contains(&bid) {
                return;
            }
            let block = &body.inner.blocks[bid];
            if block.insts.is_empty() && matches!(block.term, Terminator::Unreachable) {
                return;
            }
            visited.insert(bid);
            order.push(bid);

            let merge_bid = body.control_headers.get(&bid).map(|ctrl| match ctrl {
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
                    if let Some(&func_id) = self.constructor.functions.get(func) {
                        self.constructor.builder.function_call(result_ty, None, func_id, arg_ids)?
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
                    if let Some(&func_id) = self.constructor.functions.get(name) {
                        // Global constant function - call it with no args to get the value.
                        // This handles `def verts: [3]vec4f32 = [...]` referenced as just `verts`.
                        self.constructor.builder.function_call(result_ty, None, func_id, [])?
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
                            || *id == known.texture_sample
                            || *id == known.image_store
                            || *id == known.image_load));
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

                crate::op::OpTag::StorageView(src) => {
                    let offset = operands[0];
                    let len = operands[1];
                    let offset_id = self.get_value_ref(offset)?;
                    let len_id = self.get_value_ref(len)?;

                    match src {
                        crate::op::PureViewSource::Storage(br) => {
                            let (set, binding) = (&br.set, &br.binding);
                            // The descriptor rides in the result value's type
                            // (`Region(set, binding)`); consumers recover the
                            // buffer var from there. Validate the binding is a
                            // declared storage buffer and build the {offset,len}
                            // struct.
                            if self
                                .constructor
                                .storage_buffers
                                .contains_key(&BindingRef::new(*set, *binding))
                            {
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
                    let var = self.constructor.declare_variable("_materialize", spirv_type)?;
                    self.constructor.builder.store(var, value_id, None, [])?;
                    var
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
                    let elem_ptr_type = *self
                        .constructor
                        .builder
                        .type_pointer(spirv::StorageClass::Function, builder::TypeId::new(result_ty));
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
        let current_block = self.constructor.current_block.unwrap();

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
                if let Some(control) = self.body.control_headers.get(&_block_id) {
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
        let mut phi_map: HashMap<(BlockId, usize), Vec<(spirv::Word, spirv::Word)>> = HashMap::new();

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

    pub(super) fn lower_binop(
        &mut self,
        op: &str,
        lhs: spirv::Word,
        rhs: spirv::Word,
        lhs_ty: &PolyType<TypeName>,
        rhs_ty: &PolyType<TypeName>,
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        use PolyType::*;
        use TypeName::*;

        let bool_type = self.constructor.bool_type;

        match (op, lhs_ty, rhs_ty) {
            // Scalar-left mixed-type ops (must precede scalar catch-alls)
            ("*", Constructed(Float(_), _), Constructed(Vec, _)) => {
                Ok(self.constructor.builder.vector_times_scalar(result_ty, None, rhs, lhs)?)
            }
            ("*", Constructed(Float(_), _), Constructed(Mat, _)) => {
                Ok(self.constructor.builder.matrix_times_scalar(result_ty, None, rhs, lhs)?)
            }
            ("+" | "-" | "/" | "%", Constructed(Float(_) | Int(_) | UInt(_), _), Constructed(Vec, _)) => {
                let splat = self.splat_scalar(lhs, rhs_ty, result_ty)?;
                self.lower_binop(op, splat, rhs, rhs_ty, rhs_ty, result_ty)
            }

            // Float operations
            ("+", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_add(result_ty, None, lhs, rhs)?)
            }
            ("-", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_sub(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_mul(result_ty, None, lhs, rhs)?)
            }
            ("/", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_div(result_ty, None, lhs, rhs)?)
            }
            ("%", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_rem(result_ty, None, lhs, rhs)?)
            }
            ("<", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_less_than(bool_type, None, lhs, rhs)?)
            }
            ("<=", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_less_than_equal(bool_type, None, lhs, rhs)?)
            }
            (">", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_greater_than(bool_type, None, lhs, rhs)?)
            }
            (">=", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_greater_than_equal(bool_type, None, lhs, rhs)?)
            }
            ("==", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_equal(bool_type, None, lhs, rhs)?)
            }
            ("!=", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_not_equal(bool_type, None, lhs, rhs)?)
            }
            ("**", Constructed(Float(_), _), Constructed(Int(_), _)) => {
                // Float base, integer exponent (the spec'd het case). Convert
                // the exponent to the base's float type, then call GLSL Pow.
                let glsl = self.constructor.glsl_ext_inst_id;
                let conv = self.constructor.builder.convert_s_to_f(result_ty, None, rhs)?;
                let operands = vec![Operand::IdRef(lhs), Operand::IdRef(conv)];
                Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 26, operands)?)
            }
            ("**", Constructed(Float(_), _), Constructed(UInt(_), _)) => {
                let glsl = self.constructor.glsl_ext_inst_id;
                let conv = self.constructor.builder.convert_u_to_f(result_ty, None, rhs)?;
                let operands = vec![Operand::IdRef(lhs), Operand::IdRef(conv)];
                Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 26, operands)?)
            }
            ("**", Constructed(Float(_), _), _) => {
                // Float base, float exponent: GLSL pow (opcode 26).
                let glsl = self.constructor.glsl_ext_inst_id;
                let operands = vec![Operand::IdRef(lhs), Operand::IdRef(rhs)];
                Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 26, operands)?)
            }
            ("**", Constructed(Int(_), _), _) => self.emit_int_pow_call(lhs, rhs, result_ty, true),
            ("**", Constructed(UInt(_), _), _) => self.emit_int_pow_call(lhs, rhs, result_ty, false),

            // Integer operations (signed)
            ("+", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.i_add(result_ty, None, lhs, rhs)?)
            }
            ("-", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.i_sub(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.i_mul(result_ty, None, lhs, rhs)?)
            }
            ("/", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_div(result_ty, None, lhs, rhs)?)
            }
            ("%", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_rem(result_ty, None, lhs, rhs)?)
            }
            ("<", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_less_than(bool_type, None, lhs, rhs)?)
            }
            ("<=", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_less_than_equal(bool_type, None, lhs, rhs)?)
            }
            (">", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_greater_than(bool_type, None, lhs, rhs)?)
            }
            (">=", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_greater_than_equal(bool_type, None, lhs, rhs)?)
            }
            ("==", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.i_equal(bool_type, None, lhs, rhs)?)
            }
            ("!=", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.i_not_equal(bool_type, None, lhs, rhs)?)
            }

            // Unsigned integer operations
            ("+", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.i_add(result_ty, None, lhs, rhs)?)
            }
            ("-", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.i_sub(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.i_mul(result_ty, None, lhs, rhs)?)
            }
            ("/", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_div(result_ty, None, lhs, rhs)?)
            }
            ("%", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_mod(result_ty, None, lhs, rhs)?)
            }
            ("<", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_less_than(bool_type, None, lhs, rhs)?)
            }
            ("<=", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_less_than_equal(bool_type, None, lhs, rhs)?)
            }
            (">", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_greater_than(bool_type, None, lhs, rhs)?)
            }
            (">=", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_greater_than_equal(bool_type, None, lhs, rhs)?)
            }
            ("==", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.i_equal(bool_type, None, lhs, rhs)?)
            }
            ("!=", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.i_not_equal(bool_type, None, lhs, rhs)?)
            }

            // Bitwise and shift operations (integer operands)
            ("&", Constructed(Int(_) | UInt(_), _), _) => {
                Ok(self.constructor.builder.bitwise_and(result_ty, None, lhs, rhs)?)
            }
            ("|", Constructed(Int(_) | UInt(_), _), _) => {
                Ok(self.constructor.builder.bitwise_or(result_ty, None, lhs, rhs)?)
            }
            ("^", Constructed(Int(_) | UInt(_), _), _) => {
                Ok(self.constructor.builder.bitwise_xor(result_ty, None, lhs, rhs)?)
            }
            ("<<", Constructed(Int(_) | UInt(_), _), _) => {
                Ok(self.constructor.builder.shift_left_logical(result_ty, None, lhs, rhs)?)
            }
            // Signed `>>` is arithmetic (sign-extending); unsigned is logical.
            (">>", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.shift_right_arithmetic(result_ty, None, lhs, rhs)?)
            }
            (">>", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.shift_right_logical(result_ty, None, lhs, rhs)?)
            }

            // Boolean operations
            ("&&", Constructed(Bool, _), _) => {
                Ok(self.constructor.builder.logical_and(bool_type, None, lhs, rhs)?)
            }
            ("||", Constructed(Bool, _), _) => {
                Ok(self.constructor.builder.logical_or(bool_type, None, lhs, rhs)?)
            }
            ("==", Constructed(Bool, _), _) => {
                Ok(self.constructor.builder.logical_equal(bool_type, None, lhs, rhs)?)
            }
            ("!=", Constructed(Bool, _), _) => {
                Ok(self.constructor.builder.logical_not_equal(bool_type, None, lhs, rhs)?)
            }

            // Mixed-type multiplication: mat*mat, mat*vec, vec*mat, vec*scalar, mat*scalar
            ("*", Constructed(Mat, _), Constructed(Mat, _)) => {
                Ok(self.constructor.builder.matrix_times_matrix(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Mat, _), Constructed(Vec, _)) => {
                Ok(self.constructor.builder.matrix_times_vector(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Vec, _), Constructed(Mat, _)) => {
                Ok(self.constructor.builder.vector_times_matrix(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Vec, _), Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.vector_times_scalar(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Mat, _), Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.matrix_times_scalar(result_ty, None, lhs, rhs)?)
            }

            // Vector operations: dispatch based on element type
            (_, Constructed(Vec, _), _) => {
                // If rhs is scalar (not vec/mat), splat it to match lhs vec
                let rhs = if matches!(rhs_ty, Constructed(Float(_) | Int(_) | UInt(_), _)) {
                    self.splat_scalar(rhs, lhs_ty, result_ty)?
                } else {
                    rhs
                };

                let elem_ty = lhs_ty
                    .elem_type()
                    .ok_or_else(|| crate::err_spirv!("Vec type missing element type: {:?}", lhs_ty))?;
                match (op, elem_ty) {
                    ("+", Constructed(Float(_), _)) => {
                        Ok(self.constructor.builder.f_add(result_ty, None, lhs, rhs)?)
                    }
                    ("-", Constructed(Float(_), _)) => {
                        Ok(self.constructor.builder.f_sub(result_ty, None, lhs, rhs)?)
                    }
                    ("*", Constructed(Float(_), _)) => {
                        Ok(self.constructor.builder.f_mul(result_ty, None, lhs, rhs)?)
                    }
                    ("/", Constructed(Float(_), _)) => {
                        Ok(self.constructor.builder.f_div(result_ty, None, lhs, rhs)?)
                    }
                    ("%", Constructed(Float(_), _)) => {
                        Ok(self.constructor.builder.f_rem(result_ty, None, lhs, rhs)?)
                    }
                    ("+", Constructed(Int(_), _)) => {
                        Ok(self.constructor.builder.i_add(result_ty, None, lhs, rhs)?)
                    }
                    ("-", Constructed(Int(_), _)) => {
                        Ok(self.constructor.builder.i_sub(result_ty, None, lhs, rhs)?)
                    }
                    ("*", Constructed(Int(_), _)) => {
                        Ok(self.constructor.builder.i_mul(result_ty, None, lhs, rhs)?)
                    }
                    ("/", Constructed(Int(_), _)) => {
                        Ok(self.constructor.builder.s_div(result_ty, None, lhs, rhs)?)
                    }
                    ("%", Constructed(Int(_), _)) => {
                        Ok(self.constructor.builder.s_rem(result_ty, None, lhs, rhs)?)
                    }
                    ("+", Constructed(UInt(_), _)) => {
                        Ok(self.constructor.builder.i_add(result_ty, None, lhs, rhs)?)
                    }
                    ("-", Constructed(UInt(_), _)) => {
                        Ok(self.constructor.builder.i_sub(result_ty, None, lhs, rhs)?)
                    }
                    ("*", Constructed(UInt(_), _)) => {
                        Ok(self.constructor.builder.i_mul(result_ty, None, lhs, rhs)?)
                    }
                    ("/", Constructed(UInt(_), _)) => {
                        Ok(self.constructor.builder.u_div(result_ty, None, lhs, rhs)?)
                    }
                    ("%", Constructed(UInt(_), _)) => {
                        Ok(self.constructor.builder.u_mod(result_ty, None, lhs, rhs)?)
                    }

                    // Vector equality: componentwise compare → bvec,
                    // then `OpAll` / `OpAny` collapse to scalar bool.
                    // Matches GLSL `all(a == b)` / `any(a != b)`.
                    ("==" | "!=", _) => {
                        let vec_size = lhs_ty
                            .vec_size()
                            .ok_or_else(|| crate::err_spirv!("Vec type missing size: {:?}", lhs_ty))?
                            as u32;
                        let bvec_ty =
                            self.constructor.get_or_create_vec_type(self.constructor.bool_type, vec_size);
                        let cmp = match (op, elem_ty) {
                            ("==", Constructed(Float(_), _)) => {
                                self.constructor.builder.f_ord_equal(bvec_ty, None, lhs, rhs)?
                            }
                            ("!=", Constructed(Float(_), _)) => {
                                self.constructor.builder.f_ord_not_equal(bvec_ty, None, lhs, rhs)?
                            }
                            ("==", Constructed(Int(_) | UInt(_), _)) => {
                                self.constructor.builder.i_equal(bvec_ty, None, lhs, rhs)?
                            }
                            ("!=", Constructed(Int(_) | UInt(_), _)) => {
                                self.constructor.builder.i_not_equal(bvec_ty, None, lhs, rhs)?
                            }
                            ("==", Constructed(Bool, _)) => {
                                self.constructor.builder.logical_equal(bvec_ty, None, lhs, rhs)?
                            }
                            ("!=", Constructed(Bool, _)) => {
                                self.constructor.builder.logical_not_equal(bvec_ty, None, lhs, rhs)?
                            }
                            _ => bail_spirv!("Unsupported vector {} on element {:?}", op, elem_ty),
                        };
                        if op == "==" {
                            Ok(self.constructor.builder.all(result_ty, None, cmp)?)
                        } else {
                            Ok(self.constructor.builder.any(result_ty, None, cmp)?)
                        }
                    }

                    _ => bail_spirv!(
                        "Unsupported vector binary operation: {} on element {:?}",
                        op,
                        elem_ty
                    ),
                }
            }

            _ => bail_spirv!("Unsupported binary operation: {} on {:?}", op, lhs_ty),
        }
    }

    /// Lower an integer `**` to an `OpFunctionCall` against the
    /// compiler-generated helper emitted by `spirv::pow`. Bridges
    /// `lower_binop`'s inline dispatch to the shared helper used by
    /// `PrimOp::IntPow` in `lower_primop`.
    pub(super) fn emit_int_pow_call(
        &mut self,
        lhs: spirv::Word,
        rhs: spirv::Word,
        result_ty: spirv::Word,
        signed: bool,
    ) -> Result<spirv::Word> {
        let func_id = self
            .constructor
            .int_pow_functions
            .get(&signed)
            .copied()
            .ok_or_else(|| err_spirv!("int_pow helper not emitted (signed={})", signed))?;
        Ok(self.constructor.builder.function_call(result_ty, None, func_id, vec![lhs, rhs])?)
    }

    /// Splat a scalar SPIR-V value into a vector matching `vec_ty`.
    pub(super) fn splat_scalar(
        &mut self,
        scalar: spirv::Word,
        vec_ty: &PolyType<TypeName>,
        vec_spirv_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        let n = vec_ty.vec_size().ok_or_else(|| {
            crate::err_spirv!("Cannot splat: vec type has no concrete size: {:?}", vec_ty)
        })?;
        let components = vec![scalar; n];
        Ok(self.constructor.builder.composite_construct(vec_spirv_ty, None, components)?)
    }

    pub(super) fn lower_unaryop(
        &mut self,
        op: &str,
        operand: spirv::Word,
        operand_ty: &PolyType<TypeName>,
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        use PolyType::*;
        use TypeName::*;

        match (op, operand_ty) {
            ("-", Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.f_negate(result_ty, None, operand)?)
            }
            ("-", Constructed(Int(_), _)) => {
                Ok(self.constructor.builder.s_negate(result_ty, None, operand)?)
            }
            ("!", Constructed(Bool, _)) => {
                Ok(self.constructor.builder.logical_not(result_ty, None, operand)?)
            }
            // Vector unary operations
            ("-", Constructed(Vec, _)) => {
                let elem_ty = operand_ty
                    .elem_type()
                    .ok_or_else(|| crate::err_spirv!("Vec type missing element type: {:?}", operand_ty))?;
                match elem_ty {
                    Constructed(Float(_), _) => {
                        Ok(self.constructor.builder.f_negate(result_ty, None, operand)?)
                    }
                    Constructed(Int(_), _) => {
                        Ok(self.constructor.builder.s_negate(result_ty, None, operand)?)
                    }
                    _ => bail_spirv!(
                        "Unsupported vector unary operation: {} on element {:?}",
                        op,
                        elem_ty
                    ),
                }
            }
            _ => bail_spirv!("Unsupported unary operation: {} on {:?}", op, operand_ty),
        }
    }

    pub(super) fn lower_primop(
        &mut self,
        prim_op: &PrimOp,
        arg_ids: &[spirv::Word],
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        let glsl = self.constructor.glsl_ext_inst_id;
        let operands: Vec<Operand> = arg_ids.iter().map(|&id| Operand::IdRef(id)).collect();

        match prim_op {
            PrimOp::GlslExt(ext_op) => {
                Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, *ext_op, operands)?)
            }
            PrimOp::IntPow { signed } => {
                if arg_ids.len() != 2 {
                    bail_spirv!("int_pow requires 2 args");
                }
                // Function id was cached by `spirv::pow::emit_int_pow_helpers`
                // during module setup; missing means a backend-init bug.
                let func_id = self
                    .constructor
                    .int_pow_functions
                    .get(signed)
                    .copied()
                    .ok_or_else(|| err_spirv!("int_pow helper not emitted (signed={})", signed))?;
                Ok(self.constructor.builder.function_call(result_ty, None, func_id, arg_ids.to_vec())?)
            }
            PrimOp::Dot => {
                if arg_ids.len() != 2 {
                    bail_spirv!("dot requires 2 args");
                }
                Ok(self.constructor.builder.dot(result_ty, None, arg_ids[0], arg_ids[1])?)
            }
            PrimOp::MatrixTimesMatrix => {
                if arg_ids.len() != 2 {
                    bail_spirv!("matrix × matrix requires 2 args");
                }
                Ok(
                    self.constructor
                        .builder
                        .matrix_times_matrix(result_ty, None, arg_ids[0], arg_ids[1])?,
                )
            }
            PrimOp::MatrixTimesVector => {
                if arg_ids.len() != 2 {
                    bail_spirv!("matrix × vector requires 2 args");
                }
                Ok(
                    self.constructor
                        .builder
                        .matrix_times_vector(result_ty, None, arg_ids[0], arg_ids[1])?,
                )
            }
            PrimOp::VectorTimesMatrix => {
                if arg_ids.len() != 2 {
                    bail_spirv!("vector × matrix requires 2 args");
                }
                Ok(
                    self.constructor
                        .builder
                        .vector_times_matrix(result_ty, None, arg_ids[0], arg_ids[1])?,
                )
            }
            PrimOp::FPToSI => {
                if arg_ids.len() != 1 {
                    bail_spirv!("FPToSI requires 1 arg");
                }
                Ok(self.constructor.builder.convert_f_to_s(result_ty, None, arg_ids[0])?)
            }
            PrimOp::FPToUI => {
                if arg_ids.len() != 1 {
                    bail_spirv!("FPToUI requires 1 arg");
                }
                Ok(self.constructor.builder.convert_f_to_u(result_ty, None, arg_ids[0])?)
            }
            PrimOp::SIToFP => {
                if arg_ids.len() != 1 {
                    bail_spirv!("SIToFP requires 1 arg");
                }
                Ok(self.constructor.builder.convert_s_to_f(result_ty, None, arg_ids[0])?)
            }
            PrimOp::UIToFP => {
                if arg_ids.len() != 1 {
                    bail_spirv!("UIToFP requires 1 arg");
                }
                Ok(self.constructor.builder.convert_u_to_f(result_ty, None, arg_ids[0])?)
            }
            PrimOp::Bitcast => {
                if arg_ids.len() != 1 {
                    bail_spirv!("Bitcast requires 1 arg");
                }
                Ok(self.constructor.builder.bitcast(result_ty, None, arg_ids[0])?)
            }
            PrimOp::IsNan => {
                if arg_ids.len() != 1 {
                    bail_spirv!("isnan requires 1 arg");
                }
                Ok(self.constructor.builder.is_nan(result_ty, None, arg_ids[0])?)
            }
            PrimOp::IsInf => {
                if arg_ids.len() != 1 {
                    bail_spirv!("isinf requires 1 arg");
                }
                Ok(self.constructor.builder.is_inf(result_ty, None, arg_ids[0])?)
            }
            // Additional arithmetic ops
            PrimOp::FAdd | PrimOp::FSub | PrimOp::FMul | PrimOp::FDiv | PrimOp::FRem | PrimOp::FMod => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Float binary op requires 2 args");
                }
                match prim_op {
                    PrimOp::FAdd => {
                        Ok(self.constructor.builder.f_add(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::FSub => {
                        Ok(self.constructor.builder.f_sub(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::FMul => {
                        Ok(self.constructor.builder.f_mul(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::FDiv => {
                        Ok(self.constructor.builder.f_div(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::FRem => {
                        Ok(self.constructor.builder.f_rem(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::FMod => {
                        Ok(self.constructor.builder.f_mod(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    _ => unreachable!(),
                }
            }
            PrimOp::IAdd
            | PrimOp::ISub
            | PrimOp::IMul
            | PrimOp::SDiv
            | PrimOp::UDiv
            | PrimOp::SRem
            | PrimOp::SMod
            | PrimOp::UMod => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Integer binary op requires 2 args");
                }
                match prim_op {
                    PrimOp::IAdd => {
                        Ok(self.constructor.builder.i_add(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::ISub => {
                        Ok(self.constructor.builder.i_sub(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::IMul => {
                        Ok(self.constructor.builder.i_mul(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::SDiv => {
                        Ok(self.constructor.builder.s_div(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::UDiv => {
                        Ok(self.constructor.builder.u_div(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::SRem => {
                        Ok(self.constructor.builder.s_rem(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::SMod => {
                        Ok(self.constructor.builder.s_mod(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::UMod => {
                        Ok(self.constructor.builder.u_mod(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    _ => unreachable!(),
                }
            }
            // Comparison ops
            PrimOp::FOrdEqual
            | PrimOp::FOrdNotEqual
            | PrimOp::FOrdLessThan
            | PrimOp::FOrdGreaterThan
            | PrimOp::FOrdLessThanEqual
            | PrimOp::FOrdGreaterThanEqual => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Float comparison requires 2 args");
                }
                match prim_op {
                    PrimOp::FOrdEqual => Ok(self
                        .constructor
                        .builder
                        .f_ord_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::FOrdNotEqual => Ok(self
                        .constructor
                        .builder
                        .f_ord_not_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::FOrdLessThan => Ok(self
                        .constructor
                        .builder
                        .f_ord_less_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::FOrdGreaterThan => Ok(self
                        .constructor
                        .builder
                        .f_ord_greater_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::FOrdLessThanEqual => Ok(self
                        .constructor
                        .builder
                        .f_ord_less_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::FOrdGreaterThanEqual => Ok(self
                        .constructor
                        .builder
                        .f_ord_greater_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    _ => unreachable!(),
                }
            }
            PrimOp::IEqual
            | PrimOp::INotEqual
            | PrimOp::SLessThan
            | PrimOp::ULessThan
            | PrimOp::SGreaterThan
            | PrimOp::UGreaterThan
            | PrimOp::SLessThanEqual
            | PrimOp::ULessThanEqual
            | PrimOp::SGreaterThanEqual
            | PrimOp::UGreaterThanEqual => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Integer comparison requires 2 args");
                }
                match prim_op {
                    PrimOp::IEqual => {
                        Ok(self.constructor.builder.i_equal(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::INotEqual => Ok(self
                        .constructor
                        .builder
                        .i_not_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::SLessThan => Ok(self
                        .constructor
                        .builder
                        .s_less_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::ULessThan => Ok(self
                        .constructor
                        .builder
                        .u_less_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::SGreaterThan => Ok(self
                        .constructor
                        .builder
                        .s_greater_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::UGreaterThan => Ok(self
                        .constructor
                        .builder
                        .u_greater_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::SLessThanEqual => Ok(self
                        .constructor
                        .builder
                        .s_less_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::ULessThanEqual => Ok(self
                        .constructor
                        .builder
                        .u_less_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::SGreaterThanEqual => Ok(self
                        .constructor
                        .builder
                        .s_greater_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::UGreaterThanEqual => Ok(self
                        .constructor
                        .builder
                        .u_greater_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    _ => unreachable!(),
                }
            }
            // Bitwise ops
            PrimOp::BitwiseAnd | PrimOp::BitwiseOr | PrimOp::BitwiseXor => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Bitwise binary op requires 2 args");
                }
                match prim_op {
                    PrimOp::BitwiseAnd => Ok(self
                        .constructor
                        .builder
                        .bitwise_and(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::BitwiseOr => {
                        Ok(self.constructor.builder.bitwise_or(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::BitwiseXor => Ok(self
                        .constructor
                        .builder
                        .bitwise_xor(result_ty, None, arg_ids[0], arg_ids[1])?),
                    _ => unreachable!(),
                }
            }
            PrimOp::Not => {
                if arg_ids.len() != 1 {
                    bail_spirv!("Not requires 1 arg");
                }
                Ok(self.constructor.builder.not(result_ty, None, arg_ids[0])?)
            }
            PrimOp::ShiftLeftLogical | PrimOp::ShiftRightArithmetic | PrimOp::ShiftRightLogical => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Shift op requires 2 args");
                }
                match prim_op {
                    PrimOp::ShiftLeftLogical => Ok(self
                        .constructor
                        .builder
                        .shift_left_logical(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::ShiftRightArithmetic => Ok(self
                        .constructor
                        .builder
                        .shift_right_arithmetic(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::ShiftRightLogical => Ok(self
                        .constructor
                        .builder
                        .shift_right_logical(result_ty, None, arg_ids[0], arg_ids[1])?),
                    _ => unreachable!(),
                }
            }
            // Additional type conversions
            PrimOp::FPConvert => {
                if arg_ids.len() != 1 {
                    bail_spirv!("FPConvert requires 1 arg");
                }
                Ok(self.constructor.builder.f_convert(result_ty, None, arg_ids[0])?)
            }
            PrimOp::SConvert => {
                if arg_ids.len() != 1 {
                    bail_spirv!("SConvert requires 1 arg");
                }
                Ok(self.constructor.builder.s_convert(result_ty, None, arg_ids[0])?)
            }
            PrimOp::UConvert => {
                if arg_ids.len() != 1 {
                    bail_spirv!("UConvert requires 1 arg");
                }
                Ok(self.constructor.builder.u_convert(result_ty, None, arg_ids[0])?)
            }
            // Additional matrix/vector ops
            PrimOp::OuterProduct => {
                if arg_ids.len() != 2 {
                    bail_spirv!("OuterProduct requires 2 args");
                }
                Ok(self.constructor.builder.outer_product(result_ty, None, arg_ids[0], arg_ids[1])?)
            }
            PrimOp::VectorTimesScalar => {
                if arg_ids.len() != 2 {
                    bail_spirv!("VectorTimesScalar requires 2 args");
                }
                Ok(
                    self.constructor
                        .builder
                        .vector_times_scalar(result_ty, None, arg_ids[0], arg_ids[1])?,
                )
            }
            PrimOp::MatrixTimesScalar => {
                if arg_ids.len() != 2 {
                    bail_spirv!("MatrixTimesScalar requires 2 args");
                }
                Ok(
                    self.constructor
                        .builder
                        .matrix_times_scalar(result_ty, None, arg_ids[0], arg_ids[1])?,
                )
            }
            PrimOp::DPdx => {
                if arg_ids.len() != 1 {
                    bail_spirv!("DPdx requires 1 arg");
                }
                Ok(self.constructor.builder.d_pdx(result_ty, None, arg_ids[0])?)
            }
            PrimOp::DPdy => {
                if arg_ids.len() != 1 {
                    bail_spirv!("DPdy requires 1 arg");
                }
                Ok(self.constructor.builder.d_pdy(result_ty, None, arg_ids[0])?)
            }
            PrimOp::Fwidth => {
                if arg_ids.len() != 1 {
                    bail_spirv!("Fwidth requires 1 arg");
                }
                Ok(self.constructor.builder.fwidth(result_ty, None, arg_ids[0])?)
            }
        }
    }
}
