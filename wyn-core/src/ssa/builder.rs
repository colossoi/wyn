//! SSA function builder.
//!
//! Thin wrapper around wyn-ssa's generic `FuncBuilder` that pairs it with
//! the `FuncBody` side-map state (`control_headers`, `params`,
//! `return_ty`, `dps_output`, `places`) and exposes exactly the surface
//! the wyn-core elaborator needs.

use crate::ast::{Span, TypeName};
use polytype::Type;

use super::types::{
    BlockId, ControlHeader, FuncBody, InstId, InstKind, PlaceId, PlaceInfo, Terminator, ValueId,
};
use slotmap::SlotMap;

/// Error during function building.
pub type BuilderError = crate::ssa::framework::BuilderError;

/// Builder for constructing SSA functions.
pub struct FuncBuilder {
    inner: crate::ssa::framework::FuncBuilder<InstKind, Type<TypeName>>,
    control_headers: std::collections::HashMap<BlockId, ControlHeader>,
    params: Vec<(ValueId, Type<TypeName>, String)>,
    return_ty: Type<TypeName>,
    dps_output: Option<ValueId>,
    places: SlotMap<PlaceId, PlaceInfo>,
}

impl FuncBuilder {
    /// Create a new function builder with the given parameters and return type.
    pub fn new(params: Vec<(Type<TypeName>, String)>, return_ty: Type<TypeName>) -> Self {
        let mut inner = crate::ssa::framework::FuncBuilder::new();
        let _entry = inner.entry();

        let mut func_params = Vec::new();
        for (i, (ty, name)) in params.into_iter().enumerate() {
            let value = inner.func_mut().add_function_param(i, ty.clone());
            func_params.push((value, ty, name));
        }

        FuncBuilder {
            inner,
            control_headers: std::collections::HashMap::new(),
            params: func_params,
            return_ty,
            dps_output: None,
            places: SlotMap::with_key(),
        }
    }

    /// Get the value for a function parameter by index.
    pub fn get_param(&self, index: usize) -> ValueId {
        self.params[index].0
    }

    /// Get the number of function parameters.
    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    /// Set the DPS output parameter.
    pub fn set_dps_output(&mut self, value: ValueId) {
        self.dps_output = Some(value);
    }

    /// Get the entry block ID.
    pub fn entry(&self) -> BlockId {
        self.inner.entry()
    }

    /// Create a new basic block.
    pub fn create_block(&mut self) -> BlockId {
        self.inner.create_block()
    }

    /// Create a new basic block with parameters.
    pub fn create_block_with_params(
        &mut self,
        param_types: Vec<Type<TypeName>>,
    ) -> (BlockId, Vec<ValueId>) {
        self.inner.create_block_with_params(param_types)
    }

    /// Add a parameter to an existing block.
    pub fn add_block_param(&mut self, block: BlockId, ty: Type<TypeName>) -> ValueId {
        self.inner.add_block_param(block, ty)
    }

    /// Create a new block with named parameters (names are discarded).
    pub fn create_block_with_named_params(
        &mut self,
        params: Vec<(Type<TypeName>, String)>,
    ) -> (BlockId, Vec<ValueId>) {
        let types: Vec<Type<TypeName>> = params.into_iter().map(|(ty, _)| ty).collect();
        self.inner.create_block_with_params(types)
    }

    /// Switch to building in the specified block.
    pub fn switch_to_block(&mut self, block: BlockId) -> Result<(), BuilderError> {
        self.inner.switch_to_block(block)
    }

    /// Switch to a block without checking if the previous block is terminated.
    pub fn switch_to_block_unchecked(&mut self, block: BlockId) {
        self.inner.switch_to_block_unchecked(block)
    }

    /// Get the current block, if any.
    pub fn current_block(&self) -> Option<BlockId> {
        self.inner.current_block()
    }

    /// Check if the current block is terminated.
    pub fn is_current_terminated(&self) -> bool {
        self.current_block()
            .map(|b| !matches!(self.inner.func().blocks[b].term, Terminator::Unreachable))
            .unwrap_or(false)
    }

    /// Push an instruction that produces a value.
    pub fn push_inst(&mut self, kind: InstKind, ty: Type<TypeName>) -> Result<ValueId, BuilderError> {
        self.inner.push_inst(kind, ty)
    }

    /// Push an instruction with a source span attached for error blame.
    pub fn push_inst_with_span(
        &mut self,
        kind: InstKind,
        ty: Type<TypeName>,
        span: Option<Span>,
    ) -> Result<ValueId, BuilderError> {
        self.inner.push_inst_with_span(kind, ty, span)
    }

    /// Push an instruction that produces no value (e.g., Store).
    pub fn push_void_inst(&mut self, kind: InstKind) -> Result<InstId, BuilderError> {
        self.inner.push_void_inst(kind)
    }

    /// Push a void instruction with a source span attached.
    pub fn push_void_inst_with_span(
        &mut self,
        kind: InstKind,
        span: Option<Span>,
    ) -> Result<InstId, BuilderError> {
        self.inner.push_void_inst_with_span(kind, span)
    }

    /// Set the terminator for the current block.
    pub fn terminate(&mut self, term: Terminator) -> Result<(), BuilderError> {
        self.inner.terminate(term)
    }

    /// Finish building and return the function body.
    pub fn finish(self) -> Result<FuncBody, BuilderError> {
        let func = self.inner.finish()?;
        Ok(FuncBody {
            inner: func,
            control_headers: self.control_headers,
            params: self.params,
            return_ty: self.return_ty,
            dps_output: self.dps_output,
            places: self.places,
        })
    }

    /// Finish without checking termination (for testing).
    pub fn finish_unchecked(self) -> FuncBody {
        FuncBody {
            inner: self.inner.finish_unchecked(),
            control_headers: self.control_headers,
            params: self.params,
            return_ty: self.return_ty,
            dps_output: self.dps_output,
            places: self.places,
        }
    }

    /// Access the underlying function (read-only).
    pub fn func(&self) -> &crate::ssa::framework::Function<InstKind, Type<TypeName>> {
        self.inner.func()
    }

    /// Access the underlying function (mutable).
    pub fn func_mut(&mut self) -> &mut crate::ssa::framework::Function<InstKind, Type<TypeName>> {
        self.inner.func_mut()
    }

    /// Allocate a fresh `PlaceId` with the given element type. The
    /// caller is responsible for emitting a matching place-producing
    /// instruction (`OutputSlot` / `ViewIndex` / `Alloca`) carrying the
    /// returned `PlaceId` in its `result` field.
    pub fn new_place(&mut self, elem_ty: Type<TypeName>) -> PlaceId {
        self.places.insert(PlaceInfo { elem_ty })
    }

    pub fn set_control_header(&mut self, block: BlockId, control: ControlHeader) {
        self.control_headers.insert(block, control);
    }
}
