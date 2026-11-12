//! Direct emission of the ordered memory IR into typed SSA.
use super::entry::GeneratedBuffers;
use super::{builder_error, error, Compiler, OptimizeError, Typed};
use crate::egglog::bindings::Bindings;
use crate::egglog::query::Query;
use crate::host::{Expr, ScalarExpr};
use crate::op::OpTag;
use crate::ssa::builder::{BuilderError, FuncBuilder};
use crate::ssa::types::{BlockId, FuncBody, InstKind, PlaceId, Terminator, ValueRef};
use crate::types::{self, Type};
use crate::{BindingRef, FunctionId, LookupMap, LookupSet};
use egglog_engine::Value;

mod atomics;
mod bucket;
mod control;
mod memory;
mod subgroups;
mod values;

pub(super) struct Body<'a, 'p, 'source> {
    pub compiler: &'a mut Compiler<'p, 'source>,
    pub builder: FuncBuilder,
    pub function: Value,
    pub values: Bindings<Value, Typed>,
    local_arrays: LookupMap<ValueRef, PlaceId>,
    next_workgroup_array: u32,
    pub subgroup: Option<subgroups::Subgroup>,
    pub storage_accesses: LookupMap<BindingRef, crate::ResourceAccess>,
    pub active_dispatch: Option<Value>,
    pub host_values: Bindings<Value, ScalarExpr>,
    pub host_lengths: LookupMap<ValueRef, Expr>,
    pub host_shapes: LookupMap<ValueRef, Vec<Expr>>,
    pub collection_requests:
        LookupMap<Value, (Type, Typed, Option<Expr>, Option<Vec<(Typed, Expr)>>, BlockId)>,
    pub collection_values: LookupMap<Value, Typed>,
    pub reused_buffers: LookupSet<BindingRef>,
    pub transported_values: LookupMap<Value, ValueRef>,
    pub guard: Option<ScalarExpr>,
    pub generated: Option<GeneratedBuffers>,
}

impl<'a, 'p, 'source> Body<'a, 'p, 'source> {
    pub fn new(
        compiler: &'a mut Compiler<'p, 'source>,
        function: Value,
        parameters: Vec<(Value, Type)>,
        result: Type,
    ) -> Self {
        let builder = FuncBuilder::new(parameters.iter().map(|(_, ty)| ty.clone()).collect(), result);
        let mut values = Bindings::default();
        for (i, (name, ty)) in parameters.into_iter().enumerate() {
            values.insert(
                name,
                Typed {
                    value: builder.get_param(i).into(),
                    ty,
                },
            );
        }
        Self {
            compiler,
            builder,
            function,
            values,
            local_arrays: LookupMap::default(),
            next_workgroup_array: 0,
            subgroup: None,
            storage_accesses: LookupMap::default(),
            active_dispatch: None,
            host_values: Bindings::default(),
            host_lengths: LookupMap::default(),
            host_shapes: LookupMap::default(),
            collection_requests: LookupMap::default(),
            collection_values: LookupMap::default(),
            reused_buffers: LookupSet::default(),
            transported_values: LookupMap::default(),
            guard: None,
            generated: None,
        }
    }

    pub fn query(&self) -> Query<'_> {
        Query(&self.compiler.program.graph)
    }

    pub fn finish(mut self, result: Typed) -> Result<FuncBody, OptimizeError> {
        self.terminate(Terminator::Return(
            (result.ty != types::unit()).then_some(result.value),
        ))
        .map_err(builder_error)?;
        let mut body = self.builder.finish().map_err(builder_error)?;
        body.return_ty = result.ty;
        Ok(body)
    }

    pub fn reset_shader(&mut self) {
        self.builder = FuncBuilder::new(vec![], types::unit());
        self.values = Bindings::default();
        self.local_arrays.clear();
        self.host_lengths.clear();
        self.host_shapes.clear();
        self.collection_requests.clear();
        self.collection_values.clear();
        self.reused_buffers.clear();
        self.transported_values.clear();
        self.storage_accesses.clear();
        self.next_workgroup_array = 0;
        self.active_dispatch = None;
        self.subgroup = None;
    }

    pub fn finish_shader(&mut self) -> Result<FuncBody, OptimizeError> {
        self.terminate(Terminator::Return(None)).map_err(builder_error)?;
        let builder = std::mem::replace(&mut self.builder, FuncBuilder::new(vec![], types::unit()));
        builder.finish().map_err(builder_error)
    }

    pub fn storage_access(&mut self, ty: &Type, access: crate::ResourceAccess) {
        if let Some(binding) = crate::types::array_view_buffer(ty) {
            self.storage_accesses
                .entry(binding)
                .and_modify(|previous| *previous = previous.merge(access))
                .or_insert(access);
        }
    }

    pub fn current(&self) -> Result<BlockId, OptimizeError> {
        let Some(block) = self.builder.current_block() else {
            return Err(error("no current SSA block"));
        };
        Ok(block)
    }

    fn terminate(&mut self, term: Terminator) -> Result<(), BuilderError> {
        self.builder.terminate(term)
    }

    pub fn op(
        &mut self,
        tag: OpTag<BindingRef, FunctionId>,
        arguments: Vec<Typed>,
        ty: Type,
    ) -> Result<Typed, OptimizeError> {
        let access = if let OpTag::Intrinsic { id, .. } = &tag {
            if crate::builtins::by_id(*id).raw.purity == crate::builtins::Purity::Pure {
                crate::ResourceAccess::Read
            } else {
                crate::ResourceAccess::ReadWrite
            }
        } else {
            crate::ResourceAccess::ReadWrite
        };
        for argument in &arguments {
            if let Some(binding) = crate::types::storage_image_buffer(&argument.ty) {
                self.storage_accesses
                    .entry(binding)
                    .and_modify(|previous| *previous = previous.merge(access))
                    .or_insert(access);
            }
        }
        let operands = arguments.into_iter().map(|argument| argument.value).collect();
        let value =
            self.builder.push_inst(InstKind::Op { tag, operands }, ty.clone()).map_err(builder_error)?;
        Ok(Typed {
            value: value.into(),
            ty,
        })
    }
}
