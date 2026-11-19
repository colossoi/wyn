//! Direct emission of the ordered memory IR into typed SSA.
use super::entry::{GeneratedBuffers, Location};
use super::{builder_error, error, Compiler, OptimizeError, Typed};
use crate::ast::Span;
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

pub(super) enum CollectionShape {
    Local(Typed),
    Device(Vec<(Typed, Expr)>),
}

impl CollectionShape {
    pub fn outer(count: Typed, length: Option<Expr>) -> Self {
        match length {
            Some(length) => Self::Device(vec![(count, length)]),
            None => Self::Local(count),
        }
    }
}

pub(super) enum Collection {
    Pending {
        ty: Type,
        shape: CollectionShape,
        block: BlockId,
    },
    Allocated(Typed),
}

impl Collection {
    fn value(&self) -> Option<&Typed> {
        match self {
            Self::Allocated(value) => Some(value),
            Self::Pending { .. } => None,
        }
    }
}

pub(super) struct Body<'a, 'p, 'source> {
    pub compiler: &'a mut Compiler<'p, 'source>,
    pub builder: FuncBuilder,
    pub function: Value,
    span: Option<Span>,
    pub values: Bindings<Value, Typed>,
    pub locations: LookupMap<ValueRef, Location>,
    local_arrays: LookupMap<ValueRef, PlaceId>,
    next_workgroup_array: u32,
    pub subgroup: Option<subgroups::Subgroup>,
    pub storage_accesses: LookupMap<BindingRef, crate::ResourceAccess>,
    pub restored_reads: LookupMap<PlaceId, BindingRef>,
    pub active_dispatch: Option<Value>,
    pub host_values: Bindings<Value, ScalarExpr>,
    pub collections: LookupMap<Value, Collection>,
    pub reused_buffers: LookupSet<BindingRef>,
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
            span: None,
            values,
            locations: LookupMap::default(),
            local_arrays: LookupMap::default(),
            next_workgroup_array: 0,
            subgroup: None,
            storage_accesses: LookupMap::default(),
            restored_reads: LookupMap::default(),
            active_dispatch: None,
            host_values: Bindings::default(),
            collections: LookupMap::default(),
            reused_buffers: LookupSet::default(),
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
        self.locations.clear();
        self.collections.clear();
        self.reused_buffers.clear();
        self.storage_accesses.clear();
        self.restored_reads.clear();
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
        let value = self
            .builder
            .push_inst_with_span(InstKind::Op { tag, operands }, ty.clone(), self.span)
            .map_err(builder_error)?;
        Ok(Typed {
            value: value.into(),
            ty,
        })
    }
}
