//! Small prepared SSA inputs for backend-local contracts.
use super::{builder::FuncBuilder, types::*};
use crate::{
    ast::Span,
    flow::ExecutionModel,
    interface::StorageBindingDecl,
    interface::StorageRole,
    op::{OpTag, PureViewSource},
    types, BindingRef, CodegenTarget, EntryId, LookupMap, ResourceAccess,
};

pub(crate) fn storage_write() -> EntryPoint {
    storage_write_at(BindingRef::new(0, 0))
}

pub(crate) fn storage_write_at(binding: BindingRef) -> EntryPoint {
    storage_write_value(binding, types::i32(), |_| ValueRef::Const(ConstantValue::I32(7)))
}

pub(crate) fn storage_write_value(
    binding: BindingRef,
    ty: types::Type,
    value: impl FnOnce(&mut FuncBuilder) -> ValueRef,
) -> EntryPoint {
    let mut builder = FuncBuilder::new(vec![], types::unit());
    let value = value(&mut builder);
    let view_ty = types::view_array_with_size(
        &ty,
        types::Type::Constructed(types::TypeName::Size(4), vec![]),
        types::buffer_tag(binding),
    );
    let view = builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::StorageView(PureViewSource::Storage(binding)),
                operands: vec![
                    ValueRef::Const(ConstantValue::U32(0)),
                    ValueRef::Const(ConstantValue::U32(4)),
                ],
            },
            view_ty,
        )
        .unwrap();
    let place = builder.new_place(ty.clone());
    builder
        .push_void_inst(InstKind::ViewIndex {
            view: view.into(),
            index: ValueRef::Const(ConstantValue::U32(1)),
            result: place,
        })
        .unwrap();
    builder.push_void_inst(InstKind::Store { place, value }).unwrap();
    builder.terminate(Terminator::Return(None)).unwrap();
    EntryPoint {
        id: EntryId::from_index(0),
        name: "store".into(),
        body: builder.finish().unwrap(),
        execution_model: ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        inputs: vec![],
        parameter_inputs: vec![],
        outputs: vec![],
        storage_bindings: vec![StorageBindingDecl {
            binding,
            role: StorageRole::Output,
            elem_ty: ty,
            host: (),
        }],
        stage_descriptor_storage_accesses: LookupMap::from_iter([(binding, ResourceAccess::Write)]),
        pipeline_storage_accesses: LookupMap::from_iter([(binding, ResourceAccess::Write)]),
        span: Span::generated(),
    }
}

pub(crate) fn spirv(entries: Vec<EntryPoint>) -> Vec<u32> {
    let program = super::stage::SpirvReady::from_parts(
        vec![],
        entries,
        super::context::BackendGlobal {
            pipeline: Default::default(),
            target: CodegenTarget::Spirv,
        },
    );
    crate::spirv::lower_ssa_program(&program).unwrap()
}

pub(crate) fn wgsl(entries: Vec<EntryPoint>) -> String {
    let program = super::stage::WgslReady::from_parts(
        vec![],
        entries,
        super::context::BackendGlobal {
            pipeline: Default::default(),
            target: CodegenTarget::Wgsl,
        },
    );
    crate::wgsl::lower(&program).unwrap()
}

pub(crate) fn graphics(vertex: bool, colors: &[Vec<f32>]) -> EntryPoint {
    use crate::interface::{
        EntryInput, EntryInputKind, EntryOutput, EntryOutputDestination, EntryOutputKind, IoDecoration,
    };
    let input_ty = if vertex {
        types::Type::Constructed(types::TypeName::UInt(32), vec![])
    } else {
        types::vec(4, types::f32())
    };
    let mut builder = FuncBuilder::new(vec![input_ty.clone()], types::unit());
    let mut outputs = vec![];
    for (index, color) in colors.iter().enumerate() {
        let ty = types::vec(color.len(), types::f32());
        let value = builder
            .push_inst(
                InstKind::Op {
                    tag: OpTag::Vector,
                    operands: color
                        .iter()
                        .map(|value| ValueRef::Const(ConstantValue::from_f32(*value)))
                        .collect(),
                },
                ty.clone(),
            )
            .unwrap();
        let place = builder.new_place(ty.clone());
        builder.push_void_inst(InstKind::OutputSlot { index, result: place }).unwrap();
        builder
            .push_void_inst(InstKind::Store {
                place,
                value: value.into(),
            })
            .unwrap();
        outputs.push(EntryOutput {
            ty,
            host: (),
            kind: EntryOutputKind::Value {
                destination: if vertex && index == 0 {
                    EntryOutputDestination::BuiltIn(spirv::BuiltIn::Position)
                } else {
                    EntryOutputDestination::Location((index - usize::from(vertex)) as u32)
                },
            },
        });
    }
    builder.terminate(Terminator::Return(None)).unwrap();
    EntryPoint {
        id: EntryId::from_index(0),
        name: "main".into(),
        body: builder.finish().unwrap(),
        execution_model: if vertex { ExecutionModel::Vertex } else { ExecutionModel::Fragment },
        inputs: vec![EntryInput {
            name: "input".into(),
            ty: input_ty,
            host: (),
            kind: EntryInputKind::Value {
                decoration: Some(IoDecoration::BuiltIn(if vertex {
                    spirv::BuiltIn::VertexIndex
                } else {
                    spirv::BuiltIn::FragCoord
                })),
            },
        }],
        parameter_inputs: vec![vec![0]],
        outputs,
        storage_bindings: vec![],
        stage_descriptor_storage_accesses: Default::default(),
        pipeline_storage_accesses: Default::default(),
        span: Span::generated(),
    }
}

pub(crate) fn scalar_entry(
    params: Vec<types::Type>,
    result: types::Type,
    build: impl FnOnce(&mut FuncBuilder) -> ValueRef,
) -> EntryPoint {
    use crate::interface::{
        EntryInput, EntryInputKind, EntryOutput, EntryOutputDestination, EntryOutputKind, IoDecoration,
    };
    let mut builder = FuncBuilder::new(params.clone(), types::unit());
    let value = build(&mut builder);
    let place = builder.new_place(result.clone());
    builder
        .push_void_inst(InstKind::OutputSlot {
            index: 0,
            result: place,
        })
        .unwrap();
    builder.push_void_inst(InstKind::Store { place, value }).unwrap();
    builder.terminate(Terminator::Return(None)).unwrap();
    EntryPoint {
        id: EntryId::from_index(0),
        name: "scalar".into(),
        body: builder.finish().unwrap(),
        execution_model: ExecutionModel::Fragment,
        parameter_inputs: (0..params.len()).map(|index| vec![index]).collect(),
        inputs: params
            .into_iter()
            .enumerate()
            .map(|(index, ty)| EntryInput {
                name: format!("input_{index}"),
                ty,
                host: (),
                kind: EntryInputKind::Value {
                    decoration: Some(IoDecoration::Location(index as u32)),
                },
            })
            .collect(),
        outputs: vec![EntryOutput {
            ty: result,
            host: (),
            kind: EntryOutputKind::Value {
                destination: EntryOutputDestination::Location(0),
            },
        }],
        storage_bindings: vec![],
        stage_descriptor_storage_accesses: Default::default(),
        pipeline_storage_accesses: Default::default(),
        span: Span::generated(),
    }
}

pub(crate) fn indexed_constant(repeated: bool) -> super::stage::Elaborated {
    let entry = scalar_entry(vec![types::i32()], types::i32(), |builder| {
        let ty = types::sized_array(4, types::i32());
        let array = builder
            .push_inst(
                InstKind::Op {
                    tag: OpTag::ArrayLit,
                    operands: [10, 20, 30, 40]
                        .into_iter()
                        .map(|n| ValueRef::Const(ConstantValue::I32(n)))
                        .collect(),
                },
                ty.clone(),
            )
            .unwrap();
        let array = builder
            .push_inst(
                InstKind::Op {
                    tag: OpTag::Materialize,
                    operands: vec![array.into()],
                },
                ty,
            )
            .unwrap();
        builder
            .push_inst(
                InstKind::Op {
                    tag: OpTag::Index,
                    operands: vec![array.into(), builder.get_param(0).into()],
                },
                types::i32(),
            )
            .unwrap()
            .into()
    });
    let mut entries = vec![entry];
    if repeated {
        // Separate bodies prevent local expression sharing from making global deduplication vacuous.
        let mut second = entries[0].clone();
        second.id = EntryId::from_index(1);
        second.name = "other".into();
        entries.push(second);
    }
    super::stage::Elaborated::from_parts(
        vec![],
        entries,
        super::context::BackendGlobal {
            pipeline: Default::default(),
            target: CodegenTarget::Portable,
        },
    )
}
