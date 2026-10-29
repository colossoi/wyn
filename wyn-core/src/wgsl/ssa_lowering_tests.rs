//! WGSL backend unit tests.

// Test assertions intentionally panic on unexpected errors; production code
// retains the parent module's unwrap/expect bans.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::{validate_wgsl_identifier, wgsl_mangle, TypeEmitter};
use crate::ast::TypeName;
use crate::ssa;
use crate::types;
use crate::wgsl;
use crate::CodegenTarget;
use polytype::Type as PolyType;

// ---------- wgsl_mangle: per-char cases ----------

#[test]
fn mangle_plain_passthrough() {
    assert_eq!(wgsl_mangle("foo"), "w_foo");
    assert_eq!(wgsl_mangle("a1b2c3"), "w_a1b2c3");
}

#[test]
fn mangle_dot() {
    assert_eq!(wgsl_mangle("a.b"), "w_a_Db");
    assert_eq!(
        wgsl_mangle("materials.pbrDistributionGGX"),
        "w_materials_DpbrDistributionGGX"
    );
}

#[test]
fn mangle_dollar() {
    assert_eq!(wgsl_mangle("foo$0"), "w_foo_S0");
}

#[test]
fn mangle_underscore_inside() {
    // `_` inside a name → `_U`.
    assert_eq!(wgsl_mangle("a_b"), "w_a_Ub");
}

#[test]
fn mangle_leading_underscore_contracts_prefix() {
    // Leading non-alnum char contracts the prefix from `w_` to `w` so we
    // never produce `__` at the prefix/body boundary.
    assert_eq!(wgsl_mangle("_foo"), "w_Ufoo");
    assert_eq!(wgsl_mangle("_w_intrinsic_foo"), "w_Uw_Uintrinsic_Ufoo");
}

#[test]
fn mangle_empty() {
    // Empty input: no leading char, so no prefix underscore. Prefix alone.
    assert_eq!(wgsl_mangle(""), "w");
}

#[test]
fn mangle_non_ascii_fallback() {
    // Non-ASCII char exercises the `_X<hex>_` fallback.
    assert_eq!(wgsl_mangle("a-b"), "w_a_X2d_b");
    assert_eq!(wgsl_mangle("-ab"), "w_X2d_ab");
}

#[test]
fn mangle_no_double_underscore() {
    // Invariant: no mangled output contains `__`. Spot-check a few edge
    // cases; the mangler's design prevents it structurally.
    for input in ["_w_lambda_13", "a__b", "foo_bar_baz", "_", "__", "a._b", "a.$b"] {
        let out = wgsl_mangle(input);
        assert!(
            !out.contains("__"),
            "mangle({:?}) produced {:?} which contains `__`",
            input,
            out
        );
    }
}

// ---------- validate_wgsl_identifier ----------

#[test]
fn validate_accepts_plain_ident() {
    assert!(validate_wgsl_identifier("iResolution").is_ok());
    assert!(validate_wgsl_identifier("my_buffer_0").is_ok());
    assert!(validate_wgsl_identifier("_leading_underscore_ok_for_host").is_ok());
}

#[test]
fn validate_rejects_empty() {
    assert!(validate_wgsl_identifier("").is_err());
}

#[test]
fn validate_rejects_digit_leading() {
    assert!(validate_wgsl_identifier("0invalid").is_err());
}

#[test]
fn validate_rejects_illegal_chars() {
    assert!(validate_wgsl_identifier("foo-bar").is_err());
    assert!(validate_wgsl_identifier("foo.bar").is_err());
}

#[test]
fn validate_rejects_reserved_keyword() {
    for kw in &["fn", "let", "var", "struct", "return", "loop"] {
        assert!(
            validate_wgsl_identifier(kw).is_err(),
            "keyword {} must be rejected",
            kw
        );
    }
}

#[test]
fn validate_rejects_reserved_type_name() {
    for ty in &["f32", "i32", "u32", "bool", "vec3", "mat4x4", "array"] {
        assert!(
            validate_wgsl_identifier(ty).is_err(),
            "type name {} must be rejected",
            ty
        );
    }
}

#[test]
fn validate_rejects_double_underscore_prefix() {
    // WGSL reserves `__...` for the implementation.
    assert!(validate_wgsl_identifier("__foo").is_err());
    assert!(validate_wgsl_identifier("_").is_err());
}

// ---------- type_to_wgsl ----------

fn scalar_ty(name: TypeName) -> PolyType<TypeName> {
    PolyType::Constructed(name, vec![])
}

#[test]
fn type_f32() {
    let mut e = TypeEmitter::new();
    assert_eq!(e.type_to_wgsl(&scalar_ty(TypeName::Float(32))).unwrap(), "f32");
}

#[test]
fn type_f16_and_vector_family() {
    let mut e = TypeEmitter::new();
    assert_eq!(e.type_to_wgsl(&scalar_ty(TypeName::Float(16))).unwrap(), "f16");
    for size in 2..=4 {
        let ty = PolyType::Constructed(
            TypeName::Vec,
            vec![
                scalar_ty(TypeName::Float(16)),
                PolyType::Constructed(TypeName::Size(size), vec![]),
            ],
        );
        assert_eq!(e.type_to_wgsl(&ty).unwrap(), format!("vec{size}<f16>"));
    }
    assert!(e.uses_f16());
}

#[test]
fn type_i32() {
    let mut e = TypeEmitter::new();
    assert_eq!(e.type_to_wgsl(&scalar_ty(TypeName::Int(32))).unwrap(), "i32");
}

#[test]
fn type_u32() {
    let mut e = TypeEmitter::new();
    assert_eq!(e.type_to_wgsl(&scalar_ty(TypeName::UInt(32))).unwrap(), "u32");
}

#[test]
fn type_bool() {
    let mut e = TypeEmitter::new();
    assert_eq!(e.type_to_wgsl(&scalar_ty(TypeName::Bool)).unwrap(), "bool");
}

#[test]
fn type_f64_rejected() {
    let mut e = TypeEmitter::new();
    let result = e.type_to_wgsl(&scalar_ty(TypeName::Float(64)));
    assert!(result.is_err(), "f64 must be rejected by WGSL type lowering");
}

#[test]
fn type_u64_requires_explicit_emulation_policy() {
    let ty = scalar_ty(TypeName::UInt(64));
    let mut default_emitter = TypeEmitter::new();
    assert!(default_emitter.type_to_wgsl(&ty).is_err());

    let mut emulating = TypeEmitter::with_options(wgsl::WgslOptions::U64_EMULATION);
    assert_eq!(emulating.type_to_wgsl(&ty).unwrap(), "vec2<u32>");
}

#[test]
fn type_vec3f32() {
    let mut e = TypeEmitter::new();
    let ty = PolyType::Constructed(
        TypeName::Vec,
        vec![
            scalar_ty(TypeName::Float(32)),
            PolyType::Constructed(TypeName::Size(3), vec![]),
        ],
    );
    assert_eq!(e.type_to_wgsl(&ty).unwrap(), "vec3<f32>");
}

#[test]
fn type_mat4x4f32() {
    let mut e = TypeEmitter::new();
    let ty = PolyType::Constructed(
        TypeName::Mat,
        vec![
            scalar_ty(TypeName::Float(32)),
            PolyType::Constructed(TypeName::Size(4), vec![]),
            PolyType::Constructed(TypeName::Size(4), vec![]),
        ],
    );
    assert_eq!(e.type_to_wgsl(&ty).unwrap(), "mat4x4<f32>");
}

#[test]
fn type_array_sized() {
    let mut e = TypeEmitter::new();
    let ty = PolyType::Constructed(
        TypeName::Array,
        vec![
            scalar_ty(TypeName::Float(32)),
            PolyType::Constructed(TypeName::ArrayVariantComposite, vec![]),
            PolyType::Constructed(TypeName::Size(8), vec![]),
            types::no_buffer(),
        ],
    );
    assert_eq!(e.type_to_wgsl(&ty).unwrap(), "array<f32, 8>");
}

#[test]
fn type_tuple_creates_struct() {
    let mut e = TypeEmitter::new();
    let ty = PolyType::Constructed(
        TypeName::Tuple(2),
        vec![scalar_ty(TypeName::Float(32)), scalar_ty(TypeName::Int(32))],
    );
    let name = e.type_to_wgsl(&ty).unwrap();
    assert_eq!(name, "T0");
    assert_eq!(
        e.tuple_structs.get("T0").unwrap(),
        &vec!["f32".to_string(), "i32".to_string()]
    );
}

#[test]
fn type_tuple_caches_by_signature() {
    let mut e = TypeEmitter::new();
    let ty = PolyType::Constructed(
        TypeName::Tuple(2),
        vec![scalar_ty(TypeName::Float(32)), scalar_ty(TypeName::Int(32))],
    );
    let n1 = e.type_to_wgsl(&ty).unwrap();
    let n2 = e.type_to_wgsl(&ty).unwrap();
    assert_eq!(n1, n2);
    assert_eq!(e.tuple_structs.len(), 1);
}

#[test]
fn type_tuple_distinct_shapes_distinct_structs() {
    let mut e = TypeEmitter::new();
    let ty_a = PolyType::Constructed(
        TypeName::Tuple(2),
        vec![scalar_ty(TypeName::Float(32)), scalar_ty(TypeName::Int(32))],
    );
    let ty_b = PolyType::Constructed(
        TypeName::Tuple(2),
        vec![scalar_ty(TypeName::Int(32)), scalar_ty(TypeName::Float(32))],
    );
    let na = e.type_to_wgsl(&ty_a).unwrap();
    let nb = e.type_to_wgsl(&ty_b).unwrap();
    assert_ne!(na, nb);
    assert_eq!(e.tuple_structs.len(), 2);
}

// ---------- end-to-end lowering ----------

#[test]
fn lower_empty_program_succeeds() {
    // Empty program: no functions, no entries — lower emits just the
    // header comment and returns a string.
    let program = ssa::types::Program::<ssa::stage::WgslReadyTag, _>::from_parts(
        Vec::new(),
        Vec::new(),
        ssa::context::BackendGlobal {
            pipeline: Default::default(),
            target: CodegenTarget::Wgsl,
        },
    );
    let out = super::lower(&program).expect("empty program should lower");
    assert!(out.contains("WGSL backend"));
}

// ---------- naga-validated end-to-end ----------

/// Parse + validate WGSL text through naga. Panics with naga's diagnostic
/// message on failure so test output points directly at the offending
/// line. Used by end-to-end tests that compile a `.wyn` source through
/// the full pipeline to WGSL.
fn validate_wgsl(source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("naga parse failed:\n{}\n\n--- source ---\n{}", e, source));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("naga validation failed:\n{:?}\n\n--- source ---\n{}", e, source));
}

/// Compile a Wyn source through the full pipeline to WGSL text.

#[test]
fn wgsl_scope_tree_owns_loop_exports_and_preserves_state_swaps() {
    use crate::ast::Span;
    use crate::flow::ControlHeader;
    use crate::op::{BinaryOperator, OpTag};
    use crate::ssa::builder::FuncBuilder;
    use crate::ssa::types::{ConstantValue, InstKind, Terminator, ValueRef};
    use crate::structured::{self, Node};

    let int = types::i32();
    let number = |n| ValueRef::Const(ConstantValue::I32(n));
    let binary = |op, a, b| InstKind::Op {
        tag: OpTag::BinOp(op),
        operands: vec![a, b],
    };
    let mut builder = FuncBuilder::new(vec![types::bool_type(), int.clone()], int.clone());
    let yes = builder.create_block();
    let no = builder.create_block();
    let merge = builder.create_block();
    let result = builder.add_block_param(merge, int.clone());
    let header = builder.create_block();
    let a = builder.add_block_param(header, int.clone());
    let b = builder.add_block_param(header, int.clone());
    let i = builder.add_block_param(header, int.clone());
    let step = builder.create_block();
    let continuing = builder.create_block();
    let next = (0..3).map(|_| builder.add_block_param(continuing, int.clone())).collect::<Vec<_>>();
    let exit = builder.create_block();
    builder.set_control_header(builder.entry(), ControlHeader::Selection { merge });
    builder
        .terminate(Terminator::CondBranch {
            cond: builder.get_param(0).into(),
            then_target: yes,
            then_args: vec![],
            else_target: no,
            else_args: vec![],
        })
        .unwrap();
    builder.switch_to_block(yes).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![number(1), number(2), number(0)],
        })
        .unwrap();
    builder.switch_to_block(header).unwrap();
    builder.set_control_header(
        header,
        ControlHeader::Loop {
            merge: exit,
            continue_block: continuing,
        },
    );
    let exported =
        builder.push_inst(binary(BinaryOperator::Subtract, a.into(), b.into()), int.clone()).unwrap();
    let condition = builder
        .push_inst(
            binary(BinaryOperator::Less, i.into(), builder.get_param(1).into()),
            types::bool_type(),
        )
        .unwrap();
    builder
        .terminate(Terminator::CondBranch {
            cond: condition.into(),
            then_target: step,
            then_args: vec![],
            else_target: exit,
            else_args: vec![],
        })
        .unwrap();
    builder.switch_to_block(step).unwrap();
    let increment =
        builder.push_inst(binary(BinaryOperator::Add, i.into(), number(1)), int.clone()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: continuing,
            args: vec![b.into(), a.into(), increment.into()],
        })
        .unwrap();
    builder.switch_to_block(continuing).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: header,
            args: next.iter().copied().map(Into::into).collect(),
        })
        .unwrap();
    builder.switch_to_block(exit).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: merge,
            args: vec![exported.into()],
        })
        .unwrap();
    builder.switch_to_block(no).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: merge,
            args: vec![number(0)],
        })
        .unwrap();
    builder.switch_to_block(merge).unwrap();
    builder.terminate(Terminator::Return(Some(result.into()))).unwrap();
    let mut body = builder.finish().unwrap();
    ssa::ir::eliminate_single_input_params(&mut body.inner);
    assert!(body.inner.blocks[continuing].params.is_empty());

    let mut tree = structured::structurize(&body);
    super::super::bindings::place_bindings(&body, &mut tree);
    assert_eq!(tree.root.declarations, vec![result]);
    let Node::If {
        then_body, else_body, ..
    } = &tree.root.nodes[0]
    else {
        panic!("selection")
    };
    assert!(else_body.declarations.is_empty());
    assert_eq!(then_body.declarations.len(), 4);
    for value in [a, b, i, exported] {
        assert!(then_body.declarations.contains(&value));
    }
    let loop_body = then_body
        .nodes
        .iter()
        .find_map(|node| match node {
            Node::Loop { body } => Some(body),
            _ => None,
        })
        .unwrap();
    assert!(loop_body.declarations.is_empty());
    assert!(
        loop_body
            .nodes
            .iter()
            .any(|node| matches!(node, Node::Inst(id) if body.get_inst(*id).result == Some(exported))),
        "moving a declaration must not move its calculation"
    );

    let program = ssa::types::Program::<ssa::stage::WgslReadyTag, _>::from_parts(
        vec![],
        vec![],
        ssa::context::BackendGlobal {
            pipeline: Default::default(),
            target: CodegenTarget::Wgsl,
        },
    );
    let mut ctx = super::LowerCtx::new(&program, Default::default());
    let mut emitter = super::BodyLowerCtx::new(&mut ctx, &body, Span::generated());
    let mut wgsl = format!(
        "fn test({}: bool, {}: i32) -> i32 {{\n",
        super::wgsl_var(body.param(0).unwrap().0),
        super::wgsl_var(body.param(1).unwrap().0)
    );
    let returned = emitter.lower(&mut wgsl).unwrap();
    wgsl.push_str(&format!("return {returned};\n}}\n"));
    validate_wgsl(&wgsl);
    let save_b = wgsl.find(&format!("let _copy3: i32 = {};", super::wgsl_var(b))).unwrap();
    let save_a = wgsl.find(&format!("let _copy4: i32 = {};", super::wgsl_var(a))).unwrap();
    let write_a = wgsl.find(&format!("{} = _copy3;", super::wgsl_var(a))).unwrap();
    let write_b = wgsl.find(&format!("{} = _copy4;", super::wgsl_var(b))).unwrap();
    assert!(
        save_a < write_a && save_a < write_b && save_b < write_a && save_b < write_b,
        "both original loop values must be read before either is overwritten:\n{wgsl}"
    );
}

#[test]
fn unrelated_entries_preserve_storage_buffer_element_types() {
    use crate::ast::Span;
    use crate::interface::{StorageBindingDecl, StorageRole};
    use crate::ssa::builder::FuncBuilder;
    use crate::ssa::types::{EntryPoint, ExecutionModel, Terminator};
    use crate::BindingRef;
    use crate::{EntryId, LookupMap};
    let binding = BindingRef::new(0, 0);
    for reordered in [false, true] {
        let mut entries = Vec::new();
        for (index, ty) in [types::f32(), types::vec(4, types::f32()), types::f32()].into_iter().enumerate()
        {
            let mut builder = FuncBuilder::new(vec![], types::unit());
            builder.terminate(Terminator::Return(None)).unwrap();
            entries.push(EntryPoint {
                id: EntryId::from_index(index),
                name: format!("entry_{index}"),
                body: builder.finish().unwrap(),
                execution_model: ExecutionModel::Compute {
                    local_size: (1, 1, 1),
                },
                inputs: vec![],
                parameter_inputs: vec![],
                outputs: vec![],
                storage_bindings: vec![StorageBindingDecl {
                    binding,
                    role: StorageRole::Input,
                    elem_ty: ty,
                    host: (),
                }],
                stage_descriptor_storage_accesses: LookupMap::new(),
                pipeline_storage_accesses: LookupMap::new(),
                span: Span::generated(),
            });
        }
        if reordered {
            entries.swap(0, 1);
        }
        let program = ssa::stage::WgslReady::from_parts(
            vec![],
            entries,
            ssa::context::BackendGlobal {
                pipeline: Default::default(),
                target: CodegenTarget::Wgsl,
            },
        );
        let mut context = super::LowerCtx::new(&program, Default::default());
        let declarations = context.storage_buffer_declarations().unwrap();
        let name =
            |index| context.storage_names[&(Some(EntryId::from_index(index)), binding, false)].clone();
        assert_eq!(name(0), name(2), "identical types share a global");
        assert_ne!(name(0), name(1), "different types need distinct globals");
        assert!(declarations.contains(&format!("{}: array<f32>;", name(0))));
        assert!(declarations.contains(&format!("{}: array<vec4<f32>>;", name(1))));
        validate_wgsl(&declarations);
    }
}

fn float_bound_values(
    builder: &mut ssa::builder::FuncBuilder,
    ty: types::Type,
) -> Vec<ssa::types::ValueRef> {
    use crate::op::OpTag;
    use ssa::types::InstKind;
    let witness = builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::Float("0.0".into()),
                operands: vec![],
            },
            ty.clone(),
        )
        .unwrap();
    ["_w_intrinsic_float_highest", "_w_intrinsic_float_lowest"]
        .into_iter()
        .map(|name| {
            let id = crate::builtins::catalog().lookup_by_surface_name(name).unwrap().id;
            builder
                .push_inst(
                    InstKind::Op {
                        tag: OpTag::Intrinsic { id, overload_idx: 0 },
                        operands: vec![witness.into()],
                    },
                    ty.clone(),
                )
                .unwrap()
                .into()
        })
        .collect()
}

#[test]
fn float_bounds_in_arrays_lower_to_finite_wgsl_literals() {
    use crate::op::OpTag;
    use crate::ssa::types::InstKind;
    let entry = ssa::test_support::scalar_entry(vec![types::i32()], types::f32(), |builder| {
        let operands = float_bound_values(builder, types::f32());
        let array = builder
            .push_inst(
                InstKind::Op {
                    tag: OpTag::ArrayLit,
                    operands,
                },
                types::sized_array(2, types::f32()),
            )
            .unwrap();
        builder
            .push_inst(
                InstKind::Op {
                    tag: OpTag::Index,
                    operands: vec![array.into(), builder.get_param(0).into()],
                },
                types::f32(),
            )
            .unwrap()
            .into()
    });
    let program = ssa::stage::Elaborated::from_parts(
        vec![],
        vec![entry],
        ssa::context::BackendGlobal {
            pipeline: Default::default(),
            target: CodegenTarget::Wgsl,
        },
    );
    let wgsl = crate::lower_ssa_to_wgsl(program).unwrap();
    validate_wgsl(&wgsl);
    assert!(wgsl.contains("0x1.fffffep+127f"));
    assert!(wgsl.contains("-0x1.fffffep+127f"));
}

#[test]
fn float_bounds_preserve_infinity_bits_without_nonfinite_spirv_literals() {
    use wspirv::dr::Operand;
    for width in [32, 16] {
        use crate::op::OpTag;
        use crate::ssa::types::InstKind;
        let ty = scalar_ty(TypeName::Float(width));
        let vector = types::vec(2, ty.clone());
        let entry = ssa::test_support::storage_write_value(
            crate::BindingRef::new(0, 0),
            vector.clone(),
            |builder| {
                let operands = float_bound_values(builder, ty.clone());
                builder
                    .push_inst(
                        InstKind::Op {
                            tag: OpTag::Vector,
                            operands,
                        },
                        vector,
                    )
                    .unwrap()
                    .into()
            },
        );
        let spirv = ssa::test_support::spirv(vec![entry]);
        let module = wspirv::dr::load_words(&spirv).unwrap();
        let uint_type = module
            .types_global_values
            .iter()
            .find(|inst| {
                inst.class.opcode == spirv::Op::TypeInt
                    && inst.operands == vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)]
            })
            .unwrap()
            .result_id;
        for bits in [f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits()] {
            assert!(module.types_global_values.iter().any(|inst| {
                inst.class.opcode == spirv::Op::Constant
                    && inst.result_type == uint_type
                    && inst.operands == vec![Operand::LiteralBit32(bits)]
            }));
        }
        let bytes: Vec<u8> = spirv.iter().flat_map(|word| word.to_le_bytes()).collect();
        let module = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
fn f16_scalar_and_vectors_lower_to_valid_wgsl() {
    use crate::op::OpTag;
    use crate::ssa::types::{InstKind, ValueRef};
    let half = scalar_ty(TypeName::Float(16));
    let vector = types::vec(4, half.clone());
    let entry =
        ssa::test_support::storage_write_value(crate::BindingRef::new(0, 0), vector.clone(), |builder| {
            let operands = (1..=4)
                .map(|n| {
                    ValueRef::from(
                        builder
                            .push_inst(
                                InstKind::Op {
                                    tag: OpTag::Float(format!("{n}.0")),
                                    operands: vec![],
                                },
                                half.clone(),
                            )
                            .unwrap(),
                    )
                })
                .collect();
            builder
                .push_inst(
                    InstKind::Op {
                        tag: OpTag::Vector,
                        operands,
                    },
                    vector,
                )
                .unwrap()
                .into()
        });
    let wgsl = ssa::test_support::wgsl(vec![entry]);
    validate_wgsl(&wgsl);
    assert!(wgsl.contains("enable f16;"));
    assert!(wgsl.contains("1.0h"));
    assert!(wgsl.contains("vec4<f16>"));
}

#[test]
fn wgsl_fragment_trivial() {
    let wgsl = crate::ssa::test_support::wgsl(vec![crate::ssa::test_support::graphics(
        false,
        &[vec![1.0, 0.5, 0.0, 1.0]],
    )]);
    validate_wgsl(&wgsl);
    assert!(wgsl.contains("@fragment"));
    assert!(wgsl.contains("@builtin(position)"));
    assert!(wgsl.contains("@location(0)"));
}
#[test]
fn wgsl_vertex_full_screen_triangle() {
    let wgsl = crate::ssa::test_support::wgsl(vec![crate::ssa::test_support::graphics(
        true,
        &[vec![0.0, 0.0, 0.0, 1.0]],
    )]);
    validate_wgsl(&wgsl);
    assert!(wgsl.contains("@vertex"));
    assert!(wgsl.contains("@builtin(vertex_index)"));
}
#[test]
fn wgsl_vertex_multi_output_struct() {
    let wgsl = crate::ssa::test_support::wgsl(vec![crate::ssa::test_support::graphics(
        true,
        &[vec![0.0, 0.0, 0.0, 1.0], vec![1.0, 0.0, 0.0]],
    )]);
    validate_wgsl(&wgsl);
    assert!(wgsl.contains("struct VsOut0"));
    assert!(wgsl.contains("@builtin(position) f0: vec4<f32>,"));
    assert!(wgsl.contains("@location(0) f1: vec3<f32>,"));
    assert!(wgsl.contains("-> VsOut0"));
    assert!(wgsl.contains("var _out_struct: VsOut0;"));
    assert!(wgsl.contains("_out_struct.f0 ="));
    assert!(wgsl.contains("_out_struct.f1 ="));
    assert!(wgsl.contains("return _out_struct;"));
}

/// Compile a source file from disk through the full pipeline to WGSL
/// and naga-validate the result. Used for testfile sweeps. Resolves
/// paths relative to the workspace root so tests work regardless of
/// the crate under `-p`.

#[test]
fn wgsl_compute_reduce_writes_to_storage_buffer() {
    // A parallelized `reduce` compute shader's terminal write must hit
    // the storage buffer directly:
    //
    //     _buf_0_1[(i32(off) + i32(tid))] = v_accum;
    //
    // Writing only a local `var` that mirrors the buffer slot is a no-op at
    // runtime even though naga accepts it:
    //
    //     var v27_1: f32 = _buf_0_1[(i32(off) + i32(tid))];
    //     v27_1 = v_accum;
    let wgsl = crate::ssa::test_support::wgsl(vec![crate::ssa::test_support::storage_write()]);
    validate_wgsl(&wgsl);

    // The bug pattern: `var vNN: f32 = _buf_...[...];` immediately
    // followed by `vNN = ...;` on the next non-blank line.
    let lines: Vec<&str> = wgsl.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("var ") {
            if let Some(eq_pos) = rest.find(" = ") {
                let name = &rest[..eq_pos].split(':').next().unwrap_or("").trim();
                let init = rest[eq_pos + 3..].trim_end_matches(';').trim();
                if init.starts_with("_buf_") && init.contains('[') {
                    // Look at the next non-blank line.
                    if let Some(next) = lines[i + 1..].iter().find(|l| !l.trim().is_empty()) {
                        let nt = next.trim();
                        let expected_bug = format!("{} = ", name);
                        assert!(
                            !nt.starts_with(&expected_bug),
                            "WGSL storage-write bug: `var {0}: ... = {1};` is followed by \
                             `{0} = ...;` — the write targets a dead local instead of \
                             the storage buffer.\n\n--- offending pair ---\n{2}\n{3}\n\n\
                             --- full WGSL ---\n{4}",
                            name,
                            init,
                            line,
                            next,
                            wgsl
                        );
                    }
                }
            }
        }
    }

    // Positive assertion: at least one direct storage-buffer write must
    // appear. The phase-1 terminal store writes the partial to
    // `_buf_0_1[offset + tid]`.
    assert!(
        wgsl.lines().any(|l| {
            let t = l.trim();
            t.starts_with("_buf_") && t.contains("] = ") && !t.contains(" = _buf_")
        }),
        "expected at least one direct `_buf_N_M[idx] = val;` write in emitted WGSL:\n{}",
        wgsl
    );
}

#[test]
fn wgsl_single_use_constant_vector_is_substituted() {
    let wgsl = crate::ssa::test_support::wgsl(vec![crate::ssa::test_support::graphics(
        false,
        &[vec![0.6, 0.7, 0.8, 1.0]],
    )]);
    validate_wgsl(&wgsl);
    assert!(wgsl.contains("@fragment"));
    assert!(wgsl.contains("@location(0)"));
    assert!(
        wgsl.contains("_out0 = vec4<f32>(0.6f, 0.7f, 0.8f, 1.0f);")
            && !wgsl
                .split("@fragment")
                .nth(1)
                .unwrap()
                .lines()
                .any(|line| line.trim_start().starts_with("let v")),
        "cheap single-use expressions should be substituted into their use:\n{wgsl}"
    );
}
/// `scatter` into a `#[storage(access=write)]` framebuffer: the destination
/// binding must be emitted `read_write` (WGSL requires it for the Store, and
/// the declared `access=write` must propagate from the param), and the scatter
/// emits indexed stores into it.
#[test]
fn wgsl_const_array_dynamic_index_hoists_to_private_global() {
    // A compile-time-constant array indexed by a runtime value is hoisted
    // once to a module-scope `var<private>` (the WGSL analog of the SPIR-V
    // Private-global hoist) and indexed there, instead of being copied into
    // a per-call `var<function>` materialization.
    let wgsl = crate::lower_ssa_to_wgsl(crate::ssa::test_support::indexed_constant(false)).unwrap();
    validate_wgsl(&wgsl);
    assert!(
        wgsl.contains("var<private> _const_global_0:"),
        "expected a hoisted private global:\n{wgsl}"
    );
    assert!(
        wgsl.contains("_const_global_0["),
        "the runtime index must address the global:\n{wgsl}"
    );
    assert_eq!(
        wgsl.matches("array<i32, 4>(").count(),
        1,
        "the hoisted initializer must not also be emitted as a dead function-local array:\n{wgsl}"
    );
}

#[test]
fn wgsl_const_array_hoist_is_deduped() {
    // The same constant array indexed at two sites shares one global.
    let wgsl = crate::lower_ssa_to_wgsl(crate::ssa::test_support::indexed_constant(true)).unwrap();
    validate_wgsl(&wgsl);
    let n = wgsl.matches("var<private> _const_global").count();
    assert_eq!(
        n, 1,
        "two indexings of one constant must share one global:\n{wgsl}"
    );
}

#[test]
fn signed_shift_counts_validate_in_wgsl() {
    use crate::op::{BinaryOperator, OpTag};
    use crate::ssa::types::InstKind;
    for operator in [BinaryOperator::ShiftLeft, BinaryOperator::ShiftRight] {
        let entry =
            ssa::test_support::scalar_entry(vec![types::i32(), types::i32()], types::i32(), |builder| {
                builder
                    .push_inst(
                        InstKind::Op {
                            tag: OpTag::BinOp(operator),
                            operands: vec![builder.get_param(0).into(), builder.get_param(1).into()],
                        },
                        types::i32(),
                    )
                    .unwrap()
                    .into()
            });
        let text = ssa::test_support::wgsl(vec![entry]);
        assert!(
            text.contains(if operator == BinaryOperator::ShiftLeft { "<<" } else { ">>" }),
            "{text}"
        );
        validate_wgsl(&text);
    }
}

#[test]
fn helpers_with_identical_display_names_keep_distinct_identities() {
    use crate::op::{BinaryOperator, OpTag};
    use crate::ssa::{
        builder::FuncBuilder,
        types::{ConstantValue, Function, InstKind, Terminator, ValueRef},
    };
    let functions = [BinaryOperator::Add, BinaryOperator::Multiply]
        .into_iter()
        .enumerate()
        .map(|(index, operator)| {
            let mut builder = FuncBuilder::new(vec![types::i32()], types::i32());
            let value = builder
                .push_inst(
                    InstKind::Op {
                        tag: OpTag::BinOp(operator),
                        operands: vec![
                            builder.get_param(0).into(),
                            ValueRef::Const(ConstantValue::I32(2)),
                        ],
                    },
                    types::i32(),
                )
                .unwrap();
            builder.terminate(Terminator::Return(Some(value.into()))).unwrap();
            Function {
                id: crate::FunctionId::from_index(index as u32),
                name: "same_name".into(),
                body: builder.finish().unwrap(),
                span: crate::ast::Span::generated(),
                linkage_name: None,
            }
        })
        .collect();
    let entry = ssa::test_support::scalar_entry(vec![types::i32()], types::i32(), |builder| {
        let values = (0..2)
            .map(|index| {
                builder
                    .push_inst(
                        InstKind::Op {
                            tag: OpTag::Call(crate::FunctionId::from_index(index as u32)),
                            operands: vec![builder.get_param(0).into()],
                        },
                        types::i32(),
                    )
                    .unwrap()
                    .into()
            })
            .collect();
        builder
            .push_inst(
                InstKind::Op {
                    tag: OpTag::BinOp(BinaryOperator::Add),
                    operands: values,
                },
                types::i32(),
            )
            .unwrap()
            .into()
    });
    let program = ssa::stage::WgslReady::from_parts(
        functions,
        vec![entry],
        ssa::context::BackendGlobal {
            pipeline: Default::default(),
            target: CodegenTarget::Wgsl,
        },
    );
    let text = super::lower(&program).unwrap();
    assert_eq!(text.matches("fn w_Uw_Ufunction_").count(), 2);
    validate_wgsl(&text);
}
