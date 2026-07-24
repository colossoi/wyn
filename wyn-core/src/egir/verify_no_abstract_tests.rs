use crate::ast::TypeName;
use crate::egir::verify_no_abstract;
use crate::error::CompilerError;
use crate::flow::ExecutionModel;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::{EntryPoint, Function, Program};
use crate::{types, LookupMap};
use polytype::Type;

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn abstract_array_ty() -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            i32_ty(),
            types::array_variant_abstract(),
            Type::Constructed(TypeName::Size(4), vec![]),
            types::no_buffer(),
        ],
    )
}

fn composite_array_ty() -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            i32_ty(),
            types::array_variant_composite(),
            Type::Constructed(TypeName::Size(4), vec![]),
            types::no_buffer(),
        ],
    )
}

fn empty_program() -> Program {
    Program::bare(Vec::new(), Vec::new(), Vec::new())
}

fn function_with(params: Vec<(Type<TypeName>, String)>, return_ty: Type<TypeName>) -> Function {
    let builder = FuncBuilder::new(params, return_ty);
    Function {
        name: "f".into(),
        body: builder.finish_unchecked(),
        span: crate::ast::Span::dummy(),
        linkage_name: None,
    }
}

fn entry_with(params: Vec<(Type<TypeName>, String)>, return_ty: Type<TypeName>) -> EntryPoint {
    let builder = FuncBuilder::new(params, return_ty);
    EntryPoint {
        name: "main".into(),
        body: builder.finish_unchecked(),
        execution_model: ExecutionModel::Compute {
            local_size: (64, 1, 1),
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        storage_bindings: Vec::new(),
        pipeline_storage_accesses: LookupMap::new(),
        span: crate::ast::Span::dummy(),
    }
}

/// An Abstract variant in a function parameter slot is the canonical
/// failure mode — a non-inlined size-poly helper still carrying
/// representation polymorphism into SSA.
#[test]
fn rejects_abstract_on_function_param() {
    let mut program = empty_program();
    program.functions.push(function_with(vec![(abstract_array_ty(), "xs".into())], i32_ty()));

    let err = verify_no_abstract::run(&program).expect_err("Abstract param must be rejected");
    match err {
        CompilerError::TypeError(msg, _) => {
            assert!(
                msg.contains("abstract") && msg.contains("function `f`"),
                "error should name the variant and the scope: {msg}"
            );
        }
        other => panic!("expected TypeError, got {other:?}"),
    }
}

/// Abstract on an entry-point parameter likewise rejected; entries
/// don't get a free pass.
#[test]
fn rejects_abstract_on_entry_param() {
    let mut program = empty_program();
    program.entry_points.push(entry_with(vec![(abstract_array_ty(), "xs".into())], i32_ty()));

    let err = verify_no_abstract::run(&program).expect_err("Abstract entry param must be rejected");
    assert!(matches!(err, CompilerError::TypeError(_, _)));
}

/// Abstract nested inside a tuple field still trips the verifier; the
/// walk has to be structural.
#[test]
fn rejects_abstract_nested_in_tuple() {
    let tuple_with_abstract = Type::Constructed(TypeName::Tuple(2), vec![i32_ty(), abstract_array_ty()]);

    let mut program = empty_program();
    program.functions.push(function_with(vec![(tuple_with_abstract, "t".into())], i32_ty()));

    let err = verify_no_abstract::run(&program).expect_err("nested Abstract must be rejected");
    assert!(matches!(err, CompilerError::TypeError(_, _)));
}

/// Concrete variants (Composite, View, Bounded, Virtual) are legal —
/// the verifier is narrow on purpose.
#[test]
fn accepts_concrete_variants() {
    let view_arr = Type::Constructed(
        TypeName::Array,
        vec![
            i32_ty(),
            types::array_variant_view(),
            Type::Variable(99),
            types::no_buffer(),
        ],
    );
    let bounded_arr = Type::Constructed(
        TypeName::Array,
        vec![
            i32_ty(),
            types::array_variant_bounded(),
            Type::Constructed(TypeName::Size(8), vec![]),
            types::no_buffer(),
        ],
    );

    let mut program = empty_program();
    program.functions.push(function_with(vec![(composite_array_ty(), "c".into())], i32_ty()));
    program.functions.push(function_with(vec![(view_arr, "v".into())], i32_ty()));
    program.functions.push(function_with(vec![(bounded_arr, "b".into())], i32_ty()));

    verify_no_abstract::run(&program).expect("concrete variants must pass the verifier");
}

/// Empty program — trivially clean.
#[test]
fn accepts_empty_program() {
    verify_no_abstract::run(&empty_program()).expect("empty program passes");
}
