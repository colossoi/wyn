use super::{Expressions, MAX_FRONTIER};
use crate::egglog::data::{intern_expr, intern_type, Array, ExprKind, Ir, ParameterId};
use crate::types::{array_variant_composite, function, no_buffer, unsized_array, Type, TypeName};
use std::collections::BTreeSet;

#[test]
fn wide_shared_calculations_keep_bounded_edges_and_all_inputs() {
    let mut data = Ir::default();
    let float = Type::Constructed(TypeName::Float(32), vec![]);
    let ty = intern_type(&mut data, float.clone());
    let op_ty = intern_type(&mut data, function(float.clone(), function(float.clone(), float)));
    let op = intern_expr(&mut data, op_ty, ExprKind::BinOp("+".into()));
    let mut value = intern_expr(&mut data, ty, ExprKind::Parameter(ParameterId::from(0)));
    let mut inputs = BTreeSet::from([value]);
    let mut roots = vec![];
    for i in 1..256 {
        let input = intern_expr(&mut data, ty, ExprKind::Parameter(ParameterId::from(i)));
        inputs.insert(input);
        value = intern_expr(
            &mut data,
            ty,
            ExprKind::PureApp {
                function: op,
                args: vec![value, input],
            },
        );
        roots.push(value);
    }
    let mut expressions = Expressions::new(&data);
    let mut pending = roots;
    let mut visited = BTreeSet::new();
    let mut leaves = BTreeSet::new();
    while let Some(e) = pending.pop() {
        if !visited.insert(e) {
            continue;
        }
        let children = expressions.children(e);
        assert!(children.len() <= 2 * MAX_FRONTIER + 1);
        if children.is_empty() {
            leaves.insert(e);
        }
        pending.extend(children);
    }
    assert_eq!(leaves, inputs);
}

#[test]
fn integer_dependencies_are_compact_until_used_as_an_array_extent() {
    let mut data = Ir::default();
    let integer = Type::Constructed(TypeName::Int(32), vec![]);
    let ty = intern_type(&mut data, integer.clone());
    let op_ty = intern_type(
        &mut data,
        function(integer.clone(), function(integer.clone(), integer.clone())),
    );
    let op = intern_expr(&mut data, op_ty, ExprKind::BinOp("+".into()));
    let input = intern_expr(&mut data, ty, ExprKind::Parameter(ParameterId::from(0)));
    let one = intern_expr(&mut data, ty, ExprKind::Int("1".into()));
    let child = intern_expr(
        &mut data,
        ty,
        ExprKind::PureApp {
            function: op,
            args: vec![input, one],
        },
    );
    let result = intern_expr(
        &mut data,
        ty,
        ExprKind::PureApp {
            function: op,
            args: vec![child, one],
        },
    );
    assert_eq!(Expressions::new(&data).children(result), vec![input]);
    let array_ty = intern_type(
        &mut data,
        unsized_array(integer, array_variant_composite(), no_buffer()),
    );
    intern_expr(
        &mut data,
        array_ty,
        ExprKind::Array(Array::Range {
            start: one,
            len: result,
            step: None,
        }),
    );
    let mut expressions = Expressions::new(&data);
    assert_eq!(
        expressions.children(result),
        data.expressions[result].kind.children()
    );
    assert!(expressions.needs_size(child));
    assert!(expressions.needs_size(one));
}

#[test]
fn computed_tuple_fields_keep_their_individual_identities() {
    let mut data = Ir::default();
    let float = Type::Constructed(TypeName::Float(32), vec![]);
    let ty = intern_type(&mut data, float.clone());
    let op_ty = intern_type(
        &mut data,
        function(float.clone(), function(float.clone(), float.clone())),
    );
    let op = intern_expr(&mut data, op_ty, ExprKind::BinOp("+".into()));
    let leaf = intern_expr(&mut data, ty, ExprKind::FloatBits(0));
    let field = intern_expr(
        &mut data,
        ty,
        ExprKind::PureApp {
            function: op,
            args: vec![leaf, leaf],
        },
    );
    let tuple_ty = intern_type(
        &mut data,
        Type::Constructed(TypeName::Tuple(2), vec![float.clone(), float]),
    );
    let tuple = intern_expr(&mut data, tuple_ty, ExprKind::Tuple(vec![field, leaf]));
    assert_eq!(Expressions::new(&data).children(tuple), vec![field, leaf]);
}

#[test]
fn vector_dependencies_preserve_selected_lanes_and_size_fields() {
    let mut data = Ir::default();
    let integer = Type::Constructed(TypeName::Int(32), vec![]);
    let ty = intern_type(&mut data, integer.clone());
    let vector_ty = intern_type(&mut data, crate::types::vec(2, integer.clone()));
    let first = intern_expr(&mut data, ty, ExprKind::Parameter(ParameterId::from(0)));
    let second = intern_expr(&mut data, ty, ExprKind::Parameter(ParameterId::from(1)));
    let vector = intern_expr(&mut data, vector_ty, ExprKind::Vector(vec![first, second]));
    let lane = intern_expr(
        &mut data,
        ty,
        ExprKind::Project {
            tuple: vector,
            index: 1,
        },
    );
    let mut expressions = Expressions::new(&data);
    assert_eq!(expressions.children(vector), vec![first, second]);
    assert_eq!(expressions.children(lane), vec![second]);
    let array_ty = intern_type(
        &mut data,
        unsized_array(integer, array_variant_composite(), no_buffer()),
    );
    intern_expr(
        &mut data,
        array_ty,
        ExprKind::Array(Array::Range {
            start: first,
            len: lane,
            step: None,
        }),
    );
    let mut expressions = Expressions::new(&data);
    assert!(!expressions.compact_vector(vector));
    assert!(!expressions.compact_vector(lane));
    assert_eq!(expressions.children(lane), vec![vector]);
}
