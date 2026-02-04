#![cfg(test)]

use crate::ast::TypeName;
use crate::mir::ssa::*;
use polytype::Type;

#[test]
fn test_value_id_display() {
    assert_eq!(format!("{}", ValueId(0)), "%0");
    assert_eq!(format!("{}", ValueId(42)), "%42");
}

#[test]
fn test_block_id_display() {
    assert_eq!(format!("{}", BlockId(0)), "bb0");
    assert_eq!(format!("{}", BlockId::ENTRY), "bb0");
}

#[test]
fn test_func_body_params() {
    let body = FuncBody::new(
        vec![
            (Type::Constructed(TypeName::Int(32), vec![]), "x".to_string()),
            (Type::Constructed(TypeName::Int(32), vec![]), "y".to_string()),
        ],
        Type::Constructed(TypeName::Int(32), vec![]),
    );

    assert_eq!(body.params.len(), 2);
    assert_eq!(body.params[0].0, ValueId(0));
    assert_eq!(body.params[1].0, ValueId(1));
    assert_eq!(body.num_values(), 2);
    assert_eq!(body.num_blocks(), 1); // entry block
}
