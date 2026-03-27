#![cfg(test)]

use crate::ast::TypeName;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::Terminator;
use polytype::Type;

#[test]
fn test_func_body_params() {
    let mut builder = FuncBuilder::new(
        vec![
            (Type::Constructed(TypeName::Int(32), vec![]), "x".to_string()),
            (Type::Constructed(TypeName::Int(32), vec![]), "y".to_string()),
        ],
        Type::Constructed(TypeName::Int(32), vec![]),
    );

    let x = builder.get_param(0);
    let y = builder.get_param(1);
    let sum = builder.push_binop("+", x, y, Type::Constructed(TypeName::Int(32), vec![])).unwrap();
    builder.terminate(Terminator::Return(Some(sum))).unwrap();

    let body = builder.finish().unwrap();

    assert_eq!(body.params.len(), 2);
    assert_ne!(body.params[0].0, body.params[1].0);
    assert_eq!(body.num_blocks(), 1);
    assert_eq!(body.num_insts(), 1);
}
