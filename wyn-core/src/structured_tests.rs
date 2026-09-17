use super::{structurize, Node};
use crate::flow::ControlHeader;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::Terminator;
use crate::types::{bool_type, unit};

#[test]
fn sibling_selections_do_not_accumulate_recursion_depth() {
    let mut builder = FuncBuilder::new(vec![(bool_type(), "condition".into())], unit());
    let condition = builder.get_param(0);
    for _ in 0..128 {
        let header = builder.current_block().unwrap();
        let yes = builder.create_block();
        let no = builder.create_block();
        let merge = builder.create_block();
        builder.set_control_header(header, ControlHeader::Selection { merge });
        builder
            .terminate(Terminator::CondBranch {
                cond: condition.into(),
                then_target: yes,
                then_args: vec![],
                else_target: no,
                else_args: vec![],
            })
            .unwrap();
        for arm in [yes, no] {
            builder.switch_to_block(arm).unwrap();
            builder
                .terminate(Terminator::Branch {
                    target: merge,
                    args: vec![],
                })
                .unwrap();
        }
        builder.switch_to_block(merge).unwrap();
    }
    builder.terminate(Terminator::Return(None)).unwrap();
    let nodes = structurize(&builder.finish().unwrap());
    assert_eq!(nodes.iter().filter(|n| matches!(n, Node::If { .. })).count(), 128);
}
