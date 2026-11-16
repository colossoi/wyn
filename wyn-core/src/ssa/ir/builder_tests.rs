use super::{BuilderError, FuncBuilder, Terminator};

#[test]
fn explicit_unreachable_terminates_a_block() {
    let mut builder = FuncBuilder::<(), ()>::new();
    let dead = builder.create_block();
    builder
        .terminate(Terminator::Branch {
            target: dead,
            args: vec![],
        })
        .unwrap();
    builder.switch_to_block(dead).unwrap();
    builder.terminate(Terminator::Unreachable).unwrap();
    assert_eq!(
        builder.push_void_inst(()),
        Err(BuilderError::BlockAlreadyTerminated(dead))
    );
    builder.finish().unwrap();
}

#[test]
fn unfinished_blocks_are_still_rejected() {
    let mut builder = FuncBuilder::<(), ()>::new();
    let unfinished = builder.create_block();
    builder.terminate(Terminator::Return(None)).unwrap();
    assert_eq!(
        builder.finish().unwrap_err(),
        BuilderError::UnterminatedBlock(unfinished)
    );
}
