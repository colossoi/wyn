use super::*;

type Builder = StagedIrBuilder<&'static str, &'static str, u32, &'static str>;

#[test]
fn lowering_finishes_a_typed_resident_dag() {
    let mut builder = Builder::new();
    let producer = builder.add_stage("generated", "reduce").unwrap();
    let consumer = builder.add_stage("authored", "map").unwrap();
    builder.add_external_input("[]i32", 512, [producer]).unwrap();
    let flow = builder.add_flow(producer, "i32", 4).unwrap();
    builder.add_consumer(flow, consumer).unwrap();
    let output = builder.add_flow(consumer, "[]i32", 512).unwrap();
    builder.publish(output).unwrap();

    let ir = builder.finish().unwrap();
    assert_eq!(ir.topological_stages(), [producer, consumer]);
    assert_eq!(ir.stage(producer).unwrap().outgoing_flows(), [flow]);
    assert_eq!(ir.stage(consumer).unwrap().incoming_flows(), [flow]);
    assert_eq!(ir.flow(flow).unwrap().producer(), producer);
    assert_eq!(ir.flow(flow).unwrap().consumers(), [consumer]);
    assert!(ir.flow(output).unwrap().is_published());
}

#[test]
fn finish_rejects_a_destinationless_flow() {
    let mut builder = Builder::new();
    let stage = builder.add_stage("authored", "body").unwrap();
    let flow = builder.add_flow(stage, "i32", 4).unwrap();
    assert!(matches!(
        builder.finish(),
        Err(BuildError::FlowHasNoDestination { flow: candidate }) if candidate == flow
    ));
}

#[test]
fn checked_edges_reject_duplicates_self_dependencies_and_cycles() {
    let mut builder = Builder::new();
    let a = builder.add_stage("a", "a").unwrap();
    let b = builder.add_stage("b", "b").unwrap();
    let ab = builder.add_flow(a, "i32", 4).unwrap();
    builder.add_consumer(ab, b).unwrap();
    assert_eq!(
        builder.add_consumer(ab, b),
        Err(BuildError::DuplicateConsumer {
            flow: ab,
            consumer: b,
        })
    );
    assert_eq!(
        builder.add_consumer(ab, a),
        Err(BuildError::SelfDependency { stage: a })
    );
    let ba = builder.add_flow(b, "i32", 4).unwrap();
    assert_eq!(
        builder.add_consumer(ba, a),
        Err(BuildError::Cycle {
            producer: b,
            consumer: a,
        })
    );
}

#[test]
fn body_mutation_and_mapping_preserve_topology() {
    let mut builder = Builder::new();
    let stage = builder.add_stage("authored", "before").unwrap();
    let flow = builder.add_flow(stage, "i32", 4).unwrap();
    builder.publish(flow).unwrap();
    let mut ir = builder.finish().unwrap();

    *ir.stage_body_mut(stage).unwrap() = "after";
    let ir = ir.map_stage_bodies(|_, body| body.len());
    assert_eq!(ir.stage(stage).unwrap().body(), &5);
    assert_eq!(ir.flow(flow).unwrap().producer(), stage);
    assert!(ir.flow(flow).unwrap().is_published());
}
