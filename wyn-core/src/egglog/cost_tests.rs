use super::*;

#[test]
fn local_loop_guard_accounts_for_work_memory_and_readbacks() {
    let model = CostModel::default();
    assert!(model.prefer_local_loop(80, 20, true));
    assert!(!model.prefer_local_loop(1_000_000, 20, true));
    assert!(!model.prefer_local_loop(80, 1_000_000, true));
    let immediate = CostModel {
        launch_ns: 0.0,
        readback_ns: 0.0,
        ..model
    };
    assert!(!immediate.prefer_local_loop(80, 20, true));
    assert!(CostModel {
        readback_ns: f64::NAN,
        ..model
    }
    .validate()
    .is_err());
}

#[test]
fn cost_ranking_responds_to_work_memory_and_latency() {
    let model = CostModel::default();
    let work = CollectiveWork {
        elements: 8192,
        input_bytes: 8,
        output_bytes: 8,
        state_bytes: 64,
        map_ops: 16,
        combine_ops: 8,
        scan: true,
    };
    let base = model.estimate(work, 64, 128);
    assert!(
        model.estimate(
            CollectiveWork {
                map_ops: 10000,
                ..work
            },
            64,
            128
        ) > base
    );
    assert!(
        model.estimate(
            CollectiveWork {
                input_bytes: 10000,
                ..work
            },
            64,
            128
        ) > base
    );
    assert!(
        CostModel {
            barrier_ns: model.barrier_ns * 2.0,
            ..model
        }
        .estimate(work, 64, 128)
            > base
    );
    assert!(
        CostModel {
            launch_ns: model.launch_ns * 2.0,
            ..model
        }
        .estimate(work, 64, 128)
            > base
    );
    assert!(CostModel {
        memory_bytes_per_ns: f64::NAN,
        ..model
    }
    .validate()
    .is_err());
    for n in [0, 1, 257, 4320, 100_000] {
        let choice = model.select(CollectiveWork { elements: n, ..work }, 64, 256).unwrap();
        assert!((1..=256).contains(&choice));
    }
    assert!(model.select(work, 0, 256).is_err());
    assert!(model.select(work, 64, 0).is_err());
}
