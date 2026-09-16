//! Synthetic graph shapes isolate placement costs from parsing and EqSat.
use super::*;
use crate::egglog::{convert_program, snapshot};

fn imported(source: &str) -> AssociatedData {
    convert_program(&crate::tlc::infer_input_slice_bounds(
        crate::compile_thru_tlc(source).unwrap(),
    ))
    .unwrap()
    .data
}
fn root(data: &AssociatedData) -> RegionId {
    data.definitions[data.entries.values().next().unwrap().definition].body
}
fn maps(n: usize) -> AssociatedData {
    let mut data = imported("entry main(xs:[4]i32,bias:i32) [4]i32=map(|x:i32|x+bias*bias,xs)");
    let r = root(&data);
    let op = *data.regions[r].members.first().unwrap();
    let template = data.operations[op].clone();
    data.regions[r].members.clear();
    data.regions[r].results.clear();
    for i in 0..n {
        let op = data.operations.alloc(OperationData {
            source_position: i,
            ..template.clone()
        });
        data.regions[r].members.insert(op);
        let e = data.expressions.alloc(ExprData {
            ty: template.ty,
            kind: ExprKind::OperationResult(op),
        });
        data.regions[r].results.push(e);
    }
    data
}
fn conditionals(n: usize) -> AssociatedData {
    let mut data = imported("entry main(flag:bool,x:i32) i32=if flag then x*x+1 else x*x+2");
    let r = root(&data);
    let mut e = data.regions[r].results[0];
    let ExprKind::If { condition, .. } = data.expressions[e].kind else {
        panic!("scalar conditional")
    };
    let ty = data.expressions[e].ty;
    for _ in 0..n {
        e = data.expressions.alloc(ExprData {
            ty,
            kind: ExprKind::If {
                condition,
                then_value: e,
                else_value: e,
            },
        });
    }
    data.regions[r].results = vec![e];
    data
}
fn effects(n: usize, many_regions: bool) -> AssociatedData {
    let mut data = maps(1);
    let r = root(&data);
    let definition = data.regions[r].definition;
    let template = data.operations[*data.regions[r].members.first().unwrap()].clone();
    let ext = data.externs.alloc(ExternData {
        linkage_name: "effect".into(),
    });
    let function = data.expressions.alloc(ExprData {
        ty: template.ty,
        kind: ExprKind::Extern(ext),
    });
    data.regions[r].members.clear();
    data.regions[r].results.clear();
    for i in 0..n {
        let region = if many_regions {
            let region = data.regions.alloc(RegionData {
                parent: None,
                parameters: vec![],
                members: BTreeSet::new(),
                results: vec![],
                definition,
            });
            let mut def = data.definitions[definition].clone();
            def.body = region;
            data.definitions.alloc(def);
            region
        } else {
            r
        };
        let op = data.operations.alloc(OperationData {
            region,
            source_position: i,
            kind: OperationKind::Call {
                function,
                args: vec![],
            },
            ..template.clone()
        });
        data.regions[region].members.insert(op);
    }
    data
}

#[test]
fn nested_common_if_work_is_placed_once_above_the_chain() {
    let mut data = conditionals(256);
    let outer = data.regions[root(&data)].results[0];
    hoist::run(&mut data).unwrap();
    assert!(!data.placements.is_empty());
    assert!(data.placements.values().all(|p| p.before == PlacementSite::Expression(outer)));
}

#[test]
fn shared_callbacks_reuse_one_specialized_body() {
    let mut data = maps(64);
    let before = data.regions.len();
    hoist::run(&mut data).unwrap();
    assert_eq!(data.regions.len() - before, 1);
    let sites: BTreeSet<_> = data.placements.values().map(|p| p.before).collect();
    assert_eq!(sites.len(), 64);
}

#[test]
#[ignore = "manual release scaling measurement"]
fn placement_scaling() {
    for n in [512, 1024, 2048] {
        for shape in ["effects", "regions", "conditionals", "captures"] {
            let data = match shape {
                "effects" => effects(n, false),
                "regions" => effects(n, true),
                "conditionals" => conditionals(n),
                _ => maps(n),
            };
            let mut times = vec![];
            for _ in 0..3 {
                let mut candidate = data.clone();
                let start = std::time::Instant::now();
                if shape == "effects" || shape == "regions" {
                    let s = snapshot::analyze(&candidate);
                    s.schedules(&candidate).unwrap();
                } else {
                    hoist::run(&mut candidate).unwrap();
                }
                times.push(start.elapsed());
            }
            times.sort();
            eprintln!("{shape} n={n}: {:.3} ms", times[1].as_secs_f64() * 1000.0);
        }
    }
}

#[test]
#[ignore = "manual release placement profile"]
fn placement_profile() {
    let mut data = maps(2048);
    let before = (data.regions.len(), data.expressions.len());
    timing::with_timings(true, || hoist::run(&mut data).unwrap());
    eprintln!(
        "regions {} -> {}; expressions {} -> {}; placements {}",
        before.0,
        data.regions.len(),
        before.1,
        data.expressions.len(),
        data.placements.len()
    );
}
