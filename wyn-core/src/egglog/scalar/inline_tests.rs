use crate::op::OpTag;
use crate::ssa::types::InstKind;

#[test]
fn rotate_helper_is_inlined_and_trig_is_placed_before_the_march_loop() {
    let source = r#"
def rotate(p: vec2f32, angle: f32) vec2f32 =
  let c = f32.cos(angle)
  let s = f32.sin(angle) in
  @[c * p.x - s * p.y, s * p.x + c * p.y]
def march(p: vec2f32, angle: f32) vec2f32 =
  loop total = @[0.0, 0.0] for k < 20 do
    total + rotate(p + @[f32(k) * 0.06, 0.0], angle)
entry repro(points: []vec2f32, angle: f32) []vec2f32 =
  map(|p| march(p, angle), points)
"#;
    let program = crate::compile_thru_ssa(source).unwrap();
    for (name, expected) in [("f32.sin", 1), ("f32.cos", 1)] {
        let id = crate::builtins::catalog().lookup_by_any_name(name).unwrap().id;
        let sites = program.functions.iter().map(|f| &f.body)
            .chain(program.entry_points.iter().map(|e| &e.body))
            .flat_map(|body| body.inner.insts.values().filter_map(move |node| {
                matches!(node.data, InstKind::Op { tag: OpTag::Intrinsic { id: actual, .. }, .. } if actual == id)
                    .then_some((body, node))
            })).collect::<Vec<_>>();
        assert_eq!(
            sites.len(),
            expected,
            "{name}: sharing must precede SSA optimization"
        );
        for (body, node) in sites {
            assert_eq!(
                node.placement.block(),
                Some(body.inner.entry),
                "{name}: invariant before the loop"
            );
        }
    }
    let text = crate::lower_ssa_to_wgsl(program).unwrap();
    let module = naga::front::wgsl::parse_str(&text).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn camera_helper_above_old_inline_limit_is_placed_before_the_march_loop() {
    let source = r#"
def camera_term(angle: f32) f32 =
  let a = f32.sin(f32.sin(f32.sin(f32.sin(angle))))
  let b = f32.sin(f32.sin(f32.sin(f32.sin(a))))
  let c = f32.sin(f32.sin(f32.sin(f32.sin(b))))
  let d = f32.sin(f32.sin(f32.sin(f32.sin(c)))) in
  f32.cos(d)
def march(angle: f32) f32 =
  loop total = 0.0 for k < 20 do
    total + camera_term(angle) * f32(k)
entry repro(angles: []f32) []f32 = map(march, angles)
"#;
    let program = crate::compile_thru_ssa(source).unwrap();
    for (name, expected) in [("f32.sin", 16), ("f32.cos", 1)] {
        let id = crate::builtins::catalog().lookup_by_any_name(name).unwrap().id;
        let sites = program.functions.iter().map(|f| &f.body)
            .chain(program.entry_points.iter().map(|e| &e.body))
            .flat_map(|body| body.inner.insts.values().filter_map(move |node| {
                matches!(node.data, InstKind::Op { tag: OpTag::Intrinsic { id: actual, .. }, .. } if actual == id)
                    .then_some((body, node))
            })).collect::<Vec<_>>();
        assert_eq!(
            sites.len(),
            expected,
            "{name}: sharing must precede SSA optimization"
        );
        for (body, node) in sites {
            assert_eq!(
                node.placement.block(),
                Some(body.inner.entry),
                "{name}: invariant before the loop"
            );
        }
    }
    let text = crate::lower_ssa_to_wgsl(program).unwrap();
    let module = naga::front::wgsl::parse_str(&text).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}
