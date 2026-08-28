export const passDefinitions = {
  "egir::reify_soacs": {
    before: "Converted raw EGIR",
    after: "Segmented semantic EGIR",
    example: `entry increment(xs: [4]i32) [4]i32 =
  map(|x: i32| x + 1, xs)`,
  },
  "egir::eliminate_dead_semantic_operations": {
    before: "Segmented semantic EGIR",
    after: "Dead semantic operation removed",
    example: `entry discard_map(xs: [4]i32) [4]i32 =
  let dead = map(|x: i32| x + 99, xs) in
  xs`,
  },
  "egir::fuse_semantic_operations": {
    before: "Dead semantic operations eliminated",
    after: "One legal semantic fusion applied",
    example: `entry fuse_maps(xs: [4]i32) [4]i32 =
  let shifted = map(|x: i32| x + 1, xs) in
  map(|x: i32| x * 2, shifted)`,
  },
  "egir::lift_stage_uniform_values": {
    before: "Semantic optimization fixpoint",
    after: "Stage-uniform values lifted",
    example: `def mixed_curve(lane: f32, uniform: f32) f32 =
  let a = uniform * uniform + 1.0
  let b = a * uniform + 2.0
  let c = b * uniform + 3.0
  let d = c * uniform + 4.0 in
  lane + d
entry lift_uniform(points: [64]f32, phase: f32) [64]f32 =
  map(|point: f32| mixed_curve(point, phase), points)`,
  },
  "egir::allocate_semantic_resources": {
    before: "Stage-uniform semantic EGIR",
    after: "Logical resources allocated",
    example: `entry allocate_filter(xs: []i32) []i32 =
  filter(|x: i32| x % 2 == 0, xs)`,
  },
  "egir::resolve_residency": {
    before: "Logical resources allocated",
    after: "Cross-stage residency resolved",
    example: `entry resident_filter(xs: []i32) []i32 =
  let selected = filter(|x: i32| x > 0, xs) in
  map(|x: i32| x + 1, selected)`,
  },
  "egir::resolve_scratch_sizes": {
    before: "Cross-stage residency resolved",
    after: "Scratch sizes resolved",
    example: `entry sized_filter(xs: []i32) []i32 =
  filter(|x: i32| x % 3 == 0, xs)`,
  },
  "egir::finalize_staged_ir": {
    before: "Residency draft",
    after: "Finalized staged IR",
    example: `entry staged_filter(xs: []i32) []i32 =
  let selected = filter(|x: i32| x != 0, xs) in
  map(|x: i32| x * x, selected)`,
  },
  "egir::verify_allocated_resources": {
    before: "Finalized staged IR",
    after: "Allocated resources verified",
    example: `entry verified_sum(xs: []i32) i32 =
  reduce(|a: i32, b: i32| a + b, 0, xs)`,
  },
  "egir::bind_mapped_output_destinations": {
    before: "Verified staged IR",
    after: "Mapped output destinations bound",
    example: `entry mapped_output(xs: [4]i32) [4]i32 =
  map(|x: i32| x + 1, xs)`,
  },
  "egir::analyze_kernel_recipes": {
    before: "Mapped output destinations bound",
    after: "Kernel recipes analyzed",
    example: `entry analyzed_sum(xs: []i32) i32 =
  reduce(|a: i32, b: i32| a + b, 0, xs)`,
  },
  "egir::allocate_recipe_scratch": {
    before: "Kernel recipes analyzed",
    after: "Recipe scratch allocated",
    example: `entry scratch_sum(xs: []i32) i32 =
  reduce(|a: i32, b: i32| a + b, 0, xs)`,
  },
  "egir::build_kernel_schedule": {
    before: "Recipe scratch allocated",
    after: "Kernel schedule built",
    example: `entry scheduled_scan(xs: []i32) []i32 =
  scan(|a: i32, b: i32| a + b, 0, xs)`,
  },
  "egir::finalize_kernel_schedule": {
    before: "Kernel schedule built",
    after: "Planned physical EGIR",
    example: `entry finalized_sum(xs: []i32) i32 =
  reduce(|a: i32, b: i32| a + b, 0, xs)`,
  },
  "egir::expand_soacs": {
    before: "Planned physical EGIR",
    after: "SOACs expanded",
    example: `entry scan_offsets(xs: []i32) []i32 =
  scan(|a: i32, b: i32| a + b, 0, xs)`,
  },
  "egir::eliminate_internal_place_calls": {
    before: "SOACs expanded",
    after: "Internal place calls eliminated",
    example: `def choose_sum(values: [4]i32, flag: u32) i32 =
  let left = values[0] + values[1] in
  let right = values[2] + values[3] in
  if flag == 0u32 then left else right

entry call_place(values: [4]i32, flag: u32) i32 =
  choose_sum(values, flag)`,
  },
  "egir::partially_inline_calls": {
    before: "Internal calls place-free",
    after: "Profitable calls partially inlined",
    example: `def choose_and_scale(varying: u32, invariant: u32) u32 =
  let scale = invariant * invariant in
  if varying == 0u32 then scale else varying + scale

entry mixed_loop(seed: u32, scale: u32) u32 =
  loop value = seed for i < 4 do
    let stable = choose_and_scale(0u32, scale) in
    choose_and_scale(value + u32.i32(i), stable)`,
  },
  "egir::materialize_dynamic_extracts": {
    before: "Calls partially inlined",
    after: "Dynamic extracts materialized",
    example: `entry dynamic_local(index: i32) i32 =
  let values = [10, 20, 30, 40] in
  values[index]`,
  },
  "egir::rewrite": {
    before: "Dynamic extracts materialized",
    after: "Value rewrites applied",
    example: `entry power_chain(x: f32) f32 =
  x ** 5.0f32`,
  },
  "egir::optimize_skeleton": {
    before: "Values rewritten",
    after: "CFG skeleton optimized",
    example: `def choose_and_scale(varying: u32, invariant: u32) u32 =
  let scale = invariant * invariant in
  if varying == 0u32 then scale else varying + scale

entry mixed_loop(seed: u32, scale: u32) u32 =
  loop value = seed for i < 4 do
    let stable = choose_and_scale(0u32, scale) in
    choose_and_scale(value + u32.i32(i), stable)`,
  },
  "egir::erase_resources": {
    before: "CFG skeleton optimized",
    after: "Compile-time resources erased",
    example: `def vertex_main(vertex: vertex_invocation) vertex<vec2f32> =
  let vid = i32(vertex.vertex_index) in
  let verts = [@[-1.0, -1.0, 0.0, 1.0],
               @[3.0, -1.0, 0.0, 1.0],
               @[-1.0, 3.0, 0.0, 1.0]] in
  vertex_output(verts[vid], @[0.0, 0.0])

entry resource_handles(screen: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(direct_draw(3u32, 1u32), vertex_main) in
  shade(screen, covered,
    |fragment| @[fragment.position.x, fragment.position.y, 0.0, 1.0])`,
  },
} as const;

export type PassId = keyof typeof passDefinitions;

export const passIds = Object.keys(passDefinitions) as PassId[];
export const defaultPassId: PassId = "egir::reify_soacs";

export function isPassId(value: string | null): value is PassId {
  return value !== null && Object.hasOwn(passDefinitions, value);
}
