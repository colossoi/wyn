use crate::compile_thru_spirv;
use crate::err_spirv;
use crate::error::Result;

/// Compile a source string to SPIR-V words. Thin wrapper around the
/// canonical `compile_thru_spirv` so test failures lift through `.unwrap()`.
fn compile_to_spirv(source: &str) -> Result<Vec<u32>> {
    compile_thru_spirv(source).map(|l| l.spirv).map_err(|e| err_spirv!("{}", e))
}

#[test]
fn repeated_scan_builds_emit_identical_phi_order() {
    let source = include_str!("../../../testfiles/scan_compute.wyn");
    let expected = compile_to_spirv(source).unwrap();
    let module = wspirv::dr::load_words(&expected).unwrap();
    assert!(module.functions.iter().flat_map(|f| &f.blocks).any(|block| block
        .instructions
        .iter()
        .filter(|i| i.class.opcode == spirv::Op::Phi)
        .count()
        >= 2));
    for _ in 0..12 {
        assert_eq!(compile_to_spirv(source).unwrap(), expected);
    }
}

#[test]
fn integer_power_helpers_are_emitted_only_for_used_signedness() {
    let signed = "entry signed(x:i32, y:i32) i32 = x ** y";
    let unsigned = "entry unsigned(x:u32, y:u32) u32 = x ** y";
    for (source, helpers) in [
        ("entry plain(x:i32) i32 = x + 1".to_string(), 0),
        (signed.to_string(), 1),
        (unsigned.to_string(), 1),
        (format!("{signed}\n{unsigned}"), 2),
        (
            format!("{signed}\n{}", signed.replace("signed(", "also_signed(")),
            1,
        ),
        ("entry folded(x:i32) i32 = x + 2 ** 3".to_string(), 0),
    ] {
        let words = compile_to_spirv(&source).unwrap();
        let module = wspirv::dr::load_words(&words).unwrap();
        assert_eq!(
            module.functions.len(),
            module.entry_points.len() + helpers,
            "{source}"
        );
        let bytes = words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
        let module = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
fn literal_composite_indices_do_not_emit_unused_constant_ids() {
    let positive = ["x"; 48].join(", ");
    let negative = ["-x"; 48].join(", ");
    let source = format!(
        "def values(x:f32) [48]f32 = if x > 0.0 then [{positive}] else [{negative}]\n\
         entry at(x:f32) f32 = values(x)[38]"
    );
    let module = wspirv::dr::load_words(compile_to_spirv(&source).unwrap()).unwrap();
    assert!(
        module.functions.iter().flat_map(|f| &f.blocks).flat_map(|b| &b.instructions).any(|i| i
            .class
            .opcode
            == spirv::Op::CompositeExtract
            && i.operands.last() == Some(&wspirv::dr::Operand::LiteralBit32(38)))
    );
    assert!(!module
        .types_global_values
        .iter()
        .any(|i| i.class.opcode == spirv::Op::Constant
            && i.operands == [wspirv::dr::Operand::LiteralBit32(38)]));
}

#[test]
fn storage_length_descriptors_do_not_emit_unused_constant_ids() {
    let params = (0..39).map(|i| format!("a{i}: []f32")).collect::<Vec<_>>().join(", ");
    let source = format!("entry last_buffer({params}) []f32 = map(|x| x + 1.0, a38)");
    let module = wspirv::dr::load_words(compile_to_spirv(&source).unwrap()).unwrap();
    let buffers: Vec<_> = module
        .annotations
        .iter()
        .filter_map(|inst| match inst.operands.as_slice() {
            [wspirv::dr::Operand::IdRef(id), wspirv::dr::Operand::Decoration(spirv::Decoration::Binding), wspirv::dr::Operand::LiteralBit32(38)] => Some(*id),
            _ => None,
        })
        .collect();
    assert!(!buffers.is_empty(), "input keeps its declared binding");
    assert!(
        module.functions.iter().flat_map(|f| &f.blocks).flat_map(|b| &b.instructions).any(|i| i
            .class
            .opcode
            == spirv::Op::ArrayLength
            && matches!(i.operands.first(), Some(wspirv::dr::Operand::IdRef(id)) if buffers.contains(id)))
    );
    assert!(!module
        .types_global_values
        .iter()
        .any(|i| i.class.opcode == spirv::Op::Constant
            && i.operands == [wspirv::dr::Operand::LiteralBit32(38)]));
}

#[test]
fn unrelated_entries_preserve_storage_buffer_element_types() {
    let source = include_str!("../../../testfiles/graphics_compute_entry_buffer_types.wyn");
    for source in [
        source.to_string(),
        source.replace("weights[0]", "weights[length(weights) - 1]"),
        source
            .replace("weights: []f32", "weights: []vec4f32")
            .replace("weights[0]", "weights[0].x")
            .replace("values: []vec4f32) []vec4f32", "values: []f32) []f32"),
    ] {
        let (draw, compute) = source.split_once("entry compute").unwrap();
        let compute = format!("entry compute{compute}");
        for source in [draw.to_string(), source.to_string(), format!("{compute}\n{draw}")] {
            let words = compile_to_spirv(&source).unwrap();
            let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            let module = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
    }
}

#[test]
fn test_simple_constant() {
    let spirv = compile_to_spirv("def x = 42").unwrap();
    assert!(!spirv.is_empty());
    // SPIR-V magic number
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_simple_function() {
    let spirv = compile_to_spirv("def add(x, y) = x + y").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn fragment_helper_result_is_evaluated_once_for_multiple_projections() {
    use wspirv::spirv::Op;

    let module = wspirv::dr::load_words(
        compile_to_spirv(
            r#"
def color(x: f32) vec2f32 =
  if x < 10.0 then @[1.0, 2.0] else @[3.0, 4.0]
type output = { color: vec2f32, depth: f32 }
def fragment(_v: (), p: vec4f32, _f: bool, _i: u32, _s: u32) output =
  let c = color(p.x) in { color = @[c.x, c.y], depth = p.z }
entry repro(surface: render_target<output>) render_target<output> =
  let triangle = rasterize_triangles(direct_draw(3u32, 1u32),
    |v, _, _| vertex_output(
      @[if v == 1u32 then 3.0 else -1.0,
        if v == 2u32 then 3.0 else -1.0, 0.0, 1.0], ())) in
  shade(surface, triangle, fragment)
"#,
        )
        .unwrap(),
    )
    .unwrap();
    let calls = module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| inst.class.opcode == Op::FunctionCall)
        .count();
    assert_eq!(calls, 1, "the fragment must evaluate its helper only once");
}

#[test]
fn vertex_conditional_is_evaluated_once_for_nested_varyings() {
    use wspirv::dr::Operand;
    use wspirv::spirv::{ExecutionModel, Op};

    let module = wspirv::dr::load_words(
        compile_to_spirv(
            r#"
def color(x: f32) vec2f32 =
  if x < 10.0 then @[1.0, 2.0] else @[3.0, 4.0]
def payload(v: u32) (f32, (f32, f32)) =
  let c = color(f32(v)) in (c.x, (c.y, f32(v)))
entry repro(surface: render_target<f32>) render_target<f32> =
  let triangle = rasterize_triangles(direct_draw(3u32, 1u32),
    |v, _, _| vertex_output(@[0.0, 0.0, 0.0, 1.0], payload(v))) in
  shade(surface, triangle, |p, _, _, _, _|
    let (x, (y, z)) = p in x + y + z)
"#,
        )
        .unwrap(),
    )
    .unwrap();
    let entry = module
        .entry_points
        .iter()
        .find(|inst| inst.operands.first() == Some(&Operand::ExecutionModel(ExecutionModel::Vertex)))
        .unwrap();
    let Operand::IdRef(vertex_id) = entry.operands[1] else {
        panic!("vertex function")
    };
    let vertex = module
        .functions
        .iter()
        .find(|function| function.def.as_ref().unwrap().result_id == Some(vertex_id))
        .unwrap();
    let selections = vertex
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .filter(|inst| matches!(inst.class.opcode, Op::Select | Op::SelectionMerge))
        .count();
    assert_eq!(
        selections, 1,
        "varying projections must share the conditional result"
    );
}

#[test]
fn fixed_array_output_constructs_only_the_indexed_value() {
    use std::collections::HashSet;
    use wspirv::spirv::Op;

    let module = wspirv::dr::load_words(
        compile_to_spirv(
            "entry repro(xs: []i32) ([]i32, [1]i32) =
              let n = xs[0] in (map(|x| x + n, xs), [n])",
        )
        .unwrap(),
    )
    .unwrap();
    let arrays: HashSet<_> = module
        .types_global_values
        .iter()
        .filter(|inst| inst.class.opcode == Op::TypeArray)
        .filter_map(|inst| inst.result_id)
        .collect();
    let constructors = module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| {
            inst.class.opcode == Op::CompositeConstruct
                && inst.result_type.is_some_and(|ty| arrays.contains(&ty))
        })
        .count();
    assert_eq!(constructors, 1, "static length queries must not construct arrays");
}

#[test]
fn test_linked_extern_call_uses_structural_function_identity() {
    let spirv = compile_to_spirv(
        r#"
#[linked("external_increment")]
extern external_increment(x: i32) i32

entry main(x: i32) i32 = external_increment(x)
"#,
    )
    .expect("linked extern call should lower to an import-linked SPIR-V function");
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn distinct_entry_push_constant_layouts_have_distinct_variables() {
    use std::collections::HashSet;
    use wspirv::dr::Operand;
    use wspirv::spirv::{Op, StorageClass};
    let module = wspirv::dr::load_words(
        compile_to_spirv("entry one(x:i32) i32 = x + 1\nentry two(xs:[2]i32,i:i32) i32 = xs[i]").unwrap(),
    )
    .unwrap();
    let variables: HashSet<_> = module
        .types_global_values
        .iter()
        .filter_map(|i| {
            (i.class.opcode == Op::Variable
                && i.operands.first() == Some(&Operand::StorageClass(StorageClass::PushConstant)))
            .then_some(i.result_id)
            .flatten()
        })
        .collect();
    assert!(variables.len() >= 2);
    for entry in &module.entry_points {
        assert_eq!(
            entry
                .operands
                .iter()
                .filter(|operand| matches!(operand, Operand::IdRef(id) if variables.contains(id)))
                .count(),
            1
        );
    }
}

#[test]
fn render_target_load_fetches_only_fields_used_by_the_fragment_output() {
    use crate::host::{Binding, Pipeline};
    use std::collections::{HashMap, HashSet};
    use wspirv::dr::Operand;
    use wspirv::spirv::{Decoration, ExecutionModel, Op};

    for (scene_ty, result, expected_names) in [
        ("pair", "p.depth", vec!["scene_depth"]),
        ("pair", "p.unused", vec!["scene_unused"]),
        ("pair", "p.unused + p.depth", vec!["scene_unused", "scene_depth"]),
        ("f32", "p", vec!["scene"]),
    ] {
        let source = format!(
            r#"
type pair = {{ unused: f32, depth: f32 }}
entry repro(scene: render_target<{scene_ty}>, surface: render_target<f32>)
    render_target<f32> =
  let triangle = rasterize_triangles(direct_draw(3u32, 1u32),
    |v, _, _| vertex_output(
      @[if v == 1u32 then 3.0 else -1.0,
        if v == 2u32 then 3.0 else -1.0, 0.0, 1.0], ())) in
  shade(surface, triangle, |_, position, _, _, _|
    let p = target_load(scene, @[i32(position.x), i32(position.y)], 0u32) in {result})
"#
        );
        let lowered = compile_thru_spirv(&source).unwrap();
        let bytes: Vec<_> = lowered.spirv.iter().flat_map(|word| word.to_le_bytes()).collect();
        let validated = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&validated)
        .unwrap();

        let module = wspirv::dr::load_words(&lowered.spirv).unwrap();
        let entry = module
            .entry_points
            .iter()
            .find(|inst| inst.operands.first() == Some(&Operand::ExecutionModel(ExecutionModel::Fragment)))
            .unwrap();
        let Operand::IdRef(fragment_id) = entry.operands[1] else {
            panic!("fragment function")
        };
        let fragment = module
            .functions
            .iter()
            .find(|f| f.def.as_ref().unwrap().result_id == Some(fragment_id))
            .unwrap();
        let instructions: Vec<_> = fragment.blocks.iter().flat_map(|block| &block.instructions).collect();
        let definitions: HashMap<_, _> =
            instructions.iter().filter_map(|inst| inst.result_id.map(|id| (id, *inst))).collect();
        let fetches: Vec<_> =
            instructions.iter().filter(|inst| inst.class.opcode == Op::ImageFetch).collect();
        assert_eq!(fetches.len(), expected_names.len(), "{scene_ty}: {result}");

        // Trace each fetch through its image load and decorations to the named
        // attachment in the descriptor; neither result IDs nor binding slots
        // are fixed by the test.
        let decoration = |id, kind| {
            module
                .annotations
                .iter()
                .find_map(|inst| match inst.operands.as_slice() {
                    [Operand::IdRef(target), Operand::Decoration(actual), Operand::LiteralBit32(value)]
                        if inst.class.opcode == Op::Decorate && *target == id && *actual == kind =>
                    {
                        Some(*value)
                    }
                    _ => None,
                })
                .unwrap()
        };
        let mut fetched_names = Vec::new();
        for fetch in &fetches {
            let fetch_id = fetch.result_id.unwrap();
            assert!(
                instructions.iter().any(|inst| inst.operands.contains(&Operand::IdRef(fetch_id))),
                "unused fetch %{fetch_id}"
            );
            let Operand::IdRef(image) = fetch.operands[0] else {
                panic!("fetch image")
            };
            let load = definitions[&image];
            assert_eq!(load.class.opcode, Op::Load);
            let Operand::IdRef(variable) = load.operands[0] else {
                panic!("image variable")
            };
            let set = decoration(variable, Decoration::DescriptorSet);
            let binding = decoration(variable, Decoration::Binding);
            let name = lowered
                .program
                .interface
                .pipelines
                .iter()
                .filter_map(|p| match p {
                    Pipeline::Graphics(p) => Some(&p.bindings),
                    _ => None,
                })
                .flatten()
                .find_map(|b| match b {
                    Binding::Texture {
                        set: s,
                        binding: b,
                        name,
                        ..
                    } if *s == set && *b == binding => Some(name.as_str()),
                    _ => None,
                })
                .unwrap();
            fetched_names.push(name);
        }
        fetched_names.sort_unstable();
        let mut expected_names = expected_names;
        expected_names.sort_unstable();
        assert_eq!(fetched_names, expected_names, "{scene_ty}: {result}");

        // Every retained fetch must contribute to the fragment's output store.
        let stores: Vec<_> = instructions.iter().filter(|inst| inst.class.opcode == Op::Store).collect();
        assert_eq!(stores.len(), 1);
        let Operand::IdRef(output) = stores[0].operands[1] else {
            panic!("output value")
        };
        let mut pending = vec![output];
        let mut live = HashSet::new();
        while let Some(id) = pending.pop() {
            if live.insert(id) {
                if let Some(inst) = definitions.get(&id) {
                    pending.extend(inst.operands.iter().filter_map(|operand| match operand {
                        Operand::IdRef(id) => Some(*id),
                        _ => None,
                    }));
                }
            }
        }
        assert!(fetches.iter().all(|inst| live.contains(&inst.result_id.unwrap())));
    }
}

#[test]
fn test_let_binding() {
    let spirv = compile_to_spirv("def f = let x = 1 in x + 2").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_arithmetic() {
    let spirv = compile_to_spirv("def f(x, y) = x * y + x / y - 1").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_nested_let() {
    let spirv = compile_to_spirv("def f = let a = 1 in let b = 2 in a + b").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_if_expression() {
    let spirv = compile_to_spirv("def f(x) = if x == 0 then 1 else 2").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn folded_branches_do_not_emit_selection_merges_before_unconditional_branches() {
    for source in [
        include_str!("../../../testfiles/sum_payload_array.wyn"),
        include_str!("../../../testfiles/sum_payload_array_construct.wyn"),
    ] {
        let words = compile_to_spirv(source).unwrap();
        let module = wspirv::dr::load_words(words).unwrap();
        for block in module.functions.iter().flat_map(|f| &f.blocks) {
            for instructions in block.instructions.windows(2) {
                if instructions[0].class.opcode == wspirv::spirv::Op::SelectionMerge {
                    assert!(matches!(
                        instructions[1].class.opcode,
                        wspirv::spirv::Op::BranchConditional | wspirv::spirv::Op::Switch
                    ));
                }
            }
        }
    }
}

#[test]
fn test_if_literal_true_lowers() {
    // `fold_constant_branches` leaves the dead arm in `skeleton.blocks`
    // without a predecessor. Dominator analysis must ignore that arm when
    // deciding whether to emit the live path's merge block.
    let spirv = compile_to_spirv("def f(x) = if true then x + 1 else x + 2").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_if_literal_false_lowers() {
    let spirv = compile_to_spirv("def f(x) = if false then x + 1 else x + 2").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_comparisons() {
    let spirv = compile_to_spirv("def f(x, y) = if x < y then 1 else if x > y then 2 else 0").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_tuple_literal() {
    let spirv = compile_to_spirv("def f = (1, 2, 3)").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_array_literal() {
    let spirv = compile_to_spirv("def f = [1, 2, 3]").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_unary_negation() {
    let spirv = compile_to_spirv("def f(x) = -x").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_record_field_access() {
    let spirv = compile_to_spirv(
        r#"
def get_x(r:{x:i32, y:i32}) i32 = r.x
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_closure_capture_access() {
    // This test uses tuple_access intrinsic for closure field access
    let spirv = compile_to_spirv(
        r#"
def test(x:i32) i32 =
    let f = |y:i32| x + y in
    f(10)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_polymorphic_dot2() {
    // Test polymorphic function with type parameters that need proper instantiation
    // This reproduces the primitives.wyn issue where Vec type has unresolved size variable
    let spirv = compile_to_spirv(
        r#"
def dot2<E, T>(v: T) E = dot(v, v)

def test_dot2_vec3(v: vec3f32) f32 = dot2(v)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_polymorphic_dot2_in_expression() {
    // Test dot2 used in a more complex expression like in primitives.wyn
    // sdCappedTorus: f32.sqrt(dot2(p) + ra*ra - 2.0*ra*k) - rb
    let spirv = compile_to_spirv(
        r#"
def dot2<E, T>(v: T) E = dot(v, v)

def sdCappedTorus(p: vec3f32, ra: f32, rb: f32, k: f32) f32 =
  f32.sqrt(dot2(p) + ra*ra - 2.0*ra*k) - rb
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_polymorphic_dot2_vec2_and_vec3() {
    // Test dot2 with both vec2 and vec3 in same program (like primitives.wyn)
    let spirv = compile_to_spirv(
        r#"
def dot2<E, T>(v: T) E = dot(v, v)

def test_vec3(v: vec3f32) f32 = dot2(v)
def test_vec2(v: vec2f32) f32 = dot2(v)
def test_both(v3: vec3f32, v2: vec2f32) f32 = dot2(v3) + dot2(v2)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_scan_inclusive() {
    // Inclusive scan (prefix sum): scan (+) 0 [1,2,3] = [1, 3, 6]
    let spirv = compile_to_spirv(
        r#"
def sum_scan(arr: [4]i32) [4]i32 = scan((|a, b| a + b), 0, arr)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_map_variants() {
    // Test map with zip variants: map over zipped inputs
    let spirv = compile_to_spirv(
        r#"
def double(x: i32) i32 = x * 2

def test_map(arr: [3]i32) [3]i32 = map(double, arr)
def test_map2(xs: [3]i32, ys: [3]i32) [3]i32 = map(|(x, y)| x + y, zip(xs, ys))
def test_map3(xs: [3]i32, ys: [3]i32, zs: [3]i32) [3]i32 = map(|(x, y, z)| x + y + z, zip3(xs, ys, zs))
def test_map4(a: [3]i32, b: [3]i32, c: [3]i32, d: [3]i32) [3]i32 = map(|(a, b, c, d)| a + b + c + d, zip4(a, b, c, d))
def test_map5(a: [3]i32, b: [3]i32, c: [3]i32, d: [3]i32, e: [3]i32) [3]i32 = map(|(a, b, c, d, e)| a + b + c + d + e, zip5(a, b, c, d, e))
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_scatter_update() {
    // Scatter: write values to array at given indices
    let spirv = compile_to_spirv(
        r#"
def scatter_test(dest: [5]i32, indices: [2]i32, values: [2]i32) [5]i32 =
    scatter(dest, indices, values)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_reduce_by_index() {
    // Histogram / reduce_by_index: accumulate values at indices using operator
    let spirv = compile_to_spirv(
        r#"
def hist_test(dest: [3]i32, indices: [4]i32, values: [4]i32) [3]i32 =
    reduce_by_index(dest, |a, b| a + b, 0, indices, values)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_hist() {
    // hist is an alias for reduce_by_index
    let spirv = compile_to_spirv(
        r#"
def hist_alias_test(dest: [3]i32, indices: [4]i32, values: [4]i32) [3]i32 =
    hist(dest, |a, b| a + b, 0, indices, values)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_reduce_u32() {
    // Test reduce with u32 types - the initial value 0u32 must generate
    // an unsigned constant, not a signed one
    let spirv = compile_to_spirv(
        r#"
def sum_u32(arr: [4]u32) u32 =
    reduce(|a, b| a + b, 0u32, arr)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_reduce_with_tuple_destructuring() {
    // HOF specialization must eliminate the tuple-destructuring combiner's
    // function parameters before SPIR-V lowering.
    let result = compile_to_spirv(
        r#"
def minPair(hits: [4](f32, i32)) (f32, i32) =
  reduce(|(t1, m1): (f32, i32), (t2, m2): (f32, i32)|
           if t1 < t2 then (t1, m1) else (t2, m2),
         (1000.0, 0),
         hits)

def testHits: [4](f32, i32) = [(1.0, 1), (2.0, 2), (0.5, 3), (3.0, 4)]


entry fragment_main(pos: vec4f32) vec4f32 =
  let (t, m) = minPair(testHits) in
  @[t, t, 0.0, 1.0]
"#,
    );
    // Should compile without "Undefined global" error
    assert!(
        result.is_ok(),
        "Expected successful compilation, got: {:?}",
        result.err()
    );
    let spirv = result.unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_algebraic_simplifications() {
    // Test all algebraic identity simplifications compile correctly
    let spirv = compile_to_spirv(
        r#"
def test(x: f32) f32 =
    (0.0 - x) +     -- 0 - x → -x
    (0.0 + x) +     -- 0 + x → x
    (x + 0.0) +     -- x + 0 → x
    (x - 0.0) +     -- x - 0 → x
    (0.0 * x) +     -- 0 * x → 0
    (x * 0.0) +     -- x * 0 → 0
    (1.0 * x) +     -- 1 * x → x
    (x * 1.0) +     -- x * 1 → x
    (x / 1.0) +     -- x / 1 → x
    (-1.0 * x) +    -- -1 * x → -x
    (x * -1.0)      -- x * -1 → -x
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_partial_eval_inlined_function_local_id_collision() {
    // Partial evaluation of an inlined helper with an unknown let RHS must
    // allocate locals independently of the caller's local map.
    let spirv = compile_to_spirv(
        r#"
-- Helper function that will be inlined when called with known args.
-- The let binding 'weight' uses iTime which is unknown, so it gets residualized.
def helper(x: f32, iTime: f32) f32 =
    let weight = x * iTime in
    weight


entry fragment_main(iTime: f32, pos: vec4f32) vec4f32 =
    -- Create multiple locals to ensure LocalId collision
    let a = pos.x in
    let b = pos.y in
    let c = pos.z in
    -- Call helper with known arg - this gets inlined.
    -- helper's 'weight' local may collide with our locals.
    let d = helper(3.0, iTime) in
    @[d, a, b, c]
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_partial_eval_intrinsic_arg_types() {
    // Residualizing dot(vec3, vec3) -> f32 must reify each known argument with
    // its vec3 type rather than the scalar result type.
    let spirv = compile_to_spirv(
        r#"
def verts: [3]vec4f32 =
  [@[-1.0, -1.0, 0.0, 1.0],
   @[3.0, -1.0, 0.0, 1.0],
   @[-1.0, 3.0, 0.0, 1.0]]


entry vertex_main(vertex_id: i32) vec4f32 = verts[vertex_id]


entry fragment_main() vec4f32 =
  let g = @[1.0, 2.0, 3.0] in
  let h = dot(g, @[127.1, 311.7, 74.7]) in
  @[h, h, h, 1.0]
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_nested_if_else_in_entry_point() {
    // Entry-point return emission must preserve the multi-block structure of
    // nested conditionals.
    let result = compile_to_spirv(
        r#"

entry fragment_main(fragCoord: vec4f32) vec4f32 =
  let x = fragCoord.x in
  if x < 0.5 then @[1.0, 0.0, 0.0, 1.0]
  else if x < 1.5 then @[0.0, 1.0, 0.0, 1.0]
  else @[0.0, 0.0, 1.0, 1.0]
"#,
    );
    assert!(
        result.is_ok(),
        "Nested if-else in entry point should compile. Got: {:?}",
        result.err()
    );
    let spirv = result.unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_two_compute_entries_share_one_global_invocation_id() {
    // VUID-StandaloneSpirv-OpEntryPoint-09658 permits exactly one module-level
    // GlobalInvocationId input variable. Count its BuiltIn decoration to
    // ensure multiple compute entries share that variable.
    let src = "\
def verts: [3]vec4f32 =
  [@[0.0 - 1.0, 0.0 - 1.0, 0.0, 1.0],
   @[3.0, 0.0 - 1.0, 0.0, 1.0],
   @[0.0 - 1.0, 3.0, 0.0, 1.0]]


entry vertex_main(vid: i32)
  vec4f32 = verts[vid]

entry a(gid: vec3u32) () = ()

entry b(gid: vec3u32) () = ()


entry fragment_main(_p: vec4f32)
  vec4f32 = @[0.0, 0.0, 0.0, 1.0]
";
    let spirv = compile_to_spirv(src).expect("two-compute-entry shader should compile");
    // BuiltIn enum value for GlobalInvocationId is 28
    // (spirv::BuiltIn::GlobalInvocationId as u32).
    let global_invocation_id_builtin = spirv::BuiltIn::GlobalInvocationId as u32;
    // OpDecorate is opcode 71 with word_count 4 for BuiltIn:
    // [opcode|word_count<<16, target_id, Decoration::BuiltIn(11),
    //  BuiltIn_enum_value]. Scan for that pattern.
    const OP_DECORATE: u32 = 71;
    const DECORATION_BUILTIN: u32 = 11;
    let mut count = 0;
    let mut i = 5; // skip 5-word header
    while i + 3 < spirv.len() {
        let word = spirv[i];
        let opcode = word & 0xFFFF;
        let word_count = (word >> 16) as usize;
        if opcode == OP_DECORATE
            && word_count == 4
            && spirv[i + 2] == DECORATION_BUILTIN
            && spirv[i + 3] == global_invocation_id_builtin
        {
            count += 1;
        }
        if word_count == 0 {
            break;
        }
        i += word_count;
    }
    assert_eq!(
        count, 1,
        "expected exactly one `OpDecorate ... BuiltIn GlobalInvocationId`, got {count}"
    );
}

#[test]
fn entry_point_interfaces_are_unique_when_storage_input_and_output_alias() {
    use crate::LookupSet;
    use wspirv::dr::Operand;

    let spirv = compile_to_spirv(
        r#"
type painted = { values: []f32 }

def triangle_vertex(vertex_index: u32, instance_index: u32, draw_index: u32) =
  vertex_output(
    @[1.0, 1.0, 0.0, 1.0],
    ())

def resolve(p: painted, _fragment_value: (), _fragment_position: vec4f32, _fragment_front_facing: bool, _fragment_primitive_index: u32, _fragment_sample_index: u32) =
  let value = 0.0 in
  @[value, value, value, 1.0]

entry reproduce(values: []f32, surface: render_target<vec4f32>)
    ([]f32, render_target<vec4f32>) =
  let values_next = map(|i| 1.0, iota(0))
  let painted_scene = { values = values }
  let raster = rasterize_triangles(
    direct_draw(3u32, 1u32), triangle_vertex)
  let surface1 = shade(surface, raster, resolve(painted_scene, _, _, _, _, _)) in
  (values_next, surface1)
"#,
    )
    .expect("record-array graphics program should compile");

    let module = wspirv::dr::load_words(&spirv).expect("backend emitted parseable SPIR-V");
    for entry in &module.entry_points {
        let interface_ids = entry.operands.iter().skip(3).filter_map(|operand| match operand {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        });
        let mut seen = LookupSet::new();
        for interface_id in interface_ids {
            assert!(
                seen.insert(interface_id),
                "OpEntryPoint contains duplicate interface %{interface_id}"
            );
        }
    }
}

/// `scatter` into a `#[storage]` framebuffer lowers end-to-end: the full
/// `SoacKind::Scatter` → `SoacOp::Scatter` → `egglog::OperationKind::Scatter` →
/// `build_scatter_loop` path emits indexed `OpStore`s into the destination
/// view (one per scattered element; N=5 here, unrolled).
#[test]
fn scatter_into_storage_buffer_lowers() {
    let spirv = compile_to_spirv(
        r#"
def N:i32 = 5
entry rasterize(positions: []vec4f32,
                fb: []vec4f32) () =
  let pts  = positions[0..N] in
  let idxs = map(|p:vec4f32| i32.f32(p.y) * 512 + i32.f32(p.x), pts) in
  let vals = map(|p:vec4f32| @[1.0, 1.0, 1.0, 1.0], pts) in
  let _ = scatter(fb, idxs, vals) in ()
"#,
    )
    .expect("scatter rasterizer must lower to SPIR-V");
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic number");
    const OP_STORE: u32 = 62;
    let stores = spirv.iter().skip(5).filter(|w| (*w & 0xFFFF) == OP_STORE).count();
    assert!(
        stores >= 5,
        "expected >= 5 OpStore (one per scattered particle), got {stores}"
    );
}

/// Disassemble SPIR-V words to text for instruction-level assertions.
fn disasm(words: &[u32]) -> String {
    use wspirv::binary::Disassemble;
    wspirv::dr::load_words(words).expect("backend emitted valid SPIR-V").disassemble()
}

/// Lines per `(set, binding)` storage class help locate hoisted globals: a
/// compile-time-constant array indexed by a runtime value is materialized once
/// into a module-scope `Private` global (initializer = the constant), not
/// `OpStore`d wholesale into a per-occurrence `Function` array variable.
#[test]
fn dynamic_const_array_index_hoists_to_private_global() {
    let spirv = compile_to_spirv(
        "def t: [4]i32 = [10, 20, 30, 40]\n\
         \n\
         entry pick() []i32 = map(|i| t[i % 4], iota(100))",
    )
    .unwrap();
    let text = disasm(&spirv);
    let private_vars: Vec<&str> =
        text.lines().filter(|l| l.contains("OpVariable") && l.contains(" Private ")).collect();
    assert_eq!(
        private_vars.len(),
        1,
        "expected one hoisted Private global:\n{text}"
    );
    // The Private global carries a constant initializer (trailing operand).
    let last = private_vars[0].trim_end().rsplit(' ').next().unwrap();
    assert!(
        last.starts_with('%'),
        "Private global must have an initializer: {}",
        private_vars[0]
    );
    // The constant is not also materialized into a Function-storage array.
    assert!(
        !text.contains("_ptr_Function__arr"),
        "constant array must not be stored into a Function variable:\n{text}"
    );
}

/// The hoist is deduped by constant value: the same constant array indexed at
/// two sites collapses to a single `Private` global.
#[test]
fn const_array_hoist_is_deduped() {
    let spirv = compile_to_spirv(
        "def t: [4]i32 = [10, 20, 30, 40]\n\
         \n\
         entry pick() []i32 = map(|i| t[i % 4] + t[(i + 1) % 4], iota(100))",
    )
    .unwrap();
    let text = disasm(&spirv);
    let n = text.lines().filter(|l| l.contains("OpVariable") && l.contains(" Private ")).count();
    assert_eq!(
        n, 1,
        "two indexings of one constant must share one Private global:\n{text}"
    );
}
