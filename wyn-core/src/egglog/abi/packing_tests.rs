use super::*;
use crate::types;
use egglog_engine::{Core, EGraph};

fn writers() -> [Value; 2] {
    let mut graph = EGraph::default();
    graph
        .update(|sink| Ok::<_, egglog_engine::Error>([sink.base_to_value(0i64), sink.base_to_value(1i64)]))
        .unwrap()
}
fn output(writer: Value, slot: u32, element: Type) -> Output {
    Output {
        writer,
        binding: BindingRef::new(0, slot),
        element,
        name: format!("field_{slot}"),
    }
}

#[test]
fn mixed_outputs_have_disjoint_aligned_ranges_and_one_capacity_per_group() {
    let [writer, _] = writers();
    let outputs = (0..9)
        .map(|i| {
            output(
                writer,
                i,
                match i % 3 {
                    0 => types::f32(),
                    1 => types::i32(),
                    _ => types::vec(3, types::f32()),
                },
            )
        })
        .collect::<Vec<_>>();
    let layout = plan(outputs.clone()).unwrap();
    assert_eq!(
        layout,
        plan(outputs).unwrap(),
        "allocation decisions are deterministic"
    );
    for i in 0..9 {
        let field = &layout[&BindingRef::new(0, i)];
        let stride = if i % 3 == 2 { 16 } else { 4 };
        assert_eq!(field.binding, BindingRef::new(0, i % 3));
        assert_eq!(field.range(), (i / 3 * stride, stride));
        assert_eq!(field.allocation_bytes, 3 * stride);
        assert_eq!(field.name, format!("field_{}", i % 3));
    }
}

#[test]
fn independent_writers_never_share_physical_storage() {
    let writers = writers();
    let outputs = (0..16).map(|i| output(writers[(i / 8) as usize], i, types::f32())).collect();
    let layout = plan(outputs).unwrap();
    assert_eq!(layout.len(), 16);
    for i in 0..16 {
        assert_eq!(
            layout[&BindingRef::new(0, i)].binding,
            BindingRef::new(0, i / 8 * 8)
        );
        assert_eq!(layout[&BindingRef::new(0, i)].allocation_bytes, 32);
    }
}

#[test]
fn small_interfaces_explicit_bindings_and_special_widths_keep_their_contracts() {
    let [writer, _] = writers();
    assert!(plan((0..7).map(|i| output(writer, i, types::f32())).collect()).unwrap().is_empty());
    let mut explicit = (0..8).map(|i| output(writer, i, types::f32())).collect::<Vec<_>>();
    for field in &mut explicit {
        field.binding.set = 1;
    }
    assert!(plan(explicit).unwrap().is_empty());
    assert!(
        plan((0..8).map(|i| output(writer, i, types::vec(2, types::f16()))).collect()).unwrap().is_empty()
    );
}
