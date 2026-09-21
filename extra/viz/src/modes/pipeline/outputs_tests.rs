use super::*;
use wyn_host_interp::read;

fn layout(source: &str) -> Value {
    Value::quoted(&read(source).unwrap()[0])
}

#[test]
fn decodes_signed_arrays_and_record_fields() {
    let bytes = [-1i32, 42].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    assert_eq!(
        decode(
            &layout("(:array :length 2 :stride 4 :range :caller :element :i32)"),
            &bytes,
            0
        )
        .unwrap(),
        json!({"backing_buffer":[-1,42],"byte_length":8,"declared_count":2})
    );
    assert_eq!(
        decode(
            &layout(
                "(:record :size 8 :fields ((\"x\" :offset 0 :layout :i32) (\"y\" :offset 4 :layout :i32)))"
            ),
            &bytes,
            0
        )
        .unwrap(),
        json!({"x":-1,"y":42})
    );
}

#[test]
fn dynamic_arrays_are_identified_as_backing_storage() {
    let bytes = 99u32.to_le_bytes();
    assert_eq!(
        decode(
            &layout("(:array :length :dynamic :stride 4 :range :caller :element :u32)"),
            &bytes,
            0
        )
        .unwrap(),
        json!({"backing_buffer":[99],"byte_length":4})
    );
    assert!(decode(
        &layout("(:array :length 2 :stride 4 :range :caller :element :u32)"),
        &bytes,
        0
    )
    .is_err());
}
