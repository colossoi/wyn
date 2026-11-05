use super::*;

#[test]
fn native_literals_preserve_bits_and_integer_widths() {
    for bits in [0_u32, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_1234] {
        let ty = scalar_type("f32").unwrap();
        assert_eq!(encode(&ty, decode(&ty, bits as i64).unwrap()), Some(bits as i64));
        assert_eq!(
            evaluate_unary("bitcast", "f32", "u32", bits as i64),
            Some(bits as i64)
        );
    }
    for (name, bits) in [
        ("i64", i64::MIN),
        ("u64", -1),
        ("i32", i32::MIN as i64),
        ("u32", u32::MAX as i64),
        ("i8", -128),
        ("u8", 255),
        ("bool", 0),
        ("bool", 1),
    ] {
        let ty = scalar_type(name).unwrap();
        assert_eq!(encode(&ty, decode(&ty, bits).unwrap()), Some(bits), "{name}");
    }
    assert_eq!(evaluate_unary("bitcast", "u32", "i32", u32::MAX as i64), Some(-1));
}

#[test]
fn native_literal_primitives_fold_without_text_payloads() {
    let mut graph = EGraph::default();
    register(&mut graph);
    graph
        .parse_and_run_program(
            None,
            r#"
        (check (= (wyn-binary "+" "u8" "u8" 255 1) 0))
        (check (= (wyn-binary "+" "u64" "u64" -1 1) 0))
        (check (= (wyn-binary "==" "i32" "bool" 7 7) 1))
        (check (= (wyn-unary "bitcast" "u32" "f32" 2147483648) 2147483648))
        (check (= (wyn-unary "!" "bool" "bool" 0) 1))
    "#,
        )
        .unwrap();
}
