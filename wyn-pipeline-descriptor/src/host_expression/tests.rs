use super::*;
use crate::{BufferLen, DispatchLen, DispatchSize};
use serde_json::json;

fn count(text: &str) -> Result<HostExpression, serde_json::Error> {
    serde_json::from_value(json!({
        "inputs": { "width": {"set": 0, "binding": 0, "offset": 16, "type": "f32"},
                    "height": {"set": 0, "binding": 0, "offset": 20, "type": "f32"} },
        "count": text
    }))
}

#[test]
fn descriptor_has_named_inputs_and_string_count() {
    let expression = count("i32(width) * i32(height)").unwrap();
    let len = BufferLen::HostExpression {
        count: expression.clone(),
        elem_bytes: 4,
    };
    let value = serde_json::to_value(&len).unwrap();
    assert_eq!(
        value,
        json!({
            "kind": "host_expression", "elem_bytes": 4,
            "inputs": { "width": {"set": 0, "binding": 0, "offset": 16, "type": "f32"},
                        "height": {"set": 0, "binding": 0, "offset": 20, "type": "f32"} },
            "count": "i32(width) * i32(height)"
        })
    );
    assert_eq!(serde_json::from_value::<BufferLen>(value.clone()).unwrap(), len);
    let dispatch = DispatchSize::DerivedFrom {
        len: DispatchLen::HostExpression { count: expression },
        workgroup_size: 64,
    };
    let json = serde_json::to_value(&dispatch).unwrap();
    assert_eq!(json["len"]["inputs"], value["inputs"]);
    assert_eq!(json["len"]["count"], value["count"]);
    assert_eq!(serde_json::from_value::<DispatchSize>(json).unwrap(), dispatch);
    assert_eq!(
        len.resolve_host_bytes(&|s, b, o| match (s, b, o) {
            (0, 0, 16) => Some(1280f32.to_bits()),
            (0, 0, 20) => Some(800f32.to_bits()),
            _ => None,
        })
        .unwrap(),
        4096000
    );
}

#[test]
fn precedence_casts_literals_and_parentheses() {
    for (text, expected) in [
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("20 / 2 / 2", 5),
        ("20 / (2 / 2)", 20),
        ("20 - (3 - 2)", 19),
        ("17 % 5", 2),
        ("i32(17.9f32) * i32(9.8f32)", 153),
        ("i32(1.25e2f32)", 125),
        ("u32(-1)", u32::MAX as u64),
        ("4294967295u32", u32::MAX as u64),
        ("i32(-2147483648) + 2147483647 + 1", 0),
        ("-(2 - 5)", 3),
        ("((i32(width) + 7) / 8) * ((i32(height) + 7) / 8)", 16000),
    ] {
        let expression = count(text).unwrap();
        let encoded = serde_json::to_value(&expression).unwrap();
        let decoded: HostExpression = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, expression, "{text}");
        assert_eq!(
            decoded
                .element_count(&|_, _, o| Some(if o == 16 { 1280f32.to_bits() } else { 800f32.to_bits() }))
                .unwrap(),
            expected,
            "{text}"
        );
    }
    let minus_zero = count("-0.0f32").unwrap();
    assert_eq!(
        minus_zero.evaluate(&|_, _, _| None).unwrap(),
        (HostScalar::F32, (-0.0f32).to_bits())
    );
}

#[test]
fn invalid_syntax_and_types_are_rejected() {
    for text in [
        "",
        "width +",
        "unknown",
        "width.height",
        "width[0]",
        "eval(width)",
        "1; 2",
        "(1 + 2",
        "1 2",
        "1 /",
        "1 + 2u32",
        "width + 2",
        "NaNf32",
        "1e99f32",
        "2147483648",
        "-1u32",
    ] {
        assert!(count(text).is_err(), "accepted {text}");
    }
    assert!(count(&format!("{}1{}", "(".repeat(100), ")".repeat(100))).is_err());
    assert!(count(&"1 + ".repeat(1000)).is_err());
    assert!(serde_json::from_value::<DispatchLen>(
        json!({"kind":"host_expression", "count":{"kind":"fixed","count":1}})
    )
    .is_err());
}

#[test]
fn evaluation_retains_checked_arithmetic() {
    for text in [
        "2147483647 + 1",
        "4294967295u32 + 1u32",
        "1 / 0",
        "1 % 0",
        "-1",
        "-2147483648 / -1",
        "i32(2147483648.0f32)",
        "1.0f32",
        "i32(3.4e38f32 * 2.0f32)",
    ] {
        assert!(
            count(text).unwrap().element_count(&|_, _, _| None).is_err(),
            "accepted {text}"
        );
    }
    assert!(count("i32(width)").unwrap().element_count(&|_, _, _| None).is_err());
    for bits in [f32::NAN.to_bits(), f32::INFINITY.to_bits()] {
        assert!(count("i32(width)").unwrap().element_count(&|_, _, _| Some(bits)).is_err());
    }
}
