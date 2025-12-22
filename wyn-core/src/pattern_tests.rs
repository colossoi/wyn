use crate::ast::{Header, NodeId, Pattern, PatternKind, Span};
use crate::pattern::{PatternError, PatternValue, bound_names, extract_bindings};

// Simple test value that can be a scalar or tuple
#[derive(Clone, Debug, PartialEq)]
enum TestValue {
    Scalar(i32),
    Tuple(Vec<TestValue>),
}

impl PatternValue for TestValue {
    fn tuple_element(&self, index: usize) -> Option<Self> {
        match self {
            TestValue::Tuple(elems) => elems.get(index).cloned(),
            _ => None,
        }
    }

    fn tuple_len(&self) -> Option<usize> {
        match self {
            TestValue::Tuple(elems) => Some(elems.len()),
            _ => None,
        }
    }
}

fn mk_pattern(kind: PatternKind) -> Pattern {
    Pattern {
        h: Header {
            id: NodeId(0),
            span: Span {
                start_line: 0,
                start_col: 0,
                end_line: 0,
                end_col: 0,
            },
        },
        kind,
    }
}

#[test]
fn test_simple_name() {
    let pattern = mk_pattern(PatternKind::Name("x".to_string()));
    let value = TestValue::Scalar(42);

    let bindings = extract_bindings(&pattern, value).unwrap();
    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings[0].0, "x");
    assert_eq!(bindings[0].1, TestValue::Scalar(42));
}

#[test]
fn test_wildcard() {
    let pattern = mk_pattern(PatternKind::Wildcard);
    let value = TestValue::Scalar(42);

    let bindings = extract_bindings(&pattern, value).unwrap();
    assert_eq!(bindings.len(), 0);
}

#[test]
fn test_tuple_pattern() {
    let pattern = mk_pattern(PatternKind::Tuple(vec![
        mk_pattern(PatternKind::Name("x".to_string())),
        mk_pattern(PatternKind::Name("y".to_string())),
    ]));
    let value = TestValue::Tuple(vec![TestValue::Scalar(1), TestValue::Scalar(2)]);

    let bindings = extract_bindings(&pattern, value).unwrap();
    assert_eq!(bindings.len(), 2);
    assert_eq!(bindings[0], ("x".to_string(), TestValue::Scalar(1)));
    assert_eq!(bindings[1], ("y".to_string(), TestValue::Scalar(2)));
}

#[test]
fn test_nested_tuple() {
    // (x, (y, z))
    let pattern = mk_pattern(PatternKind::Tuple(vec![
        mk_pattern(PatternKind::Name("x".to_string())),
        mk_pattern(PatternKind::Tuple(vec![
            mk_pattern(PatternKind::Name("y".to_string())),
            mk_pattern(PatternKind::Name("z".to_string())),
        ])),
    ]));
    let value = TestValue::Tuple(vec![
        TestValue::Scalar(1),
        TestValue::Tuple(vec![TestValue::Scalar(2), TestValue::Scalar(3)]),
    ]);

    let bindings = extract_bindings(&pattern, value).unwrap();
    assert_eq!(bindings.len(), 3);
    assert_eq!(bindings[0], ("x".to_string(), TestValue::Scalar(1)));
    assert_eq!(bindings[1], ("y".to_string(), TestValue::Scalar(2)));
    assert_eq!(bindings[2], ("z".to_string(), TestValue::Scalar(3)));
}

#[test]
fn test_tuple_with_wildcard() {
    // (x, _, z)
    let pattern = mk_pattern(PatternKind::Tuple(vec![
        mk_pattern(PatternKind::Name("x".to_string())),
        mk_pattern(PatternKind::Wildcard),
        mk_pattern(PatternKind::Name("z".to_string())),
    ]));
    let value = TestValue::Tuple(vec![
        TestValue::Scalar(1),
        TestValue::Scalar(2),
        TestValue::Scalar(3),
    ]);

    let bindings = extract_bindings(&pattern, value).unwrap();
    assert_eq!(bindings.len(), 2);
    assert_eq!(bindings[0], ("x".to_string(), TestValue::Scalar(1)));
    assert_eq!(bindings[1], ("z".to_string(), TestValue::Scalar(3)));
}

#[test]
fn test_bound_names() {
    // (x, (y, _))
    let pattern = mk_pattern(PatternKind::Tuple(vec![
        mk_pattern(PatternKind::Name("x".to_string())),
        mk_pattern(PatternKind::Tuple(vec![
            mk_pattern(PatternKind::Name("y".to_string())),
            mk_pattern(PatternKind::Wildcard),
        ])),
    ]));

    let names = bound_names(&pattern);
    assert_eq!(names, vec!["x", "y"]);
}

#[test]
fn test_tuple_length_mismatch() {
    let pattern = mk_pattern(PatternKind::Tuple(vec![
        mk_pattern(PatternKind::Name("x".to_string())),
        mk_pattern(PatternKind::Name("y".to_string())),
    ]));
    let value = TestValue::Tuple(vec![TestValue::Scalar(1)]);

    let result = extract_bindings(&pattern, value);
    assert!(matches!(result, Err(PatternError::TupleLengthMismatch { .. })));
}
