use crate::{Backend, Error, Form, Number, Options, Program, Result, Value};

#[path = "test_backend.rs"]
mod test_backend;
use test_backend::Trace;

fn expression(body: &str) -> Result<Value> {
    let program = Program::parse(&format!(
        "(define-host-program :version 1)
         (define-host-entry 'main :source-name \"main\" :function 'host-main :parameters '() :results '((answer :i64)))
         (defun host-main () {body})"
    ))?;
    crate::eval::run(&program, "host-main", &[], &mut Trace::default()).and_then(Value::materialize)
}

#[test]
fn lexical_bindings_short_circuit_and_parallel_loop_steps() {
    assert_eq!(
        expression("(let ((x 4)) (let ((x 7) (y x)) (+ x y)))").unwrap(),
        Value::Number(Number::I32(11))
    );
    assert_eq!(
        expression("(let* ((x 4) (y (+ x 3))) (+ x y))").unwrap(),
        Value::Number(Number::I32(11))
    );
    assert_eq!(
        expression("(if (and nil (/ 1 0)) 3 (or nil 9))").unwrap(),
        Value::Number(Number::I32(9))
    );
    assert_eq!(
        expression("(do ((a 1 b) (b 2 a) (i 0 (+ i 1))) ((= i 3) a))").unwrap(),
        Value::Number(Number::I32(2))
    );
    assert_eq!(
        expression("(let ((sum 0)) (dotimes (i 5 sum) (setq sum (+ sum i))))").unwrap(),
        Value::Number(Number::I32(10))
    );
    assert!(expression("(dotimes (i 2) (setq i 0))").is_err());
}

#[test]
fn fixed_width_numeric_contracts() {
    assert_eq!(
        expression("(* (u64 4294967295) (u64 16))").unwrap(),
        Value::Number(Number::U64(68_719_476_720))
    );
    assert_eq!(
        expression("(u32-add (u32 4294967295) 1)").unwrap(),
        Value::Number(Number::U32(0))
    );
    assert_eq!(
        expression("(i32-add 2147483647 1)").unwrap(),
        Value::Number(Number::I32(i32::MIN))
    );
    assert_eq!(
        expression("(ceiling (u32 4294967295) 64)").unwrap(),
        Value::Number(Number::U32(67_108_864))
    );
    assert_eq!(
        expression("(floor -5 2)").unwrap(),
        Value::Number(Number::I32(-3))
    );
    assert_eq!(expression("(mod 5 -2)").unwrap(), Value::Number(Number::I32(-1)));
    for source in [
        "(+ 2147483647 1)",
        "(+ (u32 1) (i32 1))",
        "(/ 3 2)",
        "(/ 1 0)",
        "(i32 4294967295)",
        "(* (u64 18446744073709551615) 2)",
        "(floor (i64 -9223372036854775808) (i64 -1))",
    ] {
        assert!(expression(source).is_err(), "{source}");
    }
}

#[test]
fn reader_handles_quotes_comments_case_and_literal_string_escapes() {
    let forms = crate::read(";comment\n'(ABC :READ \"C:\\\\x\\\"y\" 3.5d0)").unwrap();
    let Form::List(quote) = &forms[0] else {
        panic!("quote");
    };
    let Form::List(values) = &quote[1] else {
        panic!("list");
    };
    assert_eq!(values[0], Form::Symbol("abc".into()));
    assert_eq!(values[2], Form::String("C:\\x\"y".into()));
    for bad in [
        "#.(evil)",
        "(a . b)",
        "`(a)",
        "(a",
        "\"unfinished",
        "1/3",
        "18446744073709551616",
        "9999999999999999999999999999999999999999999999",
    ] {
        assert!(crate::read(bad).is_err(), "{bad}");
    }
    assert_eq!(
        expression("(LET* ((X 1) (Y (+ X 2))) Y)").unwrap(),
        Value::Number(Number::I32(3))
    );
}

#[test]
fn rejects_duplicate_options_and_recursion_without_executing_code() {
    assert!(Program::parse("(define-host-program :version 1 :version 1)").is_err());
    assert!(Program::parse("(define-host-program :version 1)(defun recurse () (recurse))").is_err());
    assert!(Program::parse(
        "(define-host-program :version 1)(defun first () (second))(defun second () (first))"
    )
    .is_err());
    assert!(expression("(setq missing 1)").is_err());
    assert!(expression("(let ((t 4)) t)").is_err());
    assert!(expression("(let)").is_err());
    assert!(expression("(dotimes (i 1) tag)").is_err());
    assert!(expression("(mod -2147483648 -1)").is_err());
}

#[test]
fn entries_apply_declared_literal_types_and_reject_mismatched_values() {
    let program = Program::parse(
        "(define-host-program :version 1)
        (define-host-entry 'main :function 'host-main :parameters '((n :u32)) :results '((result :u32)))
        (defun host-main (n) (if (= n 0) 4294967295 n))",
    )
    .unwrap();
    let mut backend = Trace::default();
    assert_eq!(
        program.run("main", &[Value::Number(Number::Literal(0))], &mut backend).unwrap(),
        Value::Number(Number::U32(u32::MAX))
    );
    assert!(program.run("main", &[Value::Number(Number::I32(0))], &mut backend).is_err());
    assert!(program.run("main", &[], &mut backend).is_err());
}

#[test]
fn invalid_return_shape_is_reported_to_backend_cleanup() {
    struct Cleanup(bool);
    impl Backend for Cleanup {
        fn call(&mut self, _: &Program, _: &str, _: &[Value]) -> Result<Value> {
            unreachable!()
        }
        fn finish(&mut self, result: &Result<Value>) -> Result<()> {
            self.0 = result.is_err();
            Ok(())
        }
    }
    let program = Program::parse(
        "(define-host-program :version 1)
        (define-host-entry 'main :function 'host-main :parameters '() :results '())
        (defun host-main () 42)",
    )
    .unwrap();
    let mut backend = Cleanup(false);
    assert!(program.run("main", &[], &mut backend).is_err());
    assert!(backend.0);
}
