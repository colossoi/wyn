use wyn_core::{Compiler, init_compiler};

fn main() {
    let source = r#"
def compute(x:i32, y:i32) i32 =
  let sum = x + y in
  let product = x * y in
  sum + product

"#;
    let (mut node_counter, mut module_manager) = init_compiler();
    let typed = Compiler::parse(source, &mut node_counter)
        .expect("parse failed")
        .resolve(&module_manager)
        .expect("resolve failed")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type check failed");

    let tlc = typed.to_tlc(&module_manager, false);
    println!("{}", tlc.0.tlc);
}
