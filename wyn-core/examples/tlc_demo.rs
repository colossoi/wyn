use wyn_core::{Compiler, FrontEnd, tlc};

fn main() {
    let source = r#"
def compute(x:i32, y:i32) i32 =
  let sum = x + y in
  let product = x * y in
  sum + product

"#;
    let mut frontend = FrontEnd::new();
    let parsed = Compiler::parse(source, &mut frontend.node_counter).expect("parse failed");
    let desugared = parsed.desugar(&mut frontend.node_counter).expect("desugar failed");
    let resolved = desugared.resolve(&mut frontend.module_manager).expect("resolve failed");
    let folded = resolved.fold_ast_constants();
    let typed =
        folded.type_check(&mut frontend.module_manager, &mut frontend.schemes).expect("type check failed");

    let tlc_program = tlc::transform(&typed.ast, &typed.type_table);
    println!("{}", tlc_program);
}
