// SOAC Analysis Demo
// Demonstrates TLC-level SOAC analysis for compute shader workload estimation

use wyn_core::{Compiler, FrontEnd, tlc};

fn main() {
    let source = r#"
-- SOAC Analysis Demo
-- Tests: size hints, call chains, interprocedural propagation

-- Helper function: does the actual map
def process_array(data: []f32) []f32 =
    map(|x| x * 2.0, data)

-- Sum an array using reduce
def sum_array(arr: []f32) f32 =
    reduce(|a, b| a + b, 0.0, arr)

-- Chain two maps together
def double_process(data: []f32) []f32 =
    let step1 = map(|x| x + 1.0, data) in
    map(|x| x * 2.0, step1)

-- Main compute entry point with size hints
#[compute]
entry compute_main(
    #[size_hint(1024)] vectors: []f32
) ([]f32, f32) =
    let processed = double_process(vectors) in
    let total = sum_array(processed) in
    (processed, total)
"#;

    println!("=== SOAC Analysis Demo ===\n");
    println!("Source program:");
    println!("---------------");
    for (i, line) in source.lines().enumerate() {
        println!("{:3} | {}", i + 1, line);
    }
    println!();

    // Parse and type-check using full pipeline
    let mut frontend = FrontEnd::new();
    let parsed = Compiler::parse(source, &mut frontend.node_counter).expect("parse failed");
    let desugared = parsed.desugar(&mut frontend.node_counter).expect("desugar failed");
    let resolved = desugared.resolve(&mut frontend.module_manager).expect("resolve failed");
    let folded = resolved.fold_ast_constants();
    let typed =
        folded.type_check(&mut frontend.module_manager, &mut frontend.schemes).expect("type check failed");

    // Alias check and transform to TLC with prelude
    let alias_checked = typed.alias_check().expect("alias check failed");
    let known_defs = frontend.intrinsics.all_names();
    let tlc_transformed = alias_checked.to_tlc(known_defs, &frontend.schemes, &mut frontend.module_manager);

    // Run SOAC analysis on the TLC program (before defunctionalization)
    let analysis = tlc::soac_analysis::analyze_program(&tlc_transformed.tlc);

    // Print results
    println!("=== Analysis Results ===\n");
    println!("Total compute entry points: {}", analysis.by_entry.len());
    println!("Total SOACs found: {}\n", analysis.total_soac_count());

    for (entry_name, soacs) in &analysis.by_entry {
        println!("Entry point: {}", entry_name);
        println!("{}", "-".repeat(40));

        if soacs.is_empty() {
            println!("  No SOACs found");
        } else {
            for (i, soac) in soacs.iter().enumerate() {
                println!("  SOAC #{}", i + 1);
                println!("    Kind:           {}", soac.kind);
                println!("    Location:       line {}", soac.span.start_line);
                println!("    Input size:     {}", soac.input_size);
                println!("    Output size:    {}", soac.output_size);
                println!("    Nesting:        depth {}", soac.nesting_depth);
                if let Some(parent) = soac.parent {
                    println!("    Parent:         TermId({})", parent.0);
                }
                println!("    Independent:    {}", soac.kind.is_independent());
                println!();
            }
        }
        println!();
    }

    // Summary
    println!("=== Summary ===");
    let total_independent =
        analysis.by_entry.values().flat_map(|s| s.iter()).filter(|s| s.kind.is_independent()).count();
    let total_dependent = analysis.total_soac_count() - total_independent;
    println!("Independent SOACs (parallelizable): {}", total_independent);
    println!("Dependent SOACs (need coordination): {}", total_dependent);
}
