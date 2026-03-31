// SOAC Analysis Demo
// Demonstrates SSA-level SOAC analysis for compute shader parallelization

use wyn_core::{Compiler, FrontEnd, ssa};

fn main() {
    let source = r#"
-- SOAC Analysis Demo
-- Tests: map loop detection, array provenance tracking

-- Helper function: does the actual map
def process_array(data: []f32) []f32 =
    map(|x| x * 2.0, data)

-- Main compute entry point with size hints
#[compute]
entry compute_main(
    #[size_hint(1024)] vectors: []f32
) []f32 =
    process_array(vectors)
"#;

    println!("=== SSA SOAC Analysis Demo ===\n");
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

    // Continue through TLC transformations
    let defunctionalized = tlc_transformed.fuse_maps().defunctionalize();
    let monomorphized = defunctionalized.monomorphize();

    // Convert to SSA using the new direct path
    let ssa_converted =
        monomorphized.soa_transform().buffer_specialize().inline().to_ssa().expect("SSA conversion failed");

    // Run SSA SOAC analysis
    let analysis = ssa::soac_analysis::analyze_program(&ssa_converted.ssa);

    // Print results
    println!("=== Analysis Results ===\n");
    println!("Total compute entry points analyzed: {}", analysis.by_entry.len());
    println!();

    for (entry_name, entry_analysis) in &analysis.by_entry {
        println!("Entry point: {}", entry_name);
        println!("{}", "-".repeat(40));
        println!("  Local size: {:?}", entry_analysis.local_size);

        if let Some(ref par_soac) = entry_analysis.parallelizable_soac {
            println!("  Parallelizable SOAC found: {:?}", par_soac);
        } else {
            println!("  No parallelizable SOAC found");
        }
        println!();
    }

    // Summary
    println!("=== Summary ===");
    let parallelizable_count =
        analysis.by_entry.values().filter(|a| a.parallelizable_soac.is_some()).count();
    let non_parallelizable_count = analysis.by_entry.len() - parallelizable_count;
    println!("Parallelizable compute shaders: {}", parallelizable_count);
    println!("Non-parallelizable compute shaders: {}", non_parallelizable_count);
}
