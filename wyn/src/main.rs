use clap::{Parser, Subcommand, ValueEnum};
use log::info;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use thiserror::Error;
use wyn_core::{Compiler, lexer, parser::Parser as WynParser};

/// Target output format
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum Target {
    /// SPIR-V binary (default)
    #[default]
    Spirv,
    /// GLSL source code
    Glsl,
    /// GLSL for Shadertoy (fragment shader only, mainImage entry point)
    Shadertoy,
}

/// Times the execution of a closure and prints the elapsed time if verbose.
fn time<T, F: FnOnce() -> T>(name: &str, verbose: bool, f: F) -> T {
    let start = Instant::now();
    let result = f();
    if verbose {
        let elapsed = start.elapsed().as_millis();
        eprintln!("{}: {}ms", name, elapsed);
    }
    result
}

#[derive(Parser)]
#[command(name = "wyn")]
#[command(about = "A minimal Futhark-like language compiler targeting SPIR-V", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a source file to SPIR-V or GLSL
    Compile {
        /// Input source file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output file (defaults to input name with .spv or .glsl extension)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Target output format
        #[arg(short, long, default_value = "spirv")]
        target: Target,

        /// Output annotated source code with block IDs and locations
        #[arg(long, value_name = "FILE")]
        output_annotated: Option<PathBuf>,

        /// Output typed lambda calculus representation
        #[arg(long, value_name = "FILE")]
        output_tlc: Option<PathBuf>,

        /// Output initial SSA (right after conversion from TLC, before optimization)
        #[arg(long, value_name = "FILE")]
        output_init_ssa: Option<PathBuf>,

        /// Output optimized SSA (after optimization, before SOAC lowering)
        #[arg(long, value_name = "FILE")]
        output_opt_ssa: Option<PathBuf>,

        /// Print verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Validate a source file without generating output
    Check {
        /// Input source file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output annotated source code with block IDs and locations
        #[arg(long, value_name = "FILE")]
        output_annotated: Option<PathBuf>,

        /// Print verbose output
        #[arg(short, long)]
        verbose: bool,
    },
}

#[derive(Debug, Error)]
enum DriverError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Compilation error: {0}")]
    CompilationError(#[from] wyn_core::error::CompilerError),

    #[error("SSA conversion error: {0}")]
    SsaConversionError(#[from] wyn_core::tlc::to_ssa::ConvertError),

    #[error("EGraph conversion error: {0}")]
    EgirConversionError(#[from] wyn_core::egir::from_tlc::ConvertError),
}

fn main() -> Result<(), DriverError> {
    env_logger::init();
    let cli = Cli::parse();

    // Spawn on a thread with a larger stack to avoid stack overflow
    // on deeply recursive type/SSA operations.
    let result = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024) // 16 MB
        .spawn(move || run(cli))
        .expect("Failed to spawn compiler thread")
        .join()
        .expect("Compiler thread panicked");
    result
}

fn run(cli: Cli) -> Result<(), DriverError> {
    match cli.command {
        Commands::Compile {
            input,
            output,
            target,
            output_annotated,
            output_tlc,
            output_init_ssa,
            output_opt_ssa,
            verbose,
        } => {
            compile_file(
                input,
                output,
                target,
                output_annotated,
                output_tlc,
                output_init_ssa,
                output_opt_ssa,
                verbose,
            )?;
        }
        Commands::Check {
            input,
            output_annotated,
            verbose,
        } => {
            check_file(input, output_annotated, verbose)?;
        }
    }

    Ok(())
}

fn compile_file(
    input: PathBuf,
    output: Option<PathBuf>,
    target: Target,
    output_annotated: Option<PathBuf>,
    output_tlc: Option<PathBuf>,
    output_init_ssa: Option<PathBuf>,
    output_opt_ssa: Option<PathBuf>,
    verbose: bool,
) -> Result<(), DriverError> {
    if verbose {
        info!("Compiling {}...", input.display());
    }

    // Read source file
    let source = fs::read_to_string(&input)?;

    // Generate annotated source if requested
    if let Some(ref annotated_path) = output_annotated {
        generate_annotated_source(&source, annotated_path, verbose)?;
    }

    // Compile through the pipeline
    // Create FrontEnd first - it owns the node counter and loads prelude
    let mut frontend = time("frontend", verbose, wyn_core::FrontEnd::new);
    // Parse user code using the same counter (so IDs don't collide with prelude)
    let parsed = time("parse", verbose, || {
        Compiler::parse(&source, &mut frontend.node_counter)
    })?;
    // Elaborate inline modules so they're available during resolution
    let parsed = time("elaborate_modules", verbose, || {
        parsed.elaborate_modules(&mut frontend.module_manager)
    })?;
    // Desugar ranges/slices early, before name resolution and type checking
    let desugared = time("desugar", verbose, || parsed.desugar(&mut frontend.node_counter))?;
    let resolved = time("resolve", verbose, || desugared.resolve(&frontend.module_manager))?;
    let ast_folded = time("fold_ast_constants", verbose, || resolved.fold_ast_constants());
    let type_checked = time("type_check", verbose, || {
        ast_folded.type_check(&mut frontend.module_manager, &mut frontend.schemes)
    })?;

    type_checked.print_warnings();

    let alias_checked = time("alias_check", verbose, || type_checked.alias_check())?;
    if alias_checked.has_alias_errors() {
        alias_checked.print_alias_errors();
        return Err(wyn_core::err_alias!("alias checking failed").into());
    }

    // Transform to TLC (including prelude code - transformed here for consistent type variables)
    let tlc_transformed = time("to_tlc", verbose, || {
        alias_checked.to_tlc(&frontend.schemes, &frontend.module_manager)
    });

    // Output TLC if requested (before optimization)
    if let Some(ref tlc_path) = output_tlc {
        fs::write(tlc_path, format!("{}", tlc_transformed.tlc))?;
        if verbose {
            info!("Wrote TLC to {}", tlc_path.display());
        }
    }

    let tlc_optimized = time("tlc_partial_eval", verbose, || tlc_transformed.partial_eval());

    // Fuse consecutive map operations
    let tlc_fused = time("fuse_maps", verbose, || {
        tlc_optimized.normalize_soacs().fuse_maps()
    });

    // Defunctionalize: lift lambdas and flatten SOAC captures
    let tlc_defunc = time("defunctionalize", verbose, || tlc_fused.defunctionalize());

    // Monomorphize polymorphic functions at TLC level
    let tlc_mono = time("tlc_monomorphize", verbose, || tlc_defunc.monomorphize());

    // Buffer-specialize view-array params per-buffer
    let tlc_buf = time("buffer_specialize", verbose, || tlc_mono.buffer_specialize());

    // Inline compiler-generated lambda defs + DCE
    let tlc_inlined = time("inline", verbose, || tlc_buf.inline());

    // Inline small user functions and constants at TLC level
    let tlc_inlined = time("tlc_inline_small", verbose, || tlc_inlined.inline_small());

    // Parallelize SOACs in compute shaders (structural decisions at TLC level)
    let tlc_parallel = time("tlc_parallelize", verbose, || tlc_inlined.parallelize_soacs());

    // Eliminate dead TLC defs
    let tlc_reachable = time("tlc_filter_reachable", verbose, || {
        tlc_parallel.filter_reachable()
    });

    // Transform TLC to SSA via EGraph (GVN + DCE for free)
    let ssa = time("to_egir", verbose, || tlc_reachable.to_egir())?;

    // Dump initial SSA if requested
    if let Some(ref path) = output_init_ssa {
        fs::write(path, wyn_core::ssa::print::format_program(&ssa.ssa))?;
        if verbose {
            info!("Wrote initial SSA to {}", path.display());
        }
    }

    // SSA peephole optimizations
    let optimized = time("ssa_opt", verbose, || ssa.optimize());

    // Dump optimized SSA if requested
    if let Some(ref path) = output_opt_ssa {
        fs::write(path, wyn_core::ssa::print::format_program(&optimized.ssa))?;
        if verbose {
            info!("Wrote optimized SSA to {}", path.display());
        }
    }

    // Lower first-class SOAC instructions to explicit loops
    let soac_lowered = time("soac_lower", verbose, || optimized.lower_soacs());

    match target {
        Target::Spirv => {
            let lowered = time("lower", verbose, || soac_lowered.lower())?;

            // Determine output path
            let output_path = output.unwrap_or_else(|| {
                let mut path = input.clone();
                path.set_extension("spv");
                path
            });

            // Write SPIR-V binary
            let mut file = fs::File::create(&output_path)?;
            let spirv_len = lowered.spirv.len();
            for word in &lowered.spirv {
                file.write_all(&word.to_le_bytes())?;
            }

            // Write pipeline descriptor if there are any pipelines
            if !lowered.pipeline.pipelines.is_empty() {
                let descriptor_path = {
                    let mut p = output_path.clone();
                    p.set_extension("json");
                    p
                };
                let json = serde_json::to_string_pretty(&lowered.pipeline)
                    .map_err(|e| wyn_core::err_spirv!("Failed to serialize pipeline descriptor: {}", e))?;
                fs::write(&descriptor_path, json)?;
                if verbose {
                    info!("Wrote pipeline descriptor to {}", descriptor_path.display());
                }
            }

            if verbose {
                info!("Successfully compiled to {}", output_path.display());
                info!("Generated {} words of SPIR-V", spirv_len);
            }
        }
        Target::Glsl => {
            let glsl_output = time("glsl_lower", verbose, || wyn_core::glsl::lower(&soac_lowered.ssa))?;

            let output_path = output.unwrap_or_else(|| {
                let mut path = input.clone();
                path.set_extension("glsl");
                path
            });

            let mut combined = String::new();
            if let Some(vert) = &glsl_output.vertex {
                combined.push_str("// === VERTEX SHADER ===\n");
                combined.push_str(vert);
                combined.push('\n');
            }
            if let Some(frag) = &glsl_output.fragment {
                combined.push_str("// === FRAGMENT SHADER ===\n");
                combined.push_str(frag);
                combined.push('\n');
            }
            fs::write(&output_path, &combined)?;

            if verbose {
                info!("Successfully compiled to {}", output_path.display());
            }
        }
        Target::Shadertoy => {
            let glsl = time("glsl_lower_shadertoy", verbose, || {
                wyn_core::glsl::lower_shadertoy(&soac_lowered.ssa)
            })?;

            let output_path = output.unwrap_or_else(|| {
                let mut path = input.clone();
                path.set_extension("glsl");
                path
            });

            fs::write(&output_path, &glsl)?;

            if verbose {
                info!("Successfully compiled to {}", output_path.display());
            }
        }
    }

    Ok(())
}

fn check_file(input: PathBuf, output_annotated: Option<PathBuf>, verbose: bool) -> Result<(), DriverError> {
    if verbose {
        info!("Checking {}...", input.display());
    }

    // Read source file
    let source = fs::read_to_string(&input)?;

    // Generate annotated source if requested
    if let Some(ref annotated_path) = output_annotated {
        generate_annotated_source(&source, annotated_path, verbose)?;
    }

    // Type check and alias check, don't generate code
    let mut frontend = wyn_core::FrontEnd::new();
    let parsed = Compiler::parse(&source, &mut frontend.node_counter)?;
    let desugared = parsed.desugar(&mut frontend.node_counter)?;
    let resolved = desugared.resolve(&frontend.module_manager)?;
    let type_checked =
        resolved.fold_ast_constants().type_check(&mut frontend.module_manager, &mut frontend.schemes)?;

    type_checked.print_warnings();

    let alias_checked = type_checked.alias_check()?;
    if alias_checked.has_alias_errors() {
        alias_checked.print_alias_errors();
        return Err(wyn_core::err_alias!("alias checking failed").into());
    }

    if verbose {
        info!("✓ {} is valid", input.display());
    }

    Ok(())
}

fn generate_annotated_source(
    source: &str,
    output_path: &PathBuf,
    verbose: bool,
) -> Result<(), DriverError> {
    // Parse the source to get the AST
    let tokens = lexer::tokenize(source).map_err(|e| wyn_core::err_parse!("{}", e))?;
    let mut node_counter = wyn_core::ast::NodeCounter::new();
    let mut parser = WynParser::new(tokens, &mut node_counter);
    let _program = parser.parse()?;

    // Generate annotated code - temporarily disabled
    // let mut annotator = CodeAnnotator::new();
    // let annotated = annotator.annotate_program(&program);

    // Write annotated source
    // fs::write(output_path, annotated)?;

    // Temporary placeholder
    fs::write(output_path, "// Annotated code generation temporarily disabled\n")?;

    if verbose {
        info!("Generated annotated source: {}", output_path.display());
    }

    Ok(())
}
