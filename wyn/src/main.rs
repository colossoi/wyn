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

        /// Enable partial evaluation (compile-time function inlining and loop unrolling)
        #[arg(long)]
        partial_eval: bool,

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
}

fn main() -> Result<(), DriverError> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile {
            input,
            output,
            target,
            output_annotated,
            output_tlc,
            partial_eval,
            verbose,
        } => {
            compile_file(
                input,
                output,
                target,
                output_annotated,
                output_tlc,
                partial_eval,
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
    partial_eval: bool,
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

    // Build builtins set for lambda lifting (names that should not be captured)
    let known_defs = wyn_core::build_known_defs(&alias_checked.ast, &frontend.module_manager);

    // Transform to TLC (including prelude code - transformed here for consistent type variables)
    let tlc_transformed = time("to_tlc", verbose, || {
        alias_checked.to_tlc(known_defs, &frontend.schemes, &frontend.module_manager)
    });

    // Output TLC if requested (before optimization)
    if let Some(ref tlc_path) = output_tlc {
        fs::write(tlc_path, format!("{}", tlc_transformed.tlc))?;
        if verbose {
            info!("Wrote TLC to {}", tlc_path.display());
        }
    }

    // Apply TLC partial evaluation if enabled
    let tlc_optimized = if partial_eval {
        time("tlc_partial_eval", verbose, || tlc_transformed.partial_eval())
    } else {
        tlc_transformed.skip_partial_eval()
    };

    // Defunctionalize: lift lambdas and flatten SOAC captures
    let tlc_defunc = time("defunctionalize", verbose, || tlc_optimized.defunctionalize());

    // Monomorphize polymorphic functions at TLC level
    let tlc_mono = time("tlc_monomorphize", verbose, || tlc_defunc.monomorphize());

    // Transform TLC to SSA
    let ssa = time("to_ssa", verbose, || tlc_mono.to_ssa())?;

    // Parallelize SOACs in compute shaders
    let parallelized = time("parallelize_soacs", verbose, || ssa.parallelize_soacs());

    match target {
        Target::Spirv => {
            let lowered = time("lower", verbose, || parallelized.lower())?;

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

            if verbose {
                info!("Successfully compiled to {}", output_path.display());
                info!("Generated {} words of SPIR-V", spirv_len);
            }
        }
        Target::Glsl | Target::Shadertoy => {
            return Err(
                wyn_core::err_spirv!("GLSL/Shadertoy targets not yet supported with SSA pipeline").into(),
            );
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
