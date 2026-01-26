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

        /// Output initial MIR (right after flattening, before optimizations)
        #[arg(long, value_name = "FILE")]
        output_init_mir: Option<PathBuf>,

        /// Output final MIR (right before lowering)
        #[arg(long, value_name = "FILE")]
        output_final_mir: Option<PathBuf>,

        /// Output annotated source code with block IDs and locations
        #[arg(long, value_name = "FILE")]
        output_annotated: Option<PathBuf>,

        /// Output typed lambda calculus representation
        #[arg(long, value_name = "FILE")]
        output_tlc: Option<PathBuf>,

        /// Output shader interface as JSON (bindings, entry points, types)
        #[arg(long, value_name = "FILE")]
        output_interface: Option<PathBuf>,

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
}

fn main() -> Result<(), DriverError> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile {
            input,
            output,
            target,
            output_init_mir,
            output_final_mir,
            output_annotated,
            output_tlc,
            output_interface,
            partial_eval,
            verbose,
        } => {
            compile_file(
                input,
                output,
                target,
                output_init_mir,
                output_final_mir,
                output_annotated,
                output_tlc,
                output_interface,
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
    output_init_mir: Option<PathBuf>,
    output_final_mir: Option<PathBuf>,
    output_annotated: Option<PathBuf>,
    output_tlc: Option<PathBuf>,
    output_interface: Option<PathBuf>,
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
    let builtins = wyn_core::build_builtins(&alias_checked.ast, &frontend.module_manager);

    // Transform to TLC (including prelude code - transformed here for consistent type variables)
    let tlc_transformed = time("to_tlc", verbose, || {
        alias_checked.to_tlc(builtins, &frontend.schemes, &frontend.module_manager)
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

    // Transform TLC to MIR
    let flattened = time("to_mir", verbose, || tlc_defunc.to_mir());

    // Write initial MIR if requested (right after TLC→MIR)
    write_mir_if_requested(&flattened.mir, &output_init_mir, "initial MIR", verbose)?;

    let hoisted = time("hoist_materializations", verbose, || {
        flattened.hoist_materializations()
    });
    let normalized = time("normalize", verbose, || hoisted.normalize());
    let monomorphized = time("monomorphize", verbose, || normalized.monomorphize())?;
    let defaulted = monomorphized.default_address_spaces();
    let reachable = time("filter_reachable", verbose, || defaulted.filter_reachable());
    let lifted = time("lift_bindings", verbose, || reachable.lift_bindings());

    // Write final MIR if requested (right before lowering)
    write_mir_if_requested(&lifted.mir, &output_final_mir, "final MIR", verbose)?;

    // Write interface JSON if requested
    if let Some(ref interface_path) = output_interface {
        let interface = wyn_core::interface::extract_interface(&lifted.mir);
        let json = wyn_core::interface::to_json(&interface).expect("Failed to serialize interface");
        fs::write(interface_path, json)?;
        if verbose {
            info!("Wrote interface to {}", interface_path.display());
        }
    }

    match target {
        Target::Spirv => {
            let lowered = time("lower", verbose, || lifted.lower())?;

            // Determine output path
            let output_path = output.unwrap_or_else(|| {
                let mut path = input.clone();
                path.set_extension("spv");
                path
            });

            // Write SPIR-V binary
            let mut file = fs::File::create(&output_path)?;
            let spirv_len = lowered.spirv.len();
            for word in lowered.spirv {
                file.write_all(&word.to_le_bytes())?;
            }

            if verbose {
                info!("Successfully compiled to {}", output_path.display());
                info!("Generated {} words of SPIR-V", spirv_len);
            }
        }
        Target::Glsl => {
            let lowered = time("lower_glsl", verbose, || lifted.lower_glsl())?;

            // Determine base output path
            let base_path = output.unwrap_or_else(|| input.clone());

            // Write vertex shader if present
            if let Some(ref vert_src) = lowered.glsl.vertex {
                let mut vert_path = base_path.clone();
                vert_path.set_extension("vert.glsl");
                fs::write(&vert_path, vert_src)?;
                if verbose {
                    info!("Wrote vertex shader to {}", vert_path.display());
                }
            }

            // Write fragment shader if present
            if let Some(ref frag_src) = lowered.glsl.fragment {
                let mut frag_path = base_path.clone();
                frag_path.set_extension("frag.glsl");
                fs::write(&frag_path, frag_src)?;
                if verbose {
                    info!("Wrote fragment shader to {}", frag_path.display());
                }
            }

            if verbose && lowered.glsl.vertex.is_none() && lowered.glsl.fragment.is_none() {
                info!("No entry points found - no shaders generated");
            }
        }
        Target::Shadertoy => {
            let shader_src = time("lower_shadertoy", verbose, || lifted.lower_shadertoy())?;

            // Determine output path
            let output_path = output.unwrap_or_else(|| {
                let mut path = input.clone();
                path.set_extension("shadertoy.glsl");
                path
            });

            fs::write(&output_path, &shader_src)?;
            if verbose {
                info!("Wrote Shadertoy shader to {}", output_path.display());
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

fn write_mir_if_requested(
    mir: &wyn_core::mir::Program,
    output_mir: &Option<PathBuf>,
    description: &str,
    verbose: bool,
) -> Result<(), DriverError> {
    if let Some(ref mir_path) = output_mir {
        fs::write(mir_path, format!("{}", mir))?;
        if verbose {
            info!("Wrote {} to {}", description, mir_path.display());
        }
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
