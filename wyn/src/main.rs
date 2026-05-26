use clap::{Parser, Subcommand, ValueEnum};
use log::info;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;
use thiserror::Error;
use wyn_core::Compiler;

/// Target output format
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum Target {
    /// SPIR-V binary (default)
    #[default]
    Spirv,
    /// WGSL source code (WebGPU shading language)
    Wgsl,
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
    /// Compile one or more source files to SPIR-V or WGSL
    Compile {
        /// Input source file(s). When multiple files are given, each
        /// is compiled in turn within a single process — useful for
        /// batch compilation and profiling.
        #[arg(value_name = "FILE", required = true)]
        inputs: Vec<PathBuf>,

        /// Output file (only valid with a single input; multi-input
        /// runs auto-derive each output's name from its input).
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Target output format
        #[arg(short, long, default_value = "spirv")]
        target: Target,

        /// Output typed lambda calculus representation
        #[arg(long, value_name = "FILE")]
        output_tlc: Option<PathBuf>,

        /// Output MIR (SSA post-EGIR, pre-backend-lowering)
        #[arg(long, value_name = "FILE")]
        output_mir: Option<PathBuf>,

        /// Disable multi-stage SOAC parallelization. Compute SOACs emit
        /// as a single sequential loop instead of chunk/combine phases;
        /// graphical-entry SOACs are not lifted to pre-pass kernels.
        #[arg(long)]
        single_stage: bool,

        /// Treat any `???` type hole as a default value of its inferred
        /// type and continue compilation. Default: holes are a hard
        /// error (exit code 2). Default fills: numeric 0, bool false,
        /// tuples/vectors/arrays filled componentwise, unit `()`.
        /// Types that can't be default-filled (unresolved type variables,
        /// function types, view/virtual arrays, records) still produce
        /// an error.
        #[arg(long)]
        fill_holes: bool,

        /// Print verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Validate a source file without generating output
    Check {
        /// Input source file
        #[arg(value_name = "FILE")]
        input: PathBuf,

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

    #[error("EGraph conversion error: {0}")]
    EgirConversionError(#[from] wyn_core::egir::from_tlc::ConvertError),
}

fn main() -> ExitCode {
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

    // Exit-code convention:
    //   0 — success
    //   1 — generic compilation failure (parse, type, alias, backend)
    //   2 — program contains unresolved `???` type holes
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(DriverError::CompilationError(wyn_core::error::CompilerError::TypeHole(msg))) => {
            eprintln!("{msg}");
            ExitCode::from(2)
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<(), DriverError> {
    match cli.command {
        Commands::Compile {
            inputs,
            output,
            target,
            output_tlc,
            output_mir,
            single_stage,
            fill_holes,
            verbose,
        } => {
            // Output handling for multi-input:
            //   omitted     → each output written next to its input
            //   directory   → DIR/<input-stem>.<ext> per file
            //   regular file → only valid with a single input
            let out_dir: Option<PathBuf> = match (inputs.len(), &output) {
                (n, Some(p)) if n > 1 => {
                    if p.is_dir() {
                        Some(p.clone())
                    } else {
                        eprintln!(
                            "error: --output must be an existing directory when compiling multiple files (got {})",
                            p.display()
                        );
                        std::process::exit(1);
                    }
                }
                _ => None,
            };
            for (i, input) in inputs.iter().enumerate() {
                let per_output = match (inputs.len(), &out_dir, &output) {
                    (1, _, opt) => opt.clone(),
                    (_, Some(dir), _) => {
                        let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
                        let ext = match target {
                            Target::Spirv => "spv",
                            Target::Wgsl => "wgsl",
                        };
                        Some(dir.join(format!("{stem}.{ext}")))
                    }
                    _ => None,
                };
                if verbose && inputs.len() > 1 {
                    eprintln!("[{}/{}] {}", i + 1, inputs.len(), input.display());
                }
                compile_file(
                    input.clone(),
                    per_output,
                    target,
                    output_tlc.clone(),
                    output_mir.clone(),
                    single_stage,
                    fill_holes,
                    verbose,
                )?;
            }
        }
        Commands::Check { input, verbose } => {
            check_file(input, verbose)?;
        }
    }

    Ok(())
}

fn compile_file(
    input: PathBuf,
    output: Option<PathBuf>,
    target: Target,
    output_tlc: Option<PathBuf>,
    output_mir: Option<PathBuf>,
    single_stage: bool,
    fill_holes: bool,
    verbose: bool,
) -> Result<(), DriverError> {
    if verbose {
        info!("Compiling {}...", input.display());
    }

    // Wall-clock start for the always-printed timing summary below.
    let compile_start = Instant::now();

    // Read source file
    let source = fs::read_to_string(&input)?;

    let (mut node_counter, mut module_manager) = time("frontend", verbose, wyn_core::init_compiler);
    let parsed = time("parse", verbose, || Compiler::parse(&source, &mut node_counter))?;
    // Resolve `import "..."` against the entry file's directory so
    // user code can split across files. Imports are looked up
    // relative to the file containing the import statement.
    let base_dir = input.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| std::path::PathBuf::from("."));
    let parsed = time("resolve_imports", verbose, || {
        parsed.resolve_imports(&base_dir, &mut node_counter)
    })?;
    // Elaborate inline modules so they're available during resolution
    let parsed = time("elaborate_modules", verbose, || {
        parsed.elaborate_modules(&mut module_manager, &mut node_counter)
    })?;
    let resolved = time("resolve", verbose, || parsed.resolve(&module_manager))?;
    let ast_folded = time("fold_ast_constants", verbose, || resolved.fold_ast_constants());
    let type_checked = time("type_check", verbose, || {
        ast_folded.type_check(&mut module_manager)
    })?;

    type_checked.print_warnings();
    let type_checked = if fill_holes {
        // `--fill-holes`: skip the hole-gate; TLC will default-fill.
        type_checked
    } else {
        type_checked.reject_type_holes()?
    };

    // Transform to TLC (including prelude code - transformed here for consistent type variables)
    let tlc_transformed = time("to_tlc", verbose, || {
        type_checked.to_tlc(&module_manager, fill_holes)
    });

    // Surface any hole-fill errors collected during TLC transform.
    // Aggregate into one multi-line error so the exit-code branch in
    // `main` prints them once with code 2. Access the inner field
    // directly (not through the Deref) so we can take ownership.
    if !tlc_transformed.0.fill_hole_errors.is_empty() {
        let mut tlc_transformed = tlc_transformed;
        let msgs: Vec<String> =
            tlc_transformed.0.fill_hole_errors.drain(..).map(|e| format!("{e}")).collect();
        return Err(wyn_core::err_type_hole!("{}", msgs.join("\n")).into());
    }

    // Output TLC if requested (before optimization)
    if let Some(ref tlc_path) = output_tlc {
        fs::write(tlc_path, format!("{}", tlc_transformed.tlc))?;
        if verbose {
            info!("Wrote TLC to {}", tlc_path.display());
        }
    }

    let tlc_optimized = time("tlc_partial_eval", verbose, || tlc_transformed.partial_eval());

    // SOA + SOAC normalize, map fusion, then ownership-driven rewrites.
    // Each is its own pipeline step so timing breaks down per pass in
    // verbose mode.
    let tlc_normed = time("normalize_soacs", verbose, || tlc_optimized.normalize_soacs());
    let tlc_fused = time("fuse_maps", verbose, || tlc_normed.fuse_maps());
    let tlc_owned = time("apply_ownership", verbose, || tlc_fused.apply_ownership())?;

    // Defunctionalize: lift lambdas and flatten SOAC captures
    let tlc_defunc = time("defunctionalize", verbose, || tlc_owned.defunctionalize());

    // Monomorphize polymorphic functions at TLC level
    let tlc_mono = time("tlc_monomorphize", verbose, || tlc_defunc.monomorphize());

    // Buffer-specialize view-array params per-buffer
    let tlc_buf = time("buffer_specialize", verbose, || tlc_mono.buffer_specialize());

    // Inline compiler-generated lambda defs + DCE
    let tlc_folded = time("inline", verbose, || tlc_buf.fold_generated_lambdas());

    // Inline small user functions and constants at TLC level
    let tlc_inlined = time("tlc_inline_small", verbose, || tlc_folded.inline_small());

    // Parallelize SOACs in compute shaders (structural decisions at TLC
    // level). `--single-stage` disables this pass entirely; compute SOACs
    // collapse to sequential loops and graphical entries are not
    // restructured.
    let tlc_parallel = time("tlc_parallelize", verbose, || {
        tlc_inlined.parallelize_soacs(single_stage)
    })?;

    // Eliminate dead TLC defs
    let tlc_reachable = time("tlc_filter_reachable", verbose, || {
        tlc_parallel.filter_reachable()
    });

    // Build the raw EGIR program, then chain the passes.
    let raw = time("to_egraph", verbose, || tlc_reachable.to_egraph())?;
    let expanded = time("expand_soacs", verbose, || raw.expand_soacs(true));
    let ssa = time("egir_passes_full", verbose, || {
        expanded.materialize().optimize_skeleton().elaborate()
    });

    // Dump MIR if requested
    if let Some(ref path) = output_mir {
        fs::write(path, wyn_core::ssa::print::format_program(&ssa.ssa))?;
        if verbose {
            info!("Wrote MIR to {}", path.display());
        }
    }

    let soac_lowered = ssa;

    // Output path (default: input name with the target's extension).
    let output_path = output.unwrap_or_else(|| {
        let mut path = input.clone();
        path.set_extension(match target {
            Target::Spirv => "spv",
            Target::Wgsl => "wgsl",
        });
        path
    });

    match target {
        Target::Spirv => {
            let lowered = time("lower", verbose, || soac_lowered.lower())?;

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
        Target::Wgsl => {
            let wgsl = time("wgsl_lower", verbose, || wyn_core::wgsl::lower(&soac_lowered.ssa))?;

            fs::write(&output_path, &wgsl)?;

            if verbose {
                info!("Successfully compiled to {}", output_path.display());
            }
        }
    }

    // Always-on wall-clock summary (per-pass breakdown is available via
    // `-v`). Printed to stderr so it doesn't pollute any piped output.
    eprintln!(
        "Compiled {} → {} in {:.2}s",
        input.display(),
        output_path.display(),
        compile_start.elapsed().as_secs_f64()
    );

    Ok(())
}

fn check_file(input: PathBuf, verbose: bool) -> Result<(), DriverError> {
    if verbose {
        info!("Checking {}...", input.display());
    }

    // Read source file
    let source = fs::read_to_string(&input)?;

    // Type check and alias check, don't generate code
    let (mut node_counter, mut module_manager) = wyn_core::init_compiler();
    let parsed = Compiler::parse(&source, &mut node_counter)?;
    let base_dir = input.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| std::path::PathBuf::from("."));
    let parsed = parsed.resolve_imports(&base_dir, &mut node_counter)?;
    let resolved = parsed.resolve(&module_manager)?;
    let type_checked = resolved.fold_ast_constants().type_check(&mut module_manager)?;

    type_checked.print_warnings();

    let tlc_after_norm = type_checked.to_tlc(&module_manager, false).partial_eval().normalize_soacs();
    wyn_core::tlc::ownership::check(&tlc_after_norm.0.tlc)?;

    if verbose {
        info!("✓ {} is valid", input.display());
    }

    Ok(())
}
