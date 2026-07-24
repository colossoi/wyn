use clap::{Parser, Subcommand, ValueEnum};
use log::info;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;
use thiserror::Error;
use wyn_core::{CodegenTarget, Compiler, LoweringProfile, SchedulePolicy};

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

        /// Output file, or an existing directory to write
        /// <input-stem>.<ext> into. Omitted: each output is written
        /// next to its input. A non-directory path is only valid with
        /// a single input.
        #[arg(short, long, value_name = "FILE|DIR")]
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

    // `ConvertError`'s own Display carries the right per-variant label
    // (EGraph/internal prefixes, or a clean user message for InvalidDispatch),
    // so render it directly rather than force one prefix onto every variant.
    #[error("{0}")]
    EgirConversionError(#[from] wyn_core::egir::from_tlc::ConvertError),
}

struct FrontendFile {
    type_checked: wyn_core::TypeChecked,
    module_manager: wyn_core::module_manager::ModuleManager,
}

fn type_check_frontend_file(
    input: &Path,
    reject_holes: bool,
    verbose: bool,
) -> Result<FrontendFile, DriverError> {
    let source = fs::read_to_string(input)?;
    let (mut node_counter, mut module_manager) = time("frontend", verbose, wyn_core::init_compiler)?;
    let parsed = time("parse", verbose, || Compiler::parse(&source, &mut node_counter))?;
    // Resolve `import "..."` against the entry file's directory so
    // user code can split across files. Imports are looked up
    // relative to the file containing the import statement.
    let base_dir = input.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| std::path::PathBuf::from("."));
    let parsed = time("resolve_imports", verbose, || {
        parsed.resolve_imports(&base_dir, &mut node_counter)
    })?;
    // Elaborate inline modules so they're available during resolution.
    let parsed = time("elaborate_modules", verbose, || {
        parsed.elaborate_modules(&mut module_manager, &mut node_counter)
    })?;
    let resolved = time("resolve", verbose, || parsed.resolve(&module_manager))?;
    let ast_folded = time("fold_ast_constants", verbose, || resolved.fold_ast_constants());
    let type_checked = time("type_check", verbose, || {
        ast_folded.type_check(&mut module_manager)
    })?;

    type_checked.print_warnings();
    let type_checked = if reject_holes { type_checked.reject_type_holes()? } else { type_checked };

    Ok(FrontendFile {
        type_checked,
        module_manager,
    })
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
        Err(DriverError::CompilationError(e)) => {
            match e.span() {
                Some(span) if !span.is_generated() => eprintln!("{span}: {e}"),
                _ => eprintln!("{e}"),
            }
            ExitCode::from(1)
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
            // Output handling:
            //   omitted            → each output written next to its input
            //   existing directory → DIR/<input-stem>.<ext> per file
            //   regular file path  → only valid with a single input
            let out_dir: Option<PathBuf> = match &output {
                Some(p) if p.is_dir() => Some(p.clone()),
                Some(p) if inputs.len() > 1 => {
                    eprintln!(
                        "error: --output must be an existing directory when compiling multiple files (got {})",
                        p.display()
                    );
                    std::process::exit(1);
                }
                _ => None,
            };
            for (i, input) in inputs.iter().enumerate() {
                let per_output = if let Some(dir) = &out_dir {
                    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
                    let ext = match target {
                        Target::Spirv => "spv",
                        Target::Wgsl => "wgsl",
                    };
                    Some(dir.join(format!("{stem}.{ext}")))
                } else if inputs.len() == 1 {
                    output.clone()
                } else {
                    None
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

    let FrontendFile {
        type_checked,
        module_manager,
    } = type_check_frontend_file(&input, !fill_holes, verbose)?;

    // Transform to TLC (including prelude code - transformed here for consistent type variables)
    let mut program = time("to_tlc", verbose, || {
        type_checked.to_tlc(&module_manager, fill_holes)
    });

    // Surface any hole-fill errors collected during TLC transform.
    // Aggregate into one multi-line error so the exit-code branch in
    // `main` prints them once with code 2.
    if !program.global_context.fill_hole_errors.is_empty() {
        let msgs: Vec<String> =
            program.global_context.fill_hole_errors.drain(..).map(|e| format!("{e}")).collect();
        return Err(wyn_core::err_type_hole!("{}", msgs.join("\n")).into());
    }

    // Output TLC if requested (before optimization)
    if let Some(ref tlc_path) = output_tlc {
        fs::write(tlc_path, format!("{program}"))?;
        if verbose {
            info!("Wrote TLC to {}", tlc_path.display());
        }
    }

    let program = time("pin_entry_buffers", verbose, || {
        wyn_core::tlc::pin_entry_buffers(program)
    })?;
    let program = time("validate_ownership", verbose, || {
        wyn_core::tlc::validate_ownership(program)
    })?;
    let program = time("tlc_partial_eval", verbose, || {
        wyn_core::tlc::partial_eval(program)
    });
    let program = time("normalize_soacs", verbose, || {
        wyn_core::tlc::normalize_soacs(program)
    });
    let program = time("tlc_monomorphize", verbose, || {
        wyn_core::tlc::monomorphize(program)
    });
    let program = time("tlc_rep_specialize", verbose, || {
        wyn_core::tlc::rep_specialize(program)
    });
    let program = time("tlc_inline_small", verbose, || {
        wyn_core::tlc::inline_small(program)
    });
    let program = time("force_inline_soac_helpers", verbose, || {
        wyn_core::tlc::force_inline_soac_helpers(program)
    });
    let program = time("renormalize_inlined_soa", verbose, || {
        wyn_core::tlc::renormalize_inlined_soa(program)
    });
    let program = time("canonicalize_conditional_producers", verbose, || {
        wyn_core::tlc::canonicalize_conditional_producers(program)
    });
    let program = time("normalize_soacs_to_anf", verbose, || {
        wyn_core::tlc::normalize_soacs_to_anf(program)
    });
    let program = time("expose_runtime_index_producers", verbose, || {
        wyn_core::tlc::float_runtime_index_nested_producers(program)
    });
    let program = time("defunctionalize", verbose, || {
        wyn_core::tlc::defunctionalize(program)
    });
    let program = time("inline", verbose, || {
        wyn_core::tlc::fold_generated_lambdas(program)
    });

    // TLC establishes uniqueness candidates. EGIR owns post-fusion liveness,
    // output routes, resources, and physical entry structure.
    let program = time("apply_ownership", verbose, || {
        wyn_core::tlc::apply_ownership(program)
    });
    // Eliminate dead TLC defs
    let program = time("tlc_filter_reachable", verbose, || {
        wyn_core::tlc::filter_reachable(program)
    });

    // Build raw EGIR, then cross each semantic and physical typestate boundary.
    let program = time("infer_input_slice_bounds", verbose, || {
        wyn_core::tlc::infer_input_slice_bounds(program)
    });
    let raw = time("to_egraph", verbose, || wyn_core::to_egraph(program))?;
    let outputs_realized = time("egir_realize_outputs", verbose, || raw.realize_outputs())?;
    let segmented = time("egir_segment", verbose, || outputs_realized.segment());
    let optimized = time("egir_optimize", verbose, || segmented.optimize());
    let allocated = time("egir_allocate", verbose, || optimized.allocate())?;
    let profile = LoweringProfile::new(
        match target {
            Target::Spirv => CodegenTarget::Spirv,
            Target::Wgsl => CodegenTarget::Wgsl,
        },
        if single_stage { SchedulePolicy::Serial } else { SchedulePolicy::Parallel },
    );
    let planned = time("egir_plan", verbose, || allocated.plan(profile))?;
    let ssa = time("egir_lower_to_ssa", verbose, || planned.lower_to_ssa())?;

    // Dump MIR if requested
    if let Some(ref path) = output_mir {
        fs::write(path, wyn_core::ssa::print::format_program(&ssa))?;
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
            let lowered = time("lower", verbose, || wyn_core::lower_ssa_to_spirv(soac_lowered))?;

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
            let wgsl = time("wgsl_lower", verbose, || {
                wyn_core::lower_ssa_to_wgsl(soac_lowered)
            })?;

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

    let FrontendFile {
        type_checked,
        module_manager,
    } = type_check_frontend_file(&input, true, verbose)?;

    let program = type_checked.to_tlc(&module_manager, false);
    let program = wyn_core::tlc::pin_entry_buffers(program)?;
    wyn_core::tlc::validate_ownership(program)?;

    if verbose {
        info!("✓ {} is valid", input.display());
    }

    Ok(())
}
