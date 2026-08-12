use clap::{Parser, Subcommand, ValueEnum};
use log::info;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;
use thiserror::Error;
use wyn_core::{CodegenTarget, LoweringProfile, SchedulePolicy};

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

        /// Enable software emulation of unsigned 64-bit integers in WGSL.
        /// This option is invalid for the SPIR-V target.
        #[arg(long)]
        wgsl_emulate_u64: bool,

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

    #[error("Pipeline descriptor serialization error: {0}")]
    DescriptorSerialization(#[from] serde_json::Error),

    // `ConvertError`'s own Display carries the right per-variant label
    // (EGraph/internal prefixes, or a clean user message for InvalidDispatch),
    // so render it directly rather than force one prefix onto every variant.
    #[error("{0}")]
    EgirConversionError(#[from] wyn_core::egir::from_tlc::ConvertError),

    #[error("invalid command-line option: {0}")]
    InvalidOption(String),
}

struct FrontendFile {
    program: wyn_core::ast_type_holes::HolesResolved,
}

fn type_check_frontend_file(
    input: &Path,
    reject_holes: bool,
    verbose: bool,
) -> Result<FrontendFile, DriverError> {
    let source = fs::read_to_string(input)?;
    let (node_counter, module_manager) = time("frontend", verbose, wyn_core::init_compiler)?;
    let program = time("parse", verbose, || {
        wyn_core::parser::parse(&source, node_counter, module_manager)
    })?;
    // Resolve `import "..."` against the entry file's directory so
    // user code can split across files. Imports are looked up
    // relative to the file containing the import statement.
    let base_dir = input.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| std::path::PathBuf::from("."));
    let program = time("resolve_imports", verbose, || {
        wyn_core::resolve_imports::resolve_imports(program, &base_dir)
    })?;
    let program = time("elaborate_modules", verbose, || {
        wyn_core::elaborate_modules::elaborate_modules(program)
    })?;
    let program = time("resolve_names", verbose, || {
        wyn_core::name_resolution::resolve_names(program)
    });
    let program = time("resolve_resources", verbose, || {
        wyn_core::resolve_resources::resolve_resources(program)
    })?;
    let program = time("fold_ast_constants", verbose, || {
        wyn_core::ast_const_fold::fold_constants(program)
    });
    let program = time("resolve_type_placeholders", verbose, || {
        wyn_core::resolve_placeholders::resolve_type_placeholders(program)
    });
    let program = time("resolve_opens", verbose, || {
        wyn_core::resolve_opens::resolve_opens(program)
    })?;
    let program = time("type_check", verbose, || {
        wyn_core::types::run::type_check(program)
    })?;

    for warning in &program.global_context.warnings {
        eprintln!(
            "{}: warning: {}",
            warning.span(),
            warning.message(&wyn_core::types::format_type)
        );
    }
    let program = if reject_holes {
        wyn_core::ast_type_holes::reject_type_holes(program)?
    } else {
        wyn_core::ast_type_holes::fill_type_holes(program)?
    };

    Ok(FrontendFile { program })
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
            wgsl_emulate_u64,
            fill_holes,
            verbose,
        } => {
            if wgsl_emulate_u64 && !matches!(target, Target::Wgsl) {
                return Err(DriverError::InvalidOption(
                    "--wgsl-emulate-u64 requires --target wgsl".to_string(),
                ));
            }
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
                    wgsl_emulate_u64,
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
    wgsl_emulate_u64: bool,
    fill_holes: bool,
    verbose: bool,
) -> Result<(), DriverError> {
    if verbose {
        info!("Compiling {}...", input.display());
    }

    // Wall-clock start for the always-printed timing summary below.
    let compile_start = Instant::now();

    let FrontendFile { program } = type_check_frontend_file(&input, !fill_holes, verbose)?;

    let program = time("to_tlc", verbose, || wyn_core::tlc::lower_from_ast(program))?;

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
    let program = time("to_egraph", verbose, || wyn_core::to_egraph(program))?;
    let program = time("egir_realize_outputs", verbose, || {
        wyn_core::egir::realize_outputs(program)
    })?;
    let program = time("egir_reify_soacs", verbose, || {
        wyn_core::egir::reify_soacs(program)
    });
    let program = time("egir_optimize_semantics", verbose, || {
        wyn_core::egir::optimize_semantics(program)
    });
    let program = time("egir_plan_logical_resources", verbose, || {
        wyn_core::egir::plan_logical_resources(program)
    })?;
    let profile = LoweringProfile::new(
        match target {
            Target::Spirv => CodegenTarget::Spirv,
            Target::Wgsl => CodegenTarget::Wgsl,
        },
        if single_stage { SchedulePolicy::Serial } else { SchedulePolicy::Parallel },
    );
    let program = time("egir_plan", verbose, || wyn_core::egir::plan(program, profile))?;
    let ssa = time("egir_lower_to_ssa", verbose, || {
        wyn_core::lower_egir_to_ssa(program)
    })?;

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

    let pipeline = match target {
        Target::Spirv => {
            let lowered = time("lower", verbose, || wyn_core::lower_ssa_to_spirv(soac_lowered))?;

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

            lowered.pipeline
        }
        Target::Wgsl => {
            let options = if wgsl_emulate_u64 {
                wyn_core::wgsl::WgslOptions::U64_EMULATION
            } else {
                wyn_core::wgsl::WgslOptions::default()
            };
            let lowered = time("wgsl_lower", verbose, || {
                wyn_core::lower_ssa_to_wgsl_with_pipeline_and_options(soac_lowered, options)
            })?;

            fs::write(&output_path, &lowered.wgsl)?;

            if verbose {
                info!("Successfully compiled to {}", output_path.display());
            }

            lowered.pipeline
        }
    };

    // Both executable backends share the same planned runtime contract.
    if !pipeline.pipelines.is_empty() {
        let descriptor_path = {
            let mut p = output_path.clone();
            p.set_extension("json");
            p
        };
        fs::write(&descriptor_path, serde_json::to_string_pretty(&pipeline)?)?;
        if verbose {
            info!("Wrote pipeline descriptor to {}", descriptor_path.display());
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

    let FrontendFile { program } = type_check_frontend_file(&input, true, verbose)?;
    let program = wyn_core::tlc::lower_from_ast(program)?;
    let program = wyn_core::tlc::pin_entry_buffers(program)?;
    wyn_core::tlc::validate_ownership(program)?;

    if verbose {
        info!("✓ {} is valid", input.display());
    }

    Ok(())
}
