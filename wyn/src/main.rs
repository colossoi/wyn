use clap::{Parser, Subcommand, ValueEnum};
use log::info;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;
use thiserror::Error;
use wyn_core::pipeline_descriptor::PipelineDescriptor;
use wyn_core::{
    CodegenTarget, CompilationFailure, CompilerOptions, LoadModulesError, LoweringProfile, ParsedModules,
    PipelineTopologyPolicy, SchedulePolicy,
};
use wyn_module_graph::{PackagePlan, SourceGraph};
use wyn_package_manager::{
    find_build_input, prepare_package, prepare_standalone, BuildInput, PreparationError,
};

/// Target output format
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum Target {
    /// SPIR-V binary (default)
    #[default]
    Spirv,
    /// WGSL source code (WebGPU shading language)
    Wgsl,
}

impl Target {
    const fn extension(self) -> &'static str {
        match self {
            Self::Spirv => "spv",
            Self::Wgsl => "wgsl",
        }
    }
}

struct CompileOptions {
    target: Target,
    direct: bool,
    wgsl_emulate_u64: bool,
    fill_holes: bool,
    output_tlc: Option<PathBuf>,
    output_mir: Option<PathBuf>,
    verbose: bool,
}

struct Compilation {
    code: CompiledCode,
    pipeline: PipelineDescriptor,
    auxiliary: Vec<TextArtifact>,
}

enum CompiledCode {
    Spirv(Vec<u32>),
    Wgsl(String),
}

struct TextArtifact {
    path: PathBuf,
    contents: String,
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
    /// Build a source program or local package as SPIR-V or WGSL
    Build {
        /// Input source file or package directory
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Output file, or an existing directory in which to write it
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

        /// Enable the unified graphics pipeline vocabulary.
        #[arg(long)]
        graphics: bool,

        /// Emit the authored pipeline directly, without compiler-created
        /// prepasses, entry points, or host-visible resources.
        #[arg(long)]
        direct: bool,

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

    /// Validate a source file or local package without generating output
    Check {
        /// Input source file or package directory
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Enable the unified graphics pipeline vocabulary.
        #[arg(long)]
        graphics: bool,

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

    #[error(transparent)]
    LoadModules(#[from] LoadModulesError),

    #[error("{0}")]
    Compilation(#[from] CompilationFailure),

    #[error("{0}")]
    PackagePreparation(#[from] PreparationError),

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

fn retain_source<T>(
    result: wyn_core::error::Result<T>,
    source_graph: &SourceGraph,
) -> Result<T, DriverError> {
    result.map_err(|error| CompilationFailure::new(error, source_graph.clone()).into())
}

fn normalize_input(input: &Path) -> Result<PathBuf, DriverError> {
    let input = input.canonicalize()?;
    if input.is_dir() || input.extension().and_then(|extension| extension.to_str()) == Some("wyn") {
        return Ok(input);
    }
    Err(DriverError::InvalidOption(format!(
        "input `{}` must be a package directory or `.wyn` source file",
        input.display()
    )))
}

fn output_path(input: &Path, output: Option<PathBuf>, target: Target) -> Result<PathBuf, DriverError> {
    match output {
        Some(directory) if directory.is_dir() => {
            let Some(stem) = input.file_stem().and_then(|stem| stem.to_str()) else {
                return Err(DriverError::InvalidOption(format!(
                    "input `{}` has no UTF-8 file stem",
                    input.display()
                )));
            };
            Ok(directory.join(format!("{stem}.{}", target.extension())))
        }
        Some(path) => Ok(path),
        None => {
            let mut path = input.to_path_buf();
            path.set_extension(target.extension());
            Ok(path)
        }
    }
}

fn type_check_input(
    input: &Path,
    reject_holes: bool,
    graphics: bool,
    verbose: bool,
) -> Result<wyn_core::ast_type_holes::HolesResolved, DriverError> {
    let input = normalize_input(input)?;
    let package_plan = match find_build_input(&input)? {
        BuildInput::Package { root, root_module } => prepare_package(root, root_module)?,
        BuildInput::Standalone(source) => prepare_standalone(source)?,
    };
    type_check_package_plan(package_plan, reject_holes, graphics, verbose)
}

fn type_check_package_plan(
    plan: PackagePlan,
    reject_holes: bool,
    graphics: bool,
    verbose: bool,
) -> Result<wyn_core::ast_type_holes::HolesResolved, DriverError> {
    let modules = time("load_modules", verbose, || {
        ParsedModules::load(plan, CompilerOptions { graphics })
    })?;
    finish_type_check(modules, reject_holes, verbose)
}

fn finish_type_check(
    modules: ParsedModules,
    reject_holes: bool,
    verbose: bool,
) -> Result<wyn_core::ast_type_holes::HolesResolved, DriverError> {
    let program = time("type_check", verbose, || modules.type_check())?;

    for warning in &program.global_context.warnings {
        let message = warning.message(&wyn_core::types::format_type);
        match program.source_graph().display_location(*warning.span()) {
            Ok(location) => eprintln!("{location}: warning: {message}"),
            Err(_) => eprintln!("warning: {message}"),
        }
    }
    let program = if reject_holes {
        wyn_core::ast_type_holes::reject_type_holes(program)?
    } else {
        wyn_core::ast_type_holes::fill_type_holes(program)?
    };

    Ok(program)
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
        Err(DriverError::Compilation(failure)) => {
            eprintln!("{failure}");
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
        Commands::Build {
            input,
            output,
            target,
            output_tlc,
            output_mir,
            graphics,
            direct,
            wgsl_emulate_u64,
            fill_holes,
            verbose,
        } => build(
            input,
            output,
            target,
            output_tlc,
            output_mir,
            graphics,
            direct,
            wgsl_emulate_u64,
            fill_holes,
            verbose,
        ),
        Commands::Check {
            input,
            graphics,
            verbose,
        } => check(input, graphics, verbose),
    }
}

fn build(
    input: PathBuf,
    output: Option<PathBuf>,
    target: Target,
    output_tlc: Option<PathBuf>,
    output_mir: Option<PathBuf>,
    graphics: bool,
    direct: bool,
    wgsl_emulate_u64: bool,
    fill_holes: bool,
    verbose: bool,
) -> Result<(), DriverError> {
    if wgsl_emulate_u64 && !matches!(target, Target::Wgsl) {
        return Err(DriverError::InvalidOption(
            "--wgsl-emulate-u64 requires --target wgsl".to_string(),
        ));
    }

    if verbose {
        info!("Building {}...", input.display());
    }

    // Wall-clock start for the always-printed timing summary below.
    let build_start = Instant::now();

    let normalized_input = normalize_input(&input)?;
    let output_path = output_path(&normalized_input, output, target)?;
    let package_plan = match find_build_input(&normalized_input)? {
        BuildInput::Package { root, root_module } => prepare_package(root, root_module)?,
        BuildInput::Standalone(source) => prepare_standalone(source)?,
    };
    let parsed_modules = time("load_modules", verbose, || {
        ParsedModules::load(package_plan, CompilerOptions { graphics })
    })?;
    let compilation = compile(
        parsed_modules,
        CompileOptions {
            target,
            direct,
            wgsl_emulate_u64,
            fill_holes,
            output_tlc,
            output_mir,
            verbose,
        },
    )?;
    write_artifacts(&output_path, compilation, verbose)?;

    // Always-on wall-clock summary (per-pass breakdown is available via
    // `-v`). Printed to stderr so it doesn't pollute any piped output.
    eprintln!(
        "Built {} → {} in {:.2}s",
        input.display(),
        output_path.display(),
        build_start.elapsed().as_secs_f64()
    );

    Ok(())
}

fn compile(modules: ParsedModules, options: CompileOptions) -> Result<Compilation, DriverError> {
    let CompileOptions {
        target,
        direct,
        wgsl_emulate_u64,
        fill_holes,
        output_tlc,
        output_mir,
        verbose,
    } = options;
    let program = finish_type_check(modules, !fill_holes, verbose)?;
    let source_graph = program.source_graph().clone();

    let program = retain_source(
        time("to_tlc", verbose, || wyn_core::tlc::lower_from_ast(program)),
        &source_graph,
    )?;
    let mut auxiliary = Vec::new();
    if let Some(path) = output_tlc {
        auxiliary.push(TextArtifact {
            path,
            contents: format!("{program}"),
        });
    }

    let program = retain_source(
        time("pin_entry_buffers", verbose, || {
            wyn_core::tlc::pin_entry_buffers(program)
        }),
        &source_graph,
    )?;
    let program = retain_source(
        time("validate_ownership", verbose, || {
            wyn_core::tlc::validate_ownership(program)
        }),
        &source_graph,
    )?;
    let program = time("tlc_partial_eval", verbose, || {
        wyn_core::tlc::partial_eval(program)
    });
    let program = time("normalize_soacs", verbose, || {
        wyn_core::tlc::normalize_soacs(program)
    });
    let program = retain_source(
        time("tlc_monomorphize", verbose, || {
            wyn_core::tlc::monomorphize(program)
        }),
        &source_graph,
    )?;
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
    let program = time("egir_reify_soacs", verbose, || {
        wyn_core::egir::reify_soacs(program)
    });
    let program = retain_source(
        time("egir_optimize_semantic_operations", verbose, || {
            wyn_core::egir::optimize_semantic_operations(program)
        }),
        &source_graph,
    )?;
    let profile = if direct {
        LoweringProfile::with_topology(
            match target {
                Target::Spirv => CodegenTarget::Spirv,
                Target::Wgsl => CodegenTarget::Wgsl,
            },
            SchedulePolicy::Serial,
            PipelineTopologyPolicy::AuthoredOnly,
        )
    } else {
        LoweringProfile::new(
            match target {
                Target::Spirv => CodegenTarget::Spirv,
                Target::Wgsl => CodegenTarget::Wgsl,
            },
            SchedulePolicy::Parallel,
        )
    };
    let program = time("egir_apply_pipeline_topology_policy", verbose, || {
        wyn_core::egir::apply_pipeline_topology_policy(program, profile.topology)
    });
    let program = time("egir_plan_logical_resources", verbose, || {
        wyn_core::egir::plan_logical_resources_with_policy(program, profile.topology)
    })?;
    let program = time("egir_plan", verbose, || wyn_core::egir::plan(program, profile))?;
    let ssa = time("egir_lower_to_ssa", verbose, || {
        wyn_core::lower_egir_to_ssa(program)
    })?;

    if let Some(path) = output_mir {
        auxiliary.push(TextArtifact {
            path,
            contents: wyn_core::ssa::print::format_program(&ssa),
        });
    }

    let soac_lowered = ssa;

    let (code, pipeline) = match target {
        Target::Spirv => {
            let lowered = retain_source(
                time("lower", verbose, || wyn_core::lower_ssa_to_spirv(soac_lowered)),
                &source_graph,
            )?;
            (CompiledCode::Spirv(lowered.spirv), lowered.pipeline)
        }
        Target::Wgsl => {
            let options = if wgsl_emulate_u64 {
                wyn_core::wgsl::WgslOptions::U64_EMULATION
            } else {
                wyn_core::wgsl::WgslOptions::default()
            };
            let lowered = retain_source(
                time("wgsl_lower", verbose, || {
                    wyn_core::lower_ssa_to_wgsl_with_pipeline_and_options(soac_lowered, options)
                }),
                &source_graph,
            )?;

            (CompiledCode::Wgsl(lowered.wgsl), lowered.pipeline)
        }
    };

    Ok(Compilation {
        code,
        pipeline,
        auxiliary,
    })
}

fn write_artifacts(output_path: &Path, compilation: Compilation, verbose: bool) -> Result<(), DriverError> {
    let Compilation {
        code,
        pipeline,
        auxiliary,
    } = compilation;

    match code {
        CompiledCode::Spirv(words) => {
            let mut file = fs::File::create(output_path)?;
            for word in &words {
                file.write_all(&word.to_le_bytes())?;
            }
            if verbose {
                info!(
                    "Wrote {} words of SPIR-V to {}",
                    words.len(),
                    output_path.display()
                );
            }
        }
        CompiledCode::Wgsl(source) => {
            fs::write(output_path, source)?;
            if verbose {
                info!("Wrote WGSL to {}", output_path.display());
            }
        }
    }

    for artifact in auxiliary {
        fs::write(&artifact.path, artifact.contents)?;
        if verbose {
            info!("Wrote compiler output to {}", artifact.path.display());
        }
    }

    // Both executable backends share the same planned runtime contract.
    if !pipeline.pipelines.is_empty() {
        let mut descriptor_path = output_path.to_owned();
        descriptor_path.set_extension("json");
        fs::write(&descriptor_path, serde_json::to_string_pretty(&pipeline)?)?;
        if verbose {
            info!("Wrote pipeline descriptor to {}", descriptor_path.display());
        }
    }

    Ok(())
}

fn check(input: PathBuf, graphics: bool, verbose: bool) -> Result<(), DriverError> {
    if verbose {
        info!("Checking {}...", input.display());
    }

    let program = type_check_input(&input, true, graphics, verbose)?;
    let source_graph = program.source_graph().clone();
    let program = retain_source(wyn_core::tlc::lower_from_ast(program), &source_graph)?;
    let program = retain_source(wyn_core::tlc::pin_entry_buffers(program), &source_graph)?;
    retain_source(wyn_core::tlc::validate_ownership(program), &source_graph)?;

    if verbose {
        info!("✓ {} is valid", input.display());
    }

    Ok(())
}
