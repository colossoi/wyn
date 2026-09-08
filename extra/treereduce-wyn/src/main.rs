use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{ArgAction, Parser, ValueEnum};
use regex::Regex;
use tree_sitter::{Language, Node, Parser as TreeSitterParser, Tree};
use treereduce::{Check, CmdCheck, CmdCheckState, Config, NodeTypes, Original};

const DEFAULT_PASSES: usize = 2;
const DEFAULT_MIN_REDUCTION: usize = 2;

const INTEGER_EXPRESSIONS: &[&str] = &["0", "1"];
const FLOAT_EXPRESSIONS: &[&str] = &["0.0", "1.0"];
const BOOLEAN_EXPRESSIONS: &[&str] = &["false", "true"];
const HOLE: &[&str] = &["???"];
const WILDCARD: &[&str] = &["_"];

static UNUSED_WARNING_DIAGNOSTIC: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)^warning: unused (parameter|binding|loop variable|pattern binding|definition|external declaration) `([^`\r\n]+)`[^\r\n]*\r?\n\s*-->\s+.*:(\d+):(\d+)\r?$",
    )
    .expect("unused-warning diagnostic regex should compile")
});

const COMPOSITE_EXPRESSION_KINDS: &[&str] = &[
    "call_expression",
    "let_expression",
    "if_expression",
    "loop_expression",
    "match_expression",
    "field_expression",
    "index_expression",
    "unary_expression",
    "binary_expression",
    "type_ascription",
    "type_coercion",
    "array_with",
    "lambda_expression",
    "parenthesized_expression",
    "tuple_expression",
    "array_literal",
    "vec_literal",
    "record_expression",
];

#[derive(Clone, Debug, Default, ValueEnum)]
enum OnParseError {
    Ignore,
    #[default]
    Warn,
    Error,
}

/// Fast, syntax-aware test-case reducer for Wyn.
#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// Source code to consume; if omitted, read stdin.
    #[arg(short, long, value_name = "FILE")]
    source: Option<PathBuf>,

    /// Behavior when the initial source has Tree-sitter parse errors.
    #[arg(long, value_enum, default_value_t)]
    on_parse_error: OnParseError,

    /// Number of parallel interestingness checks used by treereduce.
    #[arg(short, long, default_value_t = default_jobs())]
    jobs: usize,

    /// Wyn compiler checked at startup and used to infer expression defaults.
    #[arg(long, value_name = "FILE", help_heading = "Reduction options")]
    wyn: Option<PathBuf>,

    /// Extra argument passed to `wyn check` while inferring hole types.
    #[arg(long, value_name = "ARG", allow_hyphen_values = true, help_heading = "Reduction options")]
    wyn_check_arg: Vec<String>,

    /// Emit treereduce logs as JSON.
    #[arg(long)]
    json: bool,

    /// Output file, or '-' for stdout.
    #[arg(short, long, default_value = "treereduce.out")]
    output: String,

    /// Print a compact final statistics block.
    #[arg(long)]
    stats: bool,

    /// Increase logging verbosity.
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,

    /// Exit code to consider interesting; may be repeated.
    #[arg(
        long,
        default_values_t = vec![0],
        value_name = "CODE",
        help_heading = "Interestingness check options"
    )]
    interesting_exit_code: Vec<i32>,

    /// Regex to match interesting stdout.
    #[arg(long, value_name = "REGEX", help_heading = "Interestingness check options")]
    interesting_stdout: Option<String>,

    /// Regex to match interesting stderr.
    #[arg(long, value_name = "REGEX", help_heading = "Interestingness check options")]
    interesting_stderr: Option<String>,

    /// Regex on stdout that overrides an interesting result.
    #[arg(
        long,
        value_name = "REGEX",
        requires = "interesting_stdout",
        help_heading = "Interestingness check options"
    )]
    uninteresting_stdout: Option<String>,

    /// Regex on stderr that overrides an interesting result.
    #[arg(
        long,
        value_name = "REGEX",
        requires = "interesting_stderr",
        help_heading = "Interestingness check options"
    )]
    uninteresting_stderr: Option<String>,

    /// Do not verify that the initial test case is interesting.
    #[arg(long, help_heading = "Interestingness check options")]
    no_verify: bool,

    /// Inherit stdout from the interestingness check.
    #[arg(
        long,
        conflicts_with_all = ["interesting_stdout", "uninteresting_stdout"],
        help_heading = "Interestingness check options"
    )]
    inherit_stdout: bool,

    /// Inherit stderr from the interestingness check.
    #[arg(
        long,
        conflicts_with_all = ["interesting_stderr", "uninteresting_stderr"],
        help_heading = "Interestingness check options"
    )]
    inherit_stderr: bool,

    /// Directory in which to place temporary @@ files.
    #[arg(long, value_name = "DIR", help_heading = "Interestingness check options")]
    temp_dir: Option<PathBuf>,

    /// Timeout for each interestingness check in seconds.
    #[arg(long, value_name = "SECS", help_heading = "Interestingness check options")]
    timeout: Option<u64>,

    /// One outer reduction pass with a four-byte minimum.
    #[arg(long, conflicts_with = "slow", help_heading = "Reduction options")]
    fast: bool,

    /// Reduce to a byte-size fixpoint and try non-optional deletions.
    #[arg(long, conflicts_with = "fast", help_heading = "Reduction options")]
    slow: bool,

    /// Maximum outer reduction passes unless --stable or --slow is used.
    #[arg(long, default_value_t = DEFAULT_PASSES, help_heading = "Reduction options")]
    passes: usize,

    /// Minimum byte reduction to attempt in the generic pass.
    #[arg(
        long,
        default_value_t = DEFAULT_MIN_REDUCTION,
        value_name = "BYTES",
        help_heading = "Reduction options"
    )]
    min_reduction: usize,

    /// Continue until an entire outer pass makes no byte-size progress.
    #[arg(long, help_heading = "Reduction options")]
    stable: bool,

    /// Interestingness command; use @@.wyn for a temporary Wyn source file.
    #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
    check: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum CandidateKind {
    Warning,
    Promotion,
    ListDeletion,
    Concrete,
    InferDefault,
}

impl CandidateKind {
    fn priority(self) -> u8 {
        match self {
            Self::Warning => 0,
            Self::Promotion => 1,
            Self::ListDeletion => 2,
            Self::Concrete => 3,
            Self::InferDefault => 4,
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Warning => "unused-warning cleanup",
            Self::Promotion => "child promotion",
            Self::ListDeletion => "list element deletion",
            Self::Concrete => "concrete replacement",
            Self::InferDefault => "inferred default replacement",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum UnusedWarningKind {
    Parameter,
    LetBinding,
    LoopVariable,
    MatchBinding,
    Declaration,
}

impl UnusedWarningKind {
    fn from_label(label: &str) -> Option<Self> {
        match label {
            "parameter" => Some(Self::Parameter),
            "binding" => Some(Self::LetBinding),
            "loop variable" => Some(Self::LoopVariable),
            "pattern binding" => Some(Self::MatchBinding),
            "definition" | "external declaration" => Some(Self::Declaration),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct UnusedWarning {
    kind: UnusedWarningKind,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Candidate {
    start: usize,
    end: usize,
    replacement: Vec<u8>,
    kind: CandidateKind,
}

impl Candidate {
    fn reduction(&self) -> usize {
        (self.end - self.start).saturating_sub(self.replacement.len())
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct StructuralStats {
    attempts: usize,
    accepted: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct WarningStats {
    attempts: usize,
    accepted: usize,
}

#[derive(Clone, Debug, Default)]
struct AcceptedWarnings(Arc<Mutex<HashMap<Vec<u8>, Vec<UnusedWarning>>>>);

impl AcceptedWarnings {
    fn record(&self, source: Vec<u8>, diagnostics: &[u8]) -> io::Result<()> {
        let warnings = parse_unused_warnings(&source, diagnostics);
        self.0
            .lock()
            .map_err(|_| io::Error::other("accepted-warning cache lock was poisoned"))?
            .insert(source, warnings);
        Ok(())
    }

    fn take(&self, source: &[u8]) -> io::Result<Option<Vec<UnusedWarning>>> {
        let mut accepted =
            self.0.lock().map_err(|_| io::Error::other("accepted-warning cache lock was poisoned"))?;
        let warnings = accepted.remove(source);
        // Parallel generic checks may have recorded interesting candidates
        // that lost the final edit race. Only the final accepted source can
        // guide the next cleanup step.
        accepted.clear();
        Ok(warnings)
    }
}

#[derive(Clone, Debug)]
struct ObservedCheck {
    inner: CmdCheck,
    accepted: AcceptedWarnings,
}

#[derive(Debug)]
struct ObservedCheckState {
    source: Vec<u8>,
    inner: CmdCheckState,
}

impl Check for ObservedCheck {
    type State = ObservedCheckState;

    fn start(&self, source: &[u8]) -> io::Result<Self::State> {
        Ok(ObservedCheckState {
            source: source.to_vec(),
            inner: self.inner.start(source)?,
        })
    }

    fn cancel(&self, state: Self::State) -> io::Result<()> {
        self.inner.cancel(state.inner)
    }

    fn try_wait(&self, state: &mut Self::State) -> io::Result<Option<bool>> {
        // treereduce 0.4.1 does not expose output from try_wait. Its active
        // reduction paths use wait, where the accepted output is retained.
        self.inner.try_wait(&mut state.inner)
    }

    fn wait(&self, state: Self::State) -> io::Result<bool> {
        let (interesting, _, _, stderr) = self.inner.wait_with_output(state.inner)?;
        if interesting {
            self.accepted.record(state.source, &stderr)?;
        }
        Ok(interesting)
    }
}

#[derive(Clone, Debug)]
struct SyntaxCheck<C> {
    inner: C,
    language: Language,
    reject_errors: bool,
}

#[derive(Debug)]
enum SyntaxCheckState<S> {
    Rejected,
    Inner(S),
}

impl<C: Check> Check for SyntaxCheck<C> {
    type State = SyntaxCheckState<C::State>;

    fn start(&self, source: &[u8]) -> io::Result<Self::State> {
        let has_error = self.reject_errors
            && parse(&self.language, source)
                .map_err(|error| io::Error::other(error.to_string()))?
                .root_node()
                .has_error();
        if has_error {
            Ok(SyntaxCheckState::Rejected)
        } else {
            self.inner.start(source).map(SyntaxCheckState::Inner)
        }
    }

    fn cancel(&self, state: Self::State) -> io::Result<()> {
        match state {
            SyntaxCheckState::Rejected => Ok(()),
            SyntaxCheckState::Inner(state) => self.inner.cancel(state),
        }
    }

    fn try_wait(&self, state: &mut Self::State) -> io::Result<Option<bool>> {
        match state {
            SyntaxCheckState::Rejected => Ok(Some(false)),
            SyntaxCheckState::Inner(state) => self.inner.try_wait(state),
        }
    }

    fn wait(&self, state: Self::State) -> io::Result<bool> {
        match state {
            SyntaxCheckState::Rejected => Ok(false),
            SyntaxCheckState::Inner(state) => self.inner.wait(state),
        }
    }
}

#[derive(Debug)]
struct TypeProbe {
    wyn: PathBuf,
    check_args: Vec<String>,
    temp_dir: Option<PathBuf>,
    inferred_type: Regex,
}

impl TypeProbe {
    fn new(args: &Args) -> Result<Self> {
        let wyn = args
            .wyn
            .clone()
            .or_else(|| std::env::var_os("WYN").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("./target/release/wyn"));
        Ok(Self {
            wyn,
            check_args: args.wyn_check_arg.clone(),
            temp_dir: args.temp_dir.clone(),
            inferred_type: Regex::new(r"type hole inferred as `([^`]+)`")?,
        })
    }

    fn verify_compiler(&self) -> Result<()> {
        let output = Command::new(&self.wyn).arg("--help").output().with_context(|| {
            format!(
                "failed to run Wyn compiler preflight `{} --help`",
                self.wyn.display()
            )
        })?;
        if output.status.success() {
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let detail = if stderr.trim().is_empty() { stdout.trim() } else { stderr.trim() };
        if detail.is_empty() {
            bail!(
                "Wyn compiler preflight `{} --help` exited with {}",
                self.wyn.display(),
                output.status
            );
        }
        bail!(
            "Wyn compiler preflight `{} --help` exited with {}: {detail}",
            self.wyn.display(),
            output.status
        )
    }

    fn resolve(&self, source: &[u8], candidate: Candidate) -> Result<Option<Candidate>> {
        // Human-readable diagnostics do not identify holes in a form that is
        // safe to parse when the source already contains another hole.
        if source.windows(HOLE[0].len()).any(|window| window == HOLE[0].as_bytes()) {
            return Ok((candidate.kind != CandidateKind::InferDefault).then_some(candidate));
        }

        if candidate.kind != CandidateKind::InferDefault {
            return Ok(Some(candidate));
        }

        let Some(ty) = self.infer_type(source, candidate.start, candidate.end)? else {
            return Ok(None);
        };
        let Some(replacement) = default_literal(&ty) else {
            return Ok(None);
        };
        if replacement.len() >= candidate.end - candidate.start {
            return Ok(None);
        }
        Ok(Some(Candidate {
            replacement: replacement.into_bytes(),
            ..candidate
        }))
    }

    fn infer_type(&self, source: &[u8], start: usize, end: usize) -> Result<Option<String>> {
        let mut hole_source = Vec::with_capacity(source.len() - (end - start) + HOLE[0].len());
        hole_source.extend_from_slice(&source[..start]);
        hole_source.extend_from_slice(HOLE[0].as_bytes());
        hole_source.extend_from_slice(&source[end..]);
        let mut builder = tempfile::Builder::new();
        builder.prefix("treereduce-type-").suffix(".wyn");
        let mut file =
            if let Some(dir) = &self.temp_dir { builder.tempfile_in(dir)? } else { builder.tempfile()? };
        file.write_all(&hole_source)?;
        file.flush()?;
        let output = Command::new(&self.wyn)
            .arg("check")
            .args(&self.check_args)
            .arg(file.path())
            .output()
            .with_context(|| format!("failed to run type probe {}", self.wyn.display()))?;
        let stderr = String::from_utf8_lossy(&output.stderr);
        Ok(self
            .inferred_type
            .captures(&stderr)
            .and_then(|captures| captures.get(1))
            .map(|capture| capture.as_str().to_owned()))
    }
}

fn default_literal(ty: &str) -> Option<String> {
    let ty = ty.trim();
    if ty == "bool" {
        return Some("false".to_owned());
    }
    if ty == "()" {
        return Some("()".to_owned());
    }
    if ty.len() >= 2
        && matches!(ty.as_bytes()[0], b'i' | b'u')
        && ty.as_bytes()[1..].iter().all(u8::is_ascii_digit)
    {
        return Some("0".to_owned());
    }
    if ty.len() >= 2 && ty.starts_with('f') && ty.as_bytes()[1..].iter().all(u8::is_ascii_digit) {
        return Some("0.0".to_owned());
    }
    if let Some(rest) = ty.strip_prefix("vec") {
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        let size: usize = rest[..digits].parse().ok()?;
        let element = default_literal(&rest[digits..])?;
        return Some(format!("@[{}]", vec![element; size].join(", ")));
    }
    if ty.starts_with('(') && ty.ends_with(')') {
        let elements = split_type_list(&ty[1..ty.len() - 1])?;
        let defaults = elements.into_iter().map(default_literal).collect::<Option<Vec<_>>>()?;
        return Some(format!("({})", defaults.join(", ")));
    }
    if ty.starts_with('[') {
        let end = ty.find(']')?;
        let size: usize = ty[1..end].parse().ok()?;
        let element = default_literal(&ty[end + 1..])?;
        return Some(format!("[{}]", vec![element; size].join(", ")));
    }
    None
}

fn split_type_list(types: &str) -> Option<Vec<&str>> {
    let mut result = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (index, byte) in types.bytes().enumerate() {
        match byte {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth = depth.checked_sub(1)?,
            b',' if depth == 0 => {
                result.push(types[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return None;
    }
    result.push(types[start..].trim());
    Some(result)
}

fn default_jobs() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

fn main() -> Result<()> {
    let args = Args::parse();
    init_tracing(&args);

    let type_probe = TypeProbe::new(&args)?;
    if args.verbose > 0 {
        eprintln!("checking Wyn compiler: {} --help", type_probe.wyn.display());
    }
    type_probe.verify_compiler()?;

    if let Some(dir) = &args.temp_dir {
        fs::create_dir_all(dir)
            .with_context(|| format!("failed to create temporary directory {}", dir.display()))?;
    }

    let accepted_warnings = AcceptedWarnings::default();
    let check = build_check(&args, accepted_warnings.clone())?;
    let language: Language = tree_sitter_wyn::LANGUAGE.into();
    let mut source = read_source(&args)?;
    let initial_size = source.len();
    let initial_tree = parse(&language, &source)?;
    handle_initial_parse_errors(&args, &initial_tree)?;
    let (comments_count, comments_size) = remove_comments(&initial_tree, &mut source);
    if args.verbose > 0 && comments_count > 0 {
        eprintln!("removed {comments_count} comments before reduction (-{comments_size} bytes)");
    }
    let check = SyntaxCheck {
        inner: check,
        language: language.clone(),
        reject_errors: !parse(&language, &source)?.root_node().has_error(),
    };
    reduce(
        &args,
        &language,
        source,
        initial_size,
        check,
        type_probe,
        accepted_warnings,
    )
}

fn reduce<C>(
    args: &Args,
    language: &Language,
    mut source: Vec<u8>,
    initial_size: usize,
    check: C,
    type_probe: TypeProbe,
    accepted_warnings: AcceptedWarnings,
) -> Result<()>
where
    C: Check + Clone + Debug + Send + Sync + 'static,
{
    if args.verbose > 0 {
        eprintln!("verifying initial test case ({} bytes)", source.len());
    }
    let start = Instant::now();
    let mut warning_stats = WarningStats::default();
    if !args.no_verify {
        if !check.interesting(&source)? {
            bail!("initial test case is not interesting");
        }
        let (next, stats) =
            postprocess_unused_warnings(language, source, &check, &accepted_warnings, args.verbose)?;
        source = next;
        warning_stats.attempts += stats.attempts;
        warning_stats.accepted += stats.accepted;
    }

    let node_types = NodeTypes::new(tree_sitter_wyn::NODE_TYPES)
        .context("failed to read tree-sitter-wyn node-types.json")?;
    let mut structural_stats = StructuralStats::default();
    let mut passes_done = 0;
    let max_passes = if args.fast {
        Some(1)
    } else if args.stable || args.slow {
        None
    } else {
        Some(args.passes)
    };
    let min_reduction = if args.fast {
        4
    } else if args.slow {
        1
    } else {
        args.min_reduction.max(1)
    };

    loop {
        if max_passes.is_some_and(|limit| passes_done >= limit) {
            break;
        }
        passes_done += 1;
        let pass_start_size = source.len();

        if args.verbose > 0 {
            eprintln!("outer pass {passes_done}: structural reduction ({pass_start_size} bytes)");
        }
        let (next, stats) =
            structural_reduce(language, source, &check, &type_probe, args.jobs, args.verbose)?;
        source = next;
        structural_stats.attempts += stats.attempts;
        structural_stats.accepted += stats.accepted;
        let (next, stats) =
            postprocess_unused_warnings(language, source, &check, &accepted_warnings, args.verbose)?;
        source = next;
        warning_stats.attempts += stats.attempts;
        warning_stats.accepted += stats.accepted;

        if args.verbose > 0 {
            eprintln!(
                "outer pass {passes_done}: structural reduction finished ({} bytes, {}/{} accepted)",
                source.len(),
                stats.accepted,
                stats.attempts
            );
            eprintln!(
                "outer pass {passes_done}: generic reduction ({} bytes)",
                source.len()
            );
        }
        source = generic_pass(
            language,
            &node_types,
            source,
            &check,
            args.jobs,
            min_reduction,
            args.slow,
            hole_replacements(),
        )?;
        let (next, stats) =
            postprocess_unused_warnings(language, source, &check, &accepted_warnings, args.verbose)?;
        source = next;
        warning_stats.attempts += stats.attempts;
        warning_stats.accepted += stats.accepted;

        if args.verbose > 0 {
            eprintln!(
                "outer pass {passes_done}: {pass_start_size} -> {} bytes",
                source.len()
            );
        }
        // Do not use treereduce 0.4.1's Edits::is_empty here: it ignores
        // replacement-only progress. Comparing rendered size gives --stable
        // the fixpoint behavior users expect.
        if source.len() == pass_start_size {
            break;
        }
    }

    let compacted = collapse_blank_lines(&source);
    if compacted.len() < source.len() {
        let reduction = source.len() - compacted.len();
        if check.interesting(&compacted)? {
            if args.verbose > 0 {
                eprintln!("collapsed repeated blank lines (-{reduction} bytes)");
            }
            source = compacted;
        } else if args.verbose > 0 {
            eprintln!("kept blank lines because collapsing them made the test uninteresting");
        }
    }

    write_output(&args.output, &source)?;
    if args.stats {
        println!("start size: {initial_size}");
        println!("end size: {}", source.len());
        println!("outer passes: {passes_done}");
        println!("structural attempts: {}", structural_stats.attempts);
        println!("structural accepted: {}", structural_stats.accepted);
        println!("unused-warning attempts: {}", warning_stats.attempts);
        println!("unused-warning accepted: {}", warning_stats.accepted);
        println!("duration: {:.3}s", start.elapsed().as_secs_f64());
    }
    Ok(())
}

fn init_tracing(args: &Args) {
    use tracing_subscriber::filter::LevelFilter;

    let level = match args.verbose {
        0 => LevelFilter::WARN,
        1 => LevelFilter::INFO,
        2 => LevelFilter::DEBUG,
        _ => LevelFilter::TRACE,
    };
    if args.json {
        let _ = tracing_subscriber::fmt().json().with_max_level(level).try_init();
    } else {
        let _ = tracing_subscriber::fmt().with_max_level(level).try_init();
    }
}

fn build_check(args: &Args, accepted: AcceptedWarnings) -> Result<ObservedCheck> {
    let (cmd, command_args) = args.check.split_first().context("missing interestingness command")?;
    let regex = |value: &Option<String>| -> Result<Option<Regex>> {
        value
            .as_ref()
            .map(|pattern| Regex::new(pattern).with_context(|| format!("invalid regex: {pattern}")))
            .transpose()
    };

    let interesting_stderr = regex(&args.interesting_stderr)?;
    let capture_direct_wyn = !args.inherit_stderr
        && interesting_stderr.is_none()
        && PathBuf::from(cmd).file_stem().is_some_and(|name| name.eq_ignore_ascii_case("wyn"));
    // An impossible selector asks CmdCheck to retain stderr without changing
    // the interestingness decision. Avoid imposing capture on arbitrary
    // commands; direct Wyn invocations have capped diagnostic output.
    let interesting_stderr = if capture_direct_wyn {
        Some(Regex::new(r"\z.").expect("capture-only regex should compile"))
    } else {
        interesting_stderr
    };

    let inner = CmdCheck::new(
        cmd.clone(),
        command_args.to_vec(),
        args.interesting_exit_code.clone(),
        args.temp_dir.as_ref().map(|path| path.to_string_lossy().into_owned()),
        regex(&args.interesting_stdout)?,
        interesting_stderr,
        regex(&args.uninteresting_stdout)?,
        regex(&args.uninteresting_stderr)?,
        args.inherit_stdout,
        args.inherit_stderr,
        args.timeout.map(Duration::from_secs),
    );
    Ok(ObservedCheck { inner, accepted })
}

fn read_source(args: &Args) -> Result<Vec<u8>> {
    if let Some(path) = &args.source {
        return fs::read(path).with_context(|| format!("failed to read {}", path.display()));
    }
    let mut source = Vec::new();
    io::stdin().read_to_end(&mut source)?;
    Ok(source)
}

fn comment_ranges(tree: &Tree) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        if node.kind() == "comment" {
            ranges.push((node.start_byte(), node.end_byte()));
            continue;
        }
        let mut cursor = node.walk();
        stack.extend(node.children(&mut cursor));
    }
    ranges.sort_unstable();
    ranges
}

fn remove_comments(tree: &Tree, source: &mut Vec<u8>) -> (usize, usize) {
    let ranges = comment_ranges(tree);
    let count = ranges.len();
    let size = ranges.iter().map(|(start, end)| end - start).sum();
    for (start, end) in ranges.into_iter().rev() {
        source.drain(start..end);
    }
    (count, size)
}

fn collapse_blank_lines(source: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(source.len());
    let mut in_blank_run = false;
    for line in source.split_inclusive(|byte| *byte == b'\n') {
        let content = line.strip_suffix(b"\n").unwrap_or(line);
        let content = content.strip_suffix(b"\r").unwrap_or(content);
        let blank = content.iter().all(|byte| matches!(byte, b' ' | b'\t'));
        if blank {
            if !in_blank_run && line.ends_with(b"\n") {
                output.push(b'\n');
            }
            in_blank_run = true;
        } else {
            output.extend_from_slice(line);
            in_blank_run = false;
        }
    }
    output
}

fn write_output(output: &str, source: &[u8]) -> Result<()> {
    if output == "-" {
        io::stdout().lock().write_all(source)?;
    } else {
        fs::write(output, source).with_context(|| format!("failed to write {output}"))?;
    }
    Ok(())
}

fn parse(language: &Language, source: &[u8]) -> Result<Tree> {
    let mut parser = TreeSitterParser::new();
    parser.set_language(language).context("failed to load tree-sitter-wyn")?;
    parser.parse(source, None).context("tree-sitter returned no parse tree")
}

fn handle_initial_parse_errors(args: &Args, tree: &Tree) -> Result<()> {
    if !tree.root_node().has_error() {
        return Ok(());
    }
    match args.on_parse_error {
        OnParseError::Ignore => Ok(()),
        OnParseError::Warn => {
            eprintln!("warning: initial source contains Tree-sitter parse errors");
            Ok(())
        }
        OnParseError::Error => bail!("initial source contains Tree-sitter parse errors"),
    }
}

fn parse_unused_warnings(source: &[u8], diagnostics: &[u8]) -> Vec<UnusedWarning> {
    let Ok(source) = std::str::from_utf8(source) else {
        return Vec::new();
    };
    let diagnostics = String::from_utf8_lossy(diagnostics);
    UNUSED_WARNING_DIAGNOSTIC
        .captures_iter(&diagnostics)
        .filter_map(|captures| {
            let kind = UnusedWarningKind::from_label(captures.get(1)?.as_str())?;
            let name = captures.get(2)?.as_str();
            let line = captures.get(3)?.as_str().parse().ok()?;
            let column = captures.get(4)?.as_str().parse().ok()?;
            let start = line_column_offset(source, line, column)?;
            let end = start.checked_add(name.len())?;
            (source.as_bytes().get(start..end)? == name.as_bytes()).then_some(UnusedWarning {
                kind,
                start,
                end,
            })
        })
        .collect()
}

fn line_column_offset(source: &str, line: usize, column: usize) -> Option<usize> {
    let line = line.checked_sub(1)?;
    let column = column.checked_sub(1)?;
    let mut line_start = 0;
    for _ in 0..line {
        line_start += source.as_bytes().get(line_start..)?.iter().position(|byte| *byte == b'\n')? + 1;
    }
    let line_end = source.as_bytes()[line_start..]
        .iter()
        .position(|byte| *byte == b'\n')
        .map_or(source.len(), |offset| line_start + offset);
    let line_source = &source[line_start..line_end];
    let column_offset = if column == line_source.chars().count() {
        line_source.len()
    } else {
        line_source.char_indices().nth(column)?.0
    };
    Some(line_start + column_offset)
}

fn postprocess_unused_warnings<C: Check>(
    language: &Language,
    mut source: Vec<u8>,
    check: &C,
    accepted_warnings: &AcceptedWarnings,
    verbose: u8,
) -> Result<(Vec<u8>, WarningStats)> {
    let mut stats = WarningStats::default();
    while let Some(warnings) = accepted_warnings.take(&source)? {
        if warnings.is_empty() {
            break;
        }
        let tree = parse(language, &source)?;
        let candidates = collect_unused_warning_candidates(&tree, &source, &warnings);
        if verbose > 1 && !candidates.is_empty() {
            eprintln!(
                "trying {} cleanup candidates from {} accepted-check unused warnings",
                candidates.len(),
                warnings.len()
            );
        }

        let mut reduced = false;
        for candidate in candidates {
            let next = apply_candidate(&source, &candidate);
            stats.attempts += 1;
            if check.wait(check.start(&next)?)? {
                if verbose > 1 {
                    eprintln!(
                        "accepted {} at {}..{} (-{} bytes)",
                        candidate.kind.description(),
                        candidate.start,
                        candidate.end,
                        candidate.reduction()
                    );
                }
                source = next;
                stats.accepted += 1;
                reduced = true;
                break;
            }
        }
        if !reduced {
            break;
        }
    }
    Ok((source, stats))
}

fn collect_unused_warning_candidates(
    tree: &Tree,
    source: &[u8],
    warnings: &[UnusedWarning],
) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    for warning in warnings {
        let Some(node) = tree.root_node().descendant_for_byte_range(warning.start, warning.end) else {
            continue;
        };
        match warning.kind {
            UnusedWarningKind::Declaration => {
                if let Some(declaration) =
                    ancestor_of_kind(node, &["def_declaration", "extern_declaration"])
                {
                    let (start, end) = declaration_deletion_range(declaration, source);
                    candidates.push(Candidate {
                        start,
                        end,
                        replacement: Vec::new(),
                        kind: CandidateKind::Warning,
                    });
                }
            }
            UnusedWarningKind::LetBinding => {
                if let Some(let_expression) = ancestor_of_kind(node, &["let_expression"]) {
                    if let Some(body) = let_expression.child_by_field_name("body") {
                        candidates.push(Candidate {
                            start: let_expression.start_byte(),
                            end: let_expression.end_byte(),
                            replacement: source[body.start_byte()..body.end_byte()].to_vec(),
                            kind: CandidateKind::Warning,
                        });
                    }
                }
                add_wildcard_candidate(warning, &mut candidates);
            }
            UnusedWarningKind::Parameter => {
                if let Some(candidate) = parameter_deletion_candidate(node, source) {
                    candidates.push(candidate);
                }
                add_wildcard_candidate(warning, &mut candidates);
            }
            UnusedWarningKind::LoopVariable | UnusedWarningKind::MatchBinding => {
                add_wildcard_candidate(warning, &mut candidates);
            }
        }
    }

    let mut unique = HashSet::new();
    candidates.retain(|candidate| {
        candidate.end > candidate.start
            && candidate.replacement.len() < candidate.end - candidate.start
            && unique.insert((candidate.start, candidate.end, candidate.replacement.clone()))
    });
    candidates.sort_by(|left, right| {
        right.reduction().cmp(&left.reduction()).then_with(|| left.start.cmp(&right.start))
    });
    candidates
}

fn ancestor_of_kind<'tree>(mut node: Node<'tree>, kinds: &[&str]) -> Option<Node<'tree>> {
    loop {
        if kinds.contains(&node.kind()) {
            return Some(node);
        }
        node = node.parent()?;
    }
}

fn declaration_deletion_range(declaration: Node<'_>, source: &[u8]) -> (usize, usize) {
    let start = declaration.start_byte();
    let mut end = declaration.end_byte();
    while matches!(source.get(end), Some(b' ' | b'\t')) {
        end += 1;
    }
    if source.get(end..end + 2) == Some(b"\r\n") {
        end += 2;
    } else if source.get(end) == Some(&b'\n') {
        end += 1;
    }
    (start, end)
}

fn add_wildcard_candidate(warning: &UnusedWarning, candidates: &mut Vec<Candidate>) {
    if warning.end - warning.start > WILDCARD[0].len() {
        candidates.push(Candidate {
            start: warning.start,
            end: warning.end,
            replacement: WILDCARD[0].as_bytes().to_vec(),
            kind: CandidateKind::Warning,
        });
    }
}

fn parameter_deletion_candidate(node: Node<'_>, source: &[u8]) -> Option<Candidate> {
    let mut element = node;
    let container = loop {
        let parent = element.parent()?;
        if matches!(parent.kind(), "params" | "entry_params" | "lambda_params") {
            break parent;
        }
        element = parent;
    };
    let mut cursor = container.walk();
    let elements: Vec<_> = container.named_children(&mut cursor).collect();
    let mut candidates = Vec::new();
    add_comma_deletions(container, &elements, true, false, source, &mut candidates);
    candidates
        .into_iter()
        .find(|candidate| candidate.start <= element.start_byte() && candidate.end >= element.end_byte())
        .map(|candidate| Candidate {
            kind: CandidateKind::Warning,
            ..candidate
        })
}

fn hole_replacements() -> HashMap<&'static str, &'static [&'static str]> {
    COMPOSITE_EXPRESSION_KINDS.iter().map(|kind| (*kind, HOLE)).collect()
}

#[allow(clippy::too_many_arguments)]
fn generic_pass<C>(
    language: &Language,
    node_types: &NodeTypes,
    source: Vec<u8>,
    check: &C,
    jobs: usize,
    min_reduction: usize,
    delete_non_optional: bool,
    replacements: HashMap<&'static str, &'static [&'static str]>,
) -> Result<Vec<u8>>
where
    C: Check + Clone + Debug + Send + Sync + 'static,
{
    let tree = parse(language, &source)?;
    let config = Config {
        check: check.clone(),
        delete_non_optional,
        jobs,
        min_reduction,
        replacements,
    };
    let (original, edits) = treereduce::treereduce(node_types, Original::new(tree, source), &config)
        .context("treereduce pass failed")?;
    let mut rendered = Vec::new();
    tree_sitter_edit::render(&mut rendered, &original.tree, &original.text, &edits)
        .context("failed to render treereduce edits")?;
    Ok(rendered)
}

fn structural_reduce<C: Check>(
    language: &Language,
    mut source: Vec<u8>,
    check: &C,
    type_probe: &TypeProbe,
    jobs: usize,
    verbose: u8,
) -> Result<(Vec<u8>, StructuralStats)> {
    let mut stats = StructuralStats::default();
    loop {
        let tree = parse(language, &source)?;
        let candidates = collect_candidates(&tree, &source);
        let mut accepted = false;
        let mut candidates = candidates.into_iter();
        let jobs = jobs.max(1);
        loop {
            // Candidates are ordered by reduction size, then source position.
            // Start one ordered window concurrently and consume results in
            // that same order, so parallelism cannot change which edit wins.
            let batch: Vec<_> = candidates.by_ref().take(jobs).collect();
            if batch.is_empty() {
                break;
            }
            let mut pending = Vec::with_capacity(batch.len());
            for candidate in batch {
                let Some(candidate) = type_probe.resolve(&source, candidate)? else {
                    continue;
                };
                let next = apply_candidate(&source, &candidate);
                let state = match check.start(&next) {
                    Ok(state) => state,
                    Err(error) => {
                        for (_, _, state) in pending {
                            let _ = check.cancel(state);
                        }
                        return Err(error.into());
                    }
                };
                stats.attempts += 1;
                pending.push((candidate, next, state));
            }

            let mut pending = pending.into_iter();
            while let Some((candidate, next, state)) = pending.next() {
                let interesting = match check.wait(state) {
                    Ok(interesting) => interesting,
                    Err(error) => {
                        for (_, _, state) in pending {
                            let _ = check.cancel(state);
                        }
                        return Err(error.into());
                    }
                };
                if interesting {
                    for (_, _, state) in pending {
                        // These are speculative checks whose answers can no
                        // longer affect the selected candidate. Cancellation
                        // failure must not discard an accepted reduction.
                        let _ = check.cancel(state);
                    }
                    if verbose > 1 {
                        eprintln!(
                            "accepted {} at {}..{} (-{} bytes)",
                            candidate.kind.description(),
                            candidate.start,
                            candidate.end,
                            candidate.reduction()
                        );
                    }
                    source = next;
                    stats.accepted += 1;
                    accepted = true;
                    break;
                }
            }
            if accepted {
                break;
            }
        }
        if !accepted {
            return Ok((source, stats));
        }
    }
}

fn apply_candidate(source: &[u8], candidate: &Candidate) -> Vec<u8> {
    let mut next = Vec::with_capacity(source.len() - candidate.reduction());
    next.extend_from_slice(&source[..candidate.start]);
    next.extend_from_slice(&candidate.replacement);
    next.extend_from_slice(&source[candidate.end..]);
    next
}

fn collect_candidates(tree: &Tree, source: &[u8]) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        collect_promotions(node, source, &mut candidates);
        collect_list_deletions(node, source, &mut candidates);
        collect_concrete_replacements(node, &mut candidates);

        let mut cursor = node.walk();
        stack.extend(node.children(&mut cursor));
    }

    let mut unique = HashSet::new();
    candidates.retain(|candidate| {
        candidate.end > candidate.start
            && candidate.replacement.len() < candidate.end - candidate.start
            && unique.insert((candidate.start, candidate.end, candidate.replacement.clone()))
    });
    candidates.sort_by(|left, right| {
        left.kind
            .priority()
            .cmp(&right.kind.priority())
            .then_with(|| right.reduction().cmp(&left.reduction()))
            .then_with(|| left.start.cmp(&right.start))
    });
    candidates
}

fn collect_concrete_replacements(node: Node<'_>, candidates: &mut Vec<Candidate>) {
    let replacements = match node.kind() {
        "integer_literal" => INTEGER_EXPRESSIONS,
        "float_literal" => FLOAT_EXPRESSIONS,
        "boolean_literal" => BOOLEAN_EXPRESSIONS,
        kind if COMPOSITE_EXPRESSION_KINDS.contains(&kind) => {
            candidates.push(Candidate {
                start: node.start_byte(),
                end: node.end_byte(),
                replacement: b"???".to_vec(),
                kind: CandidateKind::InferDefault,
            });
            return;
        }
        "tuple_pattern"
        | "record_pattern"
        | "typed_pattern"
        | "attributed_pattern"
        | "constructor_pattern"
        | "parenthesized_pattern" => WILDCARD,
        _ => return,
    };
    for replacement in replacements {
        candidates.push(Candidate {
            start: node.start_byte(),
            end: node.end_byte(),
            replacement: replacement.as_bytes().to_vec(),
            kind: CandidateKind::Concrete,
        });
    }
}

fn collect_promotions(node: Node<'_>, source: &[u8], candidates: &mut Vec<Candidate>) {
    match node.kind() {
        "let_expression" => {
            if let Some(body) = node.child_by_field_name("body") {
                if !let_body_references_bindings(node, body, source) {
                    add_promotion(node, body, source, candidates);
                }
            }
        }
        "if_expression" => promote_fields(node, &["then", "else"], source, candidates),
        "binary_expression" => promote_fields(
            node,
            &["left", "right", "start", "step", "end"],
            source,
            candidates,
        ),
        "unary_expression" => promote_fields(node, &["operand"], source, candidates),
        "type_ascription" | "type_coercion" => promote_fields(node, &["expression"], source, candidates),
        "field_expression" | "index_expression" => {
            promote_fields(node, &["object", "start", "end"], source, candidates);
            promote_direct_expressions(node, source, candidates);
        }
        "array_with" => promote_fields(node, &["array", "index", "value"], source, candidates),
        "lambda_expression" => promote_fields(node, &["body"], source, candidates),
        "loop_expression" => {
            promote_fields(node, &["init", "body"], source, candidates);
            if let Some(form) = node.child_by_field_name("form") {
                promote_fields(form, &["bound", "iterable", "condition"], source, candidates);
            }
        }
        "match_expression" => {
            promote_fields(node, &["value"], source, candidates);
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if child.kind() == "case_clause" {
                    promote_fields(child, &["body"], source, candidates);
                }
            }
        }
        "call_expression" => promote_call_arguments(node, source, candidates),
        "record_expression" => {
            let mut cursor = node.walk();
            for field in node.named_children(&mut cursor) {
                if field.kind() == "record_field" {
                    if let Some(value) = field.child_by_field_name("value") {
                        add_promotion(node, value, source, candidates);
                    } else if let Some(value) = field.named_child(0) {
                        add_promotion(node, value, source, candidates);
                    }
                }
            }
        }
        "parenthesized_expression" => {
            if let Some(child) = node.named_child(0) {
                add_promotion(node, child, source, candidates);
            }
        }
        _ => {}
    }
}

fn let_body_references_bindings(let_expression: Node<'_>, body: Node<'_>, source: &[u8]) -> bool {
    let mut bindings = Vec::new();
    for field in ["pattern", "name", "params"] {
        if let Some(node) = let_expression.child_by_field_name(field) {
            collect_identifiers(node, source, &mut bindings);
        }
    }
    if bindings.is_empty() {
        return false;
    }

    let mut references = Vec::new();
    collect_identifiers(body, source, &mut references);
    references.iter().any(|reference| bindings.contains(reference))
}

fn collect_identifiers(node: Node<'_>, source: &[u8], identifiers: &mut Vec<Vec<u8>>) {
    let mut stack = vec![node];
    while let Some(node) = stack.pop() {
        if node.kind() == "identifier" {
            identifiers.push(source[node.start_byte()..node.end_byte()].to_vec());
            continue;
        }
        let mut cursor = node.walk();
        stack.extend(node.named_children(&mut cursor));
    }
}

fn promote_fields(parent: Node<'_>, fields: &[&str], source: &[u8], candidates: &mut Vec<Candidate>) {
    for field in fields {
        if let Some(child) = parent.child_by_field_name(field) {
            add_promotion(parent, child, source, candidates);
        }
    }
}

fn promote_direct_expressions(parent: Node<'_>, source: &[u8], candidates: &mut Vec<Candidate>) {
    let mut cursor = parent.walk();
    for child in parent.named_children(&mut cursor) {
        if is_expression_kind(child.kind()) {
            add_promotion(parent, child, source, candidates);
        }
    }
}

fn promote_call_arguments(parent: Node<'_>, source: &[u8], candidates: &mut Vec<Candidate>) {
    let function_end =
        parent.child_by_field_name("function").map_or(parent.start_byte(), |function| function.end_byte());
    let mut cursor = parent.walk();
    for child in parent.named_children(&mut cursor) {
        if child.start_byte() >= function_end && is_expression_kind(child.kind()) {
            add_promotion(parent, child, source, candidates);
        }
    }
}

fn add_promotion(parent: Node<'_>, child: Node<'_>, source: &[u8], candidates: &mut Vec<Candidate>) {
    let start = parent.start_byte();
    let end = parent.end_byte();
    let child_start = child.start_byte();
    let child_end = child.end_byte();
    if start <= child_start && child_end <= end {
        candidates.push(Candidate {
            start,
            end,
            replacement: source[child_start..child_end].to_vec(),
            kind: CandidateKind::Promotion,
        });
    }
}

fn is_expression_kind(kind: &str) -> bool {
    matches!(
        kind,
        "identifier"
            | "qualified_name"
            | "integer_literal"
            | "float_literal"
            | "boolean_literal"
            | "type_hole"
            | "call_expression"
            | "let_expression"
            | "if_expression"
            | "loop_expression"
            | "match_expression"
            | "field_expression"
            | "index_expression"
            | "unary_expression"
            | "binary_expression"
            | "type_ascription"
            | "type_coercion"
            | "array_with"
            | "lambda_expression"
            | "parenthesized_expression"
            | "tuple_expression"
            | "array_literal"
            | "vec_literal"
            | "record_expression"
    )
}

fn collect_list_deletions(node: Node<'_>, source: &[u8], candidates: &mut Vec<Candidate>) {
    if node.kind() == "match_expression" {
        let clauses = named_children_of_kind(node, "case_clause");
        if clauses.len() > 1 {
            for clause in clauses {
                candidates.push(Candidate {
                    start: clause.start_byte(),
                    end: clause.end_byte(),
                    replacement: Vec::new(),
                    kind: CandidateKind::ListDeletion,
                });
            }
        }
        return;
    }

    let (elements, allow_empty, collapse_single) = match node.kind() {
        "call_expression" => {
            let function_end = node
                .child_by_field_name("function")
                .map_or(node.start_byte(), |function| function.end_byte());
            let mut cursor = node.walk();
            (
                node.named_children(&mut cursor)
                    .filter(|child| child.start_byte() >= function_end)
                    .filter(|child| is_expression_kind(child.kind()) || child.kind() == "call_placeholder")
                    .collect(),
                true,
                false,
            )
        }
        "params" => (named_children_of_kind(node, "param"), true, false),
        "extern_params" => (named_children_of_kind(node, "extern_param"), true, false),
        "functor_params" => (named_children_of_kind(node, "functor_param"), false, false),
        "generic_params" => {
            let mut cursor = node.walk();
            (
                node.named_children(&mut cursor)
                    .filter(|child| matches!(child.kind(), "size_param" | "type_variable"))
                    .collect(),
                false,
                false,
            )
        }
        "lambda_params" => {
            let mut cursor = node.walk();
            (node.named_children(&mut cursor).collect(), true, false)
        }
        "array_literal" | "vec_literal" | "tuple_expression" => {
            let mut cursor = node.walk();
            (
                node.named_children(&mut cursor).filter(|child| is_expression_kind(child.kind())).collect(),
                true,
                node.kind() == "tuple_expression",
            )
        }
        "record_expression" => (named_children_of_kind(node, "record_field"), true, false),
        "record_pattern" => (named_children_of_kind(node, "record_field_pattern"), true, false),
        "record_type" => (named_children_of_kind(node, "record_field_type"), true, false),
        "tuple_pattern" | "tuple_type" => {
            let mut cursor = node.walk();
            (node.named_children(&mut cursor).collect(), true, true)
        }
        _ => return,
    };

    add_comma_deletions(node, &elements, allow_empty, collapse_single, source, candidates);
}

fn named_children_of_kind<'tree>(node: Node<'tree>, kind: &str) -> Vec<Node<'tree>> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor).filter(|child| child.kind() == kind).collect()
}

fn add_comma_deletions(
    container: Node<'_>,
    elements: &[Node<'_>],
    allow_empty: bool,
    collapse_single: bool,
    source: &[u8],
    candidates: &mut Vec<Candidate>,
) {
    if elements.is_empty() || (elements.len() == 1 && !allow_empty) {
        return;
    }
    if elements.len() == 1 {
        let end = trailing_comma_end(container, elements[0], source).unwrap_or(elements[0].end_byte());
        candidates.push(Candidate {
            start: elements[0].start_byte(),
            end,
            replacement: Vec::new(),
            kind: CandidateKind::ListDeletion,
        });
        return;
    }

    for (index, element) in elements.iter().enumerate() {
        let range = if let Some(next) = elements.get(index + 1) {
            let gap = &source[element.end_byte()..next.start_byte()];
            gap.contains(&b',').then_some((element.start_byte(), next.start_byte()))
        } else {
            let previous = elements[index - 1];
            let gap = &source[previous.end_byte()..element.start_byte()];
            let mut end = element.end_byte();
            if collapse_single && elements.len() == 2 {
                end = trailing_comma_end(container, *element, source).unwrap_or(end);
            }
            gap.contains(&b',').then_some((previous.end_byte(), end))
        };
        if let Some((start, end)) = range {
            candidates.push(Candidate {
                start,
                end,
                replacement: Vec::new(),
                kind: CandidateKind::ListDeletion,
            });
        }
    }
}

fn trailing_comma_end(container: Node<'_>, element: Node<'_>, source: &[u8]) -> Option<usize> {
    let tail_start = element.end_byte();
    let tail = &source[tail_start..container.end_byte()];
    tail.iter().position(|byte| *byte == b',').map(|offset| tail_start + offset + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone, Debug)]
    struct Contains(&'static [u8]);

    impl Check for Contains {
        type State = bool;

        fn start(&self, source: &[u8]) -> io::Result<Self::State> {
            Ok(source.windows(self.0.len()).any(|window| window == self.0))
        }

        fn cancel(&self, _state: Self::State) -> io::Result<()> {
            Ok(())
        }

        fn try_wait(&self, state: &mut Self::State) -> io::Result<Option<bool>> {
            Ok(Some(*state))
        }

        fn wait(&self, state: Self::State) -> io::Result<bool> {
            Ok(state)
        }
    }

    #[derive(Clone, Debug)]
    struct RecordingContains {
        needle: &'static [u8],
        accepted: AcceptedWarnings,
        checks: Arc<AtomicUsize>,
    }

    impl Check for RecordingContains {
        type State = Vec<u8>;

        fn start(&self, source: &[u8]) -> io::Result<Self::State> {
            self.checks.fetch_add(1, Ordering::Relaxed);
            Ok(source.to_vec())
        }

        fn cancel(&self, _state: Self::State) -> io::Result<()> {
            Ok(())
        }

        fn try_wait(&self, state: &mut Self::State) -> io::Result<Option<bool>> {
            Ok(Some(
                state.windows(self.needle.len()).any(|window| window == self.needle),
            ))
        }

        fn wait(&self, state: Self::State) -> io::Result<bool> {
            let interesting = state.windows(self.needle.len()).any(|window| window == self.needle);
            if interesting {
                self.accepted.record(state, b"")?;
            }
            Ok(interesting)
        }
    }

    fn language() -> Language {
        tree_sitter_wyn::LANGUAGE.into()
    }

    fn type_probe() -> TypeProbe {
        TypeProbe {
            wyn: std::env::current_exe().unwrap(),
            check_args: Vec::new(),
            temp_dir: None,
            inferred_type: Regex::new(r"type hole inferred as `([^`]+)`").unwrap(),
        }
    }

    #[test]
    fn compiler_preflight_runs_help_and_requires_an_executable() {
        let mut probe = type_probe();
        probe.wyn = std::env::current_exe().unwrap();
        probe.verify_compiler().unwrap();

        let temp = tempfile::tempdir().unwrap();
        probe.wyn = temp.path().join("missing-wyn-compiler");
        let error = probe.verify_compiler().unwrap_err().to_string();
        assert!(error.contains("failed to run Wyn compiler preflight"));
    }

    #[test]
    fn parses_unused_warning_locations_as_utf8_byte_offsets() {
        let source = "def café = 1\nentry main() i32 = 0\n";
        let diagnostics = b"warning: unused definition `caf\xC3\xA9` is not reachable from any entry point\n  --> candidate.wyn:1:5\n";
        assert_eq!(
            parse_unused_warnings(source.as_bytes(), diagnostics),
            vec![UnusedWarning {
                kind: UnusedWarningKind::Declaration,
                start: 4,
                end: 9,
            }]
        );
    }

    #[test]
    fn warning_locations_create_semantic_cleanup_candidates() {
        let source =
            b"def dead = 1\nentry main(unused_parameter: i32) i32 = let unused_binding = 2 in ???\n";
        let diagnostics = b"warning: unused definition `dead` is not reachable from any entry point\n  --> candidate.wyn:1:5\nwarning: unused parameter `unused_parameter`; prefix its name with `_` to silence this warning\n  --> candidate.wyn:2:12\nwarning: unused binding `unused_binding`; prefix its name with `_` to silence this warning\n  --> candidate.wyn:2:45\n";
        let warnings = parse_unused_warnings(source, diagnostics);
        assert_eq!(warnings.len(), 3);
        let tree = parse(&language(), source).unwrap();
        let rendered: Vec<_> = collect_unused_warning_candidates(&tree, source, &warnings)
            .iter()
            .map(|candidate| String::from_utf8(apply_candidate(source, candidate)).unwrap())
            .collect();
        assert!(rendered.iter().any(|candidate| candidate
            == "entry main(unused_parameter: i32) i32 = let unused_binding = 2 in ???\n"));
        assert!(rendered.iter().any(|candidate| candidate.contains("entry main() i32")));
        assert!(rendered
            .iter()
            .any(|candidate| candidate.contains("entry main(unused_parameter: i32) i32 = ???")));
    }

    #[test]
    fn warning_postprocessing_reuses_the_accepted_check_output() {
        let source = b"def dead = 1\nentry main() i32 = bug()\n".to_vec();
        let diagnostics = b"warning: unused definition `dead` is not reachable from any entry point\n  --> candidate.wyn:1:5\n";
        let accepted = AcceptedWarnings::default();
        accepted.record(source.clone(), diagnostics).unwrap();
        let checks = Arc::new(AtomicUsize::new(0));
        let check = RecordingContains {
            needle: b"bug()",
            accepted: accepted.clone(),
            checks: checks.clone(),
        };

        let (reduced, stats) =
            postprocess_unused_warnings(&language(), source, &check, &accepted, 0).unwrap();

        assert_eq!(reduced, b"entry main() i32 = bug()\n");
        assert_eq!(stats.attempts, 1);
        assert_eq!(stats.accepted, 1);
        assert_eq!(checks.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn promotes_interesting_branch_out_of_if_expression() {
        let source = b"def main = if true then bug() else other()".to_vec();
        let (reduced, stats) =
            structural_reduce(&language(), source, &Contains(b"bug()"), &type_probe(), 1, 0).unwrap();
        assert_eq!(String::from_utf8(reduced).unwrap(), "def main = bug()");
        assert!(stats.accepted > 0);
    }

    #[test]
    fn parallel_structural_reduction_preserves_candidate_order() {
        let source = b"def main = if true then bug() else other()".to_vec();
        let sequential = structural_reduce(
            &language(),
            source.clone(),
            &Contains(b"bug()"),
            &type_probe(),
            1,
            0,
        )
        .unwrap();
        let parallel =
            structural_reduce(&language(), source, &Contains(b"bug()"), &type_probe(), 4, 0).unwrap();
        assert_eq!(parallel.0, sequential.0);
    }

    #[test]
    fn does_not_promote_elements_out_of_collections() {
        for source in [
            b"def f = [x, y]".as_slice(),
            b"def f = @[x, y]",
            b"def f = (x, y)",
        ] {
            let tree = parse(&language(), source).unwrap();
            let promoted: Vec<_> = collect_candidates(&tree, source)
                .iter()
                .filter(|candidate| candidate.kind == CandidateKind::Promotion)
                .map(|candidate| apply_candidate(source, candidate))
                .collect();
            assert!(!promoted.iter().any(|candidate| candidate == b"def f = x"));
            assert!(!promoted.iter().any(|candidate| candidate == b"def f = y"));
        }
    }

    #[test]
    fn avoids_low_probability_promotions() {
        let cases: &[(&[u8], &[&str], &[&str])] = &[
            (
                b"def f = let x = scalar in vector",
                &["def f = vector"],
                &["def f = scalar"],
            ),
            (
                b"def f = if condition then yes else no",
                &["def f = yes", "def f = no"],
                &["def f = condition"],
            ),
            (
                b"def f = normalize(vector)",
                &["def f = vector"],
                &["def f = normalize"],
            ),
        ];
        for (source, expected, forbidden) in cases {
            let tree = parse(&language(), source).unwrap();
            let promoted: Vec<_> = collect_candidates(&tree, source)
                .iter()
                .filter(|candidate| candidate.kind == CandidateKind::Promotion)
                .map(|candidate| String::from_utf8(apply_candidate(source, candidate)).unwrap())
                .collect();
            for candidate in *expected {
                assert!(promoted.iter().any(|value| value == candidate));
            }
            for candidate in *forbidden {
                assert!(!promoted.iter().any(|value| value == candidate));
            }
        }
    }

    #[test]
    fn promotes_let_body_only_when_it_does_not_reference_the_binding() {
        let cases: &[(&[u8], &str, bool)] = &[
            (b"def f = let x = value in unrelated", "def f = unrelated", true),
            (b"def f = let x = value in use(x)", "def f = use(x)", false),
            (b"def f = let (x, y) = pair in x + z", "def f = x + z", false),
        ];
        for (source, promoted_body, expected) in cases {
            let tree = parse(&language(), source).unwrap();
            let promoted: Vec<_> = collect_candidates(&tree, source)
                .iter()
                .filter(|candidate| candidate.kind == CandidateKind::Promotion)
                .map(|candidate| String::from_utf8(apply_candidate(source, candidate)).unwrap())
                .collect();
            assert_eq!(
                promoted.iter().any(|candidate| candidate == promoted_body),
                *expected
            );
        }
    }

    #[test]
    fn syntax_check_rejects_new_parse_errors() {
        let check = SyntaxCheck {
            inner: Contains(b"bug"),
            language: language(),
            reject_errors: true,
        };
        assert!(check.interesting(b"def f = bug()").unwrap());
        assert!(!check.interesting(b"def f = (bug()").unwrap());
    }

    #[test]
    fn list_deletion_consumes_an_adjacent_comma() {
        let source = b"def f(x, y) = x";
        let tree = parse(&language(), source).unwrap();
        let rendered: Vec<_> = collect_candidates(&tree, source)
            .iter()
            .filter(|candidate| candidate.kind == CandidateKind::ListDeletion)
            .map(|candidate| String::from_utf8(apply_candidate(source, candidate)).unwrap())
            .collect();
        assert!(rendered.iter().any(|candidate| candidate == "def f(y) = x"));
        assert!(rendered.iter().any(|candidate| candidate == "def f(x) = x"));
    }

    #[test]
    fn list_deletion_consumes_a_single_trailing_comma() {
        let source = b"def f(x,) = 0";
        let tree = parse(&language(), source).unwrap();
        let rendered: Vec<_> = collect_candidates(&tree, source)
            .iter()
            .filter(|candidate| candidate.kind == CandidateKind::ListDeletion)
            .map(|candidate| String::from_utf8(apply_candidate(source, candidate)).unwrap())
            .collect();
        assert!(rendered.iter().any(|candidate| candidate == "def f() = 0"));
    }

    #[test]
    fn tuple_deletion_can_collapse_a_trailing_comma_tuple() {
        let source = b"def f = (1, 2,)";
        let tree = parse(&language(), source).unwrap();
        let rendered: Vec<_> = collect_candidates(&tree, source)
            .iter()
            .filter(|candidate| candidate.kind == CandidateKind::ListDeletion)
            .map(|candidate| String::from_utf8(apply_candidate(source, candidate)).unwrap())
            .collect();
        assert!(rendered.iter().any(|candidate| candidate == "def f = (1)"));
    }

    #[test]
    fn concrete_replacements_run_before_holes() {
        let source = b"def f = 12345";
        let tree = parse(&language(), source).unwrap();
        let concrete: Vec<_> = collect_candidates(&tree, source)
            .iter()
            .filter(|candidate| candidate.kind == CandidateKind::Concrete)
            .map(|candidate| String::from_utf8(apply_candidate(source, candidate)).unwrap())
            .collect();
        let holes = hole_replacements();
        assert!(concrete.iter().any(|candidate| candidate == "def f = 0"));
        assert!(concrete.iter().any(|candidate| candidate == "def f = 1"));
        assert_eq!(holes["binary_expression"], ["???"]);
    }

    #[test]
    fn renders_defaults_for_compiler_type_names() {
        assert_eq!(default_literal("i32").as_deref(), Some("0"));
        assert_eq!(default_literal("f32").as_deref(), Some("0.0"));
        assert_eq!(default_literal("vec3f32").as_deref(), Some("@[0.0, 0.0, 0.0]"));
        assert_eq!(
            default_literal("(bool, vec2i32)").as_deref(),
            Some("(false, @[0, 0])")
        );
        assert_eq!(default_literal("[2]f32").as_deref(), Some("[0.0, 0.0]"));
        assert_eq!(default_literal("i32 -> i32"), None);
    }

    #[test]
    fn removes_all_comments_before_reduction() {
        let mut source = b"-- first\ndef f(x:f32) f32 = x -- second\n".to_vec();
        let tree = parse(&language(), &source).unwrap();
        assert_eq!(remove_comments(&tree, &mut source), (2, 17));
        assert_eq!(source, b"\ndef f(x:f32) f32 = x \n");
    }

    #[test]
    fn collapses_whitespace_only_line_runs() {
        let source = b"def f = 0\n   \n\t\n\n-- comment\n\n\n";
        assert_eq!(collapse_blank_lines(source), b"def f = 0\n\n-- comment\n\n");
    }

    #[test]
    fn uninteresting_stdout_requires_stdout_capture() {
        let error = Args::try_parse_from(["treereduce-wyn", "--uninteresting-stdout", "BLOCK", "true"])
            .unwrap_err();
        assert_eq!(error.kind(), clap::error::ErrorKind::MissingRequiredArgument);

        Args::try_parse_from([
            "treereduce-wyn",
            "--interesting-stdout",
            "BUG",
            "--uninteresting-stdout",
            "BLOCK",
            "true",
        ])
        .unwrap();
    }

    #[test]
    fn uninteresting_stderr_requires_stderr_capture() {
        let error = Args::try_parse_from(["treereduce-wyn", "--uninteresting-stderr", "BLOCK", "true"])
            .unwrap_err();
        assert_eq!(error.kind(), clap::error::ErrorKind::MissingRequiredArgument);

        Args::try_parse_from([
            "treereduce-wyn",
            "--interesting-stderr",
            "BUG",
            "--uninteresting-stderr",
            "BLOCK",
            "true",
        ])
        .unwrap();
    }

    #[test]
    fn inherited_output_cannot_also_be_matched() {
        let stdout_error = Args::try_parse_from([
            "treereduce-wyn",
            "--inherit-stdout",
            "--interesting-stdout",
            "BUG",
            "true",
        ])
        .unwrap_err();
        assert_eq!(stdout_error.kind(), clap::error::ErrorKind::ArgumentConflict);

        let stderr_error = Args::try_parse_from([
            "treereduce-wyn",
            "--inherit-stderr",
            "--interesting-stderr",
            "BUG",
            "true",
        ])
        .unwrap_err();
        assert_eq!(stderr_error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }
}
