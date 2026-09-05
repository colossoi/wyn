use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{ArgAction, Parser, ValueEnum};
use regex::Regex;
use tree_sitter::{Language, Node, Parser as TreeSitterParser, Tree};
use treereduce::{Check, CmdCheck, Config, NodeTypes, Original};

const DEFAULT_PASSES: usize = 2;
const DEFAULT_MIN_REDUCTION: usize = 2;

// These are tried sequentially before the type-hole fallback. Wyn's type
// checker acts as the oracle: candidates with the wrong type are simply
// uninteresting. Keeping this pass separate from treereduce prevents `???`
// from winning a race against a smaller, ordinary Wyn expression.
const CONCRETE_EXPRESSIONS: &[&str] = &["0", "1", "0.0", "1.0", "false", "true", "()", "[]", "@[]", "{}"];
const INTEGER_EXPRESSIONS: &[&str] = &["0", "1"];
const FLOAT_EXPRESSIONS: &[&str] = &["0.0", "1.0"];
const BOOLEAN_EXPRESSIONS: &[&str] = &["false", "true"];
const HOLE: &[&str] = &["???"];
const WILDCARD: &[&str] = &["_"];

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

#[derive(Clone, Debug, Default, Eq, PartialEq, ValueEnum)]
enum Comments {
    #[default]
    Remove,
    Keep,
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

    /// Remove all comments before reduction, or preserve every comment.
    #[arg(long, value_enum, default_value_t)]
    comments: Comments,

    /// Number of parallel interestingness checks used by treereduce.
    #[arg(short, long, default_value_t = default_jobs())]
    jobs: usize,

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

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Candidate {
    start: usize,
    end: usize,
    replacement: Vec<u8>,
    description: &'static str,
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

#[derive(Clone, Debug)]
struct PreserveComments<C> {
    inner: C,
    comments: Vec<Vec<u8>>,
}

#[derive(Debug)]
enum PreserveCommentsState<S> {
    Rejected,
    Inner(S),
}

impl<C: Check> Check for PreserveComments<C> {
    type State = PreserveCommentsState<C::State>;

    fn start(&self, source: &[u8]) -> io::Result<Self::State> {
        if has_exact_comments(source, &self.comments) {
            self.inner.start(source).map(PreserveCommentsState::Inner)
        } else {
            Ok(PreserveCommentsState::Rejected)
        }
    }

    fn cancel(&self, state: Self::State) -> io::Result<()> {
        match state {
            PreserveCommentsState::Rejected => Ok(()),
            PreserveCommentsState::Inner(state) => self.inner.cancel(state),
        }
    }

    fn try_wait(&self, state: &mut Self::State) -> io::Result<Option<bool>> {
        match state {
            PreserveCommentsState::Rejected => Ok(Some(false)),
            PreserveCommentsState::Inner(state) => self.inner.try_wait(state),
        }
    }

    fn wait(&self, state: Self::State) -> io::Result<bool> {
        match state {
            PreserveCommentsState::Rejected => Ok(false),
            PreserveCommentsState::Inner(state) => self.inner.wait(state),
        }
    }
}

fn default_jobs() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

fn main() -> Result<()> {
    let args = Args::parse();
    init_tracing(&args);

    if let Some(dir) = &args.temp_dir {
        fs::create_dir_all(dir)
            .with_context(|| format!("failed to create temporary directory {}", dir.display()))?;
    }

    let check = build_check(&args)?;
    let language: Language = tree_sitter_wyn::LANGUAGE.into();
    let mut source = read_source(&args)?;
    let initial_size = source.len();
    let initial_tree = parse(&language, &source)?;
    handle_initial_parse_errors(&args, &initial_tree)?;
    let comments = collect_comments(&initial_tree, &source);
    let preserved_comments = match args.comments {
        Comments::Remove => {
            let comments_size: usize = comments.iter().map(Vec::len).sum();
            remove_comments(&initial_tree, &mut source);
            if args.verbose > 0 {
                eprintln!(
                    "removed {} comments before reduction (-{comments_size} bytes)",
                    comments.len()
                );
            }
            Vec::new()
        }
        Comments::Keep => comments,
    };
    let check = PreserveComments {
        inner: check,
        comments: preserved_comments,
    };

    if args.verbose > 0 {
        eprintln!("verifying initial test case ({} bytes)", source.len());
    }
    if !args.no_verify && !check.interesting(&source)? {
        bail!("initial test case is not interesting");
    }

    let node_types = NodeTypes::new(tree_sitter_wyn::NODE_TYPES)
        .context("failed to read tree-sitter-wyn node-types.json")?;
    let start = Instant::now();
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
        let (next, stats) = structural_reduce(&language, source, &check, args.verbose)?;
        source = next;
        structural_stats.attempts += stats.attempts;
        structural_stats.accepted += stats.accepted;

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
            &language,
            &node_types,
            source,
            &check,
            args.jobs,
            min_reduction,
            args.slow,
            hole_replacements(),
        )?;

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

fn build_check(args: &Args) -> Result<CmdCheck> {
    let (cmd, command_args) = args.check.split_first().context("missing interestingness command")?;
    let regex = |value: &Option<String>| -> Result<Option<Regex>> {
        value
            .as_ref()
            .map(|pattern| Regex::new(pattern).with_context(|| format!("invalid regex: {pattern}")))
            .transpose()
    };

    Ok(CmdCheck::new(
        cmd.clone(),
        command_args.to_vec(),
        args.interesting_exit_code.clone(),
        args.temp_dir.as_ref().map(|path| path.to_string_lossy().into_owned()),
        regex(&args.interesting_stdout)?,
        regex(&args.interesting_stderr)?,
        regex(&args.uninteresting_stdout)?,
        regex(&args.uninteresting_stderr)?,
        args.inherit_stdout,
        args.inherit_stderr,
        args.timeout.map(Duration::from_secs),
    ))
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

fn collect_comments(tree: &Tree, source: &[u8]) -> Vec<Vec<u8>> {
    comment_ranges(tree).into_iter().map(|(start, end)| source[start..end].to_vec()).collect()
}

fn remove_comments(tree: &Tree, source: &mut Vec<u8>) {
    for (start, end) in comment_ranges(tree).into_iter().rev() {
        source.drain(start..end);
    }
}

fn has_exact_comments(source: &[u8], expected: &[Vec<u8>]) -> bool {
    let language: Language = tree_sitter_wyn::LANGUAGE.into();
    parse(&language, source).map(|tree| collect_comments(&tree, source) == expected).unwrap_or(false)
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
    verbose: u8,
) -> Result<(Vec<u8>, StructuralStats)> {
    let mut stats = StructuralStats::default();
    loop {
        let tree = parse(language, &source)?;
        let candidates = collect_candidates(&tree, &source);
        let mut accepted = false;
        for candidate in candidates {
            stats.attempts += 1;
            let next = apply_candidate(&source, &candidate);
            if check.interesting(&next)? {
                if verbose > 1 {
                    eprintln!(
                        "accepted {} at {}..{} (-{} bytes)",
                        candidate.description,
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
        right.reduction().cmp(&left.reduction()).then_with(|| left.start.cmp(&right.start))
    });
    candidates
}

fn collect_concrete_replacements(node: Node<'_>, candidates: &mut Vec<Candidate>) {
    let replacements = match node.kind() {
        "integer_literal" => INTEGER_EXPRESSIONS,
        "float_literal" => FLOAT_EXPRESSIONS,
        "boolean_literal" => BOOLEAN_EXPRESSIONS,
        kind if COMPOSITE_EXPRESSION_KINDS.contains(&kind) => CONCRETE_EXPRESSIONS,
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
            description: "concrete replacement",
        });
    }
}

fn collect_promotions(node: Node<'_>, source: &[u8], candidates: &mut Vec<Candidate>) {
    match node.kind() {
        "let_expression" => promote_fields(node, &["value", "body"], source, candidates),
        "if_expression" => promote_fields(node, &["condition", "then", "else"], source, candidates),
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
        "call_expression" | "array_literal" | "vec_literal" | "tuple_expression" => {
            promote_direct_expressions(node, source, candidates)
        }
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
            description: "child promotion",
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
                    description: "case deletion",
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
            description: "list element deletion",
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
                description: "list element deletion",
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

    fn language() -> Language {
        tree_sitter_wyn::LANGUAGE.into()
    }

    #[test]
    fn promotes_interesting_branch_out_of_if_expression() {
        let source = b"def main = if true then bug() else other()".to_vec();
        let (reduced, stats) = structural_reduce(&language(), source, &Contains(b"bug()"), 0).unwrap();
        assert_eq!(String::from_utf8(reduced).unwrap(), "def main = bug()");
        assert!(stats.accepted > 0);
    }

    #[test]
    fn list_deletion_consumes_an_adjacent_comma() {
        let source = b"def f(x, y) = x";
        let tree = parse(&language(), source).unwrap();
        let rendered: Vec<_> = collect_candidates(&tree, source)
            .iter()
            .filter(|candidate| candidate.description == "list element deletion")
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
            .filter(|candidate| candidate.description == "list element deletion")
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
            .filter(|candidate| candidate.description == "list element deletion")
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
            .filter(|candidate| candidate.description == "concrete replacement")
            .map(|candidate| String::from_utf8(apply_candidate(source, candidate)).unwrap())
            .collect();
        let holes = hole_replacements();
        assert!(concrete.iter().any(|candidate| candidate == "def f = 0"));
        assert!(concrete.iter().any(|candidate| candidate == "def f = 1"));
        assert_eq!(holes["binary_expression"], ["???"]);
    }

    #[test]
    fn removes_all_comments_before_reduction() {
        let mut source = b"-- first\ndef f(x:f32) f32 = x -- second\n".to_vec();
        let tree = parse(&language(), &source).unwrap();
        remove_comments(&tree, &mut source);
        assert_eq!(source, b"\ndef f(x:f32) f32 = x \n");
    }

    #[test]
    fn preserve_comments_check_rejects_comment_deletion() {
        let check = PreserveComments {
            inner: Contains(b"bug"),
            comments: vec![b"-- keep one".to_vec(), b"-- keep two".to_vec()],
        };
        assert!(check.interesting(b"-- keep one\nbug\n-- keep two").unwrap());
        assert!(!check.interesting(b"-- keep one\nbug").unwrap());
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
