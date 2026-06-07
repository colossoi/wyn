//! Extract `-- |` doc comments from Wyn source files into Markdown,
//! ready to drop into an mdBook.
//!
//! Convention (Haddock-style):
//!
//!   * `-- | <markdown>` opens a doc-block. Continuation lines must be
//!     plain `-- …` and stop at the first non-comment / blank line.
//!     The block attaches to the next item declaration.
//!   * `-- ^ <markdown>` attaches to the *preceding* identifier — used
//!     for inline parameter docs. v1 captures one line; multi-line `-- ^`
//!     blocks are a TODO.
//!   * "Item declarations" are top-level `def …`, `module …`, `module
//!     type …`, `type …`, and (inside module-type braces) `sig … : …`.
//!
//! Inside the doc-block the text is plain Markdown — backticks, links,
//! lists, the lot. The extractor doesn't try to validate it.
//!
//! Output: one Markdown file per input `.wyn` file plus an `index.md`
//! summary. Drop those into an mdBook `src/` and `SUMMARY.md` and you
//! have rendered prelude docs.

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    about = "Extract `-- |` doc comments from Wyn source files into Markdown",
    long_about = None,
)]
struct Cli {
    /// Source files / directories to scan. Directories are walked
    /// non-recursively for `*.wyn`.
    #[arg(value_name = "PATH", required = true)]
    inputs: Vec<PathBuf>,

    /// Output directory. Created if missing. One `.md` per input file
    /// plus an `index.md` summary.
    #[arg(short = 'o', long = "output", value_name = "DIR")]
    output: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let files = collect_inputs(&cli.inputs)?;
    if files.is_empty() {
        return Err(anyhow!("no .wyn files found in inputs"));
    }
    fs::create_dir_all(&cli.output)
        .with_context(|| format!("creating output dir {}", cli.output.display()))?;

    let mut modules: Vec<ModuleDoc> = Vec::new();
    for src in &files {
        let raw = fs::read_to_string(src).with_context(|| format!("reading {}", src.display()))?;
        let module = extract_module(src, &raw)?;
        modules.push(module);
    }

    for m in &modules {
        let path = cli.output.join(format!("{}.md", m.stem));
        fs::write(&path, render_module_md(m)).with_context(|| format!("writing {}", path.display()))?;
    }
    fs::write(cli.output.join("index.md"), render_index_md(&modules))?;

    eprintln!(
        "wrote {} module pages + index to {}",
        modules.len(),
        cli.output.display()
    );
    Ok(())
}

fn collect_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = Vec::new();
    for p in inputs {
        let meta = fs::metadata(p).with_context(|| format!("stat {}", p.display()))?;
        if meta.is_dir() {
            for entry in fs::read_dir(p)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("wyn") {
                    out.push(path);
                }
            }
        } else if p.extension().and_then(|e| e.to_str()) == Some("wyn") {
            out.push(p.clone());
        }
    }
    out.sort();
    Ok(out)
}

// -----------------------------------------------------------------------------
// Doc model
// -----------------------------------------------------------------------------

/// One source file's worth of extracted docs.
struct ModuleDoc {
    /// `prelude/math.wyn` → `math`. Used as the output filename stem.
    stem: String,
    /// File-level intro: the `-- |` block at the top of the file with
    /// no following item attachment (i.e. the first block, separated
    /// from any item by a blank line). `None` if absent.
    file_intro: Option<String>,
    items: Vec<ItemDoc>,
}

/// One documented item. Undocumented items aren't emitted.
struct ItemDoc {
    kind: ItemKind,
    /// Fully-qualified-ish display name. For `module foo = { def bar … }`
    /// this is `foo.bar`. For a bare top-level it's just the bare name.
    qualified_name: String,
    /// The raw declaration line(s) — the user's signature, lightly
    /// trimmed. Rendered as a code block under the heading.
    signature: String,
    /// Markdown body extracted from the attached `-- |` block.
    body: String,
    /// Per-parameter `-- ^` notes, in declaration order.
    param_notes: Vec<ParamNote>,
    /// 1-based line number of the declaration in the source file —
    /// used by the renderer to emit a "view source" anchor.
    line: usize,
}

#[derive(Copy, Clone)]
enum ItemKind {
    Def,
    Module,
    ModuleType,
    Functor,
    Type,
    Sig,
}

impl ItemKind {
    fn label(self) -> &'static str {
        match self {
            ItemKind::Def => "def",
            ItemKind::Module => "module",
            ItemKind::ModuleType => "module type",
            ItemKind::Functor => "functor",
            ItemKind::Type => "type",
            ItemKind::Sig => "sig",
        }
    }
}

struct ParamNote {
    /// The identifier the `-- ^` attaches to. Empty if we couldn't
    /// recover it from the preceding line (we still emit the note).
    ident: String,
    body: String,
}

// -----------------------------------------------------------------------------
// Extraction
// -----------------------------------------------------------------------------

fn extract_module(src_path: &Path, source: &str) -> Result<ModuleDoc> {
    let stem = src_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("no file stem for {}", src_path.display()))?
        .to_string();
    let lines: Vec<&str> = source.lines().collect();

    let mut items: Vec<ItemDoc> = Vec::new();
    let mut file_intro: Option<String> = None;

    // Active doc-block state: when we encounter `-- | …` we start
    // accumulating; we close (and attach or stash) when we hit a
    // non-comment line or a blank line.
    let mut pending_block: Option<DocBlock> = None;
    // Module nesting: each entry is the module-or-module-type name on
    // the stack. We track this so a `def` inside `module foo = { … }`
    // is qualified as `foo.<def>`.
    let mut scope: Vec<String> = Vec::new();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();

        // 1. Start of a new doc-block.
        if let Some(rest) = trimmed.strip_prefix("-- | ").or_else(|| {
            // Bare `-- |` with no following text is still a valid start
            // (an empty leading line).
            if trimmed.trim_end() == "-- |" { Some("") } else { None }
        }) {
            // Stash any orphan block (e.g. file intro with a blank line
            // between it and the next item).
            if let Some(block) = pending_block.take() {
                if file_intro.is_none() && items.is_empty() {
                    file_intro = Some(block.body);
                }
            }
            pending_block = Some(DocBlock {
                body: rest.to_string(),
                start_line: i + 1,
            });
            i += 1;
            continue;
        }

        // 2. Continuation of an active doc-block: plain `-- …` lines.
        if let Some(ref mut block) = pending_block {
            if let Some(rest) = strip_comment(trimmed) {
                // Don't fold `-- |` (handled above) or `-- ^` (handled
                // below) — those open new things.
                if !rest.starts_with('|') && !rest.starts_with('^') {
                    block.body.push('\n');
                    block.body.push_str(rest);
                    i += 1;
                    continue;
                }
            }
            // Blank line closes the block. Stash it as a possible
            // file intro if nothing's been documented yet; otherwise
            // it's an orphan and we drop it (with no diagnostic, to
            // keep extraction tolerant).
            if trimmed.is_empty() {
                if let Some(block) = pending_block.take() {
                    if file_intro.is_none() && items.is_empty() {
                        file_intro = Some(block.body);
                    }
                }
                i += 1;
                continue;
            }
        }

        // 3. `-- ^` notes are picked up inside `collect_signature` —
        //    no standalone handling needed here.

        // 4. An item declaration. If we have a pending doc-block, this
        //    is its target.
        if let Some(item_kind) = item_kind_of_line(trimmed) {
            // Multi-line signatures: keep collecting lines until balanced
            // parens and the body assignment `=` appears, or we run out
            // of "signature-looking" content. `-- ^` notes embedded
            // inside the signature get pulled out as param_notes.
            let (sig_text, param_notes, lines_consumed) = collect_signature(&lines, i);
            let name = guess_item_name(trimmed, item_kind);
            let qualified_name =
                if scope.is_empty() { name.clone() } else { format!("{}.{}", scope.join("."), name) };

            if let Some(block) = pending_block.take() {
                items.push(ItemDoc {
                    kind: item_kind,
                    qualified_name,
                    signature: sig_text,
                    body: block.body,
                    param_notes,
                    line: i + 1,
                });
            }
            // Track module / module-type / functor entry. Bodies open
            // with `= {` on the declaration line; v1 assumes `{` is on
            // the same line as the keyword.
            if matches!(
                item_kind,
                ItemKind::Module | ItemKind::ModuleType | ItemKind::Functor
            ) && trimmed.contains('{')
            {
                scope.push(guess_item_name(trimmed, item_kind));
            }
            i += lines_consumed;
            continue;
        }

        // 5. Module-body close: bare `}` line.
        if trimmed == "}" {
            scope.pop();
            i += 1;
            continue;
        }

        // 6. Any other line drops a pending doc-block.
        if pending_block.is_some() && !trimmed.is_empty() {
            pending_block = None;
        }
        i += 1;
    }

    // Trailing orphan (file ends in a `-- |` block).
    if let Some(block) = pending_block.take() {
        if file_intro.is_none() && items.is_empty() {
            file_intro = Some(block.body);
        }
    }

    Ok(ModuleDoc {
        stem,
        file_intro,
        items,
    })
}

struct DocBlock {
    body: String,
    #[allow(dead_code)]
    start_line: usize,
}

/// Strip a leading `-- ` (or `--` with nothing after) from `s`. Returns
/// `None` for lines that aren't `--`-prefixed comments.
fn strip_comment(s: &str) -> Option<&str> {
    let t = s.trim_end();
    if t == "--" {
        return Some("");
    }
    t.strip_prefix("-- ")
}

/// Recognise the first keyword of an item declaration. Conservative:
/// only fires on lines whose first non-whitespace word is exactly one
/// of the known item keywords.
fn item_kind_of_line(trimmed: &str) -> Option<ItemKind> {
    let first = trimmed.split_whitespace().next()?;
    match first {
        "def" => Some(ItemKind::Def),
        "type" => Some(ItemKind::Type),
        "sig" => Some(ItemKind::Sig),
        "functor" => Some(ItemKind::Functor),
        "module" => {
            // `module type foo = …` vs `module foo = …`.
            let rest = trimmed.trim_start_matches("module").trim_start();
            if rest.starts_with("type") { Some(ItemKind::ModuleType) } else { Some(ItemKind::Module) }
        }
        _ => None,
    }
}

/// Pull the item's bare identifier from its declaration line. Robust to
/// `def foo(...)` / `def (+) …` / `module foo = …` shapes; falls back to
/// `"<anon>"` for anything weird.
fn guess_item_name(trimmed: &str, kind: ItemKind) -> String {
    // Strip the kind keyword (and `module type`'s second keyword).
    let stripped = match kind {
        ItemKind::ModuleType => trimmed
            .strip_prefix("module")
            .unwrap_or(trimmed)
            .trim_start()
            .strip_prefix("type")
            .unwrap_or(trimmed)
            .trim_start(),
        _ => trimmed.strip_prefix(kind.label()).unwrap_or(trimmed).trim_start(),
    };
    // Operator name like `(+)` — keep the parens.
    if stripped.starts_with('(') {
        if let Some(end) = stripped.find(')') {
            return stripped[..=end].to_string();
        }
    }
    // Identifier: read until the next delimiter.
    let end =
        stripped.find(|c: char| !c.is_alphanumeric() && c != '_' && c != '\'').unwrap_or(stripped.len());
    if end == 0 {
        return "<anon>".to_string();
    }
    stripped[..end].to_string()
}

/// Collect signature text starting at `start`. Returns
/// `(joined_signature, param_notes, lines_consumed)`. A signature ends
/// when:
///   * we reach a line containing ` = ` at the top level (function body
///     starts), OR
///   * we close the parenthesis balance opened on the first line and
///     don't see ` = ` (a `sig … : …` declaration), OR
///   * we hit a blank line.
///
/// `-- ^` notes appearing inline on parameter lines are extracted as
/// `ParamNote`s and stripped from the rendered signature.
fn collect_signature(lines: &[&str], start: usize) -> (String, Vec<ParamNote>, usize) {
    // Bracket depth: only `()` and `[]` continue a signature line.
    // `{` opens a module / sig body and acts as a terminator — we
    // want the signature to be the declaration line(s) up to and
    // including `{`, never to span the body.
    let mut depth: i32 = 0;
    let mut out_lines: Vec<String> = Vec::new();
    let mut param_notes: Vec<ParamNote> = Vec::new();
    // True iff the previous line opened or continued a `-- ^` note,
    // so a bare `-- …` continuation on the next line appends to it.
    let mut in_caret_note = false;
    let mut i = start;
    while i < lines.len() {
        let line = lines[i];
        // Pure continuation comment: `   -- text` with no Wyn code on
        // the line. If a `-- ^` note is open, append; either way the
        // line stays out of the rendered signature.
        let trimmed_full = line.trim();
        if let Some(continuation) = trimmed_full.strip_prefix("--") {
            if !continuation.starts_with('|') && !continuation.starts_with('^') && in_caret_note {
                if let Some(last) = param_notes.last_mut() {
                    last.body.push(' ');
                    last.body.push_str(continuation.trim());
                }
                i += 1;
                continue;
            }
            if !continuation.starts_with('|') && !continuation.starts_with('^') {
                // Bare comment unattached to a `-- ^` note — drop from
                // the signature, treat as a no-op for bracket counting.
                i += 1;
                continue;
            }
        }

        // Split off any `-- ^` parameter note before counting brackets.
        let (sig_part, note) = split_caret_note(line);
        if let Some((ident, body)) = note {
            param_notes.push(ParamNote { ident, body });
            in_caret_note = true;
        } else {
            in_caret_note = false;
        }
        out_lines.push(sig_part.trim_end().to_string());
        let mut opens_body = false;
        for c in sig_part.chars() {
            match c {
                '(' | '[' => depth += 1,
                ')' | ']' => depth -= 1,
                '{' => opens_body = true,
                _ => {}
            }
        }
        let trimmed = sig_part.trim();
        let has_eq = signature_has_body_eq(&sig_part);
        if depth <= 0
            && (has_eq || opens_body || trimmed.is_empty() || is_terminal_sig_line(trimmed))
        {
            i += 1;
            break;
        }
        i += 1;
    }
    let joined = out_lines.join("\n");
    let trimmed = trim_signature_body(&joined).to_string();
    (trimmed, param_notes, i - start)
}

/// Split a source line into `(signature_part, Option<(ident, note)>)`.
/// A `-- ^` comment somewhere on the line is recognised as a parameter
/// note attached to the identifier *immediately before* it (stripped of
/// trailing `:`/`,`/`=`/`(`); everything from `--` onwards is removed
/// from `signature_part`.
fn split_caret_note(line: &str) -> (String, Option<(String, String)>) {
    let Some(idx) = line.find("-- ^") else {
        return (line.to_string(), None);
    };
    let (left, right) = line.split_at(idx);
    let body = right.trim_start_matches("-- ^").trim();
    // The parameter identifier comes from the `<name>: <type>` form on
    // the same line — i.e. the last token before the rightmost `:` on
    // the sig portion. Falls back to the last whitespace-separated
    // token if there's no `:` (e.g. a positional binding).
    let pre = left.trim_end().trim_end_matches(',').trim_end();
    let ident = if let Some(colon_idx) = pre.rfind(':') {
        let before_colon = pre[..colon_idx].trim_end();
        before_colon
            .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '\'')
            .filter(|s| !s.is_empty())
            .next_back()
            .unwrap_or("")
            .to_string()
    } else {
        pre.split_whitespace().next_back().unwrap_or("").to_string()
    };
    let sig_part = left.trim_end().to_string();
    (sig_part, Some((ident, body.to_string())))
}

/// True if the line contains a top-level ` = ` that opens the function
/// body. We don't try to handle `=` inside type expressions (Wyn
/// doesn't use it that way at this level) — a string-search is fine.
fn signature_has_body_eq(line: &str) -> bool {
    // Avoid matching inside line comments.
    let pre_comment = line.split("--").next().unwrap_or(line);
    pre_comment.contains(" = ") || pre_comment.trim_end().ends_with('=')
}

fn is_terminal_sig_line(trimmed: &str) -> bool {
    // `sig name : type` and `type t` end naturally on one line — there's
    // no body assignment.
    trimmed.starts_with("sig ") || trimmed.starts_with("type ")
}

/// Cut everything from the body-`=` onwards: docs show the signature,
/// not the implementation.
fn trim_signature_body(sig: &str) -> &str {
    if let Some(idx) = sig.find(" = ") {
        sig[..idx].trim_end()
    } else if let Some(stripped) = sig.strip_suffix('=') {
        stripped.trim_end()
    } else {
        sig.trim_end()
    }
}

// -----------------------------------------------------------------------------
// Markdown rendering
// -----------------------------------------------------------------------------

fn render_module_md(m: &ModuleDoc) -> String {
    let mut out = String::new();
    out.push_str(&format!("# {}\n\n", m.stem));
    if let Some(intro) = &m.file_intro {
        out.push_str(intro);
        out.push_str("\n\n");
    }
    if m.items.is_empty() {
        out.push_str("_(no documented items)_\n");
        return out;
    }
    for item in &m.items {
        out.push_str(&format!("## `{}`\n\n", item.qualified_name));
        out.push_str("```wyn\n");
        out.push_str(item.signature.trim_end());
        out.push_str("\n```\n\n");
        out.push_str(item.body.trim());
        out.push('\n');
        if !item.param_notes.is_empty() {
            out.push_str("\n**Parameters:**\n\n");
            for note in &item.param_notes {
                if note.ident.is_empty() {
                    out.push_str(&format!("- {}\n", note.body.trim()));
                } else {
                    out.push_str(&format!("- `{}` — {}\n", note.ident, note.body.trim()));
                }
            }
        }
        out.push_str(&format!(
            "\n<sub>{} at `{}.wyn:{}`</sub>\n\n",
            item.kind.label(),
            m.stem,
            item.line,
        ));
    }
    out
}

fn render_index_md(modules: &[ModuleDoc]) -> String {
    let mut out = String::new();
    out.push_str("# Wyn Prelude\n\n");
    out.push_str("Generated from `prelude/*.wyn` by `wyn-doc`. ");
    out.push_str("Doc-comment convention: `-- |` above an item, `-- ^` after a param.\n\n");
    for m in modules {
        let count = m.items.len();
        out.push_str(&format!(
            "- [`{}`]({}.md) — {} documented item{}\n",
            m.stem,
            m.stem,
            count,
            if count == 1 { "" } else { "s" },
        ));
    }
    out
}
