//! Recursive expansion of `Declaration::Import` nodes against the filesystem.
//!
//! Each imported file's declarations replace the import node in-place;
//! transitive imports inside the loaded file are resolved relative to that
//! file's directory. A canonical-path dedup set prevents infinite loops on
//! cyclic imports and dedupes diamond imports.

use crate::LookupSet;
use std::path::{Path, PathBuf};

use crate::ast::{self, NodeCounter};
use crate::error::Result;
use crate::{err_module, err_parse, lexer, parser};

/// Recursively expand every `Declaration::Import(path)` in `decls` by parsing
/// the referenced file (relative to `base_dir`), resolving its own imports
/// (relative to its own directory), and inlining the resolved declarations.
///
/// Path resolution: `import "foo"` looks for `<base_dir>/foo.wyn`. The `.wyn`
/// extension is appended automatically when missing.
///
/// Cycle / re-import safety: each canonical path is loaded at most once per
/// compilation. Diamond imports work; cycles are silently broken at the
/// second encounter.
pub fn run(
    decls: Vec<ast::Declaration>,
    base_dir: &Path,
    node_counter: &mut NodeCounter,
) -> Result<Vec<ast::Declaration>> {
    let mut visited: LookupSet<PathBuf> = LookupSet::new();
    expand(decls, base_dir, node_counter, &mut visited)
}

fn expand(
    decls: Vec<ast::Declaration>,
    base_dir: &Path,
    node_counter: &mut NodeCounter,
    visited: &mut LookupSet<PathBuf>,
) -> Result<Vec<ast::Declaration>> {
    let mut out: Vec<ast::Declaration> = Vec::with_capacity(decls.len());
    for decl in decls {
        let ast::Declaration::Import(rel_path) = decl else {
            out.push(decl);
            continue;
        };

        let mut joined = base_dir.join(&rel_path);
        if joined.extension().is_none() {
            joined.set_extension("wyn");
        }
        let canonical = joined.canonicalize().map_err(|e| {
            err_module!(
                "import: cannot resolve `{}` (looked for `{}`): {}",
                rel_path,
                joined.display(),
                e
            )
        })?;
        if !visited.insert(canonical.clone()) {
            continue;
        }

        let source = std::fs::read_to_string(&canonical)
            .map_err(|e| err_module!("import: failed to read `{}`: {}", canonical.display(), e))?;
        let tokens = lexer::tokenize(&source).map_err(|e| err_parse!("{}", e))?;
        let mut p = parser::Parser::new(tokens, node_counter);
        let imported_program = p.parse()?;
        let imported_dir = canonical.parent().unwrap_or(base_dir);
        let resolved = expand(imported_program.declarations, imported_dir, node_counter, visited)?;
        out.extend(resolved);
    }
    Ok(out)
}
