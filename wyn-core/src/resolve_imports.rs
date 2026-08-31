//! Recursive expansion of `Declaration::Import` nodes against the filesystem.
//!
//! Each imported file's declarations replace the import node in-place;
//! transitive imports inside the loaded file are resolved relative to that
//! file's directory. A canonical-path dedup set prevents infinite loops on
//! cyclic imports and dedupes diamond imports.

use crate::interface;
use crate::module_manager;
use crate::LookupSet;
use std::path::{Path, PathBuf};

use crate::ast::{self, ImportsResolvedFrontend, ParsedFrontend};
use crate::error::Result;
use crate::{err_module, err_parse, lexer, parser};
use wyn_module_graph::ModuleId;

pub type ImportsResolvedFamily = ast::AstFamily<
    ast::SourceTree,
    ast::DefinitionSyntax,
    ast::EntrySyntax,
    interface::Attribute,
    ast::ExternSyntax,
    ast::ImportsResolvedFrontend<ast::NestedDeclaration>,
>;

/// AST after every top-level file import has been expanded.
#[derive(Debug, Clone, Copy)]
pub enum ImportsResolvedTag {}
pub type ImportsResolved =
    ast::Program<ImportsResolvedTag, ImportsResolvedFamily, module_manager::ModuleManager>;

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
pub fn resolve_imports(program: parser::Parsed, base_dir: &Path) -> Result<ImportsResolved> {
    let mut visited: LookupSet<PathBuf> = LookupSet::new();
    let graphics = program.global_context.options().graphics;
    program.try_rebuild(|declarations, global_context, node_ids| {
        Ok((
            expand(declarations, base_dir, node_ids, &mut visited, graphics)?,
            global_context,
        ))
    })
}

fn expand(
    decls: Vec<ast::Declaration<parser::ParsedFamily>>,
    base_dir: &Path,
    node_counter: &mut ast::NodeCounter,
    visited: &mut LookupSet<PathBuf>,
    graphics: bool,
) -> Result<Vec<ast::Declaration<ImportsResolvedFamily>>> {
    let mut out = Vec::with_capacity(decls.len());
    for decl in decls {
        let rel_path = match decl {
            ast::Declaration::Decl(decl) => {
                out.push(ast::Declaration::Decl(decl));
                continue;
            }
            ast::Declaration::Entry(entry) => {
                out.push(ast::Declaration::Entry(entry));
                continue;
            }
            ast::Declaration::Extern(ext) => {
                out.push(ast::Declaration::Extern(ext));
                continue;
            }
            ast::Declaration::Frontend(frontend) => match frontend {
                ParsedFrontend::Sig(sig) => {
                    out.push(ast::Declaration::Frontend(ImportsResolvedFrontend::Sig(sig)));
                    continue;
                }
                ParsedFrontend::TypeBind(bind) => {
                    out.push(ast::Declaration::Frontend(ImportsResolvedFrontend::TypeBind(
                        bind,
                    )));
                    continue;
                }
                ParsedFrontend::Module(module) => {
                    out.push(ast::Declaration::Frontend(ImportsResolvedFrontend::Module(
                        module,
                    )));
                    continue;
                }
                ParsedFrontend::ModuleTypeBind(bind) => {
                    out.push(ast::Declaration::Frontend(
                        ImportsResolvedFrontend::ModuleTypeBind(bind),
                    ));
                    continue;
                }
                ParsedFrontend::Open(open) => {
                    out.push(ast::Declaration::Frontend(ImportsResolvedFrontend::Open(open)));
                    continue;
                }
                ParsedFrontend::Resource(resource) => {
                    out.push(ast::Declaration::Frontend(ImportsResolvedFrontend::Resource(
                        resource,
                    )));
                    continue;
                }
                ParsedFrontend::Import(import) => import,
            },
        };

        let mut joined = base_dir.join(&rel_path.path);
        if joined.extension().is_none() {
            joined.set_extension("wyn");
        }
        let canonical = joined.canonicalize().map_err(|e| {
            err_module!(
                "import: cannot resolve `{}` (looked for `{}`): {}",
                rel_path.path,
                joined.display(),
                e
            )
        })?;
        if !visited.insert(canonical.clone()) {
            continue;
        }

        let source = std::fs::read_to_string(&canonical)
            .map_err(|e| err_module!("import: failed to read `{}`: {}", canonical.display(), e))?;
        let module = ModuleId::from(0);
        let tokens = lexer::tokenize(module, &source).map_err(|e| err_parse!("{}", e))?;
        let mut p = parser::Parser::with_graphics(module, tokens, node_counter, graphics);
        let imported_declarations = p.parse()?;
        let imported_dir = canonical.parent().unwrap_or(base_dir);
        let resolved = expand(
            imported_declarations,
            imported_dir,
            node_counter,
            visited,
            graphics,
        )?;
        out.extend(resolved);
    }
    Ok(out)
}
