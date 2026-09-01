//! Resolution of parsed source imports through a closed source-module graph.

use std::sync::Arc;

use crate::ast::{
    AstFamily, Declaration, DefinitionSyntax, EntryDecl, EntrySyntax, ExternSyntax,
    ImportsResolvedFrontend, ModuleDecl, ModuleExpression, NestedDeclaration, ParsedFrontend, Program,
    SourceImport, SourceTree,
};
use crate::error::{CompilerError, FrontendFailure, Result};
use crate::frontend::ParsedModules;
use crate::interface::Attribute;
use crate::parser::ParsedFile;
use crate::semantic_modules::SemanticModules;
use crate::{err_module_at, LookupSet};
use wyn_module_graph::{ModuleGraph, ModuleId};

pub type ImportsResolvedFamily = AstFamily<
    SourceTree,
    DefinitionSyntax,
    EntrySyntax,
    Attribute,
    ExternSyntax,
    ImportsResolvedFrontend<NestedDeclaration>,
>;

/// AST after every physical source import has been replaced with loaded syntax.
#[derive(Debug, Clone, Copy)]
pub enum ImportsResolvedTag {}
pub type ImportsResolved = Program<ImportsResolvedTag, ImportsResolvedFamily, SemanticModules>;

/// Resolve physical imports and combine the loaded source modules into one
/// whole-program AST.
pub fn resolve_imports(modules: ParsedModules) -> std::result::Result<ImportsResolved, FrontendFailure> {
    let ParsedModules {
        graph,
        node_ids,
        semantic_modules,
    } = modules;
    let declarations = {
        let mut resolver = ImportResolver::new(&graph);
        resolver.resolve_top_level(graph.root())
    };
    let source_graph = Arc::new(graph.erase_syntax());
    let declarations =
        declarations.map_err(|error| FrontendFailure::new(error, Arc::clone(&source_graph)))?;
    Ok(Program {
        declarations,
        node_ids,
        source_graph,
        global_context: semantic_modules,
        state: std::marker::PhantomData,
    })
}

struct ImportResolver<'a> {
    graph: &'a ModuleGraph<ParsedFile>,
    injected_modules: LookupSet<ModuleId>,
}

impl<'a> ImportResolver<'a> {
    fn new(graph: &'a ModuleGraph<ParsedFile>) -> Self {
        Self {
            graph,
            injected_modules: LookupSet::new(),
        }
    }

    fn resolve_top_level(&mut self, module: ModuleId) -> Result<Vec<Declaration<ImportsResolvedFamily>>> {
        if !self.injected_modules.insert(module) {
            return Ok(Vec::new());
        }

        let declarations = self.parsed_file(module)?.declarations.clone();
        let mut resolved = Vec::with_capacity(declarations.len());
        for declaration in declarations {
            match declaration {
                Declaration::Decl(declaration) => {
                    resolved.push(Declaration::Decl(declaration));
                }
                Declaration::Entry(declaration) => {
                    self.require_root_entry(module, &declaration)?;
                    resolved.push(Declaration::Entry(declaration));
                }
                Declaration::Extern(declaration) => {
                    resolved.push(Declaration::Extern(declaration));
                }
                Declaration::Frontend(frontend) => match frontend {
                    ParsedFrontend::Sig(declaration) => {
                        resolved.push(Declaration::Frontend(ImportsResolvedFrontend::Sig(declaration)))
                    }
                    ParsedFrontend::TypeBind(declaration) => resolved.push(Declaration::Frontend(
                        ImportsResolvedFrontend::TypeBind(declaration),
                    )),
                    ParsedFrontend::Module(declaration) => {
                        let declaration = self.resolve_module_declaration(module, declaration)?;
                        resolved.push(Declaration::Frontend(ImportsResolvedFrontend::Module(
                            declaration,
                        )));
                    }
                    ParsedFrontend::ModuleTypeBind(declaration) => resolved.push(Declaration::Frontend(
                        ImportsResolvedFrontend::ModuleTypeBind(declaration),
                    )),
                    ParsedFrontend::Open(expression) => {
                        let expression = self.resolve_module_expression(module, expression)?;
                        resolved.push(Declaration::Frontend(ImportsResolvedFrontend::Open(expression)));
                    }
                    ParsedFrontend::Import(import) => {
                        let target = self.import_target(module, &import)?;
                        resolved.extend(self.resolve_top_level(target)?);
                    }
                    ParsedFrontend::Resource(declaration) => resolved.push(Declaration::Frontend(
                        ImportsResolvedFrontend::Resource(declaration),
                    )),
                },
            }
        }
        Ok(resolved)
    }

    fn resolve_module_declaration(&self, module: ModuleId, declaration: ModuleDecl) -> Result<ModuleDecl> {
        Ok(match declaration {
            ModuleDecl::Module {
                name,
                signature,
                body,
            } => ModuleDecl::Module {
                name,
                signature,
                body: self.resolve_module_expression(module, body)?,
            },
            ModuleDecl::Functor { name, params, body } => ModuleDecl::Functor {
                name,
                params,
                body: self.resolve_module_expression(module, body)?,
            },
        })
    }

    fn resolve_module_expression(
        &self,
        module: ModuleId,
        expression: ModuleExpression,
    ) -> Result<ModuleExpression> {
        Ok(match expression {
            ModuleExpression::Name(name) => ModuleExpression::Name(name),
            ModuleExpression::Ascription(expression, signature) => ModuleExpression::Ascription(
                Box::new(self.resolve_module_expression(module, *expression)?),
                signature,
            ),
            ModuleExpression::Lambda(parameters, signature, body) => ModuleExpression::Lambda(
                parameters,
                signature,
                Box::new(self.resolve_module_expression(module, *body)?),
            ),
            ModuleExpression::Application(function, argument) => ModuleExpression::Application(
                Box::new(self.resolve_module_expression(module, *function)?),
                Box::new(self.resolve_module_expression(module, *argument)?),
            ),
            ModuleExpression::Struct(declarations) => {
                ModuleExpression::Struct(self.resolve_nested_declarations(module, declarations)?)
            }
            ModuleExpression::Import(import) => {
                let target = self.import_target(module, &import)?;
                ModuleExpression::Struct(self.resolve_source_as_nested(target)?)
            }
        })
    }

    fn resolve_nested_declarations(
        &self,
        module: ModuleId,
        declarations: Vec<NestedDeclaration>,
    ) -> Result<Vec<NestedDeclaration>> {
        let mut resolved = Vec::with_capacity(declarations.len());
        for declaration in declarations {
            match declaration {
                NestedDeclaration::Module(declaration) => resolved.push(NestedDeclaration::Module(
                    self.resolve_module_declaration(module, declaration)?,
                )),
                NestedDeclaration::Open(expression) => resolved.push(NestedDeclaration::Open(
                    self.resolve_module_expression(module, expression)?,
                )),
                NestedDeclaration::Import(import) => {
                    let target = self.import_target(module, &import)?;
                    resolved.extend(self.resolve_source_as_nested(target)?);
                }
                NestedDeclaration::Decl(declaration) => {
                    resolved.push(NestedDeclaration::Decl(declaration));
                }
                NestedDeclaration::Entry(declaration) => {
                    return Err(err_module_at!(
                        declaration.name_span,
                        "entry `{}` is not declared directly in the root source module",
                        declaration.name
                    ));
                }
                NestedDeclaration::Sig(declaration) => {
                    resolved.push(NestedDeclaration::Sig(declaration));
                }
                NestedDeclaration::Extern(declaration) => {
                    resolved.push(NestedDeclaration::Extern(declaration));
                }
                NestedDeclaration::TypeBind(declaration) => {
                    resolved.push(NestedDeclaration::TypeBind(declaration));
                }
                NestedDeclaration::ModuleTypeBind(declaration) => {
                    resolved.push(NestedDeclaration::ModuleTypeBind(declaration));
                }
                NestedDeclaration::Resource(declaration) => {
                    resolved.push(NestedDeclaration::Resource(declaration));
                }
            }
        }
        Ok(resolved)
    }

    fn resolve_source_as_nested(&self, module: ModuleId) -> Result<Vec<NestedDeclaration>> {
        let declarations = self.parsed_file(module)?.declarations.clone();
        let mut resolved = Vec::with_capacity(declarations.len());
        for declaration in declarations {
            match declaration {
                Declaration::Decl(declaration) => {
                    resolved.push(NestedDeclaration::Decl(declaration));
                }
                Declaration::Entry(declaration) => {
                    self.require_root_entry(module, &declaration)?;
                    resolved.push(NestedDeclaration::Entry(declaration));
                }
                Declaration::Extern(declaration) => {
                    resolved.push(NestedDeclaration::Extern(declaration));
                }
                Declaration::Frontend(frontend) => match frontend {
                    ParsedFrontend::Sig(declaration) => {
                        resolved.push(NestedDeclaration::Sig(declaration));
                    }
                    ParsedFrontend::TypeBind(declaration) => {
                        resolved.push(NestedDeclaration::TypeBind(declaration));
                    }
                    ParsedFrontend::Module(declaration) => resolved.push(NestedDeclaration::Module(
                        self.resolve_module_declaration(module, declaration)?,
                    )),
                    ParsedFrontend::ModuleTypeBind(declaration) => {
                        resolved.push(NestedDeclaration::ModuleTypeBind(declaration));
                    }
                    ParsedFrontend::Open(expression) => resolved.push(NestedDeclaration::Open(
                        self.resolve_module_expression(module, expression)?,
                    )),
                    ParsedFrontend::Import(import) => {
                        let target = self.import_target(module, &import)?;
                        resolved.extend(self.resolve_source_as_nested(target)?);
                    }
                    ParsedFrontend::Resource(declaration) => {
                        resolved.push(NestedDeclaration::Resource(declaration));
                    }
                },
            }
        }
        Ok(resolved)
    }

    fn parsed_file(&self, module: ModuleId) -> Result<&ParsedFile> {
        let Some(loaded) = self.graph.module(module) else {
            return Err(CompilerError::Internal(format!(
                "source-module graph lost {module:?} during import resolution"
            )));
        };
        Ok(loaded.syntax())
    }

    fn import_target(&self, module: ModuleId, import: &SourceImport) -> Result<ModuleId> {
        let Some(target) = self.graph.import_target(module, import.site) else {
            return Err(err_module_at!(
                import.span,
                "source-module graph has no target for import site {:?}",
                import.site
            ));
        };
        Ok(target)
    }

    fn require_root_entry(&self, module: ModuleId, entry: &EntryDecl) -> Result<()> {
        if module == self.graph.root() {
            return Ok(());
        }
        Err(err_module_at!(
            entry.name_span,
            "entry `{}` is not declared directly in the root source module",
            entry.name
        ))
    }
}

#[cfg(test)]
#[path = "resolve_imports_tests.rs"]
mod resolve_imports_tests;
