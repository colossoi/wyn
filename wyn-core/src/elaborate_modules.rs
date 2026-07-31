//! AST checkpoint after source modules have been elaborated.

use crate::ast;

pub type ModulesElaboratedFamily = ast::AstFamily<
    ast::SourceTree,
    ast::DefinitionSyntax,
    ast::EntrySyntax,
    crate::interface::Attribute,
    ast::ExternSyntax,
    ast::ModulesElaboratedFrontend<ast::NestedDeclaration>,
>;

/// AST after module and module-type declarations have moved into the module
/// manager. Their absence is represented by the declaration family rather
/// than by a convention over a shared enum.
#[derive(Debug, Clone, Copy)]
pub enum ModulesElaboratedTag {}
pub type ModulesElaborated =
    ast::Program<ModulesElaboratedTag, ModulesElaboratedFamily, crate::module_manager::ModuleManager>;

/// Elaborate modules into `module_manager` and remove module declarations
/// from the ordinary program tree.
pub fn elaborate_modules(
    program: crate::resolve_imports::ImportsResolved,
) -> crate::error::Result<ModulesElaborated> {
    program.try_rebuild(|declarations, mut global_context, node_ids| {
        global_context.elaborate_modules(&declarations, node_ids)?;
        let declarations = declarations
            .into_iter()
            .filter_map(|declaration| {
                Some(match declaration {
                    ast::Declaration::Decl(decl) => ast::Declaration::Decl(decl),
                    ast::Declaration::Entry(entry) => ast::Declaration::Entry(entry),
                    ast::Declaration::Extern(ext) => ast::Declaration::Extern(ext),
                    ast::Declaration::Frontend(frontend) => {
                        let frontend = match frontend {
                            ast::ImportsResolvedFrontend::Sig(sig) => {
                                ast::ModulesElaboratedFrontend::Sig(sig)
                            }
                            ast::ImportsResolvedFrontend::TypeBind(bind) => {
                                ast::ModulesElaboratedFrontend::TypeBind(bind)
                            }
                            ast::ImportsResolvedFrontend::Open(open) => {
                                ast::ModulesElaboratedFrontend::Open(open)
                            }
                            ast::ImportsResolvedFrontend::Resource(resource) => {
                                ast::ModulesElaboratedFrontend::Resource(resource)
                            }
                            ast::ImportsResolvedFrontend::Module(_)
                            | ast::ImportsResolvedFrontend::ModuleTypeBind(_) => return None,
                        };
                        ast::Declaration::Frontend(frontend)
                    }
                })
            })
            .collect();
        Ok((declarations, global_context))
    })
}
