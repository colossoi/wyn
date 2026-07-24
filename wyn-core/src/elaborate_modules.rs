//! AST checkpoint after source modules have been elaborated.

use crate::ast;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModulesElaboratedFamily;

impl ast::Family for ModulesElaboratedFamily {
    type Tree = ast::SourceTree;
    type DefinitionData = ast::DefinitionSyntax;
    type EntryData = ast::EntrySyntax;
    type EntryParameterAttribute = crate::interface::Attribute;
    type ExternData = ast::ExternSyntax;
    type FrontendDeclaration = ast::ModulesElaboratedFrontend<ast::NestedDeclaration>;
}

/// AST after module and module-type declarations have moved into the module
/// manager. Their absence is represented by the declaration family rather
/// than by a convention over a shared enum.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModulesElaborated;

impl ast::Stage for ModulesElaborated {
    type Family = ModulesElaboratedFamily;
    type GlobalContext = crate::module_manager::ModuleManager;
}

/// Elaborate modules into `module_manager` and remove module declarations
/// from the ordinary program tree.
pub fn elaborate_modules(
    program: ast::Program<crate::resolve_imports::ImportsResolved>,
) -> crate::error::Result<ast::Program<ModulesElaborated>> {
    let ast::Program {
        declarations,
        mut node_ids,
        mut global_context,
    } = program;
    global_context.elaborate_modules(&declarations, &mut node_ids)?;

    let declarations = declarations
        .into_iter()
        .filter_map(|declaration| {
            Some(match declaration {
                ast::Declaration::Decl(decl) => ast::Declaration::Decl(decl),
                ast::Declaration::Entry(entry) => ast::Declaration::Entry(entry),
                ast::Declaration::Extern(ext) => ast::Declaration::Extern(ext),
                ast::Declaration::Frontend(frontend) => {
                    let frontend = match frontend {
                        ast::ImportsResolvedFrontend::Sig(sig) => ast::ModulesElaboratedFrontend::Sig(sig),
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

    Ok(ast::Program {
        declarations,
        node_ids,
        global_context,
    })
}
