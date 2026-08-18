//! Extract vertex and fragment stages from unified root entries.
//!
//! Source programs expose one host-visible root. Rasterization and shading calls
//! in that root delimit callbacks whose bodies execute in platform stage
//! contexts. This pass consumes those delimiters before ordinary TLC lowering
//! can mistake their opaque orchestration values for shader values.

use super::data;
use super::run::UnpinnedPolymorphic;
use super::{
    clone_term_with_fresh_ids, curried_function_type, Def, DefMeta, EntryPoint, Lambda, ProgramParts,
    RewriteDecision, Term, TermIdSource, TermKind, TermRewriter, TermVisitor, VarRef, WalkDecision,
};
use crate::ast;
use crate::ast::Span;
use crate::builtins;
use crate::egir;
use crate::err_type_at;
use crate::error;
use crate::interface::{self, Attribute, EntryKind};
use crate::op;
use crate::pipeline_descriptor;
use crate::types;
use crate::types::{Diet, Type, TypeName, TypeScheme};
use crate::BindingRef;
use crate::{LookupMap, LookupSet, SymbolId, SymbolTable};

struct InvocationBuiltins {
    direct_draw: builtins::BuiltinId,
    direct_draw_from: builtins::BuiltinId,
    indexed_draw: builtins::BuiltinId,
    indexed_draw_from: builtins::BuiltinId,
    indirect_draw: builtins::BuiltinId,
    indirect_draws: builtins::BuiltinId,
    indexed_indirect_draw: builtins::BuiltinId,
    indexed_indirect_draws: builtins::BuiltinId,
    vertex_output: builtins::BuiltinId,
    rasterizers: Vec<builtins::BuiltinId>,
    rasterizers_with: Vec<builtins::BuiltinId>,
    shade: builtins::BuiltinId,
    shade_with: builtins::BuiltinId,
    target_load: builtins::BuiltinId,
    target_sample: builtins::BuiltinId,
    texture_load: builtins::BuiltinId,
    texture_sample: builtins::BuiltinId,
}

impl InvocationBuiltins {
    fn get() -> Self {
        let catalog = builtins::catalog();
        let id = |name: &str| {
            catalog
                .lookup_by_surface_name(name)
                .unwrap_or_else(|| panic!("unified invocation builtin {name} is missing"))
                .id
        };
        Self {
            direct_draw: id("direct_draw"),
            direct_draw_from: id("direct_draw_from"),
            indexed_draw: id("indexed_draw"),
            indexed_draw_from: id("indexed_draw_from"),
            indirect_draw: id("indirect_draw"),
            indirect_draws: id("indirect_draws"),
            indexed_indirect_draw: id("indexed_indirect_draw"),
            indexed_indirect_draws: id("indexed_indirect_draws"),
            vertex_output: id("vertex_output"),
            rasterizers: [
                "rasterize_triangles",
                "rasterize_triangle_strip",
                "rasterize_lines",
                "rasterize_line_strip",
                "rasterize_points",
            ]
            .iter()
            .map(|name| id(name))
            .collect(),
            rasterizers_with: [
                "rasterize_triangles_with",
                "rasterize_triangle_strip_with",
                "rasterize_lines_with",
                "rasterize_line_strip_with",
                "rasterize_points_with",
            ]
            .iter()
            .map(|name| id(name))
            .collect(),
            shade: id("shade"),
            shade_with: id("shade_with"),
            target_load: id("target_load"),
            target_sample: id("target_sample"),
            texture_load: id("texture_load"),
            texture_sample: id("texture_sample"),
        }
    }
}
fn contains_graphics_invocation(term: &Term, builtins: &InvocationBuiltins) -> bool {
    let mut found = false;
    let mut visitor = |term: &Term| {
        let is_invocation = matches!(
            &term.kind,
            TermKind::Var(VarRef::Builtin { id, .. })
                if *id == builtins.shade
                    || *id == builtins.shade_with
                    || builtins.rasterizers.contains(id)
                    || builtins.rasterizers_with.contains(id)
        );
        if is_invocation {
            found = true;
            WalkDecision::Prune
        } else {
            WalkDecision::Recurse
        }
    };
    visitor.walk(term);
    found
}

#[derive(Clone)]
struct StageHelper {
    params: Vec<(SymbolId, Type)>,
    body: Term,
}

fn stage_helper(definition: &Def<UnpinnedPolymorphic>) -> Option<(SymbolId, StageHelper)> {
    if !matches!(definition.meta, DefMeta::Function) || definition.arity == 0 {
        return None;
    }
    let (body, params) = super::extract_lambda_params_ref(&definition.body);
    Some((
        definition.name,
        StageHelper {
            params,
            body: body.clone(),
        },
    ))
}

fn inline_stage_helpers(
    term: Term,
    helpers: &LookupMap<SymbolId, StageHelper>,
    term_ids: &mut TermIdSource,
) -> Term {
    StageHelperInliner {
        helpers,
        term_ids,
        changed: false,
    }
    .rewrite_owned(term)
}

struct StageHelperInliner<'a> {
    helpers: &'a LookupMap<SymbolId, StageHelper>,
    term_ids: &'a mut TermIdSource,
    changed: bool,
}

impl TermRewriter<data::Empty, data::Empty> for StageHelperInliner<'_> {
    fn next_term_id(&mut self) -> super::TermId {
        self.term_ids.next_id()
    }

    fn rewrite_owned_node(&mut self, term: Term) -> (Term, RewriteDecision) {
        let candidate = match &term.kind {
            TermKind::App { func, args } => match &func.kind {
                TermKind::Var(VarRef::Symbol(symbol)) => {
                    self.helpers.get(symbol).filter(|candidate| candidate.params.len() == args.len())
                }
                _ => None,
            },
            _ => None,
        };
        let Some(candidate) = candidate else {
            return (term, RewriteDecision::Unchanged);
        };
        let params = candidate.params.clone();
        let body = clone_term_with_fresh_ids(&candidate.body, self.term_ids);
        let Term { id, span, kind, .. } = term;
        let TermKind::App { args, .. } = kind else {
            unreachable!()
        };
        let mut replacement = super::inline::build_inline_lets(&params, args, body, span, self.term_ids);
        replacement.id = id;
        self.changed = true;
        (replacement, RewriteDecision::Changed)
    }
}

struct ComputeOperation<'a> {
    symbol: SymbolId,
    ty: Type,
    rhs: &'a Term,
    entry_name: String,
    outputs: Vec<ComputedLeaf>,
}

struct GraphicsOperation<'a> {
    raster_term: &'a Term,
    shade_term: &'a Term,
    target_symbol: SymbolId,
    target_name: String,
    target_color_ty: Type,
}

enum RootOperation<'a> {
    Compute(ComputeOperation<'a>),
    Graphics(GraphicsOperation<'a>),
}

#[derive(Clone)]
struct ComputedLeaf {
    path: Vec<usize>,
    ty: Type,
    output_name: String,
    binding: u32,
}

#[derive(Clone)]
struct ComputedValue {
    symbol: SymbolId,
    leaves: Vec<ComputedLeaf>,
}

#[derive(Clone)]
struct TargetValue {
    ty: Type,
    name: String,
    binding: u32,
}

struct RootShape<'a> {
    operations: Vec<RootOperation<'a>>,
    computed: Vec<ComputedValue>,
    targets: LookupMap<SymbolId, TargetValue>,
}

/// Replace every unified graphics root with the ordered internal stages selected
/// by its invocation operations.
pub(super) fn extract(
    parts: &mut ProgramParts<UnpinnedPolymorphic>,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> error::Result<()> {
    let builtins = InvocationBuiltins::get();
    let source_defs = std::mem::take(&mut parts.defs);
    let helpers = source_defs.iter().filter_map(stage_helper).collect::<LookupMap<_, _>>();
    let mut extracted = Vec::with_capacity(source_defs.len());

    for mut definition in source_defs {
        let is_root = matches!(
            &definition.meta,
            DefMeta::EntryPoint(entry) if entry.declaration.entry_kind == EntryKind::Root
        );
        if !is_root {
            extracted.push(definition);
            continue;
        }

        if let Some(stages) = extract_root(&definition, &builtins, &helpers, symbols, term_ids) {
            extracted.extend(stages);
            continue;
        }

        let contains_invocation = match &definition.body.kind {
            TermKind::Lambda(lambda) => {
                let normalized = inline_stage_helpers((*lambda.body).clone(), &helpers, term_ids);
                contains_graphics_invocation(&normalized, &builtins)
            }
            _ => false,
        };
        if contains_invocation {
            let name = root_entry_name(&definition).unwrap_or_else(|| "<entry>".to_string());
            return Err(err_type_at!(
                root_entry_span(&definition),
                "entry `{}` contains a graphics invocation that cannot be planned as an ordered rasterization and shading operation",
                name
            ));
        }

        let DefMeta::EntryPoint(entry) = &mut definition.meta else {
            unreachable!("root classification changed during stage extraction")
        };
        entry.declaration.entry_kind = EntryKind::Compute;
        extracted.push(definition);
    }

    parts.defs = extracted;
    Ok(())
}

fn extract_root(
    definition: &Def<UnpinnedPolymorphic>,
    builtins: &InvocationBuiltins,
    helpers: &LookupMap<SymbolId, StageHelper>,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<Vec<Def<UnpinnedPolymorphic>>> {
    let DefMeta::EntryPoint(root_entry) = &definition.meta else {
        return None;
    };
    if root_entry.declaration.entry_kind != EntryKind::Root {
        return None;
    }

    let TermKind::Lambda(source_root_lambda) = &definition.body.kind else {
        return None;
    };
    // Orchestration helpers have ordinary call semantics. Inline them before
    // recognizing the operation chain so a helper can forward a raster or
    // contain an invocation without becoming a separate host entry.
    let mut root_lambda = source_root_lambda.clone();
    let body = inline_stage_helpers(*root_lambda.body, helpers, term_ids);
    root_lambda.body = Box::new(normalize_root_bindings(body, builtins, term_ids));
    let root_name = root_entry_name(definition)?;
    let shape = root_shape(&root_lambda, root_entry, &root_name, builtins)?;
    let graphics_count =
        shape.operations.iter().filter(|operation| matches!(operation, RootOperation::Graphics(_))).count();
    if graphics_count == 0 {
        return None;
    }

    let mut stages = Vec::with_capacity(
        shape
            .operations
            .iter()
            .map(|operation| match operation {
                RootOperation::Compute(_) => 1,
                RootOperation::Graphics(_) => 2,
            })
            .sum(),
    );
    let mut graphics_index = 0usize;

    for operation in &shape.operations {
        match operation {
            RootOperation::Compute(operation) => {
                stages.push(build_compute_stage(
                    definition,
                    &root_lambda,
                    root_entry,
                    operation,
                    &shape.computed,
                    &shape.targets,
                    builtins,
                    symbols,
                    term_ids,
                )?);
            }
            RootOperation::Graphics(operation) => {
                let (rasterizer, raster_args, has_raster_state) =
                    rasterizer_app(operation.raster_term, builtins)?;
                let (shade_builtin, shade_args) = shade_app(operation.shade_term, builtins)?;
                let draw_index = usize::from(has_raster_state);
                let callback_index = draw_index + 1;
                if raster_args.len() <= callback_index || shade_args.len() < 3 {
                    return None;
                }

                let mut vertex_lambda =
                    callback_lambda(raster_args.get(callback_index)?, "vertex", symbols, term_ids)?;
                let mut fragment_lambda =
                    callback_lambda(shade_args.last()?, "fragment", symbols, term_ids)?;
                if vertex_lambda.params.len() != 1 || fragment_lambda.params.len() != 1 {
                    return None;
                }
                vertex_lambda.body = Box::new(inline_stage_helpers(*vertex_lambda.body, helpers, term_ids));
                fragment_lambda.body =
                    Box::new(inline_stage_helpers(*fragment_lambda.body, helpers, term_ids));

                let raster_state = if has_raster_state {
                    parse_raster_state(raster_args.first()?)?
                } else {
                    Default::default()
                };
                let fragment_state = if shade_builtin == builtins.shade_with {
                    parse_fragment_state(shade_args.first()?)?
                } else {
                    Default::default()
                };
                let graphics_invocation = graphics_invocation(
                    rasterizer,
                    raster_args.get(draw_index)?,
                    raster_state,
                    fragment_state,
                    &root_lambda,
                    root_entry,
                    &shape.computed,
                    builtins,
                )?;
                let graphics_group = interface::GraphicsStageGroup {
                    root: definition.name,
                    operation: graphics_index as u32,
                    invocation: graphics_invocation,
                };
                let owner = if graphics_count == 1 {
                    root_name.clone()
                } else {
                    format!("{root_name}__graphics_{graphics_index}")
                };

                let (vertex_external, vertex_external_decls, vertex_substitutions, vertex_target_reads) =
                    stage_captures(
                        &vertex_lambda.body,
                        &root_lambda,
                        root_entry,
                        &shape.computed,
                        &shape.targets,
                        Some(operation.target_symbol),
                        symbols,
                    );
                let (
                    fragment_external,
                    fragment_external_decls,
                    fragment_substitutions,
                    fragment_target_reads,
                ) = stage_captures(
                    &fragment_lambda.body,
                    &root_lambda,
                    root_entry,
                    &shape.computed,
                    &shape.targets,
                    Some(operation.target_symbol),
                    symbols,
                );

                stages.push(build_vertex_stage(
                    definition,
                    &owner,
                    &vertex_lambda,
                    vertex_external,
                    vertex_external_decls,
                    vertex_substitutions,
                    vertex_target_reads,
                    builtins,
                    graphics_group.clone(),
                    symbols,
                    term_ids,
                )?);
                stages.push(build_fragment_stage(
                    definition,
                    &owner,
                    &fragment_lambda,
                    fragment_external,
                    fragment_external_decls,
                    fragment_substitutions,
                    fragment_target_reads,
                    operation.target_name.clone(),
                    operation.target_color_ty.clone(),
                    builtins,
                    graphics_group,
                    symbols,
                    term_ids,
                )?);
                graphics_index += 1;
            }
        }
    }

    Some(stages)
}

/// Remove administrative root-level bindings by ordinary substitution so the
/// planner sees the operation chain independently of local naming choices.
fn normalize_root_bindings(term: Term, builtins: &InvocationBuiltins, term_ids: &mut TermIdSource) -> Term {
    let Term { id, ty, span, kind } = term;
    let TermKind::Let {
        name,
        name_ty,
        rhs,
        body,
    } = kind
    else {
        return Term { id, ty, span, kind };
    };

    let is_operation = rasterizer_app(&rhs, builtins).is_some()
        || matches!(name_ty, Type::Constructed(TypeName::Raster, _))
        || shade_app(&rhs, builtins).is_some()
        || computed_leaf_types(&name_ty).is_some();

    if is_operation {
        let body = normalize_root_bindings(*body, builtins, term_ids);
        return Term {
            id,
            ty,
            span,
            kind: TermKind::Let {
                name,
                name_ty,
                rhs,
                body: Box::new(body),
            },
        };
    }

    let replacement = *rhs;
    let body = super::subst::substitute_with(
        *body,
        name,
        &mut |occurrence, ids| {
            let mut value = clone_term_with_fresh_ids(&replacement, ids);
            value.span = occurrence.span;
            value
        },
        term_ids,
    );
    normalize_root_bindings(body, builtins, term_ids)
}

fn root_shape<'a>(
    root_lambda: &'a Lambda,
    root_entry: &EntryPoint<()>,
    root_name: &str,
    builtins: &InvocationBuiltins,
) -> Option<RootShape<'a>> {
    let mut targets = LookupMap::new();
    for ((symbol, ty), declaration) in root_lambda.params.iter().zip(&root_entry.declaration.params) {
        if is_render_target_type(ty) {
            targets.insert(
                *symbol,
                TargetValue {
                    ty: ty.clone(),
                    name: declaration.name.clone(),
                    binding: 0,
                },
            );
        }
    }

    let mut rasters = LookupMap::<SymbolId, &'a Term>::new();
    let mut operations = Vec::new();
    let mut current = root_lambda.body.as_ref();

    loop {
        let TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } = &current.kind
        else {
            break;
        };

        if rasterizer_app(rhs, builtins).is_some() {
            rasters.insert(*name, rhs.as_ref());
        } else if matches!(name_ty, Type::Constructed(TypeName::Raster, _)) {
            let TermKind::Var(VarRef::Symbol(source)) = rhs.kind else {
                return None;
            };
            let raster = rasters.remove(&source)?;
            rasters.insert(*name, raster);
        } else if shade_app(rhs, builtins).is_some() {
            let operation = graphics_operation(rhs, &mut rasters, &targets, builtins)?;
            targets.insert(
                *name,
                TargetValue {
                    ty: name_ty.clone(),
                    name: operation.target_name.clone(),
                    binding: 0,
                },
            );
            operations.push(RootOperation::Graphics(operation));
        } else if computed_leaf_types(name_ty).is_some() {
            operations.push(RootOperation::Compute(ComputeOperation {
                symbol: *name,
                ty: name_ty.clone(),
                rhs: rhs.as_ref(),
                entry_name: String::new(),
                outputs: vec![],
            }));
        } else {
            return None;
        }
        current = body;
    }

    if shade_app(current, builtins).is_some() {
        operations.push(RootOperation::Graphics(graphics_operation(
            current,
            &mut rasters,
            &targets,
            builtins,
        )?));
    }
    if !rasters.is_empty() {
        return None;
    }

    let compute_count =
        operations.iter().filter(|operation| matches!(operation, RootOperation::Compute(_))).count();
    let mut compute_index = 0usize;
    let mut next_binding = root_lambda.params.len() as u32;
    let mut computed = Vec::with_capacity(compute_count);
    for operation in &mut operations {
        let RootOperation::Compute(operation) = operation else {
            continue;
        };
        operation.entry_name = if compute_count == 1 {
            root_name.to_string()
        } else {
            format!("{root_name}__compute_{compute_index}")
        };
        let leaf_types = computed_leaf_types(&operation.ty)?;
        let multiple = leaf_types.len() > 1;
        operation.outputs = leaf_types
            .into_iter()
            .enumerate()
            .map(|(index, (path, _label, ty))| {
                let output_name = if multiple {
                    format!("{}_output_{index}", operation.entry_name)
                } else {
                    format!("{}_output", operation.entry_name)
                };
                let leaf = ComputedLeaf {
                    path,
                    ty,
                    output_name,
                    binding: next_binding,
                };
                next_binding += 1;
                leaf
            })
            .collect();
        computed.push(ComputedValue {
            symbol: operation.symbol,
            leaves: operation.outputs.clone(),
        });
        compute_index += 1;
    }

    let mut next_target_binding = next_binding;
    let mut target_bindings = LookupMap::new();
    for ((_, ty), declaration) in root_lambda.params.iter().zip(&root_entry.declaration.params) {
        let Some(color_ty) = render_target_color_type(ty) else {
            continue;
        };
        target_bindings.insert(declaration.name.clone(), next_target_binding);
        next_target_binding += varying_leaf_types(color_ty).len() as u32;
    }
    for target in targets.values_mut() {
        target.binding = *target_bindings.get(&target.name)?;
    }

    Some(RootShape {
        operations,
        computed,
        targets,
    })
}

fn graphics_operation<'a>(
    shade_term: &'a Term,
    rasters: &mut LookupMap<SymbolId, &'a Term>,
    targets: &LookupMap<SymbolId, TargetValue>,
    builtins: &InvocationBuiltins,
) -> Option<GraphicsOperation<'a>> {
    let (shade_builtin, shade_args) = shade_app(shade_term, builtins)?;
    let target_index = usize::from(shade_builtin == builtins.shade_with);
    let target_symbol = term_symbol(shade_args.get(target_index)?)?;
    let raster_symbol = term_symbol(shade_args.get(target_index + 1)?)?;
    let target = targets.get(&target_symbol)?;
    let target_name = target.name.clone();
    let target_color_ty = render_target_color_type(&target.ty)?.clone();
    let raster_term = rasters.remove(&raster_symbol)?;
    Some(GraphicsOperation {
        raster_term,
        shade_term,
        target_symbol,
        target_name,
        target_color_ty,
    })
}

fn rasterizer_app<'a>(
    term: &'a Term,
    builtins: &InvocationBuiltins,
) -> Option<(builtins::BuiltinId, &'a [Term], bool)> {
    if let Some((id, args)) = builtin_app(term, &builtins.rasterizers) {
        return Some((id, args, false));
    }
    builtin_app(term, &builtins.rasterizers_with).map(|(id, args)| (id, args, true))
}

fn shade_app<'a>(
    term: &'a Term,
    builtins: &InvocationBuiltins,
) -> Option<(builtins::BuiltinId, &'a [Term])> {
    builtin_app(term, &[builtins.shade, builtins.shade_with])
}

fn computed_leaf_types(ty: &Type) -> Option<Vec<(Vec<usize>, String, Type)>> {
    fn collect(
        ty: &Type,
        path: &mut Vec<usize>,
        labels: &mut Vec<String>,
        leaves: &mut Vec<(Vec<usize>, String, Type)>,
    ) -> bool {
        match ty {
            Type::Constructed(TypeName::Existential(_), args) if args.len() == 1 => {
                collect(&args[0], path, labels, leaves)
            }
            Type::Constructed(TypeName::Array, _) => {
                leaves.push((path.clone(), labels.join("_"), ty.clone()));
                true
            }
            Type::Constructed(TypeName::Record(fields), components) => {
                if components.is_empty() {
                    return false;
                }
                for (index, (field, component)) in fields.iter().zip(components).enumerate() {
                    path.push(index);
                    labels.push(field.clone());
                    if !collect(component, path, labels, leaves) {
                        return false;
                    }
                    labels.pop();
                    path.pop();
                }
                true
            }
            Type::Constructed(TypeName::Tuple(_), components) => {
                if components.is_empty() {
                    return false;
                }
                for (index, component) in components.iter().enumerate() {
                    path.push(index);
                    labels.push(index.to_string());
                    if !collect(component, path, labels, leaves) {
                        return false;
                    }
                    labels.pop();
                    path.pop();
                }
                true
            }
            _ => false,
        }
    }

    let mut leaves = Vec::new();
    collect(ty, &mut Vec::new(), &mut Vec::new(), &mut leaves).then_some(leaves)
}

fn term_symbol(term: &Term) -> Option<SymbolId> {
    match term.kind {
        TermKind::Var(VarRef::Symbol(symbol)) => Some(symbol),
        _ => None,
    }
}

fn projected_symbol_path(term: &Term) -> Option<(SymbolId, Vec<usize>)> {
    match &term.kind {
        TermKind::Var(VarRef::Symbol(symbol)) => Some((*symbol, vec![])),
        TermKind::TupleProj { tuple, idx } => {
            let (symbol, mut path) = projected_symbol_path(tuple)?;
            path.push(*idx);
            Some((symbol, path))
        }
        _ => None,
    }
}

fn is_render_target_type(ty: &Type) -> bool {
    matches!(ty, Type::Constructed(TypeName::RenderTarget, _))
}

fn builtin_app<'a>(
    term: &'a Term,
    candidates: &[builtins::BuiltinId],
) -> Option<(builtins::BuiltinId, &'a [Term])> {
    let TermKind::App { func, args } = &term.kind else {
        return None;
    };
    let TermKind::Var(VarRef::Builtin { id, .. }) = func.kind else {
        return None;
    };
    candidates.contains(&id).then_some((id, args))
}

fn callback_lambda(
    callback: &Term,
    stage: &str,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<Lambda> {
    if let TermKind::Lambda(lambda) = &callback.kind {
        return Some(lambda.clone());
    }
    let TermKind::Var(VarRef::Symbol(_)) = callback.kind else {
        return None;
    };
    let Type::Constructed(TypeName::Arrow, args) = &callback.ty else {
        return None;
    };
    let [param_ty, result_ty] = args.as_slice() else {
        return None;
    };
    let param = symbols.alloc(format!("_w_{stage}_callback_argument"));
    let argument = Term::fresh(
        term_ids,
        param_ty.clone(),
        callback.span,
        TermKind::Var(VarRef::Symbol(param)),
    );
    let function = clone_term_with_fresh_ids(callback, term_ids);
    let body = Term::fresh(
        term_ids,
        result_ty.clone(),
        callback.span,
        TermKind::App {
            func: Box::new(function),
            args: vec![argument],
        },
    );
    Some(Lambda {
        params: vec![(param, param_ty.clone())],
        body: Box::new(body),
        ret_ty: result_ty.clone(),
    })
}

fn graphics_invocation(
    rasterizer: builtins::BuiltinId,
    draw: &Term,
    raster_state: pipeline_descriptor::RasterState,
    fragment_state: pipeline_descriptor::FragmentState,
    root_lambda: &Lambda,
    root_entry: &EntryPoint<()>,
    computed: &[ComputedValue],
    builtins: &InvocationBuiltins,
) -> Option<pipeline_descriptor::GraphicsInvocation> {
    use crate::pipeline_descriptor::{DrawCall, DrawCount, GraphicsInvocation, PrimitiveTopology};

    let topology_index = builtins
        .rasterizers
        .iter()
        .position(|id| *id == rasterizer)
        .or_else(|| builtins.rasterizers_with.iter().position(|id| *id == rasterizer))?;
    let topology = match topology_index {
        0 => PrimitiveTopology::TriangleList,
        1 => PrimitiveTopology::TriangleStrip,
        2 => PrimitiveTopology::LineList,
        3 => PrimitiveTopology::LineStrip,
        4 => PrimitiveTopology::PointList,
        _ => return None,
    };
    let (constructor, args) = builtin_app(
        draw,
        &[
            builtins.direct_draw,
            builtins.direct_draw_from,
            builtins.indexed_draw,
            builtins.indexed_draw_from,
            builtins.indirect_draw,
            builtins.indirect_draws,
            builtins.indexed_indirect_draw,
            builtins.indexed_indirect_draws,
        ],
    )?;
    let draw = if constructor == builtins.direct_draw {
        let values = args.iter().map(u32_literal).collect::<Option<Vec<_>>>()?;
        let [vertex_count, instance_count] = values.as_slice() else {
            return None;
        };
        DrawCall::Direct {
            vertex_count: *vertex_count,
            instance_count: *instance_count,
            first_vertex: 0,
            first_instance: 0,
        }
    } else if constructor == builtins.direct_draw_from {
        let values = args.iter().map(u32_literal).collect::<Option<Vec<_>>>()?;
        let [vertex_count, instance_count, first_vertex, first_instance] = values.as_slice() else {
            return None;
        };
        DrawCall::Direct {
            vertex_count: *vertex_count,
            instance_count: *instance_count,
            first_vertex: *first_vertex,
            first_instance: *first_instance,
        }
    } else if constructor == builtins.indexed_draw {
        let [indices, instance_count] = args else {
            return None;
        };
        DrawCall::Indexed {
            indices: draw_buffer_source(indices, root_lambda, root_entry, computed)?,
            index_format: index_format(indices)?,
            index_count: array_draw_count(indices)?,
            instance_count: u32_literal(instance_count)?,
            first_index: 0,
            vertex_offset: 0,
            first_instance: 0,
        }
    } else if constructor == builtins.indexed_draw_from {
        let [indices, index_count, instance_count, first_index, vertex_offset, first_instance] = args
        else {
            return None;
        };
        DrawCall::Indexed {
            indices: draw_buffer_source(indices, root_lambda, root_entry, computed)?,
            index_format: index_format(indices)?,
            index_count: DrawCount::Fixed(u32_literal(index_count)?),
            instance_count: u32_literal(instance_count)?,
            first_index: u32_literal(first_index)?,
            vertex_offset: i32_literal(vertex_offset)?,
            first_instance: u32_literal(first_instance)?,
        }
    } else if constructor == builtins.indirect_draw {
        let [command] = args else {
            return None;
        };
        let (commands, offset) = indirect_command_source(command, 16, root_lambda, root_entry, computed)?;
        DrawCall::Indirect {
            commands,
            offset,
            draw_count: DrawCount::Fixed(1),
        }
    } else if constructor == builtins.indirect_draws {
        let [commands] = args else {
            return None;
        };
        DrawCall::Indirect {
            commands: draw_buffer_source(commands, root_lambda, root_entry, computed)?,
            offset: 0,
            draw_count: array_draw_count(commands)?,
        }
    } else if constructor == builtins.indexed_indirect_draw {
        let [indices, command] = args else {
            return None;
        };
        let (commands, offset) = indirect_command_source(command, 20, root_lambda, root_entry, computed)?;
        DrawCall::IndexedIndirect {
            indices: draw_buffer_source(indices, root_lambda, root_entry, computed)?,
            index_format: index_format(indices)?,
            commands,
            offset,
            draw_count: DrawCount::Fixed(1),
        }
    } else {
        let [indices, commands] = args else {
            return None;
        };
        DrawCall::IndexedIndirect {
            indices: draw_buffer_source(indices, root_lambda, root_entry, computed)?,
            index_format: index_format(indices)?,
            commands: draw_buffer_source(commands, root_lambda, root_entry, computed)?,
            offset: 0,
            draw_count: array_draw_count(commands)?,
        }
    };
    Some(GraphicsInvocation {
        topology,
        draw,
        raster_state,
        fragment_state,
    })
}

fn indirect_command_source(
    command: &Term,
    stride: u64,
    root_lambda: &Lambda,
    root_entry: &EntryPoint<()>,
    computed: &[ComputedValue],
) -> Option<(pipeline_descriptor::DrawBufferRef, u64)> {
    let TermKind::Index { array, index } = &command.kind else {
        return None;
    };
    let command_index = u32_literal(index)? as u64;
    Some((
        draw_buffer_source(array, root_lambda, root_entry, computed)?,
        command_index * stride,
    ))
}

fn draw_buffer_source(
    array: &Term,
    root_lambda: &Lambda,
    root_entry: &EntryPoint<()>,
    computed: &[ComputedValue],
) -> Option<pipeline_descriptor::DrawBufferRef> {
    let (symbol, path) = projected_symbol_path(array)?;
    if path.is_empty() {
        if let Some((index, _)) =
            root_lambda.params.iter().enumerate().find(|(_, (candidate, _))| *candidate == symbol)
        {
            return Some(pipeline_descriptor::DrawBufferRef {
                set: egir::from_tlc::AUTO_STORAGE_SET,
                binding: index as u32,
                name: root_entry.declaration.params.get(index)?.name.clone(),
                resource: None,
            });
        }
    }
    let value = computed.iter().find(|value| value.symbol == symbol)?;
    let leaf = value.leaves.iter().find(|leaf| leaf.path == path)?;
    Some(pipeline_descriptor::DrawBufferRef {
        set: egir::from_tlc::AUTO_STORAGE_SET,
        binding: leaf.binding,
        name: leaf.output_name.clone(),
        resource: Some(leaf.output_name.clone()),
    })
}
fn array_type_parts(mut ty: &Type) -> Option<(&Type, &Type)> {
    while let Type::Constructed(TypeName::Existential(_), args) = ty {
        ty = args.first()?;
    }
    let Type::Constructed(TypeName::Array, args) = ty else {
        return None;
    };
    Some((args.first()?, args.get(2)?))
}

fn array_draw_count(array: &Term) -> Option<pipeline_descriptor::DrawCount> {
    match array_type_parts(&array.ty).map(|(_, size)| size) {
        Some(Type::Constructed(TypeName::Size(count), _)) => {
            Some(pipeline_descriptor::DrawCount::Fixed(u32::try_from(*count).ok()?))
        }
        Some(_) => Some(pipeline_descriptor::DrawCount::BufferLength),
        None => None,
    }
}

fn index_format(array: &Term) -> Option<pipeline_descriptor::IndexFormat> {
    use crate::pipeline_descriptor::IndexFormat;
    match array_type_parts(&array.ty)?.0 {
        Type::Constructed(TypeName::UInt(16), _) => Some(IndexFormat::Uint16),
        Type::Constructed(TypeName::UInt(32), _) => Some(IndexFormat::Uint32),
        _ => None,
    }
}
fn is_u16_array(ty: &Type) -> bool {
    matches!(
        array_type_parts(ty).map(|(element, _)| element),
        Some(Type::Constructed(TypeName::UInt(16), _))
    )
}
fn parse_raster_state(term: &Term) -> Option<pipeline_descriptor::RasterState> {
    use crate::pipeline_descriptor::{CullMode, FillMode, FrontFace, RasterState, Scissor, Viewport};

    let viewport_term = record_component(term, "viewport")?;
    let viewport = match sum_tag(viewport_term)? {
        0 => Viewport::Target,
        1 => {
            let custom = sum_payload(viewport_term, 0)?;
            Viewport::Custom {
                origin: f32_vec2(record_component(custom, "origin")?)?,
                extent: f32_vec2(record_component(custom, "extent")?)?,
                depth: f32_vec2(record_component(custom, "depth")?)?,
            }
        }
        _ => return None,
    };
    let scissor_term = record_component(term, "scissor")?;
    let scissor = match sum_tag(scissor_term)? {
        0 => Scissor::Target,
        1 => {
            let custom = sum_payload(scissor_term, 0)?;
            Scissor::Custom {
                origin: i32_vec2(record_component(custom, "origin")?)?,
                extent: u32_vec2(record_component(custom, "extent")?)?,
            }
        }
        _ => return None,
    };
    let front_face = match sum_tag(record_component(term, "front_face")?)? {
        0 => FrontFace::Clockwise,
        1 => FrontFace::CounterClockwise,
        _ => return None,
    };
    let cull = match sum_tag(record_component(term, "cull")?)? {
        0 => CullMode::None,
        1 => CullMode::Front,
        2 => CullMode::Back,
        _ => return None,
    };
    let fill = match sum_tag(record_component(term, "fill")?)? {
        0 => FillMode::Fill,
        1 => FillMode::Line,
        2 => FillMode::Point,
        _ => return None,
    };
    Some(RasterState {
        viewport,
        scissor,
        front_face,
        cull,
        fill,
    })
}

fn parse_fragment_state(term: &Term) -> Option<pipeline_descriptor::FragmentState> {
    use crate::pipeline_descriptor::{BlendMode, DepthTest, FragmentState};

    let depth_test = match sum_tag(record_component(term, "depth_test")?)? {
        0 => DepthTest::Disabled,
        1 => DepthTest::Never,
        2 => DepthTest::Less,
        3 => DepthTest::LessEqual,
        4 => DepthTest::Equal,
        5 => DepthTest::GreaterEqual,
        6 => DepthTest::Greater,
        7 => DepthTest::Always,
        _ => return None,
    };
    let blend = match sum_tag(record_component(term, "blend")?)? {
        0 => BlendMode::Replace,
        1 => BlendMode::SourceOver,
        2 => BlendMode::Add,
        _ => return None,
    };
    Some(FragmentState {
        depth_test,
        depth_write: bool_literal(record_component(term, "depth_write")?)?,
        blend,
        color_write: bool_literal(record_component(term, "color_write")?)?,
    })
}

fn record_component<'a>(term: &'a Term, field: &str) -> Option<&'a Term> {
    let TermKind::Tuple(values) = &term.kind else {
        return None;
    };
    let Type::Constructed(TypeName::Record(fields), _) = &term.ty else {
        return None;
    };
    let index = fields.iter().position(|candidate| candidate == field)?;
    values.get(index)
}

fn sum_tag(term: &Term) -> Option<u32> {
    let TermKind::Tuple(values) = &term.kind else {
        return None;
    };
    u32_literal(values.first()?)
}

fn sum_payload(term: &Term, index: usize) -> Option<&Term> {
    let TermKind::Tuple(values) = &term.kind else {
        return None;
    };
    values.get(index + 1)
}

fn f32_vec2(term: &Term) -> Option<[f32; 2]> {
    let TermKind::VecLit(values) = &term.kind else {
        return None;
    };
    let [x, y] = values.as_slice() else {
        return None;
    };
    Some([f32_literal(x)?, f32_literal(y)?])
}

fn i32_vec2(term: &Term) -> Option<[i32; 2]> {
    let TermKind::VecLit(values) = &term.kind else {
        return None;
    };
    let [x, y] = values.as_slice() else {
        return None;
    };
    Some([i32_literal(x)?, i32_literal(y)?])
}

fn u32_vec2(term: &Term) -> Option<[u32; 2]> {
    let TermKind::VecLit(values) = &term.kind else {
        return None;
    };
    let [x, y] = values.as_slice() else {
        return None;
    };
    Some([u32_literal(x)?, u32_literal(y)?])
}

fn f32_literal(term: &Term) -> Option<f32> {
    match term.kind {
        TermKind::FloatLit(value) => Some(value),
        _ => None,
    }
}

fn bool_literal(term: &Term) -> Option<bool> {
    match term.kind {
        TermKind::BoolLit(value) => Some(value),
        _ => None,
    }
}

fn u32_literal(term: &Term) -> Option<u32> {
    let TermKind::IntLit(text) = &term.kind else {
        return None;
    };
    text.strip_suffix("u32").unwrap_or(text).replace('_', "").parse().ok()
}
fn i32_literal(term: &Term) -> Option<i32> {
    let TermKind::IntLit(text) = &term.kind else {
        return None;
    };
    text.strip_suffix("i32").unwrap_or(text).replace('_', "").parse().ok()
}
#[derive(Clone)]
struct TargetRead {
    color_ty: Type,
    leaves: Vec<(SymbolId, Type)>,
}

type TargetReads = LookupMap<SymbolId, TargetRead>;
type ExternalSubstitutions = LookupMap<(SymbolId, Vec<usize>), (SymbolId, Type)>;

type StageCaptures = (
    Vec<(SymbolId, Type)>,
    Vec<interface::EntryParamDecl>,
    ExternalSubstitutions,
    TargetReads,
);

fn stage_captures(
    body: &Term,
    root_lambda: &Lambda,
    root_entry: &EntryPoint<()>,
    computed: &[ComputedValue],
    targets: &LookupMap<SymbolId, TargetValue>,
    output_target: Option<SymbolId>,
    symbols: &mut SymbolTable,
) -> StageCaptures {
    let used = referenced_symbols(body);
    let mut params = Vec::new();
    let mut declarations = Vec::new();
    let mut substitutions = LookupMap::new();
    let mut target_reads = LookupMap::new();
    append_root_captures(
        &used,
        root_lambda,
        root_entry,
        symbols,
        &mut params,
        &mut declarations,
        &mut substitutions,
    );
    append_computed_captures(
        &used,
        computed,
        symbols,
        &mut params,
        &mut declarations,
        &mut substitutions,
    );
    append_target_captures(
        &used,
        targets,
        output_target,
        symbols,
        &mut params,
        &mut declarations,
        &mut target_reads,
    );
    (params, declarations, substitutions, target_reads)
}

fn referenced_symbols(term: &Term) -> LookupSet<SymbolId> {
    let mut referenced = LookupSet::new();
    let mut visitor = |term: &Term| {
        if let TermKind::Var(VarRef::Symbol(symbol)) = term.kind {
            referenced.insert(symbol);
        }
        WalkDecision::Recurse
    };
    visitor.walk(term);
    referenced
}

fn append_root_captures(
    used: &LookupSet<SymbolId>,
    root_lambda: &Lambda,
    root_entry: &EntryPoint<()>,
    symbols: &mut SymbolTable,
    params: &mut Vec<(SymbolId, Type)>,
    declarations: &mut Vec<interface::EntryParamDecl>,
    substitutions: &mut ExternalSubstitutions,
) {
    // Preserve the root descriptor interface across generated stages: later
    // scalar-prepass splitting relies on these occupied slots when allocating
    // compiler handoff buffers. A draw-only u16 index array is the exception;
    // portable shaders cannot expose u16 storage, and DrawCall reserves its
    // non-shader buffer reference separately.
    for (binding, ((old_symbol, ty), declaration)) in
        root_lambda.params.iter().zip(&root_entry.declaration.params).enumerate()
    {
        if is_render_target_type(ty) || (!used.contains(old_symbol) && is_u16_array(ty)) {
            continue;
        }
        let binding = binding as u32;
        let new_symbol = symbols.alloc(declaration.name.clone());
        let external_ty = external_parameter_type(ty, binding);
        substitutions.insert((*old_symbol, vec![]), (new_symbol, external_ty.clone()));
        params.push((new_symbol, external_ty.clone()));
        let mut declaration = declaration.clone();
        declaration.ty = external_ty.clone();
        if declaration.attributes.is_empty() {
            declaration.attributes.extend(external_binding_attribute(&external_ty, binding));
        }
        declarations.push(declaration);
    }
}

fn append_computed_captures(
    used: &LookupSet<SymbolId>,
    computed: &[ComputedValue],
    symbols: &mut SymbolTable,
    params: &mut Vec<(SymbolId, Type)>,
    declarations: &mut Vec<interface::EntryParamDecl>,
    substitutions: &mut ExternalSubstitutions,
) {
    for value in computed {
        if !used.contains(&value.symbol) {
            continue;
        }
        for leaf in &value.leaves {
            append_external_param(
                value.symbol,
                &leaf.path,
                &leaf.ty,
                &leaf.output_name,
                leaf.binding,
                symbols,
                params,
                declarations,
                substitutions,
            );
        }
    }
}

fn append_target_captures(
    used: &LookupSet<SymbolId>,
    targets: &LookupMap<SymbolId, TargetValue>,
    output_target: Option<SymbolId>,
    symbols: &mut SymbolTable,
    params: &mut Vec<(SymbolId, Type)>,
    declarations: &mut Vec<interface::EntryParamDecl>,
    target_reads: &mut TargetReads,
) {
    let mut captures = targets
        .iter()
        .filter(|(symbol, _)| used.contains(symbol) && Some(**symbol) != output_target)
        .collect::<Vec<_>>();
    captures.sort_by_key(|(symbol, _)| symbol.0);
    let mut shared = LookupMap::<String, TargetRead>::new();

    for (old_symbol, target) in captures {
        if let Some(read) = shared.get(&target.name).cloned() {
            target_reads.insert(*old_symbol, read);
            continue;
        }

        let color_ty = render_target_color_type(&target.ty)
            .expect("target capture must retain its render_target color type")
            .clone();
        let mut leaves = Vec::new();
        for (location, (name, leaf_ty)) in attachment_specs(&target.name, &color_ty).into_iter().enumerate()
        {
            let binding = target.binding + location as u32;
            let texture_ty = target_read_type();
            let symbol = symbols.alloc(name.clone());
            params.push((symbol, texture_ty.clone()));
            declarations.push(interface::EntryParamDecl {
                name: name.clone(),
                span: Span::new(0, 0, 0, 0),
                ty: texture_ty,
                attributes: vec![Attribute::Texture {
                    set: egir::from_tlc::AUTO_STORAGE_SET,
                    binding,
                    backing: None,
                    resource: Some(name),
                }],
            });
            leaves.push((symbol, leaf_ty));
        }
        let read = TargetRead { color_ty, leaves };
        shared.insert(target.name.clone(), read.clone());
        target_reads.insert(*old_symbol, read);
    }
}

fn render_target_color_type(target: &Type) -> Option<&Type> {
    match target {
        Type::Constructed(TypeName::RenderTarget, args) if args.len() == 1 => args.first(),
        _ => None,
    }
}

fn target_read_type() -> Type {
    Type::Constructed(TypeName::Texture2D, vec![])
}

fn attachment_specs(target_name: &str, color_ty: &Type) -> Vec<(String, Type)> {
    let leaves = varying_leaf_types(color_ty);
    if leaves.len() == 1 {
        return leaves.into_iter().map(|(_, ty)| (target_name.to_string(), ty)).collect();
    }
    leaves.into_iter().map(|(leaf, ty)| (format!("{target_name}_{leaf}"), ty)).collect()
}

fn append_external_param(
    old_symbol: SymbolId,
    path: &[usize],
    ty: &Type,
    name: &str,
    binding: u32,
    symbols: &mut SymbolTable,
    params: &mut Vec<(SymbolId, Type)>,
    declarations: &mut Vec<interface::EntryParamDecl>,
    substitutions: &mut ExternalSubstitutions,
) {
    let external_ty = external_parameter_type(ty, binding);
    let new_symbol = symbols.alloc(name.to_string());
    substitutions.insert((old_symbol, path.to_vec()), (new_symbol, external_ty.clone()));
    params.push((new_symbol, external_ty.clone()));
    declarations.push(interface::EntryParamDecl {
        name: name.to_string(),
        span: Span::new(0, 0, 0, 0),
        ty: external_ty.clone(),
        attributes: external_binding_attribute(&external_ty, binding).into_iter().collect(),
    });
}

fn external_parameter_type(ty: &Type, binding: u32) -> Type {
    let Type::Constructed(TypeName::Array, args) = ty else {
        return ty.clone();
    };
    let mut args = args.clone();
    if args.len() >= 4 {
        args[1] = Type::Constructed(TypeName::ArrayVariantView, vec![]);
        let slot = args.len() - 1;
        args[slot] = types::buffer_tag(BindingRef {
            set: egir::from_tlc::AUTO_STORAGE_SET,
            binding,
        });
    }
    Type::Constructed(TypeName::Array, args)
}

fn external_binding_attribute(ty: &Type, binding: u32) -> Option<interface::ResolvedAttribute> {
    match ty {
        Type::Constructed(TypeName::Array, _) => Some(Attribute::Storage {
            set: egir::from_tlc::AUTO_STORAGE_SET,
            binding,
            layout: interface::StorageLayout::Std430,
            access: interface::StorageAccess::ReadOnly,
        }),
        Type::Constructed(TypeName::Texture2D, _) => Some(Attribute::Texture {
            set: egir::from_tlc::AUTO_STORAGE_SET,
            binding,
            backing: None,
            resource: None,
        }),
        Type::Constructed(TypeName::Sampler, _) => Some(Attribute::Sampler {
            set: egir::from_tlc::AUTO_STORAGE_SET,
            binding,
        }),
        _ => Some(Attribute::Uniform {
            set: egir::from_tlc::AUTO_STORAGE_SET,
            binding,
        }),
    }
}

fn build_compute_stage(
    root: &Def<UnpinnedPolymorphic>,
    root_lambda: &Lambda,
    root_entry: &EntryPoint<()>,
    operation: &ComputeOperation<'_>,
    computed: &[ComputedValue],
    targets: &LookupMap<SymbolId, TargetValue>,
    builtins: &InvocationBuiltins,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<Def<UnpinnedPolymorphic>> {
    let used = referenced_symbols(operation.rhs);
    let mut params = Vec::new();
    let mut declarations = Vec::new();
    let mut substitutions = LookupMap::new();
    let mut target_reads = LookupMap::new();
    append_root_captures(
        &used,
        root_lambda,
        root_entry,
        symbols,
        &mut params,
        &mut declarations,
        &mut substitutions,
    );
    append_computed_captures(
        &used,
        computed,
        symbols,
        &mut params,
        &mut declarations,
        &mut substitutions,
    );
    append_target_captures(
        &used,
        targets,
        None,
        symbols,
        &mut params,
        &mut declarations,
        &mut target_reads,
    );

    let body = clone_term_with_fresh_ids(operation.rhs, term_ids);
    let body = ExternalValueRewriter {
        term_ids,
        substitutions,
        target_reads,
        target_load: builtins.target_load,
        target_sample: builtins.target_sample,
        texture_load: builtins.texture_load,
        texture_sample: builtins.texture_sample,
    }
    .rewrite_owned(body);
    let outputs = operation
        .outputs
        .iter()
        .map(|leaf| interface::EntryOutputDecl {
            ty: leaf.ty.clone(),
            attribute: Some(Attribute::Storage {
                set: egir::from_tlc::AUTO_STORAGE_SET,
                binding: leaf.binding,
                layout: interface::StorageLayout::Std430,
                access: interface::StorageAccess::WriteOnly,
            }),
        })
        .collect();
    Some(stage_def(
        operation.entry_name.clone(),
        EntryKind::Compute,
        params,
        declarations,
        outputs,
        body,
        None,
        root,
        symbols,
        term_ids,
    ))
}

struct ExternalValueRewriter<'a> {
    term_ids: &'a mut TermIdSource,
    substitutions: ExternalSubstitutions,
    target_reads: TargetReads,
    target_load: builtins::BuiltinId,
    target_sample: builtins::BuiltinId,
    texture_load: builtins::BuiltinId,
    texture_sample: builtins::BuiltinId,
}

impl TermRewriter<data::Empty, data::Empty> for ExternalValueRewriter<'_> {
    fn next_term_id(&mut self) -> super::TermId {
        self.term_ids.next_id()
    }

    fn rewrite_owned_node_before_children(&mut self, mut term: Term) -> (Term, RewriteDecision) {
        if let Some(key) = projected_symbol_path(&term) {
            if let Some((replacement, ty)) = self.substitutions.get(&key) {
                term.ty = ty.clone();
                term.kind = TermKind::Var(VarRef::Symbol(*replacement));
                return (term, RewriteDecision::Changed);
            }
        }
        let TermKind::App { func, args } = &term.kind else {
            return (term, RewriteDecision::Unchanged);
        };
        let TermKind::Var(VarRef::Builtin { id, .. }) = func.kind else {
            return (term, RewriteDecision::Unchanged);
        };
        if args.len() != 3 || (id != self.target_load && id != self.target_sample) {
            return (term, RewriteDecision::Unchanged);
        }
        let Some(target_symbol) = term_symbol(&args[0]) else {
            return (term, RewriteDecision::Unchanged);
        };
        let Some(read) = self.target_reads.get(&target_symbol).cloned() else {
            return (term, RewriteDecision::Unchanged);
        };
        let replacement = if id == self.target_load {
            build_target_load_value(&read, &args[1], term.span, self.texture_load, self.term_ids)
        } else {
            build_target_sample_value(
                &read,
                &args[1],
                &args[2],
                term.span,
                self.texture_sample,
                self.term_ids,
            )
        };
        let Some(mut replacement) = replacement else {
            return (term, RewriteDecision::Unchanged);
        };
        replacement.id = term.id;
        (replacement, RewriteDecision::Changed)
    }
}

fn build_target_load_value(
    read: &TargetRead,
    coord: &Term,
    span: Span,
    texture_load: builtins::BuiltinId,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    let mut leaves = read
        .leaves
        .iter()
        .map(|(texture, leaf_ty)| {
            sampled_target_leaf(*texture, leaf_ty, coord, span, texture_load, term_ids)
        })
        .collect::<Option<Vec<_>>>()?
        .into_iter();
    rebuild_target_value(&read.color_ty, &mut leaves, span, term_ids)
}

fn build_target_sample_value(
    read: &TargetRead,
    sampler: &Term,
    uv: &Term,
    span: Span,
    texture_sample: builtins::BuiltinId,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    let mut leaves = read
        .leaves
        .iter()
        .map(|(texture, leaf_ty)| {
            filtered_target_leaf(*texture, leaf_ty, sampler, uv, span, texture_sample, term_ids)
        })
        .collect::<Option<Vec<_>>>()?
        .into_iter();
    rebuild_target_value(&read.color_ty, &mut leaves, span, term_ids)
}

fn filtered_target_leaf(
    texture: SymbolId,
    leaf_ty: &Type,
    sampler: &Term,
    uv: &Term,
    span: Span,
    texture_sample: builtins::BuiltinId,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    let texture_ty = target_read_type();
    let sampler_ty = Type::Constructed(TypeName::Sampler, vec![]);
    let uv_ty = vec_ty(2, f32_ty());
    let lod_ty = f32_ty();
    let sampled_ty = vec_ty(4, f32_ty());
    let function_params = [texture_ty.clone(), sampler_ty, uv_ty, lod_ty.clone()];
    let function = Term::fresh(
        term_ids,
        curried_function_type(function_params.iter(), &sampled_ty),
        span,
        TermKind::Var(VarRef::Builtin {
            id: texture_sample,
            overload_idx: 0,
        }),
    );
    let texture = Term::fresh(term_ids, texture_ty, span, TermKind::Var(VarRef::Symbol(texture)));
    let sampler = clone_term_with_fresh_ids(sampler, term_ids);
    let uv = clone_term_with_fresh_ids(uv, term_ids);
    let lod = Term::fresh(term_ids, lod_ty, span, TermKind::FloatLit(0.0));
    let sampled = Term::fresh(
        term_ids,
        sampled_ty.clone(),
        span,
        TermKind::App {
            func: Box::new(function),
            args: vec![texture, sampler, uv, lod],
        },
    );
    target_leaf_from_vec4(sampled, leaf_ty, span, term_ids)
}

fn sampled_target_leaf(
    texture: SymbolId,
    leaf_ty: &Type,
    coord: &Term,
    span: Span,
    texture_load: builtins::BuiltinId,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    let texture_ty = target_read_type();
    let coord_ty = vec_ty(2, i32_ty());
    let lod_ty = i32_ty();
    let sampled_ty = vec_ty(4, f32_ty());
    let function_params = [texture_ty.clone(), coord_ty, lod_ty.clone()];
    let function = Term::fresh(
        term_ids,
        curried_function_type(function_params.iter(), &sampled_ty),
        span,
        TermKind::Var(VarRef::Builtin {
            id: texture_load,
            overload_idx: 0,
        }),
    );
    let texture = Term::fresh(term_ids, texture_ty, span, TermKind::Var(VarRef::Symbol(texture)));
    let coord = clone_term_with_fresh_ids(coord, term_ids);
    let lod = Term::fresh(term_ids, lod_ty, span, TermKind::IntLit("0".to_string()));
    let sampled = Term::fresh(
        term_ids,
        sampled_ty.clone(),
        span,
        TermKind::App {
            func: Box::new(function),
            args: vec![texture, coord, lod],
        },
    );

    target_leaf_from_vec4(sampled, leaf_ty, span, term_ids)
}

fn target_leaf_from_vec4(
    sampled: Term,
    leaf_ty: &Type,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    if *leaf_ty == vec_ty(4, f32_ty()) {
        return Some(sampled);
    }
    if *leaf_ty == f32_ty() {
        return Some(Term::fresh(
            term_ids,
            leaf_ty.clone(),
            span,
            TermKind::TupleProj {
                tuple: Box::new(sampled),
                idx: 0,
            },
        ));
    }
    let Type::Constructed(TypeName::Vec, args) = leaf_ty else {
        return None;
    };
    let [element, Type::Constructed(TypeName::Size(width), _)] = args.as_slice() else {
        return None;
    };
    if *element != f32_ty() || *width > 4 {
        return None;
    }
    let components = (0..*width)
        .map(|index| {
            let sampled = clone_term_with_fresh_ids(&sampled, term_ids);
            Term::fresh(
                term_ids,
                f32_ty(),
                span,
                TermKind::TupleProj {
                    tuple: Box::new(sampled),
                    idx: index,
                },
            )
        })
        .collect();
    Some(Term::fresh(
        term_ids,
        leaf_ty.clone(),
        span,
        TermKind::VecLit(components),
    ))
}

fn rebuild_target_value(
    ty: &Type,
    leaves: &mut impl Iterator<Item = Term>,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    let kind = match ty {
        Type::Constructed(TypeName::Record(_), components)
        | Type::Constructed(TypeName::Tuple(_), components) => TermKind::Tuple(
            components
                .iter()
                .map(|component| rebuild_target_value(component, leaves, span, term_ids))
                .collect::<Option<Vec<_>>>()?,
        ),
        _ => return leaves.next(),
    };
    Some(Term::fresh(term_ids, ty.clone(), span, kind))
}

fn varying_leaf_types(ty: &Type) -> Vec<(String, Type)> {
    fn collect(ty: &Type, prefix: &str, leaves: &mut Vec<(String, Type)>) {
        match ty {
            Type::Constructed(TypeName::Unit, _) => {}
            Type::Constructed(TypeName::Record(fields), components) => {
                for (field, component) in fields.iter().zip(components) {
                    let name = if prefix.is_empty() { field.clone() } else { format!("{prefix}_{field}") };
                    collect(component, &name, leaves);
                }
            }
            Type::Constructed(TypeName::Tuple(_), components) => {
                for (index, component) in components.iter().enumerate() {
                    let name = if prefix.is_empty() {
                        format!("value_{index}")
                    } else {
                        format!("{prefix}_{index}")
                    };
                    collect(component, &name, leaves);
                }
            }
            _ => leaves.push((
                if prefix.is_empty() { "value".to_string() } else { prefix.to_string() },
                ty.clone(),
            )),
        }
    }

    let mut leaves = Vec::new();
    collect(ty, "", &mut leaves);
    leaves
}

fn rebuild_varying_value(
    ty: &Type,
    symbols: &mut impl Iterator<Item = SymbolId>,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    let kind = match ty {
        Type::Constructed(TypeName::Unit, _) => TermKind::UnitLit,
        Type::Constructed(TypeName::Record(_), components)
        | Type::Constructed(TypeName::Tuple(_), components) => {
            let values = components
                .iter()
                .map(|component| rebuild_varying_value(component, symbols, span, term_ids))
                .collect::<Option<Vec<_>>>()?;
            TermKind::Tuple(values)
        }
        _ => TermKind::Var(VarRef::Symbol(symbols.next()?)),
    };
    Some(Term::fresh(term_ids, ty.clone(), span, kind))
}

fn flatten_varying_term(term: Term, term_ids: &mut TermIdSource, leaves: &mut Vec<Term>) {
    let components = match &term.ty {
        Type::Constructed(TypeName::Unit, _) => return,
        Type::Constructed(TypeName::Record(_), components)
        | Type::Constructed(TypeName::Tuple(_), components) => Some(components.clone()),
        _ => None,
    };
    let Some(components) = components else {
        leaves.push(term);
        return;
    };

    match term.kind {
        TermKind::Tuple(values) if values.len() == components.len() => {
            for value in values {
                flatten_varying_term(value, term_ids, leaves);
            }
        }
        _ => {
            for (index, component) in components.into_iter().enumerate() {
                let tuple = clone_term_with_fresh_ids(&term, term_ids);
                let projection = Term::fresh(
                    term_ids,
                    component,
                    term.span,
                    TermKind::TupleProj {
                        tuple: Box::new(tuple),
                        idx: index,
                    },
                );
                flatten_varying_term(projection, term_ids, leaves);
            }
        }
    }
}

fn build_vertex_stage(
    root: &Def<UnpinnedPolymorphic>,
    owner: &str,
    callback: &Lambda,
    external_params: Vec<(SymbolId, Type)>,
    external_decls: Vec<interface::EntryParamDecl>,
    substitutions: ExternalSubstitutions,
    target_reads: TargetReads,
    builtins: &InvocationBuiltins,
    graphics_group: interface::GraphicsStageGroup,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<Def<UnpinnedPolymorphic>> {
    let invocation_symbol = callback.params[0].0;
    let payload_ty = match &callback.ret_ty {
        Type::Constructed(TypeName::Vertex, args) if args.len() == 1 => args[0].clone(),
        _ => return None,
    };

    let varying_leaves = varying_leaf_types(&payload_ty);
    let result_ty = if varying_leaves.is_empty() {
        vec_ty(4, f32_ty())
    } else {
        Type::Constructed(
            TypeName::Tuple(1 + varying_leaves.len()),
            std::iter::once(vec_ty(4, f32_ty()))
                .chain(varying_leaves.iter().map(|(_, ty)| ty.clone()))
                .collect(),
        )
    };
    let used = used_projection_indices(&callback.body, invocation_symbol, 3);
    let mut mapping = vec![None; 3];
    let mut invocation_params = Vec::new();
    let mut invocation_decls = Vec::new();
    for (index, used) in used.into_iter().enumerate() {
        if !used {
            continue;
        }
        let (name, builtin) = match index {
            0 => ("vertex_index", spirv::BuiltIn::VertexIndex),
            1 => ("instance_index", spirv::BuiltIn::InstanceIndex),
            2 => ("draw_index", spirv::BuiltIn::DrawIndex),
            _ => unreachable!(),
        };
        let symbol = symbols.alloc(format!("_w_vertex_{name}"));
        let ty = u32_ty();
        mapping[index] = Some(ProjectionReplacement::Symbol(symbol));
        invocation_params.push((symbol, ty.clone()));
        invocation_decls.push(interface_param(
            name,
            ty,
            Attribute::BuiltIn(builtin),
            callback.body.span,
        ));
    }

    let mut body = clone_term_with_fresh_ids(&callback.body, term_ids);
    body = ExternalValueRewriter {
        term_ids,
        substitutions,
        target_reads,
        target_load: builtins.target_load,
        target_sample: builtins.target_sample,
        texture_load: builtins.texture_load,
        texture_sample: builtins.texture_sample,
    }
    .rewrite_owned(body);
    let mut rewriter = StageBodyRewriter {
        vertex_result_ty: Some(result_ty.clone()),
        term_ids,
        invocation_symbol,
        projections: mapping,
        vertex_output: Some(builtins.vertex_output),
    };
    body = rewriter.rewrite_owned(body);

    let mut params = invocation_params;
    params.extend(external_params);
    let mut declarations = invocation_decls;
    declarations.extend(external_decls);

    let stage_name = format!("_w_stage_{owner}__vertex");
    let mut outputs = vec![interface::EntryOutputDecl {
        ty: vec_ty(4, f32_ty()),
        attribute: Some(Attribute::BuiltIn(spirv::BuiltIn::Position)),
    }];
    outputs.extend(varying_leaves.into_iter().enumerate().map(|(location, (_, ty))| {
        interface::EntryOutputDecl {
            ty,
            attribute: Some(Attribute::Varying(location as u32)),
        }
    }));
    Some(stage_def(
        stage_name,
        EntryKind::Vertex,
        params,
        declarations,
        outputs,
        body,
        Some(graphics_group),
        root,
        symbols,
        term_ids,
    ))
}

fn fragment_output_color_type(ty: &Type) -> Option<Type> {
    let Type::Constructed(TypeName::Tuple(4), components) = ty else {
        return None;
    };
    let [tag, color, depth_color, depth] = components.as_slice() else {
        return None;
    };
    if *tag != u32_ty() || color != depth_color || *depth != f32_ty() {
        return None;
    }
    Some(color.clone())
}

fn fragment_output_projection(
    result: SymbolId,
    result_ty: &Type,
    index: usize,
    ty: Type,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Term {
    let value = Term::fresh(
        term_ids,
        result_ty.clone(),
        span,
        TermKind::Var(VarRef::Symbol(result)),
    );
    Term::fresh(
        term_ids,
        ty,
        span,
        TermKind::TupleProj {
            tuple: Box::new(value),
            idx: index,
        },
    )
}

fn fragment_tag_is(
    result: SymbolId,
    result_ty: &Type,
    tag: u32,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Term {
    let tag_value = fragment_output_projection(result, result_ty, 0, u32_ty(), span, term_ids);
    let expected = Term::fresh(term_ids, u32_ty(), span, TermKind::IntLit(format!("{tag}")));
    let function = Term::fresh(
        term_ids,
        curried_function_type([u32_ty(), u32_ty()].iter(), &bool_ty()),
        span,
        TermKind::BinOp(ast::BinaryOp {
            op: op::BinaryOperator::Equal,
        }),
    );
    Term::fresh(
        term_ids,
        bool_ty(),
        span,
        TermKind::App {
            func: Box::new(function),
            args: vec![tag_value, expected],
        },
    )
}

fn selected_term(
    condition: Term,
    then_branch: Term,
    else_branch: Term,
    ty: Type,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Term {
    Term::fresh(
        term_ids,
        ty,
        span,
        TermKind::If {
            cond: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        },
    )
}

fn lower_fragment_output(
    result: Term,
    color_ty: &Type,
    implicit_depth: Term,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    let span = result.span;
    let result_ty = result.ty.clone();
    let result_symbol = symbols.alloc("_w_fragment_output_value".to_string());

    let color = selected_term(
        fragment_tag_is(result_symbol, &result_ty, 0, span, term_ids),
        fragment_output_projection(result_symbol, &result_ty, 1, color_ty.clone(), span, term_ids),
        fragment_output_projection(result_symbol, &result_ty, 2, color_ty.clone(), span, term_ids),
        color_ty.clone(),
        span,
        term_ids,
    );
    let explicit_depth = fragment_output_projection(result_symbol, &result_ty, 3, f32_ty(), span, term_ids);
    let depth = selected_term(
        fragment_tag_is(result_symbol, &result_ty, 1, span, term_ids),
        explicit_depth,
        implicit_depth,
        f32_ty(),
        span,
        term_ids,
    );
    let discard_mask = Term::fresh(term_ids, u32_ty(), span, TermKind::IntLit("0".to_string()));
    let keep_mask = Term::fresh(
        term_ids,
        u32_ty(),
        span,
        TermKind::IntLit("4294967295".to_string()),
    );
    let sample_mask = selected_term(
        fragment_tag_is(result_symbol, &result_ty, 2, span, term_ids),
        discard_mask,
        keep_mask,
        u32_ty(),
        span,
        term_ids,
    );

    let mut values = Vec::new();
    flatten_varying_term(color, term_ids, &mut values);
    values.push(depth);
    values.push(sample_mask);
    let value_ty = Type::Constructed(
        TypeName::Tuple(values.len()),
        values.iter().map(|value| value.ty.clone()).collect(),
    );
    let value = Term::fresh(term_ids, value_ty.clone(), span, TermKind::Tuple(values));
    Some(Term::fresh(
        term_ids,
        value_ty,
        span,
        TermKind::Let {
            name: result_symbol,
            name_ty: result_ty,
            rhs: Box::new(result),
            body: Box::new(value),
        },
    ))
}

fn build_fragment_stage(
    root: &Def<UnpinnedPolymorphic>,
    owner: &str,
    callback: &Lambda,
    external_params: Vec<(SymbolId, Type)>,
    external_decls: Vec<interface::EntryParamDecl>,
    substitutions: ExternalSubstitutions,
    target_reads: TargetReads,
    target_name: String,
    target_color_ty: Type,
    builtins: &InvocationBuiltins,
    graphics_group: interface::GraphicsStageGroup,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<Def<UnpinnedPolymorphic>> {
    let invocation_symbol = callback.params[0].0;
    let payload_ty = match &callback.params[0].1 {
        Type::Constructed(TypeName::FragmentInvocation, args) if args.len() == 1 => args[0].clone(),
        _ => return None,
    };

    // The flattened representation of fragment_output<C> can coincide with
    // an ordinary tuple color type. The target's C disambiguates the two:
    // a direct callback returns C itself, while the sum representation wraps C.
    let fragment_output = if callback.ret_ty == target_color_ty {
        None
    } else {
        fragment_output_color_type(&callback.ret_ty).filter(|color| *color == target_color_ty)
    };
    let has_fragment_output = fragment_output.is_some();
    let mut used = used_projection_indices(&callback.body, invocation_symbol, 5);
    if has_fragment_output {
        used[1] = true;
    }
    let mut mapping = vec![None; 5];
    let mut invocation_params = Vec::new();
    let mut invocation_decls = Vec::new();

    if used[0] {
        let mut leaf_symbols = Vec::new();
        for (location, (name, ty)) in varying_leaf_types(&payload_ty).into_iter().enumerate() {
            let symbol = symbols.alloc(format!("_w_fragment_value_{name}"));
            leaf_symbols.push(symbol);
            invocation_params.push((symbol, ty.clone()));
            invocation_decls.push(interface_param(
                &format!("value_{name}"),
                ty,
                Attribute::Varying(location as u32),
                callback.body.span,
            ));
        }
        let mut leaf_symbols = leaf_symbols.into_iter();
        mapping[0] = Some(ProjectionReplacement::Term(rebuild_varying_value(
            &payload_ty,
            &mut leaf_symbols,
            callback.body.span,
            term_ids,
        )?));
    }

    let mut position_symbol = None;
    let field_specs = [
        ("position", vec_ty(4, f32_ty()), spirv::BuiltIn::FragCoord),
        ("front_facing", bool_ty(), spirv::BuiltIn::FrontFacing),
        ("primitive_index", u32_ty(), spirv::BuiltIn::PrimitiveId),
        ("sample_index", u32_ty(), spirv::BuiltIn::SampleId),
    ];
    for (offset, (name, ty, builtin)) in field_specs.into_iter().enumerate() {
        let index = offset + 1;
        if !used[index] {
            continue;
        }
        let symbol = symbols.alloc(format!("_w_fragment_{name}"));
        if index == 1 {
            position_symbol = Some(symbol);
        }
        mapping[index] = Some(ProjectionReplacement::Symbol(symbol));
        invocation_params.push((symbol, ty.clone()));
        invocation_decls.push(interface_param(
            name,
            ty,
            Attribute::BuiltIn(builtin),
            callback.body.span,
        ));
    }

    let mut body = clone_term_with_fresh_ids(&callback.body, term_ids);
    body = ExternalValueRewriter {
        term_ids,
        substitutions,
        target_reads,
        target_load: builtins.target_load,
        target_sample: builtins.target_sample,
        texture_load: builtins.texture_load,
        texture_sample: builtins.texture_sample,
    }
    .rewrite_owned(body);
    let mut rewriter = StageBodyRewriter {
        term_ids,
        invocation_symbol,
        vertex_result_ty: None,
        projections: mapping,
        vertex_output: None,
    };
    body = rewriter.rewrite_owned(body);
    let (color_ty, body) = if let Some(color_ty) = fragment_output {
        let position_symbol = position_symbol?;
        let position = Term::fresh(
            term_ids,
            vec_ty(4, f32_ty()),
            body.span,
            TermKind::Var(VarRef::Symbol(position_symbol)),
        );
        let implicit_depth = Term::fresh(
            term_ids,
            f32_ty(),
            body.span,
            TermKind::TupleProj {
                tuple: Box::new(position),
                idx: 2,
            },
        );
        let body = lower_fragment_output(body, &color_ty, implicit_depth, symbols, term_ids)?;
        (color_ty, body)
    } else {
        (body.ty.clone(), body)
    };
    let attachments = attachment_specs(&target_name, &color_ty);
    let body_span = body.span;
    let body = if has_fragment_output {
        body
    } else {
        let mut color_values = Vec::new();
        flatten_varying_term(body, term_ids, &mut color_values);
        if color_values.len() == 1 {
            color_values.pop()?
        } else {
            let component_types = attachments.iter().map(|(_, ty)| ty.clone()).collect::<Vec<_>>();
            Term::fresh(
                term_ids,
                Type::Constructed(TypeName::Tuple(component_types.len()), component_types),
                body_span,
                TermKind::Tuple(color_values),
            )
        }
    };

    let mut params = invocation_params;
    params.extend(external_params);
    let mut declarations = invocation_decls;
    declarations.extend(external_decls);

    let stage_name = format!("_w_stage_{owner}__fragment");
    let mut outputs = attachments
        .into_iter()
        .map(|(name, ty)| interface::EntryOutputDecl {
            ty,
            attribute: Some(Attribute::Target(name)),
        })
        .collect::<Vec<_>>();
    if has_fragment_output {
        outputs.push(interface::EntryOutputDecl {
            ty: f32_ty(),
            attribute: Some(Attribute::BuiltIn(spirv::BuiltIn::FragDepth)),
        });
        outputs.push(interface::EntryOutputDecl {
            ty: u32_ty(),
            attribute: Some(Attribute::BuiltIn(spirv::BuiltIn::SampleMask)),
        });
    }
    Some(stage_def(
        stage_name,
        EntryKind::Fragment,
        params,
        declarations,
        outputs,
        body,
        Some(graphics_group),
        root,
        symbols,
        term_ids,
    ))
}

fn root_entry_name(root: &Def<UnpinnedPolymorphic>) -> Option<String> {
    match &root.meta {
        DefMeta::EntryPoint(entry) => Some(entry.declaration.name.clone()),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn stage_def(
    name: String,
    kind: EntryKind,
    params: Vec<(SymbolId, Type)>,
    param_decls: Vec<interface::EntryParamDecl>,
    outputs: Vec<interface::EntryOutputDecl<interface::ResolvedAttribute>>,
    body: Term,
    graphics_group: Option<interface::GraphicsStageGroup>,
    root: &Def<UnpinnedPolymorphic>,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Def<UnpinnedPolymorphic> {
    let span = root_entry_span(root);
    let result_ty = body.ty.clone();
    let function_ty = curried_function_type(params.iter().map(|(_, ty)| ty), &result_ty);
    let lambda = super::rebuild_nested_lam(&params, body, span, term_ids);
    let symbol = symbols.alloc(name.clone());
    let arity = params.len();

    Def {
        data: data::PolymorphicDefinition {
            scheme: Some(TypeScheme::Monotype(function_ty.clone())),
        },
        name: symbol,
        ty: function_ty,
        body: lambda,
        meta: DefMeta::EntryPoint(EntryPoint {
            declaration: Box::new(interface::EntryDecl {
                entry_kind: kind,
                compute_dispatch: None,
                graphics_group,
                name,
                name_span: span,
                size_params: vec![],
                type_params: vec![],
                params: param_decls,
                outputs,
                feedback: vec![],
                param_diets: vec![Diet::observing(); arity],
                return_diet: Diet::observing(),
            }),
            data: (),
        }),
        arity,
        param_diets: vec![Diet::observing(); arity],
        return_diet: Diet::observing(),
    }
}

fn root_entry_span(root: &Def<UnpinnedPolymorphic>) -> Span {
    match &root.meta {
        DefMeta::EntryPoint(entry) => entry.declaration.name_span,
        _ => Span::new(0, 0, 0, 0),
    }
}

fn interface_param(
    name: &str,
    ty: Type,
    attribute: interface::ResolvedAttribute,
    span: Span,
) -> interface::EntryParamDecl {
    interface::EntryParamDecl {
        name: name.to_string(),
        span,
        ty,
        attributes: vec![attribute],
    }
}

fn used_projection_indices(term: &Term, symbol: SymbolId, count: usize) -> Vec<bool> {
    let mut used = vec![false; count];
    let mut visitor = |term: &Term| {
        if let TermKind::TupleProj { tuple, idx } = &term.kind {
            if matches!(tuple.kind, TermKind::Var(VarRef::Symbol(found)) if found == symbol)
                && *idx < used.len()
            {
                used[*idx] = true;
            }
        }
        WalkDecision::Recurse
    };
    visitor.walk(term);
    used
}

#[derive(Clone)]
enum ProjectionReplacement {
    Symbol(SymbolId),
    Term(Term),
}

struct StageBodyRewriter<'a> {
    term_ids: &'a mut TermIdSource,
    vertex_result_ty: Option<Type>,
    invocation_symbol: SymbolId,
    projections: Vec<Option<ProjectionReplacement>>,
    vertex_output: Option<builtins::BuiltinId>,
}

impl TermRewriter<data::Empty, data::Empty> for StageBodyRewriter<'_> {
    fn next_term_id(&mut self) -> super::TermId {
        self.term_ids.next_id()
    }

    fn rewrite_owned_node(&mut self, mut term: Term) -> (Term, RewriteDecision) {
        if let TermKind::TupleProj { tuple, idx } = &term.kind {
            if matches!(tuple.kind, TermKind::Var(VarRef::Symbol(found)) if found == self.invocation_symbol)
            {
                if let Some(Some(replacement)) = self.projections.get(*idx).cloned() {
                    match replacement {
                        ProjectionReplacement::Symbol(symbol) => {
                            term.kind = TermKind::Var(VarRef::Symbol(symbol));
                        }
                        ProjectionReplacement::Term(value) => {
                            term = clone_term_with_fresh_ids(&value, self.term_ids);
                        }
                    }
                    return (term, RewriteDecision::Changed);
                }
            }
        }

        if let (Some(vertex_output), TermKind::App { func, args }) = (self.vertex_output, &term.kind) {
            if matches!(func.kind, TermKind::Var(VarRef::Builtin { id, .. }) if id == vertex_output)
                && args.len() == 2
            {
                let mut values = vec![args[0].clone()];
                flatten_varying_term(args[1].clone(), self.term_ids, &mut values);
                term.ty = self.vertex_result_ty.clone().expect("vertex output rewrite has a result type");
                term.kind = if values.len() == 1 {
                    values.pop().expect("position output").kind
                } else {
                    TermKind::Tuple(values)
                };
                return (term, RewriteDecision::Changed);
            }
        }

        if let Type::Constructed(TypeName::Vertex, args) = &term.ty {
            if args.len() == 1 {
                term.ty = self.vertex_result_ty.clone().expect("vertex value rewrite has a result type");
                return (term, RewriteDecision::Changed);
            }
        }

        (term, RewriteDecision::Unchanged)
    }
}

fn bool_ty() -> Type {
    Type::Constructed(TypeName::Bool, vec![])
}

fn i32_ty() -> Type {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn u32_ty() -> Type {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn f32_ty() -> Type {
    Type::Constructed(TypeName::Float(32), vec![])
}

fn vec_ty(size: usize, element: Type) -> Type {
    Type::Constructed(
        TypeName::Vec,
        vec![element, Type::Constructed(TypeName::Size(size), vec![])],
    )
}
