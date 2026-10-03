//! Choose scopes once, over egglog's extracted DAG. Semantic safety is proved
//! by scalar rules; no relation enumerates possible expression/scope pairs.
use super::extract::{self, operands};
use super::scopes::{Forest, Scope, Scopes};
use super::Selected;
use crate::egglog::query::Query;
use crate::egglog::{timing, OptimizeError, Optimized, Program, ScalarOptimization};
use crate::{LookupMap, LookupSet};
use egglog_engine::ast::Literal;
use egglog_engine::sort::{VecContainer, F, S};
use egglog_engine::{Core, EGraph, RawValues, Read, Term, Value};

pub(super) fn select(graph: &mut EGraph, policy: ScalarOptimization) -> Result<Selected, OptimizeError> {
    let _timing = timing::span("egglog scalar / exit placement");
    let extraction = timing::span("egglog scalar / extraction");
    let mut roots = Vec::new();
    graph.constructor_enodes("ScalarRoot", |row| {
        roots.push((row.children[0], row.children[1], row.children[2], row.children[3]));
    })?;
    let (dag, terms) = extract::select(graph, &roots.iter().map(|r| r.3).collect::<Vec<_>>(), policy)?;
    let mut selected_roots = LookupMap::default();
    for ((context, _, source, _), term) in roots.into_iter().zip(terms) {
        selected_roots.insert((context, source), term);
    }
    drop(extraction);
    // Recover opaque handles for the selected egglog terms without constructing
    // another expression representation or serializing source identities.
    let values = graph.update(|mut sink| {
        let mut values = Vec::with_capacity(dag.size());
        for id in 0..dag.size() {
            let value = match dag.get(id) {
                Term::Lit(literal) => match literal {
                    Literal::Int(x) => sink.base_to_value(*x),
                    Literal::Float(x) => sink.base_to_value(F::from(*x)),
                    Literal::String(x) => sink.base_to_value(S::new(x.clone())),
                    Literal::Bool(x) => sink.base_to_value(*x),
                    Literal::Unit => sink.base_to_value(()),
                },
                Term::App(name, children) if name == "vec-of" => sink.container_to_value(VecContainer {
                    data: children.iter().map(|&child| values[child]).collect(),
                    do_rebuild: true,
                }),
                Term::App(name, children) => {
                    let args = RawValues(children.iter().map(|&child| values[child]).collect());
                    let Some(value) = sink.eclass_of(name, args)? else {
                        return Err(egglog_engine::Error::ExtractError(
                            "selected enode is missing".into(),
                        ));
                    };
                    value
                }
                Term::Var(_) => {
                    return Err(egglog_engine::Error::ExtractError(
                        "variable in scalar extraction".into(),
                    ));
                }
            };
            values.push(value);
        }
        Ok(values)
    })?;
    Ok(Selected {
        dag,
        values,
        roots: selected_roots,
    })
}

pub(in crate::egglog) fn run(program: &Program<'_, Optimized>) -> Result<Vec<Vec<Value>>, OptimizeError> {
    let _placement = timing::span("egglog scalar / scope placement");
    let graph = &program.stage.scalars;
    let facts = Query(&program.graph);
    let identities = &program.identities;
    let selected = &program.stage.selected;
    let dag = &selected.dag;
    let values = &selected.values;
    let mut uses = vec![Vec::new(); dag.size()];
    graph.constructor_enodes("ScalarRoot", |row| {
        let term = selected.roots[&(row.children[0], row.children[2])];
        if !uses[term].contains(&row.children[1]) {
            uses[term].push(row.children[1]);
        }
    })?;
    let total = (0..dag.size())
        .map(|id| {
            let Term::App(name, fields) = dag.get(id) else {
                return Ok(false);
            };
            if fields.is_empty() {
                return Ok(false);
            }
            let predicate = if matches!(name.as_str(), "ScalarCons" | "ScalarNil") {
                "ScalarArgsTotal"
            } else {
                "ScalarTotal"
            };
            graph.read(|r| r.contains(predicate, values[id]))
        })
        .collect::<Result<Vec<_>, egglog_engine::Error>>()?;
    let parent = |scope| {
        if facts.flag("ScalarRegionBoundary", (scope,))? {
            Ok(None)
        } else {
            Ok(identities.scopes.get(&scope).and_then(|s| s.0))
        }
    };
    let definitions = Forest::new(identities.scopes.keys().copied(), parent)?;
    let mut members = LookupMap::<Value, Vec<Value>>::default();
    graph.constructor_enodes("ScalarContextScope", |row| {
        members.entry(row.children[0]).or_default().push(row.children[1]);
    })?;
    let contexts = members
        .into_iter()
        .map(|(context, members)| Ok((context, Forest::new(members, parent)?)))
        .collect::<Result<_, OptimizeError>>()?;
    let mut loops = LookupSet::default();
    program.graph.constructor_enodes("SourceLoop", |row| {
        loops.insert(row.children[1]);
    })?;
    let scopes = Scopes::new(definitions, contexts, loops);
    // Requirements summarize definitions used by the selected DAG. This is a
    // dependency summary, not a second implementation of semantic safety.
    let mut requirements: Vec<Vec<Scope>> = Vec::with_capacity(dag.size());
    for id in 0..dag.size() {
        let mut required = Vec::new();
        if let Term::App(name, fields) = dag.get(id) {
            match name.as_str() {
                "ScalarLeaf" | "ScalarExecute" | "ScalarInstruction" | "ScalarCall" => {
                    if let Some(owner) = facts
                        .lookup("ScalarOwner", (values[fields[2]],))?
                        .and_then(|v| scopes.definition(v))
                    {
                        scopes.require(&mut required, owner);
                    }
                }
                "ScalarParameter" => {
                    if let Some(owner) = scopes.definition(values[fields[2]]) {
                        scopes.require(&mut required, owner);
                    }
                }
                _ => {}
            }
            for child in operands(name, fields) {
                for &scope in &requirements[child] {
                    scopes.require(&mut required, scope);
                }
            }
        }
        requirements.push(required);
    }
    let mut placements = vec![Vec::new(); dag.size()];
    // Parents follow their children in TermDag. Visiting backwards collects all
    // use sites before choosing a node's location and propagating to operands.
    for id in (0..dag.size()).rev() {
        let sites = std::mem::take(&mut uses[id]);
        if sites.is_empty() {
            continue;
        }
        let Term::App(name, fields) = dag.get(id) else {
            continue;
        };
        let context = values[fields[0]];
        let mut selected = sites;
        if total[id] {
            let mut common = LookupMap::default();
            for &site in &selected {
                let root = scopes.root(context, site);
                let previous = common.get(&root).copied().unwrap_or(site);
                common.insert(root, scopes.common(context, previous, site).unwrap_or(site));
            }
            let mut shared = Vec::new();
            for site in common.into_values() {
                if scopes.available(&requirements[id], site) {
                    let scope = scopes.outside_loops(context, site, &requirements[id]);
                    if !shared.contains(&scope) {
                        shared.push(scope);
                    }
                } else {
                    // Incomparable definitions cannot authorize shared placement.
                    shared.extend(
                        selected
                            .iter()
                            .copied()
                            .filter(|&s| scopes.root(context, s) == scopes.root(context, site)),
                    );
                }
            }
            selected = shared;
        }
        if !matches!(name.as_str(), "ScalarCons" | "ScalarNil") {
            placements[id] = selected.clone();
        }
        for child in operands(name, fields) {
            // Non-total choice arms stay under their control dependency. Their
            // source scopes have roots of their own; newly inlined arms remain
            // nested in the selected choice for eventual SSA block creation.
            if name == "ScalarChoice" && child != fields[2] && !total[child] {
                continue;
            }
            for &scope in &selected {
                if !uses[child].contains(&scope) {
                    uses[child].push(scope);
                }
            }
        }
    }
    Ok(placements)
}
