//! Choose scopes once, over egglog's extracted DAG. Semantic safety is proved
//! by scalar rules; no relation enumerates possible expression/scope pairs.
use super::extract::{self, operands};
use super::scopes::{Forest, Scope, Scopes};
use super::{Facts, Selected};
use crate::egglog::{source, timing, OptimizeError};
use crate::{LookupMap, LookupSet};
use egglog_engine::ast::Literal;
use egglog_engine::sort::{F, S};
use egglog_engine::{Core, EGraph, RawValues, Read, Term, Value, Write};

pub(super) fn run(
    graph: &mut EGraph,
    identities: &source::Identities<'_>,
    facts: &Facts,
) -> Result<Selected, OptimizeError> {
    let _timing = timing::span("egglog scalar / exit placement");
    let extraction = timing::span("egglog scalar / extraction");
    let mut roots = Vec::new();
    graph.constructor_enodes("ScalarRoot", |row| {
        roots.push((row.children[0], row.children[1], row.children[2], row.children[3]));
    })?;
    let (dag, terms) = extract::select(graph, &roots.iter().map(|r| r.3).collect::<Vec<_>>())?;
    let mut uses: Vec<Vec<Value>> = Vec::new();
    let mut selected_roots = LookupMap::default();
    for ((context, scope, source, _), term) in roots.into_iter().zip(terms) {
        uses.resize_with(dag.size(), Vec::new);
        if !uses[term].contains(&scope) {
            uses[term].push(scope);
        }
        selected_roots.insert((context, source), term);
    }
    drop(extraction);
    let _placement = timing::span("egglog scalar / scope placement");
    // Recover opaque handles for the selected egglog terms without constructing
    // another expression representation or serializing source identities.
    let values = graph.update(|sink| {
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
        if facts.boundaries.contains(&scope) {
            None
        } else {
            identities.scopes.get(&scope).and_then(|s| s.0)
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
    graph.constructor_enodes("SourceLoop", |row| {
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
                "ScalarLeaf" => {
                    if let Some(owner) =
                        facts.owners.get(&values[fields[2]]).and_then(|v| scopes.definition(*v))
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
    let mut placements = Vec::new();
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
            placements.extend(selected.iter().map(|&scope| (context, values[id], scope)));
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
    graph.update(|mut sink| {
        for (context, expression, scope) in placements {
            sink.add("ScalarSelectedPlacement", (context, expression, scope))?;
        }
        Ok(())
    })?;
    Ok(Selected {
        dag,
        values,
        roots: selected_roots,
    })
}
