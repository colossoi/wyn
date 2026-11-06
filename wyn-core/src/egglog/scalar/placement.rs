//! Plan lexical and guarded materializations over the selected native DAG.
//! SSA emission queries dependency requirements while constructing control flow.
use super::extract;
use super::scopes::{Forest, Scope, Scopes};
use super::Selected;
use crate::egglog::query::Query;
use crate::egglog::{timing, OptimizeError, Optimized, Program, ScalarOptimization};
use crate::{LookupMap, LookupSet};
use egglog_engine::ast::Literal;
use egglog_engine::sort::{VecContainer, F, S};
use egglog_engine::{Core, EGraph, RawValues, Read, Term, TermId, Value};

pub(super) fn select(graph: &mut EGraph, policy: ScalarOptimization) -> Result<Selected, OptimizeError> {
    let _timing = timing::span("egglog scalar / exit placement");
    let extraction = timing::span("egglog scalar / extraction");
    let mut roots = Vec::new();
    graph.constructor_enodes("ExprRoot", |row| {
        roots.push((row.children[0], row.children[1], row.children[2], row.children[3]));
    })?;
    let mut extents = Vec::new();
    graph.constructor_enodes("ExtentExprRoot", |row| {
        extents.push((row.children[0], row.children[1]))
    })?;
    let values = roots.iter().map(|r| r.3).chain(extents.iter().map(|r| r.1)).collect::<Vec<_>>();
    let (dag, terms) = extract::select(graph, &values, policy)?;
    let selected_extents = extents
        .into_iter()
        .zip(terms[roots.len()..].iter().copied())
        .map(|((extent, _), term)| (extent, term))
        .collect();
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
        extents: selected_extents,
    })
}

pub(in crate::egglog) struct Placement {
    pub(super) scopes: Scopes,
    requirements: Vec<Vec<Scope>>,
}

impl Placement {
    pub fn new(program: &Program<'_, Optimized>) -> Result<Self, OptimizeError> {
        let _placement = timing::span("egglog scalar / scope placement");
        let graph = &program.stage.scalars;
        let facts = Query(&program.graph);
        let identities = &program.identities;
        let selected = &program.stage.selected;
        let dag = &selected.dag;
        let parent = |scope| {
            if facts.flag("ExprContextBoundary", (scope,))? {
                Ok(None)
            } else {
                Ok(identities.scopes.get(&scope).and_then(|s| s.0))
            }
        };
        let definitions = Forest::new(identities.scopes.keys().copied(), parent)?;
        let mut members = LookupMap::<Value, Vec<Value>>::default();
        graph.constructor_enodes("ExprContextMember", |row| {
            members.entry(row.children[0]).or_default().push(row.children[1]);
        })?;
        let contexts = members
            .into_iter()
            .map(|(context, members)| Ok((context, Forest::new(members, parent)?)))
            .collect::<Result<_, OptimizeError>>()?;
        let mut loops = LookupSet::default();
        facts.for_function("SourceLoop", |_, info| {
            let fields = facts
                .enode("LoopInfo", info)?
                .ok_or_else(|| crate::egglog::output_error("missing loop info"))?;
            loops.insert(fields[0]);
            Ok(())
        })?;
        let scopes = Scopes::new(definitions, contexts, loops);
        Ok(Self {
            scopes,
            requirements: vec![Vec::new(); dag.size()],
        })
    }

    pub(super) fn load_requirements(&mut self, graph: &EGraph) -> Result<(), OptimizeError> {
        let query = Query(graph);
        query.for_each("FlowScopeRequirement", |row| {
            let Some(node) = query.enode("FlowNode", row[0])? else {
                return Err(crate::egglog::output_error("missing selected node"));
            };
            self.scopes.require(
                &mut self.requirements[graph.value_to_base::<i64>(node[0]) as usize],
                Scope(graph.value_to_base::<i64>(row[1]) as usize),
            );
            Ok(())
        })?;
        for required in &mut self.requirements {
            required.sort_by_key(|s| s.0);
        }
        Ok(())
    }

    pub fn available(&self, term: TermId, owner: Value, scope: Value) -> bool {
        self.scopes.definition(owner).is_some_and(|owner| {
            self.scopes.available(&[owner], scope) && self.scopes.available(&self.requirements[term], scope)
        })
    }

    pub fn sites(&self, context: Value, scope: Value, term: TermId) -> Vec<(Value, usize, usize)> {
        self.scopes.sites(context, scope, &self.requirements[term])
    }

    pub(super) fn required_scopes(&self, term: TermId) -> &[Scope] {
        &self.requirements[term]
    }
}
