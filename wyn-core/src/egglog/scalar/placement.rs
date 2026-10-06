//! Plan lexical and guarded materializations over the selected native DAG.
//! SSA emission queries dependency requirements while constructing control flow.
use super::extract::{self, operands};
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

pub(in crate::egglog) struct Placement {
    scopes: Scopes,
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
        let values = &selected.values;
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
        Ok(Self { scopes, requirements })
    }

    pub fn available(&self, term: TermId, owner: Value, scope: Value) -> bool {
        self.scopes.definition(owner).is_some_and(|owner| {
            self.scopes.available(&[owner], scope) && self.scopes.available(&self.requirements[term], scope)
        })
    }

    pub fn outside_loops(&self, context: Value, scope: Value, term: TermId) -> Value {
        self.scopes.outside_loops(context, scope, &self.requirements[term])
    }
}
