//! Plan lexical and guarded materializations over the selected native DAG.
//! SSA emission consumes these locations without rediscovering shared producers.
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

/// Locations for materialization, including selections sharing a predicate.
pub(in crate::egglog) enum Materialization {
    Value(TermId, Option<Value>),
    Choice {
        terms: Vec<TermId>,
        shared: Vec<Materialization>,
        arms: [Vec<Materialization>; 2],
    },
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

    pub fn readout(
        &self,
        program: &Program<'_, Optimized>,
        scope: Value,
        roots: &[TermId],
        bound: impl Fn(Value) -> bool,
    ) -> Result<Vec<Materialization>, OptimizeError> {
        Readout {
            placement: self,
            program,
            bound,
        }
        .body(scope, roots, &mut LookupSet::default())
    }

    pub fn shared(
        &self,
        program: &Program<'_, Optimized>,
        scope: Value,
        arms: [&[TermId]; 2],
        bound: impl Fn(Value) -> bool,
    ) -> Result<Vec<Materialization>, OptimizeError> {
        let readout = Readout {
            placement: self,
            program,
            bound,
        };
        readout.body(scope, &readout.common(arms)?, &mut LookupSet::default())
    }
}

struct Readout<'a, 'source, B> {
    placement: &'a Placement,
    program: &'a Program<'source, Optimized>,
    bound: B,
}

impl<B: Fn(Value) -> bool> Readout<'_, '_, B> {
    fn dependencies(&self, term: TermId) -> Vec<TermId> {
        let selected = &self.program.stage.selected;
        let Term::App(name, fields) = selected.dag.get(term) else {
            return Vec::new();
        };
        if name == "ScalarLeaf" {
            let source = selected.values[fields[2]];
            if !(self.bound)(source) {
                if let Some(&root) = selected.roots.get(&(selected.values[fields[0]], source)) {
                    if root != term {
                        return vec![root];
                    }
                }
            }
        }
        if name == "ScalarChoice" {
            return vec![fields[2]];
        }
        operands(name, fields)
    }

    fn demands(&self, roots: &[TermId]) -> Vec<TermId> {
        wyn_graph::dag_postorder(
            roots.iter().copied(),
            |_| false,
            |term, out| {
                out.extend(self.dependencies(term));
            },
        )
    }

    fn readonly(&self, roots: &[TermId]) -> Result<bool, OptimizeError> {
        let scalars = &self.program.stage.scalars;
        for &term in roots {
            let (name, _) = self.program.stage.selected.app(term)?;
            let table = if matches!(name, "ScalarCons" | "ScalarNil") {
                "ScalarArgsReadOnly"
            } else {
                "ScalarReadOnly"
            };
            let value = self.program.stage.selected.values[term];
            if !scalars
                .read(|r| r.lookup(table, value))?
                .is_some_and(|value| scalars.value_to_base::<bool>(value))
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn common(&self, arms: [&[TermId]; 2]) -> Result<Vec<TermId>, OptimizeError> {
        if !self.readonly(arms[0])? || !self.readonly(arms[1])? {
            return Ok(Vec::new());
        }
        let yes: LookupSet<_> = self.demands(arms[0]).into_iter().collect();
        Ok(self.demands(arms[1]).into_iter().filter(|term| yes.contains(term)).collect())
    }

    fn body(
        &self,
        scope: Value,
        roots: &[TermId],
        available: &mut LookupSet<TermId>,
    ) -> Result<Vec<Materialization>, OptimizeError> {
        let selected = &self.program.stage.selected;
        let mut order = self.demands(roots);
        let readonly = self.readonly(roots)?;
        let mut groups = LookupMap::<TermId, Vec<TermId>>::default();
        if readonly {
            // Strict dependencies precede their consumers. Delaying selections
            // behind other mandatory work prevents an earlier guarded use from
            // materializing a producer also needed unconditionally in this block.
            let mut levels = LookupMap::default();
            for &term in &order {
                let (name, fields) = selected.app(term)?;
                let level = self.dependencies(term).iter().map(|child| levels[child]).max().unwrap_or(0);
                levels.insert(term, level + usize::from(name == "ScalarChoice"));
                if name == "ScalarChoice" {
                    groups.entry(fields[2]).or_default().push(term);
                }
            }
            order.sort_by_key(|term| levels[term]);
        }
        let mut result = Vec::new();
        for term in order {
            if !available.insert(term) {
                continue;
            }
            let (name, fields) = selected.app(term)?;
            if matches!(name, "ScalarCons" | "ScalarNil") {
                continue;
            }
            if name == "ScalarChoice" {
                let terms = groups.remove(&fields[2]).unwrap_or_else(|| vec![term]);
                let arms = [3, 4].map(|arm| {
                    terms
                        .iter()
                        .map(|&term| selected.app(term).map(|(_, f)| f[arm]))
                        .collect::<Result<Vec<_>, _>>()
                });
                let [yes, no] = arms;
                let (yes, no) = (yes?, no?);
                let shared = self.body(scope, &self.common([&yes, &no])?, available)?;
                let arms = [
                    self.body(scope, &yes, &mut available.clone())?,
                    self.body(scope, &no, &mut available.clone())?,
                ];
                available.extend(terms.iter().copied());
                result.push(Materialization::Choice { terms, shared, arms });
            } else {
                let context = selected.values[fields[0]];
                let total = self
                    .program
                    .stage
                    .scalars
                    .read(|r| r.contains("ScalarTotal", selected.values[term]))?;
                let target = total.then(|| {
                    self.placement.scopes.outside_loops(context, scope, &self.placement.requirements[term])
                });
                result.push(Materialization::Value(term, target));
            }
        }
        Ok(result)
    }
}
