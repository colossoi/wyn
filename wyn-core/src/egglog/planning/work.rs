use crate::egglog::dataflow::ir::Code;
use crate::egglog::graph::{Formula, Graph};
use crate::egglog::{query::Query, timing, OptimizeError};
use crate::LookupMap;
use egglog_engine::sort::{Boxed, Z};
use egglog_engine::{Value, Write};
use wyn_graph::dfs_postorder;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Node {
    Region(Value),
    Operation(Value),
}

impl Node {
    fn cell(self) -> (&'static str, Value) {
        match self {
            Self::Region(region) => ("RegionWork", region),
            Self::Operation(operation) => ("OperationWork", operation),
        }
    }
}

enum Estimate {
    Sum,
    Branch,
    Repeat(Z, i64),
    Instruction(i64),
    Unknown,
}

struct Plan {
    estimate: Estimate,
    children: Vec<Node>,
}

impl Plan {
    fn read(query: Query<'_>, node: Node) -> Result<Self, OptimizeError> {
        let graph = query.0;
        let mut children = Vec::new();
        let estimate = match node {
            Node::Region(region) => {
                let operations = if let Some(operations) = graph.region_orders.get(&region) {
                    operations.clone()
                } else {
                    query.operations(region)?
                };
                children.extend(operations.into_iter().map(Node::Operation));
                Estimate::Sum
            }
            Node::Operation(operation) => {
                if let Some(work) = query.lookup("InstructionWork", operation)? {
                    return Ok(Self {
                        estimate: Estimate::Instruction(graph.value_to_base(work)),
                        children,
                    });
                }
                match graph.code(operation)? {
                    Code::Call => {
                        let target = graph.call_target(operation)?;
                        if query.lookup("RecursiveFunction", target)? == Some(graph.base_to_value(false)) {
                            if let Some(root) = graph.root_of(target) {
                                children.push(Node::Region(root));
                            }
                        }
                        Estimate::Repeat(Boxed(1.into()), 1)
                    }
                    Code::Branch => {
                        for slot in [0, 1] {
                            if let Some(region) = graph.control_at(operation, slot) {
                                children.push(Node::Region(region));
                            }
                        }
                        Estimate::Branch
                    }
                    Code::CountedLoop => {
                        let count = graph.formula_at(operation, Formula::LoopCount);
                        if let (Some(count), Some(body)) = (count, graph.control_at(operation, 0)) {
                            if let Some(bound) = query.lookup("StaticIndex", count)? {
                                children.push(Node::Region(body));
                                let bound = graph.value_to_base::<i64>(bound).max(0);
                                return Ok(Self {
                                    estimate: Estimate::Repeat(Boxed(bound.into()), 0),
                                    children,
                                });
                            }
                        }
                        Estimate::Unknown
                    }
                    Code::Parallel => {
                        let domain = graph.operation_domain(operation)?;
                        let layout = graph.iteration_domain(domain)?;
                        let mut extent = Boxed(1.into());
                        for dimension in layout.dimensions {
                            let Some(index) = dimension.extent else {
                                return Ok(Self {
                                    estimate: Estimate::Unknown,
                                    children,
                                });
                            };
                            let Some(size) = query.lookup("StaticIndex", index)? else {
                                return Ok(Self {
                                    estimate: Estimate::Unknown,
                                    children,
                                });
                            };
                            extent = Boxed(&*extent * graph.value_to_base::<i64>(size).max(0));
                        }
                        if let Some(body) = graph.control_at(operation, 0) {
                            children.push(Node::Region(body));
                        }
                        Estimate::Repeat(extent, 1)
                    }
                    _ => Estimate::Unknown,
                }
            }
        };
        Ok(Self { estimate, children })
    }

    fn work(&self, known: &LookupMap<Node, Z>) -> Option<Z> {
        match (&self.estimate, self.children.as_slice()) {
            (Estimate::Sum, terms) => {
                let mut work = Boxed(0.into());
                for term in terms {
                    work = Boxed(&*work + &**known.get(term)?);
                }
                Some(work)
            }
            (Estimate::Branch, [yes, no]) => Some(Boxed(&**known.get(yes)?.max(known.get(no)?) + 1)),
            (Estimate::Repeat(count, overhead), [body]) => {
                Some(Boxed(&**count * (&**known.get(body)? + overhead)))
            }
            (Estimate::Instruction(work), _) => Some(Boxed((*work).into())),
            _ => None,
        }
    }
}

/// Evaluate only requested work and its control or call dependencies.
pub(in crate::egglog) fn solve(graph: &mut Graph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut roots = Vec::new();
    query.for_each("RegionWorkDemand", |row| {
        roots.push(Node::Region(row[0]));
        Ok(())
    })?;
    query.for_each("OperationWorkDemand", |row| {
        roots.push(Node::Operation(row[0]));
        Ok(())
    })?;
    evaluate(graph, roots)
}

pub(in crate::egglog) fn demand_operation(
    graph: &mut Graph,
    operation: Value,
) -> Result<(), OptimizeError> {
    evaluate(graph, [Node::Operation(operation)])
}

fn evaluate(graph: &mut Graph, roots: impl IntoIterator<Item = Node>) -> Result<(), OptimizeError> {
    let _timing = timing::span("Rust work estimates");
    let query = Query(graph);
    let mut known = LookupMap::new();
    let mut pending = LookupMap::new();
    let mut result: Result<(), OptimizeError> = Ok(());
    let order = dfs_postorder(
        roots,
        |_| false,
        |node, children| {
            if result.is_err() {
                return;
            }
            result = (|| {
                let (table, key) = node.cell();
                if let Some(work) = query.lookup(table, key)? {
                    known.insert(node, graph.value_to_base::<Z>(work));
                } else {
                    let plan = Plan::read(query, node)?;
                    children.extend_from_slice(&plan.children);
                    pending.insert(node, plan);
                }
                Ok(())
            })();
        },
    );
    result?;
    graph.update(|mut sink| {
        for node in order {
            let Some(plan) = pending.remove(&node) else {
                continue;
            };
            match node {
                Node::Region(region) => {
                    sink.add("RegionWorkDemand", region)?;
                }
                Node::Operation(operation) => {
                    sink.add("OperationWorkDemand", operation)?;
                    for child in &plan.children {
                        if let Node::Region(region) = *child {
                            sink.add("RegionWorkUser", (region, operation))?;
                        }
                    }
                }
            }
            if let Some(work) = plan.work(&known) {
                let (table, key) = node.cell();
                sink.set(table, key, work.clone())?;
                known.insert(node, work);
            }
        }
        Ok(())
    })?;
    Ok(())
}
