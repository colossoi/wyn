use super::{Body, ClauseId, ClauseKind, Site};
use crate::ast::Span;
use crate::egglog::dataflow::ir::{
    self, Code, Name, OperationId as WireOperationId, RegionId as WireRegionId,
};
use crate::egglog::graph::CollectiveInput;
use crate::egglog::graph::Graph;
use crate::egglog::{output_error, query::Query, OptimizeError};
use crate::memory_ir::{
    BodyRole, Definition, Operation, OperationId, Region, RegionEdge, UseSite, ValueRef,
};
use egglog_engine::{RawValues, Value, Write};
use std::mem::take;

pub(super) struct Guard {
    operation: OperationId,
    outputs: Vec<Name>,
    returns: Vec<Value>,
}

fn capability(query: Query<'_>, mut token: Value) -> Result<Option<Value>, OptimizeError> {
    loop {
        if let Some(capability) = query.lookup("TokenCapability", token)? {
            return Ok(Some(capability));
        }
        // A guard built in this batch preserves the capability of its true result.
        let Some(fields) = query.enode("ClonedAccumulator", token)? else {
            return Ok(None);
        };
        token = fields[1];
    }
}

impl Body {
    pub(in crate::egglog::fusion::parallel) fn compact_result(
        &mut self,
        operation: OperationId,
        id: ClauseId,
        predicate: Value,
    ) {
        let Some(clause) = self.clauses.get_mut(&id) else {
            return;
        };
        let collected = clause.kind == ClauseKind::Collected;
        if collected {
            clause.kind = ClauseKind::Compacted;
        }
        let mut packed = false;
        for output in &self.results[operation].outputs {
            if output.clause == id && (collected || output.prefix) {
                self.packed_outputs.insert(output.original);
                packed = true;
            }
        }
        if packed {
            clause.predicate = Some(predicate);
            self.uses.add(Site::Result(id), predicate);
            self.changed_clauses.insert(id);
        }
    }

    pub(in crate::egglog::fusion::parallel) fn guard(
        &mut self,
        graph: &mut Graph,
        operation: OperationId,
        clause: ClauseId,
        predicate: Value,
    ) -> Result<(), OptimizeError> {
        let values = self.clause(clause).values.clone();
        let neutral = self.clause(clause).neutral.clone();
        let values: Vec<_> = values.into_iter().map(|name| self.resolve(name)).collect();
        let neutral: Vec<_> = neutral.into_iter().map(|name| self.resolve(name)).collect();
        let key = self.graph.operations[operation];
        let lane = self.lane(operation)?;
        let query = Query(graph);
        let mut otherwise = Vec::new();
        let mut types = Vec::new();
        let captured = self.tree_uses(operation);
        for (slot, &result) in values.iter().enumerate() {
            if let Some(fields) = query.enode("AccNameRef", result)? {
                let token = fields[0];
                let Some(binding) = capability(query, token)? else {
                    return Err(output_error("guarded contribution has no capability"));
                };
                let mut incoming = None;
                for &(_, name) in &captured {
                    let value = self.graph.refs[&name];
                    if !self.inside(self.definition_region(value), lane) {
                        if let Some(fields) = query.enode("AccNameRef", name)? {
                            if capability(query, fields[0])? == Some(binding) {
                                incoming = Some(name);
                                break;
                            }
                        }
                    }
                }
                let Some(incoming) = incoming else {
                    return Err(output_error("guarded contribution has no incoming capability"));
                };
                otherwise.push(incoming);
                let info = query.required("AccumulatorType", token)?;
                types.push((Name::Accumulator(token), info, info));
            } else {
                let value = query.name(result)?;
                let Some(&identity) = neutral.get(slot) else {
                    return Err(output_error("guarded reduction has no neutral value"));
                };
                otherwise.push(identity);
                let info = query.required("ValueType", value)?;
                types.push((Name::Value(value), info, query.fields("ScalarInfo", info)?[0]));
            }
        }

        // Mint bridge keys here. The branch and its arms are installed in the
        // Rust graph below and are published with the completed fusion batch.
        let (branch, arms, names, outputs, returns) = graph.update(|mut sink| {
            Ok((|| -> Result<_, OptimizeError> {
                let branch = sink.add("DerivedOperation", (key, 91_i64, 0_i64))?;
                let arms = [
                    sink.add("DerivedRegion", (key, 91_i64, 2_i64))?,
                    sink.add("DerivedRegion", (key, 91_i64, 0_i64))?,
                ];
                let mut names = Vec::new();
                let mut outputs = Vec::new();
                let mut returns = Vec::new();
                for (slot, &(original, info, returns_type)) in types.iter().enumerate() {
                    let (name, returns_type) = match original {
                        Name::Value(_) => {
                            let value = sink.add("DerivedValue", (key, 91_i64, slot as i64))?;
                            sink.set("ValueType", value, info)?;
                            (Name::Value(value), sink.add("ReturnScalar", returns_type)?)
                        }
                        Name::Accumulator(token) => {
                            let value = sink.add("ClonedAccumulator", (branch, token))?;
                            sink.set("AccumulatorType", value, info)?;
                            (
                                Name::Accumulator(value),
                                sink.add("ReturnAccumulator", returns_type)?,
                            )
                        }
                    };
                    names.push(name.fact(&mut sink)?);
                    outputs.push(name);
                    returns.push(returns_type);
                }
                Ok((branch, arms, names, outputs, returns))
            })())
        })??;
        let mut operations = take(&mut self.graph.ir.regions[lane].operations);
        let mut edges = Vec::new();
        for (slot, (key, yields)) in arms.into_iter().zip([values, otherwise]).enumerate() {
            let region = self.graph.ir.regions.insert(Region {
                operations: take(&mut operations),
                yields: yields.iter().map(|name| self.graph.refs[name]).collect(),
                parameters: Vec::new(),
            });
            self.graph.regions.insert(region, key);
            self.regions.insert(key, region);
            for (index, name) in yields.into_iter().enumerate() {
                self.uses.add(Site::Operand(UseSite::Yield(region, index)), name);
            }
            for (index, &op) in self.graph.ir.regions[region].operations.iter().enumerate() {
                self.positions[op] = (region, index);
                self.relocated.insert(op);
            }
            self.moved.insert(region);
            edges.push(RegionEdge {
                role: BodyRole::Control(slot),
                region,
                arguments: Vec::new(),
            });
        }
        let op = self.graph.ir.append(
            lane,
            Operation {
                definition: Definition::Other(Code::Branch),
                inputs: vec![self.graph.refs[&predicate]],
                bodies: edges,
                result_types: types.into_iter().map(|(_, ty, _)| ty).collect(),
                span: Span::generated(),
            },
        );
        self.graph.operations.insert(op, branch);
        self.graph.operation_ids.insert(branch, op);
        self.positions.insert(op, (lane, 0));
        for edge in &self.graph.ir.operations[op].bodies {
            self.parents.insert(edge.region, op);
        }
        self.uses.add(Site::Operand(UseSite::Input(op, 0)), predicate);
        for (slot, &name) in names.iter().enumerate() {
            let value = ValueRef::Result(op, slot);
            self.graph.refs.insert(name, value);
            self.graph.names.insert(value, name);
        }
        self.forget(Site::Result(clause));
        for &name in &names {
            self.uses.add(Site::Result(clause), name);
        }
        if let Some(predicate) = self.clause(clause).predicate {
            self.uses.add(Site::Result(clause), predicate);
        }
        let Some(result) = self.clauses.get_mut(&clause) else {
            return Err(output_error("guarded result clause disappeared"));
        };
        result.values = names;
        self.changed_clauses.insert(clause);
        self.guards.push(Guard {
            operation: op,
            outputs,
            returns,
        });
        Ok(())
    }

    pub(super) fn publish_guards(&self, graph: &mut Graph) -> Result<(), OptimizeError> {
        if self.changed_clauses.is_empty() {
            return Ok(());
        }
        let query = Query(graph);
        let name = |name| match query.value_name(name)? {
            Some(value) => Ok(Name::Value(value)),
            None => Ok::<_, OptimizeError>(Name::Accumulator(query.name(name)?)),
        };
        let mut guards = Vec::new();
        for guard in &self.guards {
            let branch = &self.graph.ir.operations[guard.operation];
            let predicate = name(self.graph.names[&branch.inputs[0]])?;
            let mut arms = Vec::new();
            for edge in &branch.bodies {
                let region = &self.graph.ir.regions[edge.region];
                let yields = region
                    .yields
                    .iter()
                    .map(|value| name(self.graph.names[value]))
                    .collect::<Result<Vec<_>, _>>()?;
                arms.push((self.graph.regions[edge.region], yields));
            }
            guards.push((guard, predicate, arms));
        }
        let mut clauses = Vec::new();
        for &id in &self.changed_clauses {
            if self.pruned.contains(&id) {
                continue;
            }
            let clause = self.clause(id);
            let predicate = clause.predicate.map(|name| query.name(name)).transpose()?;
            clauses.push((self.graph.operations[id.0], id.1, clause, predicate));
        }
        graph.update(|mut sink| {
            Ok((|| -> Result<(), OptimizeError> {
                for (guard, predicate, arms) in guards {
                    let id = self.graph.operations[guard.operation];
                    let branch =
                        ir::operation_at(&mut sink, WireOperationId(id), Code::Branch, &[predicate])?;
                    ir::outputs(&mut sink, branch, &guard.outputs)?;
                    for (slot, (region, yields)) in arms.into_iter().enumerate() {
                        sink.add("FlowScope", region)?;
                        ir::parameters(&mut sink, WireRegionId(region), &[])?;
                        ir::yields(&mut sink, WireRegionId(region), &yields)?;
                        for (field, &returns) in guard.returns.iter().enumerate() {
                            sink.set("RegionReturnType", (region, field as i64), returns)?;
                        }
                        sink.set_control(id, slot as i64, region);
                    }
                }
                let compacted = sink.add("Compacted", RawValues(vec![]))?;
                let packed = sink.add("CompactedOutput", RawValues(vec![]))?;
                for (key, slot, clause, predicate) in clauses {
                    if clause.kind == ClauseKind::Compacted {
                        sink.set_result_kind(key, slot, compacted)?;
                    }
                    if let Some(predicate) = predicate {
                        sink.set_collective_inputs(key, slot, CollectiveInput::Predicate, vec![predicate])?;
                    }
                    sink.set_collective_inputs(
                        key,
                        slot,
                        CollectiveInput::Contribution,
                        clause.values.clone(),
                    )?;
                }
                for &(op, slot) in &self.packed_outputs {
                    sink.set_output_role(self.graph.operations[op], slot, packed)?;
                }
                Ok(())
            })())
        })??;
        Ok(())
    }
}
