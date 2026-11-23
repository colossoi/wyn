use super::{Body, Site};
use crate::ast::Span;
use crate::egglog::dataflow::ir::{
    self, Code, Name, OperationId as WireOperationId, RegionId as WireRegionId,
};
use crate::egglog::graph::CertificateSite;
use crate::egglog::graph::Graph;
use crate::egglog::payload::Payload;
use crate::egglog::{output_error, query::Query, OptimizeError};
use crate::memory_ir::{
    BodyRole, Definition, Operation, OperationId, Region, RegionEdge, RegionId, UseSite, ValueRef,
};
use crate::op::{BinaryOperator, UnaryOperator};
use crate::LookupMap;
use egglog_engine::{RawValues, Value, Write};
use std::mem::take;

impl Body {
    pub(in crate::egglog::fusion::parallel) fn guard_shared_loop(
        &mut self,
        graph: &mut Graph,
        producer: OperationId,
        before: OperationId,
        conditions: Vec<(ValueRef, bool)>,
        fallback: ValueRef,
    ) -> Result<RegionId, OptimizeError> {
        let query = Query(graph);
        let output = ValueRef::Result(producer, 0);
        let name = query.name(self.graph.names[&output])?;
        let ty = self.graph.ir.operations[producer].result_types[0];
        let result_primitive = query.fields("ScalarInfo", ty)?[0];
        let fallback_name = query.name(self.graph.names[&fallback])?;
        let Some(&(first_condition, _)) = conditions.first() else {
            return Err(output_error("shared loop has no demand predicates"));
        };
        let bool_type = query.required("ValueType", query.name(self.graph.names[&first_condition])?)?;
        let inputs = conditions
            .into_iter()
            .map(|(condition, taken)| Ok((Name::Value(query.name(self.graph.names[&condition])?), taken)))
            .collect::<Result<Vec<_>, OptimizeError>>()?;
        let key = self.graph.operations[producer];
        let (guard, arms, predicate, inner, steps) = graph.update(|mut sink| {
            Ok((|| -> Result<_, OptimizeError> {
                let mut steps = Vec::new();
                let mut scalar = |code, inputs: &[Name]| -> Result<Name, OptimizeError> {
                    let slot = steps.len() as i64;
                    let operation = sink.add("DerivedOperation", (key, 94_i64, slot))?;
                    let result = sink.add("DerivedValue", (key, 94_i64, slot))?;
                    sink.set_value_type(result, bool_type)?;
                    let output = Name::Value(result);
                    let code = Code::Scalar(code);
                    let id = ir::operation_at(&mut sink, WireOperationId(operation), code.clone(), inputs)?;
                    ir::outputs(&mut sink, id, &[output])?;
                    let names =
                        inputs.iter().map(|name| name.fact(&mut sink)).collect::<Result<Vec<_>, _>>()?;
                    steps.push((operation, code, names, output.fact(&mut sink)?));
                    Ok(output)
                };
                let mut predicate = None;
                for (condition, taken) in inputs {
                    let condition = if taken {
                        condition
                    } else {
                        scalar(Payload::Unary(UnaryOperator::LogicalNot), &[condition])?
                    };
                    predicate = Some(match predicate {
                        Some(earlier) => {
                            scalar(Payload::Binary(BinaryOperator::LogicalOr), &[earlier, condition])?
                        }
                        None => condition,
                    });
                }
                let Some(predicate) = predicate else {
                    return Err(output_error("shared loop has no demand predicates"));
                };
                let guard = sink.add("DerivedOperation", (key, 94_i64, -1_i64))?;
                let inside = sink.add("DerivedValue", (key, 94_i64, -1_i64))?;
                sink.set_value_type(inside, ty)?;
                let inner = Name::Value(inside);
                let inner_name = inner.fact(&mut sink)?;
                sink.set_output(key, 0_i64, inner_name)?;
                let branch =
                    ir::operation_at(&mut sink, WireOperationId(guard), Code::Branch, &[predicate])?;
                ir::outputs(&mut sink, branch, &[Name::Value(name)])?;
                let mut arms = Vec::new();
                let return_type = sink.add("ReturnScalar", result_primitive)?;
                for (slot, value) in [inner, Name::Value(fallback_name)].into_iter().enumerate() {
                    let slot = slot as i64;
                    let arm = sink.add("DerivedRegion", (key, 94_i64, slot))?;

                    sink.add("FlowScope", arm)?;
                    ir::parameters(&mut sink, WireRegionId(arm), &[])?;
                    ir::yields(&mut sink, WireRegionId(arm), &[value])?;
                    sink.set("RegionReturnType", (arm, 0_i64), return_type)?;
                    sink.set_control(guard, slot, arm);
                    arms.push(arm);
                }
                Ok((guard, arms, predicate.fact(&mut sink)?, inner_name, steps))
            })())
        })??;

        for (key, code, names, name) in steps {
            let inputs = names.into_iter().map(|name| self.graph.refs[&name]).collect();
            let op = self.insert_before(before, key, code, inputs, bool_type, Vec::new())?;
            self.graph.refs.insert(name, ValueRef::Result(op, 0));
            self.graph.names.insert(ValueRef::Result(op, 0), name);
        }
        let edges = arms
            .into_iter()
            .zip([output, fallback])
            .enumerate()
            .map(|(slot, (key, value))| {
                let region = self.graph.ir.regions.insert(Region {
                    yields: vec![value],
                    ..Region::default()
                });
                self.graph.regions.insert(region, key);
                self.regions.insert(key, region);
                RegionEdge {
                    role: BodyRole::Control(slot),
                    region,
                    arguments: Vec::new(),
                }
            })
            .collect();
        let predicate = self.graph.refs[&predicate];
        let scope = self.insert_before(before, guard, Code::Branch, vec![predicate], ty, edges)?;
        self.move_value(output, ValueRef::Result(scope, 0));
        self.graph.refs.insert(inner, output);
        self.graph.names.insert(output, inner);
        let destination = self.control_region(scope, 0)?;
        let otherwise = self.control_region(scope, 1)?;
        self.uses.add(Site::Operand(UseSite::Yield(destination, 0)), inner);
        self.uses.add(
            Site::Operand(UseSite::Yield(otherwise, 0)),
            self.graph.names[&fallback],
        );
        if self.barriers.contains(&producer) {
            self.barriers.insert(scope);
        }
        self.sink_producers(scope, destination, &[producer]);
        Ok(destination)
    }

    fn insert_before(
        &mut self,
        before: OperationId,
        key: Value,
        code: Code,
        inputs: Vec<ValueRef>,
        ty: Value,
        bodies: Vec<RegionEdge>,
    ) -> Result<OperationId, OptimizeError> {
        let region = self.positions[before].0;
        let operations = &self.graph.ir.regions[region].operations;
        let Some(index) = operations.iter().position(|&op| op == before) else {
            return Err(output_error("branch insertion point was removed"));
        };
        let op = self.graph.ir.operations.insert(Operation {
            definition: Definition::Other(code),
            inputs,
            bodies,
            result_types: vec![ty],
            span: Span::generated(),
        });
        self.graph.operations.insert(op, key);
        self.graph.operation_ids.insert(key, op);
        let operations = &mut self.graph.ir.regions[region].operations;
        operations.insert(index, op);
        for (index, &op) in operations.iter().enumerate().skip(index) {
            self.positions.insert(op, (region, index));
        }
        for (slot, value) in self.graph.ir.operations[op].inputs.iter().enumerate() {
            self.uses.add(Site::Operand(UseSite::Input(op, slot)), self.graph.names[value]);
        }
        for edge in &self.graph.ir.operations[op].bodies {
            self.parents.insert(edge.region, op);
        }
        self.relocated.insert(op);
        self.moved.insert(region);
        Ok(op)
    }

    pub(in crate::egglog::fusion::parallel) fn join_branches(
        &mut self,
        first: OperationId,
        second: OperationId,
    ) -> Result<[RegionId; 2], OptimizeError> {
        let arms = [self.control_region(first, 0)?, self.control_region(first, 1)?];
        for (slot, destination) in arms.into_iter().enumerate() {
            let source = self.control_region(second, slot)?;
            self.append_yields(destination, source)?;
            self.append_operations(destination, source);
        }
        self.append_results(first, second);
        self.absorb_effects(first, second);
        self.joined_branches.remove(&second);
        self.joined_branches.insert(first);
        self.drop_op(second);
        Ok(arms)
    }

    pub(super) fn append_yields(
        &mut self,
        destination: RegionId,
        source: RegionId,
    ) -> Result<(), OptimizeError> {
        let yields = take(&mut self.graph.ir.regions[source].yields);
        let start = self.graph.ir.regions[destination].yields.len();
        for (slot, value) in yields.iter().enumerate() {
            self.forget(Site::Operand(UseSite::Yield(source, slot)));
            self.uses.add(
                Site::Operand(UseSite::Yield(destination, start + slot)),
                self.graph.names[value],
            );
        }
        self.graph.ir.regions[destination].yields.extend(yields);
        let region_uses = self.uses.at(Site::Region(source)).map(|(_, name)| name).collect::<Vec<_>>();
        for name in region_uses {
            self.uses.add(Site::Region(destination), name);
        }
        self.forget(Site::Region(source));
        let origins = self.joined_yields.remove(source).unwrap_or_else(|| vec![source]);
        let Some(entry) = self.joined_yields.entry(destination) else {
            return Err(output_error("joined region has no live identity"));
        };
        entry.or_insert_with(|| vec![destination]).extend(origins);
        Ok(())
    }

    pub(super) fn publish_joins(&self, graph: &mut Graph) -> Result<(), OptimizeError> {
        if self.joined_yields.is_empty() {
            return Ok(());
        }
        let query = Query(graph);
        let index = |slot: usize| graph.base_to_value(slot as i64);
        let mut rows = Vec::new();
        let mut destinations = LookupMap::new();
        let mut region_yields = Vec::new();
        for (region, origins) in &self.joined_yields {
            let target = self.graph.regions[region];
            let yields = &self.graph.ir.regions[region].yields;
            let mut base = 0;
            for &origin in origins {
                let origin = self.graph.regions[origin];
                destinations.insert(origin, (target, base));
                let size = query.0.yield_count(origin);
                for slot in 0..size {
                    if let Some(ty) = query.lookup("RegionReturnType", (origin, slot))? {
                        rows.push(("RegionReturnType", vec![target, index(base + slot as usize)], ty));
                    }
                }
                base += size as usize;
            }
            region_yields.push((
                target,
                yields.iter().map(|value| self.graph.names[value]).collect(),
            ));
        }
        let mut certificates = Vec::new();
        let mut outputs = Vec::new();
        for op in self.joined_branches.iter().copied().chain(self.accumulator_origins.keys()) {
            let key = self.graph.operations[op];
            let size = self.graph.ir.operations[op].result_types.len();
            outputs.push((
                key,
                (0..size).map(|slot| self.graph.names[&ValueRef::Result(op, slot)]).collect(),
            ));
            for (_, name) in self.uses.at(Site::Certificate(op)) {
                certificates.push((key, query.name(name)?));
            }
        }
        let mut yields = Vec::new();
        for (&source, &(destination, base)) in &destinations {
            yields.extend(
                graph
                    .yield_certificates(source)
                    .into_iter()
                    .map(|(slot, name)| (destination, slot + base as i64, name)),
            );
        }
        graph.update(|mut sink| {
            for (op, names) in outputs {
                sink.set_outputs(op, names)?;
            }
            for (region, values) in region_yields {
                sink.set_yields(region, values)?;
            }
            for (table, keys, value) in rows {
                sink.remove(table, RawValues(keys.clone()))?;
                sink.set(table, RawValues(keys), value)?;
            }
            for (operation, certificate) in certificates {
                sink.add_certificate(CertificateSite::Operation(operation), certificate);
            }
            for (region, slot, certificate) in yields {
                sink.add_certificate(CertificateSite::Yield(region, slot), certificate);
            }
            Ok(())
        })?;
        Ok(())
    }
}
