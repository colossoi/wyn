//! Instantiate a region using its actual edges and the caller's argument identities.
use super::ir::Code;
use crate::egglog::graph::{
    ArgumentSite, CertificateSite, CollectiveInput, Formula, Graph, IterationDomain, ParallelResults, State,
};
use crate::egglog::{output_error, query::Query, OptimizeError};
use crate::{LookupMap, LookupSet, StableMap};
use egglog_engine::sort::VecContainer;
use egglog_engine::{Core, Error, Read, Value, Write};

struct Region {
    source: Value,
    parameters: Vec<Value>,
    yields: Vec<Value>,
    certificates: Vec<(i64, Value)>,
    operations: Vec<Operation>,
}

struct Operation {
    source: Value,
    code: Code,
    order: i64,
    target: Option<Value>,
    controls: Vec<(i64, Value, Vec<Value>)>,
    inputs: Vec<Value>,
    outputs: Vec<Value>,
    reducers: Vec<(i64, Value)>,
    combiners: Vec<(i64, Value)>,
    collective: StableMap<(i64, CollectiveInput), Vec<Value>>,
    formulas: Vec<(Formula, Value)>,
    results: Option<ParallelResults>,
    policies: Vec<Option<Value>>,
    domain: Option<Value>,
    certificates: Vec<Value>,
}

impl Operation {
    fn read(graph: &Graph, source: Value) -> Result<Self, OptimizeError> {
        let code = graph.code(source)?;
        let target = matches!(code, Code::Call).then(|| graph.call_target(source)).transpose()?;
        Ok(Self {
            source,
            code,
            order: graph.operation_order(source)?,
            target,
            controls: graph
                .controls(source)
                .map(|(slot, region)| (slot, region, graph.arguments(source, slot)))
                .collect(),
            inputs: graph.inputs(source),
            outputs: graph.outputs(source),
            reducers: graph.reducers(source),
            combiners: graph.combiners(source),
            collective: graph.collective_groups(source),
            formulas: graph.formulas(source),
            results: graph.parallel_results_at(source),
            policies: graph.accumulator_policies(source),
            domain: graph.operation_domain_at(source),
            certificates: graph.operation_certificates(source),
        })
    }
}

pub(in crate::egglog) struct ClonedRegion {
    pub region: Value,
    names: LookupMap<Value, Value>,
}

impl ClonedRegion {
    pub fn name(&self, source: Value) -> Value {
        self.names.get(&source).copied().unwrap_or(source)
    }
}

pub(in crate::egglog) fn clone_region(
    graph: &mut Graph,
    context: Value,
    root: Value,
    arguments: &[Value],
) -> Result<ClonedRegion, OptimizeError> {
    let formals = graph.parameters(root);
    if formals.len() != arguments.len() {
        return Err(output_error("region entry edge has the wrong argument count"));
    }
    let mut names = LookupSet::new();
    let mut pending = vec![root];
    let mut seen = LookupSet::new();
    let mut regions = Vec::new();
    let mut domains = LookupMap::new();
    while let Some(region) = pending.pop() {
        if !seen.insert(region) {
            continue;
        }
        let parameters = graph.parameters(region);
        names.extend(parameters.iter().copied());
        let mut operations = Vec::new();
        for source in Query(graph).operations(region)? {
            let operation = Operation::read(graph, source)?;
            names.extend(operation.outputs.iter().copied());
            pending.extend(graph.attached_regions(source));
            if let Some(domain) = operation.domain {
                if let std::collections::hash_map::Entry::Vacant(entry) = domains.entry(domain) {
                    entry.insert(graph.iteration_domain(domain)?);
                }
            }
            operations.push(operation);
        }
        regions.push(Region {
            source: region,
            parameters,
            yields: graph.yield_values(region),
            certificates: graph.yield_certificates(region),
            operations,
        });
    }
    let query = Query(graph);
    let describe = |name| -> Result<(&str, Value), OptimizeError> {
        let (kind, fields) = query.application(name, &["ValueNameRef", "AccNameRef"])?;
        Ok((kind, fields[0]))
    };
    let bindings = formals
        .into_iter()
        .zip(arguments.iter().copied())
        .map(|(old, new)| {
            names.remove(&old);
            let (kind, original) = describe(old)?;
            let (replacement_kind, replacement) = describe(new)?;
            if kind != replacement_kind {
                return Err(output_error("cloned argument changes the kind of its formal"));
            }
            Ok((old, original, new, replacement, kind))
        })
        .collect::<Result<Vec<_>, OptimizeError>>()?;
    let definitions = names
        .into_iter()
        .map(|name| {
            let (kind, raw) = describe(name)?;
            Ok((name, kind, raw))
        })
        .collect::<Result<Vec<_>, OptimizeError>>()?;

    graph
        .update(|mut sink| {
            let mut clone = CloneNames::default();
            for (old, original, new, replacement, kind) in bindings {
                clone.names.insert(old, new);
                if kind == "ValueNameRef" {
                    clone.values.insert(original, replacement);
                }
            }
            let mut accumulators = Vec::new();
            for (name, kind, raw) in definitions {
                let replacement = if kind == "ValueNameRef" {
                    let value = sink.add("ClonedValue", (context, raw))?;
                    if let Some(ty) = sink.lookup("ValueType", raw)? {
                        sink.set("ValueType", value, ty)?;
                    }
                    clone.values.insert(raw, value);
                    value
                } else {
                    let value = sink.add("ClonedAccumulator", (context, raw))?;
                    if let Some(info) = sink.lookup("AccumulatorType", raw)? {
                        accumulators.push((value, info));
                    }
                    value
                };
                clone.names.insert(name, sink.add(kind, replacement)?);
            }
            for (value, info) in accumulators {
                let info = clone.accumulator(&mut sink, info)?;
                sink.set("AccumulatorType", value, info)?;
            }
            for (domain, layout) in domains {
                let renamed = sink.add("ClonedDomain", (context, domain))?;
                let layout = clone.domain(&mut sink, layout)?;
                sink.set_domain(renamed, layout);
            }
            for region in regions {
                let target = sink.add("ClonedRegion", (context, region.source))?;
                sink.add("FlowScope", target)?;
                for (slot, _) in region.yields.iter().enumerate() {
                    let slot = slot as i64;
                    if let Some(info) = sink.lookup("RegionReturnType", (region.source, slot))? {
                        let info = clone.return_type(&mut sink, info)?;
                        sink.set("RegionReturnType", (target, slot), info)?;
                    }
                }
                for (slot, certificate) in region.certificates {
                    sink.add_certificate(CertificateSite::Yield(target, slot), clone.value(certificate));
                }
                if region.source != root {
                    for (slot, _) in region.parameters.iter().enumerate() {
                        let slot = slot as i64;
                        if let Some(diet) = sink.lookup("ParameterDiet", (region.source, slot))? {
                            sink.set("ParameterDiet", (target, slot), diet)?;
                        }
                    }
                    sink.set_parameters(target, clone.names(region.parameters))?;
                }
                sink.set_yields(target, clone.names(region.yields))?;
                for operation in region.operations {
                    clone.operation(&mut sink, context, target, operation)?;
                }
            }
            Ok(ClonedRegion {
                region: sink.add("ClonedRegion", (context, root))?,
                names: clone.names,
            })
        })
        .map_err(Into::into)
}

#[derive(Default)]
struct CloneNames {
    names: LookupMap<Value, Value>,
    values: LookupMap<Value, Value>,
    indices: LookupMap<Value, Value>,
    formulas: LookupMap<(Formula, Value), Value>,
}

impl CloneNames {
    fn value(&self, name: Value) -> Value {
        self.values.get(&name).copied().unwrap_or(name)
    }

    fn names(&self, mut names: Vec<Value>) -> Vec<Value> {
        for name in &mut names {
            *name = self.names.get(name).copied().unwrap_or(*name);
        }
        names
    }

    fn index(&mut self, sink: &mut State<'_, '_, '_>, term: Value) -> Result<Value, Error> {
        if let Some(&existing) = self.indices.get(&term) {
            return Ok(existing);
        }
        let mut replacement = term;
        if let Some(fields) = fields(sink, "IndexValue", term)? {
            let name = self.value(fields[0]);
            if name != fields[0] {
                replacement = sink.add("IndexValue", name)?;
            }
        } else {
            for constructor in [
                "IndexAdd", "IndexSub", "IndexMul", "IndexDiv", "IndexMod", "IndexMin", "IndexMax",
            ] {
                if let Some(args) = fields(sink, constructor, term)? {
                    let left = self.index(sink, args[0])?;
                    let right = self.index(sink, args[1])?;
                    if (left, right) != (args[0], args[1]) {
                        replacement = sink.add(constructor, (left, right))?;
                    }
                    break;
                }
            }
        }
        self.indices.insert(term, replacement);
        Ok(replacement)
    }

    fn formula(
        &mut self,
        sink: &mut State<'_, '_, '_>,
        kind: Formula,
        term: Value,
    ) -> Result<Value, Error> {
        match kind {
            Formula::Index | Formula::LoopCount => return self.index(sink, term),
            Formula::Layout => return Ok(term),
            Formula::Extents | Formula::Slices => {}
        }
        if let Some(&existing) = self.formulas.get(&(kind, term)) {
            return Ok(existing);
        }
        let original = sink
            .value_to_container::<VecContainer>(term)
            .ok_or_else(|| Error::BackendError("expected a cloned shape or slice vector".into()))?
            .data
            .clone();
        let mut rewritten = Vec::with_capacity(original.len());
        for &item in &original {
            let replacement = if kind == Formula::Extents {
                self.index(sink, item)?
            } else if let Some(at) = fields(sink, "At", item)? {
                let index = self.index(sink, at[0])?;
                if at[0] == index {
                    item
                } else {
                    sink.add("At", index)?
                }
            } else if let Some(slice) = fields(sink, "Slice", item)? {
                let start = self.index(sink, slice[0])?;
                let length = self.index(sink, slice[1])?;
                let stride = self.index(sink, slice[2])?;
                if (start, length, stride) == (slice[0], slice[1], slice[2]) {
                    item
                } else {
                    sink.add("Slice", (start, length, stride))?
                }
            } else {
                return Err(Error::BackendError("expected a cloned slice dimension".into()));
            };
            rewritten.push(replacement);
        }
        let replacement = if rewritten == original {
            term
        } else {
            sink.container_to_value(VecContainer {
                data: rewritten,
                do_rebuild: true,
            })
        };
        self.formulas.insert((kind, term), replacement);
        Ok(replacement)
    }

    fn accumulator(&mut self, sink: &mut State<'_, '_, '_>, info: Value) -> Result<Value, Error> {
        let Some(args) = fields(sink, "AccumulatorInfo", info)? else {
            return Ok(info);
        };
        let shape = self.formula(sink, Formula::Extents, args[0])?;
        if shape == args[0] {
            Ok(info)
        } else {
            sink.add("AccumulatorInfo", (shape, args[1]))
        }
    }

    fn return_type(&mut self, sink: &mut State<'_, '_, '_>, info: Value) -> Result<Value, Error> {
        let Some(args) = fields(sink, "ReturnAccumulator", info)? else {
            return Ok(info);
        };
        let accumulator = self.accumulator(sink, args[0])?;
        if accumulator == args[0] {
            Ok(info)
        } else {
            sink.add("ReturnAccumulator", accumulator)
        }
    }

    fn domain(
        &mut self,
        sink: &mut State<'_, '_, '_>,
        mut domain: IterationDomain,
    ) -> Result<IterationDomain, Error> {
        domain.ordinal = domain.ordinal.map(|name| self.value(name));
        for dimension in &mut domain.dimensions {
            dimension.coordinate = dimension.coordinate.map(|name| self.value(name));
            dimension.extent = dimension.extent.map(|index| self.index(sink, index)).transpose()?;
        }
        Ok(domain)
    }

    fn operation(
        &mut self,
        sink: &mut State<'_, '_, '_>,
        context: Value,
        region: Value,
        operation: Operation,
    ) -> Result<(), Error> {
        let key = sink.add("ClonedOperation", (context, operation.source))?;
        sink.copy_occurrence(operation.source, key)?;
        self.properties(sink, key, &operation)?;
        for certificate in operation.certificates {
            sink.add_certificate(CertificateSite::Operation(key), self.value(certificate));
        }
        if let Some(domain) = operation.domain {
            let domain = sink.add("ClonedDomain", (context, domain))?;
            sink.set_operation_domain(key, domain);
        }
        if let Some(results) = operation.results {
            sink.set_parallel_results(key, results);
        }
        if !operation.policies.is_empty() {
            sink.set_accumulator_policies(key, operation.policies);
        }
        for (kind, value) in operation.formulas {
            let value = self.formula(sink, kind, value)?;
            sink.set_formula(key, kind, value)?;
        }
        for ((group, kind), mut values) in operation.collective {
            if kind.uses_raw_name() {
                for name in &mut values {
                    *name = self.value(*name);
                }
            } else {
                values = self.names(values);
            }
            sink.set_collective_inputs(key, group, kind, values)?;
        }
        for (slot, reducer) in operation.reducers {
            let reducer = sink.add("ClonedRegion", (context, reducer))?;
            sink.set_reducer(key, slot, reducer);
        }
        for (slot, combiner) in operation.combiners {
            let combiner = sink.add("ClonedRegion", (context, combiner))?;
            sink.set_combiner(key, slot, combiner);
        }
        sink.set_inputs(key, self.names(operation.inputs))?;
        sink.set_outputs(key, self.names(operation.outputs))?;
        for (slot, child, arguments) in operation.controls {
            let child = sink.add("ClonedRegion", (context, child))?;
            sink.set_control(key, slot, child);
            for (index, name) in self.names(arguments).into_iter().enumerate() {
                sink.set_argument(ArgumentSite(key, slot, index), name)?;
            }
        }
        sink.set_region(key, region);
        sink.set_order(key, operation.order);
        if let Some(target) = operation.target {
            sink.set_call_target(key, target);
        }
        Ok(())
    }

    fn properties(
        &mut self,
        sink: &mut State<'_, '_, '_>,
        target: Value,
        operation: &Operation,
    ) -> Result<(), Error> {
        let source = operation.source;
        for table in [
            "InstructionWork",
            "LiteralBits",
            "OperationWidth",
            "OperationExecution",
            "ScalarPayload",
        ] {
            if let Some(value) = sink.lookup(table, source)? {
                sink.set(table, target, value)?;
            }
        }
        for table in [
            "MaterializedCopyOperation",
            "ViewOperation",
            "ArrayLiteralOperation",
            "RangeOperation",
            "ConstantOperation",
            "ProductOperation",
            "IndexOperation",
            "IndexCalculationOperation",
            "CallOperation",
            "BranchOperation",
            "CountedLoopOperation",
            "WhileLoopOperation",
            "ParallelOperation",
            "AccumulatorOperation",
            "DiscardableOperation",
            "RepeatableOperation",
            "OrderedOperation",
            "LaneDependentOperation",
            "ScratchAllocation",
            "ProvenTotalOperation",
        ] {
            if sink.contains(table, source)? {
                sink.add(table, target)?;
            }
        }
        match operation.code {
            Code::Length { axis } => {
                sink.add("LengthOperation", (target, axis as i64))?;
            }
            Code::Project { index } => {
                sink.add("ProjectionOperation", (target, index as i64))?;
            }
            _ => {}
        }
        for slot in 0..operation.inputs.len() as i64 {
            if let Some(diet) = sink.lookup("InputDiet", (source, slot))? {
                sink.set("InputDiet", (target, slot), diet)?;
            }
            for table in ["ReadsInput", "WritesInput", "ConsumesInput"] {
                if sink.contains(table, (source, slot))? {
                    sink.add(table, (target, slot))?;
                }
            }
        }
        for slot in 0..operation.outputs.len() as i64 {
            if let Some(info) = sink.lookup("CallReturnType", (source, slot))? {
                let info = self.return_type(sink, info)?;
                sink.set("CallReturnType", (target, slot), info)?;
            }
            if sink.contains("FreshOutput", (source, slot))? {
                sink.add("FreshOutput", (target, slot))?;
            }
            for input in 0..operation.inputs.len() as i64 {
                for table in ["AliasesInput", "PreservesInputShape"] {
                    if sink.contains(table, (source, slot, input))? {
                        sink.add(table, (target, slot, input))?;
                    }
                }
            }
        }
        Ok(())
    }
}

fn fields(sink: &State<'_, '_, '_>, constructor: &str, term: Value) -> Result<Option<Vec<Value>>, Error> {
    let mut fields = None;
    let mut ambiguous = false;
    sink.enodes_for_eclass(constructor, term, |row| {
        ambiguous |= fields.is_some();
        fields = Some(row.children.to_vec());
    })?;
    if ambiguous {
        return Err(Error::BackendError(format!(
            "ambiguous {constructor} in cloned region"
        )));
    }
    Ok(fields)
}
