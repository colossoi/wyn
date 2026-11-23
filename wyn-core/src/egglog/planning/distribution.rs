//! Apply an admitted domain composition to keyed cells and coordinate proofs.
use crate::egglog::cost::CostModel;
use crate::egglog::dataflow::edit;
use crate::egglog::dataflow::ir::{self, Code, Name, OperationId, Parameter, RegionId};
use crate::egglog::graph::Formula;
use crate::egglog::graph::Graph;
use crate::egglog::graph::{CertificateSite, Dimension};
use crate::egglog::query::Query;
use crate::egglog::OptimizeError;
use egglog_engine::{Value, Write};

/// Rank only the pairs that passed the structural and legality rules.
pub(super) fn select_cost(graph: &mut Graph, costs: CostModel) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut selected = Vec::new();
    query.for_each("ReadyDistribution", |row| {
        let (outer, inner) = (row[0], row[1]);
        let (Some(coordinates), Some(elements)) =
            (estimated_extent(query, outer)?, estimated_extent(query, inner)?)
        else {
            return Ok(());
        };
        let prefix = graph.region_order(graph.control_region(outer, 0_i64)?)?.len().saturating_sub(1);
        let work = graph.region_order(graph.control_region(inner, 0_i64)?)?.len();
        if costs.prefer_distribution(
            coordinates,
            prefix as u64,
            work as u64,
            elements,
            query.flag("DistributionHasFold", inner)?,
        ) {
            selected.push((outer, inner));
        }
        Ok(())
    })?;
    graph.update(|mut sink| {
        for candidate in selected {
            sink.add("DistributeNest", candidate)?;
        }
        Ok(())
    })?;
    Ok(())
}

fn estimated_extent(query: Query<'_>, operation: Value) -> Result<Option<u64>, OptimizeError> {
    let domain = query.0.iteration_domain(query.0.operation_domain(operation)?)?;
    if domain.dimensions.is_empty() {
        return Ok(None);
    }
    let mut size = Some(1_u64);
    for dimension in domain.dimensions {
        let Some(extent) = dimension.extent else {
            return Ok(None);
        };
        let count = query
            .lookup("StaticIndex", extent)?
            .map_or(1024, |value| query.0.value_to_base::<i64>(value).max(0)) as u64;
        size = if count == 0 { Some(0) } else { size.and_then(|size| size.checked_mul(count)) };
    }
    Ok(size)
}

pub(super) fn lift_row_length(graph: &mut Graph) -> Result<bool, OptimizeError> {
    let Some(candidate) = Query(graph).first("LiftUniformRowLength")? else {
        return Ok(false);
    };
    let (length, outer, array) = (candidate[0], candidate[1], candidate[2]);
    edit::set_code(graph, length, Code::Length { axis: 1 })?;
    edit::inputs(graph, length, &[array])?;
    edit::move_before(graph, length, outer)?;
    Ok(true)
}

/// A solved extent is a bound witness for the same symbolic term.
pub(super) fn prove_static_domains(graph: &mut Graph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut bounds = Vec::new();
    for &index in graph.structure().read().domains.extent_uses.keys() {
        if let Some(count) = query.lookup("StaticIndex", index)? {
            if !query.contains("ConstantBound", (index, count))? {
                bounds.push((index, count));
            }
        }
    }
    if bounds.is_empty() {
        return Ok(());
    }
    graph.update(|mut sink| {
        for bound in bounds {
            sink.add("ConstantBound", bound)?;
        }
        Ok(())
    })?;
    super::super::analysis::totality::refine(graph)
}

pub(super) fn apply(graph: &mut Graph, outer: Value, inner: Value) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let domain = query.0.operation_domain(outer)?;
    let inner_domain = query.0.operation_domain(inner)?;
    let body = query.0.control_region(outer, 0_i64)?;
    let inner_body = query.0.control_region(inner, 0_i64)?;
    let rank = query.0.domain_rank(domain)?;
    let inner_rank = query.0.domain_rank(inner_domain)?;
    let mut layout = graph.iteration_domain(domain)?;
    let flat = query.0.domain_ordinal(domain)?;
    let inner_flat = query.0.domain_ordinal(inner_domain)?;
    let mut axes = Vec::new();
    let mut inner_axes = Vec::new();
    for slot in 0..rank {
        axes.push((
            query.0.domain_axis(domain, slot)?,
            query.0.domain_extent(domain, slot)?,
        ));
    }
    for slot in 0..inner_rank {
        inner_axes.push((
            query.0.domain_axis(inner_domain, slot)?,
            query.0.domain_extent(inner_domain, slot)?,
        ));
    }
    let mut parameters = Vec::new();
    for (region, rank) in [(body, rank), (inner_body, inner_rank)] {
        for slot in 1..=rank {
            let name = query.0.parameter(region, slot)?;
            let value = query.fields("ValueNameRef", name)?[0];
            parameters.push(Parameter {
                name: Name::Value(value),
                info: query.required("ValueType", value)?,
                diet: query.required("ParameterDiet", (region, slot))?,
            });
        }
    }
    let info = query.required("ValueType", flat)?;
    let diet = query.required("ParameterDiet", (body, 0_i64))?;
    let prefix = query
        .operations(body)?
        .into_iter()
        .filter(|&operation| operation != inner)
        .map(|operation| Ok((operation, query.0.operation_order(operation)?)))
        .collect::<Result<Vec<_>, OptimizeError>>()?;
    let nested = query
        .operations(inner_body)?
        .into_iter()
        .map(|operation| Ok((operation, query.0.operation_order(operation)?)))
        .collect::<Result<Vec<_>, OptimizeError>>()?;
    let base = query.0.operation_order(inner)? + 2;
    let results = graph.parallel_results(inner)?;
    let reducers = graph.reducers(inner);
    let contributions = graph.collective_groups(inner);
    let certificates = graph.operation_certificates(inner);
    // Only identities and small immutable coordinate metadata cross this edit boundary.
    graph.update(|mut sink| {
        Ok((|| -> Result<(), OptimizeError> {
            sink.set_parallel_results(outer, results);
            sink.set_reducers(outer, reducers);
            sink.clear_collective_inputs(outer);
            for ((clause, kind), values) in contributions {
                sink.set_collective_inputs(outer, clause, kind, values)?;
            }
            let ordinal = sink.add("DerivedValue", (inner, 40_i64, 0_i64))?;
            sink.set("ValueType", ordinal, info)?;
            layout.ordinal = Some(ordinal);
            layout.dimensions.extend(inner_axes.iter().map(|&(axis, extent)| Dimension::new(axis, extent)));
            sink.set_domain(domain, layout);
            for slot in 0..=rank {
                sink.remove("ParameterDiet", (body, slot))?;
            }
            sink.set_parameters(body, Vec::new())?;
            parameters.insert(
                0,
                Parameter {
                    name: Name::Value(ordinal),
                    info,
                    diet,
                },
            );
            ir::parameters(&mut sink, RegionId(body), &parameters)?;
            for (slot, (name, axes)) in [(flat, &axes), (inner_flat, &inner_axes)].into_iter().enumerate() {
                let operation = sink.add("DerivedOperation", (inner, 40_i64, slot as i64))?;
                let operation = ir::operation_at(&mut sink, OperationId(operation), Code::IndexValue, &[])?;
                ir::outputs(&mut sink, operation, &[Name::Value(name)])?;
                let mut index = sink.add("IndexConstant", 0_i64)?;
                for &(axis, extent) in axes {
                    let scaled = sink.add("IndexMul", (index, extent))?;
                    let coordinate = sink.add("IndexValue", axis)?;
                    index = sink.add("IndexAdd", (scaled, coordinate))?;
                }
                sink.set_formula(operation.0, Formula::Index, index)?;
                sink.set_region(operation.0, body);
                sink.set_order(operation.0, slot as i64);
            }
            for (offset, operations) in [(2_i64, prefix), (base, nested)] {
                for (operation, order) in operations {
                    let order = order + offset;
                    sink.set_region(operation, body);
                    sink.set_order(operation, order);
                }
            }
            for value in certificates {
                sink.add_certificate(CertificateSite::Operation(outer), value);
            }
            Ok(())
        })())
    })??;
    edit::remove(graph, &[inner])
}
