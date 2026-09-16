use super::super::analysis::{counts, inputs};
use super::{element, flatten, input, wire_input, Wiring};
use crate::egglog::data::{AssociatedData, BucketShapeData, OperationId, OperationKind};
use crate::types;

pub(super) fn envelope(
    data: &mut AssociatedData,
    producer: OperationId,
    consumer: OperationId,
) -> Option<()> {
    let OperationKind::Screma {
        form: a, inputs: ai, ..
    } = data.operations[producer].kind.clone()
    else {
        return None;
    };
    if counts(&a) != (0, 0) {
        return None;
    }
    let body = match &data.operations[consumer].kind {
        OperationKind::Filter { map, .. } | OperationKind::ReduceByIndex { map, .. } => map.clone(),
        OperationKind::Scatter { body, .. } => body.clone(),
        OperationKind::BucketScatter { body, shape, .. } if data.bucket_shapes[*shape].domain_rank == 1 => {
            body.clone()
        }
        _ => return None,
    };
    let region = data.operations[consumer].region;
    let trees_a: Vec<_> = ai.iter().map(|a| input(data, a, None)).collect();
    let trees_b: Vec<_> = inputs(&data.operations[consumer].kind)
        .into_iter()
        .map(|a| input(data, a, Some(producer)))
        .collect();
    let mut arrays = vec![];
    for t in trees_a.iter().chain(&trees_b) {
        flatten(t, &mut arrays);
    }
    let parameters = arrays.iter().map(|a| element(data, a)).collect::<Option<Vec<_>>>()?;
    let mut w = Wiring::new(parameters);
    let args = trees_a
        .iter()
        .map(|t| wire_input(data, region, &mut w, t, &arrays, &[]))
        .collect::<Option<Vec<_>>>()?;
    let av = w.call(a.pre, args);
    let produced = w.call(a.post, av);
    let args = trees_b
        .iter()
        .map(|t| wire_input(data, region, &mut w, t, &arrays, &produced))
        .collect::<Option<Vec<_>>>()?;
    let values = w.call(body, args);
    let body = w.finish(values);
    match &mut data.operations[consumer].kind {
        OperationKind::Filter {
            map,
            inputs,
            ownership,
            ..
        } => {
            *map = body;
            *inputs = arrays;
            *ownership = types::SoacOwnership::Fresh;
        }
        OperationKind::ReduceByIndex { map, inputs, .. } => {
            *map = body;
            *inputs = arrays;
        }
        OperationKind::Scatter { body: b, inputs, .. } => {
            *b = body;
            *inputs = arrays;
        }
        OperationKind::BucketScatter {
            body: b,
            inputs,
            shape,
            ..
        } => {
            *b = body;
            *inputs = arrays;
            let shape_data = BucketShapeData {
                domain_rank: 1,
                input_dimensions: vec![vec![0]; inputs.len()],
            };
            *shape = data.bucket_shapes.alloc(shape_data);
        }
        _ => return None,
    }
    data.regions[data.operations[producer].region].members.remove(&producer);
    Some(())
}
