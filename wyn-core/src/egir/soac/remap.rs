//! Structural SOAC remapping shared by resource allocation and physicalization.

use crate::LookupMap;

use super::{filter, hist, screma};
use crate::egir::ir::PlaceId;
use crate::egir::types::{GraphResource, SegExtent, SegResourceAccess, SegSpace, ValueId};

pub(crate) struct Remap<'a, R, S, E, F> {
    nodes: &'a LookupMap<ValueId, ValueId>,
    places: &'a LookupMap<PlaceId, PlaceId>,
    resource: F,
    marker: std::marker::PhantomData<fn(R) -> Result<S, E>>,
}

impl<'a, R, S, E, F> Remap<'a, R, S, E, F>
where
    R: GraphResource,
    S: GraphResource,
    F: FnMut(R) -> Result<S, E>,
{
    pub(crate) fn new(
        nodes: &'a LookupMap<ValueId, ValueId>,
        places: &'a LookupMap<PlaceId, PlaceId>,
        resource: F,
    ) -> Self {
        Self {
            nodes,
            places,
            resource,
            marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn resource(&mut self, resource: R) -> Result<S, E> {
        (self.resource)(resource)
    }

    pub(crate) fn space(&mut self, space: SegSpace<R>) -> Result<SegSpace<S>, E> {
        let dimensions = space
            .into_dims()
            .into_iter()
            .map(|extent| {
                Ok(match extent {
                    SegExtent::Fixed(value) => SegExtent::Fixed(value),
                    SegExtent::PushConstant { node, offset } => SegExtent::PushConstant {
                        node: self.nodes[&node],
                        offset,
                    },
                    SegExtent::ResourceLength {
                        mut view,
                        resource,
                        elem_bytes,
                    } => {
                        view.remap_value(|value| self.nodes[&value]);
                        SegExtent::ResourceLength {
                            view,
                            resource: self.resource(resource)?,
                            elem_bytes,
                        }
                    }
                    SegExtent::Value(node) => SegExtent::Value(self.nodes[&node]),
                })
            })
            .collect::<Result<Vec<_>, E>>()?;
        Ok(SegSpace::from_dims(dimensions).expect("remapping cannot empty a segmented space"))
    }

    pub(crate) fn lambda(&self, mut lambda: screma::Lambda) -> screma::Lambda {
        if let Some(body) = lambda.seg_body_mut() {
            for capture in body.captures_mut() {
                *capture = capture
                    .try_map(
                        |value| Ok::<_, std::convert::Infallible>(self.nodes[&value]),
                        |view| {
                            view.try_remap(|value| Ok::<_, std::convert::Infallible>(self.nodes[&value]))
                        },
                        |place| Ok::<_, std::convert::Infallible>(self.places[&place]),
                    )
                    .unwrap();
            }
        }
        lambda
    }

    pub(crate) fn screma_form(&self, mut form: screma::ScremaForm) -> screma::ScremaForm {
        form.pre = self.lambda(form.pre);
        for scan in &mut form.scans {
            scan.operator = self.lambda(scan.operator.clone());
            scan.neutral.iter_mut().for_each(|value| *value = self.nodes[value]);
        }
        for reduction in &mut form.reductions {
            reduction.operator = self.lambda(reduction.operator.clone());
            reduction.neutral.iter_mut().for_each(|value| *value = self.nodes[value]);
        }
        form.post = self.lambda(form.post);
        form
    }

    pub(crate) fn filter_body(&self, mut body: filter::Body) -> filter::Body {
        body.map = self.lambda(body.map);
        body.predicate = self.lambda(body.predicate);
        body
    }

    pub(crate) fn filter_output(&mut self, output: filter::Output<R>) -> Result<filter::Output<S>, E> {
        Ok(match output {
            filter::Output::Local { capacity, ownership } => filter::Output::Local { capacity, ownership },
            filter::Output::Runtime(runtime) => filter::Output::Runtime(filter::RuntimeOutput {
                capacity: runtime.capacity,
                backing: match runtime.backing {
                    filter::RuntimeBacking::Deferred => filter::RuntimeBacking::Deferred,
                    filter::RuntimeBacking::Bound(resource) => {
                        filter::RuntimeBacking::Bound(self.resource(resource)?)
                    }
                },
                length: match runtime.length {
                    filter::RuntimeLength::Implicit => filter::RuntimeLength::Implicit,
                    filter::RuntimeLength::Required => filter::RuntimeLength::Required,
                    filter::RuntimeLength::Stored(resource) => {
                        filter::RuntimeLength::Stored(self.resource(resource)?)
                    }
                },
            }),
        })
    }

    pub(crate) fn segment(&mut self, segment: screma::Segmented<R>) -> Result<screma::Segmented<S>, E> {
        Ok(screma::Segmented {
            space: self.space(segment.space)?,
            output_slots: segment.output_slots,
            resources: segment
                .resources
                .into_iter()
                .map(|access| {
                    Ok(SegResourceAccess {
                        resource: self.resource(access.resource)?,
                        access: access.access,
                    })
                })
                .collect::<Result<_, E>>()?,
        })
    }

    pub(crate) fn hist_form(&self, mut form: hist::HistForm) -> hist::HistForm {
        form.bucket = self.lambda(form.bucket);
        for operation in &mut form.operations {
            operation.shape.iter_mut().for_each(|value| *value = self.nodes[value]);
            operation.race_factor = self.nodes[&operation.race_factor];
            operation.destinations.iter_mut().for_each(|view| view.remap_value(|value| self.nodes[&value]));
            match &mut operation.update {
                hist::Update::Reduce { operator, neutral } => {
                    *operator = self.lambda(operator.clone());
                    neutral.iter_mut().for_each(|value| *value = self.nodes[value]);
                }
                hist::Update::BucketInsert { capacity, .. } => *capacity = self.nodes[capacity],
                hist::Update::OrderedOverwrite { .. } => {}
            }
        }
        form
    }

    pub(crate) fn bucket_storage(
        &mut self,
        storage: hist::BucketStorage<R>,
    ) -> Result<hist::BucketStorage<S>, E> {
        Ok(hist::BucketStorage {
            counts: self.resource(storage.counts)?,
            overflow: self.resource(storage.overflow)?,
        })
    }

    pub(crate) fn runtime_storage(
        &mut self,
        storage: filter::RuntimeStorage<R>,
    ) -> Result<filter::RuntimeStorage<S>, E> {
        Ok(filter::RuntimeStorage {
            data: self.resource(storage.data)?,
            length: self.resource(storage.length)?,
        })
    }

    pub(crate) fn work_buffers(
        &mut self,
        buffers: filter::WorkBuffers<R>,
    ) -> Result<filter::WorkBuffers<S>, E> {
        Ok(filter::WorkBuffers {
            flags: self.resource(buffers.flags)?,
            offsets: self.resource(buffers.offsets)?,
            block_sums: self.resource(buffers.block_sums)?,
            block_offsets: self.resource(buffers.block_offsets)?,
        })
    }
}
