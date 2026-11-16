use super::{builder_error, error, Body, OptimizeError, Typed};
use crate::host::Expr;
use crate::op::BinaryOperator;
use crate::ssa::types::InstKind;
use crate::types::{self, Type, TypeExt, TypeName};
use egglog_engine::Value;

impl Body<'_, '_, '_> {
    pub(super) fn bucket_scatter(
        &mut self,
        operation: Value,
        coordinates: &[Vec<i64>],
        rank: usize,
    ) -> Result<Typed, OptimizeError> {
        let names = self
            .query()
            .sequence("Input", operation)?
            .into_iter()
            .map(|name| self.query().name(name))
            .collect::<Result<Vec<_>, _>>()?;
        let destination_name = names[0];
        let callback = self.query().required("ControlRegion", (operation, 0_i64))?;
        let mut destination = self.named(destination_name)?;
        if destination.ty.array_variant().is_some_and(types::is_array_variant_composite) {
            let source = self.snapshot(destination)?;
            destination = self.local_array(source.ty.clone())?;
            let Some(&place) = self.local_arrays.get(&destination.value) else {
                return Err(error("bucket destination place missing"));
            };
            self.builder
                .push_void_inst(InstKind::Store {
                    place,
                    value: source.value,
                })
                .map_err(builder_error)?;
            self.values.insert(destination_name, destination.clone());
        }
        let buckets = self.array_length(destination.clone())?;
        let uint = Type::Constructed(TypeName::UInt(32), vec![]);
        let counts = if self.active_dispatch.is_none() {
            let length = self.host_array_length(&destination)?;
            let bytes = Expr::Multiply(Box::new(length.clone()), Box::new(Expr::Integer(4)));
            let (_, counts) = self.device_buffer(&uint, bytes, buckets.clone())?;
            self.set_shape(&counts, vec![length])?;
            counts
        } else {
            let Some(Type::Constructed(TypeName::Size(capacity), _)) = destination.ty.array_size() else {
                return Err(error("local bucket counters require a capacity bound"));
            };
            self.local_array(types::sized_array(*capacity, uint))?
        };
        self.counted(
            Self::number(0),
            buckets.clone(),
            Self::number(1),
            vec![],
            |body, index, _| {
                body.store(counts.clone(), index, Self::number(0))?;
                Ok(vec![])
            },
        )?;
        let sources = names[1..].iter().map(|&name| self.named(name)).collect::<Result<Vec<_>, _>>()?;
        let mappings = coordinates
            .iter()
            .map(|mapping| {
                mapping
                    .iter()
                    .map(|&axis| usize::try_from(axis).map_err(|_| error("invalid bucket input axis")))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let overflow = self.bucket_axis(
            &sources,
            &mappings,
            rank,
            vec![],
            Self::number(0),
            &mut |body, coordinates, overflow| {
                let mut inputs = Vec::new();
                for (source, mapping) in sources.iter().zip(&mappings) {
                    let mut value = source.clone();
                    for &axis in mapping {
                        value = body.index(value, coordinates[axis].clone())?;
                    }
                    inputs.push(value);
                }
                let mut emission = body.lambda(callback, inputs)?;
                if emission.len() == 1 {
                    emission = body.unpack(emission.remove(0))?;
                }
                let [active, key, value]: [Typed; 3] = emission
                    .try_into()
                    .map_err(|_| error("bucket callback must return active, key, value"))?;
                body.branch(
                    active,
                    |body| {
                        let valid = body.valid_indices(destination.clone(), &[key.clone()])?;
                        body.branch(
                            valid,
                            |body| {
                                let position = body.index(counts.clone(), key.clone())?;
                                let next =
                                    body.binary(BinaryOperator::Add, position.clone(), Self::number(1))?;
                                body.store(counts.clone(), key.clone(), next)?;
                                let row = body.index(destination.clone(), key.clone())?;
                                let capacity = body.array_length(row)?;
                                let fits = body.binary(BinaryOperator::Less, position.clone(), capacity)?;
                                body.branch(
                                    fits,
                                    |body| {
                                        body.write_array(destination_name, &[key, position], value)?;
                                        Ok(overflow.clone())
                                    },
                                    |body| {
                                        body.binary(BinaryOperator::Add, overflow.clone(), Self::number(1))
                                    },
                                )
                            },
                            |body| body.binary(BinaryOperator::Add, overflow.clone(), Self::number(1)),
                        )
                    },
                    |_body| Ok(overflow.clone()),
                )
            },
        )?;
        let destination = self.snapshot(destination)?;
        let counts = self.snapshot(counts)?;
        self.tuple(vec![destination, counts, overflow])
    }

    fn bucket_axis(
        &mut self,
        sources: &[Typed],
        mappings: &[Vec<usize>],
        rank: usize,
        coordinates: Vec<Typed>,
        overflow: Typed,
        emit: &mut impl FnMut(&mut Self, &[Typed], Typed) -> Result<Typed, OptimizeError>,
    ) -> Result<Typed, OptimizeError> {
        let axis = coordinates.len();
        if axis == rank {
            return emit(self, &coordinates, overflow);
        }
        let mut extent = None;
        for (source, mapping) in sources.iter().zip(mappings) {
            if let Some(position) = mapping.iter().position(|&a| a == axis) {
                if mapping[..position].iter().all(|&a| a < axis) {
                    let mut row = source.clone();
                    for &a in &mapping[..position] {
                        row = self.index(row, coordinates[a].clone())?;
                    }
                    extent = Some(self.array_length(row)?);
                    break;
                }
            }
        }
        let Some(extent) = extent else {
            return Err(error("bucket domain extent has no input witness"));
        };
        let mut result = self.counted(
            Self::number(0),
            extent,
            Self::number(1),
            vec![overflow],
            |body, index, state| {
                let mut coordinates = coordinates.clone();
                coordinates.push(index);
                Ok(vec![body.bucket_axis(
                    sources,
                    mappings,
                    rank,
                    coordinates,
                    state[0].clone(),
                    emit,
                )?])
            },
        )?;
        result.pop().ok_or_else(|| error("bucket overflow result missing"))
    }
}
