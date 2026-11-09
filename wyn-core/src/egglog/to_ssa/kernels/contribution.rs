//! Guarded argument substitution shared by folds, packing, and indexed updates.
use super::super::{error, Body, OptimizeError, Typed};
use super::element;
use crate::types::Type;
use crate::LookupMap;
use egglog_engine::Value;

type EmitContribution<'a, T> =
    dyn FnMut(&mut Body<'_, '_, '_>, &mut LookupMap<Value, Typed>, T) -> Result<Typed, OptimizeError> + 'a;

/// A selection guards the entire continuation, including later arguments.
pub(super) struct Contribution<'a> {
    pub(super) scope: Value,
    pub(super) plan: Value,
    pub(super) indices: &'a [Typed],
    pub(super) skipped: &'a dyn Fn(&mut Body<'_, '_, '_>, &Type) -> Result<Typed, OptimizeError>,
}

impl Contribution<'_> {
    pub(super) fn emit(
        &self,
        body: &mut Body<'_, '_, '_>,
        expression: Value,
        cache: &mut LookupMap<Value, Typed>,
        emit: &mut EmitContribution<'_, Typed>,
    ) -> Result<Typed, OptimizeError> {
        if let Some(fields) = body.compiler.facts.enode("ElementAt", expression) {
            let axes = body.compiler.facts.vector(fields[1])?;
            let coordinates = axes
                .iter()
                .map(|&axis| {
                    let axis = usize::try_from(body.compiler.facts.integer(axis))
                        .map_err(|_| error("invalid domain axis"))?;
                    let Some(coordinate) = self.indices.get(axis) else {
                        return Err(error("input axis exceeds its domain rank"));
                    };
                    Ok((axis, coordinate.clone()))
                })
                .collect::<Result<Vec<_>, OptimizeError>>()?;
            // A cache entry names an element at one coordinate map, not a row
            // or an element reached through a different permutation of axes.
            let canonical = coordinates.len() == self.indices.len()
                && coordinates.iter().enumerate().all(|(i, (axis, _))| i == *axis);
            let mut other_axes = LookupMap::default();
            let coordinates = coordinates.into_iter().map(|(_, value)| value).collect::<Vec<_>>();
            let value = element(
                body,
                self.scope,
                self.plan,
                fields[0],
                &coordinates,
                if canonical { cache } else { &mut other_axes },
            )?;
            return emit(body, cache, value);
        }
        if let Some(fields) = body.compiler.facts.enode("ApplyElement", expression) {
            if let Some(value) = cache.get(&fields[0]).cloned() {
                return emit(body, cache, value);
            }
            let arguments = body.compiler.facts.vector(fields[2])?;
            return self.arguments(
                body,
                &arguments,
                cache,
                Vec::new(),
                &mut |body, cache, arguments| {
                    let value = body.apply_region(self.scope, fields[1], arguments)?;
                    cache.insert(fields[0], value.clone());
                    emit(body, cache, value)
                },
            );
        }
        if let Some(fields) = body.compiler.facts.enode("TupleElements", expression) {
            let arguments = body.compiler.facts.vector(fields[0])?;
            return self.arguments(body, &arguments, cache, vec![], &mut |body, cache, values| {
                let tuple = body.tuple(values)?;
                emit(body, cache, tuple)
            });
        }
        if let Some(fields) = body.compiler.facts.enode("SelectElement", expression) {
            return self.emit(body, fields[0], cache, &mut |body, cache, keep| {
                body.branch_with_type(
                    Some(self.scope),
                    keep,
                    |body| self.emit(body, fields[1], &mut cache.clone(), emit),
                    |body, ty| (self.skipped)(body, ty),
                    None,
                )
            });
        }
        Err(error("invalid parallel contribution"))
    }

    fn arguments(
        &self,
        body: &mut Body<'_, '_, '_>,
        arguments: &[Value],
        cache: &mut LookupMap<Value, Typed>,
        values: Vec<Typed>,
        emit: &mut EmitContribution<'_, Vec<Typed>>,
    ) -> Result<Typed, OptimizeError> {
        let Some((&argument, rest)) = arguments.split_first() else {
            return emit(body, cache, values);
        };
        self.emit(body, argument, cache, &mut |body, cache, value| {
            let mut values = values.clone();
            values.push(value);
            self.arguments(body, rest, cache, values, emit)
        })
    }
}
