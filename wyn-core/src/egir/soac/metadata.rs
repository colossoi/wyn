//! Form-owned traversal of references outside a SOAC's ordinary operands.
//!
//! Read and mutable walks use the same field definition. Phase-specific
//! iteration spaces and resource conversion remain with their phase adapters.

use super::Lambda;
use crate::egir::types::{OperandRef, PlaceId, SegBody, Soac, ValueId, ViewId, WynSoacPhase};

pub(crate) trait Visitor<'a> {
    fn lambda(&mut self, _lambda: &'a Lambda) {}
    fn value(&mut self, _value: &'a ValueId) {}
    fn view(&mut self, _view: &'a ViewId) {}
}

pub(crate) trait VisitorMut<'a> {
    fn lambda(&mut self, _lambda: &'a mut Lambda) {}
    fn value(&mut self, _value: &'a mut ValueId) {}
    fn view(&mut self, _view: &'a mut ViewId) {}
}

pub(crate) trait Metadata {
    fn visit_metadata<'a>(&'a self, visit: &mut impl Visitor<'a>);
    fn visit_metadata_mut<'a>(&'a mut self, visit: &mut impl VisitorMut<'a>);

    /// Captures precede other metadata values, retaining the dependency order
    /// used by semantic EGIR. Repeated references remain repeated.
    fn metadata_values(&self) -> Vec<ValueId> {
        let mut values = Values::default();
        self.visit_metadata(&mut values);
        values.captures.extend(values.other);
        values.captures
    }

    fn metadata_captures(&self) -> Vec<ValueId> {
        self.metadata_bodies().into_iter().flat_map(SegBody::capture_values).collect()
    }

    /// Region order is pre/scans/reductions/post for Screma, map/predicate for
    /// Filter, and bucket/reducers for Hist. Identity lambdas have no region.
    fn metadata_bodies(&self) -> Vec<&SegBody> {
        let mut bodies = Bodies(Vec::new());
        self.visit_metadata(&mut bodies);
        bodies.0
    }

    fn metadata_body_mut(&mut self, index: usize) -> Option<&mut SegBody> {
        let mut selection = BodyMut {
            remaining: index,
            body: None,
        };
        self.visit_metadata_mut(&mut selection);
        selection.body
    }

    fn remap_metadata_values(&mut self, values: impl FnMut(ValueId) -> ValueId) {
        self.remap_metadata(values, |place| place);
    }

    /// Resource-changing graph copies additionally remap captured places.
    /// Ordinary value substitution preserves the place identity instead.
    fn remap_metadata(
        &mut self,
        values: impl FnMut(ValueId) -> ValueId,
        places: impl FnMut(PlaceId) -> PlaceId,
    ) {
        self.visit_metadata_mut(&mut Remap { values, places });
    }
}

/// Match ergonomics give the same field walk shared or mutable references.
/// Each form lists its metadata once, so reads and rewrites cannot omit
/// different fields. The body uses reference destructuring, not explicit
/// borrows of fields, and never changes the form's structure.
macro_rules! impl_metadata {
    ($form:ty, |$this:ident, $visit:ident| $body:block) => {
        impl $crate::egir::soac::metadata::Metadata for $form {
            fn visit_metadata<'a>(&'a self, $visit: &mut impl $crate::egir::soac::metadata::Visitor<'a>) {
                let $this = self;
                $body
            }

            fn visit_metadata_mut<'a>(
                &'a mut self,
                $visit: &mut impl $crate::egir::soac::metadata::VisitorMut<'a>,
            ) {
                let $this = self;
                $body
            }
        }
    };
}
pub(super) use impl_metadata;

#[derive(Default)]
struct Values {
    captures: Vec<ValueId>,
    other: Vec<ValueId>,
}

impl<'a> Visitor<'a> for Values {
    fn lambda(&mut self, lambda: &'a Lambda) {
        self.captures.extend(lambda.captures().iter().filter_map(|capture| capture.value()));
    }

    fn value(&mut self, value: &'a ValueId) {
        self.other.push(*value);
    }

    fn view(&mut self, view: &'a ViewId) {
        self.other.push(view.value());
    }
}

struct Bodies<'a>(Vec<&'a SegBody>);

impl<'a> Visitor<'a> for Bodies<'a> {
    fn lambda(&mut self, lambda: &'a Lambda) {
        self.0.extend(lambda.seg_body());
    }
}

struct BodyMut<'a> {
    remaining: usize,
    body: Option<&'a mut SegBody>,
}

impl<'a> VisitorMut<'a> for BodyMut<'a> {
    fn lambda(&mut self, lambda: &'a mut Lambda) {
        if self.body.is_none() {
            if let Some(body) = lambda.seg_body_mut() {
                if self.remaining == 0 {
                    self.body = Some(body);
                } else {
                    self.remaining -= 1;
                }
            }
        }
    }
}

struct Remap<V, P> {
    values: V,
    places: P,
}

impl<'a, V, P> VisitorMut<'a> for Remap<V, P>
where
    V: FnMut(ValueId) -> ValueId,
    P: FnMut(PlaceId) -> PlaceId,
{
    fn lambda(&mut self, lambda: &'a mut Lambda) {
        if let Some(body) = lambda.seg_body_mut() {
            for capture in body.captures_mut() {
                match capture {
                    OperandRef::Value(value) => self.value(value),
                    OperandRef::View(view) => self.view(view),
                    OperandRef::Place(place) => *place = (self.places)(*place),
                }
            }
        }
    }

    fn value(&mut self, value: &'a mut ValueId) {
        *value = (self.values)(*value);
    }

    fn view(&mut self, view: &'a mut ViewId) {
        view.remap_value(&mut self.values);
    }
}

impl<P: WynSoacPhase> Metadata for Soac<P> {
    fn visit_metadata<'a>(&'a self, visit: &mut impl Visitor<'a>) {
        match self {
            Self::Screma(op) => op.form.visit_metadata(visit),
            Self::Filter(op) => op.body.visit_metadata(visit),
            Self::Hist(op) => op.form.visit_metadata(visit),
        }
    }

    fn visit_metadata_mut<'a>(&'a mut self, visit: &mut impl VisitorMut<'a>) {
        match self {
            Self::Screma(op) => op.form.visit_metadata_mut(visit),
            Self::Filter(op) => op.body.visit_metadata_mut(visit),
            Self::Hist(op) => op.form.visit_metadata_mut(visit),
        }
    }
}

#[cfg(test)]
#[path = "metadata_tests.rs"]
mod tests;
