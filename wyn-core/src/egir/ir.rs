//! Phase-agnostic, low-level data structures for EGIR.

use slotmap::{new_key_type, SlotMap};
use smallvec::SmallVec;

use crate::ast::Span;
use crate::flow::{BlockId, ControlHeader, ExecutionModel};
use crate::interface::{EntryInput as InterfaceEntryInput, EntryOutput as InterfaceEntryOutput};
use crate::op::OpTag;
use crate::ssa::types::AtomicOp;
use crate::types::ExternDecl;
use crate::{LookupMap, LookupSet, SortedSet};

pub use crate::op::PureViewSource;
pub use crate::types::SoacOwnership;

/// Effect token for ordering effectful ops during EGIR passes.
///
/// These are purely an EGIR-internal concept — they never reach the SSA
/// backend. `elaborate` emits instructions in skeleton block order and
/// doesn't pass the tokens through. The token chain only exists to support
/// rewriting passes (e.g. `soac_expand` allocating fresh tokens for new
/// Load/Store side-effects so they don't collide with existing ones).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectToken(u32);

impl From<u32> for EffectToken {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl std::fmt::Display for EffectToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "!{}", self.0)
    }
}

/// Splice a producer and consumer's effect-token chains around a fused operation.
pub fn splice_effect_tokens(
    producer: Option<(EffectToken, EffectToken)>,
    consumer: Option<(EffectToken, EffectToken)>,
) -> Option<(EffectToken, EffectToken)> {
    match (producer, consumer) {
        (Some((input, _)), Some((_, output))) => Some((input, output)),
        (Some(effects), None) | (None, Some(effects)) => Some(effects),
        (None, None) => None,
    }
}

new_key_type! {
    /// Identity of a value node in the e-graph. Every pure node, union node,
    /// value parameter, and constant gets one.
    pub struct ValueId;
    /// Identity of an addressable place. Places cannot enter value e-classes
    /// or cross CFG edges as block arguments.
    pub struct PlaceId;
    /// Identity of one complete, explicitly bound region call.
    pub struct CallSiteId;
}

/// Index into the single physical parameter sequence owned by a callable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterId(usize);

/// Index into the by-value portion of a callable's physical results.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReturnSlotId(usize);

/// A value that has been admitted to CFG-carried state. Its constructor is
/// the boundary at which materialized aggregates are rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FlowValueId(ValueId);

/// A value-sized view handle whose addressability must survive type and
/// resource representation changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ViewId(ValueId);

/// The complete operand vocabulary at function, call, capture, and result
/// boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OperandRef {
    Value(ValueId),
    View(ViewId),
    Place(PlaceId),
}

/// Storage ownership is phase-parametric so resource erasure changes only
/// the resource payload and cannot turn a place into a value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PlaceRegion<R> {
    Function,
    Workgroup,
    Parametric,
    Resource(R),
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlaceAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

impl PlaceAccess {
    pub const fn accepts(self, provided: Self) -> bool {
        matches!(
            (self, provided),
            (Self::ReadOnly, Self::ReadOnly | Self::ReadWrite)
                | (Self::WriteOnly, Self::WriteOnly | Self::ReadWrite)
                | (Self::ReadWrite, Self::ReadWrite)
        )
    }
}

/// The pointee and capabilities intrinsically owned by an addressable place.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlaceType<R, Ty> {
    pub pointee: Ty,
    pub region: PlaceRegion<R>,
    pub access: PlaceAccess,
}

/// Physical type of an addressable view passed in the value channel.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ViewType<R, Ty> {
    pub array: Ty,
    pub region: PlaceRegion<R>,
    pub access: PlaceAccess,
}

/// One physical parameter representation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OperandType<R, Ty> {
    Value(Ty),
    View(ViewType<R, Ty>),
    Place(PlaceType<R, Ty>),
}

/// A destination with fixed extent or with an explicit length sink for a
/// bounded, variable-sized result.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PlaceDestination<P> {
    Fixed(P),
    Bounded {
        storage: P,
        length: P,
    },
}

/// Every indivisible result is routed either through a by-value return slot or
/// through one or more addressable destination parameters.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResultDestination<R, P> {
    ReturnValue(R),
    Place(PlaceDestination<P>),
}

/// A typed logical result tree paired at every leaf with its physical route.
/// Product nodes carry structure only; they are not runtime aggregates.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResultTree<Ty, R, P> {
    root: ResultNode<Ty, R, P>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ResultNode<Ty, R, P> {
    Product {
        ty: Ty,
        fields: Box<[ResultNode<Ty, R, P>]>,
    },
    Destination {
        ty: Ty,
        destination: ResultDestination<R, P>,
    },
}

fn result_node_contains_return_value<Ty>(node: &ResultNode<Ty, ValueId, PlaceId>, value: ValueId) -> bool {
    match node {
        ResultNode::Product { fields, .. } => {
            fields.iter().any(|field| result_node_contains_return_value(field, value))
        }
        ResultNode::Destination {
            destination: ResultDestination::ReturnValue(candidate),
            ..
        } => *candidate == value,
        ResultNode::Destination {
            destination: ResultDestination::Place(_),
            ..
        } => false,
    }
}

pub type FunctionResult<Ty> = ResultTree<Ty, ReturnSlotId, ParameterId>;
pub type ResultBinding<Ty> = ResultTree<Ty, ValueId, PlaceId>;

/// Observable behavior of a callable after destination selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallEffects {
    Pure,
    DestinationWrite,
    General,
}

/// One parameter in the callable's complete physical argument sequence.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FuncParam<R, Ty> {
    name: String,
    representation: OperandType<R, Ty>,
}

/// One fully applied call. Arguments align exactly with the callee's physical
/// parameter sequence; results mirror its typed result tree with concrete
/// value and place bindings.
#[derive(Clone, Debug)]
pub struct CallSite<Ty> {
    callee: RegionId,
    arguments: Box<[OperandRef]>,
    result: ResultBinding<Ty>,
    effects: CallEffects,
}

impl ParameterId {
    pub(crate) const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

impl ReturnSlotId {
    pub(crate) const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

impl FlowValueId {
    pub const fn value(self) -> ValueId {
        self.0
    }

    pub(crate) fn try_remap<E>(self, map: impl FnOnce(ValueId) -> Result<ValueId, E>) -> Result<Self, E> {
        Ok(Self(map(self.0)?))
    }
}

impl ViewId {
    pub const fn value(self) -> ValueId {
        self.0
    }

    #[cfg(test)]
    pub(crate) const fn test(value: ValueId) -> Self {
        Self(value)
    }

    pub(crate) fn try_remap<E>(self, map: impl FnOnce(ValueId) -> Result<ValueId, E>) -> Result<Self, E> {
        Ok(Self(map(self.0)?))
    }

    pub(crate) fn remap_value(&mut self, map: impl FnOnce(ValueId) -> ValueId) {
        self.0 = map(self.0);
    }
}

impl From<ViewId> for OperandRef {
    fn from(view: ViewId) -> Self {
        Self::View(view)
    }
}

impl From<ValueId> for OperandRef {
    fn from(value: ValueId) -> Self {
        Self::Value(value)
    }
}

impl OperandRef {
    pub const fn value(self) -> Option<ValueId> {
        match self {
            Self::Value(value) => Some(value),
            Self::View(view) => Some(view.0),
            Self::Place(_) => None,
        }
    }

    pub const fn place(self) -> Option<PlaceId> {
        match self {
            Self::Place(place) => Some(place),
            Self::Value(_) | Self::View(_) => None,
        }
    }

    pub fn try_map<E>(
        self,
        mut map_value: impl FnMut(ValueId) -> Result<ValueId, E>,
        mut map_view: impl FnMut(ViewId) -> Result<ViewId, E>,
        mut map_place: impl FnMut(PlaceId) -> Result<PlaceId, E>,
    ) -> Result<Self, E> {
        Ok(match self {
            Self::Value(value) => Self::Value(map_value(value)?),
            Self::View(view) => Self::View(map_view(view)?),
            Self::Place(place) => Self::Place(map_place(place)?),
        })
    }

    pub fn remap_value(&mut self, mut map: impl FnMut(ValueId) -> ValueId) {
        *self = match *self {
            Self::Value(value) => Self::Value(map(value)),
            Self::View(view) => Self::View(ViewId(map(view.value()))),
            Self::Place(place) => Self::Place(place),
        };
    }
}

impl<R, Ty> OperandType<R, Ty> {
    pub fn ty(&self) -> &Ty {
        match self {
            Self::Value(ty) => ty,
            Self::View(view) => &view.array,
            Self::Place(place) => &place.pointee,
        }
    }

    pub fn ty_mut(&mut self) -> &mut Ty {
        match self {
            Self::Value(ty) => ty,
            Self::View(view) => &mut view.array,
            Self::Place(place) => &mut place.pointee,
        }
    }

    pub fn try_map<S, U, E>(
        self,
        map_resource: &mut impl FnMut(R) -> Result<S, E>,
        map_ty: &mut impl FnMut(Ty) -> Result<U, E>,
    ) -> Result<OperandType<S, U>, E> {
        fn region<R, S, E>(
            region: PlaceRegion<R>,
            map_resource: &mut impl FnMut(R) -> Result<S, E>,
        ) -> Result<PlaceRegion<S>, E> {
            Ok(match region {
                PlaceRegion::Function => PlaceRegion::Function,
                PlaceRegion::Workgroup => PlaceRegion::Workgroup,
                PlaceRegion::Parametric => PlaceRegion::Parametric,
                PlaceRegion::Resource(resource) => PlaceRegion::Resource(map_resource(resource)?),
                PlaceRegion::Output => PlaceRegion::Output,
            })
        }

        Ok(match self {
            Self::Value(ty) => OperandType::Value(map_ty(ty)?),
            Self::View(view) => OperandType::View(ViewType {
                array: map_ty(view.array)?,
                region: region(view.region, map_resource)?,
                access: view.access,
            }),
            Self::Place(place) => OperandType::Place(PlaceType {
                pointee: map_ty(place.pointee)?,
                region: region(place.region, map_resource)?,
                access: place.access,
            }),
        })
    }

    pub fn map<S, U>(
        self,
        mut map_resource: impl FnMut(R) -> S,
        mut map_ty: impl FnMut(Ty) -> U,
    ) -> OperandType<S, U> {
        self.try_map(
            &mut |resource| Ok::<_, std::convert::Infallible>(map_resource(resource)),
            &mut |ty| Ok::<_, std::convert::Infallible>(map_ty(ty)),
        )
        .unwrap()
    }
}

impl<Ty, R, P> ResultTree<Ty, R, P> {
    pub fn destination(ty: Ty, destination: ResultDestination<R, P>) -> Self {
        Self {
            root: ResultNode::Destination { ty, destination },
        }
    }

    pub fn product(ty: Ty, fields: impl IntoIterator<Item = Self>) -> Self {
        Self {
            root: ResultNode::Product {
                ty,
                fields: fields.into_iter().map(|field| field.root).collect(),
            },
        }
    }

    pub fn ty(&self) -> &Ty {
        match &self.root {
            ResultNode::Product { ty, .. } | ResultNode::Destination { ty, .. } => ty,
        }
    }

    pub fn field_count(&self) -> usize {
        match &self.root {
            ResultNode::Product { fields, .. } => fields.len(),
            ResultNode::Destination { .. } => 1,
        }
    }

    pub fn is_product(&self) -> bool {
        matches!(self.root, ResultNode::Product { .. })
    }

    pub fn field(&self, index: usize) -> Option<Self>
    where
        Ty: Clone,
        R: Clone,
        P: Clone,
    {
        match &self.root {
            ResultNode::Product { fields, .. } => fields.get(index).cloned().map(|root| Self { root }),
            ResultNode::Destination { .. } if index == 0 => Some(self.clone()),
            ResultNode::Destination { .. } => None,
        }
    }

    /// Borrow-independent views of the top-level logical result fields. A
    /// scalar destination is its own sole field.
    pub fn top_level_fields(&self) -> Vec<Self>
    where
        Ty: Clone,
        R: Clone,
        P: Clone,
    {
        (0..self.field_count()).filter_map(|field| self.field(field)).collect()
    }

    pub fn top_level_field_index(&self, field: &Self) -> Option<usize>
    where
        Ty: PartialEq,
        R: PartialEq,
        P: PartialEq,
    {
        match &self.root {
            ResultNode::Product { fields, .. } => {
                fields.iter().position(|candidate| candidate == &field.root)
            }
            ResultNode::Destination { .. } => (self == field).then_some(0),
        }
    }

    /// Borrow-independent views of the physical destination leaves, in ABI
    /// order. Product structure is retained by `top_level_fields`; this view
    /// is for consumers that operate once per physical result channel.
    pub fn destination_leaves(&self) -> Vec<Self>
    where
        Ty: Clone,
        R: Clone,
        P: Clone,
    {
        self.destination_leaves_with_paths().into_iter().map(|(_, leaf)| leaf).collect()
    }

    /// Physical destination leaves paired with their logical product path.
    pub fn destination_leaves_with_paths(&self) -> Vec<(Box<[usize]>, Self)>
    where
        Ty: Clone,
        R: Clone,
        P: Clone,
    {
        fn walk<Ty: Clone, R: Clone, P: Clone>(
            node: &ResultNode<Ty, R, P>,
            path: &mut Vec<usize>,
            leaves: &mut Vec<(Box<[usize]>, ResultTree<Ty, R, P>)>,
        ) {
            match node {
                ResultNode::Product { fields, .. } => {
                    for (index, field) in fields.iter().enumerate() {
                        path.push(index);
                        walk(field, path, leaves);
                        path.pop();
                    }
                }
                ResultNode::Destination { .. } => {
                    leaves.push((path.clone().into_boxed_slice(), ResultTree { root: node.clone() }))
                }
            }
        }

        let mut leaves = Vec::with_capacity(self.destination_count());
        walk(&self.root, &mut Vec::new(), &mut leaves);
        leaves
    }

    pub fn destination_count(&self) -> usize {
        let mut count = 0;
        self.for_each_destination(|_, _| count += 1);
        count
    }

    pub fn single_destination(&self) -> Option<(&Ty, &ResultDestination<R, P>)> {
        match &self.root {
            ResultNode::Destination { ty, destination } => Some((ty, destination)),
            ResultNode::Product { .. } => None,
        }
    }

    pub fn for_each_destination(&self, mut visit: impl FnMut(&Ty, &ResultDestination<R, P>)) {
        fn walk<Ty, R, P>(
            node: &ResultNode<Ty, R, P>,
            visit: &mut impl FnMut(&Ty, &ResultDestination<R, P>),
        ) {
            match node {
                ResultNode::Product { fields, .. } => {
                    for field in fields {
                        walk(field, visit);
                    }
                }
                ResultNode::Destination { ty, destination } => visit(ty, destination),
            }
        }

        walk(&self.root, &mut visit);
    }

    pub fn for_each_destination_mut(
        &mut self,
        mut visit: impl FnMut(&mut Ty, &mut ResultDestination<R, P>),
    ) {
        fn walk<Ty, R, P>(
            node: &mut ResultNode<Ty, R, P>,
            visit: &mut impl FnMut(&mut Ty, &mut ResultDestination<R, P>),
        ) {
            match node {
                ResultNode::Product { ty: _, fields } => {
                    for field in fields {
                        walk(field, visit);
                    }
                }
                ResultNode::Destination { ty, destination } => visit(ty, destination),
            }
        }

        walk(&mut self.root, &mut visit);
    }

    pub fn for_each_type_mut(&mut self, mut visit: impl FnMut(&mut Ty)) {
        fn walk<Ty, R, P>(node: &mut ResultNode<Ty, R, P>, visit: &mut impl FnMut(&mut Ty)) {
            match node {
                ResultNode::Product { ty, fields } => {
                    visit(ty);
                    for field in fields {
                        walk(field, visit);
                    }
                }
                ResultNode::Destination { ty, .. } => visit(ty),
            }
        }

        walk(&mut self.root, &mut visit);
    }

    pub fn map_destinations<R2, P2>(
        &self,
        mut map: impl FnMut(&Ty, &ResultDestination<R, P>) -> ResultDestination<R2, P2>,
    ) -> ResultTree<Ty, R2, P2>
    where
        Ty: Clone,
    {
        fn walk<Ty: Clone, R, P, R2, P2>(
            node: &ResultNode<Ty, R, P>,
            map: &mut impl FnMut(&Ty, &ResultDestination<R, P>) -> ResultDestination<R2, P2>,
        ) -> ResultNode<Ty, R2, P2> {
            match node {
                ResultNode::Product { ty, fields } => ResultNode::Product {
                    ty: ty.clone(),
                    fields: fields.iter().map(|field| walk(field, map)).collect(),
                },
                ResultNode::Destination { ty, destination } => ResultNode::Destination {
                    ty: ty.clone(),
                    destination: map(ty, destination),
                },
            }
        }

        ResultTree {
            root: walk(&self.root, &mut map),
        }
    }

    pub fn try_map<Ty2, R2, P2, E>(
        self,
        map_ty: &mut impl FnMut(Ty) -> Result<Ty2, E>,
        map_return: &mut impl FnMut(R) -> Result<R2, E>,
        map_place: &mut impl FnMut(P) -> Result<P2, E>,
    ) -> Result<ResultTree<Ty2, R2, P2>, E> {
        fn walk<Ty, R, P, Ty2, R2, P2, E>(
            node: ResultNode<Ty, R, P>,
            map_ty: &mut impl FnMut(Ty) -> Result<Ty2, E>,
            map_return: &mut impl FnMut(R) -> Result<R2, E>,
            map_place: &mut impl FnMut(P) -> Result<P2, E>,
        ) -> Result<ResultNode<Ty2, R2, P2>, E> {
            Ok(match node {
                ResultNode::Product { ty, fields } => ResultNode::Product {
                    ty: map_ty(ty)?,
                    fields: fields
                        .into_vec()
                        .into_iter()
                        .map(|field| walk(field, map_ty, map_return, map_place))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice(),
                },
                ResultNode::Destination { ty, destination } => ResultNode::Destination {
                    ty: map_ty(ty)?,
                    destination: match destination {
                        ResultDestination::ReturnValue(value) => {
                            ResultDestination::ReturnValue(map_return(value)?)
                        }
                        ResultDestination::Place(PlaceDestination::Fixed(place)) => {
                            ResultDestination::Place(PlaceDestination::Fixed(map_place(place)?))
                        }
                        ResultDestination::Place(PlaceDestination::Bounded { storage, length }) => {
                            ResultDestination::Place(PlaceDestination::Bounded {
                                storage: map_place(storage)?,
                                length: map_place(length)?,
                            })
                        }
                    },
                },
            })
        }

        Ok(ResultTree {
            root: walk(self.root, map_ty, map_return, map_place)?,
        })
    }

    pub fn map<Ty2, R2, P2>(
        self,
        mut map_ty: impl FnMut(Ty) -> Ty2,
        mut map_return: impl FnMut(R) -> R2,
        mut map_place: impl FnMut(P) -> P2,
    ) -> ResultTree<Ty2, R2, P2> {
        self.try_map(
            &mut |ty| Ok::<_, std::convert::Infallible>(map_ty(ty)),
            &mut |result| Ok::<_, std::convert::Infallible>(map_return(result)),
            &mut |place| Ok::<_, std::convert::Infallible>(map_place(place)),
        )
        .unwrap()
    }
}

impl<Ty> ResultBinding<Ty> {
    pub fn values(&self) -> Vec<ValueId> {
        let mut values = Vec::new();
        self.for_each_destination(|_, destination| {
            if let ResultDestination::ReturnValue(value) = destination {
                values.push(*value);
            }
        });
        values
    }

    pub fn contains_value(&self, value: ValueId) -> bool {
        result_node_contains_return_value(&self.root, value)
    }

    pub fn top_level_field_containing_value(&self, value: ValueId) -> Option<usize> {
        match &self.root {
            ResultNode::Product { fields, .. } => {
                fields.iter().position(|field| result_node_contains_return_value(field, value))
            }
            ResultNode::Destination { .. } => self.contains_value(value).then_some(0),
        }
    }

    pub fn single_value(&self) -> Option<ValueId> {
        let values = self.values();
        let [value] = values.as_slice() else {
            return None;
        };
        Some(*value)
    }

    pub fn places(&self) -> Vec<PlaceId> {
        let mut places = Vec::new();
        self.for_each_destination(|_, destination| match destination {
            ResultDestination::ReturnValue(_) => {}
            ResultDestination::Place(PlaceDestination::Fixed(place)) => places.push(*place),
            ResultDestination::Place(PlaceDestination::Bounded { storage, length }) => {
                places.push(*storage);
                places.push(*length);
            }
        });
        places
    }

    pub fn for_each_place(&self, mut visit: impl FnMut(PlaceId)) {
        for place in self.places() {
            visit(place);
        }
    }

    pub fn replace_value(&mut self, old: ValueId, new: ValueId) {
        fn walk<Ty>(node: &mut ResultNode<Ty, ValueId, PlaceId>, old: ValueId, new: ValueId) {
            match node {
                ResultNode::Product { fields, .. } => {
                    for field in fields {
                        walk(field, old, new);
                    }
                }
                ResultNode::Destination {
                    destination: ResultDestination::ReturnValue(value),
                    ..
                } if *value == old => *value = new,
                ResultNode::Destination { .. } => {}
            }
        }

        walk(&mut self.root, old, new);
    }

    pub fn replace_value_with_place(&mut self, old: ValueId, place: PlaceId) {
        fn walk<Ty>(node: &mut ResultNode<Ty, ValueId, PlaceId>, old: ValueId, place: PlaceId) {
            match node {
                ResultNode::Product { fields, .. } => {
                    for field in fields {
                        walk(field, old, place);
                    }
                }
                ResultNode::Destination { destination, .. } => {
                    if let ResultDestination::ReturnValue(value) = destination {
                        if *value == old {
                            *destination = ResultDestination::Place(PlaceDestination::Fixed(place));
                        }
                    }
                }
            }
        }

        walk(&mut self.root, old, place);
    }

    pub fn replace_place(&mut self, old: PlaceId, new: PlaceId) {
        self.for_each_place_mut(|place| {
            if *place == old {
                *place = new;
            }
        });
    }

    pub fn for_each_value_mut(&mut self, mut visit: impl FnMut(&mut ValueId)) {
        self.for_each_destination_mut(|_, destination| {
            if let ResultDestination::ReturnValue(value) = destination {
                visit(value);
            }
        });
    }

    pub fn for_each_place_mut(&mut self, mut visit: impl FnMut(&mut PlaceId)) {
        self.for_each_destination_mut(|_, destination| match destination {
            ResultDestination::ReturnValue(_) => {}
            ResultDestination::Place(PlaceDestination::Fixed(place)) => visit(place),
            ResultDestination::Place(PlaceDestination::Bounded { storage, length }) => {
                visit(storage);
                visit(length);
            }
        });
    }
}

impl<Ty> FunctionResult<Ty> {
    /// Destination parameters in physical result order. Bounded results
    /// contribute both their storage and length parameters.
    pub fn destination_parameters(&self) -> Vec<ParameterId> {
        let mut parameters = Vec::new();
        self.for_each_destination(|_, destination| match destination {
            ResultDestination::ReturnValue(_) => {}
            ResultDestination::Place(PlaceDestination::Fixed(parameter)) => parameters.push(*parameter),
            ResultDestination::Place(PlaceDestination::Bounded { storage, length }) => {
                parameters.push(*storage);
                parameters.push(*length);
            }
        });
        parameters
    }
}

impl<Ty: Clone> FunctionResult<Ty> {
    pub fn bind(
        &self,
        mut bind_return: impl FnMut(ReturnSlotId, &Ty) -> ValueId,
        mut bind_place: impl FnMut(ParameterId) -> PlaceId,
    ) -> ResultBinding<Ty> {
        fn walk<Ty: Clone>(
            node: &ResultNode<Ty, ReturnSlotId, ParameterId>,
            bind_return: &mut impl FnMut(ReturnSlotId, &Ty) -> ValueId,
            bind_place: &mut impl FnMut(ParameterId) -> PlaceId,
        ) -> ResultNode<Ty, ValueId, PlaceId> {
            match node {
                ResultNode::Product { ty, fields } => ResultNode::Product {
                    ty: ty.clone(),
                    fields: fields.iter().map(|field| walk(field, bind_return, bind_place)).collect(),
                },
                ResultNode::Destination {
                    ty,
                    destination: ResultDestination::ReturnValue(slot),
                } => ResultNode::Destination {
                    ty: ty.clone(),
                    destination: ResultDestination::ReturnValue(bind_return(*slot, ty)),
                },
                ResultNode::Destination {
                    ty,
                    destination: ResultDestination::Place(PlaceDestination::Fixed(parameter)),
                } => ResultNode::Destination {
                    ty: ty.clone(),
                    destination: ResultDestination::Place(PlaceDestination::Fixed(bind_place(*parameter))),
                },
                ResultNode::Destination {
                    ty,
                    destination: ResultDestination::Place(PlaceDestination::Bounded { storage, length }),
                } => ResultNode::Destination {
                    ty: ty.clone(),
                    destination: ResultDestination::Place(PlaceDestination::Bounded {
                        storage: bind_place(*storage),
                        length: bind_place(*length),
                    }),
                },
            }
        }

        ResultTree {
            root: walk(&self.root, &mut bind_return, &mut bind_place),
        }
    }
}

impl<R, Ty> FuncParam<R, Ty> {
    pub fn value(name: String, ty: Ty) -> Self {
        Self {
            name,
            representation: OperandType::Value(ty),
        }
    }

    pub fn view(name: String, ty: ViewType<R, Ty>) -> Self {
        Self {
            name,
            representation: OperandType::View(ty),
        }
    }

    pub fn place(name: String, ty: PlaceType<R, Ty>) -> Self {
        Self {
            name,
            representation: OperandType::Place(ty),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn representation(&self) -> &OperandType<R, Ty> {
        &self.representation
    }

    pub fn representation_mut(&mut self) -> &mut OperandType<R, Ty> {
        &mut self.representation
    }

    pub fn ty(&self) -> &Ty {
        self.representation.ty()
    }

    pub fn try_map<S, U, E>(
        self,
        map_resource: &mut impl FnMut(R) -> Result<S, E>,
        map_ty: &mut impl FnMut(Ty) -> Result<U, E>,
    ) -> Result<FuncParam<S, U>, E> {
        Ok(FuncParam {
            name: self.name,
            representation: self.representation.try_map(map_resource, map_ty)?,
        })
    }

    pub fn map<S, U>(
        self,
        mut map_resource: impl FnMut(R) -> S,
        mut map_ty: impl FnMut(Ty) -> U,
    ) -> FuncParam<S, U> {
        self.try_map(
            &mut |resource| Ok::<_, std::convert::Infallible>(map_resource(resource)),
            &mut |ty| Ok::<_, std::convert::Infallible>(map_ty(ty)),
        )
        .unwrap()
    }
}

pub fn callable_parameter<R, Lang: Language>(name: String, ty: Lang::Ty) -> FuncParam<R, Lang::Ty> {
    if Lang::is_view(&ty) {
        FuncParam::view(
            name,
            ViewType {
                array: ty,
                region: PlaceRegion::Parametric,
                access: PlaceAccess::ReadOnly,
            },
        )
    } else {
        FuncParam::value(name, ty)
    }
}

impl<Ty> CallSite<Ty> {
    fn new(
        callee: RegionId,
        arguments: Box<[OperandRef]>,
        result: ResultBinding<Ty>,
        effects: CallEffects,
    ) -> Self {
        Self {
            callee,
            arguments,
            result,
            effects,
        }
    }

    pub fn callee(&self) -> RegionId {
        self.callee
    }

    pub fn arguments(&self) -> &[OperandRef] {
        &self.arguments
    }

    pub(crate) fn arguments_mut(&mut self) -> &mut [OperandRef] {
        &mut self.arguments
    }

    pub(crate) fn replace_boundary(
        &mut self,
        arguments: Vec<OperandRef>,
        result: ResultBinding<Ty>,
        effects: CallEffects,
    ) {
        self.arguments = arguments.into_boxed_slice();
        self.result = result;
        self.effects = effects;
    }

    pub fn result(&self) -> &ResultBinding<Ty> {
        &self.result
    }

    pub fn effects(&self) -> CallEffects {
        self.effects
    }

    pub(crate) fn try_remap_bindings<E>(
        self,
        mut map_value: impl FnMut(ValueId) -> Result<ValueId, E>,
        mut map_place: impl FnMut(PlaceId) -> Result<PlaceId, E>,
    ) -> Result<Self, E> {
        let arguments = self
            .arguments
            .into_vec()
            .into_iter()
            .map(|argument| match argument {
                OperandRef::Value(value) => map_value(value).map(OperandRef::Value),
                OperandRef::View(view) => {
                    map_value(view.value()).map(|value| OperandRef::View(ViewId(value)))
                }
                OperandRef::Place(place) => map_place(place).map(OperandRef::Place),
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        let result = self.result.try_map(&mut |ty| Ok(ty), &mut map_value, &mut map_place)?;
        Ok(Self {
            callee: self.callee,
            arguments,
            result,
            effects: self.effects,
        })
    }

    pub(crate) fn retain_arguments(&mut self, retain: &[bool]) -> Result<(), String> {
        if self.arguments.len() != retain.len() {
            return Err(format!(
                "call to {:?} has {} arguments for a {}-parameter rewrite",
                self.callee,
                self.arguments.len(),
                retain.len()
            ));
        }
        self.arguments = self
            .arguments
            .iter()
            .copied()
            .zip(retain)
            .filter_map(|(argument, retain)| (*retain).then_some(argument))
            .collect();
        Ok(())
    }
}

/// A callable used as a structured SOAC region.
///
/// Regions and ordinary functions deliberately share one identity realm:
/// every call target is a [`crate::FunctionId`]. The alias preserves the
/// domain-specific vocabulary used by segmented operators without opening a
/// second allocator.
pub type RegionId = crate::FunctionId;

/// Program-owned identity arenas with names as one-way metadata.
///
/// There is intentionally no name-to-ID lookup. All resolution happens before
/// these arenas are built; their strings exist only for diagnostics, emitted
/// symbols, and host-facing entry metadata.
pub type RegionArena = crate::IdArena<RegionId, String>;
pub type GlobalArena = crate::IdArena<crate::GlobalId, String>;
pub type EntryArena = crate::IdArena<crate::EntryId, String>;

// ---------------------------------------------------------------------------
// PureOp — operator identity for hash-consing
// ---------------------------------------------------------------------------

/// The type and literal payloads stored by a core IR graph.
pub trait Language: Clone + std::fmt::Debug + Eq + std::hash::Hash {
    type Const: Clone + std::fmt::Debug + Eq + std::hash::Hash;
    type Ty: Clone + std::fmt::Debug + Eq + std::hash::Hash;

    fn is_materialized_aggregate(ty: &Self::Ty) -> bool;
    fn is_view(ty: &Self::Ty) -> bool;
    fn product_fields(ty: &Self::Ty) -> Option<&[Self::Ty]>;

    fn view_argument_matches(parameter: &Self::Ty, argument: &Self::Ty) -> bool {
        parameter == argument
    }
}

pub fn by_value_function_result<Lang: Language>(ty: Lang::Ty) -> FunctionResult<Lang::Ty> {
    fn build<Lang: Language>(
        ty: Lang::Ty,
        next_slot: &mut usize,
    ) -> ResultNode<Lang::Ty, ReturnSlotId, ParameterId> {
        if let Some(fields) = Lang::product_fields(&ty) {
            let fields = fields.to_vec();
            ResultNode::Product {
                ty,
                fields: fields.into_iter().map(|field| build::<Lang>(field, next_slot)).collect(),
            }
        } else {
            let slot = ReturnSlotId::new(*next_slot);
            *next_slot += 1;
            ResultNode::Destination {
                ty,
                destination: ResultDestination::ReturnValue(slot),
            }
        }
    }

    let mut next_slot = 0;
    ResultTree {
        root: build::<Lang>(ty, &mut next_slot),
    }
}

pub fn destination_passing_function_result<R, Lang: Language>(
    ty: Lang::Ty,
    parameters: &mut Vec<FuncParam<R, Lang::Ty>>,
) -> FunctionResult<Lang::Ty> {
    fn build<R, Lang: Language>(
        ty: Lang::Ty,
        parameters: &mut Vec<FuncParam<R, Lang::Ty>>,
        next_slot: &mut usize,
        next_destination: &mut usize,
    ) -> ResultNode<Lang::Ty, ReturnSlotId, ParameterId> {
        if let Some(fields) = Lang::product_fields(&ty) {
            let fields = fields.to_vec();
            return ResultNode::Product {
                ty,
                fields: fields
                    .into_iter()
                    .map(|field| build::<R, Lang>(field, parameters, next_slot, next_destination))
                    .collect(),
            };
        }
        let destination = if Lang::is_materialized_aggregate(&ty) {
            let parameter = ParameterId::new(parameters.len());
            parameters.push(FuncParam::place(
                format!("result_destination_{}", *next_destination),
                PlaceType {
                    pointee: ty.clone(),
                    region: PlaceRegion::Parametric,
                    access: PlaceAccess::ReadWrite,
                },
            ));
            *next_destination += 1;
            ResultDestination::Place(PlaceDestination::Fixed(parameter))
        } else {
            let slot = ReturnSlotId::new(*next_slot);
            *next_slot += 1;
            ResultDestination::ReturnValue(slot)
        };
        ResultNode::Destination { ty, destination }
    }

    let mut next_slot = 0;
    let mut next_destination = 0;
    ResultTree {
        root: build::<R, Lang>(ty, parameters, &mut next_slot, &mut next_destination),
    }
}

/// EGIR pure operators cannot encode calls. Calls exist only as complete
/// [`CallSite`] records referenced from the effect skeleton.
pub type PureOp<R> = OpTag<R, std::convert::Infallible>;

// ---------------------------------------------------------------------------
// PureValueKey — hash-cons key = operator + operands + result type
// ---------------------------------------------------------------------------

/// The full identity of a pure node for hash-consing: operator, operands
/// (already-canonical `ValueId`s), and result type. `ty` is part of the
/// key because two otherwise-equal pure ops with different result types
/// are semantically different values — e.g.
/// `_w_intrinsic_storage_len(0, 0)` can be retyped at a rewrite site
/// from its registered `u32` to a caller-required `i32`, and collapsing
/// those two interns into one node would silently let the first-inserted
/// type win at the merged site. The type distinction applies uniformly to
/// literals and every other pure operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PureValueKey<R, Lang: Language> {
    pub op: PureOp<R>,
    pub operands: SmallVec<[ValueId; 4]>,
    pub ty: Lang::Ty,
}

/// Address-producing operations are stored outside the value e-graph.
#[derive(Clone, Debug)]
pub enum PlaceOp {
    Parameter {
        parameter: ParameterId,
    },
    View {
        view: ViewId,
    },
    AllocaResult,
    Index {
        base: PlaceId,
        index: ValueId,
    },
    Slice {
        base: PlaceId,
        start: ValueId,
        length: ValueId,
    },
    ViewIndex {
        view: ViewId,
        index: ValueId,
    },
    OutputSlot {
        index: usize,
    },
}

/// One addressable identity with its representation-independent pointee type.
#[derive(Clone, Debug)]
pub struct Place<R, Ty> {
    op: PlaceOp,
    ty: PlaceType<R, Ty>,
    span: Option<Span>,
}

impl<R, Ty> Place<R, Ty> {
    pub fn op(&self) -> &PlaceOp {
        &self.op
    }

    pub fn ty(&self) -> &PlaceType<R, Ty> {
        &self.ty
    }

    pub fn span(&self) -> Option<Span> {
        self.span
    }

    pub(crate) fn remap_parameter(&mut self, map: impl FnOnce(ParameterId) -> ParameterId) {
        if let PlaceOp::Parameter { parameter } = &mut self.op {
            *parameter = map(*parameter);
        }
    }

    pub(crate) fn remap_value(&mut self, mut map: impl FnMut(ValueId) -> ValueId) {
        match &mut self.op {
            PlaceOp::View { view } => view.remap_value(&mut map),
            PlaceOp::Index { index, .. } => *index = map(*index),
            PlaceOp::Slice { start, length, .. } => {
                *start = map(*start);
                *length = map(*length);
            }
            PlaceOp::ViewIndex { view, index } => {
                view.remap_value(&mut map);
                *index = map(*index);
            }
            PlaceOp::Parameter { .. } | PlaceOp::AllocaResult | PlaceOp::OutputSlot { .. } => {}
        }
    }

    pub(crate) fn remap_place(&mut self, mut map: impl FnMut(PlaceId) -> PlaceId) {
        match &mut self.op {
            PlaceOp::Index { base, .. } | PlaceOp::Slice { base, .. } => *base = map(*base),
            PlaceOp::Parameter { .. }
            | PlaceOp::View { .. }
            | PlaceOp::AllocaResult
            | PlaceOp::ViewIndex { .. }
            | PlaceOp::OutputSlot { .. } => {}
        }
    }

    pub(crate) fn try_map<S, E>(
        self,
        mut map_resource: impl FnMut(R) -> Result<S, E>,
        mut map_value: impl FnMut(ValueId) -> Result<ValueId, E>,
        mut map_place: impl FnMut(PlaceId) -> Result<PlaceId, E>,
    ) -> Result<Place<S, Ty>, E> {
        let op = match self.op {
            PlaceOp::Parameter { parameter } => PlaceOp::Parameter { parameter },
            PlaceOp::View { view } => PlaceOp::View {
                view: ViewId(map_value(view.value())?),
            },
            PlaceOp::AllocaResult => PlaceOp::AllocaResult,
            PlaceOp::Index { base, index } => PlaceOp::Index {
                base: map_place(base)?,
                index: map_value(index)?,
            },
            PlaceOp::Slice { base, start, length } => PlaceOp::Slice {
                base: map_place(base)?,
                start: map_value(start)?,
                length: map_value(length)?,
            },
            PlaceOp::ViewIndex { view, index } => PlaceOp::ViewIndex {
                view: ViewId(map_value(view.value())?),
                index: map_value(index)?,
            },
            PlaceOp::OutputSlot { index } => PlaceOp::OutputSlot { index },
        };
        let region = match self.ty.region {
            PlaceRegion::Function => PlaceRegion::Function,
            PlaceRegion::Workgroup => PlaceRegion::Workgroup,
            PlaceRegion::Parametric => PlaceRegion::Parametric,
            PlaceRegion::Resource(resource) => PlaceRegion::Resource(map_resource(resource)?),
            PlaceRegion::Output => PlaceRegion::Output,
        };
        Ok(Place {
            op,
            ty: PlaceType {
                pointee: self.ty.pointee,
                region,
                access: self.ty.access,
            },
            span: self.span,
        })
    }
}

// ---------------------------------------------------------------------------
// ValueKind — what lives in the sea of nodes
// ---------------------------------------------------------------------------

/// A node in the e-graph.
#[derive(Clone, Debug)]
pub enum ValueKind<R, Lang: Language> {
    /// A pure instruction, hash-consed and floating.
    Pure {
        op: PureOp<R>,
        operands: SmallVec<[ValueId; 4]>,
    },
    /// Union of two equivalent representations (binary tree of eclasses).
    Union {
        left: ValueId,
        right: ValueId,
    },
    /// Function parameter.
    FuncParam {
        parameter: ParameterId,
    },
    /// Block parameter (merge point in CFG skeleton).
    BlockParam {
        block: BlockId,
        index: usize,
    },
    /// One by-value result channel of a complete call site.
    CallResult {
        call: CallSiteId,
        slot: ReturnSlotId,
    },
    /// Runtime length associated with an addressable view or bounded place.
    PlaceLength {
        place: PlaceId,
    },
    /// A value-sized addressable handle whose storage is an EGIR place.
    PlaceView {
        place: PlaceId,
    },
    /// Inline constant value.
    Constant(Lang::Const),
    /// Side-effect result — a value produced by an effectful instruction
    /// in the skeleton. Not hash-consed; each is unique.
    SideEffectResult,
}

impl<R, Lang: Language> ValueKind<R, Lang> {
    /// Return all child ValueNodeIds referenced by this node.
    pub fn children(&self) -> SmallVec<[ValueId; 4]> {
        match self {
            ValueKind::Pure { operands, .. } => operands.clone(),
            ValueKind::Union { left, right } => smallvec::smallvec![*left, *right],
            ValueKind::FuncParam { .. }
            | ValueKind::BlockParam { .. }
            | ValueKind::CallResult { .. }
            | ValueKind::PlaceLength { .. }
            | ValueKind::PlaceView { .. }
            | ValueKind::Constant(_)
            | ValueKind::SideEffectResult => SmallVec::new(),
        }
    }
}

/// One graph node together with all metadata intrinsically owned by that
/// identity.
#[derive(Clone, Debug)]
pub struct Value<R, Lang: Language> {
    pub(crate) kind: ValueKind<R, Lang>,
    pub(crate) ty: Lang::Ty,
    /// First source span attached to this hash-consed value.
    pub(crate) span: Option<Span>,
    /// Canonical replacement selected by CFG simplification, if any.
    pub(crate) alias: Option<ValueId>,
    /// Canonical result fields explicitly packed into this value.
    pub(crate) result_origins: Vec<ResultBinding<Lang::Ty>>,
}

impl<R, Lang: Language> Value<R, Lang> {
    /// Return the graph dependencies referenced by this node.
    pub fn children(&self) -> SmallVec<[ValueId; 4]> {
        self.kind.children()
    }

    pub fn kind(&self) -> &ValueKind<R, Lang> {
        &self.kind
    }

    pub fn ty(&self) -> &Lang::Ty {
        &self.ty
    }

    pub fn span(&self) -> Option<Span> {
        self.span
    }

    pub fn alias(&self) -> Option<ValueId> {
        self.alias
    }

    pub fn result_origins(&self) -> &[ResultBinding<Lang::Ty>] {
        &self.result_origins
    }
}

// ---------------------------------------------------------------------------
// Skeleton — the CFG of side-effectful instructions
// ---------------------------------------------------------------------------

/// A side effect anchored in the skeleton CFG.
#[derive(Clone, Debug)]
pub struct SideEffect<P: Family, Lang: Language> {
    pub(crate) kind: SideEffectKind<P>,
    /// Canonical value operands for non-call effects. Complete call operands
    /// live exclusively on the referenced [`CallSite`].
    pub(crate) operands: SmallVec<[OperandRef; 4]>,
    /// Structured result of this effect. Calls bind their result on the call
    /// boundary and therefore do not duplicate it here.
    pub(crate) result: Option<ResultBinding<Lang::Ty>>,
    /// Effect token chain.
    pub(crate) effects: Option<(EffectToken, EffectToken)>,
    /// Source span of the user expression that produced this side-effect,
    /// or `None` for synthesized side-effects (e.g. SOAC expansion).
    pub(crate) span: Option<Span>,
}

impl<P: Family, Lang: Language> SideEffect<P, Lang> {
    pub fn new(
        kind: SideEffectKind<P>,
        operands: SmallVec<[OperandRef; 4]>,
        result: Option<ResultBinding<Lang::Ty>>,
        effects: Option<(EffectToken, EffectToken)>,
        span: Option<Span>,
    ) -> Self {
        Self {
            kind,
            operands,
            result,
            effects,
            span,
        }
    }

    pub fn kind(&self) -> &SideEffectKind<P> {
        &self.kind
    }

    pub fn kind_mut(&mut self) -> &mut SideEffectKind<P> {
        &mut self.kind
    }

    pub fn operands(&self) -> &[OperandRef] {
        &self.operands
    }

    pub fn operands_mut(&mut self) -> &mut SmallVec<[OperandRef; 4]> {
        &mut self.operands
    }

    pub fn operand_values(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.operands.iter().filter_map(|operand| operand.value())
    }

    pub fn result(&self) -> Option<&ResultBinding<Lang::Ty>> {
        self.result.as_ref()
    }

    pub fn result_mut(&mut self) -> Option<&mut ResultBinding<Lang::Ty>> {
        self.result.as_mut()
    }

    pub fn value_result(&self) -> Option<ValueId> {
        self.result.as_ref()?.single_value()
    }

    pub fn result_values(&self) -> Vec<ValueId> {
        self.result.as_ref().map(ResultBinding::values).unwrap_or_default()
    }

    pub fn effects(&self) -> Option<(EffectToken, EffectToken)> {
        self.effects
    }

    pub fn effects_mut(&mut self) -> &mut Option<(EffectToken, EffectToken)> {
        &mut self.effects
    }

    pub fn span(&self) -> Option<Span> {
        self.span
    }
}

/// EGIR-native effect operation. Value and place operands are represented by
/// the enclosing side effect's `ValueId` operands; SSA identities are
/// introduced only when the graph is elaborated.
#[derive(Clone, Debug)]
pub enum EffectOp<R> {
    Call {
        site: CallSiteId,
    },
    Op {
        tag: PureOp<R>,
    },
    Alloca {
        result: PlaceId,
    },
    Load {
        place: PlaceId,
    },
    Store {
        place: PlaceId,
    },
    Atomic {
        place: PlaceId,
        op: AtomicOp,
    },
    ControlBarrier,
}

impl<R> EffectOp<R> {
    /// Resource identity carried directly by the operation tag, if any.
    pub fn referenced_resource(&self) -> Option<&R> {
        match self {
            Self::Op { tag } => tag.referenced_resource(),
            Self::Call { .. }
            | Self::Alloca { .. }
            | Self::Load { .. }
            | Self::Store { .. }
            | Self::Atomic { .. }
            | Self::ControlBarrier => None,
        }
    }

    pub fn try_map_resource<S, E>(self, map: &mut impl FnMut(R) -> Result<S, E>) -> Result<EffectOp<S>, E> {
        Ok(match self {
            Self::Call { site } => EffectOp::Call { site },
            Self::Op { tag } => EffectOp::Op {
                tag: tag.try_map_resource(map)?,
            },
            Self::Alloca { result } => EffectOp::Alloca { result },
            Self::Load { place } => EffectOp::Load { place },
            Self::Store { place } => EffectOp::Store { place },
            Self::Atomic { place, op } => EffectOp::Atomic { place, op },
            Self::ControlBarrier => EffectOp::ControlBarrier,
        })
    }

    pub fn try_map<S, E>(
        self,
        mut map_resource: impl FnMut(R) -> Result<S, E>,
        mut map_call: impl FnMut(CallSiteId) -> Result<CallSiteId, E>,
        mut map_place: impl FnMut(PlaceId) -> Result<PlaceId, E>,
    ) -> Result<EffectOp<S>, E> {
        Ok(match self {
            Self::Call { site } => EffectOp::Call {
                site: map_call(site)?,
            },
            Self::Op { tag } => EffectOp::Op {
                tag: tag.try_map_resource(&mut map_resource)?,
            },
            Self::Alloca { result } => EffectOp::Alloca {
                result: map_place(result)?,
            },
            Self::Load { place } => EffectOp::Load {
                place: map_place(place)?,
            },
            Self::Store { place } => EffectOp::Store {
                place: map_place(place)?,
            },
            Self::Atomic { place, op } => EffectOp::Atomic {
                place: map_place(place)?,
                op,
            },
            Self::ControlBarrier => EffectOp::ControlBarrier,
        })
    }
}

/// A skeleton side effect's concrete kind.
#[derive(Clone, Debug)]
pub enum SideEffectKind<P: Family> {
    Effect(EffectOp<P::Resource>),
    /// A placeholder for an unexpanded SOAC. Produced by `from_tlc` and
    /// consumed by `soac_expand`. Never reaches elaborate.
    Soac(P::Soac),
}

/// One concrete dimension of a segmented iteration space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SegExtent<R> {
    Fixed(u32),
    PushConstant {
        node: ValueId,
        offset: u32,
    },
    ResourceLength {
        view: ViewId,
        resource: R,
        elem_bytes: u32,
    },
    /// A concrete EGIR value whose provenance is not host-dispatchable. Such
    /// spaces remain valid for lane-local/serial lowering.
    Value(ValueId),
}

/// The parallel iteration space of a `Seg` op. Dimensions retain their
/// logical shape until scheduling; lowerings may flatten them or map up to
/// three dimensions directly onto a target invocation grid.
#[derive(Clone, Debug)]
pub struct SegSpace<R> {
    dims: Vec<SegExtent<R>>,
}

impl<R> SegSpace<R> {
    #[cfg(test)]
    pub(crate) fn new(extent: SegExtent<R>) -> Self {
        Self { dims: vec![extent] }
    }

    pub(crate) fn from_dims(dims: Vec<SegExtent<R>>) -> Option<Self> {
        (!dims.is_empty()).then_some(Self { dims })
    }

    pub(crate) fn dims(&self) -> &[SegExtent<R>] {
        &self.dims
    }

    pub(crate) fn into_dims(self) -> Vec<SegExtent<R>> {
        self.dims
    }

    pub(crate) fn referenced_nodes(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.dims.iter().filter_map(|extent| match extent {
            SegExtent::PushConstant { node, .. } | SegExtent::Value(node) => Some(*node),
            SegExtent::ResourceLength { view, .. } => Some(view.value()),
            SegExtent::Fixed(_) => None,
        })
    }

    pub(crate) fn referenced_node_slots(&mut self) -> Vec<&mut ValueId> {
        self.dims
            .iter_mut()
            .filter_map(|extent| match extent {
                SegExtent::PushConstant { node, .. } | SegExtent::Value(node) => Some(node),
                SegExtent::ResourceLength { view, .. } => Some(&mut view.0),
                SegExtent::Fixed(_) => None,
            })
            .collect()
    }
}

impl<R: Copy> SegSpace<R> {
    /// Rewrite one referenced graph value and, for a resource-length extent,
    /// keep its resource identity synchronized with the replacement view.
    pub(crate) fn replace_reference(&mut self, old: ValueId, new: ValueId, resource: R) {
        for extent in &mut self.dims {
            match extent {
                SegExtent::PushConstant { node, .. } | SegExtent::Value(node) if *node == old => {
                    *node = new;
                }
                SegExtent::ResourceLength {
                    view,
                    resource: extent_resource,
                    ..
                } if view.value() == old => {
                    *view = ViewId(new);
                    *extent_resource = resource;
                }
                _ => {}
            }
        }
    }

    /// Retarget a one-dimensional dynamic domain to the length of a concrete
    /// resource view. Returns false when the current space is not the supported
    /// one-dimensional dynamic shape.
    pub(crate) fn retarget_single_resource_length(
        &mut self,
        view: ViewId,
        resource: R,
        elem_bytes: u32,
    ) -> bool {
        let [extent] = self.dims.as_mut_slice() else {
            return false;
        };
        if !matches!(extent, SegExtent::Value(_) | SegExtent::ResourceLength { .. }) {
            return false;
        }
        *extent = SegExtent::ResourceLength {
            view,
            resource,
            elem_bytes,
        };
        true
    }
}

/// A complete callable body and the values captured from its surrounding
/// graph. Captures are explicit values, never an operand-count convention.
#[derive(Clone, Debug)]
pub struct SegBody {
    pub(crate) region: RegionId,
    /// Captures use the same representation-typed argument vocabulary as an
    /// ordinary complete call.
    pub(crate) captures: Vec<OperandRef>,
}

impl SegBody {
    pub fn new(region: RegionId, captures: Vec<OperandRef>) -> Self {
        Self { region, captures }
    }

    pub fn region(&self) -> RegionId {
        self.region
    }

    pub fn captures(&self) -> &[OperandRef] {
        &self.captures
    }

    pub fn captures_mut(&mut self) -> &mut Vec<OperandRef> {
        &mut self.captures
    }

    pub fn capture_values(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.captures.iter().filter_map(|capture| capture.value())
    }

    pub fn remap_capture_values(&mut self, mut map: impl FnMut(ValueId) -> ValueId) {
        for capture in &mut self.captures {
            capture.remap_value(&mut map);
        }
    }

    /// Number of non-capture parameters in this body's region ABI.
    ///
    /// Segment bodies bind their lane/element parameters first and append one
    /// function parameter for each capture.
    pub(crate) fn leading_parameter_count<P: Family, Lang: Language>(
        &self,
        function: &Func<P, Lang>,
    ) -> Result<usize, String> {
        function.params.len().checked_sub(self.captures.len()).ok_or_else(|| {
            format!(
                "region `{}` has {} parameters but {} captures",
                function.name,
                function.params.len(),
                self.captures.len()
            )
        })
    }

    /// Map this body's capture parameters to their enclosing graph values.
    pub(crate) fn capture_bindings<P: Family, Lang: Language>(
        &self,
        function: &Func<P, Lang>,
    ) -> Result<LookupMap<ValueId, OperandRef>, String> {
        let leading = self.leading_parameter_count(function)?;
        let mut bindings = LookupMap::new();
        for (node, definition) in &function.graph.nodes {
            let ValueKind::FuncParam { parameter } = &definition.kind else {
                continue;
            };
            let index = parameter.index();
            if index < leading || index >= function.params.len() {
                continue;
            }
            let capture = self.captures.get(index - leading).copied().ok_or_else(|| {
                format!(
                    "region `{}` has out-of-range capture parameter {index}",
                    function.name
                )
            })?;
            bindings.insert(node, capture);
        }
        Ok(bindings)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegResourceAccess<R> {
    pub resource: R,
    pub access: crate::ResourceAccess,
}

impl<R: Copy + Ord> SegResourceAccess<R> {
    pub fn merge(a: &[Self], b: &[Self]) -> Vec<Self> {
        let mut merged: std::collections::BTreeMap<R, crate::ResourceAccess> =
            std::collections::BTreeMap::new();
        for resource in a.iter().chain(b) {
            merged
                .entry(resource.resource)
                .and_modify(|access| *access = access.merge(resource.access))
                .or_insert(resource.access);
        }
        merged.into_iter().map(|(resource, access)| Self { resource, access }).collect()
    }
}

/// Phase-varying data embedded recursively in EGIR nodes.
///
/// Every associated type corresponds to an actual field stored in the graph.
/// Proof-only typestates and program-wide context stay at [`Program`].
pub trait Family: Clone + std::fmt::Debug {
    type Resource: GraphResource;
    type Soac: Clone + std::fmt::Debug;

    fn remap_soac_values(soac: &mut Self::Soac, map: &mut dyn FnMut(ValueId) -> ValueId);
}

/// The complete collection of types physically stored by an EGIR program.
///
/// [`ProgramFamily`] makes this collection explicit at each top-level
/// typestate alias. Descendants receive only the individual types they store.
pub trait ProgramShape {
    type Family: Family;
    type ResourceDecl: Clone + std::fmt::Debug;
    type OutputRoute: Clone + std::fmt::Debug;
    type ProgramData: std::fmt::Debug;
}

/// A transparent description of one EGIR tree representation.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProgramFamily<GraphFamily, ResourceDecl, OutputRoute, ProgramData>(
    std::marker::PhantomData<fn() -> (GraphFamily, ResourceDecl, OutputRoute, ProgramData)>,
);

impl<GraphFamily, ResourceDecl, OutputRoute, ProgramData> ProgramShape
    for ProgramFamily<GraphFamily, ResourceDecl, OutputRoute, ProgramData>
where
    GraphFamily: Family,
    ResourceDecl: Clone + std::fmt::Debug,
    OutputRoute: Clone + std::fmt::Debug,
    ProgramData: std::fmt::Debug,
{
    type Family = GraphFamily;
    type ResourceDecl = ResourceDecl;
    type OutputRoute = OutputRoute;
    type ProgramData = ProgramData;
}

/// Physical representation selected for one SOAC input. Logical rank and
/// coordinate mappings are intentionally separate from this choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArrayLayout {
    /// An ordinary composite array value. Tuple elements are stored AoS.
    Composite,
    /// A storage view whose tuple element is loaded as one AoS record.
    StorageAos,
    /// A tuple of component arrays.
    StructureOfArrays,
    /// A virtual or fused generator with no materialized element storage.
    Generated,
    /// Field views into interleaved records. No current frontend creates this
    /// form, but representing its byte geometry prevents future SoA rewrites
    /// from erasing the backing AoS stride.
    StridedFields {
        element_stride_bytes: u32,
        field_offsets_bytes: Vec<u32>,
    },
}

#[derive(Clone, Debug)]
pub struct SoacInputType<Ty> {
    pub array: Ty,
    /// Logical consumer dimensions used by each regular array axis. `[0]` is
    /// an ordinary one-dimensional input; `[0, 1]` is a rank-two input;
    /// `[1]` is a one-dimensional generator varying along the second axis.
    pub dimensions: Vec<u8>,
    pub layout: ArrayLayout,
}

impl<Ty> SoacInputType<Ty> {
    pub(crate) fn array(array: Ty) -> Self {
        Self {
            array,
            dimensions: vec![0],
            layout: ArrayLayout::Composite,
        }
    }

    pub(crate) fn mapped(array: Ty, dimensions: Vec<u8>) -> Self {
        assert!(!dimensions.is_empty(), "SOAC input dimensions must be non-empty");
        let mut unique = dimensions.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            dimensions.len(),
            "SOAC input dimensions must be unique"
        );
        Self {
            array,
            dimensions,
            layout: ArrayLayout::Composite,
        }
    }

    pub(crate) fn with_layout(mut self, layout: ArrayLayout) -> Self {
        self.layout = layout;
        self
    }

    pub(crate) fn rank(&self) -> u8 {
        u8::try_from(self.dimensions.len()).expect("SOAC input rank exceeds u8")
    }
}

/// EGIR conditions are values, CFG arguments are admitted flow values, and a
/// return carries the complete function-result binding.
pub type SkeletonTerminator<Lang> =
    crate::flow::Terminator<ValueId, FlowValueId, ResultBinding<<Lang as Language>::Ty>>;

impl<Ty> crate::flow::Terminator<ValueId, FlowValueId, ResultBinding<Ty>> {
    pub fn referenced_nodes(&self) -> SmallVec<[ValueId; 8]> {
        match self {
            Self::Return(result) => result.iter().flat_map(ResultBinding::values).collect(),
            Self::Branch { args, .. } => args.iter().map(|value| value.value()).collect(),
            Self::CondBranch {
                cond,
                then_args,
                else_args,
                ..
            } => std::iter::once(*cond)
                .chain(then_args.iter().map(|value| value.value()))
                .chain(else_args.iter().map(|value| value.value()))
                .collect(),
            Self::Unreachable => SmallVec::new(),
        }
    }

    pub fn for_each_value(&self, mut visit: impl FnMut(ValueId)) {
        for value in self.referenced_nodes() {
            visit(value);
        }
    }

    pub fn visit_values_mut(&mut self, mut visit: impl FnMut(&mut ValueId)) {
        match self {
            Self::Return(result) => {
                if let Some(result) = result {
                    result.for_each_value_mut(visit);
                }
            }
            Self::Branch { args, .. } => {
                for argument in args {
                    visit(&mut argument.0);
                }
            }
            Self::CondBranch {
                cond,
                then_args,
                else_args,
                ..
            } => {
                visit(cond);
                for argument in then_args {
                    visit(&mut argument.0);
                }
                for argument in else_args {
                    visit(&mut argument.0);
                }
            }
            Self::Unreachable => {}
        }
    }
}

/// A block in the skeleton CFG.
#[derive(Clone, Debug)]
pub struct SkeletonBlock<P: Family, Lang: Language> {
    /// Materialized aggregates have no representation in this sequence.
    pub params: Vec<FlowValueId>,
    /// Explicitly located instructions, in order. Pure calls are anchored here
    /// even though they do not participate in effect ordering.
    pub side_effects: Vec<SideEffect<P, Lang>>,
    /// Block terminator.
    pub term: SkeletonTerminator<Lang>,
    /// Structured-control metadata intrinsically owned by this block.
    pub control_header: Option<ControlHeader>,
}

impl<P: Family, Lang: Language> SkeletonBlock<P, Lang> {
    pub fn new() -> Self {
        SkeletonBlock {
            params: Vec::new(),
            side_effects: Vec::new(),
            term: crate::flow::Terminator::Unreachable,
            control_header: None,
        }
    }
}

/// The skeleton CFG (blocks + explicitly located instructions).
#[derive(Clone, Debug)]
pub struct Skeleton<P: Family, Lang: Language> {
    pub entry: BlockId,
    pub blocks: SlotMap<BlockId, SkeletonBlock<P, Lang>>,
}

impl<P: Family, Lang: Language> Skeleton<P, Lang> {
    pub fn new() -> Self {
        let mut blocks = SlotMap::with_key();
        let entry = blocks.insert(SkeletonBlock::new());
        Skeleton { entry, blocks }
    }

    pub fn create_block(&mut self) -> BlockId {
        self.blocks.insert(SkeletonBlock::new())
    }

    pub fn effect(&self, site: SideEffectSite) -> &SideEffect<P, Lang> {
        &self.blocks[site.block].side_effects[site.index]
    }

    pub fn effect_mut(&mut self, site: SideEffectSite) -> &mut SideEffect<P, Lang> {
        &mut self.blocks[site.block].side_effects[site.index]
    }

    pub fn get_effect(&self, site: SideEffectSite) -> Option<&SideEffect<P, Lang>> {
        self.blocks.get(site.block)?.side_effects.get(site.index)
    }

    pub fn get_effect_mut(&mut self, site: SideEffectSite) -> Option<&mut SideEffect<P, Lang>> {
        self.blocks.get_mut(site.block)?.side_effects.get_mut(site.index)
    }

    /// Remove a snapshot-stable set of side-effect sites without invalidating
    /// any site before it has been consumed. Duplicate sites are harmless.
    pub fn remove_effect_sites(&mut self, effects: impl IntoIterator<Item = SideEffectSite>) {
        let mut by_block = std::collections::HashMap::<BlockId, Vec<usize>>::new();
        for site in effects {
            by_block.entry(site.block).or_default().push(site.index);
        }
        for (block, mut indices) in by_block {
            indices.sort_unstable();
            indices.dedup();
            for index in indices.into_iter().rev() {
                self.blocks[block].side_effects.remove(index);
            }
        }
    }

    /// Remove one side effect and reconnect every direct effect-token consumer
    /// to the removed effect's dependency.
    pub fn remove_effect_splicing_dependencies(&mut self, site: SideEffectSite) -> SideEffect<P, Lang> {
        let removed = self.blocks[site.block].side_effects.remove(site.index);
        if let Some((input, output)) = removed.effects() {
            for (_, block) in &mut self.blocks {
                for effect in &mut block.side_effects {
                    if let Some((consumer_input, _)) = effect.effects_mut() {
                        if *consumer_input == output {
                            *consumer_input = input;
                        }
                    }
                }
            }
        }
        removed
    }

    /// Split a block immediately before one of its side effects.
    ///
    /// The returned continuation receives the selected effect and every
    /// following effect, plus the original terminator and any structured
    /// control metadata. The original block is terminated by an unconditional
    /// branch to the continuation.
    pub fn split_block_before_effect(&mut self, block: BlockId, effect_index: usize) -> BlockId {
        let continuation = self.create_block();
        let source = &mut self.blocks[block];
        let suffix = source.side_effects.split_off(effect_index);
        let control_header = source.control_header.take();
        let old_term = std::mem::replace(
            &mut source.term,
            crate::flow::Terminator::Branch {
                target: continuation,
                args: Vec::new(),
            },
        );
        self.blocks[continuation].side_effects = suffix;
        self.blocks[continuation].term = old_term;
        self.blocks[continuation].control_header = control_header;
        continuation
    }

    /// Verify that every CFG edge supplies one argument per target block parameter.
    pub fn verify_branch_arities(&self) -> Result<(), String> {
        for (source, block) in &self.blocks {
            let check = |target: BlockId, args: &[FlowValueId]| {
                let target_block = self
                    .blocks
                    .get(target)
                    .ok_or_else(|| format!("branch from {source:?} targets an absent block {target:?}"))?;
                if args.len() != target_block.params.len() {
                    return Err(format!(
                        "branch from {source:?} to {target:?} supplies {} arguments for {} parameters",
                        args.len(),
                        target_block.params.len()
                    ));
                }
                Ok(())
            };
            match &block.term {
                crate::flow::Terminator::Branch { target, args } => check(*target, args)?,
                crate::flow::Terminator::CondBranch {
                    then_target,
                    then_args,
                    else_target,
                    else_args,
                    ..
                } => {
                    check(*then_target, then_args)?;
                    check(*else_target, else_args)?;
                }
                crate::flow::Terminator::Return(_) | crate::flow::Terminator::Unreachable => {}
            }
        }
        Ok(())
    }
}

/// Stable-for-a-snapshot location of a side effect in the skeleton.
///
/// Side effects are still stored in ordered per-block vectors, so a site must
/// not outlive an insertion/removal/reorder in those vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SideEffectSite {
    pub block: BlockId,
    pub index: usize,
}

/// Read-side index for explicitly located instructions and their results.
///
/// Build this once for a graph snapshot and share it across related queries.
/// Rebuild it after any structural skeleton mutation.
pub struct SideEffectIndex {
    by_result: LookupMap<ValueId, SideEffectSite>,
    by_call: LookupMap<CallSiteId, SideEffectSite>,
}

impl SideEffectIndex {
    pub fn build<P: Family, Lang: Language>(graph: &EGraph<P, Lang>) -> Self {
        let mut by_result = LookupMap::new();
        let mut by_call = LookupMap::new();
        let mut result_sites = Vec::new();
        for (block, skeleton_block) in &graph.skeleton.blocks {
            for (index, effect) in skeleton_block.side_effects.iter().enumerate() {
                let site = SideEffectSite { block, index };
                if let SideEffectKind::Effect(EffectOp::Call { site: call }) = effect.kind() {
                    let previous = by_call.insert(*call, site);
                    assert!(
                        previous.is_none(),
                        "call has more than one explicit site: {call:?}"
                    );
                }
                let Some(result) = graph.effect_result_binding(effect) else {
                    continue;
                };
                for value in result.values() {
                    result_sites.push((value, site));
                    let previous = by_result.insert(value, site);
                    assert!(
                        previous.is_none() || previous == Some(site),
                        "side-effect result {value:?} has producers {previous:?} and {site:?}"
                    );
                }
            }
        }
        for (value, site) in result_sites {
            by_result.entry(graph.canonical_value(value)).or_insert(site);
        }
        Self { by_result, by_call }
    }

    pub fn site(&self, result: ValueId) -> Option<SideEffectSite> {
        self.by_result.get(&result).copied()
    }

    pub fn call_site(&self, call: CallSiteId) -> Option<SideEffectSite> {
        self.by_call.get(&call).copied()
    }

    pub fn calls(&self) -> impl Iterator<Item = (CallSiteId, SideEffectSite)> + '_ {
        self.by_call.iter().map(|(call, site)| (*call, *site))
    }

    pub fn effect<'a, P: Family, Lang: Language>(
        &self,
        graph: &'a EGraph<P, Lang>,
        result: ValueId,
    ) -> Option<&'a SideEffect<P, Lang>> {
        let site = self.site(result)?;
        let effect = graph.skeleton.blocks.get(site.block)?.side_effects.get(site.index)?;
        let result = graph.canonical_value(result);
        graph
            .effect_result_binding(effect)
            .is_some_and(|binding| {
                binding.values().into_iter().any(|value| graph.canonical_value(value) == result)
            })
            .then_some(effect)
    }

    pub fn effect_result_field<'a, P: Family, Lang: Language>(
        &self,
        graph: &'a EGraph<P, Lang>,
        value: ValueId,
    ) -> Option<(&'a SideEffect<P, Lang>, ValueId, usize)> {
        let value = graph.canonical_value(value);
        if let Some(effect) = self.effect(graph, value) {
            let field =
                graph.effect_result_binding(effect)?.top_level_fields().iter().position(|field| {
                    field
                        .values()
                        .into_iter()
                        .any(|field_value| graph.canonical_value(field_value) == value)
                })?;
            return Some((effect, value, field));
        }
        graph.value(value).result_origins().iter().find_map(|origin| {
            let representative = *origin.values().first()?;
            let effect = self.effect(graph, representative)?;
            let field = graph.effect_result_binding(effect)?.top_level_field_index(origin)?;
            Some((effect, representative, field))
        })
    }

    pub fn effect_mut<'a, P: Family, Lang: Language>(
        &self,
        graph: &'a mut EGraph<P, Lang>,
        result: ValueId,
    ) -> Option<&'a mut SideEffect<P, Lang>> {
        let result = graph.canonical_value(result);
        let site = self.site(result)?;
        let contains_result = {
            let effect = graph.skeleton.blocks.get(site.block)?.side_effects.get(site.index)?;
            graph.effect_result_binding(effect).is_some_and(|binding| {
                binding.values().into_iter().any(|value| graph.canonical_value(value) == result)
            })
        };
        if !contains_result {
            return None;
        }
        let effect = graph.skeleton.blocks.get_mut(site.block)?.side_effects.get_mut(site.index)?;
        Some(effect)
    }
}

// ---------------------------------------------------------------------------
// EGraph — the main container
// ---------------------------------------------------------------------------

/// The acyclic e-graph: a sea of pure nodes + a CFG skeleton of side effects.
#[derive(Clone, Debug)]
pub struct EGraph<P: Family, Lang: Language> {
    /// All nodes (pure, union, params, constants, side-effect results),
    /// including their type, span, and canonical alias.
    pub(crate) nodes: SlotMap<ValueId, Value<P::Resource, Lang>>,
    /// Addressable identities are not value nodes and cannot participate in
    /// e-class unioning or CFG-carried state.
    pub(crate) places: SlotMap<PlaceId, Place<P::Resource, Lang::Ty>>,
    /// Complete calls are stored once and referenced by the skeleton and their
    /// by-value result nodes.
    pub(crate) calls: SlotMap<CallSiteId, CallSite<Lang::Ty>>,
    /// Hash-cons table: PureValueKey → existing value identity.
    hash_cons: LookupMap<PureValueKey<P::Resource, Lang>, ValueId>,
    /// Constant dedup cache.
    const_cache: LookupMap<Lang::Const, ValueId>,
    /// The CFG skeleton.
    pub skeleton: Skeleton<P, Lang>,
}

/// Graph state excluding indexes derived from that state.
///
/// Transformations may consume and rebuild an `EGraph` through this boundary
/// without gaining direct access to its hash-consing internals.
pub(super) struct EGraphParts<P: Family, Lang: Language> {
    pub(super) nodes: SlotMap<ValueId, Value<P::Resource, Lang>>,
    pub(super) places: SlotMap<PlaceId, Place<P::Resource, Lang::Ty>>,
    pub(super) calls: SlotMap<CallSiteId, CallSite<Lang::Ty>>,
    pub(super) skeleton: Skeleton<P, Lang>,
}

pub trait GraphResource: Clone + std::fmt::Debug + Eq + std::hash::Hash {}

impl<T> GraphResource for T where T: Clone + std::fmt::Debug + Eq + std::hash::Hash {}

impl<P: Family, Lang: Language> EGraph<P, Lang> {
    /// The canonical result binding anchored by a skeleton effect. Call
    /// results are owned by the complete call boundary; every other effect
    /// owns its result directly.
    pub fn effect_result_binding<'a>(
        &'a self,
        effect: &'a SideEffect<P, Lang>,
    ) -> Option<&'a ResultBinding<Lang::Ty>> {
        match effect.kind() {
            SideEffectKind::Effect(EffectOp::Call { site }) => Some(self.call(*site).result()),
            _ => effect.result(),
        }
    }

    pub fn new() -> Self {
        EGraph {
            nodes: SlotMap::with_key(),
            places: SlotMap::with_key(),
            calls: SlotMap::with_key(),
            hash_cons: LookupMap::new(),
            const_cache: LookupMap::new(),
            skeleton: Skeleton::new(),
        }
    }

    pub(super) fn into_parts(self) -> EGraphParts<P, Lang> {
        let Self {
            nodes,
            places,
            calls,
            hash_cons: _,
            const_cache: _,
            skeleton,
        } = self;
        EGraphParts {
            nodes,
            places,
            calls,
            skeleton,
        }
    }

    pub(super) fn from_parts(parts: EGraphParts<P, Lang>) -> Self {
        let EGraphParts {
            nodes,
            places,
            calls,
            skeleton,
        } = parts;
        let mut graph = Self {
            nodes,
            places,
            calls,
            hash_cons: LookupMap::new(),
            const_cache: LookupMap::new(),
            skeleton,
        };
        graph.rebuild_hash_cons();
        graph.rebuild_const_cache();
        graph
    }

    pub fn side_effect_index(&self) -> SideEffectIndex {
        SideEffectIndex::build(self)
    }

    pub fn values(&self) -> &SlotMap<ValueId, Value<P::Resource, Lang>> {
        &self.nodes
    }

    pub fn value(&self, id: ValueId) -> &Value<P::Resource, Lang> {
        &self.nodes[id]
    }

    pub fn places(&self) -> &SlotMap<PlaceId, Place<P::Resource, Lang::Ty>> {
        &self.places
    }

    pub fn place(&self, id: PlaceId) -> &Place<P::Resource, Lang::Ty> {
        &self.places[id]
    }

    pub fn calls(&self) -> &SlotMap<CallSiteId, CallSite<Lang::Ty>> {
        &self.calls
    }

    pub fn call(&self, id: CallSiteId) -> &CallSite<Lang::Ty> {
        &self.calls[id]
    }

    /// Whether a skeleton entry must participate in effect ordering. Pure
    /// calls retain an explicit anchor so their complete call boundary has a
    /// unique location, but their value dependencies determine evaluation.
    pub fn effect_requires_ordering(&self, effect: &SideEffect<P, Lang>) -> bool {
        !matches!(
            effect.kind(),
            SideEffectKind::Effect(EffectOp::Call { site })
                if self.call(*site).effects() == CallEffects::Pure
        )
    }

    pub fn has_ordered_effects(&self) -> bool {
        self.skeleton
            .blocks
            .values()
            .flat_map(|block| &block.side_effects)
            .any(|effect| self.effect_requires_ordering(effect))
    }

    pub fn call_value_dependencies(&self, id: CallSiteId) -> Vec<ValueId> {
        let mut values = Vec::new();
        for argument in self.call(id).arguments() {
            match *argument {
                OperandRef::Value(value) => values.push(value),
                OperandRef::View(view) => values.push(view.value()),
                OperandRef::Place(place) => values.extend(self.place_value_dependencies(place)),
            }
        }
        values
    }

    pub fn value_dependencies(&self, id: ValueId) -> Vec<ValueId> {
        let id = self.canonical_value(id);
        match self.value(id).kind() {
            ValueKind::CallResult { call, .. } => self.call_value_dependencies(*call),
            ValueKind::PlaceLength { place } | ValueKind::PlaceView { place } => {
                self.place_value_dependencies(*place)
            }
            kind => kind.children().into_vec(),
        }
    }

    pub fn effect_boundary_value_dependencies(&self, effect: &SideEffect<P, Lang>) -> Vec<ValueId> {
        let mut values = Vec::new();
        for operand in effect.operands() {
            match *operand {
                OperandRef::Value(value) => values.push(value),
                OperandRef::View(view) => values.push(view.value()),
                OperandRef::Place(place) => values.extend(self.place_value_dependencies(place)),
            }
        }
        if let SideEffectKind::Effect(operation) = effect.kind() {
            match operation {
                EffectOp::Call { site } if self.call(*site).effects() != CallEffects::Pure => {
                    values.extend(self.call_value_dependencies(*site));
                }
                EffectOp::Call { .. } => {}
                EffectOp::Alloca { result }
                | EffectOp::Load { place: result, .. }
                | EffectOp::Store { place: result }
                | EffectOp::Atomic { place: result, .. } => {
                    values.extend(self.place_value_dependencies(*result));
                }
                EffectOp::Op { .. } | EffectOp::ControlBarrier => {}
            }
        }
        if let Some(result) = self.effect_result_binding(effect) {
            result.for_each_destination(|_, destination| match destination {
                ResultDestination::ReturnValue(_) => {}
                ResultDestination::Place(PlaceDestination::Fixed(place)) => {
                    values.extend(self.place_value_dependencies(*place));
                }
                ResultDestination::Place(PlaceDestination::Bounded { storage, length }) => {
                    values.extend(self.place_value_dependencies(*storage));
                    values.extend(self.place_value_dependencies(*length));
                }
            });
        }
        values
    }

    pub fn place_value_dependencies(&self, root: PlaceId) -> Vec<ValueId> {
        let mut values = Vec::new();
        let mut pending = vec![root];
        let mut seen = LookupSet::new();
        while let Some(place) = pending.pop() {
            if !seen.insert(place) {
                continue;
            }
            match self.place(place).op() {
                PlaceOp::Parameter { .. } | PlaceOp::AllocaResult | PlaceOp::OutputSlot { .. } => {}
                PlaceOp::View { view } => values.push(view.value()),
                PlaceOp::Index { base, index } => {
                    pending.push(*base);
                    values.push(*index);
                }
                PlaceOp::Slice { base, start, length } => {
                    pending.push(*base);
                    values.push(*start);
                    values.push(*length);
                }
                PlaceOp::ViewIndex { view, index } => {
                    values.push(view.value());
                    values.push(*index);
                }
            }
        }
        values
    }

    pub fn admit_flow_value(&self, id: ValueId) -> FlowValueId {
        let id = self.canonical_value(id);
        assert!(
            !Lang::is_materialized_aggregate(self.value(id).ty()),
            "materialized aggregate cannot enter CFG-carried state"
        );
        FlowValueId(id)
    }

    pub fn admit_flow_values(&self, values: impl IntoIterator<Item = ValueId>) -> Vec<FlowValueId> {
        values.into_iter().map(|value| self.admit_flow_value(value)).collect()
    }

    pub fn view_id(&self, id: ValueId) -> ViewId {
        let id = self.canonical_value(id);
        assert!(
            Lang::is_view(self.value(id).ty()) && self.try_view_region(id).is_some(),
            "value {id:?} is not an addressable view: type {:?}, definition {:?}",
            self.value(id).ty(),
            self.value(id).kind()
        );
        ViewId(id)
    }

    pub fn operand_ref(&self, id: ValueId) -> OperandRef {
        let id = self.canonical_value(id);
        if Lang::is_view(self.value(id).ty()) && self.try_view_region(id).is_some() {
            OperandRef::View(ViewId(id))
        } else {
            OperandRef::Value(id)
        }
    }

    pub fn canonical_operand(&self, operand: OperandRef) -> OperandRef {
        match operand {
            OperandRef::Value(value) => self.operand_ref(value),
            OperandRef::View(view) => self.operand_ref(view.value()),
            OperandRef::Place(place) => OperandRef::Place(place),
        }
    }

    pub fn canonicalize_boundary_operands(&mut self) {
        let representations =
            self.nodes.keys().map(|value| (value, self.operand_ref(value))).collect::<LookupMap<_, _>>();
        let canonicalize = |operand: &mut OperandRef| {
            if let Some(value) = operand.value() {
                *operand = representations[&value];
            }
        };
        for (_, call) in &mut self.calls {
            for argument in call.arguments_mut() {
                canonicalize(argument);
            }
        }
        for (_, block) in &mut self.skeleton.blocks {
            for effect in &mut block.side_effects {
                for operand in effect.operands_mut() {
                    canonicalize(operand);
                }
            }
        }
    }

    /// Follow the graph's canonical value substitutions.
    pub fn canonical_value(&self, mut id: ValueId) -> ValueId {
        let mut seen = LookupSet::new();
        while let Some(alias) = self.value(id).alias() {
            assert!(seen.insert(id), "value alias cycle at {id:?}");
            id = alias;
        }
        id
    }

    pub fn value_result(&self, id: ValueId) -> ResultBinding<Lang::Ty> {
        ResultBinding::destination(self.value(id).ty().clone(), ResultDestination::ReturnValue(id))
    }

    pub fn register_result_origin(&mut self, value: ValueId, origin: ResultBinding<Lang::Ty>) {
        assert_eq!(self.value(value).ty(), origin.ty());
        let origins = &mut self.nodes[value].result_origins;
        if !origins.contains(&origin) {
            origins.push(origin);
        }
    }

    pub fn value_has_result_origin(&self, value: ValueId, origin: &ResultBinding<Lang::Ty>) -> bool {
        let value = self.canonical_value(value);
        origin.single_value() == Some(value)
            || self.nodes[value].result_origins.iter().any(|candidate| candidate == origin)
    }

    pub fn emit_call(
        &mut self,
        block: BlockId,
        callee: RegionId,
        parameters: &[FuncParam<P::Resource, Lang::Ty>],
        function_result: &FunctionResult<Lang::Ty>,
        arguments: impl IntoIterator<Item = OperandRef>,
        effects: CallEffects,
        effect_tokens: Option<(EffectToken, EffectToken)>,
        span: Option<Span>,
    ) -> Result<(CallSiteId, ResultBinding<Lang::Ty>), String> {
        if !self.skeleton.blocks.contains_key(block) {
            return Err(format!("call to {callee:?} names missing block {block:?}"));
        }
        let arguments =
            arguments.into_iter().map(|argument| self.canonical_operand(argument)).collect::<Box<[_]>>();
        if arguments.len() != parameters.len() {
            return Err(format!(
                "call to {callee:?} supplies {} arguments for {} parameters",
                arguments.len(),
                parameters.len()
            ));
        }
        for (index, (argument, parameter)) in arguments.iter().zip(parameters).enumerate() {
            let matches = match (argument, parameter.representation()) {
                (OperandRef::Value(value), OperandType::Value(ty)) => {
                    self.nodes.get(*value).is_some_and(|value| Lang::view_argument_matches(ty, value.ty()))
                }
                (OperandRef::View(view), OperandType::View(ty)) => self
                    .nodes
                    .get(view.value())
                    .is_some_and(|value| Lang::view_argument_matches(&ty.array, value.ty())),
                (OperandRef::Place(place), OperandType::Place(ty)) => {
                    self.places.get(*place).is_some_and(|place| {
                        Lang::view_argument_matches(&ty.pointee, &place.ty().pointee)
                            && ty.access.accepts(place.ty().access)
                    })
                }
                _ => false,
            };
            if !matches {
                let argument_ty = argument.value().and_then(|value| self.nodes.get(value)).map(Value::ty);
                return Err(format!(
                    "call to {callee:?} argument {index} is {argument:?} with type {argument_ty:?}, expected {:?}",
                    parameter.representation()
                ));
            }
        }
        let mut destination_error = None;
        function_result.for_each_destination(|_, destination| {
            let mut require_place = |parameter: ParameterId| {
                let valid = parameters
                    .get(parameter.index())
                    .is_some_and(|parameter| matches!(parameter.representation(), OperandType::Place(_)));
                if !valid && destination_error.is_none() {
                    destination_error = Some(format!(
                        "call to {callee:?} result names non-place destination parameter {}",
                        parameter.index()
                    ));
                }
            };
            match destination {
                ResultDestination::ReturnValue(_) => {}
                ResultDestination::Place(PlaceDestination::Fixed(parameter)) => require_place(*parameter),
                ResultDestination::Place(PlaceDestination::Bounded { storage, length }) => {
                    require_place(*storage);
                    require_place(*length);
                }
            }
        });
        if let Some(error) = destination_error {
            return Err(error);
        }
        let nodes = &mut self.nodes;
        let mut result = None;
        let site = self.calls.insert_with_key(|site| {
            let binding = function_result.bind(
                |slot, ty| {
                    nodes.insert(Value {
                        kind: ValueKind::CallResult { call: site, slot },
                        ty: ty.clone(),
                        span,
                        alias: None,
                        result_origins: Vec::new(),
                    })
                },
                |parameter| {
                    arguments[parameter.index()]
                        .place()
                        .expect("destination parameter requires a place argument")
                },
            );
            result = Some(binding.clone());
            CallSite::new(callee, arguments.clone(), binding, effects)
        });
        self.skeleton.blocks[block].side_effects.push(SideEffect::new(
            SideEffectKind::Effect(EffectOp::Call { site }),
            smallvec::smallvec![],
            None,
            effect_tokens,
            span,
        ));
        Ok((
            site,
            result.expect("call result is constructed with its call site"),
        ))
    }

    pub(crate) fn add_projected_call(
        &mut self,
        source: &CallSite<Lang::Ty>,
        arguments: Box<[OperandRef]>,
        mut source_result: impl FnMut(ValueId) -> (ReturnSlotId, Lang::Ty, Option<Span>),
        mut map_place: impl FnMut(PlaceId) -> PlaceId,
    ) -> (CallSiteId, ResultBinding<Lang::Ty>, Vec<(ValueId, ValueId)>) {
        let arguments = arguments
            .into_vec()
            .into_iter()
            .map(|argument| self.canonical_operand(argument))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let nodes = &mut self.nodes;
        let mut result = None;
        let mut value_map = Vec::new();
        let site = self.calls.insert_with_key(|site| {
            let binding = source.result.clone().map(
                |ty| ty,
                |source_value| {
                    let (slot, ty, span) = source_result(source_value);
                    let target = nodes.insert(Value {
                        kind: ValueKind::CallResult { call: site, slot },
                        ty,
                        span,
                        alias: None,
                        result_origins: Vec::new(),
                    });
                    value_map.push((source_value, target));
                    target
                },
                &mut map_place,
            );
            result = Some(binding.clone());
            CallSite::new(source.callee, arguments.clone(), binding, source.effects)
        });
        (
            site,
            result.expect("projected call result is constructed with its call site"),
            value_map,
        )
    }

    pub fn add_place_length(&mut self, place: PlaceId, ty: Lang::Ty, span: Option<Span>) -> ValueId {
        self.nodes.insert(Value {
            kind: ValueKind::PlaceLength { place },
            ty,
            span,
            alias: None,
            result_origins: Vec::new(),
        })
    }

    pub fn add_place_view(&mut self, place: PlaceId, ty: Lang::Ty, span: Option<Span>) -> ViewId {
        assert!(
            Lang::is_view(&ty),
            "place view must have an addressable view type"
        );
        if let PlaceOp::View { view } = self.place(place).op() {
            return *view;
        }
        ViewId(self.nodes.insert(Value {
            kind: ValueKind::PlaceView { place },
            ty,
            span,
            alias: None,
            result_origins: Vec::new(),
        }))
    }

    fn insert_place(
        &mut self,
        op: PlaceOp,
        ty: PlaceType<P::Resource, Lang::Ty>,
        span: Option<Span>,
    ) -> PlaceId {
        self.places.insert(Place { op, ty, span })
    }

    pub fn add_place_parameter(
        &mut self,
        parameter: ParameterId,
        ty: PlaceType<P::Resource, Lang::Ty>,
    ) -> PlaceId {
        self.insert_place(PlaceOp::Parameter { parameter }, ty, None)
    }

    pub fn add_view_place(
        &mut self,
        view: ViewId,
        pointee: Lang::Ty,
        access: PlaceAccess,
        span: Option<Span>,
    ) -> PlaceId {
        let (region, view_access) = self.view_region(view.value());
        self.insert_place(
            PlaceOp::View { view },
            PlaceType {
                pointee,
                region,
                access: if view_access.accepts(access) { access } else { view_access },
            },
            span,
        )
    }

    pub fn add_alloca_place(
        &mut self,
        ty: PlaceType<P::Resource, Lang::Ty>,
        span: Option<Span>,
    ) -> PlaceId {
        self.insert_place(PlaceOp::AllocaResult, ty, span)
    }

    pub fn add_index_place(
        &mut self,
        base: PlaceId,
        index: ValueId,
        pointee: Lang::Ty,
        span: Option<Span>,
    ) -> PlaceId {
        if let PlaceOp::View { view } = self.place(base).op().clone() {
            return self.add_view_index_place(view, index, pointee, span);
        }
        if let PlaceOp::Slice { base, start, .. } = self.place(base).op().clone() {
            let index_ty = self.value(index).ty().clone();
            let index = self.intern_pure(
                OpTag::BinOp(crate::op::BinaryOperator::Add),
                smallvec::smallvec![start, index],
                index_ty,
                span,
            );
            return self.add_index_place(base, index, pointee, span);
        }
        let parent = self.place(base).ty();
        let ty = PlaceType {
            pointee,
            region: parent.region.clone(),
            access: parent.access,
        };
        self.insert_place(PlaceOp::Index { base, index }, ty, span)
    }

    pub fn add_slice_place(
        &mut self,
        base: PlaceId,
        start: ValueId,
        length: ValueId,
        pointee: Lang::Ty,
        span: Option<Span>,
    ) -> PlaceId {
        let parent = self.place(base).ty();
        let ty = PlaceType {
            pointee,
            region: parent.region.clone(),
            access: parent.access,
        };
        self.insert_place(PlaceOp::Slice { base, start, length }, ty, span)
    }

    pub fn add_view_index_place(
        &mut self,
        view: ViewId,
        index: ValueId,
        pointee: Lang::Ty,
        span: Option<Span>,
    ) -> PlaceId {
        if let ValueKind::PlaceView { place } = self.value(view.value()).kind() {
            return self.add_index_place(*place, index, pointee, span);
        }
        let (region, access) = self.view_region(view.value());
        let ty = PlaceType {
            pointee,
            region,
            access,
        };
        self.insert_place(PlaceOp::ViewIndex { view, index }, ty, span)
    }

    fn try_view_region(&self, view: ValueId) -> Option<(PlaceRegion<P::Resource>, PlaceAccess)> {
        let value = self.value(view);
        if let Some(alias) = value.alias {
            return self.try_view_region(alias);
        }
        match value.kind() {
            ValueKind::Pure {
                op: OpTag::StorageView(PureViewSource::Storage(resource)),
                ..
            } => Some((PlaceRegion::Resource(resource.clone()), PlaceAccess::ReadWrite)),
            ValueKind::Pure {
                op: OpTag::StorageView(PureViewSource::Workgroup { .. }),
                ..
            } => Some((PlaceRegion::Workgroup, PlaceAccess::ReadWrite)),
            ValueKind::Pure {
                op: OpTag::StorageView(PureViewSource::Inherited),
                operands,
            } => {
                let parent = *operands.last().expect("inherited view has a parent operand");
                self.try_view_region(parent)
            }
            ValueKind::Pure {
                op: OpTag::Project { .. },
                operands,
            } if operands.len() == 1 => Some((PlaceRegion::Parametric, PlaceAccess::ReadOnly)),
            ValueKind::FuncParam { .. }
            | ValueKind::BlockParam { .. }
            | ValueKind::CallResult { .. }
            | ValueKind::Union { .. } => Some((PlaceRegion::Parametric, PlaceAccess::ReadOnly)),
            ValueKind::PlaceView { place } => {
                let place = self.place(*place).ty();
                Some((place.region.clone(), place.access))
            }
            ValueKind::Pure { .. }
            | ValueKind::Constant(_)
            | ValueKind::SideEffectResult
            | ValueKind::PlaceLength { .. } => None,
        }
    }

    fn view_region(&self, view: ValueId) -> (PlaceRegion<P::Resource>, PlaceAccess) {
        self.try_view_region(view).unwrap_or_else(|| {
            panic!(
                "value {view:?} is marked as a view but {:?} has no addressable view producer",
                self.value(view).kind()
            )
        })
    }

    pub fn add_output_place(&mut self, index: usize, ty: PlaceType<P::Resource, Lang::Ty>) -> PlaceId {
        self.insert_place(PlaceOp::OutputSlot { index }, ty, None)
    }

    fn pure_node_key(&self, id: ValueId) -> Option<PureValueKey<P::Resource, Lang>> {
        let node = self.nodes.get(id)?;
        let ValueKind::Pure { op, operands } = &node.kind else {
            return None;
        };
        Some(PureValueKey {
            op: op.clone(),
            operands: operands.clone(),
            ty: node.ty.clone(),
        })
    }

    fn unindex_current_pure(&mut self, id: ValueId) {
        let Some(key) = self.pure_node_key(id) else {
            return;
        };
        if self.hash_cons.get(&key) == Some(&id) {
            self.hash_cons.remove(&key);
        }
    }

    fn index_current_pure(&mut self, id: ValueId) {
        let Some(key) = self.pure_node_key(id) else {
            return;
        };
        self.hash_cons.entry(key).or_insert(id);
    }

    /// Replace a node in place without changing its result type, keeping the
    /// pure-node hash-cons table consistent across the mutation.
    pub fn replace_node_preserving_type(&mut self, id: ValueId, node: ValueKind<P::Resource, Lang>) {
        self.unindex_current_pure(id);
        self.nodes[id].kind = node;
        self.index_current_pure(id);
    }

    /// Replace a pure node's operator and operands without changing its result
    /// type, keeping the hash-cons table consistent across the mutation.
    pub fn replace_pure_node(
        &mut self,
        id: ValueId,
        op: PureOp<P::Resource>,
        operands: SmallVec<[ValueId; 4]>,
    ) {
        self.replace_node_preserving_type(id, ValueKind::Pure { op, operands });
    }

    /// Mutate a pure node's operator and operands in place while maintaining
    /// the hash-cons table. Returns false if `id` is not a pure node.
    pub fn update_pure_node<F>(&mut self, id: ValueId, update: F) -> bool
    where
        F: FnOnce(&mut PureOp<P::Resource>, &mut SmallVec<[ValueId; 4]>),
    {
        if !matches!(
            self.nodes.get(id).map(|node| &node.kind),
            Some(ValueKind::Pure { .. })
        ) {
            return false;
        }
        self.unindex_current_pure(id);
        if let ValueKind::Pure { op, operands } = &mut self.nodes[id].kind {
            update(op, operands);
        }
        self.index_current_pure(id);
        true
    }

    /// Change a node's result type while maintaining the pure-node hash-cons
    /// key when the node is hash-consed.
    pub fn retype_node(&mut self, id: ValueId, ty: Lang::Ty) {
        self.unindex_current_pure(id);
        self.nodes[id].ty = ty;
        self.index_current_pure(id);
    }

    /// Remove a function-parameter node and its graph-owned metadata.
    pub fn remove_func_param(&mut self, id: ValueId) -> bool {
        if !matches!(
            self.nodes.get(id).map(|node| &node.kind),
            Some(ValueKind::FuncParam { .. })
        ) {
            return false;
        }
        self.nodes.remove(id).is_some()
    }

    /// Replace references inside graph-owned nodes. Skeleton side-effect
    /// operands are handled by higher-level graph rewriting helpers.
    pub fn replace_node_references(&mut self, old: ValueId, new: ValueId) {
        if old == new {
            return;
        }

        let ids: Vec<ValueId> = self.nodes.keys().collect();
        for id in ids {
            for origin in &mut self.nodes[id].result_origins {
                origin.replace_value(old, new);
            }
            if self.nodes[id].alias == Some(old) {
                self.nodes[id].alias = Some(new);
            }
            match self.nodes.get(id).map(|node| &node.kind) {
                Some(ValueKind::Pure { operands, .. }) if operands.contains(&old) => {
                    self.update_pure_node(id, |_, operands| {
                        for operand in operands {
                            if *operand == old {
                                *operand = new;
                            }
                        }
                    });
                }
                Some(ValueKind::Union { .. }) => {
                    if let ValueKind::Union { left, right } = &mut self.nodes[id].kind {
                        if *left == old {
                            *left = new;
                        }
                        if *right == old {
                            *right = new;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn replace_value_references(&mut self, old: ValueId, new: ValueId) {
        if old == new {
            return;
        }
        let replacement_operand = self.operand_ref(new);
        if let OperandRef::View(view) = replacement_operand {
            if let ValueKind::PlaceView { place } = self.nodes[view.value()].kind() {
                let place = *place;
                for (_, node) in &mut self.nodes {
                    for origin in &mut node.result_origins {
                        origin.replace_value_with_place(old, place);
                    }
                }
                for (_, call) in &mut self.calls {
                    call.result.replace_value_with_place(old, place);
                }
                for (_, block) in &mut self.skeleton.blocks {
                    for effect in &mut block.side_effects {
                        if let Some(result) = &mut effect.result {
                            result.replace_value_with_place(old, place);
                        }
                    }
                    if let crate::flow::Terminator::Return(Some(result)) = &mut block.term {
                        result.replace_value_with_place(old, place);
                    }
                }
            }
        }
        let swap = |value: ValueId| if value == old { new } else { value };
        self.replace_node_references(old, new);
        let replacement_place = match replacement_operand {
            OperandRef::View(view) => match self.nodes[view.value()].kind() {
                ValueKind::PlaceView { place } => Some(*place),
                _ => None,
            },
            _ => None,
        };
        for (_, place) in &mut self.places {
            place.remap_value(swap);
            if let (Some(base), PlaceOp::ViewIndex { view, index }) = (replacement_place, place.op.clone())
            {
                if view.value() == new {
                    place.op = PlaceOp::Index { base, index };
                }
            }
        }
        if let Some(base) = replacement_place {
            let inherited = self
                .nodes
                .iter()
                .filter_map(|(value, definition)| match definition.kind() {
                    ValueKind::Pure {
                        op: OpTag::StorageView(PureViewSource::Inherited),
                        operands,
                    } if operands.len() == 3 && operands[2] == new => Some((
                        value,
                        operands[0],
                        operands[1],
                        definition.ty().clone(),
                        definition.span(),
                    )),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let mut slices = LookupMap::new();
            for (value, start, length, ty, span) in inherited {
                let slice = self.add_slice_place(base, start, length, ty, span);
                self.unindex_current_pure(value);
                self.nodes[value].kind = ValueKind::PlaceView { place: slice };
                slices.insert(value, slice);
            }
            let indexed_slices = self
                .places
                .iter()
                .filter_map(|(place, definition)| {
                    let PlaceOp::ViewIndex { view, index } = definition.op() else {
                        return None;
                    };
                    let base = *slices.get(&view.value())?;
                    Some((
                        place,
                        base,
                        *index,
                        definition.ty().pointee.clone(),
                        definition.span(),
                    ))
                })
                .collect::<Vec<_>>();
            for (place, base, index, pointee, span) in indexed_slices {
                let indexed = self.add_index_place(base, index, pointee, span);
                self.replace_place_references(place, indexed);
            }
        }
        for (_, call) in &mut self.calls {
            for argument in call.arguments_mut() {
                if argument.value() == Some(old) {
                    *argument = replacement_operand;
                }
            }
            call.result.replace_value(old, new);
        }
        for (_, block) in &mut self.skeleton.blocks {
            for effect in &mut block.side_effects {
                for operand in &mut effect.operands {
                    if operand.value() == Some(old) {
                        *operand = replacement_operand;
                    }
                }
                if let Some(result) = &mut effect.result {
                    result.replace_value(old, new);
                }
                if let SideEffectKind::Soac(soac) = &mut effect.kind {
                    P::remap_soac_values(soac, &mut |value| swap(value));
                }
            }
            block.term.visit_values_mut(|value| *value = swap(*value));
        }
    }

    pub fn replace_place_references(&mut self, old: PlaceId, new: PlaceId) {
        if old == new {
            return;
        }
        for (_, place) in &mut self.places {
            place.remap_place(|place| if place == old { new } else { place });
        }
        for (_, node) in &mut self.nodes {
            match &mut node.kind {
                ValueKind::PlaceLength { place } | ValueKind::PlaceView { place } if *place == old => {
                    *place = new;
                }
                _ => {}
            }
            for origin in &mut node.result_origins {
                origin.replace_place(old, new);
            }
        }
        for (_, call) in &mut self.calls {
            for argument in &mut call.arguments {
                if *argument == OperandRef::Place(old) {
                    *argument = OperandRef::Place(new);
                }
            }
            call.result.replace_place(old, new);
        }
        for (_, block) in &mut self.skeleton.blocks {
            for effect in &mut block.side_effects {
                for operand in &mut effect.operands {
                    if *operand == OperandRef::Place(old) {
                        *operand = OperandRef::Place(new);
                    }
                }
                if let Some(result) = &mut effect.result {
                    result.replace_place(old, new);
                }
                if let SideEffectKind::Effect(operation) = &mut effect.kind {
                    match operation {
                        EffectOp::Load { place, .. }
                        | EffectOp::Store { place }
                        | EffectOp::Atomic { place, .. }
                            if *place == old =>
                        {
                            *place = new;
                        }
                        EffectOp::Call { .. }
                        | EffectOp::Alloca { .. }
                        | EffectOp::Op { .. }
                        | EffectOp::Load { .. }
                        | EffectOp::Store { .. }
                        | EffectOp::Atomic { .. }
                        | EffectOp::ControlBarrier => {}
                    }
                }
            }
            if let crate::flow::Terminator::Return(Some(result)) = &mut block.term {
                result.replace_place(old, new);
            }
        }
    }

    /// Install canonical aliases produced by a graph rewrite. Alias ownership
    /// follows the source node, so later graph copies and removals cannot leave
    /// a detached side table behind.
    pub fn install_aliases(&mut self, aliases: impl IntoIterator<Item = (ValueId, ValueId)>) {
        for (source, target) in aliases {
            self.nodes[source].alias = Some(target);
        }
    }

    /// Rebuild the pure-node hash-cons table after a bulk rewrite that may
    /// have changed pure node operands, operators, or result types in place.
    pub fn rebuild_hash_cons(&mut self) {
        let mut rebuilt = LookupMap::new();
        for (id, node) in self.nodes.iter() {
            if matches!(&node.kind, ValueKind::Pure { .. }) {
                if let Some(key) = self.pure_node_key(id) {
                    rebuilt.entry(key).or_insert(id);
                }
            }
        }
        self.hash_cons = rebuilt;
    }

    fn rebuild_const_cache(&mut self) {
        self.const_cache = self
            .nodes
            .iter()
            .filter_map(|(id, node)| match &node.kind {
                ValueKind::Constant(value) => Some((value.clone(), id)),
                _ => None,
            })
            .collect();
    }

    /// Check that every hash-cons entry points to a pure node matching its key
    /// and that every current pure-node key is represented in the table.
    pub fn verify_hash_cons(&self) -> Result<(), String> {
        for (key, &id) in &self.hash_cons {
            let Some(current) = self.pure_node_key(id) else {
                return Err(format!(
                    "hash_cons key {:?} points to non-pure node {:?}",
                    key, id
                ));
            };
            if &current != key {
                return Err(format!(
                    "hash_cons key {:?} points to node {:?} with current key {:?}",
                    key, id, current
                ));
            }
        }

        for (id, node) in self.nodes.iter() {
            if matches!(&node.kind, ValueKind::Pure { .. }) {
                let Some(key) = self.pure_node_key(id) else {
                    return Err(format!("pure node {:?} has no type", id));
                };
                match self.hash_cons.get(&key) {
                    Some(&indexed) if self.pure_node_key(indexed).as_ref() == Some(&key) => {}
                    Some(&indexed) => {
                        return Err(format!(
                            "pure node {:?} key {:?} is represented by stale node {:?}",
                            id, key, indexed
                        ));
                    }
                    None => {
                        return Err(format!(
                            "pure node {:?} key {:?} is missing from hash_cons",
                            id, key
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    fn insert_node(
        &mut self,
        kind: ValueKind<P::Resource, Lang>,
        ty: Lang::Ty,
        span: Option<Span>,
    ) -> ValueId {
        self.nodes.insert(Value {
            kind,
            ty,
            span,
            alias: None,
            result_origins: Vec::new(),
        })
    }

    /// Allocate a function parameter node.
    fn add_func_param(&mut self, parameter: ParameterId, ty: Lang::Ty) -> ValueId {
        self.insert_node(ValueKind::FuncParam { parameter }, ty, None)
    }

    #[cfg(test)]
    pub(crate) fn add_test_value_parameter(&mut self, index: usize, ty: Lang::Ty) -> ValueId {
        self.add_parameter(ParameterId::new(index), &OperandType::Value(ty))
            .value()
            .expect("value parameter must use the value channel")
    }

    pub fn add_parameter(
        &mut self,
        parameter: ParameterId,
        representation: &OperandType<P::Resource, Lang::Ty>,
    ) -> OperandRef {
        match representation {
            OperandType::Value(ty) => OperandRef::Value(self.add_func_param(parameter, ty.clone())),
            OperandType::View(view) => {
                let value = self.add_func_param(parameter, view.array.clone());
                OperandRef::View(ViewId(value))
            }
            OperandType::Place(place) => {
                OperandRef::Place(self.add_place_parameter(parameter, place.clone()))
            }
        }
    }

    /// Append a parameter to a block and allocate its corresponding node.
    pub fn add_block_param(&mut self, block: BlockId, ty: Lang::Ty) -> ValueId {
        let index = self.skeleton.blocks[block].params.len();
        let id = self.insert_node(ValueKind::BlockParam { block, index }, ty, None);
        let flow = self.admit_flow_value(id);
        self.skeleton.blocks[block].params.push(flow);
        id
    }

    /// Remove parameter slots from a block and from every incoming branch.
    ///
    /// Removed parameter nodes remain in the node sea so a caller can alias
    /// their uses before a later cleanup. Surviving parameter nodes are
    /// renumbered to match their new positions in the block parameter list.
    /// Returns the removed nodes in ascending order of their former slots.
    pub fn remove_block_param_slots(&mut self, block: BlockId, slots: &SortedSet<usize>) -> Vec<ValueId> {
        let param_count = self.skeleton.blocks[block].params.len();
        assert!(
            slots.iter().all(|&slot| slot < param_count),
            "block parameter slot out of bounds"
        );

        let removed = slots.iter().map(|&slot| self.skeleton.blocks[block].params[slot].value()).collect();

        for &slot in slots.iter().rev() {
            self.skeleton.blocks[block].params.remove(slot);
        }

        for (_, predecessor) in self.skeleton.blocks.iter_mut() {
            match &mut predecessor.term {
                crate::flow::Terminator::Branch { target, args } if *target == block => {
                    for &slot in slots.iter().rev() {
                        args.remove(slot);
                    }
                }
                crate::flow::Terminator::CondBranch {
                    then_target,
                    then_args,
                    else_target,
                    else_args,
                    ..
                } => {
                    if *then_target == block {
                        for &slot in slots.iter().rev() {
                            then_args.remove(slot);
                        }
                    }
                    if *else_target == block {
                        for &slot in slots.iter().rev() {
                            else_args.remove(slot);
                        }
                    }
                }
                _ => {}
            }
        }

        let surviving_params = self.skeleton.blocks[block].params.clone();
        for (index, param) in surviving_params.into_iter().enumerate() {
            let param = param.value();
            match &mut self.nodes[param].kind {
                ValueKind::BlockParam {
                    block: owner,
                    index: old_index,
                } if *owner == block => *old_index = index,
                _ => panic!("block parameter list contains a mismatched node"),
            }
        }

        removed
    }

    /// Intern a constant, deduplicating.
    pub fn intern_constant(&mut self, c: Lang::Const, ty: Lang::Ty) -> ValueId {
        if let Some(&existing) = self.const_cache.get(&c) {
            return existing;
        }
        let id = self.insert_node(ValueKind::Constant(c.clone()), ty, None);
        self.const_cache.insert(c, id);
        id
    }

    /// Intern a pure node with an attached source span. The span is recorded
    /// on first intern; subsequent interns of an equivalent hash-consed node
    /// keep the original span.
    pub fn intern_pure(
        &mut self,
        op: PureOp<P::Resource>,
        operands: SmallVec<[ValueId; 4]>,
        ty: Lang::Ty,
        span: Option<Span>,
    ) -> ValueId {
        let key = PureValueKey {
            op: op.clone(),
            operands: operands.clone(),
            ty: ty.clone(),
        };
        if let Some(&existing) = self.hash_cons.get(&key) {
            return existing;
        }
        let id = self.insert_node(ValueKind::Pure { op, operands }, ty, span);
        self.hash_cons.insert(key, id);
        id
    }

    /// Allocate a node for a side-effect result (not hash-consed).
    pub fn alloc_side_effect_result(&mut self, ty: Lang::Ty) -> ValueId {
        self.insert_node(ValueKind::SideEffectResult, ty, None)
    }

    /// Create a union node joining two alternatives.
    pub fn add_union(&mut self, left: ValueId, right: ValueId) -> ValueId {
        // Use the type of the left (they should be equivalent).
        let ty = self.nodes[left].ty.clone();
        self.insert_node(ValueKind::Union { left, right }, ty, None)
    }

    /// Turn a pure node into a union of itself and `alt`, in place: the
    /// original node is re-inserted under a fresh id (returned) and `id`
    /// becomes `Union { fresh, alt }`. Every existing reference to `id` —
    /// pure operands, side-effect slots, terminator args — sees both
    /// alternatives with no rewiring; extraction picks the cheaper side.
    pub fn union_pure_in_place(&mut self, id: ValueId, alt: ValueId) -> ValueId {
        assert_ne!(
            id, alt,
            "union_pure_in_place: alternative must differ from the node"
        );
        debug_assert!(matches!(&self.nodes[id].kind, ValueKind::Pure { .. }));
        let original_kind = self.nodes[id].kind.clone();
        let original_ty = self.nodes[id].ty.clone();
        let original_span = self.nodes[id].span;
        let original_origins = self.nodes[id].result_origins.clone();
        let fresh = self.nodes.insert(Value {
            kind: original_kind,
            ty: original_ty,
            span: original_span,
            alias: None,
            result_origins: original_origins,
        });
        // The hash-cons key for the original node now belongs to its fresh id.
        if let Some(key) = self.pure_node_key(fresh) {
            self.hash_cons.insert(key, fresh);
        }
        self.nodes[id].kind = ValueKind::Union {
            left: fresh,
            right: alt,
        };
        fresh
    }

    /// Discard a pure value in favor of `better`, in place: `id` becomes a
    /// degenerate union both of whose sides are `better`, so extraction can
    /// only pick `better` and existing references follow it. A discarded
    /// pure value's hash-cons key is retired.
    pub fn subsume_pure_in_place(&mut self, id: ValueId, better: ValueId) {
        assert_ne!(
            id, better,
            "subsume_pure_in_place: replacement must differ from the value"
        );
        assert!(matches!(self.nodes[id].kind, ValueKind::Pure { .. }));
        if let Some(key) = self.pure_node_key(id) {
            self.hash_cons.remove(&key);
        }
        self.nodes[id].kind = ValueKind::Union {
            left: better,
            right: better,
        };
    }

    /// Drop structured-control metadata whose header or required target block
    /// was removed by CFG simplification.
    pub fn retain_live_control_headers(&mut self) {
        let invalid = self
            .skeleton
            .blocks
            .iter()
            .filter_map(|(header, block)| {
                let control = block.control_header.as_ref()?;
                let valid = matches!(block.term, crate::flow::Terminator::CondBranch { .. })
                    && match control {
                        ControlHeader::Loop {
                            merge,
                            continue_block,
                        } => {
                            self.skeleton.blocks.contains_key(*merge)
                                && self.skeleton.blocks.contains_key(*continue_block)
                        }
                        ControlHeader::Selection { merge } => self.skeleton.blocks.contains_key(*merge),
                    };
                (!valid).then_some(header)
            })
            .collect::<Vec<_>>();
        for header in invalid {
            self.skeleton.blocks[header].control_header = None;
        }
    }
}

// ---------------------------------------------------------------------------
// Program and body containers
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Func<P: Family, Lang: Language> {
    /// Stable identity used by segmented bodies that call this region.
    pub region: RegionId,
    /// Diagnostic and emitted-symbol metadata; never used to resolve a call.
    pub name: String,
    pub span: Span,
    pub linkage_name: Option<String>,
    /// The single physical parameter sequence. Destination parameters occupy
    /// ordinary positions in this same sequence.
    pub(crate) params: Vec<FuncParam<P::Resource, Lang::Ty>>,
    /// The typed logical result tree and the resolved destination of every
    /// indivisible result.
    pub(crate) result: FunctionResult<Lang::Ty>,
    pub(crate) effects: CallEffects,
    pub graph: EGraph<P, Lang>,
}

impl<P: Family, Lang: Language> Func<P, Lang> {
    pub fn map_graph(mut self, map: impl FnOnce(EGraph<P, Lang>) -> EGraph<P, Lang>) -> Self {
        self.graph = map(self.graph);
        self
    }

    pub fn try_map_graph<E>(
        mut self,
        map: impl FnOnce(EGraph<P, Lang>) -> Result<EGraph<P, Lang>, E>,
    ) -> Result<Self, E> {
        self.graph = map(self.graph)?;
        Ok(self)
    }

    pub fn new(
        region: RegionId,
        name: String,
        span: Span,
        linkage_name: Option<String>,
        params: Vec<FuncParam<P::Resource, Lang::Ty>>,
        result: FunctionResult<Lang::Ty>,
        effects: CallEffects,
        graph: EGraph<P, Lang>,
    ) -> Self {
        Self {
            region,
            name,
            span,
            linkage_name,
            params,
            result,
            effects,
            graph,
        }
    }

    pub fn params(&self) -> &[FuncParam<P::Resource, Lang::Ty>] {
        &self.params
    }

    pub fn result(&self) -> &FunctionResult<Lang::Ty> {
        &self.result
    }

    pub fn effects(&self) -> CallEffects {
        self.effects
    }

    pub fn return_type(&self) -> &Lang::Ty {
        self.result.ty()
    }

    /// Append one value to both sides of a segmented-body capture ABI.
    pub(crate) fn push_seg_body_capture(
        &mut self,
        body: &mut SegBody,
        capture: OperandRef,
        ty: Lang::Ty,
        name: String,
    ) -> ValueId {
        let index = self.params.len();
        let parameter_id = ParameterId::new(index);
        let abi_parameter = callable_parameter::<P::Resource, Lang>(name, ty);
        let parameter = self
            .graph
            .add_parameter(parameter_id, abi_parameter.representation())
            .value()
            .expect("a captured value or view uses the value channel");
        self.params.push(abi_parameter);
        body.captures.push(capture);
        parameter
    }

    /// Retain selected capture slots and compact both sides of the ABI.
    ///
    /// Leading lane/element parameters are always retained. Removed parameter
    /// nodes remain as unique out-of-ABI tombstones because dead pure arena
    /// nodes can still refer to them until demand-driven extraction.
    pub(crate) fn retain_seg_body_captures(
        &mut self,
        body: &mut SegBody,
        retained_captures: &SortedSet<usize>,
    ) -> Result<(), String> {
        let leading = body.leading_parameter_count(self)?;
        let parameter_count = self.params.len();
        if let Some(index) = retained_captures.iter().find(|index| **index >= body.captures.len()) {
            return Err(format!(
                "region `{}` cannot retain missing capture {index}",
                self.name
            ));
        }

        let mut retained = vec![false; parameter_count];
        retained[..leading].fill(true);
        for &capture in retained_captures {
            retained[leading + capture] = true;
        }

        let mut next_index = 0;
        let remapped = retained
            .iter()
            .map(|retain| {
                retain.then(|| {
                    let index = next_index;
                    next_index += 1;
                    index
                })
            })
            .collect::<Vec<_>>();
        let parameter_nodes = self
            .graph
            .nodes
            .iter()
            .filter_map(|(node, definition)| match &definition.kind {
                ValueKind::FuncParam { parameter } => Some((node, parameter.index())),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut tombstone_index = next_index;
        for (node, old_index) in parameter_nodes {
            if let Some(new_index) = remapped.get(old_index).copied().flatten() {
                self.graph.nodes[node].kind = ValueKind::FuncParam {
                    parameter: ParameterId::new(new_index),
                };
            } else {
                self.graph.nodes[node].kind = ValueKind::FuncParam {
                    parameter: ParameterId::new(tombstone_index),
                };
                tombstone_index += 1;
            }
        }

        self.params = std::mem::take(&mut self.params)
            .into_iter()
            .enumerate()
            .filter_map(|(index, parameter)| retained[index].then_some(parameter))
            .collect();
        body.captures = std::mem::take(&mut body.captures)
            .into_iter()
            .enumerate()
            .filter_map(|(index, capture)| retained[leading + index].then_some(capture))
            .collect();
        Ok(())
    }
}

/// A body-backed compile-time constant retained in EGIR until final
/// elaboration. Constant bodies have no parameters and must be proven pure.
#[derive(Clone, Debug)]
pub struct ConstantDef<P: Family, Lang: Language> {
    pub id: crate::GlobalId,
    /// Diagnostic and emitted-symbol metadata; never used to resolve a global.
    pub name: String,
    pub span: Span,
    pub return_ty: Lang::Ty,
    pub graph: EGraph<P, Lang>,
}

impl<P: Family, Lang: Language> ConstantDef<P, Lang> {
    pub fn map_graph(mut self, map: impl FnOnce(EGraph<P, Lang>) -> EGraph<P, Lang>) -> Self {
        self.graph = map(self.graph);
        self
    }

    pub fn try_map_graph<E>(
        mut self,
        map: impl FnOnce(EGraph<P, Lang>) -> Result<EGraph<P, Lang>, E>,
    ) -> Result<Self, E> {
        self.graph = map(self.graph)?;
        Ok(self)
    }
}

/// One write site for an entry output slot.
#[derive(Debug, Clone, Copy)]
pub struct SlotSource {
    pub block: BlockId,
    pub value: ValueId,
}

/// Stable identity of a declared entry-output position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OutputSlotId(pub usize);

/// The concrete side effect that fulfils an output route after realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputWriter {
    Value(ValueId),
    Effect(EffectToken),
}

/// An output route whose concrete writers have been installed in the graph.
#[derive(Debug, Clone)]
pub struct RealizedOutputRoute {
    pub source: SlotSource,
    pub writers: Vec<OutputWriter>,
}

impl RealizedOutputRoute {
    pub fn referenced_values(&self) -> impl Iterator<Item = ValueId> + '_ {
        std::iter::once(self.source.value).chain(self.writers.iter().filter_map(|writer| match writer {
            OutputWriter::Value(value) => Some(*value),
            OutputWriter::Effect(_) => None,
        }))
    }

    pub fn replace_values(&mut self, replacements: &[(ValueId, ValueId)]) {
        let replace = |value| {
            replacements
                .iter()
                .find_map(|(source, replacement)| (*source == value).then_some(*replacement))
                .unwrap_or(value)
        };
        self.source.value = replace(self.source.value);
        for writer in &mut self.writers {
            if let OutputWriter::Value(value) = writer {
                *value = replace(*value);
            }
        }
    }
}

pub trait RemapBlockIds {
    fn remap_block_ids(&mut self, blocks: &LookupMap<BlockId, BlockId>);
}

impl RemapBlockIds for RealizedOutputRoute {
    fn remap_block_ids(&mut self, blocks: &LookupMap<BlockId, BlockId>) {
        self.source.block = blocks[&self.source.block];
    }
}

/// One entry input together with its phase-typed resource identity, when the
/// slot is backed by a logical or physical resource.
#[derive(Debug, Clone)]
pub struct EntryInput<R, Lang: Language> {
    pub inner: InterfaceEntryInput<Lang::Ty>,
    pub resource: Option<R>,
}

impl<R, Lang: Language> std::ops::Deref for EntryInput<R, Lang> {
    type Target = InterfaceEntryInput<Lang::Ty>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<R, Lang: Language> std::ops::DerefMut for EntryInput<R, Lang> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// One entry output together with its phase-typed resource identity, when the
/// slot is backed by a logical or physical resource.
#[derive(Debug, Clone)]
pub struct EntryOutput<R, Route, Lang: Language> {
    pub inner: InterfaceEntryOutput<Lang::Ty>,
    pub resource: Option<R>,
    /// Every control-flow source capable of producing this declared output.
    pub routes: Vec<Route>,
}

/// A compiler-materialized result routed to an internal resource. Keeping
/// this outside `EntryOutput` makes it impossible to publish as host ABI.
#[derive(Debug, Clone)]
pub struct InternalResultRoute<R, Route> {
    pub resource: R,
    pub route: Route,
}

impl<R, Route, Lang: Language> std::ops::Deref for EntryOutput<R, Route, Lang> {
    type Target = InterfaceEntryOutput<Lang::Ty>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<R, Route, Lang: Language> std::ops::DerefMut for EntryOutput<R, Route, Lang> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Clone, Debug)]
pub struct Entry<P: Family, ResourceDecl, Route, Lang: Language> {
    pub id: crate::EntryId,
    /// Host-facing entry symbol and diagnostic metadata.
    pub name: String,
    pub span: Span,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput<P::Resource, Lang>>,
    /// Structural association from source parameter to its ABI input slots.
    /// A parameter may expand into multiple slots (for example tuple views).
    pub parameter_inputs: Vec<Vec<super::program::InputSlotId>>,
    pub outputs: Vec<EntryOutput<P::Resource, Route, Lang>>,
    pub internal_results: Vec<InternalResultRoute<P::Resource, Route>>,
    pub resource_declarations: Vec<ResourceDecl>,
    pub(crate) params: Vec<FuncParam<P::Resource, Lang::Ty>>,
    pub(crate) result: FunctionResult<Lang::Ty>,
    pub graph: EGraph<P, Lang>,
}

impl<P: Family, ResourceDecl: Clone, Route: Clone, Lang: Language> Entry<P, ResourceDecl, Route, Lang> {
    pub fn routes(&self) -> impl Iterator<Item = &Route> {
        self.outputs
            .iter()
            .flat_map(|output| &output.routes)
            .chain(self.internal_results.iter().map(|result| &result.route))
    }

    pub fn routes_mut(&mut self) -> impl Iterator<Item = &mut Route> {
        self.outputs
            .iter_mut()
            .flat_map(|output| &mut output.routes)
            .chain(self.internal_results.iter_mut().map(|result| &mut result.route))
    }

    pub fn resource_routes(&self) -> impl Iterator<Item = (&P::Resource, &Route)> {
        self.outputs
            .iter()
            .flat_map(|output| {
                output
                    .resource
                    .iter()
                    .flat_map(move |resource| output.routes.iter().map(move |route| (resource, route)))
            })
            .chain(self.internal_results.iter().map(|result| (&result.resource, &result.route)))
    }

    pub fn map_graph(mut self, map: impl FnOnce(EGraph<P, Lang>) -> EGraph<P, Lang>) -> Self {
        self.graph = map(self.graph);
        self
    }

    pub fn try_map_graph<E>(
        mut self,
        map: impl FnOnce(EGraph<P, Lang>) -> Result<EGraph<P, Lang>, E>,
    ) -> Result<Self, E> {
        self.graph = map(self.graph)?;
        Ok(self)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_resources(
        name: String,
        id: crate::EntryId,
        span: Span,
        execution_model: ExecutionModel,
        inputs: Vec<InterfaceEntryInput<Lang::Ty>>,
        outputs: Vec<InterfaceEntryOutput<Lang::Ty>>,
        resource_declarations: Vec<ResourceDecl>,
        params: Vec<FuncParam<P::Resource, Lang::Ty>>,
        result: FunctionResult<Lang::Ty>,
        graph: EGraph<P, Lang>,
    ) -> Self {
        let parameter_inputs = (0..params.len())
            .map(|index| {
                (index < inputs.len()).then(|| vec![super::program::InputSlotId(index)]).unwrap_or_default()
            })
            .collect();
        Self {
            id,
            name,
            span,
            execution_model,
            inputs: inputs
                .into_iter()
                .map(|inner| EntryInput {
                    inner,
                    resource: None,
                })
                .collect(),
            parameter_inputs,
            outputs: outputs
                .into_iter()
                .map(|inner| EntryOutput {
                    inner,
                    resource: None,
                    routes: Vec::new(),
                })
                .collect(),
            internal_results: Vec::new(),
            resource_declarations,
            params,
            result,
            graph,
        }
    }

    pub fn params(&self) -> &[FuncParam<P::Resource, Lang::Ty>] {
        &self.params
    }

    pub fn result(&self) -> &FunctionResult<Lang::Ty> {
        &self.result
    }

    pub fn return_type(&self) -> &Lang::Ty {
        self.result.ty()
    }

    /// Retain selected original parameter indices and compact the entry
    /// interface and corresponding function-parameter nodes together.
    pub fn retain_parameter_indices(&mut self, retained: &SortedSet<usize>) {
        let mut kept = self
            .graph
            .nodes
            .iter()
            .filter_map(|(node, definition)| match &definition.kind {
                ValueKind::FuncParam { parameter } if retained.contains(&parameter.index()) => {
                    Some((parameter.index(), node))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        kept.sort_by_key(|(index, _)| *index);
        kept.dedup_by_key(|(index, _)| *index);

        let kept_slots = kept
            .iter()
            .flat_map(|(index, _)| self.parameter_inputs[*index].iter().copied())
            .collect::<LookupSet<_>>();
        let mut remapped_slots = LookupMap::new();
        self.inputs = std::mem::take(&mut self.inputs)
            .into_iter()
            .enumerate()
            .filter_map(|(old_index, input)| {
                let old_slot = super::program::InputSlotId(old_index);
                kept_slots.contains(&old_slot).then(|| {
                    let new_slot = super::program::InputSlotId(remapped_slots.len());
                    remapped_slots.insert(old_slot, new_slot);
                    input
                })
            })
            .collect();
        self.parameter_inputs = kept
            .iter()
            .map(|(index, _)| {
                self.parameter_inputs[*index].iter().map(|slot| remapped_slots[slot]).collect()
            })
            .collect();
        self.params = kept.iter().map(|(index, _)| self.params[*index].clone()).collect();

        let retained_nodes = kept.iter().map(|(_, node)| *node).collect::<LookupSet<_>>();
        let removed = self
            .graph
            .nodes
            .iter()
            .filter_map(|(node, definition)| {
                (matches!(&definition.kind, ValueKind::FuncParam { .. }) && !retained_nodes.contains(&node))
                    .then_some(node)
            })
            .collect::<Vec<_>>();
        for node in removed {
            self.graph.remove_func_param(node);
        }
        for (new_index, (_, node)) in kept.into_iter().enumerate() {
            if let Some(ValueKind::FuncParam { parameter }) =
                self.graph.nodes.get_mut(node).map(|node| &mut node.kind)
            {
                *parameter = ParameterId::new(new_index);
            }
        }
    }

    /// Drop structured-control metadata whose header or required target block
    /// was removed by CFG simplification.
    pub fn retain_live_control_headers(&mut self) {
        self.graph.retain_live_control_headers();
    }
}

impl<P: Family, ResourceDecl, Route, Lang: Language> Entry<P, ResourceDecl, Route, Lang> {
    /// Consume an entry while changing only the representation of each
    /// output route.
    pub fn map_output_routes<T>(self, mut map: impl FnMut(Route) -> T) -> Entry<P, ResourceDecl, T, Lang> {
        let Self {
            name,
            id,
            span,
            execution_model,
            inputs,
            parameter_inputs,
            outputs,
            internal_results,
            resource_declarations,
            params,
            result,
            graph,
        } = self;
        Entry {
            id,
            name,
            span,
            execution_model,
            inputs,
            parameter_inputs,
            outputs: outputs
                .into_iter()
                .map(|output| EntryOutput {
                    inner: output.inner,
                    resource: output.resource,
                    routes: output.routes.into_iter().map(&mut map).collect(),
                })
                .collect(),
            internal_results: internal_results
                .into_iter()
                .map(|result| InternalResultRoute {
                    resource: result.resource,
                    route: map(result.route),
                })
                .collect(),
            resource_declarations,
            params,
            result,
            graph,
        }
    }
}

/// Whole-program EGIR container at one externally visible checkpoint.
#[derive(Debug)]
pub struct Program<Tag, Shape: ProgramShape, GlobalContext, Lang: Language> {
    pub functions: Vec<Func<Shape::Family, Lang>>,
    pub externs: Vec<ExternDecl<Lang::Ty>>,
    pub entry_points: Vec<Entry<Shape::Family, Shape::ResourceDecl, Shape::OutputRoute, Lang>>,
    pub constants: Vec<ConstantDef<Shape::Family, Lang>>,
    /// Program-owned IR data selected by this checkpoint.
    pub data: Shape::ProgramData,
    /// Program-wide state available at this exact pipeline checkpoint.
    pub global_context: GlobalContext,
    pub(crate) state: std::marker::PhantomData<fn() -> Tag>,
}

/// Stable address of one graph-bearing body within a program snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BodySite {
    Function(RegionId),
    Entry(usize),
    Constant(usize),
}

/// One graph-bearing program member removed from a program for a consuming
/// rewrite.
pub enum Body<P: Family, ResourceDecl, OutputRoute, Lang: Language> {
    Function(Func<P, Lang>),
    Entry(Entry<P, ResourceDecl, OutputRoute, Lang>),
    Constant(ConstantDef<P, Lang>),
}

impl<Tag, Shape: ProgramShape, GlobalContext, Lang: Language> Program<Tag, Shape, GlobalContext, Lang> {
    pub fn from_parts(
        functions: Vec<Func<Shape::Family, Lang>>,
        externs: Vec<ExternDecl<Lang::Ty>>,
        entry_points: Vec<Entry<Shape::Family, Shape::ResourceDecl, Shape::OutputRoute, Lang>>,
        constants: Vec<ConstantDef<Shape::Family, Lang>>,
        data: Shape::ProgramData,
        global_context: GlobalContext,
    ) -> Self {
        Self {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state: std::marker::PhantomData,
        }
    }

    pub fn body_graph(&self, site: BodySite) -> Option<&EGraph<Shape::Family, Lang>> {
        match site {
            BodySite::Function(region) => {
                self.functions.iter().find(|function| function.region == region).map(|body| &body.graph)
            }
            BodySite::Entry(index) => self.entry_points.get(index).map(|body| &body.graph),
            BodySite::Constant(index) => self.constants.get(index).map(|body| &body.graph),
        }
    }

    /// Change only the top-level proof tag while preserving all stored types.
    pub fn retag<NewTag>(self) -> Program<NewTag, Shape, GlobalContext, Lang> {
        let Self {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state: _,
        } = self;
        Program {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state: std::marker::PhantomData,
        }
    }

    /// Consume and rebuild every graph-bearing body while moving all
    /// non-graph fields directly into the resulting program.
    pub fn map_graphs(
        self,
        mut map: impl FnMut(BodySite, EGraph<Shape::Family, Lang>) -> EGraph<Shape::Family, Lang>,
    ) -> Self {
        match self.try_map_graphs(|site, graph| Ok::<_, std::convert::Infallible>(map(site, graph))) {
            Ok(program) => program,
            Err(error) => match error {},
        }
    }

    /// Fallible counterpart to [`Self::map_graphs`].
    pub fn try_map_graphs<E>(
        self,
        mut map: impl FnMut(BodySite, EGraph<Shape::Family, Lang>) -> Result<EGraph<Shape::Family, Lang>, E>,
    ) -> Result<Self, E> {
        let Self {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state,
        } = self;
        let functions = functions
            .into_iter()
            .map(|function| {
                let site = BodySite::Function(function.region);
                function.try_map_graph(|graph| map(site, graph))
            })
            .collect::<Result<_, E>>()?;
        let entry_points = entry_points
            .into_iter()
            .enumerate()
            .map(|(index, entry)| entry.try_map_graph(|graph| map(BodySite::Entry(index), graph)))
            .collect::<Result<_, E>>()?;
        let constants = constants
            .into_iter()
            .enumerate()
            .map(|(index, constant)| constant.try_map_graph(|graph| map(BodySite::Constant(index), graph)))
            .collect::<Result<_, E>>()?;
        Ok(Self {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state,
        })
    }

    /// Fallibly rebuild every graph-bearing body while allowing the rewrite
    /// to update both program-owned data and carried global state.
    pub fn try_map_graphs_with_state<E>(
        self,
        mut map: impl FnMut(
            BodySite,
            EGraph<Shape::Family, Lang>,
            &mut Shape::ProgramData,
            &mut GlobalContext,
        ) -> Result<EGraph<Shape::Family, Lang>, E>,
    ) -> Result<Self, E> {
        let Self {
            functions,
            externs,
            entry_points,
            constants,
            mut data,
            mut global_context,
            state,
        } = self;
        let functions = functions
            .into_iter()
            .map(|function| {
                let site = BodySite::Function(function.region);
                function.try_map_graph(|graph| map(site, graph, &mut data, &mut global_context))
            })
            .collect::<Result<_, E>>()?;
        let entry_points = entry_points
            .into_iter()
            .enumerate()
            .map(|(index, entry)| {
                entry.try_map_graph(|graph| {
                    map(BodySite::Entry(index), graph, &mut data, &mut global_context)
                })
            })
            .collect::<Result<_, E>>()?;
        let constants = constants
            .into_iter()
            .enumerate()
            .map(|(index, constant)| {
                constant.try_map_graph(|graph| {
                    map(BodySite::Constant(index), graph, &mut data, &mut global_context)
                })
            })
            .collect::<Result<_, E>>()?;
        Ok(Self {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state,
        })
    }

    /// Consume the program and append synthesized callable regions while
    /// moving every existing member unchanged.
    pub fn extend_functions(self, additional: impl IntoIterator<Item = Func<Shape::Family, Lang>>) -> Self {
        let Self {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state,
        } = self;
        Self {
            functions: functions.into_iter().chain(additional).collect(),
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state,
        }
    }

    /// Consume the program and rebuild only its program-owned data.
    pub fn map_data(self, map: impl FnOnce(Shape::ProgramData) -> Shape::ProgramData) -> Self {
        let Self {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state,
        } = self;
        Self {
            functions,
            externs,
            entry_points,
            constants,
            data: map(data),
            global_context,
            state,
        }
    }

    /// Consume and replace exactly one graph-bearing body. All other bodies
    /// move directly into the rebuilt program.
    pub fn rewrite_body(
        self,
        site: BodySite,
        rewrite: impl FnOnce(
            Body<Shape::Family, Shape::ResourceDecl, Shape::OutputRoute, Lang>,
        ) -> Body<Shape::Family, Shape::ResourceDecl, Shape::OutputRoute, Lang>,
    ) -> Self {
        match self.try_rewrite_body(site, |body| Ok::<_, std::convert::Infallible>(rewrite(body))) {
            Ok(program) => program,
            Err(error) => match error {},
        }
    }

    /// Fallible counterpart to [`Self::rewrite_body`].
    pub fn try_rewrite_body<E>(
        self,
        site: BodySite,
        rewrite: impl FnOnce(
            Body<Shape::Family, Shape::ResourceDecl, Shape::OutputRoute, Lang>,
        )
            -> Result<Body<Shape::Family, Shape::ResourceDecl, Shape::OutputRoute, Lang>, E>,
    ) -> Result<Self, E> {
        let Self {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state,
        } = self;
        let mut rewrite = Some(rewrite);
        let mut rebuilt_functions = Vec::with_capacity(functions.len());
        for function in functions {
            if site != BodySite::Function(function.region) {
                rebuilt_functions.push(function);
                continue;
            }
            match rewrite.take().expect("body patch applied more than once")(Body::Function(function))? {
                Body::Function(function) => rebuilt_functions.push(function),
                _ => panic!("function body patch returned a different body kind"),
            }
        }
        let mut rebuilt_entries = Vec::with_capacity(entry_points.len());
        for (index, entry) in entry_points.into_iter().enumerate() {
            if site != BodySite::Entry(index) {
                rebuilt_entries.push(entry);
                continue;
            }
            match rewrite.take().expect("body patch applied more than once")(Body::Entry(entry))? {
                Body::Entry(entry) => rebuilt_entries.push(entry),
                _ => panic!("entry body patch returned a different body kind"),
            }
        }
        let mut rebuilt_constants = Vec::with_capacity(constants.len());
        for (index, constant) in constants.into_iter().enumerate() {
            if site != BodySite::Constant(index) {
                rebuilt_constants.push(constant);
                continue;
            }
            match rewrite.take().expect("body patch applied more than once")(Body::Constant(constant))? {
                Body::Constant(constant) => rebuilt_constants.push(constant),
                _ => panic!("constant body patch returned a different body kind"),
            }
        }
        assert!(
            rewrite.is_none(),
            "body patch targeted a body absent from the program"
        );
        Ok(Self {
            functions: rebuilt_functions,
            externs,
            entry_points: rebuilt_entries,
            constants: rebuilt_constants,
            data,
            global_context,
            state,
        })
    }
}
