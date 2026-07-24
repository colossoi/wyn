//! Concrete payloads stored on phase-typed TLC nodes.
//!
//! Pass modules select these payloads through their `Family`
//! implementations, but the payload types themselves are part of the TLC
//! representation and live here so consumers do not depend on producing
//! passes.

use crate::pipeline_descriptor::BufferLen;
use crate::{LookupMap, SymbolId};

/// No payload is stored at this position in the selected phase.
///
/// This is a type-constructor adapter, not a marker stored in the tree:
/// `Empty::With<T>` is the actual field type `()`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Empty;

impl super::Payload for Empty {
    type With<T: Clone + std::fmt::Debug> = ();

    fn map<T, U, M>((): (), _map: &mut M)
    where
        T: Clone + std::fmt::Debug,
        U: Clone + std::fmt::Debug,
        M: FnMut(T) -> U,
    {
    }

    fn for_each<T, V>((): &(), _visit: &mut V)
    where
        T: Clone + std::fmt::Debug,
        V: FnMut(&T),
    {
    }

    fn for_each_rev<T, V>((): &(), _visit: &mut V)
    where
        T: Clone + std::fmt::Debug,
        V: FnMut(&T),
    {
    }

    fn for_each_mut<T, V>((): &mut (), _visit: &mut V)
    where
        T: Clone + std::fmt::Debug,
        V: FnMut(&mut T),
    {
    }
}

/// Per-definition information retained until monomorphization consumes it.
#[derive(Debug, Clone)]
pub struct PolymorphicDefinition {
    pub scheme: Option<crate::types::TypeScheme>,
}

/// A closure value whose environment is intrinsically owned by its term.
#[derive(Debug, Clone)]
pub struct ExplicitClosure<T> {
    pub code: SymbolId,
    pub captures: Vec<T>,
    pub param_count: usize,
}

/// Selects [`ExplicitClosure`] as a recursively mapped closure payload.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExplicitClosurePayload;

impl super::Payload for ExplicitClosurePayload {
    type With<T: Clone + std::fmt::Debug> = ExplicitClosure<T>;

    fn map<T, U, M>(data: ExplicitClosure<T>, map: &mut M) -> ExplicitClosure<U>
    where
        T: Clone + std::fmt::Debug,
        U: Clone + std::fmt::Debug,
        M: FnMut(T) -> U,
    {
        ExplicitClosure {
            code: data.code,
            captures: data.captures.into_iter().map(map).collect(),
            param_count: data.param_count,
        }
    }

    fn for_each<T, V>(data: &ExplicitClosure<T>, visit: &mut V)
    where
        T: Clone + std::fmt::Debug,
        V: FnMut(&T),
    {
        data.captures.iter().for_each(visit);
    }

    fn for_each_rev<T, V>(data: &ExplicitClosure<T>, visit: &mut V)
    where
        T: Clone + std::fmt::Debug,
        V: FnMut(&T),
    {
        data.captures.iter().rev().for_each(visit);
    }

    fn for_each_mut<T, V>(data: &mut ExplicitClosure<T>, visit: &mut V)
    where
        T: Clone + std::fmt::Debug,
        V: FnMut(&mut T),
    {
        data.captures.iter_mut().for_each(visit);
    }
}

/// Explicit capture ABI stored on a closure-converted SOAC body.
#[derive(Debug, Clone)]
pub struct ExplicitCaptures<T> {
    pub captures: Vec<T>,
}

/// Selects [`ExplicitCaptures`] as recursively mapped SOAC-body data.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExplicitCapturesPayload;

impl super::Payload for ExplicitCapturesPayload {
    type With<T: Clone + std::fmt::Debug> = ExplicitCaptures<T>;

    fn map<T, U, M>(data: ExplicitCaptures<T>, map: &mut M) -> ExplicitCaptures<U>
    where
        T: Clone + std::fmt::Debug,
        U: Clone + std::fmt::Debug,
        M: FnMut(T) -> U,
    {
        ExplicitCaptures {
            captures: data.captures.into_iter().map(map).collect(),
        }
    }

    fn for_each<T, V>(data: &ExplicitCaptures<T>, visit: &mut V)
    where
        T: Clone + std::fmt::Debug,
        V: FnMut(&T),
    {
        data.captures.iter().for_each(visit);
    }

    fn for_each_rev<T, V>(data: &ExplicitCaptures<T>, visit: &mut V)
    where
        T: Clone + std::fmt::Debug,
        V: FnMut(&T),
    {
        data.captures.iter().rev().for_each(visit);
    }

    fn for_each_mut<T, V>(data: &mut ExplicitCaptures<T>, visit: &mut V)
    where
        T: Clone + std::fmt::Debug,
        V: FnMut(&mut T),
    {
        data.captures.iter_mut().for_each(visit);
    }
}

/// Minimum required storage-input lengths attached directly to an entry.
#[derive(Debug, Clone, Default)]
pub struct EntryInputBounds {
    pub by_symbol: LookupMap<SymbolId, BufferLen>,
}
