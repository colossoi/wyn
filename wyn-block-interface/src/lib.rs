//! Parameter columns over an immutable sequence of incoming edge identities.
use std::collections::HashSet;
use std::hash::Hash;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix<E, V> {
    edges: Vec<E>,
    columns: Vec<Column<V>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Column<V> {
    parameter: V,
    arguments: Vec<V>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    DuplicateEdge,
    DuplicateParameter,
    Arity,
    InvalidSelection,
}

impl<V: Copy + Eq> Column<V> {
    pub fn parameter(&self) -> V {
        self.parameter
    }

    pub fn arguments(&self) -> &[V] {
        &self.arguments
    }

    /// An empty column has no common incoming value.
    pub fn common_argument(&self) -> Option<V> {
        let first = *self.arguments.first()?;
        self.arguments.iter().all(|value| *value == first).then_some(first)
    }
}

impl<E: Copy + Eq + Hash, V: Copy + Eq + Hash> Matrix<E, V> {
    pub fn new(
        parameters: impl IntoIterator<Item = V>,
        rows: impl IntoIterator<Item = (E, Vec<V>)>,
    ) -> Result<Self, Error> {
        let mut parameters_seen = HashSet::new();
        let mut columns = parameters
            .into_iter()
            .map(|parameter| {
                if !parameters_seen.insert(parameter) {
                    return Err(Error::DuplicateParameter);
                }
                Ok(Column {
                    parameter,
                    arguments: Vec::new(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut edges = Vec::new();
        let mut edges_seen = HashSet::new();
        for (edge, arguments) in rows {
            if !edges_seen.insert(edge) {
                return Err(Error::DuplicateEdge);
            }
            if arguments.len() != columns.len() {
                return Err(Error::Arity);
            }
            edges.push(edge);
            for (column, argument) in columns.iter_mut().zip(arguments) {
                column.arguments.push(argument);
            }
        }
        Ok(Self { edges, columns })
    }

    pub fn edges(&self) -> &[E] {
        &self.edges
    }

    pub fn columns(&self) -> &[Column<V>] {
        &self.columns
    }

    pub fn parameters(&self) -> impl Iterator<Item = V> + '_ {
        self.columns.iter().map(Column::parameter)
    }

    /// Select whole columns in the requested order, preserving every edge.
    pub fn select(&self, slots: impl IntoIterator<Item = usize>) -> Result<Self, Error> {
        let mut seen = HashSet::new();
        let columns = slots
            .into_iter()
            .map(|slot| {
                if !seen.insert(slot) {
                    return Err(Error::InvalidSelection);
                }
                self.columns.get(slot).cloned().ok_or(Error::InvalidSelection)
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            edges: self.edges.clone(),
            columns,
        })
    }

    pub fn rows(&self) -> impl Iterator<Item = (E, Vec<V>)> + '_ {
        self.edges.iter().enumerate().map(|(row, edge)| {
            (
                *edge,
                self.columns.iter().map(|column| column.arguments[row]).collect(),
            )
        })
    }
}

#[cfg(test)]
mod tests;
