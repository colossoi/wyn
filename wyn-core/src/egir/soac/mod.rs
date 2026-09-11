pub mod filter;
pub mod hist;
pub(crate) mod lambda;
pub(crate) mod metadata;
pub(crate) mod remap;
pub mod screma;

pub use lambda::{Lambda, LambdaBody};

/// Segmented iteration and publication metadata shared by Screma and Filter.
#[derive(Clone, Debug)]
pub struct SegmentedMetadata<R> {
    pub space: super::types::SegSpace<R>,
    /// Host-visible slots linked from entry output routes during reification.
    pub output_slots: Vec<super::program::OutputSlotId>,
    /// Semantic resource effects, including publication writes.
    pub resources: Vec<super::types::SegResourceAccess<R>>,
}
