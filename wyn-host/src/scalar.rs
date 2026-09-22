//! Sequential scalar expressions shared by the host backends.
use crate::{Binding, HostError, Program, ResourceId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarType {
    Bool,
    I32,
    U32,
    F32,
}

impl ScalarType {
    pub fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::F32 => "f32",
        }
    }
}

/// A physical input before target-specific push-constant legalization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScalarSource {
    Binding {
        set: u32,
        binding: u32,
    },
    PushConstant {
        name: String,
        offset: u32,
    },
}

/// Expressions retain lexical binding and conditional evaluation boundaries.
/// `Loop` tests before each iteration and evaluates its step in the old scope.
#[derive(Clone, Debug)]
pub enum ScalarExpr {
    I32(i32),
    U32(u32),
    F32(u32),
    Bool(bool),
    Local(String),
    Read {
        source: ScalarSource,
        offset: u32,
        ty: ScalarType,
    },
    Apply {
        op: String,
        /// Operand type; comparisons return a boolean.
        ty: ScalarType,
        args: Vec<ScalarExpr>,
    },
    If {
        condition: Box<Self>,
        yes: Box<Self>,
        no: Box<Self>,
    },
    Let {
        bindings: Vec<(String, Self)>,
        result: Box<Self>,
    },
    Loop {
        name: String,
        initial: Box<Self>,
        condition: Box<Self>,
        step: Box<Self>,
    },
    Tuple(Vec<Self>),
    Field {
        tuple: Box<Self>,
        index: usize,
    },
}

/// One ordered host evaluation and the scalar word it supplies to device work.
#[derive(Clone, Debug)]
pub struct ScalarTask {
    pub stage: String,
    pub destination: ScalarSource,
    pub offset: u32,
    pub ty: ScalarType,
    pub value: ScalarExpr,
    pub replaces_dispatch: bool,
}

impl ScalarExpr {
    pub fn reads_mut(&mut self, visit: &mut impl FnMut(&mut ScalarSource, &mut u32)) {
        match self {
            Self::Read { source, offset, .. } => visit(source, offset),
            Self::Apply { args, .. } | Self::Tuple(args) => {
                for arg in args {
                    arg.reads_mut(visit);
                }
            }
            Self::If { condition, yes, no } => {
                condition.reads_mut(visit);
                yes.reads_mut(visit);
                no.reads_mut(visit);
            }
            Self::Let { bindings, result } => {
                for (_, value) in bindings {
                    value.reads_mut(visit);
                }
                result.reads_mut(visit);
            }
            Self::Loop {
                initial,
                condition,
                step,
                ..
            } => {
                initial.reads_mut(visit);
                condition.reads_mut(visit);
                step.reads_mut(visit);
            }
            Self::Field { tuple, .. } => tuple.reads_mut(visit),
            Self::I32(_) | Self::U32(_) | Self::F32(_) | Self::Bool(_) | Self::Local(_) => {}
        }
    }
}

impl Program {
    pub(crate) fn scalar_resource(
        &self,
        pipeline: usize,
        source: &ScalarSource,
    ) -> Result<ResourceId, HostError> {
        match source {
            ScalarSource::Binding { set, binding } => self.slot_resource(pipeline, *set, *binding),
            ScalarSource::PushConstant { name, offset } => {
                let Some(index) = self.bindings(pipeline).iter().position(|b| {
                    matches!(b,
                    Binding::PushConstant { name: n, offset: o, .. } if n == name && o == offset)
                }) else {
                    return Err(HostError::Invalid(format!("missing scalar input {name}")));
                };
                self.binding_resource(pipeline, index)
            }
        }
    }
}
