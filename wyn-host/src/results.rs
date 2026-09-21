//! Source result types and their storage-buffer layouts.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultKind {
    Value,
    RecordField,
    TupleField,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultScalar {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
}

impl ResultScalar {
    pub fn bytes(self) -> u32 {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 | Self::Bool => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResultField {
    pub name: String,
    pub offset: u32,
    pub layout: ResultLayout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResultLayout {
    Scalar(ResultScalar),
    /// Fixed-size vectors, matrix columns, and embedded arrays.
    Sequence {
        element: Box<ResultLayout>,
        count: u32,
        stride: u32,
    },
    Record {
        fields: Vec<ResultField>,
        size: u32,
    },
    Tuple {
        fields: Vec<ResultField>,
        size: u32,
    },
    /// A returned array view. Readers require its logical element range in
    /// the backing buffer; allocation capacity does not determine that range.
    Array {
        element: Box<ResultLayout>,
        stride: u32,
        length: Option<u32>,
    },
    /// Keep unsupported source types descriptive without guessing a decoder.
    Unsupported(String),
}

impl ResultLayout {
    pub fn byte_size(&self) -> Option<u32> {
        match self {
            Self::Scalar(scalar) => Some(scalar.bytes()),
            Self::Sequence { count, stride, .. } => count.checked_mul(*stride),
            Self::Record { size, .. } | Self::Tuple { size, .. } => Some(*size),
            Self::Array { .. } | Self::Unsupported(_) => None,
        }
    }
}
