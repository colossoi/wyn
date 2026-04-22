use crate::ast::Span;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompilerError {
    #[error("Parse error: {0}")]
    ParseError(String, Option<Span>),

    #[error("Type error: {0}")]
    TypeError(String, Option<Span>),

    #[error("Undefined variable '{0}'")]
    UndefinedVariable(String, Option<Span>),

    #[error("Alias error: {0}")]
    AliasError(String, Option<Span>),

    #[error("SPIR-V generation error: {0}")]
    SpirvError(String, Option<Span>),

    #[error("GLSL generation error: {0}")]
    GlslError(String, Option<Span>),

    #[error("WGSL generation error: {0}")]
    WgslError(String, Option<Span>),

    #[error("Module system error: {0}")]
    ModuleError(String, Option<Span>),

    #[error("Flattening error: {0}")]
    FlatteningError(String, Option<Span>),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("SPIR-V builder error: {0}")]
    SpirvBuilderError(#[from] rspirv::dr::Error),
}

impl CompilerError {
    pub fn span(&self) -> Option<Span> {
        match self {
            Self::ParseError(_, span) => *span,
            Self::TypeError(_, span) => *span,
            Self::UndefinedVariable(_, span) => *span,
            Self::AliasError(_, span) => *span,
            Self::SpirvError(_, span) => *span,
            Self::GlslError(_, span) => *span,
            Self::WgslError(_, span) => *span,
            Self::ModuleError(_, span) => *span,
            Self::FlatteningError(_, span) => *span,
            Self::IoError(_) | Self::SpirvBuilderError(_) => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, CompilerError>;

// Error creation macros without span

#[macro_export]
macro_rules! err_parse {
    ($($arg:tt)*) => {
        $crate::error::CompilerError::ParseError(format!($($arg)*), None)
    };
}

#[macro_export]
macro_rules! err_type {
    ($($arg:tt)*) => {
        $crate::error::CompilerError::TypeError(format!($($arg)*), None)
    };
}

#[macro_export]
macro_rules! err_undef {
    ($($arg:tt)*) => {
        $crate::error::CompilerError::UndefinedVariable(format!($($arg)*), None)
    };
}

#[macro_export]
macro_rules! err_spirv {
    ($($arg:tt)*) => {
        $crate::error::CompilerError::SpirvError(format!($($arg)*), None)
    };
}

#[macro_export]
macro_rules! err_glsl {
    ($($arg:tt)*) => {
        $crate::error::CompilerError::GlslError(format!($($arg)*), None)
    };
}

#[macro_export]
macro_rules! err_wgsl {
    ($($arg:tt)*) => {
        $crate::error::CompilerError::WgslError(format!($($arg)*), None)
    };
}

#[macro_export]
macro_rules! err_module {
    ($($arg:tt)*) => {
        $crate::error::CompilerError::ModuleError(format!($($arg)*), None)
    };
}

#[macro_export]
macro_rules! err_flatten {
    ($($arg:tt)*) => {
        $crate::error::CompilerError::FlatteningError(format!($($arg)*), None)
    };
}

#[macro_export]
macro_rules! err_alias {
    ($($arg:tt)*) => {
        $crate::error::CompilerError::AliasError(format!($($arg)*), None)
    };
}

// Error creation macros with span

#[macro_export]
macro_rules! err_parse_at {
    ($span:expr, $($arg:tt)*) => {
        $crate::error::CompilerError::ParseError(format!($($arg)*), Some($span))
    };
}

#[macro_export]
macro_rules! err_type_at {
    ($span:expr, $($arg:tt)*) => {
        $crate::error::CompilerError::TypeError(format!($($arg)*), Some($span))
    };
}

#[macro_export]
macro_rules! err_undef_at {
    ($span:expr, $($arg:tt)*) => {
        $crate::error::CompilerError::UndefinedVariable(format!($($arg)*), Some($span))
    };
}

#[macro_export]
macro_rules! err_spirv_at {
    ($span:expr, $($arg:tt)*) => {
        $crate::error::CompilerError::SpirvError(format!($($arg)*), Some($span))
    };
}

#[macro_export]
macro_rules! err_glsl_at {
    ($span:expr, $($arg:tt)*) => {
        $crate::error::CompilerError::GlslError(format!($($arg)*), Some($span))
    };
}

#[macro_export]
macro_rules! err_wgsl_at {
    ($span:expr, $($arg:tt)*) => {
        $crate::error::CompilerError::WgslError(format!($($arg)*), Some($span))
    };
}

#[macro_export]
macro_rules! err_module_at {
    ($span:expr, $($arg:tt)*) => {
        $crate::error::CompilerError::ModuleError(format!($($arg)*), Some($span))
    };
}

#[macro_export]
macro_rules! err_flatten_at {
    ($span:expr, $($arg:tt)*) => {
        $crate::error::CompilerError::FlatteningError(format!($($arg)*), Some($span))
    };
}

#[macro_export]
macro_rules! err_alias_at {
    ($span:expr, $($arg:tt)*) => {
        $crate::error::CompilerError::AliasError(format!($($arg)*), Some($span))
    };
}

// Bail macros without span (delegate to err_x)

#[macro_export]
macro_rules! bail_parse {
    ($($arg:tt)*) => {
        return Err($crate::err_parse!($($arg)*))
    };
}

#[macro_export]
macro_rules! bail_type {
    ($($arg:tt)*) => {
        return Err($crate::err_type!($($arg)*))
    };
}

#[macro_export]
macro_rules! bail_undef {
    ($($arg:tt)*) => {
        return Err($crate::err_undef!($($arg)*))
    };
}

#[macro_export]
macro_rules! bail_spirv {
    ($($arg:tt)*) => {
        return Err($crate::err_spirv!($($arg)*))
    };
}

#[macro_export]
macro_rules! bail_glsl {
    ($($arg:tt)*) => {
        return Err($crate::err_glsl!($($arg)*))
    };
}

#[macro_export]
macro_rules! bail_module {
    ($($arg:tt)*) => {
        return Err($crate::err_module!($($arg)*))
    };
}

#[macro_export]
macro_rules! bail_flatten {
    ($($arg:tt)*) => {
        return Err($crate::err_flatten!($($arg)*))
    };
}

#[macro_export]
macro_rules! bail_alias {
    ($($arg:tt)*) => {
        return Err($crate::err_alias!($($arg)*))
    };
}

// Bail macros with span (delegate to err_x_at)

#[macro_export]
macro_rules! bail_parse_at {
    ($span:expr, $($arg:tt)*) => {
        return Err($crate::err_parse_at!($span, $($arg)*))
    };
}

#[macro_export]
macro_rules! bail_type_at {
    ($span:expr, $($arg:tt)*) => {
        return Err($crate::err_type_at!($span, $($arg)*))
    };
}

#[macro_export]
macro_rules! bail_undef_at {
    ($span:expr, $($arg:tt)*) => {
        return Err($crate::err_undef_at!($span, $($arg)*))
    };
}

#[macro_export]
macro_rules! bail_spirv_at {
    ($span:expr, $($arg:tt)*) => {
        return Err($crate::err_spirv_at!($span, $($arg)*))
    };
}

#[macro_export]
macro_rules! bail_glsl_at {
    ($span:expr, $($arg:tt)*) => {
        return Err($crate::err_glsl_at!($span, $($arg)*))
    };
}

#[macro_export]
macro_rules! bail_module_at {
    ($span:expr, $($arg:tt)*) => {
        return Err($crate::err_module_at!($span, $($arg)*))
    };
}

#[macro_export]
macro_rules! bail_flatten_at {
    ($span:expr, $($arg:tt)*) => {
        return Err($crate::err_flatten_at!($span, $($arg)*))
    };
}

#[macro_export]
macro_rules! bail_alias_at {
    ($span:expr, $($arg:tt)*) => {
        return Err($crate::err_alias_at!($span, $($arg)*))
    };
}
