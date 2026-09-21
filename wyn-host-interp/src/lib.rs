mod eval;
mod number;
mod program;
mod reader;
mod value;

#[cfg(feature = "wgpu")]
pub mod gpu;

pub use number::{Number, NumberType};
pub use program::{Declaration, Entry, Parameter, Program};
pub use reader::{read, Form};
pub use value::{Options, Value};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("WHL reader at byte {offset}: {message}")]
    Reader {
        offset: usize,
        message: String,
    },
    #[error("WHL: {0}")]
    Invalid(String),
    #[error("WHL GPU: {0}")]
    Gpu(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

pub trait Backend {
    fn call(&mut self, program: &Program, operation: &str, arguments: &[Value]) -> Result<Value>;

    fn validate_parameter(&self, _parameter: &Parameter, _value: &Value) -> Result<()> {
        Ok(())
    }

    fn begin(&mut self, _arguments: &[Value]) -> Result<()> {
        Ok(())
    }

    fn finish(&mut self, _result: &Result<Value>) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "interp_tests.rs"]
mod tests;
