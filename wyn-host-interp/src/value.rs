use crate::{Error, Form, Number, Result};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Nil,
    True,
    Number(Number),
    String(String),
    Symbol(String),
    List(Vec<Value>),
    Resource(u64),
}

impl Value {
    pub fn quoted(form: &Form) -> Self {
        match form {
            Form::Symbol(s) if s == "nil" => Self::Nil,
            Form::Symbol(s) if s == "t" => Self::True,
            Form::Symbol(s) => Self::Symbol(s.clone()),
            Form::String(s) => Self::String(s.clone()),
            Form::Number(n) => Self::Number(*n),
            Form::List(items) if items.is_empty() => Self::Nil,
            Form::List(items) => Self::List(items.iter().map(Self::quoted).collect()),
        }
    }

    pub fn truth(&self) -> bool {
        !matches!(self, Self::Nil) && !matches!(self, Self::List(values) if values.is_empty())
    }
    pub fn boolean(value: bool) -> Self {
        if value {
            Self::True
        } else {
            Self::Nil
        }
    }
    pub fn list(&self) -> Result<&[Value]> {
        match self {
            Self::List(values) => Ok(values),
            Self::Nil => Ok(&[]),
            _ => Err(Error::Invalid("expected a list".into())),
        }
    }
    pub fn text(&self) -> Result<&str> {
        match self {
            Self::String(s) | Self::Symbol(s) => Ok(s),
            _ => Err(Error::Invalid("expected a string or symbol".into())),
        }
    }
    pub fn number(&self) -> Result<Number> {
        match self {
            Self::Number(n) => Ok(*n),
            _ => Err(Error::Invalid("expected a number".into())),
        }
    }
    pub fn u64(&self) -> Result<u64> {
        u64::try_from(self.number()?.integer()?)
            .map_err(|_| Error::Invalid("expected a u64 byte count".into()))
    }
    pub fn u32(&self) -> Result<u32> {
        u32::try_from(self.u64()?).map_err(|_| Error::Invalid("expected a u32 count".into()))
    }
    pub fn handle(&self) -> Result<u64> {
        match self {
            Self::Resource(id) => Ok(*id),
            _ => Err(Error::Invalid("expected a resource".into())),
        }
    }
    pub fn materialize(self) -> Result<Self> {
        match self {
            Self::Number(Number::Literal(n)) => {
                Ok(Self::Number(Number::Literal(n).convert(crate::NumberType::I32)?))
            }
            Self::List(values) => Ok(Self::List(
                values.into_iter().map(Self::materialize).collect::<Result<_>>()?,
            )),
            other => Ok(other),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Options(pub BTreeMap<String, Value>);

impl Options {
    pub fn parse(values: &[Value], allowed: &[&str]) -> Result<Self> {
        if values.len() % 2 != 0 {
            return Err(Error::Invalid("unpaired keyword option".into()));
        }
        let mut result = Self::default();
        for pair in values.chunks_exact(2) {
            let key = pair[0].text()?;
            if !allowed.contains(&key) {
                return Err(Error::Invalid(format!("unknown option {key}")));
            }
            if result.0.insert(key.into(), pair[1].clone()).is_some() {
                return Err(Error::Invalid(format!("duplicate option {key}")));
            }
        }
        Ok(result)
    }
    pub fn get(&self, name: &str) -> Result<&Value> {
        let Some(value) = self.0.get(name) else {
            return Err(Error::Invalid(format!("missing option {name}")));
        };
        Ok(value)
    }
    pub fn optional(&self, name: &str) -> Option<&Value> {
        self.0.get(name)
    }
    pub fn text(&self, name: &str) -> Result<&str> {
        self.get(name)?.text()
    }
    pub fn u32(&self, name: &str) -> Result<u32> {
        self.get(name)?.u32()
    }
}
