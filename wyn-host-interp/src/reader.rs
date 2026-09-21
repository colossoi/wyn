use crate::{Error, Number, Result};

#[derive(Clone, Debug, PartialEq)]
pub enum Form {
    Symbol(String),
    String(String),
    Number(Number),
    List(Vec<Form>),
}

pub fn read(source: &str) -> Result<Vec<Form>> {
    let mut reader = Reader { source, offset: 0 };
    let mut forms = Vec::new();
    while reader.skip() {
        forms.push(reader.form(0)?);
    }
    Ok(forms)
}

struct Reader<'a> {
    source: &'a str,
    offset: usize,
}

impl Reader<'_> {
    fn error(&self, message: impl Into<String>) -> Error {
        Error::Reader {
            offset: self.offset,
            message: message.into(),
        }
    }

    fn peek(&self) -> Option<char> {
        self.source[self.offset..].chars().next()
    }

    fn next(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.offset += c.len_utf8();
        Some(c)
    }

    fn skip(&mut self) -> bool {
        loop {
            match self.peek() {
                Some(c) if c.is_whitespace() => {
                    self.next();
                }
                Some(';') => while self.next().is_some_and(|c| c != '\n') {},
                other => return other.is_some(),
            }
        }
    }

    fn form(&mut self, depth: usize) -> Result<Form> {
        if depth > 256 {
            return Err(self.error("nesting exceeds 256 levels"));
        }
        self.skip();
        match self.next() {
            Some('(') => {
                let mut values = Vec::new();
                loop {
                    if !self.skip() {
                        return Err(self.error("unclosed list"));
                    }
                    if self.peek() == Some(')') {
                        self.next();
                        break;
                    }
                    values.push(self.form(depth + 1)?);
                }
                Ok(Form::List(values))
            }
            Some(')') => Err(self.error("unexpected closing parenthesis")),
            Some('\'') => Ok(Form::List(vec![
                Form::Symbol("quote".into()),
                self.form(depth + 1)?,
            ])),
            Some('"') => {
                let mut value = String::new();
                loop {
                    match self.next() {
                        Some('"') => break,
                        Some('\\') => {
                            let Some(c) = self.next() else {
                                return Err(self.error("unfinished string escape"));
                            };
                            value.push(c);
                        }
                        Some(c) => value.push(c),
                        None => return Err(self.error("unclosed string")),
                    }
                }
                Ok(Form::String(value))
            }
            Some(first) => {
                let start = self.offset - first.len_utf8();
                while self
                    .peek()
                    .is_some_and(|c| !c.is_whitespace() && !matches!(c, '(' | ')' | ';' | '\'' | '"'))
                {
                    self.next();
                }
                let word = &self.source[start..self.offset];
                let integer = word.trim_start_matches(['+', '-']);
                if !integer.is_empty() && integer.bytes().all(|b| b.is_ascii_digit()) {
                    let n = word
                        .parse::<i128>()
                        .map_err(|_| self.error("integer literal exceeds fixed-width ranges"))?;
                    if n < i128::from(i64::MIN) || n > i128::from(u64::MAX) {
                        return Err(self.error("integer literal exceeds fixed-width ranges"));
                    }
                    return Ok(Form::Number(Number::Literal(n)));
                }
                if first.is_ascii_digit()
                    || matches!(first, '-' | '+' | '.')
                        && word.len() > 1
                        && word.as_bytes()[1].is_ascii_digit()
                {
                    let lower = word.to_ascii_lowercase();
                    let double = lower.contains('d');
                    let number = lower
                        .replace(['d', 'f'], "e")
                        .parse::<f64>()
                        .map_err(|_| self.error(format!("invalid number {word}")))?;
                    return Ok(Form::Number(if double {
                        Number::f64(number)?
                    } else {
                        Number::f32(number as f32)?
                    }));
                }
                let lower = word.to_ascii_lowercase();
                let ident = lower.strip_prefix(':').unwrap_or(&lower);
                let valid = ident.bytes().next().is_some_and(|b| b.is_ascii_alphabetic())
                    && ident.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-');
                if !valid
                    && !matches!(
                        lower.as_str(),
                        "+" | "-" | "*" | "/" | "=" | "/=" | "<" | "<=" | ">" | ">=" | "let*"
                    )
                {
                    return Err(self.error(format!("unsupported reader token {word}")));
                }
                Ok(Form::Symbol(lower))
            }
            None => Err(self.error("expected a form")),
        }
    }
}
