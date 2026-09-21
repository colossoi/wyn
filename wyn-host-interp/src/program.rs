use crate::eval;
use crate::{read, Backend, Error, Form, NumberType, Options, Result, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub struct Parameter {
    pub name: String,
    pub kind: String,
    pub access: Option<String>,
    pub options: Options,
}

impl Parameter {
    fn parse(value: &Value) -> Result<Self> {
        let values = value.list()?;
        if values.len() < 2 {
            return Err(Error::Invalid("incomplete parameter".into()));
        }
        let name = values[0].text()?.to_owned();
        let kind = values[1].text()?.to_owned();
        if !matches!(
            kind.as_str(),
            ":bool"
                | ":i8"
                | ":u8"
                | ":i16"
                | ":u16"
                | ":i32"
                | ":u32"
                | ":i64"
                | ":u64"
                | ":f16"
                | ":f32"
                | ":f64"
                | ":buffer"
                | ":host-buffer"
                | ":texture"
                | ":texture-view"
                | ":sampler"
        ) {
            return Err(Error::Invalid(format!("unknown parameter type {kind}")));
        }
        let resource = matches!(
            kind.as_str(),
            ":buffer" | ":host-buffer" | ":texture" | ":texture-view"
        );
        let start = if resource { 3 } else { 2 };
        if values.len() < start {
            return Err(Error::Invalid(format!("missing access for {name}")));
        }
        let access = if resource {
            let access = values[2].text()?;
            let valid = if kind == ":texture-view" {
                matches!(access, ":sampled" | ":storage" | ":render-target")
            } else {
                matches!(access, ":read" | ":write" | ":read-write")
            };
            if !valid {
                return Err(Error::Invalid(format!("invalid resource access {access}")));
            }
            Some(access.into())
        } else {
            None
        };
        let options = Options::parse(
            &values[start..],
            &[
                ":element",
                ":stride",
                ":layout",
                ":min-bytes",
                ":source-name",
                ":dimension",
                ":format",
                ":samples",
                ":sample-type",
                ":kind",
                ":ownership",
                ":alias",
                ":value-layout",
            ],
        )?;
        if let Some(name) = options.optional(":source-name") {
            if !matches!(name, Value::String(_)) {
                return Err(Error::Invalid(":source-name must be a string".into()));
            }
        }
        Ok(Self {
            name,
            kind,
            access,
            options,
        })
    }

    pub fn source_name(&self) -> &str {
        match self.options.optional(":source-name") {
            Some(Value::String(name)) => name,
            _ => &self.name,
        }
    }

    fn argument(&self, value: Value) -> Result<Value> {
        if let Some(kind) = self.kind.strip_prefix(':').and_then(NumberType::parse) {
            let number = value.number()?;
            if let Some(actual) = number.kind() {
                if actual != kind {
                    return Err(Error::Invalid(format!(
                        "{} requires {}, got {actual:?}",
                        self.name, self.kind
                    )));
                }
            }
            return Ok(Value::Number(number.convert(kind)?));
        }
        match (&*self.kind, &value) {
            (":bool", Value::True | Value::Nil) => {}
            (
                ":buffer" | ":host-buffer" | ":texture" | ":texture-view" | ":sampler",
                Value::Resource(_),
            ) => {}
            _ => return Err(Error::Invalid(format!("{} requires {}", self.name, self.kind))),
        }
        Ok(value)
    }

    pub fn minimum_bytes(&self) -> Result<Option<u64>> {
        if let Some(layout) = self.options.optional(":layout") {
            let layout = Options::parse(layout.list()?, &[":size", ":alignment", ":fields"])?;
            return Ok(Some(layout.get(":size")?.u64()?));
        }
        self.options.optional(":min-bytes").map(Value::u64).transpose()
    }

    pub fn fields(&self) -> Result<Vec<(String, u64, u64)>> {
        let Some(layout) = self.options.optional(":layout") else {
            return Ok(Vec::new());
        };
        let layout = Options::parse(layout.list()?, &[":size", ":alignment", ":fields"])?;
        layout
            .get(":fields")?
            .list()?
            .iter()
            .map(|field| {
                let [name, ty, offset] = field.list()? else {
                    return Err(Error::Invalid("invalid layout field".into()));
                };
                let bytes = match ty {
                    Value::Symbol(s) => match s.as_str() {
                        ":i8" | ":u8" => 1,
                        ":i16" | ":u16" | ":f16" => 2,
                        ":i32" | ":u32" | ":f32" | ":bool" => 4,
                        ":i64" | ":u64" | ":f64" => 8,
                        _ => return Err(Error::Invalid(format!("unsupported field type {s}"))),
                    },
                    _ => {
                        let [kind, bytes] = ty.list()? else {
                            return Err(Error::Invalid("invalid opaque field type".into()));
                        };
                        if kind.text()? != ":bytes" {
                            return Err(Error::Invalid("expected opaque byte field".into()));
                        }
                        bytes.u64()?
                    }
                };
                Ok((decode_source_name(name.text()?)?, offset.u64()?, bytes))
            })
            .collect()
    }
}

pub fn decode_source_name(symbol: &str) -> Result<String> {
    let Some(mut source) = symbol.strip_prefix("source-") else {
        return Ok(symbol.into());
    };
    if !source.is_ascii() {
        return Err(Error::Invalid("source name encoding must be ASCII".into()));
    }
    let mut bytes = Vec::new();
    while !source.is_empty() {
        if source.starts_with('-') {
            if source.len() < 4 || &source[3..4] != "-" {
                return Err(Error::Invalid("invalid source name encoding".into()));
            }
            bytes.push(
                u8::from_str_radix(&source[1..3], 16)
                    .map_err(|_| Error::Invalid("invalid source name byte".into()))?,
            );
            source = &source[4..];
        } else {
            bytes.push(source.as_bytes()[0]);
            source = &source[1..];
        }
    }
    String::from_utf8(bytes).map_err(|_| Error::Invalid("source name is not UTF-8".into()))
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub function: String,
    pub source_name: String,
    pub parameters: Vec<Parameter>,
    pub results: Vec<Parameter>,
}

#[derive(Clone, Debug)]
pub struct Declaration {
    pub kind: String,
    pub name: String,
    pub options: Options,
    pub parameters: Vec<Parameter>,
}

#[derive(Clone, Debug)]
pub(crate) struct Function {
    pub parameters: Vec<String>,
    pub body: Vec<Form>,
}

#[derive(Clone, Debug, Default)]
pub struct Program {
    pub modules: BTreeMap<String, Declaration>,
    pub kernels: BTreeMap<String, Declaration>,
    pub graphics: BTreeMap<String, Declaration>,
    pub entries: BTreeMap<String, Entry>,
    pub(crate) functions: BTreeMap<String, Function>,
}

fn symbol(form: &Form) -> Result<&str> {
    match form {
        Form::Symbol(s) => Ok(s),
        _ => Err(Error::Invalid("expected a symbol".into())),
    }
}

fn literal(form: &Form) -> Result<Value> {
    if let Form::List(values) = form {
        if values.first().is_some_and(|f| matches!(f, Form::Symbol(s) if s == "quote")) {
            let [_, value] = values.as_slice() else {
                return Err(Error::Invalid("quote needs one datum".into()));
            };
            return Ok(Value::quoted(value));
        }
    }
    Ok(Value::quoted(form))
}

fn parameters(value: &Value) -> Result<Vec<Parameter>> {
    let parameters = value.list()?.iter().map(Parameter::parse).collect::<Result<Vec<_>>>()?;
    let mut names = BTreeSet::new();
    for p in &parameters {
        if !names.insert(&p.name) {
            return Err(Error::Invalid(format!("duplicate parameter {}", p.name)));
        }
    }
    Ok(parameters)
}

impl Program {
    pub fn parse(source: &str) -> Result<Self> {
        let forms = read(source)?;
        let Some(Form::List(header)) = forms.first() else {
            return Err(Error::Invalid("missing define-host-program".into()));
        };
        if header.first().map(symbol).transpose()? != Some("define-host-program") {
            return Err(Error::Invalid("expected define-host-program first".into()));
        }
        let header = header[1..].iter().map(literal).collect::<Result<Vec<_>>>()?;
        if Options::parse(&header, &[":version"])?.u32(":version")? != 1 {
            return Err(Error::Invalid("unsupported host program version".into()));
        }
        let mut program = Self::default();
        for form in &forms[1..] {
            let Form::List(parts) = form else {
                return Err(Error::Invalid("top level must contain declarations".into()));
            };
            if parts.len() < 2 {
                return Err(Error::Invalid("incomplete declaration".into()));
            }
            let kind = symbol(&parts[0])?;
            if kind == "defun" {
                if parts.len() < 3 {
                    return Err(Error::Invalid("incomplete defun".into()));
                }
                let name = symbol(&parts[1])?;
                eval::binding_name(name)?;
                let Form::List(params) = &parts[2] else {
                    return Err(Error::Invalid("function parameters must be a list".into()));
                };
                let mut names = BTreeSet::new();
                let params = params
                    .iter()
                    .map(|form| {
                        let name = symbol(form)?;
                        eval::binding_name(name)?;
                        if !names.insert(name) {
                            return Err(Error::Invalid(format!("duplicate parameter {name}")));
                        }
                        Ok(name.to_owned())
                    })
                    .collect::<Result<_>>()?;
                if program
                    .functions
                    .insert(
                        name.into(),
                        Function {
                            parameters: params,
                            body: parts[3..].to_vec(),
                        },
                    )
                    .is_some()
                {
                    return Err(Error::Invalid(format!("duplicate function {name}")));
                }
                continue;
            }
            let name = literal(&parts[1])?.text()?.to_owned();
            let values = parts[2..].iter().map(literal).collect::<Result<Vec<_>>>()?;
            let allowed: &[&str] = match kind {
                "define-gpu-module" => &[":format", ":path"],
                "define-gpu-kernel" => &[":module", ":entry", ":workgroup-size", ":parameters", ":abi"],
                "define-gpu-graphics" => &[
                    ":vertex",
                    ":fragment",
                    ":parameters",
                    ":abi",
                    ":vertex-inputs",
                    ":color-outputs",
                    ":depth-format",
                    ":samples",
                    ":topology",
                    ":front-face",
                    ":cull",
                    ":fill",
                    ":depth-test",
                    ":depth-write",
                    ":blend",
                    ":color-write",
                ],
                "define-host-entry" => &[":function", ":source-name", ":parameters", ":results"],
                _ => return Err(Error::Invalid(format!("unsupported top-level form {kind}"))),
            };
            let options = Options::parse(&values, allowed)?;
            for key in allowed {
                if *key != ":source-name" {
                    options.get(key)?;
                }
            }
            if kind == "define-host-entry" {
                let entry = Entry {
                    name: name.clone(),
                    function: options.text(":function")?.into(),
                    source_name: options
                        .optional(":source-name")
                        .map(Value::text)
                        .transpose()?
                        .unwrap_or(&name)
                        .into(),
                    parameters: parameters(options.get(":parameters")?)?,
                    results: parameters(options.get(":results")?)?,
                };
                if program.entries.insert(name.clone(), entry).is_some() {
                    return Err(Error::Invalid(format!("duplicate entry {name}")));
                }
                continue;
            }
            let params = options.optional(":parameters").map(parameters).transpose()?.unwrap_or_default();
            let declaration = Declaration {
                kind: kind.into(),
                name: name.clone(),
                options,
                parameters: params,
            };
            let declarations = match kind {
                "define-gpu-module" => &mut program.modules,
                "define-gpu-kernel" => &mut program.kernels,
                _ => &mut program.graphics,
            };
            if declarations.insert(name.clone(), declaration).is_some() {
                return Err(Error::Invalid(format!("duplicate declaration {name}")));
            }
        }
        program.validate()?;
        Ok(program)
    }

    fn validate(&self) -> Result<()> {
        let mut source_names = BTreeSet::new();
        for entry in self.entries.values() {
            let Some(function) = self.functions.get(&entry.function) else {
                return Err(Error::Invalid(format!("missing function {}", entry.function)));
            };
            if function.parameters.len() != entry.parameters.len() {
                return Err(Error::Invalid(format!(
                    "entry {} has the wrong parameter count",
                    entry.source_name
                )));
            }
            if !source_names.insert(&entry.source_name) {
                return Err(Error::Invalid(format!(
                    "duplicate source entry {}",
                    entry.source_name
                )));
            }
        }
        for kernel in self.kernels.values() {
            if !self.modules.contains_key(kernel.options.text(":module")?) {
                return Err(Error::Invalid("kernel references an unknown module".into()));
            }
            let dims = kernel.options.get(":workgroup-size")?.list()?;
            if dims.len() != 3 || dims.iter().any(|d| d.u32().map_or(true, |d| d == 0)) {
                return Err(Error::Invalid("invalid workgroup size".into()));
            }
        }
        for graphics in self.graphics.values() {
            for stage in [":vertex", ":fragment"] {
                let value = graphics.options.get(stage)?;
                if matches!(value, Value::Nil) {
                    continue;
                }
                let [module, _] = value.list()? else {
                    return Err(Error::Invalid("invalid shader stage".into()));
                };
                if !self.modules.contains_key(module.text()?) {
                    return Err(Error::Invalid("graphics references an unknown module".into()));
                }
            }
        }
        for name in self.functions.keys() {
            self.check_calls(name, &mut BTreeSet::new())?;
        }
        Ok(())
    }

    fn check_calls(&self, name: &str, active: &mut BTreeSet<String>) -> Result<()> {
        if active.len() >= 128 {
            return Err(Error::Invalid("function nesting limit exceeded".into()));
        }
        if !active.insert(name.into()) {
            return Err(Error::Invalid(format!("recursive call to {name}")));
        }
        let function = &self.functions[name];
        let mut calls = BTreeSet::new();
        for form in &function.body {
            eval::calls(form, &mut calls);
        }
        for call in calls {
            if self.functions.contains_key(&call) {
                self.check_calls(&call, active)?;
            }
        }
        active.remove(name);
        Ok(())
    }

    pub fn entry(&self, name: &str) -> Result<&Entry> {
        let Some(entry) = self.entries.values().find(|e| e.source_name == name || e.name == name) else {
            return Err(Error::Invalid(format!("unknown entry {name}")));
        };
        Ok(entry)
    }

    pub fn run(&self, name: &str, arguments: &[Value], backend: &mut impl Backend) -> Result<Value> {
        let entry = self.entry(name)?;
        if arguments.len() != entry.parameters.len() {
            return Err(Error::Invalid(format!("wrong argument count for {name}")));
        }
        let arguments = entry
            .parameters
            .iter()
            .zip(arguments)
            .map(|(p, v)| p.argument(v.clone()))
            .collect::<Result<Vec<_>>>()?;
        for (parameter, value) in entry.parameters.iter().zip(&arguments) {
            backend.validate_parameter(parameter, value)?;
        }
        backend.begin(&arguments)?;
        let result = eval::run(self, &entry.function, &arguments, backend).and_then(|result| {
            let values = match entry.results.len() {
                0 if !result.truth() => Vec::new(),
                1 => vec![result],
                _ => result.list()?.to_vec(),
            };
            if values.len() != entry.results.len() {
                return Err(Error::Invalid("entry returned the wrong result count".into()));
            }
            let values = entry
                .results
                .iter()
                .zip(values)
                .map(|(parameter, value)| {
                    let value = parameter.argument(value)?;
                    backend.validate_parameter(parameter, &value)?;
                    if let Some(alias) = parameter.options.optional(":alias") {
                        let Some(index) = entry
                            .parameters
                            .iter()
                            .position(|p| alias.text().is_ok_and(|name| name == p.name))
                        else {
                            return Err(Error::Invalid("result aliases an unknown parameter".into()));
                        };
                        if value != arguments[index] {
                            return Err(Error::Invalid("result differs from its declared alias".into()));
                        }
                    }
                    Ok(value)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(match values.as_slice() {
                [] => Value::Nil,
                [value] => value.clone(),
                _ => Value::List(values),
            })
        });
        let cleanup = backend.finish(&result);
        let result = result?;
        cleanup?;
        Ok(result)
    }
}
