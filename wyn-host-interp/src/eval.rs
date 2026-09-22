use crate::number::{calculate, compare};
use crate::scalar;
use crate::{Backend, Error, Form, Number, NumberType, Program, Result, Value};
use std::collections::{BTreeMap, BTreeSet};

type Scope = BTreeMap<String, (Value, bool)>;

pub(crate) fn builtin(name: &str) -> bool {
    name.starts_with("gpu-")
        || name.starts_with("wyn-")
        || name.starts_with("define-")
        || NumberType::parse(name).is_some()
        || matches!(
            name,
            "nil"
                | "t"
                | "quote"
                | "let"
                | "let*"
                | "setq"
                | "progn"
                | "if"
                | "cond"
                | "and"
                | "or"
                | "dotimes"
                | "do"
                | "defun"
                | "+"
                | "-"
                | "*"
                | "/"
                | "floor"
                | "ceiling"
                | "mod"
                | "min"
                | "max"
                | "="
                | "/="
                | "<"
                | "<="
                | ">"
                | ">="
                | "not"
                | "list"
                | "nth"
                | "i32-add"
                | "i32-sub"
                | "i32-mul"
                | "u32-add"
                | "u32-sub"
                | "u32-mul"
        )
}

pub(crate) fn binding_name(name: &str) -> Result<()> {
    if builtin(name) || name.starts_with(':') {
        return Err(Error::Invalid(format!("cannot bind reserved name {name}")));
    }
    Ok(())
}

fn symbol(form: &Form) -> Result<&str> {
    match form {
        Form::Symbol(name) => Ok(name),
        _ => Err(Error::Invalid("expected an identifier".into())),
    }
}

fn list(form: &Form) -> Result<&[Form]> {
    match form {
        Form::List(values) => Ok(values),
        _ => Err(Error::Invalid("expected a list form".into())),
    }
}

pub(crate) fn calls(form: &Form, result: &mut BTreeSet<String>) {
    let Form::List(items) = form else {
        return;
    };
    let Some(Form::Symbol(name)) = items.first() else {
        for item in items {
            calls(item, result);
        }
        return;
    };
    if name == "quote" {
        return;
    }
    if matches!(name.as_str(), "let" | "let*" | "do") {
        if let Some(Form::List(bindings)) = items.get(1) {
            for binding in bindings {
                if let Form::List(parts) = binding {
                    for value in parts.iter().skip(1) {
                        calls(value, result);
                    }
                }
            }
        }
        for item in items.iter().skip(2) {
            calls(item, result);
        }
    } else if name == "dotimes" {
        if let Some(Form::List(control)) = items.get(1) {
            for item in control.iter().skip(1) {
                calls(item, result);
            }
        }
        for item in items.iter().skip(2) {
            calls(item, result);
        }
    } else {
        if !builtin(name) {
            result.insert(name.clone());
        }
        for item in &items[1..] {
            calls(item, result);
        }
    }
}

pub(crate) fn run(
    program: &Program,
    function: &str,
    arguments: &[Value],
    backend: &mut impl Backend,
) -> Result<Value> {
    Interpreter {
        program,
        backend,
        remaining: 1_000_000,
        depth: 0,
    }
    .function(function, arguments, false)
}

struct Interpreter<'a, B> {
    program: &'a Program,
    backend: &'a mut B,
    remaining: usize,
    depth: usize,
}

impl<B: Backend> Interpreter<'_, B> {
    fn function(&mut self, name: &str, arguments: &[Value], materialize: bool) -> Result<Value> {
        let Some(function) = self.program.functions.get(name) else {
            return Err(Error::Invalid(format!("unknown function {name}")));
        };
        if function.parameters.len() != arguments.len() {
            return Err(Error::Invalid(format!("wrong argument count for {name}")));
        }
        if self.depth >= 128 {
            return Err(Error::Invalid("function nesting limit exceeded".into()));
        }
        let mut scopes = vec![function
            .parameters
            .iter()
            .cloned()
            .zip(arguments.iter().cloned().map(|v| (v, false)))
            .collect()];
        self.depth += 1;
        let result = self.body(&function.body, &mut scopes).and_then(|v| {
            if materialize {
                v.materialize()
            } else {
                Ok(v)
            }
        });
        self.depth -= 1;
        result
    }

    fn body(&mut self, forms: &[Form], scopes: &mut Vec<Scope>) -> Result<Value> {
        let mut result = Value::Nil;
        for form in forms {
            result = self.eval(form, scopes)?;
        }
        Ok(result)
    }

    fn eval(&mut self, form: &Form, scopes: &mut Vec<Scope>) -> Result<Value> {
        if self.remaining == 0 {
            return Err(Error::Invalid("evaluation step limit exceeded".into()));
        }
        self.remaining -= 1;
        let parts = match form {
            Form::Number(n) => return Ok(Value::Number(*n)),
            Form::String(s) => return Ok(Value::String(s.clone())),
            Form::Symbol(s) => {
                return match s.as_str() {
                    "t" => Ok(Value::True),
                    "nil" => Ok(Value::Nil),
                    _ if s.starts_with(':') => Ok(Value::Symbol(s.clone())),
                    _ => scopes
                        .iter()
                        .rev()
                        .find_map(|scope| scope.get(s).map(|(v, _)| v.clone()))
                        .ok_or_else(|| Error::Invalid(format!("unbound variable {s}"))),
                }
            }
            Form::List(parts) if parts.is_empty() => return Ok(Value::Nil),
            Form::List(parts) => parts,
        };
        let name = symbol(&parts[0])?;
        let args = &parts[1..];
        match name {
            "quote" => {
                let [datum] = args else {
                    return Err(Error::Invalid("quote needs one operand".into()));
                };
                Ok(Value::quoted(datum))
            }
            "progn" => self.body(args, scopes),
            "if" => {
                if !(2..=3).contains(&args.len()) {
                    return Err(Error::Invalid("if needs two or three operands".into()));
                }
                if self.eval(&args[0], scopes)?.truth() {
                    self.eval(&args[1], scopes)
                } else if args.len() == 3 {
                    self.eval(&args[2], scopes)
                } else {
                    Ok(Value::Nil)
                }
            }
            "cond" => {
                for clause in args {
                    let clause = list(clause)?;
                    let Some(test) = clause.first() else {
                        return Err(Error::Invalid("empty cond clause".into()));
                    };
                    let value = self.eval(test, scopes)?;
                    if value.truth() {
                        return if clause.len() == 1 { Ok(value) } else { self.body(&clause[1..], scopes) };
                    }
                }
                Ok(Value::Nil)
            }
            "and" | "or" => {
                let mut result = Value::boolean(name == "and");
                for arg in args {
                    result = self.eval(arg, scopes)?;
                    if result.truth() == (name == "or") {
                        break;
                    }
                }
                Ok(result)
            }
            "let" | "let*" => {
                let Some(bindings) = args.first() else {
                    return Err(Error::Invalid("missing let bindings".into()));
                };
                let bindings = list(bindings)?;
                let mut scope = Scope::new();
                let sequential = name == "let*";
                if sequential {
                    scopes.push(Scope::new());
                }
                for binding in bindings {
                    let [name, init] = list(binding)? else {
                        return Err(Error::Invalid("binding needs name and initializer".into()));
                    };
                    let name = symbol(name)?;
                    binding_name(name)?;
                    let value = self.eval(init, scopes)?.materialize()?;
                    if name.is_empty() {
                        return Err(Error::Invalid("empty binding name".into()));
                    }
                    if sequential {
                        let Some(scope) = scopes.last_mut() else {
                            return Err(Error::Invalid("missing lexical scope".into()));
                        };
                        scope.insert(name.into(), (value, false));
                    } else if scope.insert(name.into(), (value, false)).is_some() {
                        return Err(Error::Invalid(format!("duplicate binding {name}")));
                    }
                }
                if name == "let" {
                    scopes.push(scope);
                }
                let result = self.body(&args[1..], scopes);
                scopes.pop();
                result
            }
            "setq" => {
                if args.len() % 2 != 0 {
                    return Err(Error::Invalid("setq requires name/value pairs".into()));
                }
                let mut result = Value::Nil;
                for pair in args.chunks_exact(2) {
                    let name = symbol(&pair[0])?;
                    result = self.eval(&pair[1], scopes)?.materialize()?;
                    let Some((value, readonly)) =
                        scopes.iter_mut().rev().find_map(|scope| scope.get_mut(name))
                    else {
                        return Err(Error::Invalid(format!("setq of unbound variable {name}")));
                    };
                    if *readonly {
                        return Err(Error::Invalid(format!("cannot assign loop index {name}")));
                    }
                    *value = result.clone();
                }
                Ok(result)
            }
            "dotimes" => self.dotimes(args, scopes),
            "do" => self.do_loop(args, scopes),
            _ => {
                let values = args.iter().map(|arg| self.eval(arg, scopes)).collect::<Result<Vec<_>>>()?;
                if let Some(kind) = NumberType::parse(name) {
                    let [value] = values.as_slice() else {
                        return Err(Error::Invalid(format!("{name} needs one operand")));
                    };
                    return Ok(Value::Number(value.number()?.convert(kind)?));
                }
                match name {
                    name if name.starts_with("wyn-") => scalar::evaluate(name, &values),
                    "nth" => {
                        let [index, list] = values.as_slice() else {
                            return Err(Error::Invalid("nth needs an index and list".into()));
                        };
                        Ok(list.list()?.get(index.u32()? as usize).cloned().unwrap_or(Value::Nil))
                    }
                    "list" => Ok(if values.is_empty() { Value::Nil } else { Value::List(values) }),
                    "not" => {
                        let [value] = values.as_slice() else {
                            return Err(Error::Invalid("not needs one operand".into()));
                        };
                        Ok(Value::boolean(!value.truth()))
                    }
                    "=" | "/=" | "<" | "<=" | ">" | ">=" => Ok(Value::boolean(compare(
                        name,
                        &values.iter().map(Value::number).collect::<Result<Vec<_>>>()?,
                    )?)),
                    "+" | "-" | "*" | "/" | "floor" | "ceiling" | "mod" | "min" | "max" | "i32-add"
                    | "i32-sub" | "i32-mul" | "u32-add" | "u32-sub" | "u32-mul" => {
                        Ok(Value::Number(calculate(
                            name,
                            &values.iter().map(Value::number).collect::<Result<Vec<_>>>()?,
                        )?))
                    }
                    name if name.starts_with("gpu-") => self.backend.call(self.program, name, &values),
                    _ => self.function(
                        name,
                        &values.into_iter().map(Value::materialize).collect::<Result<Vec<_>>>()?,
                        true,
                    ),
                }
            }
        }
    }

    fn dotimes(&mut self, args: &[Form], scopes: &mut Vec<Scope>) -> Result<Value> {
        let Some(control) = args.first() else {
            return Err(Error::Invalid("missing dotimes control".into()));
        };
        if args[1..].iter().any(|form| !matches!(form, Form::List(_))) {
            return Err(Error::Invalid("loop bodies require compound forms".into()));
        }
        let control = list(control)?;
        if !(2..=3).contains(&control.len()) {
            return Err(Error::Invalid("invalid dotimes control".into()));
        }
        let name = symbol(&control[0])?;
        binding_name(name)?;
        let count = self.eval(&control[1], scopes)?.materialize()?.number()?;
        let iterations = count.integer()?.max(0);
        let kind = count.kind().unwrap_or(NumberType::I32);
        scopes.push(Scope::new());
        let result = (|| {
            for i in 0..iterations {
                let Some(scope) = scopes.last_mut() else {
                    return Err(Error::Invalid("missing loop scope".into()));
                };
                scope.insert(
                    name.into(),
                    (Value::Number(Number::Literal(i).convert(kind)?), true),
                );
                self.body(&args[1..], scopes)?;
                if self.remaining == 0 {
                    return Err(Error::Invalid("evaluation step limit exceeded".into()));
                }
                self.remaining -= 1;
            }
            let Some(scope) = scopes.last_mut() else {
                return Err(Error::Invalid("missing loop scope".into()));
            };
            scope.insert(
                name.into(),
                (Value::Number(Number::Literal(iterations).convert(kind)?), true),
            );
            if let Some(result) = control.get(2) {
                self.eval(result, scopes)
            } else {
                Ok(Value::Nil)
            }
        })();
        scopes.pop();
        result
    }

    fn do_loop(&mut self, args: &[Form], scopes: &mut Vec<Scope>) -> Result<Value> {
        if args.len() < 2 {
            return Err(Error::Invalid("do needs bindings and termination clause".into()));
        }
        if args[2..].iter().any(|form| !matches!(form, Form::List(_))) {
            return Err(Error::Invalid("loop bodies require compound forms".into()));
        }
        let bindings = list(&args[0])?;
        let termination = list(&args[1])?;
        let Some(test) = termination.first() else {
            return Err(Error::Invalid("do needs a termination test".into()));
        };
        let mut scope = Scope::new();
        let mut steps = Vec::new();
        for binding in bindings {
            let binding = list(binding)?;
            if !(2..=3).contains(&binding.len()) {
                return Err(Error::Invalid("invalid do binding".into()));
            }
            let name = symbol(&binding[0])?;
            binding_name(name)?;
            let value = self.eval(&binding[1], scopes)?.materialize()?;
            if scope.insert(name.into(), (value, false)).is_some() {
                return Err(Error::Invalid(format!("duplicate do binding {name}")));
            }
            if let Some(step) = binding.get(2) {
                steps.push((name, step));
            }
        }
        scopes.push(scope);
        let result = (|| loop {
            if self.eval(test, scopes)?.truth() {
                break self.body(&termination[1..], scopes);
            }
            self.body(&args[2..], scopes)?;
            let next = steps
                .iter()
                .map(|(name, step)| Ok((*name, self.eval(step, scopes)?.materialize()?)))
                .collect::<Result<Vec<_>>>()?;
            let Some(scope) = scopes.last_mut() else {
                return Err(Error::Invalid("missing loop scope".into()));
            };
            for (name, value) in next {
                scope.insert(name.into(), (value, false));
            }
        })();
        scopes.pop();
        result
    }
}
