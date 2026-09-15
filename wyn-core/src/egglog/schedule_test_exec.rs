//! Small test oracle for the generated CFG and dispatch recipes. Deliberately
//! independent of the lowering templates; rejects unsupported scalar operations.

use crate::builtins;
use crate::egglog::{
    Array, AssociatedData, BlockId, BodyId, BufferId, Exit, ExprId, ExprKind, FunctionKind, Instruction,
    OperationId, OperationKind, ParameterId, Storage, Value as Code,
};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Value {
    Int(i64),
    Bool(bool),
    Tuple(Vec<Value>),
    Array(Rc<RefCell<Vec<Value>>>),
    Uninitialized,
}

impl Value {
    pub(super) fn arrays(values: Vec<Self>) -> Self {
        Self::values(values)
    }
    pub(super) fn array(values: impl IntoIterator<Item = i64>) -> Self {
        Self::values(values.into_iter().map(Self::Int).collect())
    }
    fn values(values: Vec<Self>) -> Self {
        Self::Array(Rc::new(RefCell::new(values)))
    }
    fn int(&self) -> i64 {
        let Self::Int(value) = self else {
            panic!("expected integer: {self:?}");
        };
        *value
    }
    fn boolean(&self) -> bool {
        let Self::Bool(value) = self else {
            panic!("expected bool: {self:?}");
        };
        *value
    }
    fn elements(&self) -> Vec<Self> {
        match self {
            Self::Array(values) => values.borrow().clone(),
            Self::Tuple(fields) if !fields.is_empty() => (0..fields[0].elements().len())
                .map(|i| Self::Tuple(fields.iter().map(|f| f.at(i)).collect()))
                .collect(),
            _ => panic!("expected array: {self:?}"),
        }
    }
    pub(super) fn ints(&self) -> Vec<i64> {
        self.elements().iter().map(Self::int).collect()
    }
    fn at(&self, i: usize) -> Self {
        let result = self.elements()[i].clone();
        assert_ne!(result, Self::Uninitialized, "read uninitialized element {i}");
        result
    }
    fn store(&self, i: usize, value: Self) {
        match (self, value) {
            (Self::Array(values), value) => values.borrow_mut()[i] = value,
            (Self::Tuple(fields), Self::Tuple(values)) => {
                for (field, value) in fields.iter().zip(values) {
                    field.store(i, value);
                }
            }
            _ => panic!("expected writable logical array"),
        }
    }
}

#[derive(Default)]
struct Frame {
    locals: BTreeMap<String, Value>,
    parameters: BTreeMap<ParameterId, Value>,
    expressions: BTreeMap<ExprId, Value>,
    operations: BTreeMap<OperationId, Value>,
    buffers: BTreeMap<BufferId, Value>,
}

struct Machine<'a> {
    data: &'a AssociatedData,
    buffers: BTreeMap<BufferId, Value>,
    invocation: u32,
    invocations: u32,
    fuel: usize,
}

pub(super) fn run(data: &AssociatedData, args: Vec<Value>) -> Vec<Value> {
    let entry = data
        .blocks
        .iter()
        .find_map(|(&id, b)| {
            b.interface.as_ref().is_some_and(|f| matches!(f.kind, FunctionKind::Entry(_))).then_some(id)
        })
        .unwrap();
    Machine {
        data,
        buffers: BTreeMap::new(),
        invocation: 0,
        invocations: 1,
        fuel: 5_000_000,
    }
    .call(entry, args)
}

impl Machine<'_> {
    fn call(&mut self, entry: BlockId, mut args: Vec<Value>) -> Vec<Value> {
        let mut frame = Frame::default();
        let mut block = entry;
        loop {
            assert!(self.fuel > 0, "CFG did not terminate");
            self.fuel -= 1;
            let data = self.data.blocks[block].clone();
            assert_eq!(args.len(), data.parameters.len());
            for (name, value) in data.parameters.iter().zip(args) {
                frame.locals.insert(name.clone(), value);
            }
            for instruction in self.data.bodies[data.body].instructions.clone() {
                self.instruction(instruction, &mut frame);
            }
            let edge = match data.exit {
                Exit::Return(body) => return self.tuple(body, &frame),
                Exit::Jump(edge) => edge,
                Exit::Branch { condition, yes, no } => {
                    if self.tuple(condition, &frame)[0].boolean() {
                        yes
                    } else {
                        no
                    }
                }
            };
            args = self.tuple(edge.arguments, &frame);
            block = edge.target;
        }
    }

    fn tuple(&self, id: BodyId, frame: &Frame) -> Vec<Value> {
        self.data.bodies[id].results.iter().map(|v| self.value(v, frame)).collect()
    }

    fn instruction(&mut self, instruction: Instruction, frame: &mut Frame) {
        match instruction {
            Instruction::BindParameter(id, value) => {
                frame.parameters.insert(id, self.value(&value, frame));
            }
            Instruction::BindExpression(id, value) => {
                frame.expressions.insert(id, self.value(&value, frame));
            }
            Instruction::BindResult(id, value) => {
                frame.operations.insert(id, self.value(&value, frame));
            }
            Instruction::Call {
                function,
                arguments,
                results,
            } => {
                let args = arguments.iter().map(|v| self.value(v, frame)).collect();
                let values = self.call(function, args);
                assert_eq!(results.len(), values.len());
                frame.locals.extend(results.into_iter().zip(values));
            }
            Instruction::Evaluate(op) => {
                let value = match self.data.operations[op].kind {
                    OperationKind::Index { array, index } => {
                        self.source(array, frame).at(self.source(index, frame).int() as usize)
                    }
                    ref other => panic!("unsupported source effect in oracle: {other:?}"),
                };
                frame.operations.insert(op, value);
            }
            Instruction::Allocate(id) => {
                let buffer = &self.data.buffers[id];
                let n = self.value(&buffer.length, frame).int() as usize;
                let values = Value::values(vec![Value::Uninitialized; n]);
                if buffer.storage == Storage::Function {
                    frame.buffers.insert(id, values);
                } else {
                    self.buffers.insert(id, values);
                }
            }
            Instruction::Load {
                result,
                buffer,
                index,
            } => {
                let value = self.value(&buffer, frame).at(self.value(&index, frame).int() as usize);
                frame.locals.insert(result, value);
            }
            Instruction::Store { buffer, index, value } => {
                self.value(&buffer, frame).store(
                    self.value(&index, frame).int() as usize,
                    self.value(&value, frame),
                );
            }
            Instruction::Dispatch(id) => {
                let dispatch = &self.data.dispatches[id];
                let groups = self.value(&self.data.grids[dispatch.grid].groups[0], frame).int();
                let FunctionKind::Kernel([width, 1, 1]) =
                    self.data.blocks[dispatch.kernel].interface.as_ref().unwrap().kind
                else {
                    panic!("kernel");
                };
                let args: Vec<_> = dispatch.captures.iter().map(|&id| self.source(id, frame)).collect();
                self.invocations = groups as u32 * width;
                // Reverse invocation order catches accidental inter-invocation
                // dependence within a dispatch. Only dispatch boundaries sync.
                for i in (0..groups as u32 * width).rev() {
                    self.invocation = i;
                    assert!(self.call(dispatch.kernel, args.clone()).is_empty());
                }
            }
        }
    }

    fn value(&self, code: &Code, frame: &Frame) -> Value {
        match code {
            Code::Int(i) => Value::Int(i64::from(*i)),
            Code::Local(name) => {
                frame.locals.get(name).unwrap_or_else(|| panic!("unbound local {name}")).clone()
            }
            Code::Source(id) => self.source(*id, frame),
            Code::Array(array) => self.array(array, frame),
            Code::Buffer(id) => {
                if let Storage::External(expr) = self.data.buffers[*id].storage {
                    self.source(expr, frame)
                } else {
                    frame
                        .buffers
                        .get(id)
                        .or_else(|| self.buffers.get(id))
                        .unwrap_or_else(|| panic!("unallocated {id:?}"))
                        .clone()
                }
            }
            Code::Tuple(fields) => Value::Tuple(fields.iter().map(|v| self.value(v, frame)).collect()),
            Code::Field(tuple, index) => field(self.value(tuple, frame), *index),
            Code::Primitive("global_id", _) => Value::Int(i64::from(self.invocation)),
            Code::Primitive("global_size", _) => Value::Int(i64::from(self.invocations)),
            Code::Primitive(name, args) => {
                primitive(name, args.iter().map(|v| self.value(v, frame)).collect())
            }
        }
    }

    fn array(&self, array: &Array, frame: &Frame) -> Value {
        match array {
            Array::Value(id) => self.source(*id, frame),
            Array::Literal(ids) => Value::values(ids.iter().map(|&id| self.source(id, frame)).collect()),
            Array::Range { start, len, step } => {
                let start = self.source(*start, frame).int();
                let n = self.source(*len, frame).int();
                let step = step.map(|id| self.source(id, frame).int()).unwrap_or(1);
                Value::array((0..n).map(|i| start + i * step))
            }
            Array::Zip(arrays) => {
                let arrays: Vec<_> = arrays.iter().map(|a| self.array(a, frame)).collect();
                Value::values(
                    (0..arrays[0].elements().len())
                        .map(|i| Value::Tuple(arrays.iter().map(|a| a.at(i)).collect()))
                        .collect(),
                )
            }
        }
    }

    fn source(&self, id: ExprId, frame: &Frame) -> Value {
        if let Some(value) = frame.expressions.get(&id) {
            return value.clone();
        }
        match &self.data.expressions[id].kind {
            ExprKind::Int(text) => Value::Int(text.parse().unwrap()),
            ExprKind::Bool(value) => Value::Bool(*value),
            ExprKind::Unit => Value::Tuple(vec![]),
            ExprKind::Parameter(id) => {
                frame.parameters.get(id).unwrap_or_else(|| panic!("unbound parameter {id:?}")).clone()
            }
            ExprKind::OperationResult(id) => {
                frame.operations.get(id).unwrap_or_else(|| panic!("unbound operation {id:?}")).clone()
            }
            ExprKind::Array(array) => self.array(array, frame),
            ExprKind::Tuple(ids) => Value::Tuple(ids.iter().map(|&id| self.source(id, frame)).collect()),
            ExprKind::Project { tuple, index } => field(self.source(*tuple, frame), *index),
            ExprKind::Coerce(id) => self.source(*id, frame),
            ExprKind::If {
                condition,
                then_value,
                else_value,
            } => self.source(
                if self.source(*condition, frame).boolean() { *then_value } else { *else_value },
                frame,
            ),
            ExprKind::PureApp { function, args } => {
                let values = args.iter().map(|&id| self.source(id, frame)).collect();
                let name = match &self.data.expressions[*function].kind {
                    ExprKind::BinOp(name) | ExprKind::UnOp(name) => name.as_str(),
                    ExprKind::Builtin(id) => {
                        builtins::by_id(self.data.builtins[*id].builtin).raw.surface_name
                    }
                    other => panic!("unsupported pure function: {other:?}"),
                };
                primitive(name, values)
            }
            other => panic!("unsupported source value in oracle: {other:?}"),
        }
    }
}

fn field(tuple: Value, index: usize) -> Value {
    let Value::Tuple(fields) = tuple else {
        panic!("expected tuple: {tuple:?}");
    };
    fields[index].clone()
}

fn primitive(name: &str, args: Vec<Value>) -> Value {
    match name {
        "length" => Value::Int(args[0].elements().len() as i64),
        "index" => args[0].at(args[1].int() as usize),
        "slice" => Value::values(args[0].elements()[..args[1].int() as usize].to_vec()),
        "dimension" => {
            let mut value = args[0].clone();
            for _ in 0..args[1].int() {
                let values = value.elements();
                if values.is_empty() {
                    return Value::Int(0);
                }
                value = values[0].clone();
            }
            Value::Int(value.elements().len() as i64)
        }
        "bool_to_u32" => Value::Int(i64::from(args[0].boolean())),
        "!" => Value::Bool(!args[0].boolean()),
        "and" | "&&" => Value::Bool(args[0].boolean() && args[1].boolean()),
        "or" | "||" => Value::Bool(args[0].boolean() || args[1].boolean()),
        "+" | "add" => Value::Int(args[0].int() + args[1].int()),
        "-" if args.len() == 1 => Value::Int(-args[0].int()),
        "-" => Value::Int(args[0].int() - args[1].int()),
        "*" | "mul" => Value::Int(args[0].int() * args[1].int()),
        "/" | "div" => Value::Int(args[0].int() / args[1].int()),
        "%" | "rem" => Value::Int(args[0].int() % args[1].int()),
        "ceil_div" => Value::Int((args[0].int() + args[1].int() - 1) / args[1].int()),
        "max" => Value::Int(args[0].int().max(args[1].int())),
        "min" => Value::Int(args[0].int().min(args[1].int())),
        "lt" | "<" => Value::Bool(args[0].int() < args[1].int()),
        "<=" => Value::Bool(args[0].int() <= args[1].int()),
        "ge" | ">=" => Value::Bool(args[0].int() >= args[1].int()),
        ">" => Value::Bool(args[0].int() > args[1].int()),
        "==" => Value::Bool(args[0] == args[1]),
        "ne" | "!=" => Value::Bool(args[0] != args[1]),
        "u32" | "i32" => args[0].clone(),
        other => panic!("unsupported primitive in oracle: {other}"),
    }
}
