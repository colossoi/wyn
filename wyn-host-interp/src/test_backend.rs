use super::{Backend, Error, Number, Options, Program, Result, Value};
use std::collections::BTreeMap;

#[derive(Default)]
pub struct Trace {
    pub next: u64,
    pub buffers: BTreeMap<u64, Vec<u8>>,
    pub dispatches: Vec<(String, Vec<u32>)>,
    pub freed: Vec<u64>,
    pub scalar_writes: Vec<(u64, [u8; 4])>,
}

impl Trace {
    pub fn input(&mut self, bytes: Vec<u8>) -> Value {
        self.next += 1;
        self.buffers.insert(self.next, bytes);
        Value::Resource(self.next)
    }
}

impl Backend for Trace {
    fn call(&mut self, _: &Program, name: &str, args: &[Value]) -> Result<Value> {
        Ok(match name {
            "gpu-alloc" => self.input(vec![0; args[0].u64()? as usize]),
            "gpu-buffer-size" => Value::Number(Number::U64(self.buffers[&args[0].handle()?].len() as u64)),
            "gpu-free" => {
                let id = args[0].handle()?;
                self.freed.push(id);
                self.buffers.remove(&id);
                Value::Nil
            }
            "gpu-read-scalar" => {
                let bytes = &self.buffers[&args[0].handle()?];
                let offset = args[1].u64()? as usize;
                let word: [u8; 4] = bytes[offset..offset + 4].try_into().unwrap();
                Value::Number(match args[2].text()? {
                    "i32" => Number::I32(i32::from_le_bytes(word)),
                    "u32" => Number::U32(u32::from_le_bytes(word)),
                    "f32" => Number::F32(f32::from_le_bytes(word)),
                    "bool" => return Ok(Value::boolean(u32::from_le_bytes(word) != 0)),
                    _ => panic!("scalar type"),
                })
            }
            "gpu-write-scalar" => {
                let offset = args[1].u64()? as usize;
                let word = match args[2].text()? {
                    "i32" => (args[3].number()?.integer()? as i32).to_le_bytes(),
                    "u32" => args[3].u32()?.to_le_bytes(),
                    "f32" => (args[3].number()?.real() as f32).to_le_bytes(),
                    "bool" => u32::from(args[3].truth()).to_le_bytes(),
                    _ => panic!("scalar type"),
                };
                let buffer = args[0].handle()?;
                self.scalar_writes.push((buffer, word));
                self.buffers.get_mut(&buffer).unwrap()[offset..offset + 4].copy_from_slice(&word);
                Value::Nil
            }
            "gpu-dispatch" => {
                let options = Options::parse(&args[1..], &[":groups", ":args"])?;
                self.dispatches.push((
                    args[0].text()?.into(),
                    options.get(":groups")?.list()?.iter().map(Value::u32).collect::<Result<_>>()?,
                ));
                Value::Nil
            }
            _ => return Err(Error::Gpu(format!("unhandled test operation {name}"))),
        })
    }
}
