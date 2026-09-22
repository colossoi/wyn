use crate::{FrameResourceKind, HostError, Program, ScalarExpr, ScalarTask, ScalarType};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Index;

impl Program {
    fn whl_scalar(&self, pipeline: usize, value: &ScalarExpr) -> Result<String, HostError> {
        Ok(match value {
            ScalarExpr::I32(n) => format!("(i32 {n})"),
            ScalarExpr::U32(n) => format!("(u32 {n})"),
            ScalarExpr::F32(bits) => format!("(wyn-f32-bits (u32 {bits}))"),
            ScalarExpr::Bool(b) => if *b { "t" } else { "nil" }.into(),
            ScalarExpr::Local(name) => name.clone(),
            ScalarExpr::Read { source, offset, ty } => format!(
                "(gpu-read-scalar resource-{} {offset} '{})",
                self.scalar_resource(pipeline, source)?.0,
                ty.name()
            ),
            ScalarExpr::Apply { op, ty, args } => {
                let args =
                    args.iter().map(|a| self.whl_scalar(pipeline, a)).collect::<Result<Vec<_>, _>>()?;
                let function = match op.as_str() {
                    "add" => "+".into(),
                    "sub" => "-".into(),
                    "mul" => "*".into(),
                    _ => format!("wyn-{}-{op}", ty.name()),
                };
                format!("({function} {})", args.join(" "))
            }
            ScalarExpr::If { condition, yes, no } => format!(
                "(if {} {} {})",
                self.whl_scalar(pipeline, condition)?,
                self.whl_scalar(pipeline, yes)?,
                self.whl_scalar(pipeline, no)?
            ),
            ScalarExpr::Let { bindings, result } => {
                let bindings = bindings
                    .iter()
                    .map(|(name, value)| Ok(format!("({name} {})", self.whl_scalar(pipeline, value)?)))
                    .collect::<Result<Vec<_>, HostError>>()?;
                format!(
                    "(let* ({}) {})",
                    bindings.join(" "),
                    self.whl_scalar(pipeline, result)?
                )
            }
            ScalarExpr::Loop {
                name,
                initial,
                condition,
                step,
            } => format!(
                "(do (({name} {} {})) ((not {}) {name}))",
                self.whl_scalar(pipeline, initial)?,
                self.whl_scalar(pipeline, step)?,
                self.whl_scalar(pipeline, condition)?
            ),
            ScalarExpr::Tuple(fields) => format!(
                "(list {})",
                fields
                    .iter()
                    .map(|a| self.whl_scalar(pipeline, a))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(" ")
            ),
            ScalarExpr::Field { tuple, index } => {
                format!("(nth {index} {})", self.whl_scalar(pipeline, tuple)?)
            }
        })
    }

    pub(crate) fn whl_scalar_task(&self, pipeline: usize, task: &ScalarTask) -> Result<String, HostError> {
        Ok(format!(
            "(gpu-write-scalar resource-{} {} '{} {})",
            self.scalar_resource(pipeline, &task.destination)?.0,
            task.offset,
            task.ty.name(),
            self.whl_scalar(pipeline, &task.value)?
        ))
    }

    fn rust_scalar(&self, pipeline: usize, value: &ScalarExpr) -> Result<TokenStream, HostError> {
        Ok(match value {
            ScalarExpr::I32(n) => quote!(#n),
            ScalarExpr::U32(n) => quote!(#n),
            ScalarExpr::F32(bits) => quote!(f32::from_bits(#bits)),
            ScalarExpr::Bool(b) => quote!(#b),
            ScalarExpr::Local(name) => {
                let name = format_ident!("{}", name.replace('-', "_"));
                quote!(#name)
            }
            ScalarExpr::Read { source, offset, ty } => {
                let resource = self.scalar_resource(pipeline, source)?;
                let name = format_ident!("resource_{}", resource.0);
                let primitive =
                    format_ident!("{}", if *ty == ScalarType::Bool { "u32" } else { ty.name() });
                let read = if self.interface.frame_graph.resources[resource.0].kind
                    == FrameResourceKind::PushConstant
                {
                    quote!(#primitive::from_le_bytes(support::scalar_bytes(#name, #offset)?))
                } else {
                    let reader = format_ident!("read_{}", primitive);
                    quote!(support::#reader(device, queue, &#name, #offset)?)
                };
                if *ty == ScalarType::Bool {
                    quote!(#read != 0)
                } else {
                    read
                }
            }
            ScalarExpr::Apply { op, ty, args } => {
                let args =
                    args.iter().map(|a| self.rust_scalar(pipeline, a)).collect::<Result<Vec<_>, _>>()?;
                rust_operation(op, *ty, &args)?
            }
            ScalarExpr::If { condition, yes, no } => {
                let condition = self.rust_scalar(pipeline, condition)?;
                let yes = self.rust_scalar(pipeline, yes)?;
                let no = self.rust_scalar(pipeline, no)?;
                quote!(if #condition { #yes } else { #no })
            }
            ScalarExpr::Let { bindings, result } => {
                let bindings = bindings
                    .iter()
                    .map(|(name, value)| {
                        let name = format_ident!("{}", name.replace('-', "_"));
                        let value = self.rust_scalar(pipeline, value)?;
                        Ok(quote!(let #name = #value;))
                    })
                    .collect::<Result<Vec<_>, HostError>>()?;
                let result = self.rust_scalar(pipeline, result)?;
                quote!({ #(#bindings)* #result })
            }
            ScalarExpr::Loop {
                name,
                initial,
                condition,
                step,
            } => {
                let name = format_ident!("{}", name.replace('-', "_"));
                let initial = self.rust_scalar(pipeline, initial)?;
                let condition = self.rust_scalar(pipeline, condition)?;
                let step = self.rust_scalar(pipeline, step)?;
                quote!({let mut #name = #initial; while #condition { #name = #step; } #name})
            }
            ScalarExpr::Tuple(fields) => {
                let fields =
                    fields.iter().map(|a| self.rust_scalar(pipeline, a)).collect::<Result<Vec<_>, _>>()?;
                quote!((#(#fields,)*))
            }
            ScalarExpr::Field { tuple, index } => {
                let tuple = self.rust_scalar(pipeline, tuple)?;
                let index = Index::from(*index);
                quote!((#tuple).#index)
            }
        })
    }

    pub(crate) fn rust_scalar_task(
        &self,
        pipeline: usize,
        task: &ScalarTask,
    ) -> Result<TokenStream, HostError> {
        let resource = format_ident!(
            "resource_{}",
            self.scalar_resource(pipeline, &task.destination)?.0
        );
        let value = self.rust_scalar(pipeline, &task.value)?;
        let offset = u64::from(task.offset);
        let bytes = if task.ty == ScalarType::Bool {
            quote!(u32::from(value).to_le_bytes())
        } else {
            quote!(value.to_le_bytes())
        };
        Ok(quote!({let value = #value; queue.write_buffer(&#resource, #offset, &#bytes);}))
    }
}

fn rust_operation(op: &str, ty: ScalarType, args: &[TokenStream]) -> Result<TokenStream, HostError> {
    let integer = matches!(ty, ScalarType::I32 | ScalarType::U32);
    Ok(match (op, args) {
        ("add" | "sub" | "mul", [a, b]) if integer => {
            let method = format_ident!("checked_{op}");
            quote!({let Some(value) = (#a).#method(#b) else {return Err(HostError::Invalid("scalar arithmetic overflow".into()));}; value})
        }
        ("add" | "sub" | "mul", [a, b]) => {
            let value = match op {
                "add" => quote!((#a) + (#b)),
                "sub" => quote!((#a) - (#b)),
                _ => quote!((#a) * (#b)),
            };
            quote!({let value = #value; if !value.is_finite() {return Err(HostError::Invalid("non-finite scalar arithmetic".into()));} value})
        }
        ("div" | "rem", [a, b]) if integer => {
            let method = format_ident!("checked_{op}");
            quote!({let Some(value) = (#a).#method(#b) else {return Err(HostError::Invalid("invalid scalar division".into()));}; value})
        }
        ("div", [a, b]) => quote!((#a) / (#b)),
        ("rem", [a, b]) => quote!((#a) % (#b)),
        ("eq", [a, b]) => quote!((#a) == (#b)),
        ("ne", [a, b]) if ty == ScalarType::F32 => {
            quote!({let a = #a; let b = #b; !a.is_nan() && !b.is_nan() && a != b})
        }
        ("ne", [a, b]) => quote!((#a) != (#b)),
        ("lt", [a, b]) => quote!((#a) < (#b)),
        ("le", [a, b]) => quote!((#a) <= (#b)),
        ("gt", [a, b]) => quote!((#a) > (#b)),
        ("ge", [a, b]) => quote!((#a) >= (#b)),
        ("and", [a, b]) => quote!((#a) & (#b)),
        ("or", [a, b]) => quote!((#a) | (#b)),
        ("xor", [a, b]) => quote!((#a) ^ (#b)),
        ("shl" | "shr", [a, b]) => {
            let method = format_ident!("wrapping_{op}");
            quote!((#a).#method((#b) as u32))
        }
        ("neg", [a]) if integer => quote!((#a).wrapping_neg()),
        ("neg", [a]) => quote!(-(#a)),
        ("not", [a]) => quote!(!(#a)),
        ("to-i32", [a]) => quote!((#a) as i32),
        ("to-u32", [a]) => quote!((#a) as u32),
        ("to-f32", [a]) => quote!((#a) as f32),
        ("abs", [a]) if ty == ScalarType::I32 => quote!((#a).wrapping_abs()),
        ("sign", [a]) if ty == ScalarType::I32 => quote!((#a).signum()),
        ("sign", [a]) => quote!({ let value = #a; if value == 0.0 { 0.0f32 } else { value.signum() } }),
        ("fract", [a]) => quote!({let value = #a; value - value.floor()}),
        ("rsqrt", [a]) => quote!(1.0f32 / (#a).sqrt()),
        ("round-even", [a]) => quote!((#a).round_ties_even()),
        (
            "round" | "trunc" | "abs" | "floor" | "ceil" | "sin" | "cos" | "tan" | "asin" | "acos" | "atan"
            | "sinh" | "cosh" | "tanh" | "asinh" | "acosh" | "atanh" | "exp" | "exp2" | "sqrt" | "log"
            | "log2" | "radians" | "degrees" | "isnan" | "isinf",
            [a],
        ) => {
            let method = format_ident!(
                "{}",
                match op {
                    "log" => "ln",
                    "radians" => "to_radians",
                    "degrees" => "to_degrees",
                    "isnan" => "is_nan",
                    "isinf" => "is_infinite",
                    _ => op,
                }
            );
            quote!((#a).#method())
        }
        ("min" | "max" | "atan2" | "pow", [a, b]) => {
            let method = format_ident!("{}", if op == "pow" { "powf" } else { op });
            quote!((#a).#method(#b))
        }
        _ => return Err(HostError::Invalid(format!("unsupported scalar operation {op}"))),
    })
}
