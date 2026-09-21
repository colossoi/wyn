//! Rust output descriptors carrying resources and source value layouts.
use crate::rust_wgpu::resource;
use crate::{Entry, FrameResourceKind, HostError, Program, ResultKind, ResultLayout, ResultScalar};
use proc_macro2::TokenStream;
use quote::quote;

pub(crate) fn name(source: &str) -> String {
    let mut result = String::new();
    for (i, c) in source.chars().enumerate() {
        if c.is_ascii_alphabetic() || c == '_' || (i > 0 && c.is_ascii_digit()) {
            result.push(c);
        } else {
            result.push_str(&format!("_u{:x}_", u32::from(c)));
        }
    }
    if result.is_empty() {
        "output".into()
    } else {
        result
    }
}

impl Program {
    pub(crate) fn rust_results(&self, entry: &Entry) -> Result<TokenStream, HostError> {
        let mut values = vec![];
        let mut source_results =
            self.interface.source_results.iter().filter(|r| r.entry == entry.name).collect::<Vec<_>>();
        source_results.sort_by_key(|r| r.result);
        for (index, &id) in entry.results.iter().enumerate() {
            let descriptor = source_results.get(index);
            let source_name = descriptor
                .map(|r| r.name.as_str())
                .unwrap_or(&self.interface.frame_graph.resources[id.0].name);
            let kind = match descriptor.map(|r| r.kind).unwrap_or(ResultKind::Value) {
                ResultKind::Value => quote!(ResultKind::Value),
                ResultKind::RecordField => quote!(ResultKind::RecordField),
                ResultKind::TupleField => quote!(ResultKind::TupleField),
            };
            let local = resource(id);
            let id = id.0;
            let value = match self.interface.frame_graph.resources[id].kind {
                FrameResourceKind::StorageBuffer | FrameResourceKind::Uniform => {
                    let Some(descriptor) = descriptor else {
                        return Err(HostError::Invalid(format!(
                            "missing result layout for {source_name}"
                        )));
                    };
                    let layout = rust_layout(&descriptor.layout);
                    let range = match descriptor.layout.byte_size() {
                        Some(bytes) => {
                            let bytes = u64::from(bytes);
                            quote!(BufferRange::Bytes { offset: 0, size: #bytes })
                        }
                        None => quote!(BufferRange::CallerProvided),
                    };
                    quote!(OutputResource::Buffer { buffer: Buffer::clone(&#local), layout: #layout, range: #range })
                }
                FrameResourceKind::Texture | FrameResourceKind::StorageTexture => {
                    quote!(OutputResource::Texture(Texture::clone(&#local)))
                }
                FrameResourceKind::PushConstant | FrameResourceKind::Sampler => {
                    return Err(HostError::Invalid(format!(
                        "unsupported result resource {source_name}"
                    )));
                }
            };
            values.push(quote!(OutputValue {name:#source_name,kind:#kind,resource_id:#id,resource:#value}));
        }
        let entry_name = &entry.name;
        Ok(quote!(OutputDescriptor {entry:#entry_name,values:Vec::from([#(#values),*])}))
    }
}

fn rust_layout(layout: &ResultLayout) -> TokenStream {
    match layout {
        ResultLayout::Scalar(scalar) => {
            let scalar = match scalar {
                ResultScalar::I8 => quote!(I8),
                ResultScalar::I16 => quote!(I16),
                ResultScalar::I32 => quote!(I32),
                ResultScalar::I64 => quote!(I64),
                ResultScalar::U8 => quote!(U8),
                ResultScalar::U16 => quote!(U16),
                ResultScalar::U32 => quote!(U32),
                ResultScalar::U64 => quote!(U64),
                ResultScalar::F32 => quote!(F32),
                ResultScalar::F64 => quote!(F64),
                ResultScalar::Bool => quote!(Bool),
            };
            quote!(ResultLayout::Scalar(ResultScalar::#scalar))
        }
        ResultLayout::Sequence {
            element,
            count,
            stride,
        } => {
            let element = rust_layout(element);
            quote!(ResultLayout::Sequence {element:Box::new(#element),count:#count,stride:#stride})
        }
        ResultLayout::Array {
            element,
            stride,
            length,
        } => {
            let element = rust_layout(element);
            let length = match length {
                Some(n) => quote!(Some(#n)),
                None => quote!(None),
            };
            quote!(ResultLayout::Array {element:Box::new(#element),stride:#stride,length:#length})
        }
        ResultLayout::Record { fields, size } | ResultLayout::Tuple { fields, size } => {
            let kind =
                if matches!(layout, ResultLayout::Record { .. }) { quote!(Record) } else { quote!(Tuple) };
            let fields = fields.iter().map(|field| {
                let name = &field.name;
                let offset = field.offset;
                let layout = rust_layout(&field.layout);
                quote!(ResultField {name:#name.into(),offset:#offset,layout:#layout})
            });
            quote!(ResultLayout::#kind {fields:Vec::from([#(#fields),*]),size:#size})
        }
        ResultLayout::Unsupported(description) => quote!(ResultLayout::Unsupported(#description.into())),
    }
}
