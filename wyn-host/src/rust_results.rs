//! Named Rust output handles and storage-layout-aware result readers.
use crate::{Entry, FrameResourceKind, HostError, Program, ResultKind, ResultLayout, ResultScalar};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use std::collections::BTreeSet;

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

fn field_name(source: &str) -> Ident {
    let name = name(source);
    match syn::parse_str::<Ident>(&name) {
        Ok(ident) => ident,
        Err(_) => format_ident!("field_{}", name),
    }
}

fn type_name(source: &str) -> Ident {
    let name = name(source);
    let mut result = String::new();
    let mut uppercase = true;
    for c in name.chars() {
        if c == '_' {
            uppercase = true;
        } else {
            result.push(if uppercase { c.to_ascii_uppercase() } else { c });
            uppercase = false;
        }
    }
    if result.is_empty() || result.starts_with(|c: char| c.is_ascii_digit()) {
        result.insert_str(0, "Output");
    }
    // These names occur in the generated module independently of result types.
    match result.as_str() {
        "Self" | "HostError" | "Buffer" | "Texture" | "Device" | "Queue" | "Sampler" => {
            result.insert_str(0, "Wyn");
        }
        _ => {}
    }
    format_ident!("{}", result)
}

pub(crate) struct RustResults {
    pub declaration: TokenStream,
    pub handle: Ident,
    pub values: Vec<TokenStream>,
}

impl Program {
    pub(crate) fn rust_results(&self, entry: &Entry) -> Result<RustResults, HostError> {
        let result_name = type_name(&entry.name);
        let handle = format_ident!("{}Output", result_name);
        let reader = format_ident!("read_{}", name(&entry.name));
        let mut declarations = vec![];
        let mut handle_fields = vec![];
        let mut result_fields = vec![];
        let mut values = vec![];
        let mut methods = vec![];
        let mut reader_params = vec![];
        let mut reader_values = vec![];
        let mut ranges = vec![];
        let mut requests = vec![];
        let mut decoders = vec![];
        let mut names = BTreeSet::new();
        let mut source_results =
            self.interface.source_results.iter().filter(|r| r.entry == entry.name).collect::<Vec<_>>();
        source_results.sort_by_key(|r| r.result);
        let result_kind = source_results.first().map(|r| r.kind).unwrap_or(ResultKind::Value);
        for (index, &resource) in entry.results.iter().enumerate() {
            let descriptor = source_results.get(index);
            let source_name = descriptor
                .map(|r| r.name.as_str())
                .unwrap_or(&self.interface.frame_graph.resources[resource.0].name);
            let field = field_name(source_name);
            if !names.insert(field.to_string()) {
                return Err(HostError::Invalid(format!(
                    "duplicate Rust result name {field} in {}",
                    entry.name
                )));
            }
            let local = format_ident!("resource_{}", resource.0);
            let kind = self.interface.frame_graph.resources[resource.0].kind;
            match kind {
                FrameResourceKind::StorageBuffer | FrameResourceKind::Uniform => {
                    let Some(descriptor) = descriptor else {
                        return Err(HostError::Invalid(format!(
                            "missing result layout for {source_name}"
                        )));
                    };
                    let value_name = if entry.results.len() == 1 && result_kind == ResultKind::Value {
                        result_name.clone()
                    } else {
                        format_ident!("{}{}", result_name, type_name(source_name))
                    };
                    let byte_range = format_ident!("result_{}_range", index);
                    let decoded = format_ident!("result_{}_value", index);
                    let count = format_ident!("result_{}_count", index);
                    let (ty, decode) = match &descriptor.layout {
                        ResultLayout::Array {
                            element,
                            stride,
                            length,
                        } => {
                            let element_name = format_ident!("{}Element", value_name);
                            let (element_ty, decode) =
                                rust_value(element, &element_name, quote!(offset), &mut declarations)?;
                            let range = format_ident!("{}_elements", name(source_name));
                            reader_params.push(quote!(#range:std::ops::Range<u32>));
                            let length_check = length.map(|length| quote! {
                                if #range.end.checked_sub(#range.start) != Some(#length) {
                                    return Err(HostError::Invalid(format!("{} requires {} elements", #source_name, #length)));
                                }
                            });
                            ranges.push(quote! {
                                #length_check
                                let #byte_range = support::readback::element_range(#range.clone(), #stride, output.#field.size())?;
                                let #count = #range.end - #range.start;
                            });
                            (
                                quote!(Vec<#element_ty>),
                                quote!(support::readback::array(bytes, #count, #stride, |bytes,offset| Ok(#decode))?),
                            )
                        }
                        layout => {
                            let (ty, decode) =
                                rust_value(layout, &value_name, quote!(0u64), &mut declarations)?;
                            let Some(size) = layout.byte_size() else {
                                return Err(HostError::Invalid(format!(
                                    "result {source_name} has no fixed storage size"
                                )));
                            };
                            let size = u64::from(size);
                            ranges.push(quote!(let #byte_range = 0..#size;));
                            (ty, decode)
                        }
                    };
                    let accessor = format_ident!("{}_buffer", name(source_name));
                    methods.push(quote! {
                        /// Borrow this result's backing buffer for subsequent GPU work.
                        pub fn #accessor(&self) -> &Buffer { &self.#field }
                    });
                    requests.push(quote!((&output.#field, #byte_range)));
                    decoders.push(quote! {
                        let Some(bytes) = spans.next() else {
                            return Err(HostError::Invalid("missing result readback span".into()));
                        };
                        let bytes = bytes.as_slice();
                        let #decoded = #decode;
                    });
                    handle_fields.push(quote!(#field:Buffer));
                    values.push(quote!(#field:Buffer::clone(&#local)));
                    result_fields.push((field.clone(), ty));
                    reader_values.push(quote!(#decoded));
                }
                FrameResourceKind::Texture | FrameResourceKind::StorageTexture => {
                    handle_fields.push(quote!(#field:Texture));
                    values.push(quote!(#field:Texture::clone(&#local)));
                    result_fields.push((field.clone(), quote!(Texture)));
                    reader_values.push(quote!(Texture::clone(output.#field())));
                    methods.push(quote! {
                        /// Borrow the GPU texture for presentation or subsequent GPU work.
                        pub fn #field(&self)->&Texture { &self.#field }
                    });
                }
                FrameResourceKind::PushConstant | FrameResourceKind::Sampler => {
                    return Err(HostError::Invalid(format!(
                        "unsupported result resource {source_name}"
                    )));
                }
            }
        }
        let (result_type, result_value) = if result_fields.is_empty() {
            (quote!(()), quote!(()))
        } else if result_kind == ResultKind::TupleField {
            let types = result_fields.iter().map(|(_, ty)| ty);
            (quote!((#(#types,)*)), quote!((#(#reader_values,)*)))
        } else if result_fields.len() == 1 && result_kind == ResultKind::Value {
            (result_fields[0].1.clone(), quote!(#(#reader_values)*))
        } else {
            let fields = result_fields.iter().map(|(field, ty)| quote!(pub #field:#ty));
            let values =
                result_fields.iter().zip(&reader_values).map(|((field, _), value)| quote!(#field:#value));
            declarations.push(quote!(pub struct #result_name {#(#fields),*}));
            (quote!(#result_name), quote!(#result_name {#(#values),*}))
        };
        let (context, readback) = if requests.is_empty() {
            (quote!(), quote!())
        } else {
            (
                quote!(device:&Device,queue:&Queue,),
                quote! {
                    #(#ranges)*
                    let readback = support::read_buffers(device,queue,&[#(#requests),*])?;
                    let mut spans = readback.iter();
                    #(#decoders)*
                },
            )
        };
        let output_param = if result_fields.is_empty() { quote!(_output) } else { quote!(output) };
        let reader_doc = format!("Read all results of {} in one GPU copy submission and one wait. Buffer results require COPY_SRC and are decoded into Rust values; textures retain their GPU handles. Array ranges specify the returned views in their backing buffers.", entry.name);
        Ok(RustResults {
            declaration: quote! {
                #(#declarations)*
                pub struct #handle {#(#handle_fields),*}
                impl #handle {#(#methods)*}
                #[doc = #reader_doc]
                pub fn #reader(#context #output_param:&#handle,#(#reader_params),*) -> Result<#result_type,HostError> {
                    #readback
                    Ok(#result_value)
                }
            },
            handle,
            values,
        })
    }
}

fn rust_value(
    layout: &ResultLayout,
    name: &Ident,
    offset: TokenStream,
    declarations: &mut Vec<TokenStream>,
) -> Result<(TokenStream, TokenStream), HostError> {
    Ok(match layout {
        ResultLayout::Scalar(scalar) => {
            let ty = match scalar {
                ResultScalar::I8 => quote!(i8),
                ResultScalar::I16 => quote!(i16),
                ResultScalar::I32 => quote!(i32),
                ResultScalar::I64 => quote!(i64),
                ResultScalar::U8 => quote!(u8),
                ResultScalar::U16 => quote!(u16),
                ResultScalar::U32 => quote!(u32),
                ResultScalar::U64 => quote!(u64),
                ResultScalar::F32 => quote!(f32),
                ResultScalar::F64 => quote!(f64),
                ResultScalar::Bool => quote!(bool),
            };
            let bytes = scalar.bytes() as usize;
            let decode = if *scalar == ResultScalar::Bool {
                quote!(u32::from_le_bytes(support::readback::bytes::<4>(bytes,#offset)?) != 0)
            } else {
                quote!(#ty::from_le_bytes(support::readback::bytes::<#bytes>(bytes,#offset)?))
            };
            (ty, decode)
        }
        ResultLayout::Sequence {
            element,
            count,
            stride,
        } => {
            let element_name = format_ident!("{}Item", name);
            let (ty, decode) = rust_value(element, &element_name, quote!(item_offset), declarations)?;
            let count_usize = *count as usize;
            (
                quote!([#ty;#count_usize]),
                quote!({
                    let base = #offset;
                    let mut items = Vec::new();
                    items.try_reserve_exact(#count_usize).map_err(|e| HostError::Invalid(e.to_string()))?;
                    for index in 0..#count {
                    let Some(item_offset) = base.checked_add(u64::from(index) * u64::from(#stride)) else {
                        return Err(HostError::Invalid("result item offset overflow".into()));
                    };
                        items.push(#decode);
                    }
                    let items: [#ty;#count_usize] = items.try_into().map_err(|_| HostError::Invalid("invalid fixed result length".into()))?;
                    items
                }),
            )
        }
        ResultLayout::Record { fields, .. } | ResultLayout::Tuple { fields, .. } => {
            let mut names = BTreeSet::new();
            let mut types = vec![];
            let mut values = vec![];
            for field in fields {
                let ident = field_name(&field.name);
                if !names.insert(ident.to_string()) {
                    return Err(HostError::Invalid(format!(
                        "duplicate Rust field {ident} in {name}"
                    )));
                }
                let member_name = format_ident!("{}{}", name, type_name(&field.name));
                let field_offset = field.offset;
                let (ty, decode) = rust_value(
                    &field.layout,
                    &member_name,
                    quote!(support::readback::at(base,#field_offset)?),
                    declarations,
                )?;
                types.push((ident, ty));
                values.push(decode);
            }
            if matches!(layout, ResultLayout::Tuple { .. }) {
                let types = types.iter().map(|(_, ty)| ty);
                (quote!((#(#types,)*)), quote!({let base=#offset; (#(#values,)*)}))
            } else {
                let fields = types.iter().map(|(ident, ty)| quote!(pub #ident:#ty));
                let values = types.iter().zip(values).map(|((ident, _), value)| quote!(#ident:#value));
                declarations.push(quote!(#[derive(Debug,Clone,PartialEq)] pub struct #name {#(#fields),*}));
                (quote!(#name), quote!({let base=#offset; #name {#(#values),*}}))
            }
        }
        ResultLayout::Array { .. } => return Err(HostError::Invalid("nested runtime result array".into())),
        ResultLayout::Unsupported(description) => {
            return Err(HostError::Invalid(format!(
                "unsupported Rust result layout: {description}"
            )))
        }
    })
}
