// CPU-only shader parsing and validation. This binary does not link WGPU,
// request an adapter, create a GPU device, or dispatch any shader work.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected a WGSL path")?;
    let source = std::fs::read_to_string(&path)?;
    let module = naga::front::wgsl::parse_str(&source)
        .map_err(|error| error.emit_to_string(&source))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|error| error.emit_to_string(&source))?;
    println!("CPU validation passed: {path}");
    Ok(())
}
