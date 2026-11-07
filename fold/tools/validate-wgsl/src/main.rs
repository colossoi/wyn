// CPU-only shader parsing and validation. This binary does not link WGPU,
// request an adapter, create a GPU device, or dispatch any shader work.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected a WGSL path")?;
    let source = std::fs::read_to_string(&path)?;
    let module = naga::front::wgsl::parse_str(&source)
        .map_err(|error| error.emit_to_string(&source))?;
    let info=naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|error| error.emit_to_string(&source))?;
    println!("CPU validation passed: {path}");
    let args=std::env::args().collect::<Vec<_>>();
    if let Some(i)=args.iter().position(|a|a=="--spirv") {let words=naga::back::spv::write_vec(&module,&info,&Default::default(),None)?;let bytes=words.iter().flat_map(|w|w.to_le_bytes()).collect::<Vec<_>>();std::fs::write(&args[i+1],bytes)?;}
    if let Some(i)=args.iter().position(|a|a=="--msl") {let (source,_)=naga::back::msl::write_string(&module,&info,&Default::default(),&Default::default())?;std::fs::write(&args[i+1],source)?;}
    Ok(())
}
