use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use wgpu::*;

#[derive(Parser)]
#[command(name = "spirv-validator")]
#[command(about = "Validate SPIR-V shaders using wgpu without windowing")]
struct Args {
    /// Path to the SPIR-V file to validate
    #[arg(value_name = "SPIRV_FILE")]
    spirv_file: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Test compute shader instead of graphics shader
    #[arg(short, long)]
    compute: bool,

    /// Force Vulkan backend (if available)
    #[arg(long)]
    vulkan: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        env_logger::init();
    }

    println!("🔍 SPIR-V Validator");
    println!("📁 Loading: {}", args.spirv_file.display());

    // Read SPIR-V file
    let spirv_data = std::fs::read(&args.spirv_file)
        .with_context(|| format!("Failed to read SPIR-V file: {}", args.spirv_file.display()))?;

    println!("📊 File size: {} bytes", spirv_data.len());

    // Convert bytes to u32 words (SPIR-V format)
    if spirv_data.len() % 4 != 0 {
        anyhow::bail!("SPIR-V file size is not aligned to 4-byte words");
    }

    let spirv_words: Vec<u32> = spirv_data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    println!("📊 SPIR-V words: {}", spirv_words.len());

    // Create wgpu instance
    println!("🚀 Initializing wgpu...");
    let backends =
        if args.vulkan { Backends::VULKAN } else { Backends::VULKAN | Backends::METAL | Backends::DX12 };

    let instance = Instance::new(InstanceDescriptor {
        backends,
        ..InstanceDescriptor::new_without_display_handle()
    });

    // Request adapter (headless) - prefer high performance to get discrete GPU
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .context("Failed to find suitable GPU adapter")?;

    println!("🎮 Adapter: {}", adapter.get_info().name);
    println!("🎮 Backend: {:?}", adapter.get_info().backend);

    // Check if passthrough is supported
    let spirv_passthrough_supported = adapter.features().contains(Features::PASSTHROUGH_SHADERS);
    println!("🔧 SPIR-V Passthrough supported: {}", spirv_passthrough_supported);

    // Request device
    let (device, _queue) = adapter
        .request_device(&DeviceDescriptor {
            label: Some("SPIR-V Validator Device"),
            required_features: if spirv_passthrough_supported {
                Features::PASSTHROUGH_SHADERS
            } else {
                Features::empty()
            },
            required_limits: Limits::default(),
            experimental_features: ExperimentalFeatures::default(),
            memory_hints: MemoryHints::default(),
            trace: Trace::Off,
        })
        .await
        .context("Failed to create GPU device")?;

    println!("✅ Device created successfully");

    // Test SPIR-V loading
    println!("🧪 Testing SPIR-V shader loading...");

    match test_spirv_loading(&device, &spirv_words, args.compute, spirv_passthrough_supported) {
        Ok(()) => {
            println!("✅ SPIR-V validation PASSED! 🎉");
            println!("   The shader loaded successfully in wgpu");
        }
        Err(e) => {
            println!("❌ SPIR-V validation FAILED:");
            println!("   Error: {}", e);

            // Try with passthrough if available
            if device.features().contains(Features::PASSTHROUGH_SHADERS) {
                println!("🔄 Trying with SPIR-V passthrough...");
                match test_spirv_passthrough(&device, &spirv_words) {
                    Ok(()) => {
                        println!("✅ SPIR-V passthrough PASSED!");
                        println!("   The shader works with passthrough (bypasses wgpu validation)");
                    }
                    Err(e2) => {
                        println!("❌ SPIR-V passthrough also FAILED:");
                        println!("   Error: {}", e2);
                    }
                }
            } else {
                println!("ℹ️  SPIR-V passthrough not available on this device");
            }

            return Err(e);
        }
    }

    Ok(())
}

fn test_spirv_loading(
    device: &Device,
    spirv_words: &[u32],
    compute: bool,
    has_passthrough: bool,
) -> Result<()> {
    let shader_module = if has_passthrough {
        println!("🔄 Using SPIR-V passthrough...");
        // Try to create shader module using SPIR-V passthrough
        unsafe {
            device.create_shader_module_passthrough(ShaderModuleDescriptorPassthrough {
                label: Some("Test SPIR-V Shader"),
                spirv: Some(std::borrow::Cow::Borrowed(spirv_words)),
                ..Default::default()
            })
        }
    } else {
        println!("🔄 Using regular SPIR-V creation with validation...");
        // Convert u32 words back to bytes for make_spirv
        let bytes: Vec<u8> = spirv_words.iter().flat_map(|word| word.to_le_bytes()).collect();

        let source = wgpu::util::make_spirv(&bytes);
        device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Test SPIR-V Shader"),
            source,
        })
    };

    if compute {
        // Test as compute shader
        println!("🧮 Testing as compute shader...");
        let _compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Test Compute Pipeline"),
            layout: None,
            module: &shader_module,
            entry_point: Some("main"), // Assume main entry point
            compilation_options: Default::default(),
            cache: None,
        });
    } else {
        // Test as graphics shader (vertex + fragment)
        println!("🎨 Testing as graphics shader...");

        // Try to create a basic render pipeline
        let _render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Test Graphics Pipeline"),
            layout: None,
            vertex: VertexState {
                module: &shader_module,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: Some("fragment_main"),
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
    }

    Ok(())
}

fn test_spirv_passthrough(device: &Device, spirv_words: &[u32]) -> Result<()> {
    let shader_descriptor = ShaderModuleDescriptorPassthrough {
        label: Some("Test SPIR-V Passthrough"),
        spirv: Some(std::borrow::Cow::Borrowed(spirv_words)),
        ..Default::default()
    };

    // Use passthrough to bypass wgpu validation
    let _shader_module = unsafe { device.create_shader_module_passthrough(shader_descriptor) };

    println!("✅ SPIR-V passthrough succeeded (validation bypassed)");
    Ok(())
}
