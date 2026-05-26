//! Vulkan helper abstractions for compute shaders

use anyhow::{Context, Result, anyhow};
use ash::vk;
use std::ffi::CStr;

/// Turn a Vulkan submit/wait result into an `anyhow` error, calling out a lost
/// device explicitly — the usual cause is a watchdog/TDR timeout from too much
/// work in one dispatch, which otherwise surfaces as a silently-zeroed buffer.
fn check_vk(result: std::result::Result<(), vk::Result>, what: &str) -> Result<()> {
    match result {
        Ok(()) => Ok(()),
        Err(vk::Result::ERROR_DEVICE_LOST) => Err(anyhow!(
            "{what}: GPU device lost — likely a watchdog/TDR timeout; lower --chunk-size"
        )),
        Err(e) => Err(anyhow!("{what}: {e:?}")),
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let severity = if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        "ERROR"
    } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        "WARN"
    } else {
        "INFO"
    };

    let kind = if message_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION) {
        "VALIDATION"
    } else if message_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE) {
        "PERFORMANCE"
    } else {
        "GENERAL"
    };

    let message = if p_callback_data.is_null() {
        "(no message)".to_string()
    } else {
        let data = &*p_callback_data;
        if data.p_message.is_null() {
            "(null message)".to_string()
        } else {
            CStr::from_ptr(data.p_message).to_string_lossy().into_owned()
        }
    };

    eprintln!("[Vulkan {}][{}] {}", severity, kind, message);
    vk::FALSE
}

/// Vulkan compute context - instance, device, queue
pub struct ComputeContext {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    device_name: String,
    _debug_utils: Option<ash::ext::debug_utils::Instance>,
    _debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl ComputeContext {
    pub fn new() -> Result<Self> {
        unsafe {
            let entry = ash::Entry::load().context("Failed to load Vulkan library")?;

            let app_info = vk::ApplicationInfo::default()
                .application_name(CStr::from_bytes_with_nul(b"tephra\0").unwrap())
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(CStr::from_bytes_with_nul(b"tephra\0").unwrap())
                .engine_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::API_VERSION_1_2);

            // Try to enable validation layer + debug utils
            let validation_layer = CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap();
            let available_layers = entry.enumerate_instance_layer_properties().unwrap_or_default();
            let has_validation = available_layers
                .iter()
                .any(|layer| CStr::from_ptr(layer.layer_name.as_ptr()) == validation_layer);

            let layer_names: Vec<*const i8> = if has_validation {
                eprintln!("  Validation layer enabled");
                vec![validation_layer.as_ptr()]
            } else {
                eprintln!("  Validation layer not available");
                vec![]
            };

            let ext_names: Vec<*const i8> =
                if has_validation { vec![ash::ext::debug_utils::NAME.as_ptr()] } else { vec![] };

            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_layer_names(&layer_names)
                .enabled_extension_names(&ext_names);

            let instance =
                entry.create_instance(&create_info, None).context("Failed to create Vulkan instance")?;

            // Set up debug messenger
            let (debug_utils, debug_messenger) = if has_validation {
                let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
                let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                            | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(vulkan_debug_callback));
                let messenger = debug_utils.create_debug_utils_messenger(&messenger_info, None).ok();
                (Some(debug_utils), messenger)
            } else {
                (None, None)
            };

            let (physical_device, queue_family_index, device_name) = select_compute_device(&instance)?;

            let device = create_logical_device(&instance, physical_device, queue_family_index)?;
            let queue = device.get_device_queue(queue_family_index, 0);

            Ok(Self {
                _entry: entry,
                instance,
                physical_device,
                device,
                queue,
                queue_family_index,
                device_name,
                _debug_utils: debug_utils,
                _debug_messenger: debug_messenger,
            })
        }
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn max_push_constants_size(&self) -> u32 {
        unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
                .limits
                .max_push_constants_size
        }
    }

    pub fn create_compute_pipeline(&self, spirv: &[u32], entry_name: &str) -> Result<ComputePipeline<'_>> {
        ComputePipeline::new(self, spirv, entry_name, 1, 0)
    }

    pub fn create_compute_pipeline_multi(
        &self,
        spirv: &[u32],
        entry_name: &str,
        binding_count: u32,
        push_constant_size: u32,
    ) -> Result<ComputePipeline<'_>> {
        ComputePipeline::new(self, spirv, entry_name, binding_count, push_constant_size)
    }
}

impl Drop for ComputeContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            if let (Some(ref debug_utils), Some(messenger)) = (&self._debug_utils, self._debug_messenger) {
                debug_utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// Find a physical device with compute capability
unsafe fn select_compute_device(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32, String)> {
    let physical_devices =
        instance.enumerate_physical_devices().context("Failed to enumerate physical devices")?;

    for pd in physical_devices {
        let queue_families = instance.get_physical_device_queue_family_properties(pd);

        if let Some((idx, _)) = queue_families
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
        {
            let props = instance.get_physical_device_properties(pd);
            let name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned();

            return Ok((pd, idx as u32, name));
        }
    }

    Err(anyhow!("No compute-capable device found"))
}

/// Check whether a physical device supports variablePointersStorageBuffer.
unsafe fn check_variable_pointers_support(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    let mut vp_features = vk::PhysicalDeviceVariablePointersFeatures::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut vp_features);
    instance.get_physical_device_features2(physical_device, &mut features2);
    vp_features.variable_pointers_storage_buffer == vk::TRUE
}

/// Create a logical device with a single compute queue
unsafe fn create_logical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> Result<ash::Device> {
    let queue_priorities = [1.0f32];
    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities);

    // Enable VariablePointersStorageBuffer if supported (required by compiled Wyn shaders)
    let has_vp = check_variable_pointers_support(instance, physical_device);
    let mut variable_pointers =
        vk::PhysicalDeviceVariablePointersFeatures::default().variable_pointers_storage_buffer(has_vp);

    if !has_vp {
        eprintln!(
            "Warning: device does not support variablePointersStorageBuffer; \
             some shaders may fail to compile"
        );
    }

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_create_info))
        .push_next(&mut variable_pointers);

    instance
        .create_device(physical_device, &device_create_info, None)
        .context("Failed to create logical device")
}

/// Storage buffer for compute shader I/O.
///
/// Host-visible buffers (`new`, `new_host_bytes`) live in
/// `HOST_VISIBLE | HOST_COHERENT` memory and can be mapped for upload /
/// download. Device-local buffers (`new_device_local`) live in
/// `DEVICE_LOCAL` memory the CPU never maps — used for inter-stage GPU
/// scratch like the miner's `partials` buffer, which the descriptor marks
/// `usage: "intermediate"`: phase 1 writes it, phase 2 reads it, and the
/// host neither seeds nor reads it.
pub struct StorageBuffer<'a> {
    ctx: &'a ComputeContext,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    len: usize,
    host_visible: bool,
}

impl<'a> StorageBuffer<'a> {
    /// Host-visible buffer of `len` f32s.
    pub fn new(ctx: &'a ComputeContext, len: usize) -> Result<Self> {
        Self::new_bytes(ctx, len * std::mem::size_of::<f32>(), true)
    }

    /// Host-visible buffer of `byte_size` bytes, for CPU readback.
    pub fn new_host_bytes(ctx: &'a ComputeContext, byte_size: usize) -> Result<Self> {
        Self::new_bytes(ctx, byte_size, true)
    }

    /// Device-local buffer of `byte_size` bytes the CPU never maps.
    pub fn new_device_local(ctx: &'a ComputeContext, byte_size: usize) -> Result<Self> {
        Self::new_bytes(ctx, byte_size, false)
    }

    fn new_bytes(ctx: &'a ComputeContext, byte_size: usize, host_visible: bool) -> Result<Self> {
        let size = byte_size as vk::DeviceSize;

        unsafe {
            let buffer_create_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer =
                ctx.device.create_buffer(&buffer_create_info, None).context("Failed to create buffer")?;

            let mem_requirements = ctx.device.get_buffer_memory_requirements(buffer);
            let required = if host_visible {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            } else {
                vk::MemoryPropertyFlags::DEVICE_LOCAL
            };
            let memory_type_index = find_memory_type(ctx, mem_requirements.memory_type_bits, required)?;

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_requirements.size)
                .memory_type_index(memory_type_index);

            let memory =
                ctx.device.allocate_memory(&alloc_info, None).context("Failed to allocate memory")?;

            ctx.device.bind_buffer_memory(buffer, memory, 0).context("Failed to bind buffer memory")?;

            Ok(Self {
                ctx,
                buffer,
                memory,
                size,
                len: byte_size / std::mem::size_of::<f32>(),
                host_visible,
            })
        }
    }

    /// Read the first `count` u32 words back (host-visible buffers only).
    pub fn read_u32(&self, count: usize) -> Result<Vec<u32>> {
        if !self.host_visible {
            return Err(anyhow!("read_u32 called on a device-local buffer"));
        }
        unsafe {
            let ptr = self
                .ctx
                .device
                .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
                .context("Failed to map memory")? as *const u32;
            let mut out = vec![0u32; count];
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), count);
            self.ctx.device.unmap_memory(self.memory);
            Ok(out)
        }
    }

    pub fn upload(&mut self, data: &[f32]) -> Result<()> {
        if data.len() != self.len {
            return Err(anyhow!(
                "Data length mismatch: got {}, expected {}",
                data.len(),
                self.len
            ));
        }

        unsafe {
            let ptr = self
                .ctx
                .device
                .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
                .context("Failed to map memory")? as *mut f32;

            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            self.ctx.device.unmap_memory(self.memory);
        }

        Ok(())
    }

    pub fn download(&self) -> Result<Vec<f32>> {
        unsafe {
            let ptr = self
                .ctx
                .device
                .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
                .context("Failed to map memory")? as *const f32;

            let mut output = vec![0.0f32; self.len];
            std::ptr::copy_nonoverlapping(ptr, output.as_mut_ptr(), self.len);
            self.ctx.device.unmap_memory(self.memory);

            Ok(output)
        }
    }
}

impl Drop for StorageBuffer<'_> {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.free_memory(self.memory, None);
            self.ctx.device.destroy_buffer(self.buffer, None);
        }
    }
}

/// Find a memory type satisfying `type_bits` and carrying all of `required`.
unsafe fn find_memory_type(
    ctx: &ComputeContext,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let memory_properties = ctx.instance.get_physical_device_memory_properties(ctx.physical_device);

    (0..memory_properties.memory_type_count)
        .find(|&i| {
            let supported = (type_bits & (1 << i)) != 0;
            let props = memory_properties.memory_types[i as usize].property_flags;
            supported && props.contains(required)
        })
        .ok_or_else(|| anyhow!("No suitable memory type found"))
}

/// Compute pipeline with descriptor set
pub struct ComputePipeline<'a> {
    ctx: &'a ComputeContext,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    shader_module: vk::ShaderModule,
    command_pool: vk::CommandPool,
    binding_count: u32,
    push_constant_size: u32,
}

impl<'a> ComputePipeline<'a> {
    fn new(
        ctx: &'a ComputeContext,
        spirv: &[u32],
        entry_name: &str,
        binding_count: u32,
        push_constant_size: u32,
    ) -> Result<Self> {
        unsafe {
            // Descriptor set layout with multiple bindings
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..binding_count)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();

            let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

            let descriptor_set_layout = ctx
                .device
                .create_descriptor_set_layout(&layout_info, None)
                .context("Failed to create descriptor set layout")?;

            // Push constant range (if any)
            let push_constant_ranges = if push_constant_size > 0 {
                vec![
                    vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .offset(0)
                        .size(push_constant_size),
                ]
            } else {
                vec![]
            };

            // Pipeline layout
            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                .push_constant_ranges(&push_constant_ranges);

            let pipeline_layout = ctx
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .context("Failed to create pipeline layout")?;

            // Shader module
            let shader_info = vk::ShaderModuleCreateInfo::default().code(spirv);

            let shader_module = ctx
                .device
                .create_shader_module(&shader_info, None)
                .context("Failed to create shader module")?;

            // Pipeline
            let entry_cstr = std::ffi::CString::new(entry_name)?;
            let stage_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(&entry_cstr);

            let pipeline_info =
                vk::ComputePipelineCreateInfo::default().stage(stage_info).layout(pipeline_layout);

            let pipelines = ctx
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| e)
                .context("Failed to create compute pipeline")?;

            let pipeline = pipelines[0];

            // Descriptor pool
            let pool_size = vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(binding_count);

            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(std::slice::from_ref(&pool_size));

            let descriptor_pool = ctx
                .device
                .create_descriptor_pool(&pool_info, None)
                .context("Failed to create descriptor pool")?;

            // Command pool
            let command_pool_info =
                vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family_index);

            let command_pool = ctx
                .device
                .create_command_pool(&command_pool_info, None)
                .context("Failed to create command pool")?;

            Ok(Self {
                ctx,
                pipeline,
                pipeline_layout,
                descriptor_set_layout,
                descriptor_pool,
                shader_module,
                command_pool,
                binding_count,
                push_constant_size,
            })
        }
    }

    /// Dispatch with a single buffer (legacy API).
    pub fn dispatch(&self, buffer: &StorageBuffer, num_workgroups: u32) -> Result<()> {
        self.dispatch_multi(&[buffer], [num_workgroups, 1, 1], &[])
    }

    /// Dispatch with multiple buffers, 3D workgroup dimensions, and optional push constants.
    pub fn dispatch_multi(
        &self,
        buffers: &[&StorageBuffer],
        workgroups: [u32; 3],
        push_constants: &[u8],
    ) -> Result<()> {
        if buffers.len() != self.binding_count as usize {
            return Err(anyhow!(
                "Expected {} buffers, got {}",
                self.binding_count,
                buffers.len()
            ));
        }

        if !push_constants.is_empty() && push_constants.len() != self.push_constant_size as usize {
            return Err(anyhow!(
                "Expected {} bytes of push constants, got {}",
                self.push_constant_size,
                push_constants.len()
            ));
        }

        self.dispatch_multi_internal(buffers, workgroups, push_constants)
    }

    fn dispatch_multi_internal(
        &self,
        buffers: &[&StorageBuffer],
        workgroups: [u32; 3],
        push_constants: &[u8],
    ) -> Result<()> {
        unsafe {
            // Allocate descriptor set
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));

            let descriptor_sets = self
                .ctx
                .device
                .allocate_descriptor_sets(&alloc_info)
                .context("Failed to allocate descriptor set")?;

            let descriptor_set = descriptor_sets[0];

            // Update descriptors for all buffers
            let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
                .iter()
                .map(|buf| vk::DescriptorBufferInfo::default().buffer(buf.buffer).offset(0).range(buf.size))
                .collect();

            // We need to keep buffer_info slices alive for the write descriptors
            let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();

            self.ctx.device.update_descriptor_sets(&writes, &[]);

            // Allocate command buffer
            let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let cmd_buffers = self
                .ctx
                .device
                .allocate_command_buffers(&cmd_alloc_info)
                .context("Failed to allocate command buffer")?;

            let cmd = cmd_buffers[0];

            // Record commands
            let begin_info =
                vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.ctx.device.begin_command_buffer(cmd, &begin_info)?;

            self.ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);

            self.ctx.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            // Push constants if any
            if !push_constants.is_empty() {
                self.ctx.device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }

            self.ctx.device.cmd_dispatch(cmd, workgroups[0], workgroups[1], workgroups[2]);

            self.ctx.device.end_command_buffer(cmd)?;

            // Submit and wait
            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));

            let fence = self.ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;

            check_vk(
                self.ctx.device.queue_submit(self.ctx.queue, &[submit_info], fence),
                "queue submit",
            )?;
            check_vk(
                self.ctx.device.wait_for_fences(&[fence], true, u64::MAX),
                "wait for fence",
            )?;
            check_vk(self.ctx.device.device_wait_idle(), "device wait idle")?;

            self.ctx.device.destroy_fence(fence, None);

            // Reset pools so they can be reused on the next dispatch
            self.ctx.device.reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())?;
            self.ctx
                .device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())?;

            Ok(())
        }
    }
}

impl Drop for ComputePipeline<'_> {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_command_pool(self.command_pool, None);
            self.ctx.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.ctx.device.destroy_pipeline(self.pipeline, None);
            self.ctx.device.destroy_shader_module(self.shader_module, None);
            self.ctx.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.ctx.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

/// A multi-stage compute pipeline: several entry points from one SPIR-V
/// module sharing a single descriptor-set layout and push-constant range.
/// One `dispatch` runs every stage in order within a single command
/// buffer, inserting a shader-write→shader-read memory barrier between
/// stages so an earlier stage's storage writes are visible to the next.
/// Drives the Wyn miner's `phase1_chunks → phase2_combine` reduce.
pub struct MultiStagePipeline<'a> {
    ctx: &'a ComputeContext,
    pipelines: Vec<vk::Pipeline>,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    shader_module: vk::ShaderModule,
    command_pool: vk::CommandPool,
    binding_count: u32,
    push_constant_size: u32,
}

impl<'a> MultiStagePipeline<'a> {
    pub fn new(
        ctx: &'a ComputeContext,
        spirv: &[u32],
        entry_names: &[&str],
        binding_count: u32,
        push_constant_size: u32,
    ) -> Result<Self> {
        unsafe {
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..binding_count)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            let descriptor_set_layout = ctx
                .device
                .create_descriptor_set_layout(&layout_info, None)
                .context("Failed to create descriptor set layout")?;

            let push_constant_ranges = if push_constant_size > 0 {
                vec![
                    vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .offset(0)
                        .size(push_constant_size),
                ]
            } else {
                vec![]
            };
            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                .push_constant_ranges(&push_constant_ranges);
            let pipeline_layout = ctx
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .context("Failed to create pipeline layout")?;

            let shader_info = vk::ShaderModuleCreateInfo::default().code(spirv);
            let shader_module = ctx
                .device
                .create_shader_module(&shader_info, None)
                .context("Failed to create shader module")?;

            // One pipeline per entry point, each sharing the layout above.
            // Built one at a time so each entry-name CString outlives its
            // own create call.
            let mut pipelines = Vec::with_capacity(entry_names.len());
            for name in entry_names {
                let entry_cstr = std::ffi::CString::new(*name)?;
                let stage_info = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(shader_module)
                    .name(&entry_cstr);
                let pipeline_info =
                    vk::ComputePipelineCreateInfo::default().stage(stage_info).layout(pipeline_layout);
                let created = ctx
                    .device
                    .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                    .map_err(|(_, e)| e)
                    .with_context(|| format!("Failed to create compute pipeline for '{}'", name))?;
                pipelines.push(created[0]);
            }

            let pool_size = vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(binding_count);
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(std::slice::from_ref(&pool_size));
            let descriptor_pool = ctx
                .device
                .create_descriptor_pool(&pool_info, None)
                .context("Failed to create descriptor pool")?;

            let command_pool_info =
                vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family_index);
            let command_pool = ctx
                .device
                .create_command_pool(&command_pool_info, None)
                .context("Failed to create command pool")?;

            Ok(Self {
                ctx,
                pipelines,
                pipeline_layout,
                descriptor_set_layout,
                descriptor_pool,
                shader_module,
                command_pool,
                binding_count,
                push_constant_size,
            })
        }
    }

    /// Run every stage once, in order. `buffers[i]` is bound at descriptor
    /// binding `i`; `stage_workgroups[s]` is stage `s`'s dispatch size;
    /// `push_constants` is shared by all stages.
    pub fn dispatch(
        &self,
        buffers: &[&StorageBuffer],
        stage_workgroups: &[[u32; 3]],
        push_constants: &[u8],
    ) -> Result<()> {
        if buffers.len() != self.binding_count as usize {
            return Err(anyhow!(
                "Expected {} buffers, got {}",
                self.binding_count,
                buffers.len()
            ));
        }
        if stage_workgroups.len() != self.pipelines.len() {
            return Err(anyhow!(
                "Expected {} stage dispatch sizes, got {}",
                self.pipelines.len(),
                stage_workgroups.len()
            ));
        }
        if !push_constants.is_empty() && push_constants.len() != self.push_constant_size as usize {
            return Err(anyhow!(
                "Expected {} bytes of push constants, got {}",
                self.push_constant_size,
                push_constants.len()
            ));
        }

        unsafe {
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));
            let descriptor_set = self
                .ctx
                .device
                .allocate_descriptor_sets(&alloc_info)
                .context("Failed to allocate descriptor set")?[0];

            let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
                .iter()
                .map(|buf| vk::DescriptorBufferInfo::default().buffer(buf.buffer).offset(0).range(buf.size))
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            self.ctx.device.update_descriptor_sets(&writes, &[]);

            let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = self
                .ctx
                .device
                .allocate_command_buffers(&cmd_alloc_info)
                .context("Failed to allocate command buffer")?[0];

            let begin_info =
                vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.ctx.device.begin_command_buffer(cmd, &begin_info)?;

            // The descriptor set and push constants persist across pipeline
            // binds (the layout is shared), so bind them once.
            self.ctx.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            if !push_constants.is_empty() {
                self.ctx.device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }

            let last = self.pipelines.len() - 1;
            for (s, pipeline) in self.pipelines.iter().enumerate() {
                self.ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, *pipeline);
                let wg = stage_workgroups[s];
                self.ctx.device.cmd_dispatch(cmd, wg[0], wg[1], wg[2]);
                if s != last {
                    // Make this stage's storage writes visible to the next.
                    let barrier = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ);
                    self.ctx.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    );
                }
            }

            self.ctx.device.end_command_buffer(cmd)?;

            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            let fence = self.ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            check_vk(
                self.ctx.device.queue_submit(self.ctx.queue, &[submit_info], fence),
                "queue submit",
            )?;
            check_vk(
                self.ctx.device.wait_for_fences(&[fence], true, u64::MAX),
                "wait for fence",
            )?;
            // Best-effort: some drivers only report a lost device here, after
            // recovering the fence on reset.
            check_vk(self.ctx.device.device_wait_idle(), "device wait idle")?;
            self.ctx.device.destroy_fence(fence, None);

            self.ctx.device.reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())?;
            self.ctx
                .device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())?;

            Ok(())
        }
    }
}

impl Drop for MultiStagePipeline<'_> {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_command_pool(self.command_pool, None);
            self.ctx.device.destroy_descriptor_pool(self.descriptor_pool, None);
            for pipeline in &self.pipelines {
                self.ctx.device.destroy_pipeline(*pipeline, None);
            }
            self.ctx.device.destroy_shader_module(self.shader_module, None);
            self.ctx.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.ctx.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
