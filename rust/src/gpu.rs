use std::borrow::Cow;
use wgpu::util::DeviceExt;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    no_components: u32,
    num_users: u32,
    num_items: u32,
    num_predictions: u32,
}

#[pyclass]
pub struct LightFMwgpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[pymethods]
impl LightFMwgpu {
    #[new]
    pub fn new() -> Self {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .expect("Failed to find an appropriate adapter");

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits {
                            max_storage_buffers_per_shader_stage: 8,
                            ..wgpu::Limits::downlevel_defaults()
                        },
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .await
                .expect("Failed to create device");

            // Define the shader
            let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("LightFM Shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
            });

            // Define bind group layout
            // 0: User Features (ReadOnly)
            // 1: Item Features (ReadOnly)
            // 2: User Biases (ReadOnly)
            // 3: Item Biases (ReadOnly)
            // 4: Interactions (ReadOnly - [user_id, item_id] pairs)
            // 5: Output/Predictions (WriteOnly)
            // 6: Uniforms
            let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("LightFM Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("LightFM Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LightFM Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &cs_module,
                entry_point: "predict_kernel",
                compilation_options: Default::default(),
                cache: None,
            });

            Self {
                device,
                queue,
                pipeline,
                bind_group_layout,
            }
        })
    }

    #[pyo3(signature = (user_features, item_features, user_biases, item_biases, pairs, no_components))]
    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        user_features: PyReadonlyArray2<f32>,
        item_features: PyReadonlyArray2<f32>,
        user_biases: PyReadonlyArray1<f32>,
        item_biases: PyReadonlyArray1<f32>,
        pairs: PyReadonlyArray1<i32>,
        no_components: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let user_features_slice = user_features.as_slice()?;
        let item_features_slice = item_features.as_slice()?;
        let user_biases_slice = user_biases.as_slice()?;
        let item_biases_slice = item_biases.as_slice()?;
        let pairs_slice = pairs.as_slice()?;
        
        let num_users = user_biases_slice.len() as u32;
        let num_items = item_biases_slice.len() as u32;
        let num_predictions = (pairs_slice.len() / 2) as u32;

        let predictions = pollster::block_on(async {
            let device = &self.device;
            let queue = &self.queue;

            // Create buffers
            let user_features_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("User Features Buffer"),
                contents: bytemuck::cast_slice(user_features_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let item_features_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Item Features Buffer"),
                contents: bytemuck::cast_slice(item_features_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let user_biases_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("User Biases Buffer"),
                contents: bytemuck::cast_slice(user_biases_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let item_biases_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Item Biases Buffer"),
                contents: bytemuck::cast_slice(item_biases_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let pairs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Pairs Buffer"),
                contents: bytemuck::cast_slice(pairs_slice),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let predictions_size = (num_predictions as usize * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
            let predictions_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Predictions Buffer"),
                size: predictions_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let stage_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size: predictions_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Uniforms
            let uniforms = Uniforms {
                no_components,
                num_users,
                num_items,
                num_predictions,
            };
            let uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniforms Buffer"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            // Bind Group
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("LightFM Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: user_features_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: item_features_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: user_biases_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: item_biases_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: pairs_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: predictions_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: uniforms_buf.as_entire_binding(),
                    },
                ],
            });

            // Encoder
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("LightFM Encoder"),
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("LightFM Compute Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                
                let workgroup_size = 64;
                let total_workgroups = (num_predictions + workgroup_size - 1) / workgroup_size;
                
                let max_dim = 65535;
                let (x_groups, y_groups) = if total_workgroups > max_dim {
                    (max_dim, (total_workgroups + max_dim - 1) / max_dim)
                } else {
                    (total_workgroups, 1)
                };
                
                cpass.dispatch_workgroups(x_groups, y_groups, 1);
            }

            // Copy result
            encoder.copy_buffer_to_buffer(&predictions_buf, 0, &stage_buf, 0, predictions_size);

            queue.submit(Some(encoder.finish()));

            // Read back
            let slice = stage_buf.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            device.poll(wgpu::Maintain::Wait);
            receiver.receive().await.unwrap().unwrap();

            let data = slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            stage_buf.unmap();

            result
        });

        Ok(predictions.to_pyarray_bound(py).into_any())
    }
}
