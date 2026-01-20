struct Uniforms {
    no_components: u32,
    num_users: u32,
    num_items: u32,
    num_predictions: u32,
};

@group(0) @binding(0) var<storage, read> user_features: array<f32>;
@group(0) @binding(1) var<storage, read> item_features: array<f32>;
@group(0) @binding(2) var<storage, read> user_biases: array<f32>;
@group(0) @binding(3) var<storage, read> item_biases: array<f32>;
@group(0) @binding(4) var<storage, read> pairs: array<i32>; // [user_id, item_id, user_id, item_id...]
@group(0) @binding(5) var<storage, read_write> predictions: array<f32>;
@group(0) @binding(6) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn predict_kernel(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_wgs: vec3<u32>
) {
    // 2D dispatch handling: index = x + y * width
    // global_id.x runs from 0 to num_wgs.x * 64
    // global_id.y runs from 0 to num_wgs.y
    let width = num_wgs.x * 64u;
    let index = global_id.x + global_id.y * width;

    if (index >= params.num_predictions) {
        return;
    }

    let user_id = u32(pairs[index * 2]);
    let item_id = u32(pairs[index * 2 + 1]);

    let no_components = params.no_components;

    // Dot product
    var dot_prod = 0.0;
    
    // We assume features are flattened: [u1_c1, u1_c2... u1_cn, u2_c1...]
    let user_offset = user_id * no_components;
    let item_offset = item_id * no_components;

    for (var i = 0u; i < no_components; i = i + 1u) {
        dot_prod = dot_prod + user_features[user_offset + i] * item_features[item_offset + i];
    }

    // Add biases
    let result = dot_prod + user_biases[user_id] + item_biases[item_id];

    predictions[index] = result;
}
