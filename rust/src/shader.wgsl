struct Uniforms {
    no_components: u32,
    num_users: u32,
    num_items: u32,
    num_predictions: u32,
};

struct TrainParams {
    no_components: u32,
    num_users: u32,
    num_items: u32,
    num_samples: u32, // Number of samples per epoch per thread block? No, total samples.
    learning_rate: f32,
    item_alpha: f32,
    user_alpha: f32,
    seed: u32,
};

// Prediction Bindings (Group 0)
@group(0) @binding(0) var<storage, read> user_features_in: array<f32>;
@group(0) @binding(1) var<storage, read> item_features_in: array<f32>;
@group(0) @binding(2) var<storage, read> user_biases_in: array<f32>;
@group(0) @binding(3) var<storage, read> item_biases_in: array<f32>;
@group(0) @binding(4) var<storage, read> pairs: array<i32>; 
@group(0) @binding(5) var<storage, read_write> predictions: array<f32>;
@group(0) @binding(6) var<uniform> params: Uniforms;

// Training Bindings (Group 0)
// Using f32 directly (Unsafe Hogwild - Race conditions will occur)
@group(0) @binding(0) var<storage, read_write> user_features: array<f32>;
@group(0) @binding(1) var<storage, read_write> item_features: array<f32>;
@group(0) @binding(2) var<storage, read_write> user_biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> item_biases: array<f32>;
@group(0) @binding(4) var<storage, read> train_indptr: array<i32>;
@group(0) @binding(5) var<storage, read> train_indices: array<i32>;
@group(0) @binding(6) var<uniform> train_params: TrainParams;

@compute @workgroup_size(64)
fn predict_kernel(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_wgs: vec3<u32>
) {
    let width = num_wgs.x * 64u;
    let index = global_id.x + global_id.y * width;

    if (index >= params.num_predictions) {
        return;
    }

    let user_id = u32(pairs[index * 2]);
    let item_id = u32(pairs[index * 2 + 1]);

    let no_components = params.no_components;

    var dot_prod = 0.0;
    let user_offset = user_id * no_components;
    let item_offset = item_id * no_components;

    for (var i = 0u; i < no_components; i = i + 1u) {
        dot_prod = dot_prod + user_features_in[user_offset + i] * item_features_in[item_offset + i];
    }

    let result = dot_prod + user_biases_in[user_id] + item_biases_in[item_id];
    predictions[index] = result;
}

// --- Training Helpers ---

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_range(seed: ptr<function, u32>, range: u32) -> u32 {
    *seed = pcg_hash(*seed);
    return *seed % range;
}

@compute @workgroup_size(64)
fn fit_bpr_kernel(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_wgs: vec3<u32>
) {
    let width = num_wgs.x * 64u;
    let index = global_id.x + global_id.y * width;
    
    if (index >= train_params.num_samples) {
        return;
    }

    var seed = train_params.seed + index;
    
    // 1. Sample User
    let user_id = rand_range(&seed, train_params.num_users);
    
    let start_idx = train_indptr[user_id];
    let end_idx = train_indptr[user_id + 1];
    
    if (start_idx == end_idx) { return; } 
    
    // 2. Sample Positive Item
    let pos_idx = rand_range(&seed, u32(end_idx - start_idx)) + u32(start_idx);
    let pos_item_id = u32(train_indices[pos_idx]);
    
    // 3. Sample Negative Item
    let neg_item_id = rand_range(&seed, train_params.num_items);
    
    // 4. Compute Scores
    let nc = train_params.no_components;
    let u_off = user_id * nc;
    let p_off = pos_item_id * nc;
    let n_off = neg_item_id * nc;
    
    var pos_score = user_biases[user_id] + item_biases[pos_item_id];
    var neg_score = user_biases[user_id] + item_biases[neg_item_id];
    
    for (var i = 0u; i < nc; i = i + 1u) {
        let u = user_features[u_off + i];
        let p = item_features[p_off + i];
        let n = item_features[n_off + i];
        
        pos_score = pos_score + u * p;
        neg_score = neg_score + u * n;
    }
    
    // 5. Compute Gradient (BPR)
    let diff = pos_score - neg_score;
    let sigmoid = 1.0 / (1.0 + exp(-diff));
    let gradient = 1.0 - sigmoid; 
    
    let lr = train_params.learning_rate;
    let update = lr * gradient;
    
    // 6. Update Weights (Unsafe add)
    user_biases[user_id] += update;
    item_biases[pos_item_id] += update;
    item_biases[neg_item_id] -= update;
    
    // Features
    for (var i = 0u; i < nc; i = i + 1u) {
        let u = user_features[u_off + i];
        let p = item_features[p_off + i];
        let n = item_features[n_off + i];
        
        // Update User
        user_features[u_off + i] += update * (p - n);
        
        // Update Pos
        item_features[p_off + i] += update * u;
        
        // Update Neg
        item_features[n_off + i] -= update * u;
    }
}
