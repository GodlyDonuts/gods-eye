struct Params {
    dims: vec4<u32>,
    depth_size: vec4<u32>,
    origin: vec4<f32>,
    intrinsics: vec4<f32>,
    scalars: vec4<f32>,
    world_to_cam: mat4x4<f32>,
};

@group(0) @binding(0) var<storage, read_write> tsdf: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read> depth_m: array<f32>;
@group(0) @binding(3) var<storage, read> confidence: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear = gid.x;
    if (linear >= params.dims.w) {
        return;
    }

    let dim_x = params.dims.x;
    let dim_y = params.dims.y;
    let xy = dim_x * dim_y;
    let z = linear / xy;
    let rem = linear - z * xy;
    let y = rem / dim_x;
    let x = rem - y * dim_x;

    let voxel = params.scalars.x;
    let trunc = params.scalars.y;
    let half = 0.5 * voxel;
    let world = vec3<f32>(
        params.origin.x + f32(x) * voxel + half,
        params.origin.y + f32(y) * voxel + half,
        params.origin.z + f32(z) * voxel + half,
    );
    let cam = params.world_to_cam * vec4<f32>(world, 1.0);
    if (cam.z <= 0.0001) {
        return;
    }

    let u_f = params.intrinsics.x * cam.x / cam.z + params.intrinsics.z;
    let v_f = params.intrinsics.y * cam.y / cam.z + params.intrinsics.w;
    if (u_f < 0.0 || v_f < 0.0 || u_f >= f32(params.depth_size.x) || v_f >= f32(params.depth_size.y)) {
        return;
    }

    let u = u32(u_f);
    let v = u32(v_f);
    let depth_idx = v * params.depth_size.x + u;
    let d = depth_m[depth_idx];
    if (d <= 0.0 || d != d) {
        return;
    }

    var obs_weight = 1.0;
    if (params.depth_size.z != 0u) {
        obs_weight = clamp(confidence[depth_idx], 0.0, 1.0);
    }
    if (obs_weight <= 0.0) {
        return;
    }

    let sdf = d - cam.z;
    if (sdf < -trunc) {
        return;
    }

    let val = clamp(sdf / trunc, -1.0, 1.0);
    let old_weight = weight[linear];
    let new_weight = old_weight + obs_weight;
    tsdf[linear] = (tsdf[linear] * old_weight + val * obs_weight) / new_weight;
    weight[linear] = new_weight;
}
