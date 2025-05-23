#include "backward.cuh"
#include "bindings.h"
#include "forward.cuh"
#include "helpers.cuh"
#include "sh.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <tuple>

namespace cg = cooperative_groups;

__global__ void compute_cov2d_bounds_kernel(
    const unsigned num_pts, const float* __restrict__ covs2d, float* __restrict__ conics, float* __restrict__ radii
) {
    unsigned row = cg::this_grid().thread_rank();
    if (row >= num_pts) {
        return;
    }
    int index = row * 3;
    float3 conic;
    float radius;
    float3 cov2d{
        (float)covs2d[index], (float)covs2d[index + 1], (float)covs2d[index + 2]
    };
    compute_cov2d_bounds(cov2d, conic, radius);
    conics[index] = conic.x;
    conics[index + 1] = conic.y;
    conics[index + 2] = conic.z;
    radii[row] = radius;
}

std::tuple<
    torch::Tensor, // output conics
    torch::Tensor> // output radii
compute_cov2d_bounds_tensor(const int num_pts, torch::Tensor &covs2d) {
    DEVICE_GUARD(covs2d);
    CHECK_INPUT(covs2d);
    torch::Tensor conics = torch::zeros(
        {num_pts, covs2d.size(1)}, covs2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor radii =
        torch::zeros({num_pts, 1}, covs2d.options().dtype(torch::kFloat32));

    int blocks = (num_pts + N_THREADS - 1) / N_THREADS;

    compute_cov2d_bounds_kernel<<<blocks, N_THREADS>>>(
        num_pts,
        covs2d.contiguous().data_ptr<float>(),
        conics.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<float>()
    );
    return std::make_tuple(conics, radii);
}

torch::Tensor compute_sh_forward_tensor(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &coeffs
) {
    DEVICE_GUARD(viewdirs);
    unsigned num_bases = num_sh_bases(degree);
    if (coeffs.ndimension() != 3 || coeffs.size(0) != num_points ||
        coeffs.size(1) != num_bases || coeffs.size(2) != 3) {
        AT_ERROR("coeffs must have dimensions (N, D, 3)");
    }
    torch::Tensor colors = torch::empty({num_points, 3}, coeffs.options());
    compute_sh_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        degree,
        degrees_to_use,
        (float3 *)viewdirs.contiguous().data_ptr<float>(),
        coeffs.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>()
    );
    return colors;
}

torch::Tensor compute_sh_backward_tensor(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &v_colors
) {
    DEVICE_GUARD(viewdirs);
    if (viewdirs.ndimension() != 2 || viewdirs.size(0) != num_points ||
        viewdirs.size(1) != 3) {
        AT_ERROR("viewdirs must have dimensions (N, 3)");
    }
    if (v_colors.ndimension() != 2 || v_colors.size(0) != num_points ||
        v_colors.size(1) != 3) {
        AT_ERROR("v_colors must have dimensions (N, 3)");
    }
    unsigned num_bases = num_sh_bases(degree);
    torch::Tensor v_coeffs =
        torch::zeros({num_points, num_bases, 3}, v_colors.options());
    compute_sh_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        degree,
        degrees_to_use,
        (float3 *)viewdirs.contiguous().data_ptr<float>(),
        v_colors.contiguous().data_ptr<float>(),
        v_coeffs.contiguous().data_ptr<float>()
    );
    return v_coeffs;
}


std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_tensor(
    const int num_points,
    torch::Tensor &means3d,
    torch::Tensor &scales,
    const float glob_scale,
    torch::Tensor &quats,
    torch::Tensor &viewmat,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const unsigned img_height,
    const unsigned img_width,
    const unsigned block_width,
    const float clip_thresh
) {
    DEVICE_GUARD(means3d);
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = int((img_width + block_width - 1) / block_width);
    tile_bounds_dim3.y = int((img_height + block_width - 1) / block_width);
    tile_bounds_dim3.z = 1;

    float4 intrins = {fx, fy, cx, cy};

    // Triangular covariance.
    torch::Tensor cov3d_d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor xys_d =
        torch::zeros({num_points, 2}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor depths_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor radii_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));
    torch::Tensor conics_d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor compensation_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));

    project_gaussians_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float3 *)means3d.contiguous().data_ptr<float>(),
        (float3 *)scales.contiguous().data_ptr<float>(),
        glob_scale,
        (float4 *)quats.contiguous().data_ptr<float>(),
        viewmat.contiguous().data_ptr<float>(),
        intrins,
        img_size_dim3,
        tile_bounds_dim3,
        block_width,
        clip_thresh,
        // Outputs.
        cov3d_d.contiguous().data_ptr<float>(),
        (float2 *)xys_d.contiguous().data_ptr<float>(),
        depths_d.contiguous().data_ptr<float>(),
        radii_d.contiguous().data_ptr<int>(),
        (float3 *)conics_d.contiguous().data_ptr<float>(),
        compensation_d.contiguous().data_ptr<float>(),
        num_tiles_hit_d.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(
        cov3d_d, xys_d, depths_d, radii_d, conics_d, compensation_d, num_tiles_hit_d
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_backward_tensor(
    const int num_points,
    torch::Tensor &means3d,
    torch::Tensor &scales,
    const float glob_scale,
    torch::Tensor &quats,
    torch::Tensor &viewmat,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const unsigned img_height,
    const unsigned img_width,
    torch::Tensor &cov3d,
    torch::Tensor &radii,
    torch::Tensor &conics,
    torch::Tensor &compensation,
    torch::Tensor &v_xy,
    torch::Tensor &v_depth,
    torch::Tensor &v_conic,
    torch::Tensor &v_compensation
){
    DEVICE_GUARD(means3d);
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    float4 intrins = {fx, fy, cx, cy};

    const auto num_cov3d = num_points * 6;

    // Triangular covariance.
    torch::Tensor v_cov2d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_cov3d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_mean3d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_scale =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_quat =
        torch::zeros({num_points, 4}, means3d.options().dtype(torch::kFloat32));

    project_gaussians_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float3 *)means3d.contiguous().data_ptr<float>(),
        (float3 *)scales.contiguous().data_ptr<float>(),
        glob_scale,
        (float4 *)quats.contiguous().data_ptr<float>(),
        viewmat.contiguous().data_ptr<float>(),
        intrins,
        img_size_dim3,
        cov3d.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int32_t>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float *)compensation.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        v_depth.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        (float *)v_compensation.contiguous().data_ptr<float>(),
        // Outputs.
        (float3 *)v_cov2d.contiguous().data_ptr<float>(),
        v_cov3d.contiguous().data_ptr<float>(),
        (float3 *)v_mean3d.contiguous().data_ptr<float>(),
        (float3 *)v_scale.contiguous().data_ptr<float>(),
        (float4 *)v_quat.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_cov2d, v_cov3d, v_mean3d, v_scale, v_quat);
}

std::tuple<torch::Tensor, torch::Tensor> map_gaussian_to_intersects_tensor(
    const int num_points,
    const int num_intersects,
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &radii,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> tile_bounds,
    const unsigned block_width
) {
    DEVICE_GUARD(xys);
    CHECK_INPUT(xys);
    CHECK_INPUT(depths);
    CHECK_INPUT(radii);
    CHECK_INPUT(cum_tiles_hit);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    torch::Tensor gaussian_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor isect_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));

    map_gaussian_to_intersects<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float2 *)xys.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int32_t>(),
        cum_tiles_hit.contiguous().data_ptr<int32_t>(),
        tile_bounds_dim3,
        block_width,
        // Outputs.
        isect_ids_unsorted.contiguous().data_ptr<int64_t>(),
        gaussian_ids_unsorted.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(isect_ids_unsorted, gaussian_ids_unsorted);
}

torch::Tensor get_tile_bin_edges_tensor(
    int num_intersects, const torch::Tensor &isect_ids_sorted, 
    const std::tuple<int, int, int> tile_bounds
) {
    DEVICE_GUARD(isect_ids_sorted);
    CHECK_INPUT(isect_ids_sorted);
    int num_tiles = std::get<0>(tile_bounds) * std::get<1>(tile_bounds);
    torch::Tensor tile_bins = torch::zeros(
        {num_tiles, 2}, isect_ids_sorted.options().dtype(torch::kInt32)
    );
    get_tile_bin_edges<<<
        (num_intersects + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_intersects,
        isect_ids_sorted.contiguous().data_ptr<int64_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>()
    );
    return tile_bins;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &medium_rgb,
    const torch::Tensor &medium_bs,
    const torch::Tensor &medium_attn,
    const torch::Tensor &depths,
    const torch::Tensor &background
) {
    DEVICE_GUARD(xys);
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(medium_rgb);
    CHECK_INPUT(medium_bs);
    CHECK_INPUT(medium_attn);
    CHECK_INPUT(depths);
    CHECK_INPUT(background);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    dim3 block_dim3;
    block_dim3.x = std::get<0>(block);
    block_dim3.y = std::get<1>(block);
    block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    const int channels = colors.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor out_clr = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor out_med = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor depth_im = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );
    torch::Tensor first_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );

    rasterize_forward<<<tile_bounds_dim3, block_dim3>>>(
        tile_bounds_dim3,
        img_size_dim3,
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        (float3 *)medium_rgb.contiguous().data_ptr<float>(),
        (float3 *)medium_bs.contiguous().data_ptr<float>(),
        (float3 *)medium_attn.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        first_idx.contiguous().data_ptr<int>(),
        (float3 *)out_img.contiguous().data_ptr<float>(),
        (float3 *)out_clr.contiguous().data_ptr<float>(),
        (float3 *)out_med.contiguous().data_ptr<float>(),
        depth_im.contiguous().data_ptr<float>(),
        *(float3 *)background.contiguous().data_ptr<float>()
    );

    return std::make_tuple(out_img, out_clr, out_med, depth_im, final_Ts, final_idx, first_idx);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
nd_rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &medium_rgb,
    const torch::Tensor &medium_bs,
    const torch::Tensor &medium_attn,
    const torch::Tensor &depths,
    const torch::Tensor &background
) {
    DEVICE_GUARD(xys);
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(medium_rgb);
    CHECK_INPUT(medium_bs);
    CHECK_INPUT(medium_attn);
    CHECK_INPUT(depths);
    CHECK_INPUT(background);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    dim3 block_dim3;
    block_dim3.x = std::get<0>(block);
    block_dim3.y = std::get<1>(block);
    block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    const int channels = colors.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );
    torch::Tensor first_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );
    const int B = block_dim3.x * block_dim3.y;
    const uint32_t shared_mem = B*sizeof(int) + B*sizeof(float3) + B*sizeof(float3) + B*channels*sizeof(half);
    if(cudaFuncSetAttribute(nd_rasterize_forward, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem) != cudaSuccess){
        AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem, " bytes), try lowering block_size");
    }

    nd_rasterize_forward<<<tile_bounds_dim3, block_dim3, shared_mem>>>(
        tile_bounds_dim3,
        img_size_dim3,
        channels,
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        medium_rgb.contiguous().data_ptr<float>(),
        medium_bs.contiguous().data_ptr<float>(),
        medium_attn.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        first_idx.contiguous().data_ptr<int>(),
        out_img.contiguous().data_ptr<float>(),
        background.contiguous().data_ptr<float>()
    );

    return std::make_tuple(out_img, final_Ts, final_idx, first_idx);
}



std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor, // dL_dopacity
        torch::Tensor, // dL_dmedium_rgb
        torch::Tensor, // dL_dmedium_bs
        torch::Tensor  // dL_dmedium_attn
        >
    nd_rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const unsigned block_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &medium_rgb,
        const torch::Tensor &medium_bs,
        const torch::Tensor &medium_attn,
        const torch::Tensor &depths,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &first_idx,
        const torch::Tensor &v_output, // dL_dout_color
        const torch::Tensor &v_output_alpha // dL_dout_alpha
    ) {
    DEVICE_GUARD(xys);
    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }

    if (colors.ndimension() != 2) {
        AT_ERROR("colors must have 2 dimensions");
    }

    const int num_points = xys.size(0);
    const dim3 tile_bounds = {
        (img_width + block_width - 1) / block_width,
        (img_height + block_width - 1) / block_width,
        1
    };
    const dim3 block(block_width, block_width, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());
    torch::Tensor v_medium_rgb = torch::zeros({img_height, img_width, channels}, xys.options());
    torch::Tensor v_medium_bs = torch::zeros({img_height, img_width, channels}, xys.options());
    torch::Tensor v_medium_attn = torch::zeros({img_height, img_width, channels}, xys.options());

    const int B = block.x * block.y;
    //shared mem accounts for each thread having a local shared memory workspace for running sum
    const uint32_t shared_mem = B*channels*sizeof(half);
    if(cudaFuncSetAttribute(nd_rasterize_backward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem) != cudaSuccess){
        AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem, " bytes), try lowering block_size");
    }
    nd_rasterize_backward_kernel<<<tile_bounds, block, shared_mem>>>(
        tile_bounds,
        img_size,
        channels,
        gaussians_ids_sorted.contiguous().data_ptr<int>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        (float *)medium_rgb.contiguous().data_ptr<float>(),
        (float *)medium_bs.contiguous().data_ptr<float>(),
        (float *)medium_attn.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        background.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        first_idx.contiguous().data_ptr<int>(),
        v_output.contiguous().data_ptr<float>(),
        v_output_alpha.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        v_colors.contiguous().data_ptr<float>(),
        v_opacity.contiguous().data_ptr<float>(),
        (float *)v_medium_rgb.contiguous().data_ptr<float>(),
        (float *)v_medium_bs.contiguous().data_ptr<float>(),
        (float *)v_medium_attn.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_xy, v_conic, v_colors, v_opacity, v_medium_rgb, v_medium_bs, v_medium_attn);
}

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor, // dL_dopacity
        torch::Tensor, // dL_dmedium_rgb
        torch::Tensor, // dL_dmedium_bs
        torch::Tensor  // dL_dmedium_attn
        >
    rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const unsigned block_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        torch::Tensor &xys_grad_abs,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &medium_rgb,
        const torch::Tensor &medium_bs,
        const torch::Tensor &medium_attn,
        const torch::Tensor &depths,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &first_idx,
        const torch::Tensor &v_output, // dL_dout_color
        const torch::Tensor &v_out_medium, // dL_dout_med
        const torch::Tensor &v_output_alpha // dL_dout_alpha
    ) {
    DEVICE_GUARD(xys);
    CHECK_INPUT(xys_grad_abs);
    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }

    if (colors.ndimension() != 2 || colors.size(1) != 3) {
        AT_ERROR("colors must have 2 dimensions");
    }

    const int num_points = xys.size(0);
    const dim3 tile_bounds = {
        (img_width + block_width - 1) / block_width,
        (img_height + block_width - 1) / block_width,
        1
    };
    const dim3 block(block_width, block_width, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());
    torch::Tensor v_medium_rgb = torch::zeros({img_height, img_width, 3}, xys.options());
    torch::Tensor v_medium_bs = torch::zeros({img_height, img_width, 3}, xys.options());
    torch::Tensor v_medium_attn = torch::zeros({img_height, img_width, 3}, xys.options());

    rasterize_backward_kernel<<<tile_bounds, block>>>(
        tile_bounds,
        img_size,
        gaussians_ids_sorted.contiguous().data_ptr<int>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float2 *)xys_grad_abs.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        (float3 *)medium_rgb.contiguous().data_ptr<float>(),
        (float3 *)medium_bs.contiguous().data_ptr<float>(),
        (float3 *)medium_attn.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        *(float3 *)background.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        first_idx.contiguous().data_ptr<int>(),
        (float3 *)v_output.contiguous().data_ptr<float>(),
        (float3 *)v_out_medium.contiguous().data_ptr<float>(),
        v_output_alpha.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        (float3 *)v_colors.contiguous().data_ptr<float>(),
        v_opacity.contiguous().data_ptr<float>(),
        (float3 *)v_medium_rgb.contiguous().data_ptr<float>(),
        (float3 *)v_medium_bs.contiguous().data_ptr<float>(),
        (float3 *)v_medium_attn.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_xy, v_conic, v_colors, v_opacity, v_medium_rgb, v_medium_bs, v_medium_attn);
}




torch::Tensor sparse_compute_sh_forward_tensor(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &coeffs
) {
    DEVICE_GUARD(viewdirs);
    unsigned num_bases = num_sh_bases(degree);
    if (coeffs.ndimension() != 3 || coeffs.size(1) != num_bases || coeffs.size(2) != 3) {
        AT_ERROR("coeffs must have dimensions (3, N, 3)");
    }
    torch::Tensor colors = torch::empty({num_points * 3, 3}, coeffs.options());
    sparse_compute_sh_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        degree,
        degrees_to_use,
        (float3 *)viewdirs.contiguous().data_ptr<float>(),
        coeffs.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>()
    );
    return colors;
}

torch::Tensor sparse_compute_sh_backward_tensor(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &v_colors
) {
    // printf("sparse_compute_sh_backward_tensor\n");
    DEVICE_GUARD(viewdirs);
    if (viewdirs.ndimension() != 2 || viewdirs.size(0) != num_points ||
        viewdirs.size(1) != 3) {
        AT_ERROR("viewdirs must have dimensions (N, 3)");
    }
    if (v_colors.ndimension() != 2 || v_colors.size(0) != 3 * num_points ||
        v_colors.size(1) != 3) {
        AT_ERROR("v_colors must have dimensions (3N, 3)");
    }
    unsigned num_bases = num_sh_bases(degree);
    torch::Tensor v_coeffs =
        torch::zeros({3, num_bases, 3}, v_colors.options());
    sparse_compute_sh_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        degree,
        degrees_to_use,
        (float3 *)viewdirs.contiguous().data_ptr<float>(),
        v_colors.contiguous().data_ptr<float>(),
        v_coeffs.contiguous().data_ptr<float>()
    );
    return v_coeffs;
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_tensor2(const int num_points,
                                 torch::Tensor& means3d,
                                 torch::Tensor& scales,
                                 const float glob_scale,
                                 torch::Tensor& quats,
                                 torch::Tensor& viewmat,
                                 const float fx,
                                 const float fy,
                                 const float cx,
                                 const float cy,
                                 const unsigned img_height,
                                 const unsigned img_width,
                                 const unsigned block_width,
                                 const float clip_thresh,
                                 torch::Tensor& kernel) {
    DEVICE_GUARD(means3d);
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = int((img_width + block_width - 1) / block_width);
    tile_bounds_dim3.y = int((img_height + block_width - 1) / block_width);
    tile_bounds_dim3.z = 1;

    float4 intrins = {fx, fy, cx, cy};

    // Triangular covariance.
    torch::Tensor cov3d_d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor xys_d =
        torch::zeros({num_points, 2}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor depths_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor radii_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));
    torch::Tensor conics_d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor compensation_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));

    project_gaussians_forward_kernel2<<<
        (num_points + N_THREADS - 1) / N_THREADS, N_THREADS>>>(
        num_points, (float3*)means3d.contiguous().data_ptr<float>(),
        (float3*)scales.contiguous().data_ptr<float>(), glob_scale,
        (float4*)quats.contiguous().data_ptr<float>(),
        viewmat.contiguous().data_ptr<float>(), intrins, img_size_dim3,
        tile_bounds_dim3, block_width, clip_thresh,
        // kernel_size.contiguous().data_ptr<float>(),
        kernel.contiguous().data_ptr<float>(),
        // Outputs.
        cov3d_d.contiguous().data_ptr<float>(),
        (float2*)xys_d.contiguous().data_ptr<float>(),
        depths_d.contiguous().data_ptr<float>(),
        radii_d.contiguous().data_ptr<int>(),
        (float3*)conics_d.contiguous().data_ptr<float>(),
        compensation_d.contiguous().data_ptr<float>(),
        num_tiles_hit_d.contiguous().data_ptr<int32_t>());

    return std::make_tuple(cov3d_d, xys_d, depths_d, radii_d, conics_d,
                           compensation_d, num_tiles_hit_d);
}

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
project_gaussians_backward_tensor2(const int num_points,
                                  torch::Tensor& means3d,
                                  torch::Tensor& scales,
                                  const float glob_scale,
                                  torch::Tensor& quats,
                                  torch::Tensor& viewmat,
                                  const float fx,
                                  const float fy,
                                  const float cx,
                                  const float cy,
                                  const unsigned img_height,
                                  const unsigned img_width,
                                  torch::Tensor& cov3d,
                                  torch::Tensor& radii,
                                  torch::Tensor& conics,
                                  torch::Tensor& compensation,
                                  torch::Tensor& v_xy,
                                  torch::Tensor& v_depth,
                                  torch::Tensor& v_conic,
                                  torch::Tensor& v_compensation,
                                  torch::Tensor& kernel) {
    DEVICE_GUARD(means3d);
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    float4 intrins = {fx, fy, cx, cy};

    const auto num_cov3d = num_points * 6;
    // Triangular covariance.
    torch::Tensor v_cov2d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_cov3d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_mean3d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_scale =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_quat =
        torch::zeros({num_points, 4}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_kernel =
        torch::zeros({1}, means3d.options().dtype(torch::kFloat32));
    project_gaussians_backward_kernel2<<<
        (num_points + N_THREADS - 1) / N_THREADS, N_THREADS>>>(
        num_points, (float3*)means3d.contiguous().data_ptr<float>(),
        (float3*)scales.contiguous().data_ptr<float>(), glob_scale,
        (float4*)quats.contiguous().data_ptr<float>(),
        viewmat.contiguous().data_ptr<float>(), intrins, img_size_dim3,
        cov3d.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int32_t>(),
        (float3*)conics.contiguous().data_ptr<float>(),
        (float*)compensation.contiguous().data_ptr<float>(),
        (float2*)v_xy.contiguous().data_ptr<float>(),
        v_depth.contiguous().data_ptr<float>(),
        (float3*)v_conic.contiguous().data_ptr<float>(),
        (float*)v_compensation.contiguous().data_ptr<float>(),
        kernel.contiguous().data_ptr<float>(),
        // Outputs.
        v_kernel.contiguous().data_ptr<float>(),
        (float3*)v_cov2d.contiguous().data_ptr<float>(),
        v_cov3d.contiguous().data_ptr<float>(),
        (float3*)v_mean3d.contiguous().data_ptr<float>(),
        (float3*)v_scale.contiguous().data_ptr<float>(),
        (float4*)v_quat.contiguous().data_ptr<float>());

    return std::make_tuple(v_cov2d, v_cov3d, v_mean3d, v_scale, v_quat,
                           v_kernel);
}

std::tuple<torch::Tensor,  // dL_dxy
           torch::Tensor,   //dL_ddepth
           torch::Tensor,  // dL_dconic
           torch::Tensor,  // dL_dcolors
           torch::Tensor,  // dL_dopacity
           torch::Tensor,  // dL_dmedium_rgb
           torch::Tensor,  // dL_dmedium_bs
           torch::Tensor   // dL_dmedium_attn
           >
rasterize_backward_tensor2(const unsigned img_height,
                          const unsigned img_width,
                          const unsigned block_width,
                          const torch::Tensor& gaussians_ids_sorted,
                          const torch::Tensor& tile_bins,
                          const torch::Tensor& xys,
                          torch::Tensor& xys_grad_abs,
                          const torch::Tensor& conics,
                          const torch::Tensor& colors,
                          const torch::Tensor& opacities,
                          const torch::Tensor& medium_rgb,
                          const torch::Tensor& medium_bs,
                          const torch::Tensor& medium_attn,
                          const torch::Tensor& depths,
                          const torch::Tensor& background,
                          const torch::Tensor& final_Ts,
                          const torch::Tensor& final_idx,
                          const torch::Tensor& first_idx,
                          const torch::Tensor& v_output,       // dL_dout_color
                          const torch::Tensor& v_out_clr,       // dL_dout_clr
                          const torch::Tensor& v_out_medium,   // dL_dout_med
                          const torch::Tensor& v_output_alpha,  // dL_dout_alpha
                          const torch::Tensor& v_depth_im    // dL_ddepth_im
) {
    DEVICE_GUARD(xys);
    CHECK_INPUT(xys_grad_abs);
    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }

    if (colors.ndimension() != 2 || colors.size(1) != 3) {
        AT_ERROR("colors must have 2 dimensions");
    }

    const int num_points = xys.size(0);
    const dim3 tile_bounds = {(img_width + block_width - 1) / block_width,
                              (img_height + block_width - 1) / block_width, 1};
    const dim3 block(block_width, block_width, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_depths = torch::zeros({num_points}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());
    torch::Tensor v_medium_rgb =
        torch::zeros({img_height, img_width, 3}, xys.options());
    torch::Tensor v_medium_bs =
        torch::zeros({img_height, img_width, 3}, xys.options());
    torch::Tensor v_medium_attn =
        torch::zeros({img_height, img_width, 3}, xys.options());

    rasterize_backward_kernel2<<<tile_bounds, block>>>(
        tile_bounds, img_size,
        gaussians_ids_sorted.contiguous().data_ptr<int>(),
        (int2*)tile_bins.contiguous().data_ptr<int>(),
        (float2*)xys.contiguous().data_ptr<float>(),
        (float2*)xys_grad_abs.contiguous().data_ptr<float>(),
        (float3*)conics.contiguous().data_ptr<float>(),
        (float3*)colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        (float3*)medium_rgb.contiguous().data_ptr<float>(),
        (float3*)medium_bs.contiguous().data_ptr<float>(),
        (float3*)medium_attn.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        *(float3*)background.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        first_idx.contiguous().data_ptr<int>(),
        (float3*)v_output.contiguous().data_ptr<float>(),
        (float3*)v_out_clr.contiguous().data_ptr<float>(),
        (float3*)v_out_medium.contiguous().data_ptr<float>(),
        v_output_alpha.contiguous().data_ptr<float>(),
        v_depth_im.contiguous().data_ptr<float>(),
        (float2*)v_xy.contiguous().data_ptr<float>(),
        v_depths.contiguous().data_ptr<float>(),
        (float3*)v_conic.contiguous().data_ptr<float>(),
        (float3*)v_colors.contiguous().data_ptr<float>(),
        v_opacity.contiguous().data_ptr<float>(),
        (float3*)v_medium_rgb.contiguous().data_ptr<float>(),
        (float3*)v_medium_bs.contiguous().data_ptr<float>(),
        (float3*)v_medium_attn.contiguous().data_ptr<float>());

    return std::make_tuple(v_xy, v_depths, v_conic, v_colors, v_opacity, v_medium_rgb,
                           v_medium_bs, v_medium_attn);
}
