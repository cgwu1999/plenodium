#include "cuda_runtime.h"
#include "forward.cuh"
#include <cstdio>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <tuple>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)
#define DEVICE_GUARD(_ten) \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

std::tuple<
    torch::Tensor, // output conics
    torch::Tensor> // output radii
compute_cov2d_bounds_tensor(const int num_pts, torch::Tensor &A);

torch::Tensor compute_sh_forward_tensor(
    unsigned num_points,
    unsigned degree,
    unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &coeffs
);

torch::Tensor compute_sh_backward_tensor(
    unsigned num_points,
    unsigned degree,
    unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &v_colors
);

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
);

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
);


std::tuple<torch::Tensor, torch::Tensor> map_gaussian_to_intersects_tensor(
    const int num_points,
    const int num_intersects,
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &radii,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> tile_bounds,
    const unsigned block_width
);

torch::Tensor get_tile_bin_edges_tensor(
    int num_intersects,
    const torch::Tensor &isect_ids_sorted,
    const std::tuple<int, int, int> tile_bounds
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> rasterize_forward_tensor(
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
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> nd_rasterize_forward_tensor(
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
);


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
        const torch::Tensor &v_output_alpha
    );

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
        const torch::Tensor &v_out_medium,
        const torch::Tensor &v_output_alpha
    );


torch::Tensor sparse_compute_sh_forward_tensor(
    unsigned num_points,
    unsigned degree,
    unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &coeffs
);

torch::Tensor sparse_compute_sh_backward_tensor(
    unsigned num_points,
    unsigned degree,
    unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &v_colors
);

std::tuple<torch::Tensor,
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
                                  torch::Tensor& kernel);

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
                                   torch::Tensor& kernel);

std::tuple<torch::Tensor,  // dL_dxy
           torch::Tensor,  // dL_ddepth
           torch::Tensor,  // dL_dconic
           torch::Tensor,  // dL_dcolors
           torch::Tensor,  // dL_dopacity
           torch::Tensor,  // dL_dmedium_rgb
           torch::Tensor,  // dL_dmedium_bs
           torch::Tensor   // dL_dmedium_attn
           >
rasterize_backward_tensor2(
    const unsigned img_height,
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
    const torch::Tensor& v_output,        // dL_dout_color
    const torch::Tensor& v_out_clr,       // dL_dout_clr
    const torch::Tensor& v_out_medium,    // dL_dout_med
    const torch::Tensor& v_output_alpha,  // dL_dout_alpha
    const torch::Tensor& v_depth_im       // dL_ddepth_im
);