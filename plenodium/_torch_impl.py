"""Pure PyTorch implementations of various functions"""

import struct

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from typing import Tuple, Optional
from jaxtyping import Float, Int
from .utils import bin_and_sort_gaussians, compute_cumulative_intersects


def compute_sh_color(
    viewdirs: Float[Tensor, "*batch 3"], sh_coeffs: Float[Tensor, "*batch D C"]
):
    """
    :param viewdirs (*, C)
    :param sh_coeffs (*, D, C) sh coefficients for each color channel
    return colors (*, C)
    """
    *dims, dim_sh, C = sh_coeffs.shape
    bases = eval_sh_bases(dim_sh, viewdirs)  # (*, dim_sh)
    return (bases[..., None] * sh_coeffs).sum(dim=-2)


"""
Taken from https://github.com/sxyu/svox2
"""

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

MAX_SH_BASIS = 10


def eval_sh_bases(basis_dim: int, dirs: torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.

    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device
    )
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y
        result[..., 2] = SH_C1 * z
        result[..., 3] = -SH_C1 * x
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy
            result[..., 5] = SH_C2[1] * yz
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
            result[..., 7] = SH_C2[3] * xz
            result[..., 8] = SH_C2[4] * (xx - yy)

        if basis_dim > 9:
            result[..., 9] = SH_C3[0] * y * (3 * xx - yy)
            result[..., 10] = SH_C3[1] * xy * z
            result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy)
            result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
            result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy)
            result[..., 14] = SH_C3[5] * z * (xx - yy)
            result[..., 15] = SH_C3[6] * x * (xx - 3 * yy)

        if basis_dim > 16:
            result[..., 16] = SH_C4[0] * xy * (xx - yy)
            result[..., 17] = SH_C4[1] * yz * (3 * xx - yy)
            result[..., 18] = SH_C4[2] * xy * (7 * zz - 1)
            result[..., 19] = SH_C4[3] * yz * (7 * zz - 3)
            result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
            result[..., 21] = SH_C4[5] * xz * (7 * zz - 3)
            result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
            result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy)
            result[..., 24] = SH_C4[8] * (
                xx * (xx - 3 * yy) - yy * (3 * xx - yy)
                    )
    return result


def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y ** 2 + z ** 2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x ** 2 + z ** 2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x ** 2 + y ** 2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def quat_to_rotmat(quat: Tensor) -> Tensor:
    assert quat.shape[-1] == 4, quat.shape
    return normalized_quat_to_rotmat(F.normalize(quat, dim=-1))


def scale_rot_to_cov3d(scale: Tensor, glob_scale: float, quat: Tensor) -> Tensor:
    assert scale.shape[-1] == 3, scale.shape
    assert quat.shape[-1] == 4, quat.shape
    assert scale.shape[:-1] == quat.shape[:-1], (scale.shape, quat.shape)
    R = normalized_quat_to_rotmat(quat)  # (..., 3, 3)
    M = R * glob_scale * scale[..., None, :]  # (..., 3, 3)
    # TODO: save upper right because symmetric
    return M @ M.transpose(-1, -2)  # (..., 3, 3)


def project_cov3d_ewa(
    mean3d: Tensor,
    cov3d: Tensor,
    viewmat: Tensor,
    fx: float,
    fy: float,
    tan_fovx: float,
    tan_fovy: float,
) -> Tuple[Tensor, Tensor]:
    assert mean3d.shape[-1] == 3, mean3d.shape
    assert cov3d.shape[-2:] == (3, 3), cov3d.shape
    assert viewmat.shape[-2:] == (4, 4), viewmat.shape
    W = viewmat[..., :3, :3]  # (..., 3, 3)
    p = viewmat[..., :3, 3]  # (..., 3)
    t = torch.einsum("...ij,...j->...i", W, mean3d) + p  # (..., 3)

    rz = 1.0 / t[..., 2]  # (...,)
    rz2 = rz ** 2  # (...,)

    lim_x = 1.3 * torch.tensor([tan_fovx], device=mean3d.device)
    lim_y = 1.3 * torch.tensor([tan_fovy], device=mean3d.device)
    x_clamp = t[..., 2] * torch.clamp(t[..., 0] * rz, min=-lim_x, max=lim_x)
    y_clamp = t[..., 2] * torch.clamp(t[..., 1] * rz, min=-lim_y, max=lim_y)
    t = torch.stack([x_clamp, y_clamp, t[..., 2]], dim=-1)

    O = torch.zeros_like(rz)
    J = torch.stack(
        [fx * rz, O, -fx * t[..., 0] * rz2, O, fy * rz, -fy * t[..., 1] * rz2],
        dim=-1,
    ).reshape(*rz.shape, 2, 3)
    T = torch.matmul(J, W)  # (..., 2, 3)
    cov2d = torch.einsum("...ij,...jk,...kl->...il", T, cov3d, T.transpose(-1, -2))
    # add a little blur along axes and (TODO save upper triangular elements)
    det_orig = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] * cov2d[..., 0, 1]
    cov2d[..., 0, 0] = cov2d[..., 0, 0] + 0.3
    cov2d[..., 1, 1] = cov2d[..., 1, 1] + 0.3
    det_blur = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] * cov2d[..., 0, 1]
    compensation = torch.sqrt(torch.clamp(det_orig / det_blur, min=0))
    return cov2d[..., :2, :2], compensation


def compute_compensation(cov2d_mat: Tensor):
    """
    params: cov2d matrix (*, 2, 2)
    returns: compensation factor as calculated in project_cov3d_ewa
    """
    det_denom = cov2d_mat[..., 0, 0] * cov2d_mat[..., 1, 1] - cov2d_mat[..., 0, 1] ** 2
    det_nomin = (cov2d_mat[..., 0, 0] - 0.3) * (cov2d_mat[..., 1, 1] - 0.3) - cov2d_mat[
        ..., 0, 1
    ] ** 2
    return torch.sqrt(torch.clamp(det_nomin / det_denom, min=0))


def compute_cov2d_bounds(cov2d_mat: Tensor):
    """
    param: cov2d matrix (*, 2, 2)
    returns: conic parameters (*, 3)
    """
    det_all = cov2d_mat[..., 0, 0] * cov2d_mat[..., 1, 1] - cov2d_mat[..., 0, 1] ** 2
    valid = det_all != 0
    # det = torch.clamp(det, min=eps)
    det = det_all[valid]
    cov2d = cov2d_mat[valid]
    conic = torch.stack(
        [
            cov2d[..., 1, 1] / det,
            -cov2d[..., 0, 1] / det,
            cov2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # (..., 3)
    b = (cov2d[..., 0, 0] + cov2d[..., 1, 1]) / 2  # (...,)
    v1 = b + torch.sqrt(torch.clamp(b ** 2 - det, min=0.1))  # (...,)
    v2 = b - torch.sqrt(torch.clamp(b ** 2 - det, min=0.1))  # (...,)
    radius = torch.ceil(3.0 * torch.sqrt(torch.max(v1, v2)))  # (...,)
    radius_all = torch.zeros(*cov2d_mat.shape[:-2], device=cov2d_mat.device)
    conic_all = torch.zeros(*cov2d_mat.shape[:-2], 3, device=cov2d_mat.device)
    radius_all[valid] = radius
    conic_all[valid] = conic
    return conic_all, radius_all, valid


def project_pix(fxfy, p_view, center, eps=1e-6):
    fx, fy = fxfy
    cx, cy = center

    rw = 1.0 / (p_view[..., 2] + 1e-6)
    p_proj = (p_view[..., 0] * rw, p_view[..., 1] * rw)
    u, v = (p_proj[0] * fx + cx, p_proj[1] * fy + cy)
    return torch.stack([u, v], dim=-1)


def clip_near_plane(p, viewmat, clip_thresh=0.01):
    R = viewmat[:3, :3]
    T = viewmat[:3, 3]
    p_view = torch.einsum("ij,nj->ni", R, p) + T[None]
    return p_view, p_view[..., 2] < clip_thresh


def get_tile_bbox(pix_center, pix_radius, tile_bounds, block_width):
    tile_size = torch.tensor(
        [block_width, block_width], dtype=torch.float32, device=pix_center.device
    )
    tile_center = pix_center / tile_size
    tile_radius = pix_radius[..., None] / tile_size

    top_left = (tile_center - tile_radius).to(torch.int32)
    bottom_right = (tile_center + tile_radius).to(torch.int32) + 1
    tile_min = torch.stack(
        [
            torch.clamp(top_left[..., 0], 0, tile_bounds[0]),
            torch.clamp(top_left[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    tile_max = torch.stack(
        [
            torch.clamp(bottom_right[..., 0], 0, tile_bounds[0]),
            torch.clamp(bottom_right[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    return tile_min, tile_max


def project_gaussians_forward(
    means3d,
    scales,
    glob_scale,
    quats,
    viewmat,
    intrins,
    img_size,
    block_width,
    clip_thresh=0.01,
):
    tile_bounds = (
        (img_size[0] + block_width - 1) // block_width,
        (img_size[1] + block_width - 1) // block_width,
        1,
    )
    fx, fy, cx, cy = intrins
    tan_fovx = 0.5 * img_size[0] / fx
    tan_fovy = 0.5 * img_size[1] / fy
    p_view, is_close = clip_near_plane(means3d, viewmat, clip_thresh)
    cov3d = scale_rot_to_cov3d(scales, glob_scale, quats)
    cov2d, compensation = project_cov3d_ewa(
        means3d, cov3d, viewmat, fx, fy, tan_fovx, tan_fovy
    )
    conic, radius, det_valid = compute_cov2d_bounds(cov2d)
    xys = project_pix((fx, fy), p_view, (cx, cy))
    tile_min, tile_max = get_tile_bbox(xys, radius, tile_bounds, block_width)
    tile_area = (tile_max[..., 0] - tile_min[..., 0]) * (
        tile_max[..., 1] - tile_min[..., 1]
    )
    mask = (tile_area > 0) & (~is_close) & det_valid

    num_tiles_hit = tile_area
    depths = p_view[..., 2]
    radii = radius.to(torch.int32)

    radii = torch.where(~mask, 0, radii)
    conic = torch.where(~mask[..., None], 0, conic)
    xys = torch.where(~mask[..., None], 0, xys)
    cov3d = torch.where(~mask[..., None, None], 0, cov3d)
    cov2d = torch.where(~mask[..., None, None], 0, cov2d)
    compensation = torch.where(~mask, 0, compensation)
    num_tiles_hit = torch.where(~mask, 0, num_tiles_hit)
    depths = torch.where(~mask, 0, depths)

    i, j = torch.triu_indices(3, 3)
    cov3d_triu = cov3d[..., i, j]
    i, j = torch.triu_indices(2, 2)
    cov2d_triu = cov2d[..., i, j]
    # return (
    #     cov3d_triu,
    #     cov2d_triu,
    #     xys,
    #     depths,
    #     radii,
    #     conic,
    #     compensation,
    #     num_tiles_hit,
    #     mask,
    # )
    return (xys, depths, radii, conic, compensation, num_tiles_hit, cov3d)


def map_gaussian_to_intersects(
    num_points, xys, depths, radii, cum_tiles_hit, tile_bounds, block_width
):
    num_intersects = cum_tiles_hit[-1]
    isect_ids = torch.zeros(num_intersects, dtype=torch.int64, device=xys.device)
    gaussian_ids = torch.zeros(num_intersects, dtype=torch.int32, device=xys.device)

    for idx in range(num_points):
        if radii[idx] <= 0:
            break

        tile_min, tile_max = get_tile_bbox(
            xys[idx], radii[idx], tile_bounds, block_width
        )

        cur_idx = 0 if idx == 0 else cum_tiles_hit[idx - 1].item()

        # Get raw byte representation of the float value at the given index
        raw_bytes = struct.pack("f", depths[idx])

        # Interpret those bytes as an int32_t
        depth_id_n = struct.unpack("i", raw_bytes)[0]

        for i in range(tile_min[1], tile_max[1]):
            for j in range(tile_min[0], tile_max[0]):
                tile_id = i * tile_bounds[0] + j
                isect_ids[cur_idx] = (tile_id << 32) | depth_id_n
                gaussian_ids[cur_idx] = idx
                cur_idx += 1

    return isect_ids, gaussian_ids


def get_tile_bin_edges(num_intersects, isect_ids_sorted, tile_bounds):
    tile_bins = torch.zeros(
        (tile_bounds[0] * tile_bounds[1], 2),
        dtype=torch.int32,
        device=isect_ids_sorted.device,
    )

    for idx in range(num_intersects):
        cur_tile_idx = isect_ids_sorted[idx] >> 32

        if idx == 0:
            tile_bins[cur_tile_idx, 0] = 0
            continue

        if idx == num_intersects - 1:
            tile_bins[cur_tile_idx, 1] = num_intersects
            break

        prev_tile_idx = isect_ids_sorted[idx - 1] >> 32

        if cur_tile_idx != prev_tile_idx:
            tile_bins[prev_tile_idx, 1] = idx
            tile_bins[cur_tile_idx, 0] = idx

    return tile_bins


def rasterize_forward(
    tile_bounds,
    block,
    img_size,
    gaussian_ids_sorted,
    tile_bins,
    xys,
    conics,
    colors,
    opacities,
    background,
):
    channels = colors.shape[1]
    out_img = torch.zeros(
        (img_size[1], img_size[0], channels), dtype=torch.float32, device=xys.device
    )
    final_Ts = torch.zeros(
        (img_size[1], img_size[0]), dtype=torch.float32, device=xys.device
    )
    final_idx = torch.zeros(
        (img_size[1], img_size[0]), dtype=torch.int32, device=xys.device
    )
    for i in range(img_size[1]):
        for j in range(img_size[0]):
            tile_id = (i // block[0]) * tile_bounds[0] + (j // block[1])
            tile_bin_start = tile_bins[tile_id, 0]
            tile_bin_end = tile_bins[tile_id, 1]
            T = 1.0

            for idx in range(tile_bin_start, tile_bin_end):
                gaussian_id = gaussian_ids_sorted[idx]
                conic = conics[gaussian_id]
                center = xys[gaussian_id]
                delta = center - torch.tensor(
                    [j, i], dtype=torch.float32, device=xys.device
                )

                sigma = (
                    0.5
                    * (conic[0] * delta[0] * delta[0] + conic[2] * delta[1] * delta[1])
                    + conic[1] * delta[0] * delta[1]
                )

                if sigma < 0:
                    continue

                opac = opacities[gaussian_id]
                alpha = min(0.999, opac * torch.exp(-sigma))

                if alpha < 1 / 255:
                    continue

                next_T = T * (1 - alpha)

                if next_T <= 1e-4:
                    idx -= 1
                    break

                vis = alpha * T

                out_img[i, j] += vis * colors[gaussian_id]
                T = next_T

            final_Ts[i, j] = T
            final_idx[i, j] = idx
            out_img[i, j] += T * background

    return out_img, final_Ts, final_idx


def rasterize_gaussians_forward(
    xys: Float[Tensor, "*batch 2"],
    depths: Float[Tensor, "*batch 1"],
    radii: Float[Tensor, "*batch 1"],
    conics: Float[Tensor, "*batch 3"],
    num_tiles_hit: Int[Tensor, "*batch 1"],
    colors: Float[Tensor, "*batch channels"],
    opacity: Float[Tensor, "*batch 1"],
    img_height: int,
    img_width: int,
    block_width: int,
    background: Optional[Float[Tensor, "channels"]] = None,
    return_alpha: Optional[bool] = False,
) -> Tensor:
    """Rasterizes 2D gaussians by sorting and binning gaussian intersections for each tile and returns an N-dimensional output using alpha-compositing.

    Note:
        This function is differentiable w.r.t the xys, conics, colors, and opacity inputs.

    Args:
        xys (Tensor): xy coords of 2D gaussians.
        depths (Tensor): depths of 2D gaussians.
        radii (Tensor): radii of 2D gaussians
        conics (Tensor): conics (inverse of covariance) of 2D gaussians in upper triangular format
        num_tiles_hit (Tensor): number of tiles hit per gaussian
        colors (Tensor): N-dimensional features associated with the gaussians.
        opacity (Tensor): opacity associated with the gaussians.
        img_height (int): height of the rendered image.
        img_width (int): width of the rendered image.
        block_width (int): MUST match whatever block width was used in the project_gaussians call. integer number of pixels between 2 and 16 inclusive
        background (Tensor): background color
        return_alpha (bool): whether to return alpha channel

    Returns:
        A Tensor:

        - **out_img** (Tensor): N-dimensional rendered output image.
        - **out_alpha** (Optional[Tensor]): Alpha channel of the rendered output image.
    """
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"
    if colors.dtype == torch.uint8:
        # make sure colors are float [0,1]
        colors = colors.float() / 255

    if background is not None:
        assert (
            background.shape[0] == colors.shape[-1]
        ), f"incorrect shape of background color tensor, expected shape {colors.shape[-1]}"
    else:
        background = torch.ones(
            colors.shape[-1], dtype=torch.float32, device=colors.device
        )

    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")

    if colors.ndimension() != 2:
        raise ValueError("colors must have dimensions (N, D)")

    # below from _RasterizeGaussians
    num_points = xys.size(0)
    tile_bounds = (
        (img_width + block_width - 1) // block_width,
        (img_height + block_width - 1) // block_width,
        1,
    )
    block = (block_width, block_width, 1)
    img_size = (img_width, img_height, 1)

    num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)

    if num_intersects < 1:
        out_img = (
            torch.ones(img_height, img_width, colors.shape[-1], device=xys.device)
            * background
        )
        gaussian_ids_sorted = torch.zeros(0, 1, device=xys.device)
        tile_bins = torch.zeros(0, 2, device=xys.device)
        final_Ts = torch.zeros(img_height, img_width, device=xys.device)
        final_idx = torch.zeros(img_height, img_width, device=xys.device)
    else:
        (
            isect_ids_unsorted,
            gaussian_ids_unsorted,
            isect_ids_sorted,
            gaussian_ids_sorted,
            tile_bins,
        ) = bin_and_sort_gaussians(
            num_points,
            num_intersects,
            xys,
            depths,
            radii,
            cum_tiles_hit,
            tile_bounds,
            block_width,
        )
        if colors.shape[-1] == 3:
            rasterize_fn = rasterize_forward
        else:
            rasterize_fn = rasterize_forward

        out_img, final_Ts, final_idx = rasterize_fn(
            tile_bounds,
            block,
            img_size,
            gaussian_ids_sorted,
            tile_bins,
            xys,
            conics,
            colors,
            opacity,
            background,
        )

    if return_alpha:
        out_alpha = 1 - final_Ts
        return out_img, out_alpha
    else:
        return out_img