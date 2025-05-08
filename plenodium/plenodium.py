# ruff: noqa: E741
# Copyright 2024 Huapeng Li, Wenxuan Song, Tianao Xu, Alexandre Elsig and Jonas KulhanekS. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Python package for combining 3DGS with volume rendering to enable water/fog modeling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn

# from nerfstudio.data.datamanagers.base_datamanager import DataManager
import torchvision.transforms.functional as TF
from gsplat import quat_scale_to_covar_preci
from pytorch_msssim import MS_SSIM, SSIM
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import k_nearest_sklearn, random_quat_tensor
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB
from plenodium._torch_impl import quat_to_rotmat
from plenodium.project_gaussians import project_gaussians
from plenodium.rasterize import rasterize_gaussians, rasterize_gaussians2
from plenodium.sh import num_sh_bases, sparse_spherical_harmonics, spherical_harmonics


# @torch_compile
def interpolation_medium(T,medium_dc,medium_rest):
    """
    Interpolates the medium SH
    """
    medium_feature = torch.cat((medium_dc[:,None],medium_rest),dim=1)
    medium_size = medium_feature.shape
    medium_feature = medium_feature.reshape(1,-1,2,2,2)
    position = T.reshape(1,1,1,1,3)
    medium = torch.nn.functional.grid_sample(medium_feature, position, mode='bilinear', align_corners=True)
    return medium.reshape(*medium_size[:3])


# @torch_compile
def depth_ranking_loss_fn(pred, disparity, n_patch = 16):
    pred = pred.permute(2, 0, 1)
    disparity = disparity.permute(2, 0, 1)
    disparity = disparity / disparity.max()
    pred = pred/(pred.max().detach())
    pred = nn.functional.adaptive_avg_pool2d(pred, n_patch)
    disparity = nn.functional.adaptive_avg_pool2d(disparity,n_patch)
    pred_diff = pred.reshape(1,-1,1)-pred.reshape(1,1,-1)
    disparity_diff = disparity.reshape(1, -1, 1) - disparity.reshape(1, 1, -1)
    return (pred_diff * disparity_diff).relu().mean()


@dataclass
class WaterModelConfig(ModelConfig):
    """Water Splatting Model Config"""

    _target: Type = field(default_factory=lambda: WaterModel)
    num_steps: int = 15001
    """Number of steps to train the model"""
    warmup_length: int = 600
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "black"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.5
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_alpha_thresh_post: float = 0.1
    """threshold of opacity for post culling gaussians"""
    reset_alpha_thresh: float = 0.45
    """threshold of opacity for resetting alpha"""
    cull_scale_thresh: float = 10.
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    zero_medium: bool = False
    """If True, zero out the medium field"""
    reset_alpha_every: int = 5
    """Every this many refinement steps, reset the alpha"""
    abs_grad_densification: bool = True
    """If True, use absolute gradient for densification"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians (0.0004, 0.0008)"""
    densify_size_thresh: float = 0.001
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    clip_thresh: float = 0.01
    """minimum depth threshold"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    # stop_screen_size_at: int = 10000
    stop_screen_size_at: int = 0
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    lpips_lambda: float = 0.00
    """weight of lpips loss"""
    main_loss: Literal["l1", "reg_l1", "reg_l2",'water_reg_l1'] = "reg_l1"
    """main loss to use"""
    ssim_loss: Literal["reg_ssim", "ssim","water_reg_ssim"] = "reg_ssim"
    """ssim loss to use"""
    stop_split_at: int = 10000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """

    medium_sh_degree: int = 3
    """degree of the spherical harmonics to use for the medium field"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    inject_noise_to_position: bool = False
    """If enabled, inject noise to the gaussians position"""
    noise_lr: float = 2e5
    """The noise learning rate"""
    use_depth_ranking_loss: bool = True
    """If enabled, use depth ranking loss"""
    point_complementation: bool = True
    """If enabled, use point complementation"""
    point_complementation_at: int = 100
    """"""
    multi_scale_simm: bool = True
    """If enabled, use multi-scale ssim"""
    correct_transform: bool = False
    """use the correct version of the transformation"""
    near_far_thr: float = 0.5
    well_init_thr: float = 0.999
    poor_init_thr: float = 0.975


class WaterModel(Model):
    """
    Args:
        config: Water configuration to instantiate model22222
    """

    config: WaterModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        self.colour_activation = nn.Sigmoid()
        self.sigma_activation = nn.Softplus()

        #! medium_SH
        self.medium_feature_dc = nn.Parameter(
            torch.tensor([[0., 0., 0.], [-5., -5., -5.], [-5., -5., -5.]])
        )
        self.medium_feature_rest = nn.Parameter(
            torch.zeros(3, num_sh_bases(self.config.medium_sh_degree) - 1, 3)
        )

        # self.medium_feature_dc = nn.Parameter(
        #     torch.tensor([[0., 0., 0.], [-5., -5., -5.], [-5., -5., -5.]]).reshape(3,3,1,1,1).repeat(1,1,2,2,2)
        # )
        # self.medium_feature_rest = nn.Parameter(
        #     torch.zeros(3, num_sh_bases(self.config.medium_sh_degree) - 1, 3).reshape(3,-1,3,1,1,1).repeat(1,1,1,2,2,2)
        # )

        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = k_nearest_sklearn(means.data, 3)
        # distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        self.avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(self.avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.ssim_fn = MS_SSIM(data_range=1.0, size_average=True, channel=3, weights=[0.6,0.3,0.1]) if self.config.multi_scale_simm else SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        self.background_color = torch.nn.Parameter(torch.tensor([0.,0.,0.]))

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = self.config.num_steps
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):
        assert step == self.step
        if step<self.config.point_complementation_at:
            return
        # to save some training time, we no longer need to update those stats post refinement
        # if self.step >= self.config.stop_split_at:
        #     return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            if self.config.abs_grad_densification:
                assert self.xys_grad_abs is not None
                grads = self.xys_grad_abs.detach().norm(dim=-1)
            else:
                assert self.xys.grad is not None
                grads = self.xys.grad.detach().norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.depths_accum = self.depths
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]
                self.depths_accum[visible_mask] = self.depths[visible_mask] + self.depths_accum[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = self.step < self.config.stop_split_at and (
                self.step % reset_interval
                > self.num_train_data + self.config.refine_every
            )
            if do_densification:
                # then we densify
                assert (
                    self.xys_grad_norm is not None
                    and self.vis_counts is not None
                    and self.max_2Dsize is not None
                )
                avg_grad_norm = (
                    (self.xys_grad_norm / self.vis_counts)
                    * 0.5
                    * max(self.last_size[0], self.last_size[1])
                )

                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()

                splits = (
                    self.scales.exp().max(dim=-1).values
                    > self.config.densify_size_thresh
                ).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (
                        self.max_2Dsize > self.config.split_screen_size
                    ).squeeze()
                splits &= high_grads

                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)

                dups = (
                    self.scales.exp().max(dim=-1).values
                    <= self.config.densify_size_thresh
                ).squeeze()
                dups &= high_grads

                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat(
                            [param.detach(), split_params[name], dup_params[name]],
                            dim=0,
                        )
                    )

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # if self.step < self.config.stop_screen_size_at:
                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )
                deleted_mask = self.cull_gaussians(splits_mask)
            elif (
                self.step >= self.config.stop_split_at
                and self.config.continue_cull_post_densification
            ):
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

                # reset the exp of optimizer
                for key in [
                    "medium_feature_dc",
                    "medium_feature_rest",
                ]:
                    optim = optimizers.optimizers[key]
                    param = optim.param_groups[0]["params"][0]
                    param_state = optim.state[param]
                    if "exp_avg" in param_state:
                        param_state["exp_avg"] = torch.zeros_like(
                            param_state["exp_avg"]
                        )
                        param_state["exp_avg_sq"] = torch.zeros_like(
                            param_state["exp_avg_sq"]
                        )

            if (
                self.step < self.config.stop_split_at
                and self.step % reset_interval == self.config.refine_every
            ):
                # Reset value is set to be reset_alpha_thresh
                reset_value = self.config.reset_alpha_thresh
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(
                        torch.tensor(reset_value, device=self.device)
                    ).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

                self.inject_noise_to_position(optimizers, step)

            self.xys_grad_norm = None
            self.vis_counts = None
            self.depths_accum = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        if self.step < self.config.stop_split_at:
            cull_alpha_thresh = self.config.cull_alpha_thresh
        else:
            cull_alpha_thresh = self.config.cull_alpha_thresh_post
        culls = (torch.sigmoid(self.opacities) < cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            # related_scale = (self.scales.max(dim=-1).values - self.scales.max(dim=-1).values.min()) / (math.log(self.config.cull_scale_thresh)- self.scales.max(dim=-1).values.min())
            # toobigs = (
            #     toobigs | ((related_scale - self.opacities.sigmoid().squeeze()) > 1-self.config.cull_alpha_thresh)
            # )
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        # CONSOLE.log(
        #     f"Culled {n_bef - self.num_points} gaussians "
        #     f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        # )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        # CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        # CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    @torch.no_grad()
    def inject_noise_to_position(self,optimers: Optimizers,step):
        if (not self.config.inject_noise_to_position) or  step < self.config.warmup_length: 
            return
        lr = optimers.optimizers["means"].param_groups[0]["lr"]
        assert step == self.step
        opacities = torch.sigmoid(self.opacities.flatten())
        scales = torch.exp(self.scales)
        covars, _ = quat_scale_to_covar_preci(
            self.quats,
            scales,
            compute_covar=True,
            compute_preci=False,
            triu=False,
        )

        def op_sigmoid(x, k=100, x0=0.995):
            return 1 / (1 + torch.exp(-k * (x - x0)))

        noise = (
            torch.randn_like(self.means)
            * (op_sigmoid(1 - opacities)).unsqueeze(-1)
            * lr * self.config.noise_lr # 这是 lr noisescalelr
        )
        noise = torch.einsum("bij,bj->bi", covars, noise)
        self.means.add_(noise)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        # cbs.append(
        #     TrainingCallback(
        #         [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
        #         self.inject_noise_to_position,
        #         args=[
        #             training_callback_attributes.optimizers,
        #         ],
        #     )
        # )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[
                    training_callback_attributes.optimizers,
                ],
            )
        )
        if self.config.point_complementation:
            cbs.append(
                TrainingCallback(
                    [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    self.point_complementation,
                    update_every_num_iters=self.config.point_complementation_at,
                    args=[
                        training_callback_attributes.pipeline.datamanager,
                        training_callback_attributes.optimizers,
                    ],
                )
            )
        return cbs

    @torch.no_grad()
    def point_complementation(self, datamanager, optimizers: Optimizers, step):
        if self.step == 0 or self.step > self.config.point_complementation_at+1:
            return
        MEAN = []
        RGB=[]
        SIZE= []
        for i in range(len(datamanager.train_dataset.cameras)):
            self.eval()
            camera, batch = datamanager.next_train(self.step)
            camera.rescale_output_resolution(1 / (2 ** (self.config.num_downscales+2)))
            psuedo_disparity = batch["depth"]
            image = batch["image"]
            model_outputs = self(camera)
            pred_depth = model_outputs["depth"]
            pred_alpha = model_outputs["accumulation"]
            new_size = pred_depth.shape[:2]
            psuedo_disparity = TF.resize(
                psuedo_disparity.permute(2, 0, 1), new_size, antialias=True
            ).permute(1, 2, 0)
            image = TF.resize(image.permute(2, 0, 1), new_size,antialias=True).permute(1, 2, 0)
            psuedo_disparity = psuedo_disparity/psuedo_disparity.max()
            obj = (pred_alpha >= self.config.well_init_thr) & (psuedo_disparity > self.config.near_far_thr)
            hole = (psuedo_disparity > self.config.near_far_thr) & (pred_alpha < self.config.poor_init_thr)
            if (hole).sum() == 0:
                continue

            pred_depth = pred_depth * pred_alpha
            y = pred_depth[obj]
            x = psuedo_disparity[obj]
            a = ((y-y.mean())*(x-x.mean())).sum()/((x-x.mean())**2).sum()
            b = y.mean() - a * x.mean()
            psuedo_depth_refine = psuedo_disparity * a + b

            cx = camera.cx.item()
            cy = camera.cy.item()
            W, H = int(camera.width.item()), int(camera.height.item())
            y = torch.linspace(0.0, H, H, device=self.device)
            x = torch.linspace(0.0, W, W, device=self.device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            yy = (yy - cy) / camera.fy.item()
            xx = (xx - cx) / camera.fx.item()
            yy = yy[hole.squeeze()]
            xx = xx[hole.squeeze()]

            directions = torch.stack([xx,yy,torch.ones_like(xx)], dim=-1)

            # shift the camera to center of scene looking at center
            R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
            T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
            # flip the z and y axes to align with gsplat conventions
            R_edit = torch.diag(
                torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype)
            )
            R = R @ R_edit

            pos = psuedo_depth_refine[hole][:,None].repeat(1,3) * directions @ R.T + T[ None, :, 0] # [1,0,0]
            color = image[hole.squeeze()]
            size = 4 * psuedo_depth_refine[hole][:,None]/ (camera.fx.item() + camera.fy.item())
            MEAN.append(pos)
            RGB.append(color)
            SIZE.append(size)
            camera.rescale_output_resolution((2 ** (self.config.num_downscales+2)))
            self.train()

        if not MEAN:
            return
        MEAN = torch.cat(MEAN, dim=0)
        RGB = torch.cat(RGB, dim=0)
        SIZE= torch.cat(SIZE, dim=0)
        n_add = MEAN.shape[0]
        if n_add < 50: 
            return
        CONSOLE.log(f"Add {n_add}/{self.num_points} points to the model")
        self.add_points(MEAN,RGB,SIZE,optimizers)
        # self.vis_counts = torch.cat(
        #     [self.vis_counts, torch.zeros(n_add, device=self.device)], dim=0
        # )
        # self.depths_accum = torch.cat(
        #     [self.depths_accum, torch.zeros(n_add, device=self.device)], dim=0
        # )
        # self.max_2Dsize = torch.cat(
        #     [self.max_2Dsize, torch.zeros(n_add, device=self.device)], dim=0
        # )
        # self.xys_grad_norm = torch.cat(
        #     [self.xys_grad_norm, torch.zeros(n_add, device=self.device)], dim=0
        # )
        # pass

    def add_points(self, means,rgb,size, optimizers):
        pre_num = self.num_points
        add_num = means.size(0)
        means = torch.cat([self.gauss_params["means"].detach(), means], dim=0)
        distances, _ = k_nearest_sklearn(means.data, 3)
        self.avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(
            torch.cat(
                [
                    self.gauss_params["scales"].detach(),
                    torch.log(
                        torch.min(self.avg_dist[pre_num:].to(self.device), size).repeat(
                            1, 3
                        )
                    ),
                ]
            )
        )
        quats = torch.nn.Parameter(
            torch.cat(
                [
                    self.gauss_params["quats"].detach(),
                    random_quat_tensor(add_num).to(self.device),
                ]
            )
        )
        dim_sh = num_sh_bases(self.config.sh_degree)

        features_dc = torch.cat(
            [
                self.gauss_params["features_dc"].detach(),
                RGB2SH(rgb / 255),
            ],
            dim=0,
        )
        features_rest = torch.cat(
            [
                self.gauss_params["features_rest"].detach(),
                torch.zeros(add_num, dim_sh - 1, 3, device=self.device),
            ],
            dim=0,
        )
        opacities = torch.nn.Parameter(
            torch.cat(
                [
                    self.gauss_params["opacities"].detach(),
                    torch.logit(
                        0.1 * torch.ones(add_num, 1, device=self.device)
                    ),  #! 0.99 for debug, 0.1 for the use
                ]
            )
        )
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )
        self.dup_in_all_optim(optimizers, torch.multinomial(torch.ones(pre_num),add_num,replacement=True), 1)

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        gps["medium_feature_dc"] = [self.medium_feature_dc]
        gps["medium_feature_rest"] = [self.medium_feature_rest]
        # gps["background_color"] = [self.background_color]
        # gps["direction_encoding"] = list(self.direction_encoding.parameters())
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [image.shape[0] // d, image.shape[1] // d]

            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            return TF.resize(image.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        return image

    def get_outputs(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)

        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv

        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        self.last_fx = camera.fx.item()
        self.last_fy = camera.fy.item()

        # Medium
        # Encode directions
        y = torch.linspace(0., H, H, device=self.device)
        x = torch.linspace(0., W, W, device=self.device)
        yy, xx = torch.meshgrid(y, x,indexing="ij")
        yy = (yy - cy) / camera.fy.item()
        xx = (xx - cx) / camera.fx.item()

        directions = (
            -1 * torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
            if self.config.correct_transform
            else torch.stack([yy, xx, -1 * torch.ones_like(xx)], dim=-1)
        )
        norms = torch.linalg.norm(directions, dim=-1, keepdim=True)
        directions = directions / norms
        directions = directions @ R.T if self.config.correct_transform else directions @ R

        directions_flat = directions.view(-1, 3)
        outputs_shape = directions.shape[:-1]
        self.medium_feature = torch.cat([self.medium_feature_dc[:,None],self.medium_feature_rest],dim=1)
        # self.medium_feature = interpolation_medium(T, self.medium_feature_dc,self.medium_feature_rest)
        n = min(
            self.step // self.config.sh_degree_interval, self.config.medium_sh_degree
        )

        medium_base_out = sparse_spherical_harmonics(
            n,
            directions_flat,
            self.medium_feature,
        )

        medium_rgb, medium_bs, medium_attn=medium_base_out.chunk(3,0)
        medium_rgb = torch.clamp(medium_rgb.view(*outputs_shape, -1) + 0.5, min=0.0)
        medium_bs =  self.sigma_activation(medium_bs.view(*outputs_shape, -1))
        medium_attn = self.sigma_activation(medium_attn.view(*outputs_shape, -1))

        if self.config.zero_medium:
            medium_rgb = torch.zeros_like(medium_rgb)
            medium_bs = torch.zeros_like(medium_bs)
            medium_attn = torch.zeros_like(medium_attn)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = medium_rgb
                depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = medium_rgb.new_zeros(*rgb.shape[:2], 1)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": medium_rgb, 
                        "rgb_object": torch.zeros_like(rgb), "rgb_medium": medium_rgb, "pred_image": rgb,
                        "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn}
        else:
            crop_ids = None

        if crop_ids is not None and crop_ids.sum() != 0:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default

        self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = (
            project_gaussians(  # type: ignore
                means_crop,
                torch.exp(scales_crop),
                1,
                quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                viewmat.squeeze()[:3, :],
                camera.fx.item(),
                camera.fy.item(),
                cx,
                cy,
                H,
                W,
                BLOCK_WIDTH,
                clip_thresh=self.config.clip_thresh,
            )
        )  # type: ignore

        self.depths = depths.detach()

        # rescale the camera back to original dimensions before returning
        camera.rescale_output_resolution(camera_downscale)

        if (self.radii).sum() == 0:
            rgb = medium_rgb
            depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = medium_rgb.new_zeros(*rgb.shape[:2], 1)
            return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": medium_rgb, 
                    "rgb_object": torch.zeros_like(rgb), "rgb_clear": torch.zeros_like(rgb), "rgb_clear_clamp": torch.zeros_like(rgb), "rgb_medium": medium_rgb, "pred_image": rgb,
                    "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn}

        if self.training:
            self.xys.retain_grad()

        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        assert (num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        opacities = None
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * comp[:, None]
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        self.xys_grad_abs = torch.zeros_like(self.xys)

        rasterize_fn = rasterize_gaussians2 if self.config.use_depth_ranking_loss else rasterize_gaussians

        rgb_object, rgb_clear, rgb_medium, depth_im, alpha = rasterize_fn(  # type: ignore
            self.xys,
            self.xys_grad_abs,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            medium_rgb,
            medium_bs,
            medium_attn,
            H,
            W,
            BLOCK_WIDTH,
            background=medium_rgb,
            return_alpha=True,
            step=self.step,
        )  # type: ignore

        rgb = rgb_object + rgb_medium
        rgb_clear_clamp = torch.clamp(rgb_clear, 0., 1.)
        rgb_clear_unclamp = rgb_clear
        rgb_clear = rgb_clear / (rgb_clear + 1.)

        depth_im = depth_im[..., None]
        alpha = alpha[..., None]
        depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())
        # rgb = rgb + (1 - alpha) @ self.background_color[None, :].clamp(0.0, 1.0)
        return {
            "rgb": rgb,
            "depth": depth_im,
            "accumulation": alpha,
            "background": medium_rgb,
            "rgb_object": rgb_object,
            "rgb_clear": rgb_clear,
            "rgb_clear_unclamp": rgb_clear_unclamp,
            "rgb_clear_clamp": rgb_clear_clamp,
            "rgb_medium": rgb_medium,
            "pred_image": rgb,
            "medium_rgb": medium_rgb,
            "medium_bs": medium_bs,
            "medium_attn": medium_attn,
        }  # type: ignore

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            # alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return image[..., :3]
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["pred_image"]
        predicted_rgb = torch.clamp(predicted_rgb, 0.0, 1.0)
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points
        for i in range(3):
            # 3 channels
            metrics_dict[f"medium_attn_{i}"] = outputs["medium_attn"][:, :, i].mean()
            metrics_dict[f"medium_bs_{i}"] = outputs["medium_bs"][:, :, i].mean()
            metrics_dict[f"medium_rgb_{i}"] = outputs["medium_rgb"][:, :, i].mean()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["pred_image"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        if self.config.main_loss == "l1":
            recon_loss = torch.abs(gt_img - pred_img).mean()
        elif self.config.main_loss == "reg_l1":
            recon_loss = torch.abs((gt_img - pred_img) / (pred_img.detach() + 1e-3)).mean()
        elif self.config.main_loss == "reg_l2":
            recon_loss = (((pred_img - gt_img) / (pred_img.detach() + 1e-3)) ** 2).mean()
        elif self.config.main_loss == "water_reg_l1":
            # water_ref = (pred_img - outputs["medium_rgb"]).abs().detach() + 1e-3
            recon_loss = (
                0.05+torch.abs(
                    (gt_img/pred_img.detach() - pred_img/pred_img.detach())
                ).mean()
                + 0.95 * torch.abs((gt_img - pred_img) / (pred_img.detach() + 1e-3)).mean()
            )
        else:
            raise ImportError(f"Unknown main loss: {self.config.main_loss}")

        if self.config.ssim_loss == "reg_ssim":
            simloss = 1 - self.ssim_fn((gt_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...], (pred_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...])
        elif self.config.ssim_loss == "ssim":
            simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        elif self.config.ssim_loss == "water_reg_ssim":
            water_ref = (pred_img - outputs["medium_rgb"]).abs().detach() + 1e-3
            simloss = 1 - self.ssim(
                (gt_img / water_ref).permute(2, 0, 1)[None, ...],
                (pred_img / water_ref).permute(2, 0, 1)[None, ...],
            )
        else: 
            raise ImportError(f"Unknown ssim loss: {self.config.ssim_loss}")

        lpips_loss = (
            self.lpips(
                    gt_img.permute(2, 0, 1)[None, ...],
                    pred_img.permute(2, 0, 1)[None, ...].clamp(0.0, 1.0),
            )
            if self.config.lpips_lambda > 0
            else torch.tensor(0.0).to(self.device)
        )

        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        depth_ranking_loss = 5 * depth_ranking_loss_fn(
            outputs["depth"], batch["depth"]
        ) if self.config.use_depth_ranking_loss and self.step<self.config.stop_split_at else torch.tensor(0.0).to(self.device)

        return {
            "main_loss": (1 - self.config.ssim_lambda - self.config.lpips_lambda) * recon_loss
            + self.config.ssim_lambda * simloss
            + self.config.lpips_lambda * lpips_loss
            ,
            "scale_reg": scale_reg,
            "depth_ranking_loss": depth_ranking_loss,
        }

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device), obb_box=obb_box)
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])

        predicted_rgb = outputs["pred_image"]
        predicted_rgb = torch.clamp(predicted_rgb, 0.0, 1.0)

        d = self._get_downscale_factor()
        if d > 1:
            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            predicted_rgb = TF.resize(predicted_rgb.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            predicted_rgb = predicted_rgb

        output_gt_rgb = gt_rgb.cpu()

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {
            "gt": output_gt_rgb,
            "rgb_medium": outputs["rgb_medium"],
            "rgb_object": outputs["rgb_object"],
            "depth": outputs["depth"],
            "rgb": outputs["rgb"],
            "rgb_clear": outputs["rgb_clear"],
            "rgb_clear_clamp": outputs["rgb_clear_clamp"]}
        return metrics_dict, images_dict
