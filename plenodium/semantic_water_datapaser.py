from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParser
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Type
from pathlib import Path
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    Semantics,
)

import torch
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras
from nerfstudio.cameras import camera_utils
import numpy as np
import os
from nerfstudio.utils.io import load_from_json

@dataclass
class SemanticWaterDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: SemanticWaterDataParser)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    downscale_rounding_mode: Literal["floor", "round", "ceil"] = "floor"
    """How to round downscale image height and Image width."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    assume_colmap_world_coordinate_convention: bool = True
    """Colmap optimized world often have y direction of the first camera pointing towards down direction,
    while nerfstudio world set z direction to be up direction for viewer. Therefore, we usually need to apply an extra
    transform when orientation_method=none. This parameter has no effects if orientation_method is set other than none.
    When this parameter is set to False, no extra transform is applied when reading data from colmap.
    """
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "interval"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval (used by most academic papers, e.g. MipNerf360, GSplat).
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""

    images_path: Path = Path("images")
    """Path to images directory relative to the data path."""
    masks_path: Optional[Path] = None
    """Path to masks directory. If not set, masks are not loaded."""
    depths_path: Optional[Path] = None
    """Path to depth maps directory. If not set, depths are not loaded."""
    colmap_path: Path = Path("colmap/sparse/0")
    """Path to the colmap reconstruction directory relative to the data path."""
    load_3D_points: bool = True
    """Whether to load the 3D points from the colmap reconstruction. This is helpful for Gaussian splatting and
    generally unused otherwise, but it's typically harmless so we default to True."""
    max_2D_matches_per_3D_point: int = 0
    """Maximum number of 2D matches per 3D point. If set to -1, all 2D matches are loaded. If set to 0, no 2D matches are loaded."""
    include_semantics: bool = True
    """whether or not to include loading of semantics data""" #! this new for the data parser.
    label_path: Path = Path("label")
    """Path to semantic labels directory relative to the data path."""


class SemanticWaterDataParser(ColmapDataParser):

    config: SemanticWaterDataParserConfig

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):
        assert (
            self.config.data.exists()
        ), f"Data directory {self.config.data} does not exist."
        colmap_path = self.config.data / self.config.colmap_path
        assert colmap_path.exists(), f"Colmap path {colmap_path} does not exist."

        meta = self._get_all_images_and_cameras(colmap_path)
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            fx.append(float(frame["fl_x"]))
            fy.append(float(frame["fl_y"]))
            cx.append(float(frame["cx"]))
            cy.append(float(frame["cy"]))
            height.append(int(frame["h"]))
            width.append(int(frame["w"]))
            if any([k in frame and float(frame[k]) != 0.0 for k in ["k4", "k5", "k6"]]):
                raise ValueError(
                    "K4/K5/K6 is non-zero! Note that Nerfstudio camera model's K4 has different meaning than colmap "
                    "OPENCV camera model K4. Nerfstudio's K4 is the 4-th order of radial distortion coefficient, while "
                    "colmap/OPENCV's K4 is 4-th coefficient in fractional radial distortion model."
                )
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

            image_filenames.append(Path(frame["file_path"]))
            poses.append(frame["transform_matrix"])
            if "mask_path" in frame:
                mask_filenames.append(Path(frame["mask_path"]))
            if "depth_path" in frame:
                depth_filenames.append(Path(frame["depth_path"]))

        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        indices = self._get_image_indices(image_filenames, split)
        image_filenames, mask_filenames, depth_filenames, downscale_factor = (
            self._setup_downscale_factor(
                image_filenames, mask_filenames, depth_filenames
            )
        )

        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = (
            [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        )
        depth_filenames = (
            [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        )

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-aabb_scale, -aabb_scale, -aabb_scale],
                    [aabb_scale, aabb_scale, aabb_scale],
                ],
                dtype=torch.float32,
            )
        )

        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        cameras.rescale_output_resolution(
            scaling_factor=1.0 / downscale_factor,
            scale_rounding_mode=self.config.downscale_rounding_mode,
        )

        if "applied_transform" in meta:
            applied_transform = torch.tensor(
                meta["applied_transform"], dtype=transform_matrix.dtype
            )
            transform_matrix = transform_matrix @ torch.cat(
                [
                    applied_transform,
                    torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype),
                ],
                0,
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        metadata = {}
        if self.config.load_3D_points:
            # Load 3D points
            metadata.update(
                self._load_3D_points(colmap_path, transform_matrix, scale_factor)
            )

        #! --- semantics ---
        semantics = None
        if self.config.include_semantics:
            label_path = self.config.data / self.config.label_path
            filenames = [
                Path(str(label_path / os.path.basename(str(image_filename)))) for image_filename in image_filenames
            ]
            # ============================================================================================#
            # TODO: modify
            panoptic_classes = load_from_json(label_path / "classes.json")
            classes = panoptic_classes["thing"]
            colors = (
                torch.tensor(panoptic_classes["thing_colors"], dtype=torch.float32)
                / 255.0
            )
            # ============================================================================================#
            semantics = Semantics(
                filenames=filenames,
                classes=classes,
                colors=colors,
                mask_classes=["fish"],
            )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": (
                    depth_filenames if len(depth_filenames) > 0 else None
                ),
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "semantics": semantics,
                **metadata,
            },
        )
        return dataparser_outputs
