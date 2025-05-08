from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification
from water_splatting.water_splatting import WaterSplattingModelConfig

from plenodium.plenodium import WaterModelConfig

NUM_STEPS = 15001
plenodium_method_0 = MethodSpecification(
    config=TrainerConfig(
        method_name="plenodium-rec",
        save_only_latest_checkpoint=True,
        steps_per_eval_image=0,
        steps_per_eval_batch=0,
        steps_per_save=1000,
        steps_per_eval_all_images=1000,
        max_num_iterations=15001,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                _target=FullImageDatamanager[DepthDataset],
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=WaterModelConfig(
                num_steps=15001,
                main_loss="reg_l1",
                ssim_loss="reg_ssim",
                ssim_lambda=0.2,
                lpips_lambda=0.00,
                zero_medium=False,
                use_scale_regularization=False,
                medium_sh_degree=3,
                inject_noise_to_position=True,
                use_depth_ranking_loss=True,
                point_complementation=True,
                multi_scale_simm=True,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5,
                    max_steps=15001,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.00025,
                    max_steps=15001,
                ),
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.00025 / 20,
                    max_steps=15001,
                ),
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.05,
                    max_steps=15001,
                ),
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.005,
                    max_steps=15001,
                ),
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.001,
                    max_steps=15001,
                ),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=15001
                ),
            },
            "medium_feature_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.00025,
                    max_steps=15001,
                ),
            },
            "medium_feature_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.00025 / 20,
                    max_steps=15001,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="plenodium for underwater scenes reconstruction.",
)

plenodium_method_1 = MethodSpecification(
    config=TrainerConfig(
        method_name="plenodium-res",
        save_only_latest_checkpoint=True,
        steps_per_eval_image=0,
        steps_per_eval_batch=0,
        steps_per_save=1000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30001,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                _target=FullImageDatamanager[DepthDataset],
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=WaterModelConfig(
                num_steps=30001,
                main_loss="reg_l1",
                ssim_loss="reg_ssim",
                ssim_lambda=0.2,
                lpips_lambda=0.00,
                zero_medium=False,
                use_scale_regularization=False,
                medium_sh_degree=3,
                inject_noise_to_position=True,
                use_depth_ranking_loss=True,
                point_complementation=True,
                multi_scale_simm=False,
                correct_transform=True,
                point_complementation_at=300,
                cull_alpha_thresh=0.2,
                reset_alpha_every=30,
                stop_split_at=15000,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5,
                    max_steps=30001,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.00025,
                    max_steps=30001,
                ),
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.00025 / 20,
                    max_steps=30001,
                ),
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.05,
                    max_steps=30001,
                ),
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.005,
                    max_steps=30001,
                ),
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.001,
                    max_steps=30001,
                ),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=30001
                ),
            },
            "medium_feature_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.00025,
                    max_steps=30001,
                ),
            },
            "medium_feature_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.00025 / 20,
                    max_steps=30001,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="plenodium for underwater scenes restoration.",
)

