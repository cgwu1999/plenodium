<p align="center">
    <h1>
        <span class="title-main"><span>Plenodium</span></span>
        <span class="title-small">UnderWater 3D Scene Reconstruction with Plenoptic Medium Representation</span>
      </h1>
<p align="justify">


## Installation

Our method is based on [nerfstudio](https://docs.nerf.studio/index.html).


```bash
# Install PyTorch
pip uninstall torch torchvision functorch tinycudann
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install tiny-cuda-nn
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install our modified nerfstudio
cd nerfstudio-v1.5
pip install -e .
ns-install-cli
cd ..

# WaterSplatting
cd plenodium
pip install --no-use-pep517 -e .
```

## Training
To start the training on the [SeaThru-NeRF](https://sea-thru-nerf.github.io/) dataset, run the following commands:

```bash
ns-train plenodium-rec  --vis viewer+wandb colmap --downscale-factor 1 --colmap-path sparse/0 --data /your_path_to_dataset/ --images-path Images_wb --depths_path depths 
```
To start the training on our [Simulated](https://figshare.com/s/78307bef0248d4577880) dataset, run the following commands:
```bash
ns-train plenodium-res  --vis viewer+wandb colmap --downscale-factor 1 --data /your_path_to_dataset/ --colmap-path sparse --images-path train --depths_path depths --eval-mode interval --eval_interval 2
```

## Interactive viewer
To start the viewer and explore the trained models, run one of the following:
```bash
ns-viewer --load-config outputs/unnamed/plenodium/your_timestamp/config.yml
```

## Rendering videos
To render a video on a trajectory (e.g., generated from the interactive viewer), run:
```bash
ns-render camera-path --load-config outputs/unnamed/plenodium/your_timestamp/config.yml --camera-path-filename /your_path_to_dataset/SeathruNeRF_dataset/IUI3-RedSea/camera_paths/your_trajectory.json --output-path renders/IUI3-RedSea/water_splatting.mp4
```

## Rendering dataset
To render testing set for a checkpoint, run:
```bash
ns-render dataset --load-config outputs/unnamed/plenodium/your_timestamp/config.yml --data /your_path_to_dataset/
```