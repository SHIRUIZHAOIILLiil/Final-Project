# Final Project: RGB-D Fusion for Indoor Object Classification

## Project overview

This project studies how RGB and depth information can be combined for indoor visual recognition on the SUN RGB-D dataset. The main goal is to compare RGB-only, Depth-only, early-fusion RGB-D, and mid-fusion RGB-D models, and to examine how fusion strategy, backbone choice, and low-light conditions affect performance.

The implementation focuses on two tasks:

- scene classification on the `kv1/NYUdata` subset of SUN RGB-D as a preliminary baseline;
- object region-of-interest classification as the main task of the project.

The codebase supports experiments with:

- RGB-only, Depth-only, and RGB-D input;
- early fusion and dual-branch mid-fusion;
- ResNet18, ViT, and hybrid ResNet+ViT style models;
- normal-light, low-light robustness, and low-light adaptation settings.

## What the project can do

The project can:

- load scene-level and ROI-level SUN RGB-D data from configuration files;
- generate ROI index files from SUN RGB-D object annotations;
- train single-stream RGB, depth, and early-fusion RGB-D classifiers;
- train configurable dual-branch mid-fusion RGB-D models;
- run controlled low-light experiments by degrading RGB while keeping depth unchanged;
- evaluate saved checkpoints on held-out test data;
- log experiment results for later comparison and analysis.

## Installation

Python dependencies are listed in `requirements.txt`.

Install them with:

```bash
pip install -r requirements.txt
```

## Repository structure

### Root files

- `README.md`: overview of the project and code structure.
- `LICENSE`: licence for the code in this repository.
- `.gitignore`: ignore rules for generated files and local artifacts.
- `test.py`: small inspection script used during development for checking ROI label distributions.

### `configs/`

Configuration files for dataset paths, preprocessing, split settings, model settings, and low-light parameters.

- `dataset_sun_rgb_d.yaml`: scene-classification configuration.
- `dataset_sun_rgbd_object.yaml`: ROI object-classification configuration.

### `datasets/`

Dataset definitions for the main experiment settings.

- `sunrgbd_scene.py`: scene-level dataset loader for RGB, depth, and RGB-D.
- `sunrgbd_object.py`: ROI dataset for single-stream RGB, depth, and early-fusion RGB-D experiments.
- `sunrgbd_object_midfusion.py`: ROI dataset for dual-branch mid-fusion experiments.
- `sunrgbd_object_midfusion_lowlight.py`: low-light mid-fusion ROI dataset with synthetic RGB degradation.
- `__init__.py`: package exports.

### `models/`

Model implementations used in the experiments.

- `resnet_sun.py`: ResNet18 classifier adapted for RGB, depth, and early-fusion RGB-D input.
- `resnet_vit_sun.py`: ViT classifier and hybrid ResNet+Transformer model.
- `resnet_vit_sun_midfusion.py`: dual-branch mid-fusion model and gated mid-fusion variant.
- `__init__.py`: shared model-construction entry point.

### `train/`

Training scripts for each experiment family.

- `train_sun_rgbd_scene.py`: scene-classification training.
- `train_sun_rgbd_object.py`: single-stream ROI object-classification training.
- `train_sun_rgbd_object_lowlight.py`: low-light single-stream ROI training.
- `train_sun_rgbd_object_midfusion.py`: normal-light mid-fusion ROI training.
- `train_sun_rgbd_object_midfusion_lowlight.py`: low-light mid-fusion ROI training.
- `__init__.py`: package initialisation.

### `test/`

Evaluation scripts for trained checkpoints.

- `test_sun.py`: evaluation for scene-classification models.
- `test_sun_object.py`: evaluation for single-stream ROI models.
- `test_sun_object_midfusion.py`: evaluation for mid-fusion ROI models.
- `__init__.py`: package initialisation.

### `utilities/`

Shared helper code used by the training and evaluation pipeline.

- `load_sun_scene.py`: YAML loading, modality input handling, and CSV experiment logging.
- `midfusion_presets.py`: preset configurations for mid-fusion ablations and low-light experiments.
- `utility_sun_objects.py`: ROI index generation from SUN RGB-D annotations with image-level splitting.
- `__init__.py`: package exports.

## How the pipeline works

1. Dataset and model settings are defined in `configs/`.
2. ROI metadata can be generated from SUN RGB-D annotations using `utilities/utility_sun_objects.py`.
3. A training script in `train/` loads the chosen dataset and model.
4. The model is trained and the best checkpoint is saved.
5. An evaluation script in `test/` reloads the saved checkpoint and computes held-out performance.

## Main outcome

The final implementation provides a reusable RGB-D experimental pipeline for comparing modality choice, fusion strategy, backbone design, and low-light robustness on indoor object classification.

## Notes

- The project assumes local access to the SUN RGB-D dataset.
- The dataset paths in the YAML files currently use local Windows paths and must be updated before running the code on another machine.
