# DeRF-pytorch

This repository is an unofficial blind reproduction attempt of [DeRF: Decomposed Radiance Fields](https://arxiv.org/abs/2011.12490), appearing at CVPR2021.
We make no guarantees on accuracy - this effort is for educational and exploratory purposes only.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

Ensure settings in `config/__init__.py` are configured to the desired dataset.
Config options include:

```configs
DEVICE -> cpu, cuda
DATASET_NAME -> blender (for synthetic), llff (for real)
DATASET_TYPE -> lego, drums (blender examples), fern, flower (llff examples)
TRAINING_ACCELERATION -> 1 or 2 (use only if you want quick, cheap results)
TEST_ALL_EPOCHS -> True, or False (will only test the checkpoint for last found epoch)
HEAD_COUNT -> any integer (number of heads for Voronoi partitioning)
VORONOI_INIT_SCHEME -> 'uniform', 'stratified_uniform', 'deterministic_grid' (view volume init)
```

Before training the model, data must be properly packaged for the model to use.
First, ensure the [official NeRF data](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) is in the `data` directory, following the format:

```data
data/nerf_synthetic/lego
```

for the Blender (synthetic) lego dataset, for example.
Then, with the matching configurations set, run this command:

```pickle
python pickler.py
```

This will convert the dataset to ray origins, directions, ground truth pixel colors, and the near and far planes of each dataset, which is all the networks need.

To train the NeRF model, run this command:

```train_nerf
python nerf_train.py
```

Likewise, to train the DeRF model, run this command:

```train_derf
python derf_train.py
```

## Testing

To test the NeRF model, run this command:

```eval
python nerf_test.py
```

Likewise, to test the DeRF model, run this command:

```eval
python derf_test.py
```

Results located in the `results` directory will contain PNGs of rendered images, a GIF of the combined images, RMSE, PSNR, SSIM, and LPIPS of each image, the means of each across all images, and if testing on multiple epochs, means across epochs.

## Pre-trained Models

This repository includes pre-trained models for NeRF and DeRF in the `checkpoints` directory. Most should work without much code modification.

In order to use a pre-trained model, load the state dict in the `training_loop()` of the respective model's `x_test.py` file. This will have to be done in code.

## Results

Refer to associated report for results, otherwise feel free to generate your own to confirm.
All results in our report were performed benchmarking on 1 NVIDIA GeForce GTX 1080 Ti.
