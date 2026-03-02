# ScaleFormer

This folder contains the implementation of the paper Cross-Scale Pansharpening via ScaleFormer and the PanScale Datasets (CVPR 2026).


## Environment Setup

Recommended: Python 3.10+ with CUDA-enabled PyTorch.

Install a minimal dependency set:

```bash
pip install torch torchvision numpy pyyaml pillow opencv-python tifffile scipy tqdm tensorboardX einops rotary-embedding-torch matplotlib
```

## Dataset Format

PanScale is publicly available on Hugging Face:

- https://huggingface.co/datasets/kecao/PanScale

Download example:

```bash
pip install -U huggingface_hub
huggingface-cli download kecao/PanScale --repo-type dataset --local-dir ./datasets/hddataset
```

## Configuration

Edit one of the YAML files before running (for example `options/option_jilin.yml`).
All option files are now organized under the `options/` folder:

- `options/option_jilin.yml`
- `options/option_landsat.yml`
- `options/option_skysat.yml`
- `options/option_full_jilin.yml`
- `options/option_full_landsat.yml`
- `options/option_full_skysat.yml`

- `algorithm`: model name (must match a python file under `model/`)
- `checkpoint`: root directory for checkpoints
- `log_dir`: training log directory
- `data_dir_train`: training split root (contains `ms/` and `pan/`)
- `data_dir_eval`: validation/test split root (contains `ms/` and `pan/`)
- `test.data_dir`: dataset used by test scripts
- `test.model`: relative checkpoint path under `checkpoint`
- `test.save_dir`: output directory for fused images

Scale settings used in this project:

- Jilin: `upsacle: 4`
- Landsat: `upsacle: 2`
- SkySat: `upsacle: 2.5`

## Training

Examples:

```bash
python runners/main_jilin.py --option_path options/option_jilin.yml
python runners/main_landsat.py --option_path options/option_landsat.yml
python runners/main_skysat.py --option_path options/option_skysat.yml
```

The training loop saves:

- `latest.pth`
- `bestPSNR.pth`
- `bestSSIM.pth`
- additional best checkpoints for no-reference metrics

under:

```text
<checkpoint>/<algorithm>_<upsacle>_<timestamp>/
```

## Inference (Reduced Resolution)

```bash
python runners/test_jilin.py
python runners/test_landsat.py
python runners/test_skysat.py
```

## Inference (Full Resolution / Large Images)

```bash
python runners/test_full_jilin.py
python runners/test_full_landsat.py
python runners/test_full_skysat.py
```

## Citation

```bibtex
@inproceedings{scaleformer2026,
  title     = {},
  author    = {},
  booktitle = {},
  year      = {}
}
```
