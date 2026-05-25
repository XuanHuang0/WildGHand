# WildGHand

WildGHand is a research codebase for wild hand reconstruction and training experiments built around MANO, triplane features, and Gaussian-splatting-style rendering components.

This repository contains the code only. Large datasets, checkpoints, and downloaded model weights are intentionally excluded from git.

## Published Repositories

The project is split across GitHub and Hugging Face so that code stays lightweight while large assets are hosted separately.

- GitHub code repository: `https://github.com/XuanHuang0/WildGHand`
- Hugging Face dataset repository: `https://huggingface.co/datasets/XuanHuang0/WildGHand`
- Hugging Face checkpoint repository: `https://huggingface.co/XuanHuang0/WildGHand-checkpoints`

## Repository Scope

This release keeps the core training and preprocessing pipeline, including:

- training entrypoints
- dataset preprocessing scripts
- model code under `tgs/`
- helper scripts and configs

Large artifacts are not tracked in git:

- datasets under `dataset/`
- experiment outputs under `EXPERIMENTS/`
- pretrained checkpoints and weight files such as `*.pth`, `*.pt`, `*.ckpt`, `*.bin`, and `*.safetensors`

## Code and Asset Status

All required code files, configs, and helper scripts are present in the GitHub repository. The large assets needed to reproduce training and inference are hosted on Hugging Face:

- Dataset content: `XuanHuang0/WildGHand` on Hugging Face Datasets
- Checkpoint assets: `XuanHuang0/WildGHand-checkpoints` on Hugging Face Models

If you plan to reproduce experiments, clone the GitHub repo and download the dataset and checkpoint from the Hugging Face repositories.

## Environment

The released commands were checked in the `interhand` conda environment. A local setup should expose the environment in the same style:

```bash
conda env list
```

Expected environment list on the development machine:

```text
# conda environments:
#
base                     /data/huangx/anaconda3
interhand             *  /data/huangx/anaconda3/envs/interhand
```

Activate the environment before running preprocessing, training, or validation:

```bash
conda activate interhand
python -V
python -m pip --version
```

The tested environment uses:

```text
Python 3.9.12
PyTorch 1.11.0
CUDA 11.3
pytorch-lightning 1.6.5
numpy 1.23.5
opencv-python 4.6.0
omegaconf 2.0.6
transformers 4.33.0
huggingface-hub 0.36.2
trimesh 3.12.3
```

Verify that PyTorch can see the GPUs:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

The multi-GPU command expects four visible GPUs. The development machine used for testing exposes four NVIDIA RTX A6000 GPUs.

## Dataset

The WildGHand dataset content is not stored in this git repository.

Use the dataset release instead:

- Hugging Face dataset: `https://huggingface.co/datasets/XuanHuang0/WildGHand`

The dataset has been published to Hugging Face and can be synchronized from the local `dataset/WildGHand` folder.

Use the Hugging Face CLI to upload or download the dataset:

```bash
python -m huggingface_hub.commands.huggingface_cli login
python -m huggingface_hub.commands.huggingface_cli upload --repo-id XuanHuang0/WildGHand --repo-type dataset dataset/WildGHand
```

Expected local path layout:

```text
WildGHand/
  dataset/
    WildGHand/
      capture0_subsample3_single/
      capture1_subsample2_single/
      ...
```

The default configs in this repository point to paths under `dataset/WildGHand/...`.

## Checkpoints And Assets

This repository does not ship the large model files needed for training or inference.
Checkpoint assets are consolidated under `EXPERIMENTS/` in the Hugging Face
model repository so they can be restored with a single download.

Expected local files:

- `vit_h.pth`: SAM ViT-H checkpoint
- `EXPERIMENTS/checkpoints/0.ckpt`: optional pretrained initialization checkpoint
- `EXPERIMENTS/c0_single/ckpts/last.ckpt`: released single-sequence checkpoint
- `EXPERIMENTS/c4_multi/ckpts/last.ckpt`: released multi-sequence checkpoint
- `EXPERIMENTS/<expname>/ckpts/last.ckpt`: latest checkpoint produced by training
- `EXPERIMENTS/<expname>/ckpts/model-XXXX.ckpt`: saved validation checkpoints

The code also supports environment variables when you want to keep assets outside the repo:

- `WILDG_HAND_ASSET_ROOT`: root for auxiliary assets such as `InterHand2.6M`, `processed_dataset`, `mano_uv`, and `change`
- `WILDG_HAND_SAM_CHECKPOINT`: explicit path to the SAM ViT-H checkpoint
- `WILDG_HAND_PRETRAINED_CKPT`: explicit path to the pretrained training checkpoint

Download the published checkpoint bundle into the repository root with:

```bash
python -m huggingface_hub.commands.huggingface_cli download \
  XuanHuang0/WildGHand-checkpoints \
  --repo-type model \
  --local-dir .
```

The released hand configs point to this consolidated layout:

```text
WildGHand/
  EXPERIMENTS/
    checkpoints/
      0.ckpt
    c0_single/
      ckpts/
        last.ckpt
    c4_multi/
      ckpts/
        last.ckpt
```

## Publishing checkpoints to Hugging Face

The recommended workflow is to publish the consolidated `EXPERIMENTS/` folder
to the Hugging Face model repository.

1. Log in and create a model repo (if needed):

```bash
python -m huggingface_hub.commands.huggingface_cli login
python -m huggingface_hub.commands.huggingface_cli repo create <user>/<repo-name> --type model --private
```

2. Upload the checkpoint bundle:

```bash
python -m huggingface_hub.commands.huggingface_cli upload \
  --repo-id <user>/<repo-name> \
  --repo-type model \
  ./EXPERIMENTS \
  EXPERIMENTS
```

3. Or upload with Git + Git LFS:

```bash
git clone https://huggingface.co/<user>/<repo-name>
cd <repo-name>
git lfs install
cp -r ../WildGHand/EXPERIMENTS .
git add EXPERIMENTS
git commit -m "Upload consolidated experiment checkpoints"
git push
```

4. If you publish a model repo, keep a short `README.md` there describing the expected checkpoint layout and inference command.

## Main Entry Points

The helper script supports the four common modes:

```bash
bash scripts/train.sh single_train
bash scripts/train.sh single_val
bash scripts/train.sh multi_train
bash scripts/train.sh multi_val
```

Equivalent raw commands:

Single-sequence training:

```bash
CUDA_VISIBLE_DEVICES='0' python -m wildghand.train_single --config configs/train_treg_tsp10.yaml --config_hand configs/hand_single_c0.json
```

Single-sequence validation:

```bash
CUDA_VISIBLE_DEVICES='0' python -m wildghand.train_single --config configs/eval_nt_tsp10.yaml --config_hand configs/hand_single_c0.json --run_val
```

Multi-GPU training:

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' WILDG_HAND_NUM_GPUS='4' python -m wildghand.train_multi --config configs/train_treg_tsp10.yaml --config_hand configs/hand_multi_c4.json --num_gpus 4
```

Multi-sequence validation:

```bash
CUDA_VISIBLE_DEVICES='0' python -m wildghand.train_multi --config configs/eval_nt_tsp10.yaml --config_hand configs/hand_multi_c4.json --run_val
```

Preprocessing example:

```bash
bash scripts/process.sh dataset/WildGHand/capture10_subsample3/output.pkl
```

## Notes

- The project has substantial GPU memory requirements.
- Dependency versions matter. In particular, older `diffusers` releases may require an older `huggingface_hub` version.
- Public entrypoints now live under `scripts/`, while configs live under `configs/` and core Python modules live under `wildghand/`.
- Some deeper research code and historical variants may still contain local experimental assumptions and may need additional cleanup if you plan to fully productize the repository.

## License

License is not set yet in this repository snapshot. Add a `LICENSE` file before publishing a public release.
