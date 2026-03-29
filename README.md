# WildGHand

WildGHand is a research codebase for wild hand reconstruction and training experiments built around MANO, triplane features, and Gaussian-splatting-style rendering components.

This repository contains the code only. Large datasets, checkpoints, and downloaded model weights are intentionally excluded from git.

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

## Dataset

The WildGHand dataset content is not stored in this git repository.

Use the dataset release instead:

- Hugging Face dataset: `https://huggingface.co/datasets/XuanHuang0/WildGHand`

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

This repository does not ship the large model files needed for training.

Expected local files:

- `vit_h.pth`: SAM ViT-H checkpoint
- `checkpoints/0.ckpt`: optional pretrained initialization checkpoint

The code also supports environment variables when you want to keep assets outside the repo:

- `WILDG_HAND_ASSET_ROOT`: root for auxiliary assets such as `InterHand2.6M`, `processed_dataset`, `mano_uv`, and `change`
- `WILDG_HAND_SAM_CHECKPOINT`: explicit path to the SAM ViT-H checkpoint
- `WILDG_HAND_PRETRAINED_CKPT`: explicit path to the pretrained training checkpoint

## Main Entry Points

Single-sequence training example:

```bash
bash scripts/train.sh single_train
```

Multi-GPU training example:

```bash
bash scripts/train.sh multi_train
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
