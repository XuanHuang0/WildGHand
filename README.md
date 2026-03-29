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
CUDA_VISIBLE_DEVICES='0' python infer_hand_train_shade_mano_pose_wild_4d_fenli_loss_mano_lmask_t_reg_sym_sam1_pt_lr4_single.py \
  --config online_shade_mano_pose_wild_time_tex_fenli_t_reg_tsp10.yaml \
  --config_hand online_shade_mano_pose_wild_time_tex_4d_fenli_loss_mano_sam_res_lmask_magic_t_reg_sym_res1_sam1_vtreg_lr_c0_x_50_tw_tsp10_single.json
```

Multi-GPU training example:

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python infer_hand_train_shade_mano_pose_wild_4d_fenli_loss_mano_lmask_t_reg_sym_sam1_pt_lr4.py \
  --config online_shade_mano_pose_wild_time_tex_fenli_t_reg_tsp10.yaml \
  --config_hand online_shade_mano_pose_wild_time_tex_4d_fenli_loss_mano_sam_res_lmask_magic_t_reg_sym_res1_sam1_vtreg_lr_c4_x_50_tw_tsp10.json \
  --num_gpus 4
```

Preprocessing example:

```bash
python dataset_identity_mano_sam1_wild_process_4d_.py \
  --input_path dataset/WildGHand/capture10_subsample3/output.pkl
```

## Notes

- The project has substantial GPU memory requirements.
- Dependency versions matter. In particular, older `diffusers` releases may require an older `huggingface_hub` version.
- Top-level training and preprocessing scripts have been reduced to minimal public examples.
- Some deeper research code and historical variants may still contain local experimental assumptions and may need additional cleanup if you plan to fully productize the repository.

## License

License is not set yet in this repository snapshot. Add a `LICENSE` file before publishing a public release.
