# WildGHand

WildGHand is a research codebase for wild hand reconstruction and training experiments built around MANO, triplane features, and Gaussian-splatting-style rendering components.

This repository is being cleaned for open-source release. The code is included here, while large datasets, checkpoints, and downloaded model weights are intentionally excluded from git.

## Repository Scope

This code release keeps the core training and preprocessing pipeline under `/home/huangx/WildGHand`, including:

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

## Notes

- Several configs and scripts still contain local absolute-path assumptions and should be cleaned further before a polished public release.
- The project has substantial GPU memory requirements.
- Dependency versions matter. In particular, older `diffusers` releases may require an older `huggingface_hub` version.

## License

License is not set yet in this repository snapshot. Add a `LICENSE` file before publishing a public release.
