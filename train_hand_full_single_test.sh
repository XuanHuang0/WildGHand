#!/usr/bin/env bash

# Example evaluation-only command for the open-source repository.
CUDA_VISIBLE_DEVICES='0' python infer_hand_train_shade_mano_pose_wild_4d_fenli_loss_mano_lmask_t_reg_sym_sam1_pt_lr4_single.py \
  --config online_shade_mano_pose_wild_time_tex_fenli_nt_tsp10.yaml \
  --config_hand online_shade_mano_pose_wild_time_tex_4d_fenli_loss_mano_sam_res_lmask_magic_t_reg_sym_res1_sam1_vtreg_lr_c0_x_50_tw_tsp10_single.json \
  --run_val
