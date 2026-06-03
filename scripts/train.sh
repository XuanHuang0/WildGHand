# Single train:
CUDA_VISIBLE_DEVICES='0' python -m wildghand.train_single --config configs/train_treg_tsp10.yaml --config_hand configs/hand_single_c0.json

# Single validation:
# CUDA_VISIBLE_DEVICES='0' python -m wildghand.train_single --config configs/eval_nt_tsp10.yaml --config_hand configs/hand_single_c0.json --run_val

# Multi-train:
# CUDA_VISIBLE_DEVICES='0,1,2,3' WILDG_HAND_NUM_GPUS='4' python -m wildghand.train_multi --config configs/train_treg_tsp10.yaml --config_hand configs/hand_multi_c4.json --num_gpus 4

# Multi validation:
# CUDA_VISIBLE_DEVICES='0' python -m wildghand.train_multi --config configs/eval_nt_tsp10.yaml --config_hand configs/hand_multi_c4.json --run_val
