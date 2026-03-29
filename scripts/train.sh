#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

mode="${1:-single_train}"
shift || true

case "$mode" in
  single_train)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      python -m wildghand.train_single \
      --config configs/train_treg_tsp10.yaml \
      --config_hand configs/hand_single_c0.json \
      "$@"
    ;;
  single_val)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      python -m wildghand.train_single \
      --config configs/eval_nt_tsp10.yaml \
      --config_hand configs/hand_single_c0.json \
      --run_val \
      "$@"
    ;;
  multi_train)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
      python -m wildghand.train_multi \
      --config configs/train_treg_tsp10.yaml \
      --config_hand configs/hand_multi_c4.json \
      --num_gpus "${WILDG_HAND_NUM_GPUS:-4}" \
      "$@"
    ;;
  multi_val)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      python -m wildghand.train_multi \
      --config configs/eval_nt_tsp10.yaml \
      --config_hand configs/hand_multi_c4.json \
      --run_val \
      "$@"
    ;;
  *)
    echo "Usage: bash scripts/train.sh {single_train|single_val|multi_train|multi_val} [extra args]" >&2
    exit 1
    ;;
esac
