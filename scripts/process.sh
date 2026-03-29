#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ "$#" -gt 0 ]; then
  input_paths=("$@")
else
  input_paths=(
    "dataset/WildGHand/capture4_subsample4/output.pkl"
    "dataset/WildGHand/capture5_subsample4/output.pkl"
    "dataset/WildGHand/capture6_subsample5/output.pkl"
    "dataset/WildGHand/capture10_subsample3/output.pkl"
  )
fi

for input_path in "${input_paths[@]}"; do
  python -m wildghand.process_dataset --input_path "$input_path"
done
