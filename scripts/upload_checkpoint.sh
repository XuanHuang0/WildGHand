#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <hf-repo-id> <checkpoint-path> [--commit-message MESSAGE]"
  exit 1
fi

HF_REPO_ID="$1"
CKPT_PATH="$2"
shift 2
COMMIT_MESSAGE="Upload checkpoint ${CKPT_PATH}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --commit-message)
      shift
      COMMIT_MESSAGE="$1"
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [ ! -f "${CKPT_PATH}" ]; then
  echo "Checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

python -m huggingface_hub.commands.huggingface_cli login
python -m huggingface_hub.commands.huggingface_cli upload \
  --repo-id "${HF_REPO_ID}" \
  --repo-type model \
  "${CKPT_PATH}"

echo "Uploaded ${CKPT_PATH} to ${HF_REPO_ID}"
