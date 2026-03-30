#!/usr/bin/env bash
# Export and release a single model component to GitHub.
#
# Usage:
#   ./export/create_release.sh yolo          # re-release after retraining YOLO
#   ./export/create_release.sh mobilenetv4   # re-release embedding model
#   ./export/create_release.sh embeddings    # re-release after card catalog updates
#   ./export/create_release.sh all           # release all three components
#
# Each component is released independently under its own tag (e.g. yolo-abc1234),
# so retraining one model doesn't require re-releasing the others.
#
# Prerequisites:
#   - venv activated (source venv/bin/activate)
#   - gh CLI authenticated (gh auth login)
#   - YOLO weights present at outputs/detection/train/weights/best.pt (for yolo/embeddings)
#   - Clean working tree (uncommitted changes will abort the script)

set -euo pipefail

COMPONENT="${1:?Usage: $0 <component>  (yolo | mobilenetv4 | embeddings)}"

case "$COMPONENT" in
  yolo|mobilenetv4|embeddings|all) ;;
  *) echo "Error: unknown component '$COMPONENT'. Choose: yolo, mobilenetv4, embeddings, all" >&2; exit 1 ;;
esac

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: uncommitted changes present. Commit or stash them before releasing." >&2
  git status --short >&2
  exit 1
fi

HASH="$(git rev-parse --short HEAD)"
TAG="${COMPONENT}-${HASH}"

echo "==> Releasing $TAG (commit $(git rev-parse HEAD))"

release_component() {
  local component="$1"
  local tag="${component}-${HASH}"
  echo "==> [$component] Creating release $tag..."
  gh release create "$tag" --title "$tag" --generate-notes "${@:2}"
  echo "==> [$component] Done. Release: $(gh release view "$tag" --json url --jq .url)"
}

if [[ "$COMPONENT" == "all" ]]; then
  python export/export_yolo.py
  python export/export_mobilenetv4.py
  python export/export_reference_embeddings.py
  release_component yolo        outputs/export/yolo.onnx outputs/export/yolo.json
  release_component mobilenetv4 outputs/export/mobilenetv4.onnx outputs/export/mobilenetv4.json
  release_component embeddings  outputs/export/reference_embeddings.npz outputs/export/reference_embeddings_meta.json
else
  case "$COMPONENT" in
    yolo)
      python export/export_yolo.py
      release_component yolo outputs/export/yolo.onnx outputs/export/yolo.json
      ;;
    mobilenetv4)
      python export/export_mobilenetv4.py
      release_component mobilenetv4 outputs/export/mobilenetv4.onnx outputs/export/mobilenetv4.json
      ;;
    embeddings)
      python export/export_reference_embeddings.py
      release_component embeddings outputs/export/reference_embeddings.npz outputs/export/reference_embeddings_meta.json
      ;;
  esac
fi
