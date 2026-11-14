#!/usr/bin/env bash
# bash run_tag.sh <TAG> <card...> : assets/<card><TAG>에 뽑는다. 장면 그림은 assets/<card>에서 가져와 다시 그리지 않는다
cd "$(dirname "$0")"
TAG=$1; shift
for c in "$@"; do
  if systemctl is-active -q lolog-worker@comic; then echo "워커가 켜져서 멈춤: $c 전" >> chain$TAG.log; exit 1; fi
  mkdir -p assets/$c$TAG; [ -d assets/$c ] && cp -n assets/$c/scene*.png assets/$c$TAG/ 2>/dev/null
  ASSET_TAG=$TAG PYTHONPATH=${PYDEPS:-} LIPSYNC_FACE_MODEL=${LIPSYNC_FACE_MODEL:-$HOME/.cache/lolog/models/mediapipe/face_landmarker.task} \
  ANIMEFACE_MODEL=/opt/lolog/.cache/lolog/models/animeface/model.onnx EXTRA=${EXTRA:-B} \
  FLUX_NF4_DIR=${FLUX_NF4_DIR:-/opt/lolog/models/flux2-klein-4b-nf4} HF_HOME=/opt/lolog/hf-cache HF_HUB_OFFLINE=1 \
  /opt/lolog/venv/bin/python gen_assets.py $c > assets_$c$TAG.log 2>&1
  echo "$c$TAG $(date +%H:%M:%S) $(tail -1 assets_$c$TAG.log | cut -c1-160)" >> chain$TAG.log
done
echo ALLDONE >> chain$TAG.log
