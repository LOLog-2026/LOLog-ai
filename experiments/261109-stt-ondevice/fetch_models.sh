#!/usr/bin/env bash
# sherpa-onnx 모델을 models/ 에 받는다 (github.com 접속 필요). 이미 있으면 건너뛴다.
# 쓰는 법: bash fetch_models.sh [모델폴더이름 ...]   (안 주면 기본 세트)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
DEFAULT=(
  sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17
  sherpa-onnx-zipformer-korean-2024-06-24
  sherpa-onnx-whisper-small
)
names=("${@:-${DEFAULT[@]}}")
[ -f models/silero_vad.onnx ] || curl -fL -o models/silero_vad.onnx "$BASE/silero_vad.onnx"
for n in "${names[@]}"; do
  [ -d "models/$n" ] && { echo "있음: $n"; continue; }
  echo "받는 중: $n"
  curl -fL "$BASE/$n.tar.bz2" | tar xj -C models
done
ls models
