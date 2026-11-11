#!/usr/bin/env bash
# Jetson 3호 TTS 환경 만들기 (#33). 여러 번 돌려도 된다.
#
#   bash setup_tts_jetson.sh VOICES            # VOICES: 인사 클립 <id>.wav 24개가 든 폴더
#
# 만드는 것 (LOLOG_HOME 기본 /opt/lolog, 1·2호와 같은 자리)
#   $LOLOG_HOME/tts-venv     qwen-tts 전용 venv. 워커 venv와 transformers 버전이 달라 따로 둔다
#   $LOLOG_HOME/voices       인사 클립
#   $LOLOG_HOME/hf-cache     TTS 모델(약 2.4GB). 받은 뒤에는 HF_HUB_OFFLINE=1로 쓴다. HF_HOME을 주면 그 자리를 쓴다
# 끝에 한 줄을 읽어 $LOLOG_HOME/tts-hello.wav로 남긴다. "[ok]" 줄이 나오면 준비가 끝난 것이다.
#
# PyPI가 막힌 망에서는: UV_INDEX_URL=https://mirror.kakao.com/pypi/simple bash setup_tts_jetson.sh VOICES
set -euo pipefail

VOICES_SRC=${1:?"인사 클립 폴더를 주세요: bash setup_tts_jetson.sh VOICES"}
LOLOG_HOME=${LOLOG_HOME:-/opt/lolog}
MODEL=${TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}
HERE=$(cd "$(dirname "$0")" && pwd)
# tts.py: 묶음으로 받았으면 이 파일 옆, 레포에서 돌리면 src/lolog_ai
TTS_PY=$HERE/tts.py; [ -f "$TTS_PY" ] || TTS_PY=$HERE/../src/lolog_ai/tts.py
[ -f "$TTS_PY" ] || { echo "tts.py를 찾지 못했어요"; exit 1; }

n=$(find "$VOICES_SRC" -maxdepth 1 -name '*.wav' | wc -l)
[ "$n" -ge 24 ] || { echo "인사 클립이 ${n}개예요(24개 필요): $VOICES_SRC"; exit 1; }

if ! mkdir -p "$LOLOG_HOME" 2>/dev/null || [ ! -w "$LOLOG_HOME" ]; then
  echo "$LOLOG_HOME 에 쓸 수 없어요. 먼저 아래를 실행한 뒤 다시 돌려 주세요:"
  echo "  sudo mkdir -p $LOLOG_HOME && sudo chown $(id -un):$(id -gn) $LOLOG_HOME"
  exit 1
fi

if ! command -v uv >/dev/null; then
  [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

VENV=$LOLOG_HOME/tts-venv
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
uv pip install -p "$VENV" qwen-tts soundfile

mkdir -p "$LOLOG_HOME/voices" "$LOLOG_HOME/hf-cache"
cp "$VOICES_SRC"/*.wav "$LOLOG_HOME/voices/"
cp "$TTS_PY" "$LOLOG_HOME/tts.py"

export HF_HOME=${HF_HOME:-$LOLOG_HOME/hf-cache} TTS_MODEL=$MODEL
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'GPU가 안 잡혀요 (torch.cuda.is_available() False)'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
"$VENV/bin/python" -c "from huggingface_hub import snapshot_download as d; d('$MODEL')" >/dev/null
HF_HUB_OFFLINE=1 "$VENV/bin/python" -I "$LOLOG_HOME/tts.py" "$LOLOG_HOME/voices" --out "$LOLOG_HOME/tts-hello.wav" 2>&1 | grep -E '^\[ok\]|Error|error' \
  || { echo "읽기 확인에 실패했어요. 아래로 전체 출력을 보세요:"; echo "  HF_HOME=$HF_HOME $VENV/bin/python -I $LOLOG_HOME/tts.py $LOLOG_HOME/voices"; exit 1; }

echo "준비 끝. 워커 환경 변수: HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 TTS_MODEL=$MODEL TTS_VOICES_DIR=$LOLOG_HOME/voices"
