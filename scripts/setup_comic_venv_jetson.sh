#!/usr/bin/env bash
# Jetson 2호 comic 워커의 패키지를 실험용 venv에서 워커 전용 venv로 옮긴다 (#69).
#
# 2026-11-24에 2호에서 돌려 옮겼다. 되돌리기(rollback)와 기록용으로 둔다.
#
# 옮기기 전 2호는 /opt/lolog/venv가 비어 있고 flux-venv.pth 한 줄로 /home/minju/LOLog-ai/.venv를 봤다.
# 그 폴더에서 uv sync를 돌리면 워커가 쓰는 패키지가 지워진다. 그래서 패키지를 /opt/lolog/venv로 복사한다.
# 새로 내려받지 않고 지금 도는 파일을 그대로 복사하므로 버전·빌드가 바뀌지 않는다(두 venv 모두 /usr/bin/python3.12).
#
#   bash setup_comic_venv_jetson.sh check          # 읽기만 한다. sudo 없이 돈다
#   sudo bash setup_comic_venv_jetson.sh           # 옮긴다. 여러 번 돌려도 된다
#   sudo bash setup_comic_venv_jetson.sh rollback  # 되돌린다
#
# 순서: 복사(워커는 도는 채로, 약 6GB) → 워커 멈춤(하던 작업은 끝내고 멈춘다) → 폴더 바꿔치기
#       → import·CUDA 확인 → 워커 켬. 확인에 실패하면 원래대로 되돌리고 워커를 켠다.
# 워커가 멈춘 동안 들어온 작업은 Redis에 남아 있다가 다시 켜지면 처리된다.
# 옮기지 않는 것: /opt/lolog/pydeps-lipsync(mediapipe·opencv), 실험 venv의 bin/ 스크립트(워커가 안 쓴다).
# /home/minju/LOLog-ai/.venv는 지우지 않는다.
set -euo pipefail

SRC=${SRC:-/home/minju/LOLog-ai/.venv/lib/python3.12/site-packages}
VENV=/opt/lolog/venv
SITE=$VENV/lib/python3.12/site-packages
NEW=$SITE.new       # 복사해 두는 곳
OLD=$SITE.pth-only  # 바꾸기 전 폴더(flux-venv.pth만 든 것). rollback이 쓴다
UNIT=lolog-worker@comic.service
NEED_GB=8
MODE=${1:-run}

state() {  # empty-pth: 아직 안 옮김 / moved: 옮김 / unknown: 둘 다 아님
  if [ -f "$SITE/flux-venv.pth" ] && [ ! -d "$SITE/torch" ]; then echo empty-pth
  elif [ ! -f "$SITE/flux-venv.pth" ] && [ -d "$SITE/torch" ]; then echo moved
  else echo unknown; fi
}

worker_pythonpath() {  # 서비스에 걸린 PYTHONPATH를 그대로 쓴다(lipsync.conf가 pydeps-lipsync를 넣는다)
  systemctl show -p Environment --value "$UNIT" | tr ' ' '\n' | sed -n 's/^PYTHONPATH=//p' | tail -1
}

smoke() {  # 워커와 같은 사용자·파이썬·PYTHONPATH로 import와 CUDA를 본다. 모델은 올리지 않는다
  local pp; pp=$(worker_pythonpath)
  [ -n "$pp" ] || { echo "서비스에서 PYTHONPATH를 못 읽었어요"; return 1; }
  runuser -u lolog -- env PYTHONPATH="$pp" HF_HUB_OFFLINE=1 "$VENV/bin/python" - <<'PY'
import importlib, sys

bad = [p for p in sys.path if p.startswith("/home/")]
if bad:
    sys.exit(f"sys.path에 홈 폴더가 남아 있어요: {bad}")
names = ["redis", "numpy", "PIL", "torch", "transformers", "diffusers", "bitsandbytes", "accelerate", "peft",
         "onnxruntime", "cv2", "mediapipe", "lolog_ai.worker.draw", "lolog_ai.lipsync", "lolog_ai.faces"]
for n in names:
    m = importlib.import_module(n)
    f = getattr(m, "__file__", "") or ""
    if not f.startswith("/opt/lolog/"):
        sys.exit(f"{n}이 /opt/lolog 밖에서 잡혀요: {f}")
    print(f"  {n:22s} {getattr(m, '__version__', '')}")
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel  # noqa: F401  worker/draw.py가 쓰는 것
import torch

if not torch.cuda.is_available():
    sys.exit("torch가 CUDA를 못 잡아요")
print("  CUDA", torch.cuda.get_device_name(0))
PY
}

start_worker() {
  systemctl start "$UNIT"
  sleep 8
  if systemctl is-active -q "$UNIT"; then
    echo "워커가 켜졌어요"
    journalctl -u "$UNIT" --no-pager -n 5 -o cat | cut -c1-200 | sed 's/^/  /'
  else
    echo "워커가 안 켜져요. journalctl -u $UNIT -n 50 으로 확인해 주세요"; return 1
  fi
}

swap_back() {  # 옮긴 폴더를 치우고 flux-venv.pth만 든 폴더를 되돌린다
  [ -d "$OLD" ] || { echo "$OLD 가 없어서 되돌릴 수 없어요"; return 1; }
  rm -rf "$NEW"
  mv "$SITE" "$NEW"
  mv "$OLD" "$SITE"
}

# ---------- check ----------
if [ "$MODE" = check ]; then
  echo "상태: $(state)"
  echo "원본: $SRC ($(du -sh "$SRC" 2>/dev/null | cut -f1), 항목 $(ls "$SRC" | wc -l)개)"
  echo "원본 파이썬: $(sed -n 's/^home = //p' "$SRC/../../../pyvenv.cfg") / 워커 파이썬: $(sed -n 's/^home = //p' "$VENV/pyvenv.cfg")"
  echo "빈 자리: $(df -BG --output=avail "$VENV" | tail -1 | tr -d ' ') (필요 ${NEED_GB}G)"
  echo "워커: $(systemctl is-active "$UNIT")"
  echo "PYTHONPATH: $(worker_pythonpath)"
  [ -d "$NEW" ] && echo "주의: 앞서 복사하다 만 폴더가 있어요($NEW). 다시 돌리면 지우고 새로 복사해요"
  [ -d "$OLD" ] && echo "되돌릴 폴더가 있어요($OLD)"
  exit 0
fi

[ "$(id -u)" -eq 0 ] || { echo "sudo로 돌려 주세요: sudo bash $0 $*"; exit 1; }

# ---------- rollback ----------
if [ "$MODE" = rollback ]; then
  [ "$(state)" = moved ] || { echo "옮긴 상태가 아니에요(상태: $(state)). 되돌릴 것이 없어요"; exit 1; }
  [ -d "$SRC/torch" ] || { echo "원본($SRC)에 torch가 없어요. 되돌리면 워커가 못 켜져요. 그만둘게요"; exit 1; }
  [ -d "$OLD" ] || { echo "$OLD 가 없어서 되돌릴 수 없어요. 워커는 그대로예요"; exit 1; }
  echo "워커를 멈춰요(하던 작업이 있으면 끝날 때까지 기다려요)"
  systemctl stop "$UNIT"
  swap_back
  echo "되돌렸어요. 복사해 둔 것은 $NEW 에 남겨 뒀어요"
  start_worker
  exit 0
fi

[ "$MODE" = run ] || { echo "쓰는 법: bash $0 [check|rollback]"; exit 1; }

# ---------- run ----------
case "$(state)" in
  moved) echo "이미 옮겼어요. 할 일이 없어요"; exit 0 ;;
  unknown) echo "$SITE 가 예상한 모양이 아니에요(flux-venv.pth와 torch 폴더를 봐 주세요). 그만둘게요"; exit 1 ;;
esac
[ -d "$SRC/torch" ] || { echo "원본($SRC)에 torch가 없어요"; exit 1; }
[ "$(sed -n 's/^home = //p' "$SRC/../../../pyvenv.cfg")" = "$(sed -n 's/^home = //p' "$VENV/pyvenv.cfg")" ] \
  || { echo "두 venv의 파이썬이 달라요. 복사해서 쓸 수 없어요"; exit 1; }
avail=$(df -BG --output=avail "$VENV" | tail -1 | tr -dc '0-9')
[ "$avail" -ge "$NEED_GB" ] || { echo "빈 자리가 ${avail}G뿐이에요(${NEED_GB}G 필요)"; exit 1; }
[ ! -e "$OLD" ] || { echo "$OLD 가 이미 있어요. 앞선 실행의 흔적이니 확인하고 지운 뒤 다시 돌려 주세요"; exit 1; }

echo "[1/5] 복사해요: $SRC → $NEW (약 6GB, 몇 분 걸려요. 워커는 도는 채로 둬요)"
rm -rf "$NEW"
mkdir "$NEW"
cp -a "$SRC/." "$NEW/"
# 실험용 체크아웃을 가리키는 editable 설치는 뺀다. 워커 코드는 PYTHONPATH(/opt/lolog/LOLog-ai/src)로 잡는다
rm -rf "$NEW"/lolog_ai.pth "$NEW"/lolog_ai-*.dist-info
chown -R lolog:lolog "$NEW"
want=$(find "$SRC" -type f ! -name lolog_ai.pth ! -path '*/lolog_ai-*.dist-info/*' | wc -l)
got=$(find "$NEW" -type f | wc -l)
[ "$want" -eq "$got" ] || { echo "파일 수가 달라요(원본 $want, 복사본 $got). 그만둘게요. 워커는 그대로예요"; exit 1; }
echo "      파일 $got개"

echo "[2/5] 워커를 멈춰요(하던 작업이 있으면 끝날 때까지, 길면 15분 기다려요)"
systemctl stop "$UNIT"

echo "[3/5] 폴더를 바꿔요"
mv "$SITE" "$OLD"
if ! mv "$NEW" "$SITE"; then
  mv "$OLD" "$SITE"
  start_worker || true
  echo "폴더를 못 바꿨어요. 원래대로 두고 워커를 켰어요"
  exit 1
fi

echo "[4/5] 워커와 같은 환경으로 import·CUDA를 확인해요"
if ! smoke; then
  echo "확인에 실패했어요. 원래대로 되돌려요"
  swap_back
  start_worker || true
  echo "되돌렸어요. 복사본은 $NEW 에 남아 있어요"
  exit 1
fi

echo "[5/5] 워커를 켜요"
if ! start_worker; then
  echo "워커가 안 켜져서 원래대로 되돌려요"
  systemctl stop "$UNIT" || true
  swap_back
  start_worker || true
  echo "되돌렸어요. 복사본은 $NEW 에 남아 있어요"
  exit 1
fi
echo
echo "끝났어요. 워커는 이제 $SITE 만 봐요."
echo "  되돌리기: sudo bash $0 rollback"
echo "  며칠 써 보고 괜찮으면 $OLD 는 지워도 돼요. /home/minju/LOLog-ai/.venv는 건드리지 않았어요."
