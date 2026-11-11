#!/usr/bin/env bash
# 젯슨에서 추측 디코딩이 화자 나누기를 얼마나 빠르게 하는지 잰다 (README E·G절).
# 서비스 LLM과 감지 워커를 잠깐 멈추고, 같은 옵션의 llama-server를 8001번 포트에 설정만 바꿔 세 번 띄운다.
#   plain  지금 서비스 옵션 + --cache-ram 0
#   ngram  + n-gram 추측 디코딩(입력에 있던 글을 옮겨 쓸 때 여러 토큰을 한 번에 확정. 메모리 그대로)
#   draft  + 작은 모델(Qwen3-0.6B Q8_0, 약 640MB)이 먼저 쓰고 4B가 확인
# 이야기 구간은 정답을 주고(--gold-detect) 화자 나누기만 돌린다. 끝나면 서비스를 다시 켠다(중간에 멈춰도).
# tmux 안에서 돌릴 것(SSH가 끊겨도 계속 돌고 서비스가 다시 켜지게).
#   bash jetson_spec_bench.sh            # 약 20분
set -u
cd "$(dirname "$0")"
BIN=${BIN:-/opt/lolog/llama.cpp/build/bin/llama-server}
MODEL=${MODEL:-/opt/lolog/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf}
DRAFT=${DRAFT:-$HOME/llm-bench/Qwen3-0.6B-Q8_0.gguf}
PORT=8001
OUT=jetson-spec-$(date +%m%d-%H%M).log
BASE_OPTS="--host 127.0.0.1 --port $PORT -c 24576 -np 1 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --jinja --cache-ram 0"

if [ ! -s "$DRAFT" ]; then
  echo "draft 모델 받는 중: $DRAFT"
  mkdir -p "$(dirname "$DRAFT")"
  curl -L --fail -o "$DRAFT" https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf || { echo "받기 실패"; rm -f "$DRAFT"; }
fi

restore() {
  [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null
  [ -n "${TEG:-}" ] && kill "$TEG" 2>/dev/null
  sudo systemctl start lolog-llm lolog-worker@detect
  echo "서비스 다시 켬: $(systemctl is-active lolog-llm lolog-worker@detect | paste -sd' ')"
}
trap restore EXIT
sudo systemctl stop lolog-worker@detect lolog-llm || exit 1

export LLM_BASE_URL=http://127.0.0.1:$PORT/v1
unset LLM_API_KEY
run() {  # run 이름 "추가 옵션"
  name=$1; extra=$2
  echo "===== $name ($extra)" | tee -a "$OUT"
  # shellcheck disable=SC2086
  $BIN -m "$MODEL" $BASE_OPTS $extra > "llm-$name.log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 90); do curl -s -m 3 127.0.0.1:$PORT/health | grep -q ok && break; sleep 2; done
  if ! curl -s -m 3 127.0.0.1:$PORT/health | grep -q ok; then
    echo "서버가 안 뜸: $(tail -3 "llm-$name.log")" | tee -a "$OUT"; kill $SRV 2>/dev/null; SRV=; return
  fi
  tegrastats --interval 1000 --logfile "tegra-$name.log" & TEG=$!
  t0=$(date +%s)
  python3 run_eval.py --tag "jet-spec-$name" --variant base --gold-detect --runs 1 --limit 2 --sets sets/short-stt 2>&1 | grep -E "short0|^\| short" | tee -a "$OUT"
  python3 run_eval.py --tag "jet-spec-$name" --variant base --gold-detect --runs 1 --limit 1 --sets sets/longstory-stt 2>&1 | grep -E "lstory|^\| longstory" | tee -a "$OUT"
  echo "걸린 시간 $(( $(date +%s) - t0 ))초" | tee -a "$OUT"
  kill $TEG 2>/dev/null; TEG=
  # 출력 속도(토큰/초)와 draft 적중률: 서버 로그에서 합산
  python3 - "llm-$name.log" "tegra-$name.log" <<'PY' | tee -a "$OUT"
import re, sys
log = open(sys.argv[1], errors="ignore").read()
ev = [(float(a), int(b)) for a, b in re.findall(r"\|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens", log)]
pe = [(float(a), int(b)) for a, b in re.findall(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens", log)]
if ev:
    ms, n = sum(a for a, _ in ev), sum(b for _, b in ev)
    print(f"출력 {n}토큰 / {ms / 1000:.0f}초 = {n / (ms / 1000):.1f} 토큰/초 (호출 {len(ev)}번)")
if pe:
    ms, n = sum(a for a, _ in pe), sum(b for _, b in pe)
    print(f"입력 {n}토큰 / {ms / 1000:.0f}초 = {n / (ms / 1000):.0f} 토큰/초")
acc = [(int(a), int(b)) for a, b in re.findall(r"\(\s*(\d+) accepted /\s*(\d+) generated\)", log)]
if acc:
    print(f"draft: 받아들임 {sum(a for a, _ in acc)} / 만든 것 {sum(b for _, b in acc)}")
teg = open(sys.argv[2], errors="ignore").read()
ram = [int(x) for x in re.findall(r"RAM (\d+)/", teg)]
swap = [int(x) for x in re.findall(r"SWAP (\d+)/", teg)]
if ram:
    print(f"RAM 최대 {max(ram)}MB, 스왑 최대 {max(swap) if swap else 0}MB")
PY
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=
  sleep 3
}

run plain ""
run ngram "--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 12"
[ -s "$DRAFT" ] && run draft "-md $DRAFT --spec-type draft-simple --spec-draft-n-max 8"
echo "결과: $(pwd)/$OUT"
