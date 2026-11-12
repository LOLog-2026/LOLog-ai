#!/usr/bin/env bash
# 젯슨 1호에서 이야기 감지 모델을 하나씩 바꿔 올리며 같은 세트로 잰다 (README I절).
# 연구 서버 없이 젯슨만으로 모델을 비교할 때 쓴다.
#   1) jetson_models.tsv 의 모델을 모두 받는다(서비스는 켜 둔 채)
#   2) 서비스 LLM과 감지 워커를 멈추고, 모델마다 8001번 포트에 llama-server를 띄워
#      short-stt(전부) · chat-stt(전부) · long2-stt(앞 LONG_LIMIT개)를 화자 생략으로 1번씩 잰다
#   3) 끝나면(중간에 멈춰도) 서비스를 다시 켜고, 모델별 표를 jetson-models-<시각>.md 로 남긴다
# tmux 안에서 돌릴 것. 4B 한 모델에 약 30~40분, 작은 모델은 더 빠르다.
#   bash jetson_model_sweep.sh                     # tsv 모델 전부
#   ONLY="qwen3.5-4b exaone4-1.2b" bash jetson_model_sweep.sh
#   VARIANT=kc8+verify-s bash jetson_model_sweep.sh    # 채택안(나누기+확인 단계)으로
#   DOWNLOAD_ONLY=1 bash jetson_model_sweep.sh         # 받기만
set -u
cd "$(dirname "$0")"
BIN=${BIN:-/opt/lolog/llama.cpp/build/bin/llama-server}
MODELS_DIR=${MODELS_DIR:-$HOME/llm-bench/models}
LIST=${LIST:-jetson_models.tsv}
VARIANT=${VARIANT:-base}
LONG_LIMIT=${LONG_LIMIT:-2}
ONLY=${ONLY:-}
PORT=8001
STAMP=$(date +%m%d-%H%M)
OUT=jetson-models-$STAMP.md
LOGDIR=logs-models-$STAMP
BASE_OPTS="--host 127.0.0.1 --port $PORT -c 24576 -np 1 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --jinja --cache-ram 0"
mkdir -p "$LOGDIR"

# 0) 평가셋: 없으면 만든다(방송 녹음이 있어야 지금까지와 같은 세트가 나온다)
if [ ! -d sets/short-stt ] || [ ! -d sets/chat-stt ] || [ ! -d sets/long2-stt ]; then
  TV=""
  for p in "$HOME/llm-bench/sample_tv_37min.json" "$HOME/jetson-bench/llm-bench/sample_tv_37min.json"; do
    [ -s "$p" ] && TV=$p && break
  done
  [ -z "$TV" ] && { echo "sample_tv_37min.json 이 없어 평가셋을 못 만듦(~/llm-bench/ 에 두기)"; exit 1; }
  python3 make_evalset.py --out sets --tv "$TV" || exit 1
fi

# 1) 모델 받기
declare -a NAMES PATHS OPTS
while IFS=$'\t' read -r name src globs extra; do
  case "$name" in ''|'#'*) continue ;; esac
  if [ -n "$ONLY" ] && ! grep -qw -- "$name" <<<"$ONLY"; then continue; fi
  echo "== $name 받기"
  if path=$(python3 -I hf_fetch.py "$src" "$globs" "$MODELS_DIR"); then
    NAMES+=("$name"); PATHS+=("$path"); OPTS+=("${extra:--}")
    echo "   $path ($(du -h "$path" | cut -f1))"
  else
    echo "   받기 실패, 건너뜀" | tee -a "$LOGDIR/skipped.txt"
  fi
done < "$LIST"
[ "${DOWNLOAD_ONLY:-0}" = 1 ] && exit 0
[ ${#NAMES[@]} -eq 0 ] && { echo "잴 모델이 없음"; exit 1; }

# 2) 측정
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
{
  echo "# 젯슨 모델 비교 ($STAMP, variant=$VARIANT, 화자 생략, 1번, long2 앞 ${LONG_LIMIT}개)"
  echo
  echo "| 모델 | 파일 | short F1 | short P | short R | short kind | chat fp/h | long2 F1 | long2 R | fmt | 출력 토큰/초 | 입력 토큰/초 | RAM 최대 | 스왑 최대 | 걸린 시간 |"
  echo "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
} > "$OUT"

for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]}; path=${PATHS[$i]}; extra=${OPTS[$i]}
  [ "$extra" = "-" ] && extra=""
  tag="jet-m-$name-$VARIANT"
  echo "===== $name ($(basename "$path")) $extra"
  sync; sleep 2
  # shellcheck disable=SC2086
  $BIN -m "$path" $BASE_OPTS $extra > "$LOGDIR/llm-$name.log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 120); do curl -s -m 3 127.0.0.1:$PORT/health | grep -q ok && break; kill -0 $SRV 2>/dev/null || break; sleep 2; done
  if ! curl -s -m 3 127.0.0.1:$PORT/health | grep -q ok; then
    why=$(grep -iE "error|fail|unknown|unsupported|out of memory" "$LOGDIR/llm-$name.log" | tail -1 | tr '|' '/' | cut -c1-120)
    echo "| $name | $(basename "$path") | 서버가 안 뜸: ${why:-로그 확인} |||||||||||||" >> "$OUT"
    echo "   서버가 안 뜸: $why"
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=; continue
  fi
  tegrastats --interval 1000 --logfile "$LOGDIR/tegra-$name.log" & TEG=$!
  t0=$(date +%s)
  python3 run_eval.py --tag "$tag" --variant "$VARIANT" --skip-speakers --runs 1 --sets sets/short-stt sets/chat-stt 2>&1 | tee "$LOGDIR/eval-$name.log" | grep -E "^\| (short|chat)"
  python3 run_eval.py --tag "$tag-long" --variant "$VARIANT" --skip-speakers --runs 1 --limit "$LONG_LIMIT" --sets sets/long2-stt 2>&1 | tee -a "$LOGDIR/eval-$name.log" | grep -E "^\| long2"
  secs=$(( $(date +%s) - t0 ))
  kill $TEG 2>/dev/null; wait $TEG 2>/dev/null; TEG=
  python3 - "$name" "$path" "$tag" "$LOGDIR" "$secs" <<'PY' | tee -a "$OUT"
import json, os, re, sys
name, path, tag, logdir, secs = sys.argv[1:6]
def summ(t):
    try:
        return json.load(open(f"results/{t}/summary.json"))
    except Exception:
        return {}
s = {**summ(tag), **summ(tag + "-long")}
def g(st, k):
    v = s.get(st, {}).get(k)
    return "-" if not v or v[0] is None else f"{v[0]:.3g}"
def read(p):
    try:
        return open(p, errors="ignore").read()
    except OSError:
        return ""
log = read(f"{logdir}/llm-{name}.log")
ev = [(float(a), int(b)) for a, b in re.findall(r"\|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens", log)]
pe = [(float(a), int(b)) for a, b in re.findall(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens", log)]
rate = lambda xs: f"{sum(b for _, b in xs) / (sum(a for a, _ in xs) / 1000):.1f}" if xs and sum(a for a, _ in xs) else "-"
teg = read(f"{logdir}/tegra-{name}.log")
ram = [int(x) for x in re.findall(r"RAM (\d+)/", teg)]
swap = [int(x) for x in re.findall(r"SWAP (\d+)/", teg)]
fmt = sum((s.get(st, {}).get("fmt") or [0])[0] or 0 for st in s)
gb = os.path.getsize(path) / 1e9
print(f"| {name} | {os.path.basename(path)} ({gb:.1f}GB) | {g('short-stt','F1')} | {g('short-stt','P')} | {g('short-stt','R')} | "
      f"{g('short-stt','kind')} | {g('chat-stt','fp_per_h')} | {g('long2-stt','F1')} | {g('long2-stt','R')} | {fmt:g} | "
      f"{rate(ev)} | {rate(pe)} | {max(ram) if ram else '-'}MB | {max(swap) if swap else '-'}MB | {int(secs) // 60}분 |")
PY
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=
  sleep 3
done
echo
echo "결과: $(pwd)/$OUT  (로그 $LOGDIR/, 녹음별 결과 results/jet-m-*)"
