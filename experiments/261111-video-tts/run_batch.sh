#!/usr/bin/env bash
# 카드 12개로 영상을 끝까지 만들고 단계별 결과를 batch.jsonl에 남긴다 (#33 할 일 3)
# 도는 동안 코드를 고쳐도 섞이지 않게, 시작할 때 src를 결과 폴더에 복사해 그것으로 돌린다(3차 시드 34 실행이
# 도중에 바뀐 코드로 돌아 버려졌다)
cd /workspace/ai-video
CH=/tmp/claude-1001/-workspace-ai/f41d7271-15e2-428d-a388-267ae3505b78/scratchpad/docs/assets/default-characters
B=${B:-/workspace/video-demo/batch}
# 같은 결과 폴더로 두 번 돌면 GPU를 나눠 쓰다 메모리 부족으로 실패했다(5차 첫 시도). 잠금으로 막는다
exec 9>"$B.lock"; flock -n 9 || { echo "이미 돌고 있어요: $B"; exit 1; }
mkdir -p $B; : > $B/batch.jsonl
rm -rf $B/src && cp -r src $B/src && git rev-parse --short HEAD > $B/COMMIT && git diff --quiet || echo dirty >> $B/COMMIT
export PYTHONPATH=$B/src
for f in ${CARDS:-/workspace/video-demo}/batch_card_*.json; do
  id=$(basename $f .json | sed 's/batch_card_//'); O=$B/$id; rm -rf $O
  t0=$(date +%s)
  /workspace/tts-voices/.venv/bin/python scripts/video_demo.py plan $f $O --voices /workspace/tts-eval/voices-final > $B/$id.plan.log 2>&1; p=$?
  t1=$(date +%s); d=1; r=1
  if [ $p -eq 0 ]; then
    FLUX_NF4_DIR=/workspace/models/flux2-klein-4b-nf4 HF_HUB_OFFLINE=1 .venv/bin/python scripts/video_demo.py draw $O --seed ${SEED:-33} --chars $CH > $B/$id.draw.log 2>&1; d=$?
    FFMPEG=/workspace/bin/ffmpeg .venv/bin/python scripts/video_demo.py render $O > $B/$id.render.log 2>&1; r=$?
  fi
  t2=$(date +%s)
  tries=$(grep -c "콘티 [0-9]번째" $B/$id.plan.log)
  echo "{\"id\":\"$id\",\"plan\":$p,\"draw\":$d,\"render\":$r,\"tries\":$tries,\"plan_s\":$((t1-t0)),\"rest_s\":$((t2-t1))}" >> $B/batch.jsonl
done
echo done > $B/done
