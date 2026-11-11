#!/usr/bin/env bash
# 카드 12개로 영상을 끝까지 만들고 단계별 결과를 batch.jsonl에 남긴다 (#33 할 일 3)
cd /workspace/ai-video
CH=/tmp/claude-1001/-workspace-ai/f41d7271-15e2-428d-a388-267ae3505b78/scratchpad/docs/assets/default-characters
B=${B:-/workspace/video-demo/batch}; mkdir -p $B; : > $B/batch.jsonl
for f in /workspace/video-demo/batch_card_*.json; do
  id=$(basename $f .json | sed 's/batch_card_//'); O=$B/$id; rm -rf $O
  t0=$(date +%s)
  PYTHONPATH=src /workspace/tts-voices/.venv/bin/python scripts/video_demo.py plan $f $O --voices /workspace/tts-eval/voices-final > $B/$id.plan.log 2>&1; p=$?
  t1=$(date +%s); d=1; r=1
  if [ $p -eq 0 ]; then
    FLUX_NF4_DIR=/workspace/models/flux2-klein-4b-nf4 HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/video_demo.py draw $O --seed 33 --chars $CH > $B/$id.draw.log 2>&1; d=$?
    PYTHONPATH=src FFMPEG=/workspace/bin/ffmpeg .venv/bin/python scripts/video_demo.py render $O > $B/$id.render.log 2>&1; r=$?
  fi
  t2=$(date +%s)
  tries=$(grep -c "콘티 [0-9]번째" $B/$id.plan.log)
  echo "{\"id\":\"$id\",\"plan\":$p,\"draw\":$d,\"render\":$r,\"tries\":$tries,\"plan_s\":$((t1-t0)),\"rest_s\":$((t2-t1))}" >> $B/batch.jsonl
done
echo done > $B/done
