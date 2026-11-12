#!/usr/bin/env bash
# Jetson 3호 영상 워커(lolog-worker@video)와 서버 터널(lolog-tunnel)을 systemd로 올린다 (#33).
# setup_tts_jetson.sh(모델·목소리)를 먼저 끝낸 뒤, 묶음 폴더에서 한 번 돌린다. 여러 번 돌려도 된다.
#
#   sudo bash setup_video_worker_jetson.sh
#
# 미리 둘 것 (1·2호에서 복사, 비밀값이라 root만 읽게 둔다)
#   /etc/lolog/worker.env    REDIS_URL, API_INTERNAL_URL, INTERNAL_TOKEN, LLM_BASE_URL, LLM_API_KEY, WORKER_NAME
#   /etc/lolog/tunnel.env    SSH_HOST, SSH_PORT, SSH_USER, TUNNEL_REDIS_PORT, TUNNEL_API_PORT
#   /etc/lolog/known_hosts   서버 호스트 키
# 터널 키(/etc/lolog/tunnel_ed25519)는 Jetson마다 따로다(spec/deployment.md 7장). 없으면 만들고 공개키를 보여 준다.
# 그 공개키가 서버 터널 계정에 등록되기 전에는 터널과 워커를 켜지 않는다.
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "sudo로 돌려 주세요: sudo bash $0"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
LOLOG_HOME=/opt/lolog
CODE=$LOLOG_HOME/LOLog-ai
[ -d "$HERE/lolog_ai" ] || { echo "묶음 폴더(lolog_ai가 있는 곳)에서 돌려 주세요"; exit 1; }
[ -x "$LOLOG_HOME/tts-venv/bin/python" ] || { echo "먼저 bash setup_tts_jetson.sh voices 를 끝내 주세요"; exit 1; }
for f in worker.env tunnel.env known_hosts; do
  [ -f /etc/lolog/$f ] || { echo "/etc/lolog/$f 가 없어요. 1·2호에서 복사해 주세요"; exit 1; }
done

command -v autossh >/dev/null || apt-get install -y autossh

# 1·2호와 같은 실행 계정. GPU를 쓰려면 video·render 그룹이 필요하다
id lolog >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/lolog --shell /usr/sbin/nologin lolog
for g in video render; do getent group $g >/dev/null && usermod -aG $g lolog; done

# 3호 이름. 2호 파일을 복사했으면 jetson-2로 되어 있다
sed -i 's/^WORKER_NAME=.*/WORKER_NAME=jetson-3/' /etc/lolog/worker.env
grep -q '^WORKER_NAME=' /etc/lolog/worker.env || echo 'WORKER_NAME=jetson-3' >> /etc/lolog/worker.env
chmod 600 /etc/lolog/worker.env /etc/lolog/tunnel.env
chmod 644 /etc/lolog/known_hosts

# 코드: 묶음의 lolog_ai를 1·2호와 같은 자리(/opt/lolog/LOLog-ai/src)에 둔다
mkdir -p $CODE/src
rm -rf $CODE/src/lolog_ai && cp -r "$HERE/lolog_ai" $CODE/src/lolog_ai
[ -f "$HERE/COMMIT" ] && cp "$HERE/COMMIT" $CODE/COMMIT
chown -R lolog:lolog $LOLOG_HOME

new_key=0
if [ ! -f /etc/lolog/tunnel_ed25519 ]; then
  ssh-keygen -q -t ed25519 -N "" -C "lolog-jetson-3-tunnel" -f /etc/lolog/tunnel_ed25519
  new_key=1
fi
chown lolog:lolog /etc/lolog/tunnel_ed25519 /etc/lolog/tunnel_ed25519.pub
chmod 600 /etc/lolog/tunnel_ed25519

cat > /etc/systemd/system/lolog-tunnel.service <<'EOF'
# 교육기관 서버의 Redis와 메인 백엔드를 젯슨의 localhost:6379, :18080으로 당겨 오는 SSH 터널 (TR-10). 1·2호와 같다
[Unit]
Description=LOLog SSH tunnel to school server (Redis, internal API)
Wants=network-online.target
After=network-online.target

[Service]
User=lolog
EnvironmentFile=/etc/lolog/tunnel.env
Environment=AUTOSSH_GATETIME=0
ExecStart=/usr/bin/autossh -M 0 -N \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/etc/lolog/known_hosts \
  -i /etc/lolog/tunnel_ed25519 -p ${SSH_PORT} \
  -L 127.0.0.1:6379:127.0.0.1:${TUNNEL_REDIS_PORT} \
  -L 127.0.0.1:18080:127.0.0.1:${TUNNEL_API_PORT} \
  ${SSH_USER}@${SSH_HOST}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/lolog-worker@.service <<'EOF'
# 추론 워커. 인스턴스 이름이 역할이다. 1호 detect, 2호 comic, 3호 video. 1·2호와 같다
[Unit]
Description=LOLog worker (%i)
Wants=lolog-tunnel.service
After=network-online.target lolog-tunnel.service lolog-llm.service

[Service]
User=lolog
WorkingDirectory=/opt/lolog/LOLog-ai
EnvironmentFile=/etc/lolog/worker.env
Environment=PYTHONPATH=/opt/lolog/LOLog-ai/src PYTHONUNBUFFERED=1
ExecStart=/opt/lolog/venv/bin/python -m lolog_ai.worker --role %i
Restart=always
RestartSec=10
TimeoutStopSec=900
NoNewPrivileges=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
EOF

# 3호만 다른 것: TTS 전용 venv와 모델·목소리 자리
mkdir -p /etc/systemd/system/lolog-worker@video.service.d
cat > /etc/systemd/system/lolog-worker@video.service.d/tts.conf <<EOF
[Service]
Environment=HF_HOME=$LOLOG_HOME/hf-cache HF_HUB_OFFLINE=1 TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
Environment=TTS_VOICES_DIR=$LOLOG_HOME/voices NUMBA_CACHE_DIR=/tmp
ExecStart=
ExecStart=$LOLOG_HOME/tts-venv/bin/python -m lolog_ai.worker --role video
EOF
systemctl daemon-reload
systemctl enable lolog-tunnel.service lolog-worker@video.service >/dev/null

if [ $new_key -eq 1 ]; then
  echo
  echo "터널 키를 새로 만들었어요. 아래 한 줄을 인프라 담당자에게 주고 서버 터널 계정에 등록해 달라고 하세요:"
  echo
  cat /etc/lolog/tunnel_ed25519.pub
  echo
  echo "등록된 뒤:  sudo systemctl start lolog-tunnel lolog-worker@video"
  exit 0
fi

systemctl restart lolog-tunnel.service
sleep 5
systemctl restart lolog-worker@video.service
echo "켰어요. 확인:"
echo "  systemctl status lolog-tunnel --no-pager | head -5"
echo "  journalctl -u lolog-worker@video -f        # 'TTS 모델 올림' 뒤 큐를 기다리면 정상"
