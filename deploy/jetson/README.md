# 젯슨 설치

명령에서 `<...>`는 실제 값으로 바꿔 넣는다(그대로 치면 bash 문법 오류).
이 폴더가 아직 레포에 없으면 맥의 `jetson-bench/jetson-infra`를 `scp -r`로 젯슨에 복사한 뒤
`sudo cp -r deploy src /opt/lolog/LOLog-ai/ && sudo chown -R lolog:lolog /opt/lolog/LOLog-ai`로 넣는다.

젯슨 두 대를 역할로 나눈다. 모델을 번갈아 올리지 않는다(TR-08 대신).

| | 1호 `jetson-1` | 2호 `jetson-2` |
|---|---|---|
| 모델 | LLM (Qwen3-4B-Instruct-2507 Q4_K_M, llama.cpp) | FLUX.2 klein 4B nf4 + LoRA |
| 워커 | `lolog-worker@detect` (스트림 `lolog:detect`) | `lolog-worker@comic` (스트림 `lolog:comic`) |
| 서비스 | `lolog-tunnel`, `lolog-llm`, `lolog-worker@detect` | `lolog-tunnel`, `lolog-worker@comic` |
| LLM 부르는 곳 | 자기 LAN 주소 `:8000` | 1호 LAN 주소 `:8000` (콘티 만들 때) |

```
교육기관 서버 ──SSH 터널(각 젯슨이 붙음)── 1호: Redis·/internal → localhost:6379, :18080
  127.0.0.1:<TUNNEL_REDIS_PORT> Redis       2호: 〃
  127.0.0.1:<TUNNEL_API_PORT>   lolog-api
                                            1호 llama-server :8000 ← 1호 감지 워커, 2호 만화 워커 (LAN, API 키)
```

두 젯슨이 같은 컨슈머 그룹 `jetson-workers`를 쓰지만 각자 자기 스트림만 읽으므로 백엔드는 바꿀 것이 없다.

## 0. 공통 준비 (두 대 모두)

```bash
# 메모리 확보: 데스크톱 끄기(약 1GB). 되돌릴 때는 graphical.target
sudo systemctl set-default multi-user.target
# 전원 모드: FLUX 측정과 같은 15W + 최대 클럭 (모드 번호는 sudo nvpmodel -q --verbose 로 확인)
sudo nvpmodel -m 0 && sudo jetson_clocks
# 스왑 8GB (/swapfile)는 측정 때 만든 것을 그대로 둔다. 없으면 만들고 /etc/fstab에 추가

sudo apt install -y autossh python3-venv nano
sudo useradd --system --create-home --home-dir /opt/lolog --shell /usr/sbin/nologin lolog
sudo usermod -aG video,render lolog   # GPU 권한. 없으면 서비스가 CUDA를 못 찾고 CPU로 돎
sudo chmod 755 /opt/lolog   # 코드·모델만 있는 곳이라 읽기는 열어 둠(비밀값은 /etc/lolog)
sudo mkdir -p /etc/lolog && sudo chown root:lolog /etc/lolog && sudo chmod 750 /etc/lolog

# 코드
sudo -u lolog git clone https://github.com/LOLog-2026/LOLog-ai /opt/lolog/LOLog-ai
```

파이썬 환경
- 1호: `sudo -u lolog python3 -m venv /opt/lolog/venv && sudo -u lolog /opt/lolog/venv/bin/pip install "redis>=5"`
- 2호: FLUX 측정 venv(torch, diffusers)를 같이 쓰는 lolog 전용 venv를 만든다. 측정 venv를 그대로 링크하면
  거기 개발 모드로 설치된 예전 `lolog_ai`가 `PYTHONPATH`보다 먼저 잡혀 `No module named 'lolog_ai.worker'`가 난다.
  ```bash
  ~/LOLog-ai/.venv/bin/python -m pip install "redis>=5" 2>/dev/null || (cd ~/LOLog-ai && uv pip install "redis>=5")
  sudo chmod 711 ~                                        # lolog가 측정 venv를 읽도록 지나가기만 허용
  PY=$(readlink -f ~/LOLog-ai/.venv/bin/python)
  SP=$(~/LOLog-ai/.venv/bin/python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
  cd /opt/lolog && sudo -u lolog $PY -m venv --without-pip /opt/lolog/venv
  NEWSP=$(sudo -u lolog /opt/lolog/venv/bin/python -c "import site;print(site.getsitepackages()[0])")
  echo "$SP" | sudo -u lolog tee $NEWSP/flux-venv.pth
  ```

## 1. SSH 터널 (두 대 모두)

교육기관 서버 쪽(한 번만). 서버는 학생 공용(s2.java21.net, SSH 8822, 민주님 학번 계정)이라 6379·18080을 직접 못 열고,
Redis와 lolog-api를 개인 포트 범위의 `127.0.0.1:<TUNNEL_REDIS_PORT>`, `127.0.0.1:<TUNNEL_API_PORT>`에 열어 둔다
(LOLog-api `deploy/README.md` "Jetson 터널"). 젯슨에서는 그대로 localhost:6379, :18080으로 보인다.
```bash
# 학번 계정의 ~/.ssh/authorized_keys 에 젯슨마다 한 줄. 이 키로는 터널만 되고 셸·다른 포트는 안 된다
restrict,port-forwarding,permitopen="127.0.0.1:<TUNNEL_REDIS_PORT>",permitopen="127.0.0.1:<TUNNEL_API_PORT>" ssh-ed25519 AAAA... jetson-1
```

젯슨 쪽
```bash
sudo ssh-keygen -t ed25519 -N '' -C "$(hostname)" -f /etc/lolog/tunnel_ed25519   # .pub을 서버에 등록
sudo ssh-keyscan -p 8822 s2.java21.net | sudo tee /etc/lolog/known_hosts
sudo chown root:lolog /etc/lolog/tunnel_ed25519 /etc/lolog/known_hosts && sudo chmod 640 /etc/lolog/tunnel_ed25519
sudo cp tunnel.env.example /etc/lolog/tunnel.env && sudo nano /etc/lolog/tunnel.env
```

## 2. LLM 서버 (1호만)

```bash
sudo apt install -y cmake build-essential libcurl4-openssl-dev
sudo -u lolog git clone https://github.com/ggml-org/llama.cpp /opt/lolog/llama.cpp
cd /opt/lolog/llama.cpp
sudo -u lolog env PATH=/usr/local/cuda/bin:$PATH cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release
sudo -u lolog cmake --build build -j4 --target llama-server      # 20~40분

sudo -u lolog mkdir -p /opt/lolog/models
# 받기 전에 Hugging Face에서 저장소·파일 이름 확인
sudo -u lolog /opt/lolog/venv/bin/pip install -U "huggingface_hub[cli]"
sudo -u lolog /opt/lolog/venv/bin/hf download unsloth/Qwen3-4B-Instruct-2507-GGUF \
  Qwen3-4B-Instruct-2507-Q4_K_M.gguf --local-dir /opt/lolog/models

sudo cp llm.env.example /etc/lolog/llm.env && sudo nano /etc/lolog/llm.env   # LAN 주소, API 키

# LLM_HOST를 Tailscale 주소로 두면 tailnet 밖에서는 안 보이므로 방화벽은 생략해도 된다
# LAN 주소로 열 때만: 8000번은 1호 자신과 2호만
sudo apt install -y ufw
sudo ufw allow OpenSSH   # SSH가 끊기지 않게 먼저
sudo ufw allow from <2호 LAN 주소> to any port 8000 proto tcp
sudo ufw enable
```

- `-np 1`: 요청을 하나씩만 받는다. 기본값(여러 슬롯)이면 문맥이 슬롯 수로 쪼개져 1시간 묶음이 안 들어간다
- `-c 24576`: 1시간 묶음(약 1.5만 토큰 추정) + 지시문 + 출력
- `-ctk/-ctv q8_0`: KV 캐시를 8비트로 저장해 문맥 메모리를 반으로
- 모델을 바꿀 때는 `llm.env`의 `LLM_MODEL_PATH`만 고치고 `sudo systemctl restart lolog-llm`

## 3. 워커

```bash
sudo cp worker.env.example /etc/lolog/worker.env && sudo nano /etc/lolog/worker.env   # WORKER_NAME, 비밀값
sudo chown root:lolog /etc/lolog/*.env && sudo chmod 640 /etc/lolog/*.env
```

비밀값(`INTERNAL_TOKEN`, Redis 비밀번호, `LLM_API_KEY`)은 레포에 올리지 않는다.

### 2호만: 그림 모델 준비

FLUX 측정 때 받아 둔 파일을 `lolog` 사용자 쪽으로 복사한다(측정한 사용자 홈에서).
```bash
sudo mkdir -p /opt/lolog/.cache /opt/lolog/models
sudo cp -r ~/.cache/huggingface /opt/lolog/hf-cache          # FLUX.2-klein-4B 원본
sudo cp -r ~/.cache/lolog /opt/lolog/.cache/                 # 한글 폰트(fonts/Jua-Regular.ttf), 얼굴 검출(models/animeface/model.onnx)
sudo chown -R lolog:lolog /opt/lolog/hf-cache /opt/lolog/.cache /opt/lolog/models
```

- 첫 만화를 그릴 때 텍스트 인코더와 이미지 모델을 nf4로 바꿔 `/opt/lolog/models/flux2-klein-4b-nf4/`에 저장한다(각 약 100초, 한 번만).
  다음부터는 저장한 것을 바로 읽는다
- 만화 하나에 텍스트 인코더 → 이미지 모델 순서로 올렸다 내린다. 8GB에 둘을 같이 둘 수 없어서다
- 화풍 LoRA가 나오면 `worker.env`의 `FLUX_LORA`에 경로를 넣고 `sudo systemctl restart lolog-worker@comic`

## 4. 켜기

```bash
# 폴더가 lolog 전용이라 * 는 sudo 안에서 풀어야 한다
sudo sh -c 'cp /opt/lolog/LOLog-ai/deploy/jetson/systemd/*.service /etc/systemd/system/' && sudo systemctl daemon-reload

# 1호
sudo systemctl enable --now lolog-tunnel lolog-llm lolog-worker@detect
# 2호 (1호 LLM이 먼저 켜져 있어야 함. 워커가 켜질 때 1호 LLM을 기다린다)
sudo systemctl enable --now lolog-tunnel lolog-worker@comic
```

## 5. 확인

```bash
systemctl status lolog-tunnel lolog-llm lolog-worker@detect
journalctl -u lolog-worker@detect -f            # "시작: detect · lolog:detect · 컨슈머 jetson-1"
nc -z localhost 6379 && nc -z localhost 18080 && echo 터널OK
curl -s http://100.98.23.5:8000/health             # {"status":"ok"} (키 없이도 health는 열림)
sudo reboot   # 다시 켜진 뒤 위 status가 모두 active면 끝
```

시험 메시지 (서버의 Redis에 직접 넣음. 백엔드에 없는 recordingId면 cards 보내기에서 400이 나는 게 정상)
```bash
redis-cli -a <비밀번호> XADD lolog:detect '*' payload '{"jobId":1,"recordingId":1,"partNo":1,"isLast":true,"sentences":[{"startMs":0,"endMs":2000,"text":"야 너 어제 그거 봤어"}]}'
```

## 코드 올리기 (main에 머지된 뒤)

젯슨은 Tailscale 안에만 있어 GitHub Actions로 배포하지 않는다. 각 젯슨에서 직접 받는다.
```bash
# 처음 한 번: 손으로 복사해 둔 폴더를 git 저장소로 바꾼다
sudo mv /opt/lolog/LOLog-ai /opt/lolog/LOLog-ai.old
sudo -u lolog git clone -b main https://github.com/LOLog-2026/LOLog-ai /opt/lolog/LOLog-ai
# 그 뒤로는
sudo -u lolog git -C /opt/lolog/LOLog-ai pull --ff-only
sudo systemctl restart lolog-worker@detect   # 2호는 lolog-worker@comic
```
- 서비스 파일(`systemd/`)이 바뀌었으면 4장의 복사와 `daemon-reload`도 다시 한다
- 끝까지 되는지 확인은 `RUNBOOK.md`의 E

## 로그 규칙

- 워커 로그는 journald로만 간다. 대화 글, 토큰, 비밀값은 남기지 않는다(TR-12). jobId, 개수, 걸린 시간만
- `journalctl --vacuum-time=14d`로 2주만 둔다
