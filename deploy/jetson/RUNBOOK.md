# 젯슨 끝까지 한 번 돌리기

> 2026-11-08 처음 구축할 때 쓴 순서다. 그때는 이 레포에 코드가 없어서 맥의 `~/jetson-bench/jetson-infra`(이 레포의 `deploy/`, `src/`와 같은 내용)를 `scp`로 복사했다. 지금은 README의 "코드 올리기"처럼 git으로 받는다.

이 순서대로 A부터 G까지 끝나면 젯슨 쪽 인프라는 끝이다.
각 단계 제목에 **어디서 치는지**를 적었다. `<...>`는 실제 값으로 바꿔 넣는다.

| 단계 | 내용 | 어디서 | 막힌 것 |
|---|---|---|---|
| A | 1호 LLM을 서비스로 | 맥 → 1호 | 없음. 지금 할 수 있음 |
| B | 2호 준비 | 맥 → 2호 | 2호가 Tailscale에 붙어 있어야 함 |
| C | 터널 | 1호·2호·공용 서버 | 서버 포트(`TUNNEL_REDIS_PORT`, `TUNNEL_API_PORT`)가 정해져야 함 |
| D | 워커 켜기 | 1호·2호 | 서버에 Redis·백엔드가 떠 있어야 함 |
| E | 끝까지 한 번 | 맥(아무 컴퓨터) | 서버 공개 주소 |
| F | 재부팅 뒤 다시 | 1호·2호 → 맥 | |
| G | 시간 적기 | 1호·2호 | |

---

## A. 1호 LLM을 서비스로 (지금 가능)

손으로 띄운 `llama-server`가 있으면 먼저 끈다(그 tmux 창에서 Ctrl+C).

**맥**에서 코드를 1호로 보낸다.
```bash
scp -r ~/jetson-bench/jetson-infra <1호 사용자>@100.98.23.5:~/
```

**1호**에서
```bash
cd ~/jetson-infra
sudo mkdir -p /opt/lolog/LOLog-ai
sudo cp -r deploy src /opt/lolog/LOLog-ai/ && sudo chown -R lolog:lolog /opt/lolog/LOLog-ai
sudo -u lolog /opt/lolog/venv/bin/pip install "redis>=5"

sudo usermod -aG video,render lolog   # GPU 권한. 없으면 CPU로 돎(no CUDA-capable device)

# API 키를 만들어 바로 넣는다. 나중에 볼 때는 sudo grep LLM_API_KEY /etc/lolog/llm.env (레포·채팅에 올리지 않기)
sudo cp deploy/jetson/llm.env.example /etc/lolog/llm.env
KEY=$(openssl rand -hex 32)
sudo sed -i "s/^LLM_API_KEY=.*/LLM_API_KEY=$KEY/" /etc/lolog/llm.env
sudo chown root:lolog /etc/lolog/llm.env && sudo chmod 640 /etc/lolog/llm.env

sudo cp deploy/jetson/systemd/*.service /etc/systemd/system/ && sudo systemctl daemon-reload
sudo systemctl enable --now lolog-llm
journalctl -u lolog-llm -f            # "offloaded 37/37 layers to GPU", "listening on" 나오면 Ctrl+C
```

확인 (1호에서)
```bash
curl -s http://100.98.23.5:8000/health                        # {"status":"ok"}
curl -s http://100.98.23.5:8000/v1/models | head -c 200        # 키 없이: 401이 나와야 정상
curl -s -H "Authorization: Bearer $KEY" http://100.98.23.5:8000/v1/models | head -c 200   # qwen3-4b-q4 보임
```

메모리를 아끼려고 데스크톱을 아직 안 껐다면 지금 끈다.
```bash
sudo systemctl set-default multi-user.target && sudo reboot
# 다시 켜진 뒤 health가 또 ok면 자동 시작도 된 것
```

## B. 2호 준비

먼저 2호에서 `tailscale ip -4`를 쳐 본다. 주소가 나오면 그대로 진행하고, 없으면 `sudo tailscale up`으로 1호와 같은 계정에 붙인다.

**2호**에서 (README 0장과 같음)
```bash
sudo systemctl set-default multi-user.target
sudo apt install -y autossh nano
sudo useradd --system --create-home --home-dir /opt/lolog --shell /usr/sbin/nologin lolog
sudo usermod -aG video,render lolog
sudo chmod 755 /opt/lolog
sudo mkdir -p /etc/lolog && sudo chown root:lolog /etc/lolog && sudo chmod 750 /etc/lolog
```

**맥**에서 코드를 2호로
```bash
scp -r ~/jetson-bench/jetson-infra <2호 사용자>@<2호 주소>:~/
```

**2호**에서
```bash
cd ~/jetson-infra
sudo mkdir -p /opt/lolog/LOLog-ai
sudo cp -r deploy src /opt/lolog/LOLog-ai/ && sudo chown -R lolog:lolog /opt/lolog/LOLog-ai

# FLUX 측정 때 쓴 venv에 redis 추가하고 /opt/lolog/venv로 연결
<측정 venv 경로>/bin/pip install "redis>=5" numpy pillow
sudo ln -s <측정 venv 경로> /opt/lolog/venv
sudo -u lolog /opt/lolog/venv/bin/python -c "import torch, diffusers, redis; print(torch.cuda.is_available())"   # True
#  → Permission denied면 측정 venv가 있는 홈 폴더를 lolog가 못 읽는 것: sudo chmod o+rx ~ 후 다시

# 그림 모델·폰트·얼굴 모델 복사 (README 3장 "2호만")
sudo mkdir -p /opt/lolog/.cache /opt/lolog/models
sudo cp -r ~/.cache/huggingface /opt/lolog/hf-cache
sudo cp -r ~/.cache/lolog /opt/lolog/.cache/
sudo chown -R lolog:lolog /opt/lolog/hf-cache /opt/lolog/.cache /opt/lolog/models

# 1호 LLM이 2호에서 보이는지
curl -s http://100.98.23.5:8000/health          # {"status":"ok"}
```

## C. 터널 (서버 포트가 정해지면)

**1호와 2호 각각**에서 키를 만든다.
```bash
sudo ssh-keygen -t ed25519 -N '' -C "$(hostname)" -f /etc/lolog/tunnel_ed25519
sudo cat /etc/lolog/tunnel_ed25519.pub          # 이 한 줄을 복사
sudo ssh-keyscan -p 8822 s2.java21.net | sudo tee /etc/lolog/known_hosts
sudo chown root:lolog /etc/lolog/tunnel_ed25519 /etc/lolog/known_hosts && sudo chmod 640 /etc/lolog/tunnel_ed25519
```

**공용 서버**(학번 계정)에서 `nano ~/.ssh/authorized_keys`로 젯슨마다 한 줄 추가한다.
```
restrict,port-forwarding,permitopen="127.0.0.1:<TUNNEL_REDIS_PORT>",permitopen="127.0.0.1:<TUNNEL_API_PORT>" <1호 .pub 한 줄>
restrict,port-forwarding,permitopen="127.0.0.1:<TUNNEL_REDIS_PORT>",permitopen="127.0.0.1:<TUNNEL_API_PORT>" <2호 .pub 한 줄>
```

**1호와 2호 각각**
```bash
sudo apt install -y nano                 # 1호에는 nano가 없음
sudo cp /opt/lolog/LOLog-ai/deploy/jetson/tunnel.env.example /etc/lolog/tunnel.env
sudo nano /etc/lolog/tunnel.env          # SSH_USER=학번, 두 포트
sudo chown root:lolog /etc/lolog/tunnel.env && sudo chmod 640 /etc/lolog/tunnel.env
sudo sh -c 'cp /opt/lolog/LOLog-ai/deploy/jetson/systemd/*.service /etc/systemd/system/' && sudo systemctl daemon-reload   # * 는 sudo 안에서 풀어야 함(폴더가 lolog 전용)
sudo systemctl enable --now lolog-tunnel
nc -z localhost 6379 && nc -z localhost 18080 && echo 터널OK
```
`터널OK`가 안 나오면 `journalctl -u lolog-tunnel -n 30`. `Permission denied`면 authorized_keys 줄, `Connection refused`면 서버에 Redis·백엔드가 아직 안 뜬 것이다.

## D. 워커 켜기 (서버 스택이 떠 있으면)

### D-1. 고친 워커 코드 넣기 (Redis가 없으면 죽지 않고 기다리게 바꾼 것)

**맥에서**
```bash
scp ~/jetson-bench/jetson-infra/src/lolog_ai/worker/__main__.py aip-1@100.98.23.5:~/
scp ~/jetson-bench/jetson-infra/src/lolog_ai/worker/__main__.py minju@100.73.209.39:~/
```
**1호와 2호 각각**
```bash
sudo install -o lolog -g lolog -m 644 ~/__main__.py /opt/lolog/LOLog-ai/src/lolog_ai/worker/__main__.py
```

### D-2. 비밀값 넣기 (nano 없이, 화면에 안 보이게)

공용 서버에서 `grep -E 'REDIS_PASSWORD|INTERNAL_TOKEN' ~/lolog-infra/.env`로 두 값을 본다. 채팅에는 붙이지 않는다.

**1호**
```bash
read -s -p "REDIS_PASSWORD: " RP; echo
read -s -p "INTERNAL_TOKEN: " IT; echo
LK=$(sudo sed -n 's/^LLM_API_KEY=//p' /etc/lolog/llm.env)
sudo cp /opt/lolog/LOLog-ai/deploy/jetson/worker.env.example /etc/lolog/worker.env
sudo sed -i -e "s|^REDIS_URL=.*|REDIS_URL=redis://:$RP@localhost:6379/0|" -e "s|^INTERNAL_TOKEN=.*|INTERNAL_TOKEN=$IT|" \
  -e "s|^LLM_API_KEY=.*|LLM_API_KEY=$LK|" -e "s|^WORKER_NAME=.*|WORKER_NAME=jetson-1|" /etc/lolog/worker.env
sudo chown root:lolog /etc/lolog/worker.env && sudo chmod 640 /etc/lolog/worker.env
```

**2호** (LLM 키는 1호에서 `sudo sed -n 's/^LLM_API_KEY=//p' /etc/lolog/llm.env`로 보고 입력)
```bash
read -s -p "REDIS_PASSWORD: " RP; echo
read -s -p "INTERNAL_TOKEN: " IT; echo
read -s -p "LLM_API_KEY: " LK; echo
sudo cp /opt/lolog/LOLog-ai/deploy/jetson/worker.env.example /etc/lolog/worker.env
sudo sed -i -e "s|^REDIS_URL=.*|REDIS_URL=redis://:$RP@localhost:6379/0|" -e "s|^INTERNAL_TOKEN=.*|INTERNAL_TOKEN=$IT|" \
  -e "s|^LLM_API_KEY=.*|LLM_API_KEY=$LK|" -e "s|^WORKER_NAME=.*|WORKER_NAME=jetson-2|" \
  -e 's|^# HF_HOME=|HF_HOME=|' -e 's|^# HF_HUB_OFFLINE=|HF_HUB_OFFLINE=|' -e 's|^# FLUX_NF4_DIR=|FLUX_NF4_DIR=|' /etc/lolog/worker.env
sudo chown root:lolog /etc/lolog/worker.env && sudo chmod 640 /etc/lolog/worker.env
```

확인 (값은 가리고 이름만): `sudo sed 's/=.*/=…/' /etc/lolog/worker.env | grep -v '^#'`

### D-3. 켜기

```bash
# 1호
sudo systemctl enable lolog-worker@detect && sudo systemctl restart lolog-worker@detect
sudo journalctl -u lolog-worker@detect -n 5 --no-pager    # "시작: detect · lolog:detect · 컨슈머 jetson-1"
# 2호
sudo systemctl enable lolog-worker@comic && sudo systemctl restart lolog-worker@comic
sudo journalctl -u lolog-worker@comic -n 5 --no-pager     # "시작: comic · lolog:comic · 컨슈머 jetson-2"
```
`NOAUTH`/`WRONGPASS`가 나오면 REDIS_PASSWORD, 백엔드 호출에서 401이 나오면 INTERNAL_TOKEN이 서버 `.env`와 다른 것이다.

## E. 끝까지 한 번 (맥에서)

시험용 대화 `e2e/sample_friends_2min.json`을 쓴다(친구끼리 편의점 이야기, 14구간). 37분 방송 샘플은 규칙상 카드가 거의 안 나와서 시험에 맞지 않다.

```bash
cd deploy/jetson/e2e        # 맥에서 복사해 쓸 때는 ~/jetson-bench/jetson-infra/e2e
BASE=https://lolog.java21.net
jget() { python3 -c "import sys,json;d=json.load(sys.stdin);print($1)"; }   # zsh에서도 됨

# 1) 가입(처음 한 번만), 로그인 → 토큰 (30분 뒤 만료되면 TOKEN 줄만 다시)
curl -s -X POST $BASE/auth/signup -H 'Content-Type: application/json' -d '{"email":"e2e@lolog.test","password":"<비밀번호>"}'; echo
TOKEN=$(curl -s -X POST $BASE/auth/login -H 'Content-Type: application/json' \
  -d '{"email":"e2e@lolog.test","password":"<비밀번호>"}' | jget 'd["accessToken"]')

# 2) 녹음 글 올리기. 같은 clientRecordingId는 다시 처리하지 않으므로 매번 새 값으로
python3 -c 'import json,uuid;d=json.load(open("sample_friends_2min.json"));d["clientRecordingId"]=str(uuid.uuid4());print(json.dumps(d,ensure_ascii=False))' > /tmp/rec.json
REC=$(curl -s -X POST $BASE/api/recordings -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d @/tmp/rec.json | jget 'd["recordingId"]')
echo "녹음 $REC"

# 3) 1~2분 뒤: 감지 결과(DONE, cardCount 1 이상)와 만화 요청
curl -s $BASE/api/recordings/$REC -H "Authorization: Bearer $TOKEN"; echo
CARD=$(curl -s $BASE/api/cards -H "Authorization: Bearer $TOKEN" | jget 'd["cards"][0]["id"]')
COMIC=$(curl -s -X POST $BASE/api/cards/$CARD/comics -H "Authorization: Bearer $TOKEN" | jget 'd["comicId"]')
echo "카드 $CARD, 만화 $COMIC"

# 4) 몇 분 간격으로: QUEUED → STORYBOARD → DRAWING → DONE (약 6~9분)
curl -s $BASE/api/comics/$COMIC -H "Authorization: Bearer $TOKEN"; echo
#  → DONE이면 imageUrl을 브라우저로 연다(10분 뒤 만료, 다시 GET하면 새 주소)
```

여기까지 되면 1번(카드)과 2번(만화)이 끝난 것이다.

막히면 볼 곳
| 증상 | 볼 곳 |
|---|---|
| 녹음이 RECEIVED에서 안 움직임 | 1호 `journalctl -u lolog-worker@detect -n 50`. 아무것도 없으면 터널(`nc -z localhost 6379`) |
| 녹음 FAILED | 1호 워커 로그의 실패 이유 |
| cardCount 0 | 실패는 아님. 다른 대화로 다시 해 보거나 알려 주기 |
| 만화가 QUEUED에서 안 움직임 | 2호 `journalctl -u lolog-worker@comic -n 50` |
| 만화 FAILED | 2호 로그. `StoryboardInvalid`면 콘티, `CUDA out of memory`면 2호 메모리 |

## F. 재부팅 뒤 다시

```bash
# 1호, 2호 둘 다
sudo reboot
# 다시 켜진 뒤
systemctl is-active lolog-tunnel lolog-llm lolog-worker@detect    # 1호: active 세 줄
systemctl is-active lolog-tunnel lolog-worker@comic               # 2호: active 두 줄
```
그다음 E의 2)부터 다시 한다(토큰이 30분 지났으면 1)의 로그인부터). 또 DONE이 나오면 3번도 끝이다.

## G. 시간 적기

```bash
journalctl -u lolog-worker@detect --since today | grep "처리\|이야기 찾기\|화자 나누기"   # 1호: 감지 시간
journalctl -u lolog-worker@comic  --since today | grep "처리\|콘티\|그림:"             # 2호: 콘티·그림 시간
```
두 번째 만화(nf4 저장이 끝난 뒤)의 시간이 실제 시간이다. 그 값과 F 결과를 알려 주면 정리하고 PR을 연다.
