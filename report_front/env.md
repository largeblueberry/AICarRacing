## 0. 실행 환경 및 설치 (재현성)

본 프로젝트는 **학습**과 **평가/녹화**를 서로 다른 환경에서 수행한다. 무거운 PPO 학습은 원격 A100 GPU 서버(conda 환경 `teamB_env`, CUDA 빌드 PyTorch)에서, 학습이 끝난 체크포인트의 평가·영상 녹화는 로컬 macOS(conda 환경 `racing`, CPU)에서 진행했다. 아래 버전은 로컬 `racing` 환경에서 `importlib.metadata`로 실측한 값이며, 채점 환경 재현 시 그대로 맞추면 된다.

### 0.1 사용 환경 / 버전

| 구분 | 패키지 / 항목 | 버전 | 용도 |
|------|---------------|------|------|
| 공통 | Python | 3.11.15 | 인터프리터 |
| 학습(원격) | 환경 | conda `teamB_env` / A100 CUDA | PPO 학습 (GPU) |
| 평가·녹화(로컬) | 환경 | conda `racing` / macOS CPU | 체크포인트 평가, mp4 녹화 |
| 딥러닝 | torch | 2.12.0 (로컬 CPU 빌드) | 학습/추론 — 원격 학습은 `teamB_env`의 CUDA 빌드 torch 사용 |
| 강화학습 | gymnasium | 1.3.0 | 환경 API (CarRacing) |
| 물리엔진 | box2d-py | 2.3.8 | CarRacing/Box2D 물리 (`gymnasium[box2d]`가 끌어옴) |
| 렌더링 | pygame | 2.6.1 | 환경 렌더링 (`gymnasium[box2d]`가 끌어옴) |
| 영상 전처리 | **opencv-python (cv2)** | 4.x | grayscale 변환 (`src/env_wrappers.py`의 `import cv2`) — **필수** |
| 수치연산 | numpy | 2.4.6 | 배열 연산 |
| 녹화 | imageio / imageio-ffmpeg | 2.37.3 / 0.6.0 | mp4 영상 저장 |
| 로깅·시각화 | tensorboard / matplotlib | 2.20.0 / 3.10.9 | 학습 곡선, 그림 |

> 제출물에 **`requirements.txt`를 동봉**했다(`pip install -r requirements.txt`). 아래는 그 내용을 풀어 쓴 것이다.

### 0.2 설치 (기존 실습 docker image에 추가 설치)

기존 실습용 docker base image를 그대로 사용하되, 아래 pip 패키지만 추가로 설치하면 된다.

복붙용 한 줄(권장):

```bash
pip install "torch>=2.6" "gymnasium[box2d]==1.3.0" opencv-python numpy==2.4.6 imageio==2.37.3 imageio-ffmpeg==0.6.0 tensorboard==2.20.0 matplotlib==3.10.9
```

개별 줄(필요한 것만 골라 설치):

```bash
# base 이미지에 PyTorch가 이미 있으면 아래 torch 줄은 건너뛴다.
# (단, 체크포인트 로드에 torch>=2.6 필요 — 0.4 참조)
pip install "torch>=2.6"

pip install "gymnasium[box2d]==1.3.0"   # CarRacing 환경 + Box2D 물리 + pygame(렌더링)을 함께 끌어옴
pip install opencv-python                # ★ 필수: src/env_wrappers.py 의 cv2 (없으면 모든 스크립트가 import 단계에서 ModuleNotFoundError: cv2)
pip install numpy==2.4.6
pip install imageio==2.37.3 imageio-ffmpeg==0.6.0   # mp4 녹화(record_video.py)
pip install tensorboard==2.20.0 matplotlib==3.10.9  # 로깅 / 그림
```

> **각주 (Box2D 설치):** `gymnasium[box2d]`는 `box2d-py`(본 환경 2.3.8)와 `pygame`을 함께 설치한다. extras 설치가 실패하면 `pip install box2d-py pygame`로 직접 설치할 수 있다. 일부 환경에서는 C++ 바인딩 빌드를 위해 `swig`가 필요하다(예: `apt-get install -y swig` 또는 `conda install -c conda-forge swig` 후 재시도). `pygame`을 별도 버전으로 핀하면 `gymnasium`이 요구하는 버전과 충돌할 수 있으니 가급적 `gymnasium[box2d]`가 끌어오게 둔다.

### 0.3 실행 커맨드 (학습 / 평가 / 영상)

모든 스크립트는 **모듈 방식**(`python -m scripts.NAME`)으로 실행한다(패키지 상대 import 해석을 위함). 저장소 루트에서 실행할 것.

```bash
# --- 학습 (원격 A100 / teamB_env 권장) ---
# 베이스(장애물 없음)
python -m scripts.train_ppo_2action2 --steps 6000000

# 장애물 학습 (2-action)
python -m scripts.train_ppo_2action_obstacles \
    --accel-turn-weight 0.2 --obstacle-size-min 0.25 --obstacle-size-max 0.6

# 3-action 대조군
python -m scripts.train_ppo_3action_obstacles \
    --ent-coef 0.003 --obstacle-size-min 0.25 --obstacle-size-max 0.6

# --- 평가 (로컬 macOS / racing) ---
# 베이스
python -m scripts.evaluate_agent_2action \
    --model ./models/ppo_2action4/best_model.pth --episodes 100 --seed 42

# 장애물 (동봉 체크포인트: models/obs_small/best_model.pth)
python -m scripts.evaluate_agent_2action_obstacles \
    --model ./models/obs_small/best_model.pth \
    --obstacle-size-min 0.25 --obstacle-size-max 0.6 --episodes 50 --seed 42

# --- 영상 녹화 (mp4) ---
python scripts/record_video.py --model ./models/ppo_2action4/best_model.pth --episodes 5
```

> **장애물 환경 자동 등록:** 장애물 관련 스크립트는 `import src.car_racing_obstacles` 시점에 `gym.register(id="CarRacingObstacles-v0", ...)`가 자동 실행되므로(`src/car_racing_obstacles.py`), 별도 등록 절차 없이 `gym.make("CarRacingObstacles-v0", ...)`가 동작한다.
>
> **torch 버전 요구:** 평가/녹화 스크립트는 체크포인트를 `torch.load(..., weights_only=False)`로 로드한다(체크포인트에 numpy 스칼라 등이 포함되어 있어 `weights_only=True`로는 로드 불가). 이 인자 기본값 변경 때문에 **`torch>=2.6`이 필요**하다.

### 0.4 디렉토리 / 체크포인트 경로 관례 및 가용성 (중요)

체크포인트는 실행(run)별로 `./models/<run_name>/best_model.pth`에 저장되며, 학습 로그는 `./logs/`, 녹화 영상은 `./videos*/`에 위치한다.

> **채점자 주의 — 체크포인트 입수:** `.gitignore`에 `/models`가 있어 학습된 체크포인트는 git에 포함되지 않는다. **제출 번들에 핵심 평가용 체크포인트를 별도 동봉**했다:
>
> | 모델 | 경로 | clean 성능 | 비고 |
> |---|---|---|---|
> | 2-action 베이스(최종) | `models/ppo_2action4/best_model.pth` | 667 / 745 | 무장애물 |
> | 2-action 장애물 | `models/obs_small/best_model.pth` | 365 / 325 | 위 평가 명령이 이 파일을 가리킴 |
> | 3-action 베이스(참고) | `BestSavedAgents/evaluated641.pth` | ~675(임베디드) | git 추적본 |
>
> 보고서에 등장하는 **최고 2-action 장애물 모델(`obs_small_p02`, clean 415)와 3-action 장애물 모델(`obs_3action_lowent`, clean 229)은 원격 학습 서버에만 존재**하여 번들에서 제외했다(용량/접근). 이 수치들은 §8·장애물 보고서에 기록돼 있으며, 위 학습 커맨드(§0.3)로 재현 가능하다.
