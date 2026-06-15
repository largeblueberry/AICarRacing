## 1-A. State(상태) 정의

본 에이전트의 상태(state)는 게임 화면의 픽셀(이미지)이다. 즉 차량 좌표·속도·각도 같은 저차원 수치 벡터가 아니라, 환경이 매 스텝 렌더링한 톱다운(top-down) 화면을 그대로 정책 네트워크의 입력으로 사용하는 **순수 픽셀 기반(vision-only) 상태**다. 아래에서 원시 관측이 정책 입력 텐서가 되기까지의 전처리 파이프라인과, 장애물 환경에서 상태에 추가되는 정보, 그리고 학습에 쓰는 상태와 화면 표시용 화면의 차이를 코드 근거와 함께 정의한다.

### 상태 전처리 파이프라인

원시 관측 → grayscale → FrameStack(4) → 정책 입력 `(4, 96, 96)` → `/255` 정규화의 순서로 변환되며, 각 단계는 `gym` Wrapper로 구현되어 환경 생성 시 다음 순서로 합성된다(예: `scripts/train_ppo_2action_obstacles.py:117-120`).

```python
env = ActionWrapper(env)          # 행동 좌표계 변환 (상태 아님 — 아래 참고)
env = GrayScaleObservation(env)   # RGB(96,96,3) -> Gray(96,96)
env = TimeLimit(env, max_episode_steps=max_episode_steps)
env = FrameStack(env, frame_stack)# Gray(96,96) -> (4,96,96)
```

| 단계 | 입력 shape | 출력 shape | 근거 (file:line) |
|------|-----------|-----------|------------------|
| 원시 관측 (`state_pixels`) | — | `(96, 96, 3)` uint8 RGB | `car_racing_obstacles.py:133` (`self._render("state_pixels")`) |
| GrayScaleObservation | `(96, 96, 3)` | `(96, 96)` uint8 | `env_wrappers.py:22-23, 28` |
| FrameStack(k=4) | `(96, 96)` | `(4, 96, 96)` uint8 | `env_wrappers.py:95, 119-123` |
| 정책 입력 정규화 | `(B, 4, 96, 96)` uint8 | `[0,1]` float | `cnn_model.py:98` |

1. **원시 관측 — 96×96 `state_pixels`.** 기본 환경의 관측은 96×96 해상도의 RGB 화면이다. 장애물 환경에서는 `reset()`이 장애물 생성 직후 `self._render("state_pixels")`로 첫 관측을 다시 렌더링해 0프레임부터 장애물이 보이도록 보장한다(`car_racing_obstacles.py:131-134`).

2. **Grayscale 변환.** `GrayScaleObservation`이 RGB `(H,W,3)`를 OpenCV `cv2.cvtColor(..., COLOR_RGB2GRAY)`로 단일 채널 `(H,W)` uint8로 축소한다(`env_wrappers.py:25-29`). 관측 공간도 `Box(low=0, high=255, shape=(96,96), dtype=uint8)`로 갱신된다(`env_wrappers.py:22-23`). 채널 수를 1/3로 줄여 입력 차원을 낮추는 동시에, 뒤이은 FrameStack의 채널 축을 "시간(프레임)" 전용으로 비워 둔다.

3. **FrameStack(4) — 시간 정보 주입.** `FrameStack`은 최근 `k=4`개의 그레이스케일 프레임을 `deque(maxlen=k)`에 모아 `np.stack(..., axis=0)`로 채널 축에 쌓는다(`env_wrappers.py:88, 101-104, 119-123`). 그 결과 정책에 들어가는 단일 상태 텐서는 `(4, 96, 96)`이 된다(관측 공간 `(k,)+(H,W)`, `env_wrappers.py:95`). 단일 정지 프레임만으로는 차량의 **속도·진행 방향 같은 동역학을 추론할 수 없기** 때문에, 연속 4프레임을 함께 제공해 CNN이 프레임 간 차이로 운동 정보를 복원하도록 한다(클래스 docstring `env_wrappers.py:70-72`). `reset()` 시에는 첫 관측을 4번 복제해 버퍼를 채운다(`env_wrappers.py:113-117`).

4. **`/255` 정규화.** uint8 `[0,255]` 상태는 CNN 특징 추출기 내부에서 `observations.float() / 255.0`로 `[0,1]`로 정규화된 뒤 합성곱 층에 들어간다(`cnn_model.py:96-98`). 정규화가 환경 Wrapper가 아니라 모델 `forward` 내부에서 일어나므로, 리플레이/버퍼에는 메모리 효율이 좋은 uint8 그대로 저장하고 정규화는 순전파 시점에 1회 수행된다(`ppo_agent_2.py:289, 376, 493` 주석 "normalization inside extractor"). 더미 forward로 flatten 크기를 잴 때도 동일하게 `/255.0`를 적용한다(`cnn_model.py:73`). 첫 합성곱의 입력 채널 수는 관측 공간의 0번째 축, 즉 스택 프레임 수(`n_input_channels = observation_space.shape[0]` = 4)로 설정된다(`cnn_model.py:49, 53`).

요약하면 정책이 보는 상태 텐서는 **`(4, 96, 96)`의 그레이스케일 프레임 스택이며, `[0,1]`로 정규화된 픽셀 강도 값**이다.

### 학습용 상태(`state_pixels`) vs 화면 표시용(`rgb_array`)

상태로 쓰는 관측과 사람이 보기 위한 영상은 별개다.

- **학습/평가 입력**은 위의 96×96 `state_pixels` 관측이다. 학습·평가는 `render_mode=None`으로 환경을 만들고(`train_ppo_2action_obstacles.py:101`, `evaluate_agent_2action_obstacles.py:52`의 `"render_mode": None`), 관측은 내부적으로 `state_pixels`로 렌더된 96×96 프레임을 사용한다(`car_racing_obstacles.py:133`).
- **표시/녹화용**은 `render_mode="rgb_array"`로 생성한 고해상도(400×600) 화면이다. 영상 녹화 스크립트만 이 모드를 쓰며(`record_video.py:316`의 `render_mode="rgb_array"`), 그 docstring은 *"render mode does not affect the physics, track ... compared to the render_mode=None evaluation run"*라고 명시한다(`record_video.py:95-98`). 즉 표시 화면은 보기 좋은 큰 프레임일 뿐 **에이전트의 상태가 아니며**, 정책 입력에 쓰이는 것은 어디까지나 96×96 `state_pixels`다. 두 경로 모두 동일한 Wrapper 스택(GrayScale → TimeLimit → FrameStack)을 적용한다(`record_video.py:124-126`).

### 장애물 환경에서의 상태 — 흰색 quad가 상태에 들어가는 이유

장애물 환경의 핵심 설계는, 장애물을 **관측 픽셀 안에 흰색(255)으로 그려 넣어 픽셀 기반 에이전트가 인지 가능하게** 만든 점이다. 각 장애물은 실제 Box2D 정적 물체(`CreateStaticBody`)인 동시에(`car_racing_obstacles.py:188-194`), 그 사각형 quad가 `self.road_poly`에 `(quad, OBSTACLE_COLOR)`로 추가되어 렌더 프레임에 그려진다(`car_racing_obstacles.py:196-203`).

이것이 중요한 이유는 **에이전트의 유일한 입력이 픽셀이기 때문**이다. 물리 충돌만 존재하고 화면에 그려지지 않으면, 상태(이미지)에는 장애물이 전혀 나타나지 않아 에이전트가 회피를 학습할 단서가 없다. 모듈 docstring이 이를 명시한다: *"the agent's input is pixels: the obstacle must be visible in the 96x96 observation to be avoidable"*(`car_racing_obstacles.py:7-9`).

색을 흰색으로 고른 것도 상태 가시성 때문이다. 그레이스케일 변환 후 도로는 약 102, 잔디는 약 162인데 장애물은 255가 되어 **가장 강한 대비**로 부각된다(`car_racing_obstacles.py:9-10, 35-36`의 `OBSTACLE_COLOR = (255, 255, 255)`). 정규화 후에도 장애물 픽셀은 1.0, 도로는 약 0.4로 분리되어 CNN이 쉽게 식별한다. 또한 장애물 배치는 `self.np_random`을 사용해 시드별로 재현 가능하므로(`car_racing_obstacles.py:14, 175, 182`), 동일 시드에서 동일한 상태 분포가 보장된다.

### 행동 좌표계는 상태가 아님 (별도 절 참고)

`ActionWrapper`는 정책이 출력하는 2차원 행동 `[steering, throttle]`을 환경 네이티브 3차원 행동 `[steering, gas, brake]`로 변환한다(`throttle>0`이면 gas, `throttle<0`이면 brake; `env_wrappers.py:124-142`). 이는 **상태가 아니라 행동(action) 좌표계** 변환이며 상태 정의에는 영향을 주지 않는다 — 자세한 행동 공간 설계는 별도 절에서 다룬다.
