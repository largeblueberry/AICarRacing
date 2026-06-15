## 1-B. Reward(보상) 정의

본 절은 학습/평가에 사용된 보상 신호를 세 층(layer)으로 나누어 정의한다. (1) 네이티브 CarRacing 보상, (2) 장애물 충돌 패널티, (3) 학습 전용 reward shaping. 모든 수식과 가중치는 실제 코드에서 그대로 인용했다.

### 보상 신호의 3개 층 구조

```
최종 step reward
  = [네이티브 CarRacing 보상]                  ← 항상 적용 (학습/평가 공통)
  + [장애물 충돌 패널티]                        ← 장애물 환경에서만 적용 (학습/평가 공통)
  + [reward shaping 항들]                       ← 학습 시에만 적용, 평가 시 제거
```

**핵심 구분 — shaped reward vs clean reward**
- **shaped reward (학습 시):** 위 3개 층을 모두 합산한 값. 에이전트가 실제로 받는 학습 신호이며, 속도/이탈/조향 등 행동 유도 항이 포함된다. shaping wrapper는 학습 스크립트에서 `use_reward_shaping=True`일 때만 환경에 부착된다 (`scripts/train_ppo_2action_obstacles.py:109-115`).
- **clean reward (평가 시):** shaping을 제거하고 **네이티브 보상 + 장애물 패널티만** 합산한 값. 보고서의 결과 수치(베이스 트랙 667점, 장애물 환경 415점 등)는 모두 이 **clean reward** 기준이다. shaping 항(특히 velocity 보상)은 점수를 인위적으로 부풀리므로, 모델 간/환경 간 비교는 반드시 네이티브 척도로만 수행했다.

---

### (1) 네이티브 CarRacing 보상 (gymnasium `car_racing`)

학습/평가 양쪽에서 항상 적용되는 원본 환경 보상이다. (gymnasium 설치본 `gymnasium/envs/box2d/car_racing.py` 기준)

| 항목 | 수식 / 값 | 적용 조건 | file:line |
|---|---|---|---|
| 프레임 패널티 | `reward -= 0.1` (스텝당 -0.1) | `action is not None`인 모든 스텝 | `car_racing.py:568` |
| 타일 통과 보상 | `reward += 1000.0 / len(track)` (= +1000/N, N=총 타일 수) | 새 트랙 타일을 처음 통과할 때마다 | `car_racing.py:92` |
| 플레이필드 이탈 | `step_reward = -100` + **에피소드 종료**(`terminated=True`) | `abs(x) > PLAYFIELD or abs(y) > PLAYFIELD` | `car_racing.py:579-582` |
| 랩 완주 | (보상 없음) + **에피소드 종료** | 모든 타일 통과 또는 새 랩 | `car_racing.py:574-577` |

설계 의미: 매 프레임 -0.1이 부과되므로 빨리 완주할수록 점수가 높다. 트랙 N개 타일을 모두 통과하면 누적 +1000, 732프레임에 완주 시 `1000 - 0.1*732 = 926.8`이 만점에 가까운 기준값이다 (`car_racing.py:140-141` docstring). 단, 플레이필드 이탈은 -100 + 즉시 사망이라 가장 큰 페널티다.

---

### (2) 장애물 충돌 패널티 (`src/car_racing_obstacles.py`)

네이티브 환경을 상속한 장애물 변형 환경의 추가 패널티. 학습/평가 공통으로 적용되며, **penalty-only 설계**(충돌해도 에피소드를 종료시키지 않음)다.

| 항목 | 수식 / 값 | 적용 조건 | file:line |
|---|---|---|---|
| 충돌 패널티 | `step_reward -= obstacle_penalty` (기본 `obstacle_penalty=15.0` → 스텝당 -15) | 해당 스텝에 **새 car↔obstacle 접촉이 시작**되고 `action is not None` | `src/car_racing_obstacles.py:144-145`, 기본값 `:87` |
| 스텝당 1회 한정 | `hits` 여러 번이어도 패널티는 1회만 차감 | 한 충돌이 hull+wheel 동시 접촉을 일으켜 과패널티되는 것 방지 | `src/car_racing_obstacles.py:140-146` |
| 비종료 | `terminated` 변경 없음 (super().step 결과 그대로) | 항상 (penalty-only) | `src/car_racing_obstacles.py:13`, `:137-148` |

충돌 검출은 `ObstacleFrictionDetector.BeginContact`가 장애물 접촉마다 `obstacle_hits_pending += 1`로 카운트하고(`src/car_racing_obstacles.py:49-53`), `step()`에서 그 값을 읽어 0보다 크면 한 번만 -15를 차감하는 구조다. 패널티 기본값과 장애물 수(`n_obstacles=10`)는 학습 config에서 지정한다 (`scripts/train_ppo_2action_obstacles.py:22-23`).

---

### (3) 학습 전용 Reward Shaping (`RewardShapingWrapper`)

`RewardShapingWrapper`는 두 학습 스크립트에 인라인으로 정의되어 있다 (`scripts/train_ppo_2action_obstacles.py:125-230`, `scripts/train_ppo_2action2.py:114~`). 아래 표의 가중치는 두 스크립트의 `config` 실제 사용값이다.

| 항목 | 수식 | 가중치(최종 사용값) | 적용 조건 | file:line |
|---|---|---|---|---|
| velocity (속도 보상) | `reward += speed * velocity_weight` | `velocity_reward_weight = 0.003` | **on-track**(`off_track=False`)이고 `speed > 0`일 때만 | wrapper `:196-199`, config `:55` |
| survival (생존 보상) | `reward += survival_reward` | `survival_reward = 0.0` (비활성) | (현재 0이라 무효과; 정지 버티기 꼼수 차단 목적으로 제거) | config `:56` |
| track_penalty (이탈 패널티) | `reward -= track_penalty` | `track_penalty = 1.0` | **off-track**일 때 (이때 velocity 보상은 미지급) | wrapper `:188-192`, config `:57` |
| steering_smooth (조향 급변) | `reward -= |steer - last_steer| * steering_smooth_weight` | `steering_smooth_weight = 0.001` | 매 스텝 | wrapper `:202-205`, config `:58` |
| accel_turn (코너 가속) | `reward -= weight * gas * |steer|` (`gas = max(0, action[1])`) | 장애물: **최종 0.2** (config 기본 0.5; CLI로 하향) / 베이스: **항 자체가 미구현** | `weight > 0`이고 조향 중 가속할 때만(직진 가속은 비용 0) | wrapper `:213-217`, config `:59`, CLI `:268-269` |

**속도(speed) 계산 주의점:** CarRacing-v3는 `info`에 `speed`를 제공하지 않아(항상 0) 속도 보상이 죽는다. 따라서 차량 물리에서 직접 계산한다 — `vx, vy = car.hull.linearVelocity; speed = sqrt(vx² + vy²)` (`scripts/train_ppo_2action_obstacles.py:166-173`).

**off-track 판정:** 관측 픽셀의 차량 영역(`obs[84:94, 42:54]`)에서 초록 채널 평균 > 150 이고 빨강 채널 평균 < 100 이면 잔디 위로 간주한다 (`scripts/train_ppo_2action_obstacles.py:181-185`).

**weight 튜닝 메모(코드 주석 근거):**
- `velocity_weight`는 0.03이면 step당 ~0.9로 네이티브 보상을 압도해 0.003으로 낮춤 (실속도 ≈ 20~60) (`:55`).
- `accel_turn_weight`(코너 가속 억제)는 장애물 스크립트에만 구현돼 있고 **베이스(train2) RewardShapingWrapper에는 항 자체가 없다**(인자만 받고 버림). 장애물 스크립트도 config 기본값은 0.5이나, 실험 결과 **0.5는 reward만 깎고 트랙 이탈(실제 원인은 장애물 충돌)을 줄이지 못해** CLI로 **0.2로 하향한 것이 최종값**이다(상세는 장애물 보고서 §5 진단 참조). 즉 "코너 패널티"는 이탈의 진짜 레버가 아니었다.
- `steering_smooth_weight`는 0.01이 움직임을 과도하게 방해해 0.001로 대폭 하향했다 (`:58`).

---

### 요약

학습 신호(shaped) = 네이티브 보상(`-0.1`/프레임, `+1000/N`/타일, 이탈 `-100`+종료) + 장애물 패널티(충돌 스텝당 `-15`, 비종료, 스텝당 1회) + shaping(velocity `+speed*0.003` on-track, off-track `-1.0`, steering_smooth `-0.001*Δsteer`, accel_turn `-weight*gas*|steer|`). 평가/보고 점수는 shaping을 제거한 **clean reward**(네이티브 + 장애물 패널티)로만 측정하여 모델·환경 간 비교의 공정성을 확보했다.

**관련 파일 (절대경로)**
- `/Users/younghoon-kang/AICarRacing/src/car_racing_obstacles.py` — 장애물 환경 및 충돌 패널티
- `/Users/younghoon-kang/AICarRacing/scripts/train_ppo_2action_obstacles.py` — RewardShapingWrapper(장애물, accel_turn=0.5) + config
- `/Users/younghoon-kang/AICarRacing/scripts/train_ppo_2action2.py` — RewardShapingWrapper(베이스, accel_turn=0.0) + config
- `/Users/younghoon-kang/anaconda3/envs/racing/lib/python3.11/site-packages/gymnasium/envs/box2d/car_racing.py` — 네이티브 CarRacing 보상
