# AICarRacing 장애물 회피 실험 보고서 — `CarRacingObstacles-v0` 구축, 코너 감속 오진(誤診), 그리고 진짜 실패 원인 규명 (팀 B)

> 본 보고서는 2-action PPO 베이스 모델(별도 보고서 `REPORT.md` 참조, clean 667)을 출발점으로 한 **무작위 정적 장애물 회피 태스크**의 환경 구축·실험·진단을 다룬다. `ent_coef` 축소 실험(§7)에 더해, **2-action vs native 3-action 대조 실험(§8)**까지 학습·평가·코드 진단을 완료했다.

---

## 1. 개요 / 목표

베이스 2-action 모델(장애물 없는 트랙에서 clean 667)에 **도로 위 무작위 정적 장애물을 회피**하는 능력을 추가하는 것이 목표였다. 진행 흐름:

1. **환경 신설** — `CarRacingObstacles-v0`: CarRacing-v3에 Box2D 정적 장애물을 뿌리고, 픽셀 관측에 보이게 그려 넣고, 충돌에 패널티를 준다.
2. **장애물 회피 학습** — 베이스 best 모델에서 파인튜닝 (round 1).
3. **약점 진단** — 회원의 관찰("급커브에서 이탈")을 출발점으로, **계측 평가(instrumented eval)** 로 실제 실패 원인을 규명.
4. **개선 시도** — 코너 감속 패널티(0.5→0.2), 장애물 크기 랜덤화, (예정) 행동 노이즈 축소.

**핵심 결론(미리)**: 장애물 회피 자체는 학습됐다(동일 시드 before/after seed42 **-68 → +323**). 그러나 회원이 제기한 "급커브 이탈"은 **이 모델의 주 실패 원인이 아니었다.** 계측 결과 주 실패는 **장애물 충돌**이며, 그 충돌의 상당수는 **정책의 무능이 아니라 평가 시 행동 샘플링 노이즈**로 보였다(같은 트랙을 결정론으로 돌리면 충돌 0). 이에 따라 추가한 코너 감속 패널티(0.5)는 **엉뚱한 문제를 겨냥**해 reward만 깎았고(clean 365), 0.2로 낮추자 reward가 회복됐다(clean **415**). 마지막으로 행동 노이즈를 직접 축소(`ent_coef` 0.01→0.005)해 가설을 검증했더니, **`entropy`는 1.3→0.91로 내렸지만 clean 성능은 415→418로 무변화(Min은 오히려 악화).** → 충돌의 일부는 노이즈로 제거되지 않는 **구조적 실패**이며, **2-action 장애물 task의 clean 천장은 ~415로 확정**된다(시도한 세 레버 모두 못 넘음). 끝으로 **3-action 대조군**(native gas/brake, §8)을 동일 task·하이퍼파라미터로 학습했더니 `ent_coef 0.01`에선 std 발산(2.0→5.33), 공정 재튜닝(`ent 0.003`) 후에도 **clean 229로 2-action(415)의 55%** — 독립 gas/brake는 이 task에 이득 없이 최적화만 어렵게 했고, **2-action의 ActionWrapper가 유용한 inductive bias**임이 확인됐다.

---

## 2. 환경 설계 — `CarRacingObstacles-v0` (`src/car_racing_obstacles.py`)

CarRacing의 `CarRacing` 클래스를 상속한 변형 환경. 등록 id `CarRacingObstacles-v0`, `max_episode_steps=1000`.

### 2.1 장애물 배치
- 트랙 타일 위에 **`n_obstacles=10`개의 Box2D 정적 바디**를 배치.
- **재현성**: 배치 타일·위치·크기 모두 `self.np_random`(reset seed로 시드됨)에서 샘플 → **같은 시드 = 같은 장애물 배치**.
- **양끝 클리어**: 트랙은 **루프**라서 끝 타일이 시작선 바로 뒤에 온다. `range(start_clear_tiles, n_tiles - start_clear_tiles)`로 **양끝 모두** 비워 차 위에 스폰되는 것을 방지(`start_clear_tiles=30`).
- **최소 간격** `min_tile_gap=12` 타일.
- **통과 틈 보장**: 장애물 바깥 모서리를 도로 반폭의 ~80% 이내로 제한.

### 2.2 픽셀 가시성 (중요)
에이전트 입력은 픽셀(96×96 grayscale ×4)이므로, **장애물이 관측에 그려져야** 회피를 학습할 수 있다. 각 장애물을 **흰색 `(255,255,255)` 사각형으로 `road_poly`에 추가** → grayscale에서 도로(~102)·잔디(~162) 대비 **255로 고대비**. (`reset()`에서 장애물 생성 후 관측을 다시 렌더해 프레임 0부터 보이게 함.)

### 2.3 크기 랜덤화
- 장애물마다 `size_frac = uniform(obstacle_size_min, obstacle_size_max)`를 샘플, 한 변 = `size_frac × TRACK_WIDTH / 2`.
- **기하 관계**: 장애물 전체폭 = `size_frac × TRACK_WIDTH`, 도로 전체폭 = `2 × TRACK_WIDTH`. 즉 **`size_frac = 2.0`이 도로폭과 같고, `> 2.0`이면 도로보다 크다.**
- 기본 범위 **0.25–0.6** (도로폭의 ~12~30% → 항상 통과 틈 존재). `min==max`로 고정 크기 가능.
- **대형 변형(설계 완료)**: `2.0–3.0`으로 주면 **도로를 가로막는 벽** → 통과하려면 잔디로 우회해야 함(별도 난이도, 4절 참조).

### 2.4 충돌 검출 및 패널티
- `ObstacleFrictionDetector`(부모 `FrictionDetector` 상속)가 차–장애물 접촉을 감지.
- **새 접촉이 시작된 스텝마다 `obstacle_penalty=15` 차감** (패널티 전용 — 에피소드는 계속, 차는 물리적으로 튕기며 감속). 한 스텝에 hull+바퀴 동시 접촉이 잡혀도 **스텝당 1회만** 차감.

---

## 3. 환경 구축 시 해결한 버그 (엔지니어링 핵심)

### 3.1 SEGFAULT — 본체 파괴 중 contact 콜백 재진입 (Linux box2d-py)
- **증상**: 64개 env가 첫 에피소드를 끝내고 autoreset하는 순간 전원 segfault(`pygame_parachute`).
- **원인**: 차가 장애물에 닿은 채 에피소드가 끝나면, reset의 `_destroy()`에서 `DestroyBody(장애물)` 도중 Box2D가 **EndContact 콜백을 발사** → 반쯤 파괴된 바디의 `userData`를 Python에서 접근 → segfault. (macOS는 우연히 살아남아 로컬 테스트가 못 잡음.)
- **수정**: `_destroy()`에서 바디 파괴 **전에 contact listener를 분리**(`self.world.contactListener = None`). 리스너는 `CarRacing.reset()`이 직후 재설치하므로 게임플레이 영향 없음.

### 3.2 차 위에 장애물 스폰 (loop-spawn)
- **원인**: 트랙이 루프라 끝 타일이 시작선 뒤 → 한쪽 끝만 비우면 차 스폰 위치에 장애물.
- **수정**: 배치 후보를 `range(start_clear_tiles, n_tiles - start_clear_tiles)`로 **양끝 제외**.

### 3.3 로깅 무력화 — 벡터 env info 포맷 (셰이핑 지표 전부 미기록)
- **증상**: 학습 TB 로그에 `penalties/mean_accel_turn`, `driving/mean_percent_off_track`, `driving/mean_obstacle_hits`, `rewards/mean_velocity`가 **하나도 안 찍힘** (charts/losses만).
- **원인**: gymnasium 1.x 벡터 env의 `infos`는 **`{키: (num_envs,) 배열}` dict**인데, 코드가 `infos.get(i)`(정수 인덱스)로 읽어 **항상 `None`** → 셰이핑 지표 수집 블록 전체 스킵.
- **수정**: `infos[key][i]` 배열 인덱싱으로 교체(probe로 `'_final_info' 없음` + 배열 포맷 확정). **검증**: 회전+가속 액션에서 `accel_turn` 10.8 정상 수집. → 이후 실험부터 지표 정상 기록.

> 이 세 버그(특히 3.3)는 "진단"의 전제였다. 3.3을 고치기 전 run들의 로그로는 코너 감속·이탈·충돌 추세를 알 수 없었고, 실제 진단은 **모델을 직접 계측**해서 했다(5절).

---

## 4. 장애물 크기 실험 (소형 vs 대형)

| 실험 | size_frac | 도로 대비 | 과제 성격 |
|---|---|---|---|
| **소형(기본)** | 0.25–0.6 | 12~30% | 틈으로 weaving 회피 (통과 틈 보장) |
| **대형(설계 완료)** | 2.0–3.0 | 100~150% | **도로 차단 → 잔디 우회 필수** |

- 검증: 소형은 장애물 전체폭 1.7–3.9 vs 도로 13.3 (전부 작음), 대형은 13.4–19.6 (전부 도로보다 큼). 시드 재현성·통과틈·도로차단 모두 확인.
- **대형의 보상 설계 긴장**: 도로를 막으면 **off-road 우회만이 유일 통과법**인데, 셰이핑은 off-track에 패널티(+속도보상 차단)를 줘 **유일한 통과법을 벌준다.** → 대형 실험은 `--track-penalty`를 낮춰야 가능(예 0.2). CLI로 노출함.
- 대형 실험(B)은 **학습 미수행**(설계·검증만). 본 보고서 결과는 모두 **소형(0.25–0.6)** 기준.

---

## 5. 진단 — "급커브 이탈"은 오진이었다 (계측 평가)

회원 관찰: *"급커브에서 액셀을 밟는지 도로에서 크게 벗어난다."* → 코너 감속 패널티를 추가했으나, **계측으로 실제 실패 원인을 측정**한 결과 가설이 틀렸다.

### 5.1 코너 감속 패널티 구현
- 발견: `acceleration_while_turning_penalty_weight`가 config에서 0이었을 뿐 아니라 **wrapper에 구현 자체가 없었다**(= 신호 부재).
- 구현: `step_penalty = weight × gas × |steering|`. **직진 가속(|steer|≈0)은 무비용**, 조향하며 밟을 때만 벌점. 검증: 직진 풀가스 30스텝 패널티 0.000 / 풀조향+풀가스 30스텝 15.000.

### 5.2 계측 결과 — 메커니즘은 작동, 그러나 결과는 무관
동일 env(소형)·동일 시드, 20ep 샘플링, **코너 가속 지표 `gas×|steer|`**:

| 모델 | clean reward | 코너가속 | off-track%* | 충돌 |
|---|---|---|---|---|
| obs_small (패널티 0.5) | 372 | **0.151** | 29.0 | 2.5 |
| round1 (패널티 없음) | 455 | 0.263 | 31.6 | 1.9 |

- ✅ **코너 가속 42% 감소(0.263→0.151)** — 패널티는 의도한 직접 목표를 달성.
- ⚠️ 그러나 **off-track은 거의 불변(-2.6%p), reward는 오히려 하락(455→372).** 패널티가 정책을 소심하게 만들어 reward만 깎음.
- (`*` off-track% 계측은 신뢰도 낮음 — "전 바퀴 도로 밖" 기준이 타일 경계/정체를 과대계상. 절대값은 버리고 추세만 참고.)

### 5.3 실패 원인은 충돌, 그것도 대부분 "샘플링 노이즈"
obs_small의 에피소드별 분석(시드별):

| seed | reward | off% | 충돌 |
|---|---|---|---|
| 53 (최악) | **-36** | **0.0** | **6** |
| 57 | 127 | 0.0 | 6 |
| 52 (최고) | 760 | 12.5 | **0** |
| 56 | 757 | 13.1 | 0 |

- **망한 판 = 도로 위에 잘 있는데(off 0%) 장애물 다발 충돌.** 잘한 판 = 충돌 0, off-track은 오히려 높음(회피하느라 가장자리로 weaving). → **off-track은 실패가 아니라 회피 성공의 부산물.**
- **결정적**: 최악 seed 53을 **결정론(mean action)으로 돌리니 -36 → 486, 충돌 0.** → 그 충돌들은 정책 무능이 아니라 **평가 시 가우시안 행동 노이즈**가 가끔 장애물로 틀어버린 것.

### 5.4 결론: 코너 패널티는 비(非)레버 — 0.5→0.2로 reward 회복
동일 설정(소형, 50ep 샘플링)에서 패널티만 바꿔 비교:

| 모델 | 패널티 | Mean | Median | Q1 / Q3 | Min / Max |
|---|---|---|---|---|---|
| obs_small | **0.5** | 365 | 325 | 221 / 482 | -37 / 837 |
| **obs_small_p02** | **0.2** | **415** | **422** | **285 / 553** | -65 / 842 |

→ **패널티를 낮추니 평균·중앙값·사분위 전부 개선(median +97).** 코너 패널티는 reward만 깎았을 뿐 실패 원인(충돌)과 무관했음이 확정됨.

---

## 6. 실험 결과 종합

### 6.1 학습 곡선 — 이 태스크는 전반부에 정점 찍고 정체
| run | 패널티 / 크기 | 스텝 | best(shaped) @ step | 비고 |
|---|---|---|---|---|
| round1 (`ppo_2action_obstacles`) | 0 / 고정 0.4 | 10M | **474.8 @ 4.9M** | 52→360→431 상승 후 정체 |
| obs_small | 0.5 / 랜덤 0.25–0.6 | 6M | **447.4 @ 1.64M** | 정점 매우 이름 |
| **obs_small_p02** | 0.2 / 랜덤 0.25–0.6 | 6M | **465.4 @ 3.28M** | 정점 후 6M엔 453로 퇴보 |

→ **세 run 모두 전반부(1.6M / 3.3M / 4.9M)에 정점 후 정체/퇴보.** (베이스 20M도 9.8M 정점) → **순수 스텝 증가는 ROI가 낮고, 다음 실험은 4~5M이면 충분.**

### 6.2 Clean 평가 (shaping 제외, 50ep 샘플링, seed 42)
| 모델 | Mean | Median | Min / Max | 비고 |
|---|---|---|---|---|
| round1 (고정 0.4) | 331 | 322 | -99 / 810 | 고정 0.4 분포 |
| obs_small (0.5) | 365 | 325 | -37 / 837 | 랜덤 0.25–0.6 |
| **obs_small_p02 (0.2)** | **415** | **422** | -65 / 842 | 랜덤 0.25–0.6, **현 최고** |

> 주의: round1은 고정 0.4로 학습/평가, obs_small·p02는 랜덤 0.25–0.6 → **크기 분포가 달라 직접 비교는 obs_small vs p02만** 깨끗하다. round1 수치는 분포가 다른 참고값.

### 6.3 정성 (before/after, 동일 시드)
| | 학습 전(베이스) | 학습 후(round1) |
|---|---|---|
| seed42 | **-68** (장애물에 정면 충돌, step300에 reward~1.9) | **+323** (깨끗이 통과) |
| seed43 | 237 | **+707** (장애물 옆으로 비켜가며 스키드마크) |

→ 영상: `videos_obstacles_before/`(충돌), `videos_obstacles_after/`·`videos_obs_small/`(회피), `videos_diag/`(진단용).

---

## 7. 마지막 실험 — 행동 노이즈 축소(`ent_coef`)는 비(非)레버였다

진단(5절)에 따르면 이 모델의 천장은 "학습 부족"이 아니라 **장애물 충돌**이고, 그 충돌의 상당수가 **평가 시 샘플링 노이즈**로 보였다(결정론에서 충돌이 거의 사라짐). 그렇다면 마지막 레버는 코너 패널티가 아니라 **행동 노이즈 축소**여야 한다 — `ent_coef`를 낮춰 정책을 더 결정론적으로 만들면 샘플링 충돌이 줄어 실성능이 오를 것이라는 가설.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_ppo_2action_obstacles \
  --ent-coef 0.005 --accel-turn-weight 0.2 \
  --obstacle-size-min 0.25 --obstacle-size-max 0.6 \
  --steps 4000000 \
  --save-dir ./models/obs_p02_lowent --log-dir ./logs/obs_p02_lowent
```

### 7.1 결과 — 레버는 움직였으나 성능은 안 따라옴
학습 4M 완료(clean exit). 학습 지표: **`losses/entropy` 1.3 → 0.91로 하락** → 노이즈 축소 레버 자체는 의도대로 작동. 그러나 50ep clean(seed 42, 동일 소형 분포):

| 모델 | ent_coef | Mean | Median | Q1 / Q3 | **Min(충돌 바닥)** | Max |
|---|---|---|---|---|---|---|
| obs_small_p02 | 0.01 | 415 | **422** | 285 / 553 | **-65** | 842 |
| **obs_p02_lowent** | **0.005** | **418** | 396 | 252 / 591 | **-112** | 811 |

- **Mean 415 → 418**: 사실상 무변화(편차 ±231 안). **Median 422 → 396**으로 오히려 ↓.
- **Min(충돌 바닥) -65 → -112로 악화.** 최악 판(ep32 -112)은 entropy를 낮췄는데도 살아남음.

### 7.2 해석 — 진단의 수정: 남은 충돌은 "구조적"이지 단순 노이즈가 아니다
5.3절에서 "충돌은 대부분 샘플링 노이즈"로 보였으나(seed53 결정론 486 vs 샘플링 -36), **노이즈를 실제로 줄여봤더니(entropy 1.3→0.91) 충돌이 줄지 않았다.** Min은 오히려 악화. 즉 **샘플링 노이즈는 충돌의 *충분조건* 중 하나였을 뿐, 제거해도 사라지지 않는 구조적 실패 케이스가 따로 남아 있다** — 정책이 특정 장애물 배치를 아직 못 푸는 것이다. (`ent_coef`를 더 낮추면 탐험까지 줄어 학습 자체가 나빠질 수 있어 더 내리는 것은 권장 안 함.)

### 7.3 판정 — task 천장(~415) 확정
- **가설 기각.** 행동 노이즈 축소는 `losses/entropy`는 내렸지만 clean 성능(Mean·Median·Min)을 못 올렸다.
- **2-action 장애물 task의 clean 천장은 ~415 (Mean)으로 확정.** 코너 패널티(5절)·스텝 증가(6.1절)·노이즈 축소(7절) — 시도한 세 레버 모두 이 천장을 넘지 못했다.
- **최종 채택 모델: `obs_small_p02`(ent_coef 0.01, clean 415/422).** lowent는 동급이나 Median·Min에서 미세 열위라 채택하지 않음.

---

## 8. 3-action 대조 실험 — 2 vs 3 action (native gas/brake)

### 8.1 동기와 절차
2-action(throttle 1축, ActionWrapper)이 장애물 task에서 정착한 천장(clean ~415)이 **action 파라미터화** 탓인지 확인하기 위해, **native 3-action `[steer, gas, brake]`** 정책으로 동일 task를 학습했다. 환경·보상·하이퍼파라미터는 2-action과 **바이트 단위로 동일**(n_obstacles=10, penalty=15, size 0.25–0.6, LR 1e-4 cosine, target_kl 0.03)하고 **유일한 차이는 ActionWrapper 미적용**(정책이 3D 네이티브 출력). 3-action 베이스 `evaluated641`(무장애물 base에서 ~675)에서 fine-tune. 전용 스크립트 `scripts/train_ppo_3action_obstacles.py`·`scripts/evaluate_agent_3action_obstacles.py`.

### 8.2 B (ent 0.01) — 발산
2-action과 동일한 `ent_coef 0.01`로 학습하자 **entropy(=std)가 2.0→5.33으로 폭증**, reward가 피크 325(@1.1M) 후 156으로 붕괴. TB 곡선 진단(`scripts/dump_tb_scalars.py`): `ent_coef×entropy`(0.053)가 `policy_loss`(0.008)를 **6배 압도** → 옵티마이저가 보상 대신 std 키우기로 폭주. `obstacle_hits`는 4.8→1~2로 줄어(회피는 학습) 실패 원인은 충돌이 아니라 **std 발산**. → **B는 불공정**(2D에 맞춘 ent가 3D엔 과대).

### 8.3 B2 (ent 0.003) — 발산 차단, 그러나 낮은 천장
`ent_coef`를 0.003으로 낮추자(ent항/policy항 ≈ 1:1) entropy가 ~2–3.3에서 안정, 발산 소멸. 피크도 1.1M→**3.28M으로 이동**(2-action p02와 동일 시점) — 저엔트로피가 조기 발산 대신 후반까지 개선을 허용. clean 50ep(동일 분포·seed 42):

| 지표 | 2-action p02 (ent0.01) | **3-action B2 (ent0.003)** | 3/2 비율 |
|---|---|---|---|
| best shaped @step | 465 @3.28M | 334 @3.28M | 72% |
| **clean Mean** | **415** | **229** | **55%** |
| clean Median | 422 | 245 | 58% |
| Q1 / Q3 | 285 / 553 | 179 / 317 | — |
| **Min (충돌 바닥)** | −65 | **−171** | 악화 |
| Max | 842 | 730 | — |

→ **발산을 고쳐 공정(오히려 3-action에 유리한 ent 튜닝)하게 비교해도 3-action은 2-action의 ~55%.** Mean·Median 모두 큰 폭 하회, **Min은 오히려 악화**(−171, 깊은 충돌 판 다수).

### 8.4 왜 — 코드 근거 (멀티에이전트 반증 검증 완료)
동일 task·하이퍼파라미터에서 3-action만 낮은 건 **native 독립 gas/brake action space** 때문이며, 코드에 대조·반증했다:

1. **gas+brake 동시 입력이 실재하는 퇴화영역** (검증 holds=true). native step은 `car.gas(action[1])`·`car.brake(action[2])`를 독립 적용(`car_racing.py:545-546`), 물리엔진에서 같은 `w.omega`를 한 틱에 gas가 올리고 brake가 깎는 **상쇄 낭비 구간**(`car_dynamics.py:199-216`). 2-action ActionWrapper는 단일 throttle을 `gas=max(0,t)/brake=max(0,−t)`로 분해해 이 구간을 **구조적으로 제거**(`env_wrappers.py:140-141`).
2. **무방비 brake 채널.** `gas()`는 [0,1] 클립 + 스텝당 +0.1 램프로 보호되나 `brake()`는 **클립·램프 없이 즉발**(`car_dynamics.py:146-160`). 노이즈 큰 brake가 즉시 속도를 깎아, 유일한 양의 셰이핑인 `velocity=speed*0.003`(survival=0)을 직접 잠식.
3. **탐색 부피 ~2.9배 + 합산 엔트로피.** 차원이 하나 더라 entropy/log_prob이 brake축까지 합산(`ppo_agent_2.py:126-127`) → 같은 ent_coef에서 std가 발산하기 쉬움(B에서 실증).

### 8.5 결론과 한계
- **결론**: **native 3-action은 이 장애물 task에서 구조적으로 불리**하다. 발산을 막는 공정한(오히려 관대한) 튜닝 후에도 clean 229 vs 415 — ActionWrapper의 signed-throttle이 *퇴화영역 제거 + 탐색 축소*라는 **유용한 inductive bias**임이 확인된다. 이 프로젝트가 처음부터 2-action을 택한 근거를 사후 정당화한다.
- **한계(정직하게)**: 3-action의 clean 천장을 2-action처럼 여러 lever로 **확정한 것은 아니다**(B2 단일 공정 run). 정확한 진술은 "3-action이 415를 못 넘는다"가 아니라 **"동일·관대 setting에서 3-action은 415 도달 전에 발산(ent0.01)하거나 ~229에서 정체(ent0.003)한다"**. 잔여 탐색 여지(brake축만 std 축소, gas/brake squash, 더 긴 학습)는 있으나 ROI는 낮다고 판단.

---

## 9. 핵심 교훈

1. **픽셀 입력 에이전트는 장애물을 관측에 "그려" 넣어야 한다** — 물리적으로만 존재해선 학습 불가.
2. **Linux box2d-py는 본체 파괴 중 contact 콜백에서 segfault** — 파괴 전 리스너 분리. (macOS가 가려서 로컬 테스트로 못 잡음.)
3. **보상 셰이핑 항은 조용한 no-op일 수 있다 — 실제로 발화하는지 검증하라.** 코너 패널티(미구현)와 (베이스 보고서의) 속도 보상(`info['speed']`=0)이 둘 다 무발화였다.
4. **벡터 env 로깅은 `infos[key][i]`(dict-of-arrays)로 읽어야** 한다 — `infos.get(i)`(정수키)는 무음 실패.
5. **셰이핑을 추가하기 전에 실패 모드를 측정하라.** 코너 패널티는 오진이었다 — 계측 평가(코너가속·off%·충돌/에피소드)가 진짜 원인(충돌·노이즈)을 드러냈다.
6. **샘플링 vs 결정론 평가는 std가 클 때 크게 다르다** — 샘플링 노이즈가 장애물 충돌을 유발. (단, 평가 표준은 일관성을 위해 기존처럼 샘플링 유지.)
7. **그러나 "노이즈가 원인"은 가설로 끝내지 말고 직접 검증하라.** `ent_coef`로 노이즈를 실제로 줄였더니(`entropy` 1.3→0.91) 충돌이 안 줄었다 — 충돌의 일부는 노이즈로 제거 안 되는 **구조적 실패**였다. 진단은 한 번의 결정론 평가로 단정하지 말고 **레버를 움직여 반증을 시도**해야 한다.
8. **이 태스크는 전반부에 정점** — 향후 run은 짧게(4~5M). 그리고 코너 패널티·스텝 증가·노이즈 축소 **세 레버 모두 clean ~415 천장을 못 넘음** → 다음 개선은 하이퍼파라미터가 아니라 **표현/아키텍처(예: 명시적 장애물 채널, recurrent, 더 큰 CNN)** 쪽이어야 함을 시사.
9. **action 파라미터화는 공짜가 아니다 — ActionWrapper(2-action)가 inductive bias로 작동(§8).** native 3-action은 동일 task·관대 튜닝에도 clean 229(2-action 415의 55%): 독립 gas/brake가 (a) 동시 입력 퇴화영역, (b) 무방비 brake 채널, (c) ~2.9배 탐색 부피를 만들어 std 발산/저성능을 유발. 자유도를 늘리기 전에 **그 자유도가 task에 이득인지** 따져라 — brake 독립 제어는 이 트랙에서 이득 없이 최적화만 어렵게 했다.
10. **하이퍼파라미터는 차원에 따라 재튜닝하라.** 2D에서 안정적이던 `ent_coef 0.01`이 3D에선 entropy를 발산시켰다(2.0→5.33). 같은 설정을 차원만 바꿔 재사용하면 불공정/오결론 — `ent항/policy항` 비율로 균형을 확인하라.

---

## 10. 구현 산출물 (커밋 안 함)

| 파일 | 상태 | 내용 |
|---|---|---|
| `src/car_racing_obstacles.py` | 신규 | `CarRacingObstacles-v0` 환경 (랜덤 크기·픽셀 렌더·충돌 패널티·segfault·loop-spawn 픽스) |
| `scripts/train_ppo_2action_obstacles.py` | 신규 | 장애물 학습 fork (코너 패널티 구현, 로깅 픽스, CLI 다수) |
| `scripts/record_video.py` | 수정 | `--obstacles`, `--n-obstacles`, `--obstacle-size-min/max` 추가 |
| `scripts/evaluate_agent_2action.py` | 분리(베이스) | 장애물 코드 제거 → **베이스(CarRacing-v3) 전용** clean 평가기 |
| `scripts/evaluate_agent_2action_obstacles.py` | 신규 | **장애물 전용** 평가기 — `CarRacingObstacles-v0` 항상 ON, ActionWrapper 적용. `--obstacles`는 하위호환 no-op |
| `scripts/train_ppo_3action_obstacles.py` | 신규 | **3-action 대조군** 학습 (ActionWrapper 미적용, native 3D, evaluated641서 fine-tune) — §8 |
| `scripts/evaluate_agent_3action_obstacles.py` | 신규 | 3-action 장애물 평가기 (ActionWrapper 미적용) |
| `scripts/dump_tb_scalars.py` | 신규 | TB 이벤트→텍스트 표 + PNG 덤프 (다중 run 오버레이). 3-action 발산 진단(§8.2)에 사용 |

**학습 스크립트 CLI 오버라이드**: `--checkpoint --steps --seed --log-dir --save-dir --n-obstacles --obstacle-size-min --obstacle-size-max --track-penalty --accel-turn-weight --ent-coef`

---

## 11. 부록 — 재현 커맨드 (로컬 `racing` env / 원격 `teamB_env`)

```bash
# 학습 (장애물 회피 fine-tune, 소형 랜덤, 코너 패널티 0.2)
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_ppo_2action_obstacles \
  --accel-turn-weight 0.2 --obstacle-size-min 0.25 --obstacle-size-max 0.6 \
  --save-dir ./models/obs_p02 --log-dir ./logs/obs_p02

# 평가 (clean, 학습과 동일 크기 분포로! 장애물 전용 스크립트)
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python -m scripts.evaluate_agent_2action_obstacles \
  --model models/obs_p02/best_model.pth \
  --obstacle-size-min 0.25 --obstacle-size-max 0.6 --episodes 50 --seed 42

# 영상 (로컬)
python scripts/record_video.py --model models/obs_p02/best_model.pth \
  --obstacles --obstacle-size-min 0.25 --obstacle-size-max 0.6 --episodes 5

# 대형 장애물(도로보다 큼) 실험 — track_penalty 완화 필요
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_ppo_2action_obstacles \
  --obstacle-size-min 2.0 --obstacle-size-max 3.0 --track-penalty 0.2 \
  --save-dir ./models/obs_big --log-dir ./logs/obs_big
```
