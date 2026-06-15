# 작업 핸드오프 (이어서 진행용) — 2026-06-14

> 컨텍스트가 길어져 이 파일에 현재 상태/다음 단계를 박아둔다. 새 세션/압축 후 이 파일부터 읽고 이어서 진행.
> 로컬 env: conda `racing` (`~/anaconda3/bin/conda run -n racing python ...`). 원격: A100 conda `teamB_env`, `/home/data/teamB/AICarRacing`.

## 지금 당장 할 일 (다음 액션)
**A(ent_coef 0.005) 완료 → 가설 기각, task 천장(~415) 확정.** clean 50ep: Mean **418** / Median 396 / Min **-112** / Max 811 (p02 415/422/-65 대비 무변화~미세악화). entropy 1.3→0.91로 내렸으나 성능 안 따라옴 = 남은 충돌은 구조적. **REPORT_obstacles.md §7 작성 완료.**

**다음: B (3-action 대조군) — 미시작.** 아래 "B 실행법" 참조. (또는 여기서 마무리하고 두 보고서 합본/문서화로 종료 — 사용자 결정 대기.)

## 실험 현황
| 실험 | 모델 경로 | 패널티/설정 | clean(50ep) | 상태 |
|---|---|---|---|---|
| 2-action 베이스(장애물X) | `models/ppo_2action4/best_model.pth` | — | **667/745** | 완료(최종 베이스) |
| 장애물 round1 | `models/ppo_2action_obstacles/best_model.pth` | accel0, 고정0.4 | 331/322 | 완료 |
| obs_small | `models/obs_small/best_model.pth` | accel0.5, 랜덤0.25-0.6 | 365/325 | 완료 |
| **obs_p02** (현 2-action 최고) | `models/obs_small_p02/best_model.pth`(원격) | accel0.2, 랜덤 | **415/422** | 완료, best_step 3.28M |
| **A: obs_p02_lowent** | `models/obs_p02_lowent/best_model.pth` | accel0.2 + **ent_coef0.005**, 4M | **418/396 (Min -112)** | 완료 — 가설 기각(천장~415 확정) |
| **B: 3-action 대조군** | `models/obs_3action/best_model.pth` | accel0.2, ent**0.01**, 3D 네이티브 | shaped peak 325@1.1M | **완료-발산**. entropy 2.0→**5.33 폭증**(std 발산), reward 피크 325후 156으로 추락. obst_hits는 4.8→1~2로 줄어듦(회피는 배움). 원인=ent_coef 0.01이 3D엔 과대 |
| **B2: 3-action lowent** | `models/obs_3action_lowent/best_model.pth` | accel0.2, **ent0.003**, 3D | **229/245 (Min -171)** | **완료**. entropy ~2-3.3 안정(발산 차단 성공), best_step 3.28M shaped 334. clean 229 = 2-action 415의 **55%** → native 3-action 구조적 불리 확정 |

## B (3-action) 실행법 — 새 파일 2개 원격 업로드 후
파일: `scripts/train_ppo_3action_obstacles.py`, `scripts/evaluate_agent_3action_obstacles.py` (ActionWrapper 미적용, 3-action 베이스 evaluated641에서 fine-tune)
```bash
# 업로드 확인
python -m scripts.train_ppo_3action_obstacles --help 2>/dev/null | grep accel-turn-weight   # 1줄
# 학습 (4M)
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_ppo_3action_obstacles \
  --accel-turn-weight 0.2 --obstacle-size-min 0.25 --obstacle-size-max 0.6 \
  --steps 4000000 --save-dir ./models/obs_3action --log-dir ./logs/obs_3action
# 평가
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python -m scripts.evaluate_agent_3action_obstacles \
  --model models/obs_3action/best_model.pth --obstacles \
  --obstacle-size-min 0.25 --obstacle-size-max 0.6 --episodes 50 --seed 42
```
→ 비교: 2-action p02(415/422) vs 3-action. 질문 = "장애물 회피에 3D 독립 gas/brake가 유리한가?"

## 핵심 발견 (보존)
- **2-action 복구**: per-minibatch KL early-stop(`src/ppo_agent_2.py`) + LR 호출 버그(global_step 전달) + 죽은 속도보상(`car.hull.linearVelocity`) 수정 → clean 667. 베이스 20M은 9.8M서 정점.
- **장애물 실패 원인**: "급커브 이탈"은 오진. 실제는 **장애물 충돌**, 그것도 대부분 **샘플링 노이즈**(seed53 결정론 486 vs 샘플링 -36). off-track은 회피(weaving) 부산물.
- **코너 패널티는 비레버**: 0.5→reward만 깎음(365), 0.2→회복(415). 충돌과 무관.
- **task는 전반부(~3M) 정점 후 정체** → 향후 run 4M이면 충분.
- **로깅 버그 수정**: 벡터env infos는 `infos[key][i]`(dict-of-arrays)로 읽어야 함(`infos.get(i)` 무음실패). → 이후 run부터 `penalties/mean_accel_turn` 등 정상 기록.
- **3-action은 동일 ent_coef(0.01)에서 std 발산**: TB 곡선상 entropy 2.0→5.33, reward 피크 325(1.1M)후 156 추락. `ent_coef×entropy`가 `policy_loss`를 6배 압도 → std(특히 brake 채널) 폭주 → 주행 난조. obst_hits는 줄어듦(회피는 학습) → 실패는 충돌 아닌 std 발산. 2-action은 entropy~1.0 안정. **결론: B는 불공정(3D엔 ent 0.01 과대) → B2(ent0.003)로 재시도.** TB 진단은 `scripts/dump_tb_scalars.py <logdir>...`로.

## 보고서
- `REPORT.md` — 2-action 베이스 복구 보고서 (완료)
- `REPORT_obstacles.md` — 장애물 보고서. §7(ent_coef A)·**§8(2 vs 3 action 비교) 완료**. 섹션 재번호(교훈9/산출물10/부록11). **모든 실험·보고 완료.** 남은 선택: 두 보고서 합본 docx/PDF 문서화(원하면).

## 변경 파일 (로컬, 커밋 안 함)
신규: `src/car_racing_obstacles.py`, `src/ppo_agent_2.py`, `scripts/train_ppo_2action2.py`, `scripts/train_ppo_2action_obstacles.py`, `scripts/train_ppo_3action_obstacles.py`, `scripts/evaluate_agent_2action_obstacles.py`(2-action 장애물 전용 평가기, 신규), `scripts/evaluate_agent_3action_obstacles.py`, `scripts/record_video.py`, `scripts/dump_tb_scalars.py`(TB 곡선→텍스트표+PNG 덤프, 신규), `REPORT.md`, `REPORT_obstacles.md`
수정: `scripts/evaluate_agent_2action.py`(**베이스 전용으로 분리** — 장애물 코드는 `*_2action_obstacles.py`로 빠짐), `.gitignore`(/videos)

## 평가 스크립트 매핑 (분리 후)
| 라인 | 평가 스크립트 | env | 장애물 인자 |
|---|---|---|---|
| 2-action 베이스 | `evaluate_agent_2action.py` | CarRacing-v3 | 없음 |
| 2-action 장애물 | `evaluate_agent_2action_obstacles.py` | CarRacingObstacles-v0(항상) | `--n-obstacles --obstacle-size-min/max` (`--obstacles`는 no-op) |
| 3-action 장애물 | `evaluate_agent_3action_obstacles.py` | CarRacingObstacles-v0 | `--obstacles --obstacle-size-min/max` |

## 반복된 함정 (주의)
1. **원격 부분 업로드** → `unrecognized arguments`. 실행 전 항상 `--help | grep <인자>` 확인. (rsync 전체 동기화 권장)
2. **평가는 학습과 같은 `--obstacle-size-min/max`** (분포 일치).
3. **3-action은 새 스크립트 2개**(`*_3action_obstacles.py`) 사용 — 2-action 파일 아님.
