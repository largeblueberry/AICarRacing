"""Build COMBINED_REPORT.md = REPORT.md (phase 1) + bridge + REPORT_obstacles.md
(phase 2), with figures/frames inserted near relevant sections and a figure/video
appendix. Tolerant: if an anchor is missing, the figure falls back to the appendix
so nothing is silently dropped. Then convert to docx/pdf via pandoc/soffice (separately)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = "report_assets"

def img(path, caption, width="80%"):
    return f'\n![{caption}]({A}/{path}){{width={width}}}\n'

def pair(left, lcap, right, rcap):
    # side-by-side via a 2-col pipe table with images in cells
    return (
        f'\n| ![{lcap}]({A}/{left}){{width=98%}} | ![{rcap}]({A}/{right}){{width=98%}} |\n'
        f'|:--:|:--:|\n'
        f'| **{lcap}** | **{rcap}** |\n'
    )

# (target_file, anchor_substring, block_to_insert_after_anchor_line)
INSERTIONS = [
    ("REPORT.md",
     "**남은 약점**: bimodal 분포",
     img("frame_base_r838.png", "그림. 베이스 2-action 최종 모델(9.8M) 주행 — seed46, 보상 838 (clean 평균 667의 상단)", "62%")),
    ("REPORT.md",
     "(`videos_obstacles_after` 5개 에피소드",
     pair("frame_before_seed42_r-68.png", "Before (장애물 학습 전): seed42 step240 — 흰 장애물로 정면 돌진, 누적 −68",
          "frame_after_seed42_r323.png", "After (장애물 학습 후): seed42 step240 — 도로 위 정상 주행·회피, 누적 +323")),
    ("REPORT_obstacles.md",
     "→ **패널티를 낮추니 평균·중앙값·사분위 전부 개선(median +97).**",
     img("fig_corner_lever.png", "그림. 코너 감속 패널티는 비(非)레버 — 0.5→0.2로 낮추니 clean 보상 회복(365→415)", "62%")),
    ("REPORT_obstacles.md",
     "### 6.2 Clean 평가 (shaping 제외, 50ep 샘플링, seed 42)",
     img("fig_clean_progression.png", "그림. 장애물 task clean 성능 진행 — 2-action 천장 ~415 vs 3-action 229 (회색 점선=무장애물 베이스 667)", "85%")),
    ("REPORT_obstacles.md",
     "→ 영상: `videos_obstacles_before/`",
     img("frame_after_seed43_r707.png", "그림. 장애물 회피(weaving) — seed43, 보상 707: 곡선 구간에서 장애물 사이로 우회 조향", "60%")),
    ("REPORT_obstacles.md",
     "### 8.2 B (ent 0.01) — 발산",
     img("fig_entropy_divergence.png", "그림. Entropy(=std) 곡선 — 3-action(ent0.01)은 2.0→5.3 발산, 2-action 및 3-action(ent0.003)은 안정", "85%")
     + img("fig_reward_curve.png", "그림. shaped 학습 곡선 — 2-action ~450 유지 vs 3-action 피크 후 정체/추락", "85%")),
    ("REPORT_obstacles.md",
     "→ **발산을 고쳐 공정(오히려 3-action에 유리한 ent 튜닝)하게 비교해도 3-action은 2-action의 ~55%.**",
     img("fig_2v3_grouped.png", "그림. 2 vs 3 action (동일 장애물 task, clean 50ep) — 3-action은 2-action의 ~55%, 충돌 바닥(Min)도 악화", "70%")),
]

def load(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()

reports = {n: load(n) for n in ("REPORT.md", "REPORT_obstacles.md")}
leftover = []
for fname, anchor, block in INSERTIONS:
    text = reports[fname]
    idx = text.find(anchor)
    if idx == -1:
        print(f"  [WARN] anchor not found in {fname}: {anchor[:40]!r} -> appendix")
        leftover.append(block)
        continue
    # insert after the end of the anchor's line
    line_end = text.find("\n", idx)
    if line_end == -1:
        line_end = len(text)
    reports[fname] = text[:line_end+1] + block + text[line_end+1:]

TITLE = """---
title: "AICarRacing 종합 기술 보고서"
subtitle: "2-Action PPO: 붕괴 복구 · 장애물 회피 · 2 vs 3 Action 대조"
author: "팀 B"
date: "2026-06-15"
---

# AICarRacing 종합 기술 보고서 {-}

**2-Action PPO 에이전트: 붕괴 복구 → 장애물 회피 확장 → 2 vs 3 Action 대조 실험**

팀 B · 2026-06-15 · 환경: Gymnasium CarRacing-v3 / CarRacingObstacles-v0

> 본 문서는 채점 루브릭 순서를 따른다. **Part 0**(실행 환경·설치 / 문제·State·Reward 정의 / PPO 이론) → **Part I**(붕괴 복구 + 베이스 스케일업 + 장애물 round-1) → **Part II**(장애물 심화: 진단·크기·엔트로피·**2 vs 3 action 대조**) → 부록(그림·GIF). 표는 원문 그대로, 그림·주행 프레임은 관련 절에 삽입했다.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Part 0 — 실행 환경 · 문제 정의(State/Reward) · 알고리즘

"""

# --- Part 0 front matter (rubric-required): env/install, problem, state, reward, PPO theory ---
def load_front(name):
    with open(os.path.join(ROOT, "report_front", name), encoding="utf-8") as f:
        return f.read().strip()

PROBLEM_LEADIN = """

## 1. 문제 및 환경 정의

본 프로젝트가 선택한 환경은 Gymnasium **`CarRacing-v3`**(연속 제어, top-down 레이싱)이며, 이를 상속해 **무작위 정적 장애물 회피** 과제(`CarRacingObstacles-v0`)를 직접 정의해 확장했다. 목표는 (1) 픽셀 관측만으로 트랙을 빠르고 안정적으로 주행하는 2-action PPO 에이전트를 학습하고, (2) 도로 위 장애물을 회피하도록 일반화하며, (3) 2-action(`[steering, throttle]`)과 native 3-action(`[steering, gas, brake]`) 행동 파라미터화를 동일 조건에서 비교하는 것이다. 강화학습의 핵심인 **상태(State)와 보상(Reward)** 정의를 아래에 상세히 기술한다.

"""

FRONT = (
    load_front("env.md") + "\n\n"
    + "```{=openxml}\n<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>\n```\n\n"
    + PROBLEM_LEADIN
    + load_front("state.md") + "\n\n"
    + load_front("reward.md") + "\n\n"
    + "```{=openxml}\n<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>\n```\n\n"
    + load_front("ppo.md") + "\n\n"
    + "```{=openxml}\n<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>\n```\n\n"
    + "# Part I — 베이스 복구 · 스케일업 · 장애물 round-1\n\n"
)

BRIDGE = """

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Part II — 장애물 심화 실험 (진단 · 크기 · 엔트로피 · 2 vs 3 action)

> **편집자 주(브리지)**: Part I은 프로젝트 phase-1(베이스 2-action 붕괴 복구와 장애물 round-1) 시점의 기록으로, 작성 당시 코너 감속(accel-turn) 패널티는 *구현만 완료·미학습* 상태였다. Part II는 그 이후 **실제로 수행된** 장애물 심화 실험을 다룬다 — 코너 패널티(0.5→0.2)의 실험적 기각, 장애물 크기 랜덤화, 행동 노이즈(ent_coef) 축소, 그리고 native **3-action 대조군**. 따라서 Part II는 Part I의 §6(round-2 계획)을 **갱신·대체**한다.

"""

APPENDIX = """

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 부록 Z — 그림 · 영상(GIF) 인덱스

## Z.1 생성 그림 (figures)
모든 차트는 본문 실험 수치로 생성(`scripts/make_report_figs.py`). 파일은 `report_assets/`에 있다.

- `fig_entropy_divergence.png` — entropy 발산(3-action ent0.01) vs 안정(2-action / 3-action ent0.003)
- `fig_reward_curve.png` — shaped 학습 곡선 2 vs 3 action
- `fig_clean_progression.png` — 장애물 clean 성능 진행 + 3-action
- `fig_2v3_grouped.png` — 2 vs 3 action clean 비교(Mean/Median/Min)
- `fig_corner_lever.png` — 코너 패널티 0.5→0.2 효과

## Z.2 주행 프레임 (정지 컷)
"""

# Append leftover figures (anchors that were missing), if any.
if leftover:
    APPENDIX += "\n### (본문 미삽입 그림)\n" + "".join(leftover)

APPENDIX += (
    pair("frame_before_seed42_r-68.png", "Before — seed42, 누적 −68 (장애물 정면 충돌)",
         "frame_after_seed42_r323.png", "After — seed42, 누적 +323 (회피 성공)")
    + img("frame_after_seed43_r707.png", "장애물 회피 weaving — seed43, 보상 707", "55%")
    + img("frame_obs_small_r645.png", "랜덤 크기 장애물 회피 — obs_small seed44, 보상 645", "55%")
    + img("frame_base_r838.png", "베이스 무장애물 주행 — seed46, 보상 838", "55%")
)

APPENDIX += """

## Z.3 애니메이션 GIF (별도 첨부)
> docx/PDF는 애니메이션을 표시하지 못하므로(첫 프레임만 정지), 아래 GIF 파일을 **별도로 열어** 동작을 확인할 것. 파일은 `report_assets/`에 있다. 원본 mp4는 각 `videos_*/` 폴더.

| 장면 | GIF 파일 | 원본 mp4 |
|---|---|---|
| 베이스 주행 (r838) | `report_assets/gif_base_r838.gif` | `videos_2action_final/best_model_ep5_seed46_r838.mp4` |
| 장애물 학습 전 충돌 (seed42, −68) | `report_assets/gif_before_seed42_r-68.gif` | `videos_obstacles_before/best_model_ep1_seed42_r-68.mp4` |
| 장애물 회피 (seed42, +323) | `report_assets/gif_after_seed42_r323.gif` | `videos_obstacles_after/best_model_ep1_seed42_r323.mp4` |
| 장애물 회피 weaving (seed43, 707) | `report_assets/gif_after_seed43_r707.gif` | `videos_obstacles_after/best_model_ep2_seed43_r707.mp4` |
| 랜덤크기 회피 (seed44, 645) | `report_assets/gif_obs_small_r645.gif` | `videos_obs_small/best_model_ep3_seed44_r645.mp4` |
"""

combined = TITLE + FRONT + reports["REPORT.md"] + BRIDGE + reports["REPORT_obstacles.md"] + APPENDIX
out = os.path.join(ROOT, "COMBINED_REPORT.md")
with open(out, "w", encoding="utf-8") as f:
    f.write(combined)
print("Wrote", out, f"({len(combined)} chars)")
print("leftover figs to appendix:", len(leftover))
