"""Generate report figures (charts) from the experiment numbers into report_assets/.
Self-contained: data is hardcoded from the eval/TB results in REPORT*.md."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Korean-capable font so Hangul labels don't render as tofu boxes.
for _f in ("AppleGothic", "Apple SD Gothic Neo", "Nanum Gothic"):
    try:
        matplotlib.rcParams["font.family"] = _f
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report_assets")
os.makedirs(OUT, exist_ok=True)

C2 = "#1f77b4"   # 2-action
C3 = "#d62728"   # 3-action (diverged)
C3b = "#ff7f0e"  # 3-action B2 (stabilized)

# ---------- TB curve data (from dump_tb_scalars output) ----------
# 2-action obstacle A (obs_p02_lowent, ent 0.005): step -> (reward, entropy)
A_step = [196608,360448,524288,688128,851968,1015808,1212416,1376256,1540096,1703936,1867776,2031616,2195456,2359296,2523136,2686976,2850816,3047424,3211264,3375104,3538944,3702784,3866624,4000000]
A_rew  = [329.5,406.5,409.3,356.3,424.9,407.0,382.5,390.2,442.3,428.4,404.8,460.0,490.6,461.6,444.1,427.1,484.2,478.4,486.6,449.7,417.2,422.5,460.6,401.7]
A_ent  = [1.727,1.335,1.316,1.556,1.224,1.427,1.541,1.008,1.325,1.127,1.185,1.258,1.030,1.219,1.069,1.189,1.011,1.140,0.984,0.946,1.057,1.192,0.971,0.959]

# 3-action obstacle B (ent 0.01, diverged)
B_step = [196608,360448,524288,688128,851968,1015808,1212416,1376256,1540096,1703936,1867776,2031616,2195456,2359296,2523136,2686976,2850816,3047424,3211264,3375104,3538944,3702784,3866624,4000000]
B_rew  = [98.1,160.8,147.9,196.3,242.7,293.4,258.6,281.7,207.2,211.9,197.1,174.8,217.9,193.1,200.0,190.3,214.0,167.9,170.6,167.8,168.9,141.8,175.2,156.3]
B_ent  = [2.988,2.917,3.143,3.360,2.841,3.708,4.529,2.861,5.225,3.680,4.417,4.487,3.458,5.815,3.123,5.625,3.227,4.083,4.664,4.887,3.680,5.552,3.467,5.326]

# 3-action obstacle B2 (ent 0.003, stabilized) - sparse points from rollout logs
B2_step = [1179648,1212416,2392064,2424832,3997696,4000000]
B2_rew  = [281.7,281.7,228.1,228.1,282.3,282.3]
B2_ent  = [2.092,3.662,2.272,3.767,1.934,3.325]

def Mstep(xs):
    return [x/1e6 for x in xs]

# ---------- Fig 1: entropy divergence ----------
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(Mstep(A_step), A_ent, color=C2, lw=2, marker="o", ms=3, label="2-action (ent 0.01-line)  — 안정 ~1.0")
ax.plot(Mstep(B_step), B_ent, color=C3, lw=2, marker="s", ms=3, label="3-action B (ent 0.01)  — 발산 → 5.3")
ax.plot(Mstep(B2_step), B2_ent, color=C3b, lw=2.4, marker="^", ms=6, label="3-action B2 (ent 0.003)  — 안정 ~2.5")
ax.axhspan(0.8, 3.6, color="green", alpha=0.05)
ax.set_xlabel("env step (M)")
ax.set_ylabel("entropy_loss (std proxy)")
ax.set_title("Entropy: 3-action(ent0.01) std 발산 vs 2-action / 3-action(ent0.003) 안정")
ax.grid(True, ls="--", lw=0.4, alpha=0.6)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_entropy_divergence.png"), dpi=130); plt.close(fig)

# ---------- Fig 2: shaped reward curve ----------
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(Mstep(A_step), A_rew, color=C2, lw=2, marker="o", ms=3, label="2-action (obs_p02_lowent)")
ax.plot(Mstep(B_step), B_rew, color=C3, lw=2, marker="s", ms=3, label="3-action B (ent0.01, 발산)")
ax.plot(Mstep(B2_step), B2_rew, color=C3b, lw=2.4, marker="^", ms=6, label="3-action B2 (ent0.003)")
ax.set_xlabel("env step (M)")
ax.set_ylabel("shaped mean reward (last 100)")
ax.set_title("학습 곡선: 2-action ~450 유지 vs 3-action 피크 후 정체/추락")
ax.grid(True, ls="--", lw=0.4, alpha=0.6)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_reward_curve.png"), dpi=130); plt.close(fig)

# ---------- Fig 3: obstacle clean-reward progression (Mean) ----------
labels = ["round1\n(고정0.4)", "obs_small\n(corner0.5)", "p02\n(corner0.2)", "lowent\n(ent0.005)", "3-action\n(ent0.003)"]
means  = [331, 365, 415, 418, 229]
colors = [C2, C2, C2, C2, C3b]
fig, ax = plt.subplots(figsize=(8, 4.6))
bars = ax.bar(labels, means, color=colors, alpha=0.85)
ax.axhline(667, color="gray", ls="--", lw=1.2, label="2-action 베이스 (무장애물) 667")
for b, m in zip(bars, means):
    ax.text(b.get_x()+b.get_width()/2, m+8, str(m), ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("clean Mean reward (50ep, seed42)")
ax.set_title("장애물 task clean 성능: 2-action 천장 ~415  vs  3-action 229")
ax.set_ylim(0, 720)
ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.6)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_clean_progression.png"), dpi=130); plt.close(fig)

# ---------- Fig 4: 2 vs 3 grouped (Mean/Median/Min) ----------
metrics = ["Mean", "Median", "Min (충돌 바닥)"]
two = [415, 422, -65]
three = [229, 245, -171]
x = np.arange(len(metrics)); w = 0.36
fig, ax = plt.subplots(figsize=(8, 4.6))
b1 = ax.bar(x-w/2, two, w, color=C2, label="2-action p02")
b2 = ax.bar(x+w/2, three, w, color=C3b, label="3-action B2")
for bars in (b1, b2):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h + (6 if h>=0 else -18), f"{int(h)}", ha="center", fontsize=10, fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylabel("clean reward (50ep, seed42)")
ax.set_title("2 vs 3 action (동일 장애물 task): 3-action은 2-action의 ~55%")
ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.6)
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_2v3_grouped.png"), dpi=130); plt.close(fig)

# ---------- Fig 5: corner-penalty lever effect ----------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
pens = ["0.5", "0.2"]; mr = [365, 415]; md = [325, 422]
x = np.arange(2); w = 0.36
ax.bar(x-w/2, mr, w, color=C2, label="Mean")
ax.bar(x+w/2, md, w, color=C3b, label="Median")
for i,(a,b) in enumerate(zip(mr,md)):
    ax.text(i-w/2, a+5, str(a), ha="center", fontsize=9, fontweight="bold")
    ax.text(i+w/2, b+5, str(b), ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"corner penalty {p}" for p in pens])
ax.set_ylabel("clean reward (50ep)")
ax.set_title("코너 패널티는 비(非)레버: 0.5→0.2 낮추니 회복")
ax.set_ylim(0, 470); ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.6); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_corner_lever.png"), dpi=130); plt.close(fig)

print("Saved figures to", OUT)
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"):
        print("  ", f)
