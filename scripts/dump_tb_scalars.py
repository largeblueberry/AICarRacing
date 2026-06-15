"""
Dump TensorBoard scalar curves to a text table + PNG, without a browser.

Usage:
    python scripts/dump_tb_scalars.py logs/obs_3action
    python scripts/dump_tb_scalars.py logs/obs_3action logs/obs_small_p02   # overlay/compare

For each run dir it:
  - loads all scalar events (recursively finds the event file),
  - prints a downsampled table of the most diagnostic tags so it can be pasted,
  - saves a multi-panel PNG (reward / entropy / value_loss / approx_kl /
    clip_fraction / obstacle_hits). When several run dirs are given they are
    overlaid in one figure for direct comparison.

Why: start/end rollout snapshots can't show WHEN/HOW a run diverged. The full
entropy + reward curve pinpoints the peak and the divergence onset.
"""
import os
import sys
import argparse

import numpy as np

try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception as e:  # pragma: no cover
    print("ERROR: could not import tensorboard's event_accumulator.")
    print("Install it in this env:  pip install tensorboard")
    print(f"(import error: {e})")
    sys.exit(1)

# Tags we care about, in display order. Missing tags are silently skipped.
TABLE_TAGS = [
    "charts/mean_reward",
    "losses/entropy_loss",
    "losses/value_loss",
    "losses/approx_kl",
    "losses/clip_fraction",
    "driving/mean_obstacle_hits",
]
# Short column headers for the printed table.
SHORT = {
    "charts/mean_reward": "reward",
    "losses/entropy_loss": "entropy",
    "losses/value_loss": "val_loss",
    "losses/approx_kl": "kl",
    "losses/clip_fraction": "clipfrac",
    "driving/mean_obstacle_hits": "obst_hits",
}
# Panels for the PNG (tag -> subplot title).
PLOT_TAGS = [
    ("charts/mean_reward", "shaped mean_reward"),
    ("losses/entropy_loss", "entropy_loss (std proxy)"),
    ("losses/value_loss", "value_loss"),
    ("losses/approx_kl", "approx_kl"),
    ("losses/clip_fraction", "clip_fraction"),
    ("driving/mean_obstacle_hits", "obstacle_hits"),
]


def load_run(run_dir):
    """Return {tag: (steps_array, values_array)} for one run dir."""
    if not os.path.isdir(run_dir):
        print(f"  ! not a directory: {run_dir}")
        return {}
    ea = event_accumulator.EventAccumulator(
        run_dir,
        size_guidance={event_accumulator.SCALARS: 0},  # 0 == load everything
    )
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out = {}
    for tag in available:
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=np.int64)
        vals = np.array([e.value for e in events], dtype=np.float64)
        # A run dir may hold several event files (restarts) concatenated in
        # file order, not step order. Sort by step so tables/plots are monotonic.
        order = np.argsort(steps, kind="stable")
        out[tag] = (steps[order], vals[order])
    return out, available


def downsample_idx(n, rows):
    if n <= rows:
        return list(range(n))
    return [int(round(i)) for i in np.linspace(0, n - 1, rows)]


def print_table(name, data, available, rows=25):
    print("=" * 78)
    print(f"RUN: {name}")
    tags = [t for t in TABLE_TAGS if t in available]
    if not tags:
        print("  (none of the expected tags found; available scalars:)")
        print("   " + ", ".join(sorted(available)))
        return
    # Use mean_reward's step axis as the row index (fallback to first tag).
    ref = "charts/mean_reward" if "charts/mean_reward" in data else tags[0]
    ref_steps = data[ref][0]
    idx = downsample_idx(len(ref_steps), rows)

    header = f"{'step':>10} | " + " | ".join(f"{SHORT.get(t, t):>9}" for t in tags)
    print(header)
    print("-" * len(header))
    for i in idx:
        step = ref_steps[i]
        cells = []
        for t in tags:
            s, v = data[t]
            # nearest value at-or-before this step
            j = np.searchsorted(s, step, side="right") - 1
            j = max(0, min(j, len(v) - 1))
            cells.append(f"{v[j]:>9.3f}")
        print(f"{step:>10} | " + " | ".join(cells))

    # Peak + final summary for the headline metric.
    if "charts/mean_reward" in data:
        s, v = data["charts/mean_reward"]
        pk = int(np.argmax(v))
        print(f"  -> reward PEAK {v[pk]:.1f} @ step {s[pk]}  |  FINAL {v[-1]:.1f} @ step {s[-1]}")
    if "losses/entropy_loss" in data:
        s, v = data["losses/entropy_loss"]
        print(f"  -> entropy START {v[0]:.2f} -> FINAL {v[-1]:.2f}  (rise => std inflating/diverging)")


def make_plot(runs, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping PNG: matplotlib unavailable: {e})")
        return
    panels = PLOT_TAGS
    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, (tag, title) in zip(axes, panels):
        for name, (data, _avail) in runs.items():
            if tag in data:
                s, v = data[tag]
                # Mask non-finite values (e.g. mean_reward is -inf before the
                # first 100 episodes complete) so they don't wreck the y-axis.
                finite = np.isfinite(v)
                if finite.any():
                    ax.plot(s[finite], v[finite], label=name, alpha=0.85, linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("env step")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
        ax.legend(fontsize=8)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=110)
        print(f"\nSaved comparison PNG: {out_path}")
    except Exception as e:
        print(f"(error saving PNG: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump TensorBoard scalar curves to text + PNG.")
    parser.add_argument("run_dirs", nargs="+", help="One or more TB log directories.")
    parser.add_argument("--rows", type=int, default=25, help="Rows in the printed table (default 25).")
    parser.add_argument("--out", type=str, default="tb_curves.png", help="Output PNG path.")
    args = parser.parse_args()

    runs = {}
    for d in args.run_dirs:
        name = os.path.basename(os.path.normpath(d))
        loaded = load_run(d)
        if not loaded:
            continue
        data, available = loaded
        runs[name] = (data, available)
        print_table(name, data, available, rows=args.rows)
        print()

    if runs:
        make_plot(runs, args.out)
    else:
        print("No runs loaded.")
