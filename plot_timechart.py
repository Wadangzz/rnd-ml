"""
plot_timechart.py — 스펙 타임차트를 로직 애널라이저 풍 PNG로 렌더

연구일지/보고 자료화 용도.
  - 입력 파형: 파란색 (전이 스캔 기준 forward-fill)
  - 기대 출력 파형: 빨간색
  - don't-care 스캔: 회색 음영 + 해당 구간 출력 파형 생략

실행 (폴더 안에서, matplotlib 있는 인터프리터로):
  python plot_timechart.py            # 전체 과제
  python plot_timechart.py interlock seq2
  → charts/{과제}_s{N}.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ladder_benchmark import make_tasks

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).parent / "charts"

IN_COLOR = "#1f77b4"
OUT_COLOR = "#d62728"
HIGH = 0.85                    # 파형 1 레벨 높이
ROW_H = 1.3                    # 디바이스 행 간격


def filled_inputs(input_trace, devices):
    """스캔별 갱신 dict → 전 스캔 forward-fill 값"""
    cur = {d: 0 for d in devices}
    rows = []
    for upd in input_trace:
        cur.update(upd)
        rows.append(dict(cur))
    return rows


def draw_wave(ax, base, vals, color):
    """0/1/None 목록을 디지털 파형으로. None 구간은 비움"""
    prev = None
    for t, v in enumerate(vals):
        if v is None:
            prev = None
            continue
        y = base + v * HIGH
        ax.plot([t, t + 1], [y, y], color=color, lw=2.2,
                solid_capstyle="butt")
        if prev is not None and prev != v:
            ax.plot([t, t], [base + prev * HIGH, y], color=color, lw=2.2)
        prev = v


def plot_scenario(task, idx):
    sc = task.spec.scenarios[idx]
    n = len(sc.input_trace)
    in_rows = filled_inputs(sc.input_trace, task.spec.inputs)

    rows = [(d, [r[d] for r in in_rows], IN_COLOR)
            for d in task.spec.inputs]
    rows += [(d, [w.get(d) for w in sc.expected], OUT_COLOR)
             for d in task.spec.outputs]

    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.42 * n + 1.5), 0.62 * len(rows) + 1.1))

    # don't-care 스캔 음영 (마스킹은 스캔 단위 — 출력 전부 None)
    for t, want in enumerate(sc.expected):
        if all(v is None for v in want.values()):
            ax.axvspan(t, t + 1, color="0.90", zorder=0)

    yticks, ylabels = [], []
    for i, (dev, vals, color) in enumerate(reversed(rows)):
        base = i * ROW_H
        ax.axhline(base, color="0.85", lw=0.6, zorder=1)
        draw_wave(ax, base, vals, color)
        yticks.append(base + HIGH / 2)
        ylabels.append(dev)

    ax.set_yticks(yticks, ylabels, fontsize=10)
    ax.set_xticks(range(0, n + 1))
    ax.set_xlim(0, n)
    ax.set_ylim(-0.4, (len(rows) - 1) * ROW_H + HIGH + 0.4)
    ax.set_xlabel("scan")
    ax.grid(axis="x", color="0.9", lw=0.5)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)
    ax.set_title(f"{task.name} — 시나리오 {idx}"
                 f"  (회색 = don't-care)", fontsize=11, loc="left")
    fig.tight_layout()
    return fig


def main():
    tasks = {t.name: t for t in make_tasks()}
    names = sys.argv[1:] or list(tasks)
    unknown = set(names) - set(tasks)
    assert not unknown, f"없는 과제: {unknown}"

    OUT_DIR.mkdir(exist_ok=True)
    for name in names:
        task = tasks[name]
        for idx in range(len(task.spec.scenarios)):
            fig = plot_scenario(task, idx)
            path = OUT_DIR / f"{name}_s{idx}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"저장: {path}")


if __name__ == "__main__":
    main()
