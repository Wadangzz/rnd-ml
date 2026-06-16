"""plot_metrics.py — metrics/history.jsonl 누적 성능 궤적 차트.

실험별 PNG 1장: x = 측정 시점(run_id 순서), 시드 평균 best_acc 선 + min/max
밴드(시드 분산), 발견(★) 마커, gp/고원 참조선. 개선 수술이 누적될수록 선이
위로 밀리는지(또는 발견 ★ 가 찍히는지)를 한눈에.

사용:
  python tools/plot_metrics.py                # 전 실험 각각 PNG
  python tools/plot_metrics.py seq4_lenext    # 특정 실험만
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # headless (WSL, 디스플레이 없음)
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from ladder.metrics import METRICS_DIR, load_history

# 한글 라벨 — Noto Sans CJK KR (WSL 기본 설치). 없으면 DejaVu(tofu) 폴백.
for _fp in (
  '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
  '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
):
  if Path(_fp).exists():
    fm.fontManager.addfont(_fp)
    plt.rcParams['font.family'] = fm.FontProperties(fname=_fp).get_name()
    break
plt.rcParams['axes.unicode_minus'] = False  # 음수 부호 깨짐 방지

# 변형별 고정 색 (재실행해도 일관)
COLORS = {
  'net-rollout': '#1f77b4',
  'puct+net': '#d62728',
  'puct+w': '#2ca02c',
  'mcts_w': '#ff7f0e',
  'gp': '#9467bd',
  'random': '#8c564b',
}


def aggregate(records: list[dict]):
  """(experiment, variant, run_id) → 시드 집계 (mean/min/max acc, 발견 수)."""
  buckets = defaultdict(list)
  for r in records:
    buckets[(r['experiment'], r['variant'], r['run_id'])].append(r)
  agg = defaultdict(dict)  # experiment → variant → [ (run_id, mean, lo, hi, n_disc, n) ]
  for (exp, var, run_id), rs in buckets.items():
    accs = [x['best_acc'] for x in rs]
    n_disc = sum(1 for x in rs if x['discovered'])
    note = rs[0].get('note', '')
    agg[exp].setdefault(var, []).append(
      (run_id, sum(accs) / len(accs), min(accs), max(accs), n_disc, len(rs), note)
    )
  # run_id 시간순 정렬
  for exp in agg:
    for var in agg[exp]:
      agg[exp][var].sort(key=lambda t: t[0])
  return agg


def plot_experiment(exp: str, by_var: dict):
  fig, ax = plt.subplots(figsize=(11, 6))
  # x 축 = 전 변형 통합 run_id 순서 (시점 인덱스)
  run_ids = sorted({row[0] for rows in by_var.values() for row in rows})
  xpos = {rid: i for i, rid in enumerate(run_ids)}

  for var, rows in sorted(by_var.items()):
    color = COLORS.get(var, None)
    xs = [xpos[r[0]] for r in rows]
    means = [r[1] for r in rows]
    los = [r[2] for r in rows]
    his = [r[3] for r in rows]
    ax.plot(xs, means, '-o', color=color, label=var, lw=1.8, ms=5, zorder=3)
    ax.fill_between(xs, los, his, color=color, alpha=0.12, zorder=1)
    # 발견(★) 마커
    for x, row in zip(xs, rows):
      if row[4] > 0:  # n_disc
        ax.scatter(
          [x], [row[1]], marker='*', s=240, color=color,
          edgecolors='black', linewidths=0.6, zorder=5,
        )
        ax.annotate(
          f'{row[4]}/{row[5]} 발견', (x, row[1]),
          textcoords='offset points', xytext=(0, 10),
          ha='center', fontsize=8, color=color,
        )

  ax.axhline(1.0, color='green', ls='--', lw=1, alpha=0.6)
  ax.text(0, 1.002, 'acc=1.0 (발견)', fontsize=8, color='green')
  ax.set_ylim(min(0.85, ax.get_ylim()[0]), 1.02)
  ax.set_xticks(list(xpos.values()))
  ax.set_xticklabels(
    [rid[4:] for rid in run_ids], rotation=45, ha='right', fontsize=7
  )
  ax.set_xlabel('측정 시점 (run_id, 시간순)')
  ax.set_ylabel('best accuracy (시드 평균, 밴드=min/max)')
  ax.set_title(f'{exp} — 누적 성능 궤적 (★ = 발견)')
  ax.legend(loc='lower right', fontsize=9)
  ax.grid(True, alpha=0.25)
  fig.tight_layout()
  out = METRICS_DIR / f'perf_{exp}.png'
  fig.savefig(out, dpi=120)
  plt.close(fig)
  return out


def main():
  records = load_history()
  if not records:
    print('history.jsonl 비어 있음 — 측정 먼저 실행')
    return
  agg = aggregate(records)
  wanted = sys.argv[1:] or list(agg.keys())
  for exp in wanted:
    if exp not in agg:
      print(f'  (없음) {exp}')
      continue
    out = plot_experiment(exp, agg[exp])
    print(f'  저장: {out}')


if __name__ == '__main__':
  main()
