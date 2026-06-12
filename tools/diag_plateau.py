"""
diag_plateau.py — 고원에 박힌 최고 후보 회로의 실패 지점 진단

벤치마크에서 acc 고원(예: interlock 0.967)에 갇혔을 때, 그 1~3점이
  - 전이 직후 스캔(width=2 마스킹으로 풀릴 타이밍 부산물)인지
  - 정상 채점 스캔(진짜 의미 차이 = 탐색/스펙 본질 문제)인지
를 구분한다.

실행:
  python -m diag_plateau interlock seq2
"""

import sys

from ladder.benchmark import (
    make_tasks,
    run_random,
)
from ladder.search import program_str
from ladder.sim import simulate

BUDGET = 200_000
SEED = 0


def transition_scans(input_trace):
    cur, trans = {}, set()
    for t, upd in enumerate(input_trace):
        if any(cur.get(k, 0) != v for k, v in upd.items()):
            trans.add(t)
        cur.update(upd)
    return trans


def diagnose(task):
    found, acc, prog = run_random(task, BUDGET, SEED)
    print(
        f'=== {task.name}  best acc={acc:.3f}  ({"완벽해" if found else "고원"}) ==='
    )
    print(program_str(prog))
    for i, sc in enumerate(task.spec.scenarios):
        trans = transition_scans(sc.input_trace)
        trace = simulate(prog, sc.input_trace, task.spec.outputs)
        for s, (got, want) in enumerate(zip(trace, sc.expected)):
            for dev, wv in want.items():
                if wv is None:
                    continue
                if got.get(dev, 0) != wv:
                    near = (
                        '전이+1 → width=2면 마스킹됨'
                        if (s - 1) in trans
                        else '정상 채점 스캔'
                    )
                    print(
                        f'  시나리오{i} scan{s} {dev}:'
                        f' got {got.get(dev, 0)} want {wv}  [{near}]'
                    )
    print()


if __name__ == '__main__':
    names = sys.argv[1:] or ['interlock', 'seq2']
    tasks = {t.name: t for t in make_tasks()}
    for name in names:
        diagnose(tasks[name])
