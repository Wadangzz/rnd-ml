"""
motif_generalize.py — 신규 일반화 (leave-one-out): 학습 안 한 모티프 발견?

[ 왜 ]
  지금까지는 타깃 모티프의 변형 커리큘럼을 학습에 넣고 그 canonical 을
  발견했다 (인스턴스 일반화). 제품의 진짜 가치는 "**학습 안 한** 구조도 합성"
  = 구조 외삽. TARGET 의 커리큘럼은 빼고 다른 모티프(TRAIN_MOTIFS)의 커리큘럼
  + 체인/타이머만 학습 → TARGET 을 발견하는지. 길이 외삽 벽처럼 안 될 수도
  있음 — 되면 신규 일반화 첫 증거, 안 되면 "라이브러리를 넓게 깔아야" 경계 확정.

[ 무엇 ]
  prior 학습 = make_tasks + 체인 K2/3 + TRAIN_MOTIFS 변형 커리큘럼 (TARGET 제외)
  측정 = TARGET held-out canonical 발견? + 대조용 with-target 표시는 motif_discover

[ 실행 ]
  uv run experiments/motif_generalize.py
  uv run experiments/motif_generalize.py --target actuator/action_cond \\
      --train actuator/sensor_input
"""

import sys

from motif_discover import make_motif_task

from ladder.benchmark import make_tasks
from ladder.curriculum import make_chain_curriculum, make_motif_curriculum
from ladder.metrics import log_run
from ladder.parallel import parallel_search
from ladder.policy import build_samples, extract_weights, train
from ladder.search import program_size, program_str

SEEDS = (0, 1, 2)


def _arg(flag, default):
  return int(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def _arg_str(flag, default):
  return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


if __name__ == '__main__':
  budget = _arg('--budget', 200_000)
  target = _arg_str('--target', 'actuator/action_cond')
  train_motifs = _arg_str('--train', 'actuator/sensor_input').split(',')
  motif_n = _arg('--motif-n', 12)

  tgt = make_motif_task(target)
  print(f'=== TARGET (held-out, 커리큘럼 없음) = {target} ===')
  print(program_str(tgt.reference))
  print(f'학습 모티프(커리큘럼 제공): {train_motifs}\n')

  # TARGET 의 커리큘럼은 빼고 학습 — 신규 구조 외삽 시험
  train_tasks = (
    make_tasks() + make_chain_curriculum(8, K=2) + make_chain_curriculum(8, K=3)
  )
  for name in train_motifs:
    t = make_motif_task(name)
    train_tasks += make_motif_curriculum(t.reference, n_variants=motif_n)
  samples, _ = build_samples(train_tasks)
  print(f'과제 {len(train_tasks)} → 라벨 {len(samples)} (TARGET 변형 미포함)')
  model, _ = train(samples)
  w = extract_weights(model)

  print(f'\n=== TARGET 발견 측정 (예산 {budget:,} × 시드 {SEEDS}, 병렬) ===')
  jobs, labels = [], []
  for mname, use_prior in [('net-rollout', False), ('puct+net', True)]:
    for s in SEEDS:
      jobs.append((tgt.spec, tgt.mcts_kwargs, budget, s, w, use_prior))
      labels.append((mname, s))
  exp = 'generalize_' + target.replace('/', '_')
  tbl = {}
  for (mname, s), (found, acc, prog) in zip(labels, parallel_search(jobs)):
    tbl.setdefault(mname, []).append((found, acc))
    stat = f'{found:,}회 발견' if found else f'실패 (acc {acc:.3f})'
    print(f'  {mname:<12} seed{s}: {stat}', flush=True)
    log_run(exp, mname, s, found, acc, ref_size=program_size(tgt.reference),
            prog_size=program_size(prog) if prog else None, note='leave_one_out')

  print(f'\n{"method":<14}발견')
  for mname in ('net-rollout', 'puct+net'):
    nf = sum(1 for f, _ in tbl[mname] if f)
    print(f'{mname:<14}{nf}/3')
  total = sum(1 for fs in tbl.values() for f, _ in fs if f)
  print(f'\n신규 일반화: {total}/{len(jobs)} '
        f'({"발견됨 = 구조 외삽 성립 🎉" if total else "0 = 학습 안 한 구조 미발견 (라이브러리 확장 필요 경계)"})')
