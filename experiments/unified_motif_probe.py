"""
unified_motif_probe.py — 여러 추출 실전 모티프를 단일 prior 로 (누적 증명)

[ 왜 ]
  지금까지 추출 모티프(sensor_input/action_cond)를 건바이건 발견했다. 제품
  prior 는 여러 모티프를 한 번에 담아야 한다. 단일 prior 를 여러 모티프의 변형
  커리큘럼으로 학습 → 각 held-out canonical 을 전부 발견하는지(간섭 없는 누적).
  unified_probe(체인/타이머/seq)의 실전 추출 모티프 버전.

[ 무엇 ]
  TARGETS 각각: 추출→역할정규화 Task + 변형 커리큘럼(canonical 제외).
  단일 prior = make_tasks + 체인 K2/3 + 모든 TARGET 의 변형 커리큘럼.
  측정: 각 TARGET held-out canonical 을 puct+net/net-rollout ×3시드 발견?

[ 실행 ]
  uv run experiments/unified_motif_probe.py
  uv run experiments/unified_motif_probe.py --budget 50000   # 예산 축소
"""

import sys

from motif_discover import make_motif_task  # 같은 experiments 디렉토리

from ladder.benchmark import make_tasks
from ladder.curriculum import make_chain_curriculum, make_motif_curriculum
from ladder.metrics import log_run
from ladder.parallel import parallel_search
from ladder.policy import build_samples, extract_weights, train
from ladder.search import program_size, program_str

TARGETS = ['actuator/sensor_input', 'actuator/action_cond']
SEEDS = (0, 1, 2)


def _arg(flag, default):
  if flag in sys.argv:
    return int(sys.argv[sys.argv.index(flag) + 1])
  return default


if __name__ == '__main__':
  budget = _arg('--budget', 200_000)
  motif_n = _arg('--motif-n', 12)

  tasks = {name: make_motif_task(name) for name in TARGETS}
  print('=== 타깃 (held-out canonical) ===')
  for name, t in tasks.items():
    print(f'  {name}: {program_str(t.reference).strip()}')

  # 단일 prior 학습셋 = 기본 + 체인 + 모든 타깃의 변형 커리큘럼 (canonical 제외)
  print(f'\n=== 단일 prior 학습 (기본+체인+모티프 변형×{len(TARGETS)}) ===')
  train_tasks = (
    make_tasks() + make_chain_curriculum(8, K=2) + make_chain_curriculum(8, K=3)
  )
  for name, t in tasks.items():
    train_tasks += make_motif_curriculum(t.reference, n_variants=motif_n)
  samples, _ = build_samples(train_tasks)
  print(f'과제 {len(train_tasks)} → 라벨 {len(samples)}')
  model, _ = train(samples)
  w = extract_weights(model)

  # 모든 타깃 × 2방법 × 3시드 한 번에 병렬
  print(f'\n=== 측정 (예산 {budget:,} × 시드 {SEEDS}, 전 타깃 병렬) ===')
  jobs, labels = [], []
  for name, t in tasks.items():
    for mname, use_prior in [('net-rollout', False), ('puct+net', True)]:
      for s in SEEDS:
        jobs.append((t.spec, t.mcts_kwargs, budget, s, w, use_prior))
        labels.append((name, mname, s))

  results = parallel_search(jobs)
  found_tbl = {}  # (target, method) -> [found per seed]
  for (name, mname, s), (found, acc, prog) in zip(labels, results):
    found_tbl.setdefault((name, mname), []).append(found)
    log_run('unified_motif', f'{name.split("/")[-1]}:{mname}', s, found, acc,
            ref_size=program_size(tasks[name].reference),
            prog_size=program_size(prog) if prog else None, note='unified')

  print(f'\n{"target":<24}{"method":<14}{"발견 (시드별)"}')
  print('-' * 60)
  for name in TARGETS:
    for mname in ('net-rollout', 'puct+net'):
      fs = found_tbl[(name, mname)]
      n_found = sum(1 for f in fs if f)
      cells = ' '.join(f'{f:,}' if f else '✗' for f in fs)
      print(f'{name:<24}{mname:<14}{n_found}/3  [{cells}]')

  total = sum(1 for fs in found_tbl.values() for f in fs if f)
  print(f'\n누적: {total}/{len(jobs)} 발견 '
        f'({"전부 발견 = 간섭 없음 ✅" if total == len(jobs) else "일부 미발견 — 간섭 점검"})')
