"""
k4_probe.py — K=4 체인 탐침: 새 한계 설정 + 학습 prior 의 인스턴스 일반화 재시험

[ 왜 ]
  8차에서 seq3(3단)가 풀렸으니 한계를 한 단 밀어 본다. canonical 4단
  체인(seq4, ref 크기 30)을 새 held-out 으로:

    ① 무학습 베이스라인(random/mcts_w/gp)이 못 푸는지 — 새 경계 확인
    ② K≤4 변형(canonical 제외)으로 학습한 prior 가 푸는지 — 8차 패턴
       (같은 길이·다른 배정 인스턴스 일반화)이 한 단 위에서도 성립하는지

  이번엔 seq3 는 더 이상 held-out 이 아님 (목표가 seq4 로 이동) —
  8과제 reference 전부 + 2/3/4단 변형이 학습 데이터.

[ 실행 ]
  python k4_probe.py               # 베이스라인 3종×3시드 + policy 2구성×3시드
  python k4_probe.py --skip-base   # 베이스라인 생략 (policy 만)
"""

import sys

from ladder.benchmark import (
  GenCfg,
  make_tasks,
  run_gp,
  run_mcts_w,
  run_random,
)
from ladder.curriculum import make_chain_curriculum, make_chain_task
from ladder.metrics import log_run
from ladder.parallel import parallel_search
from ladder.policy import build_samples, extract_weights, train
from ladder.search import program_size, program_str

BUDGET = 200_000
SEEDS = (0, 1, 2)


def make_seq4():
  """canonical 4단 체인 — 새 held-out 과제 (학습 변형 생성기는 이 배정을 제외)"""
  t = make_chain_task(
    4,
    [f'X{i}' for i in range(5)],
    [f'Y{i}' for i in range(4)],
    [f'X{i}' for i in range(5)],
    [f'Y{i}' for i in range(4)],
    'seq4',
  )
  # 탐색용 설정 (make_chain_task 는 라벨 전용이라 기본값) — SET/RST 자연형 8 rung
  t.gen_cfg = GenCfg(max_rungs=8, max_depth=4, setrst_p=0.3)
  # max_actions 42 = 구 EMIT 그래머의 34 + max_rungs(8): OPEN+CLOSE 가 rung 당
  # 1수 더 들어 동일 도달성 유지하려면 +max_rungs (2026-06-15 베이스라인 등가).
  t.mcts_kwargs = dict(max_actions=42, max_stack=3, max_rungs=8, allow_setrst=True)
  return t


if __name__ == '__main__':
  skip_base = '--skip-base' in sys.argv
  lenext = '--lenext' in sys.argv  # K=4 변형 제외 = 순수 길이 외삽 시험
  seq4 = make_seq4()
  print(f'seq4 (canonical 4단 체인) — ref 크기 {program_size(seq4.reference)}')
  print(program_str(seq4.reference))

  if not skip_base:
    print(f'\n=== 무학습 베이스라인 (예산 {BUDGET:,} × 시드 {SEEDS}) ===')
    for name, runner in [
      ('random', run_random),
      ('mcts_w', run_mcts_w),
      ('gp', run_gp),
    ]:
      for s in SEEDS:
        found, best_acc, _ = runner(seq4, BUDGET, s)
        stat = f'{found:,}회 발견' if found else f'실패 (acc {best_acc:.3f})'
        print(f'  {name:<7} seed{s}: {stat}', flush=True)

  mode = 'K<=3 (순수 길이 외삽)' if lenext else 'K<=4 (인스턴스 일반화)'
  print(f'\n=== policy 학습 — {mode} ===')
  train_tasks = (
    make_tasks()
    + make_chain_curriculum(8, K=2)
    + make_chain_curriculum(8, K=3)
  )
  if not lenext:
    train_tasks += make_chain_curriculum(8, K=4)  # canonical(=seq4) 자동 제외
  samples, _ = build_samples(train_tasks)
  print(f'라벨 {len(samples)}')
  model, dev = train(samples)
  w = extract_weights(model)

  print(f'\n=== policy 측정 — seq4 (예산 {BUDGET:,} × 시드 {SEEDS}, 병렬) ===')
  jobs, labels = [], []
  for name, use_prior in [('net-rollout', False), ('puct+net', True)]:
    for s in SEEDS:
      jobs.append((seq4.spec, seq4.mcts_kwargs, BUDGET, s, w, use_prior))
      labels.append((name, s))
  exp = 'seq4_lenext' if lenext else 'seq4_instgen'
  ref_sz = program_size(seq4.reference)
  for (name, s), (found, acc, prog) in zip(labels, parallel_search(jobs)):
    stat = f'{found:,}회 발견' if found else f'실패 (acc {acc:.3f})'
    print(f'  {name:<12} seed{s}: {stat}', flush=True)
    try:  # 로깅 실패가 비싼 탐색 결과를 날리지 않도록 방어
      log_run(
        exp, name, s, found, acc,
        ref_size=ref_sz,
        prog_size=program_size(prog) if prog else None,
      )
    except Exception as e:  # noqa: BLE001
      print(f'  [metrics 기록 실패: {e}]', flush=True)
    if prog:  # 실패해도 최선 회로 출력 — 고원에서 무엇을 못 닫나 진단
      print(program_str(prog))
