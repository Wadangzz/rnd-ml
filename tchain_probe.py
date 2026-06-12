"""
tchain_probe.py — 모티프 확장 시험: 지연 핸드오프 타이머 체인 (tchain3)

[ 질문 ]
  레시피(역할 featurizer + 변형 커리큘럼 + prior 탐색)가 래치 전용인가,
  모티프 일반인가. 타이머 체인은 상태 소자(TON)가 관용구 안에 들어오고
  클리어 방식이 다름 (센서 직접 → 다음 단계 핸드오프) — 질적으로 다른
  두 번째 모티프.

[ 구성 ]
  held-out: canonical tchain3 (ref 크기 24, TON 2개)
  학습: 8과제 ref + 래치 체인 변형 (2단 8 + 3단 8) + 타이머 체인 변형
       (2단 6 + 3단 8, canonical 제외) — **혼합 모티프 커리큘럼**
       (한 prior 에 두 관용구 공존 시험 포함)

[ 실행 ]
  python tchain_probe.py               # 베이스라인 + policy
  python tchain_probe.py --skip-base
"""

import sys

from ladder_benchmark import (
  GenCfg,
  make_tasks,
  run_gp,
  run_mcts_w,
  run_random,
)
from ladder_curriculum import (
  make_chain_curriculum,
  make_timer_chain_curriculum,
  make_timer_chain_task,
)
from ladder_mcts import BuildState, mcts_search
from ladder_policy import (
  build_samples,
  extract_weights,
  make_net_rollout,
  make_prior_fn,
  train,
)
from ladder_search import program_size, program_str

BUDGET = 200_000
SEEDS = (0, 1, 2)


def make_tchain3():
  xs = [f'X{i}' for i in range(4)]
  ys = [f'Y{i}' for i in range(3)]
  t = make_timer_chain_task(3, xs, ys, xs, ys, 'tchain3')
  t.gen_cfg = GenCfg(
    max_rungs=4, max_depth=4, timer_presets=(3,), wrap_p=0.2
  )
  t.mcts_kwargs = dict(
    max_actions=30,
    max_stack=3,
    max_rungs=4,
    timer_presets=(3,),
    max_timers=2,
  )
  return t


if __name__ == '__main__':
  skip_base = '--skip-base' in sys.argv
  tchain3 = make_tchain3()
  print(f'tchain3 (지연 핸드오프 3단) — ref 크기 {program_size(tchain3.reference)}')
  print(program_str(tchain3.reference))

  if not skip_base:
    print(f'\n=== 무학습 베이스라인 (예산 {BUDGET:,} × 시드 {SEEDS}) ===')
    for name, runner in [
      ('random', run_random),
      ('mcts_w', run_mcts_w),
      ('gp', run_gp),
    ]:
      for s in SEEDS:
        found, best_acc, _ = runner(tchain3, BUDGET, s)
        stat = f'{found:,}회 발견' if found else f'실패 (acc {best_acc:.3f})'
        print(f'  {name:<7} seed{s}: {stat}', flush=True)

  print('\n=== policy 학습 (혼합 모티프 커리큘럼) ===')
  train_tasks = (
    make_tasks()
    + make_chain_curriculum(8, K=2)
    + make_chain_curriculum(8, K=3)
    + make_timer_chain_curriculum(6, K=2)
    + make_timer_chain_curriculum(8, K=3)  # canonical(=tchain3) 자동 제외
  )
  samples, _ = build_samples(train_tasks)
  print(f'라벨 {len(samples)}')
  model, dev = train(samples)
  w = extract_weights(model)

  print(f'\n=== policy 측정 — tchain3 (예산 {BUDGET:,} × 시드 {SEEDS}) ===')
  rollout = make_net_rollout(w)
  prior = make_prior_fn(w)
  for name, kw in [
    ('net-rollout', dict(rollout_policy=rollout)),
    ('puct+net', dict(rollout_policy=rollout, prior_fn=prior)),
  ]:
    for s in SEEDS:
      ev = mcts_search(
        tchain3.spec,
        BUDGET,
        s,
        state_factory=lambda: BuildState(tchain3.spec, **tchain3.mcts_kwargs),
        **kw,
      )
      stat = (
        f'{ev.found_at:,}회 발견' if ev.found_at else f'실패 (acc {ev.best_acc:.3f})'
      )
      print(f'  {name:<12} seed{s}: {stat}', flush=True)
      if ev.found_at and ev.best_prog:
        print(program_str(ev.best_prog))
