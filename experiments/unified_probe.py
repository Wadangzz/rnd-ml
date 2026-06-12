"""
unified_probe.py — 통합 회귀 프로브: 단일 prior 로 3 held-out 동시 발견

[ 왜 ]
  지금까지의 프로브는 판마다 따로 학습했다 (8차=seq3용, k4=seq4용,
  tchain=tchain3용). "건바이건 땜빵이 아니라 누적"을 주장하려면
  **하나의 prior 가 세 held-out 을 동시에** 풀어야 한다 — 래치 3단 +
  래치 4단 + 타이머 3단을 한 가중치에 들고. 모티프 간 간섭이 있다면
  여기서 드러난다 (그것대로 다음 연구 대상).

[ 순수성 ]
  학습에서 세 표적의 라벨 전부 제외 — seq3 reference 제외 + 변형
  생성기의 canonical 상시 제외 (seq4/tchain3 은 애초에 benchmark 밖).

[ 실행 ]
  python unified_probe.py      # 학습 1회 + 3표적 × puct+net × 3시드
"""

from k4_probe import make_seq4
from ladder.benchmark import make_tasks
from ladder.curriculum import (
  make_chain_curriculum,
  make_timer_chain_curriculum,
)
from ladder.mcts import BuildState, mcts_search
from ladder.policy import (
  build_samples,
  extract_weights,
  make_net_rollout,
  make_prior_fn,
  train,
)
from ladder.search import program_str
from tchain_probe import make_tchain3

BUDGET = 200_000
SEEDS = (0, 1, 2)

if __name__ == '__main__':
  bench = make_tasks()
  seq3 = next(t for t in bench if t.name == 'seq3')
  targets = [seq3, make_seq4(), make_tchain3()]

  train_tasks = (
    [t for t in bench if t.name != 'seq3']
    + make_chain_curriculum(8, K=2)
    + make_chain_curriculum(8, K=3)
    + make_chain_curriculum(8, K=4)
    + make_timer_chain_curriculum(6, K=2)
    + make_timer_chain_curriculum(8, K=3)
  )
  print('=== 통합 학습 (단일 prior, 3 held-out 라벨 전부 제외) ===')
  samples, _ = build_samples(train_tasks)
  print(f'학습 과제 {len(train_tasks)}개 / 라벨 {len(samples)}')
  model, dev = train(samples)
  w = extract_weights(model)
  rollout = make_net_rollout(w)
  prior = make_prior_fn(w)

  print(f'\n=== 3 held-out 동시 측정 (puct+net, {BUDGET:,} × 시드 {SEEDS}) ===')
  results = {}
  for t in targets:
    for s in SEEDS:
      ev = mcts_search(
        t.spec,
        BUDGET,
        s,
        state_factory=lambda t=t: BuildState(t.spec, **t.mcts_kwargs),
        rollout_policy=rollout,
        prior_fn=prior,
      )
      results.setdefault(t.name, []).append(ev.found_at)
      stat = (
        f'{ev.found_at:,}회 발견' if ev.found_at else f'실패 (acc {ev.best_acc:.3f})'
      )
      print(f'  {t.name:<8} seed{s}: {stat}', flush=True)
      if ev.best_prog:
        print(program_str(ev.best_prog))

  print('\n' + '=' * 50)
  print('통합 회귀 결과 — 단일 prior 의 3 held-out 발견')
  ok = True
  for name, costs in results.items():
    cells = ' / '.join(f'{c:,}' if c else '실패' for c in costs)
    ok = ok and all(costs)
    print(f'  {name:<8}: {cells}')
  print('-' * 50)
  print('판정: ' + ('누적 증명 — 전 표적 전 시드 발견' if ok else '간섭 발견 — 실패 지점이 다음 연구 대상'))
