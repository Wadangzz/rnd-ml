"""
temp_sweep.py — 롤아웃 softmax 온도 스윕 (7차 과적합 가설 진단)

7차: featurizer v3 로 학습은 완벽(loss 0.036)했으나 holdout seq3 전이
붕괴(0.893). puct≤net-rollout 로 '과신 오답' 교차검증됨. 가설: prior 가
너무 뾰족 → 본 적 없는 과제에서 과신. 검증 = 온도로 prior 를 펴서
3차의 '느슨함'을 인위 복원하면 회복되나.

저장된 policy_weights.npz 재사용 (재학습 없음). 1차는 seed 0 만 빠르게.

  python temp_sweep.py                  # seed 0, temp 1/2/3/5
  python temp_sweep.py 1 2 4 8          # 온도 직접 지정
"""

import sys

import numpy as np

from ladder_benchmark import make_tasks
from ladder_mcts import BuildState, mcts_search
from ladder_policy import WEIGHTS_PATH, make_net_rollout

temps = [float(x) for x in sys.argv[1:]] or [1.0, 2.0, 3.0, 5.0]
data = np.load(WEIGHTS_PATH)
w = {k: data[k] for k in data.files}
seq3 = next(t for t in make_tasks() if t.name == 'seq3')

print('온도 스윕 — seq3 net-rollout, seed 0, 예산 200k')
print('기준: 7차 temp=1.0 → 0.893 / 목표: 3차 0.953 회복 또는 발견')
print('-' * 50)
for temp in temps:
  ev = mcts_search(
    seq3.spec,
    200_000,
    0,
    state_factory=lambda: BuildState(seq3.spec, **seq3.mcts_kwargs),
    rollout_policy=make_net_rollout(w, temp=temp),
  )
  stat = f'{ev.found_at:,}회 발견' if ev.found_at else f'미발견 best {ev.best_acc:.3f}'
  print(f'  temp {temp:>4.1f}:  {stat}', flush=True)
