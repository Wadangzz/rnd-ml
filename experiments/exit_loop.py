"""
exit_loop.py — Expert Iteration(ExIt) 1회전: 탐색이 만든 데이터로 탐색을 강화

[ 왜 ]
  8차에서 held-out seq3 첫 발견 (puct+net 1~17회) — 그 순간 ExIt 의
  입구가 열렸다. 탐색(전문가)이 찾은 검증된 해는 무비용 라벨이고,
  그걸 정책망(도제)에 환류하면 다음 탐색이 강해진다. AlphaZero 의
  자기대국과 같은 구조 — 여기선 자기'탐색'.

[ 순수성 규칙 ]
  seq3 의 사람 레퍼런스 라벨은 끝까지 쓰지 않는다. 환류되는 것은
  round 0 탐색이 **스스로 발견한 해**뿐 — 검증 게이트(acc==1.0 &
  invariant 0 & 이중코일 0) 통과 후 polish(단순화→스펙축소)를 거쳐
  분해한다. 사람 개입 0 인 자기개선 사이클의 최소 실증.

[ 측정 ]
  seq3 발견 비용 (puct+net, 3시드, 200k) — round 0 vs round 1.
  기대: 환류 후 seq3 가 분포 안으로 들어와 발견 비용 급감 (→ ~1회).
  이건 루프 배관의 검증이고, 진짜 가치는 다음 단계 — 환류로 강해진
  prior 가 '더 어려운' 과제(K=4 체인)의 문턱을 낮추는가.

[ 실행 ]
  python exit_loop.py          # round 0 → 환류 → round 1 (수 분)
"""

import numpy as np

from ladder.benchmark import make_tasks
from ladder.curriculum import make_chain_curriculum
from ladder.decompose import decompose_with_states
from ladder.mcts import BuildState, mcts_search
from ladder.policy import (
  build_samples,
  candidate_actions,
  extract_weights,
  featurize_pairs,
  make_net_rollout,
  make_prior_fn,
  train,
)
from ladder.search import (
  evaluate,
  find_coil_conflicts,
  program_size,
  program_str,
)
from ladder.simplify import polish_program

BUDGET = 200_000
SEEDS = (0, 1, 2)


def measure(w, task, label):
  """puct+net 으로 발견 비용 측정 + 발견 해 수집"""
  rollout = make_net_rollout(w)
  prior = make_prior_fn(w)
  costs, found = [], []
  for s in SEEDS:
    ev = mcts_search(
      task.spec,
      BUDGET,
      s,
      state_factory=lambda: BuildState(task.spec, **task.mcts_kwargs),
      rollout_policy=rollout,
      prior_fn=prior,
    )
    costs.append(ev.found_at)
    if ev.found_at and ev.best_prog is not None:
      found.append(ev.best_prog)
    stat = f'{ev.found_at:,}회' if ev.found_at else f'실패 {ev.best_acc:.3f}'
    print(f'  [{label}] seed{s}: {stat}', flush=True)
  return costs, found


def expert_labels(progs, spec):
  """발견 해 → 검증 게이트 → polish → (특징행렬, 정답) 라벨"""
  samples, kept = [], []
  seen = set()
  for prog in progs:
    acc, viol = evaluate(prog, spec)
    if not (acc >= 1.0 and viol == 0 and not find_coil_conflicts(prog)):
      print(f'  검증 탈락 (acc={acc:.3f} viol={viol})')
      continue
    p = polish_program(prog, spec)
    key = program_str(p)
    if key in seen:
      continue  # 동일 해 중복 환류 방지
    seen.add(key)
    kept.append(p)
    for state, act in decompose_with_states(p, spec):
      cands = candidate_actions(state, act)
      samples.append((featurize_pairs(state, cands), cands.index(act)))
  return samples, kept


if __name__ == '__main__':
  tasks = make_tasks()
  seq3 = next(t for t in tasks if t.name == 'seq3')
  train_tasks = (
    [t for t in tasks if t.name != 'seq3']
    + make_chain_curriculum(8, K=2)
    + make_chain_curriculum(8, K=3)
  )

  print('=== round 0: 기반 학습 (8차 구성 재현) ===')
  base_samples, _ = build_samples(train_tasks)
  print(f'기반 라벨 {len(base_samples)}')
  model, dev = train(base_samples)
  w0 = extract_weights(model)

  print('\n=== round 0 측정 + 전문가 해 수집 (holdout seq3, puct+net) ===')
  costs0, found = measure(w0, seq3, 'r0')
  assert found, 'round 0 에서 발견 0 — ExIt 환류 불가 (8차 재현 실패?)'

  print('\n=== 환류: 검증 게이트 → polish → 분해 ===')
  extra, kept = expert_labels(found, seq3.spec)
  for p in kept:
    print(f'  환류 해 (크기 {program_size(p)}):')
    print(program_str(p))
  print(f'  전문가 라벨 +{len(extra)} (해 {len(kept)}개)')

  print('\n=== round 1: 환류 재학습 ===')
  model1, _ = train(base_samples + extra)
  w1 = extract_weights(model1)
  np.savez('policy_weights_exit1.npz', **w1)

  print('\n=== round 1 측정 (동일 조건) ===')
  costs1, _ = measure(w1, seq3, 'r1')

  print('\n' + '=' * 50)
  print('ExIt 1회전 결과 — seq3 발견 비용 (puct+net)')
  for s, (a, b) in zip(SEEDS, zip(costs0, costs1)):
    fa = f'{a:,}' if a else '실패'
    fb = f'{b:,}' if b else '실패'
    print(f'  seed{s}: {fa} → {fb}')
