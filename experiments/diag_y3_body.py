"""
diag_y3_body.py — Y3 body 조립 붕괴의 잔여 OOD 진단 (학습 불필요)

[ 왜 ]
  c_opengap 실측: puct+net 0.971, Y0/Y1/Y2 ref 정확 일치, **최상단 Y3 만**
  `X3 -> Y3` 로 조기 종료. OPEN-선택은 해결됐는데(Y3 rung 을 열긴 함) body
  를 못 짓는다. Y2 body 는 완벽 전이됐는데 Y3 body 만 무너지는 차이가 어느
  특징의 OOD 인지.

[ 무엇 ]
  학습(8 ref + K=2/3 변형, seq4 미공급)에서 '정답 라벨 액션'이 받는 각 문맥
  특징의 support 를 모으고, seq4 의 **최상단 타깃(Y3) rung body** 정답 액션이
  받는 값과 대조 → 어느 dim 이 분포 밖인지.

  대상 특징 (featurize_action 의 해석 가능 스칼라):
    c_idx, c_rel, c_tgtrel, c_opengap, role(on/off/self rank·delay·flag)

[ 판정 ]
  Y3 body 액션에서만 OOD 인 특징 = 다음 상대화 수술 대상.

[ 실행 ]
  uv run experiments/diag_y3_body.py
"""

from collections import defaultdict

from k4_probe import make_seq4

from ladder.benchmark import make_tasks
from ladder.curriculum import make_chain_curriculum
from ladder.decompose import decompose_with_states
from ladder.policy import (
  Ctx,
  dev_idx,
  dev_role,
  role_feats,
  spec_roles,
)

# 해석 가능 특징 추출기 — featurize_action 의 문맥 부분을 이름붙여 재현
def named_feats(spec, state, act):
  """정답 라벨 액션 1개 → {특징명: 값} (PUSH/OPEN 의 디바이스 문맥만)."""
  ctx = Ctx(state)
  kind = act[0]
  dev = act[1] if kind in ('PUSH', 'OPEN') else None
  if dev is None:
    return {}
  roles = ctx.roles
  out = {}
  idx = dev_idx(spec, dev)
  out['c_idx'] = round(idx / 5.0, 3)
  out['c_rel'] = round((idx - ctx.next_out) / 5.0, 3)
  # c_tgtrel (열린 타깃 대비)
  if ctx.target_rank is not None:
    dr = roles['out_rank'].get(dev)
    if dr is not None:
      out['c_tgtrel'] = round(max(min(dr - ctx.target_rank, 1), -3) / 3.0, 3)
  # c_opengap (OPEN-선택)
  if ctx.max_uncoiled_rank is not None:
    dr2 = roles['out_rank'].get(dev)
    if dr2 is not None:
      out['c_opengap'] = round(min(max(ctx.max_uncoiled_rank - dr2, 0), 3) / 3.0, 3)
  # role_feats 9 — 이름 부여
  rf = role_feats(dev, roles)
  names = [
    'r_start', 'r_has_on', 'r_on_rank', 'r_on_delay',
    'r_has_off', 'r_off_rank', 'r_off_delay', 'r_has_self', 'r_self_rank',
  ]
  for n, v in zip(names, rf):
    out[n] = round(v, 3)
  return out


def collect_support(tasks):
  """학습 정답 라벨 전체 → {특징: 등장 값 집합}."""
  sup = defaultdict(set)
  for t in tasks:
    spec = t.spec
    for state, correct in decompose_with_states(t.reference, spec):
      for k, v in named_feats(spec, state, correct).items():
        sup[k].add(v)
  return sup


def collect_target(task, target):
  """seq4 의 특정 타깃 rung body 정답 액션 → [(act, {특징:값})]."""
  spec = task.spec
  rows = []
  for state, correct in decompose_with_states(task.reference, spec):
    # OPEN target 자체 + 그 rung 조립 중(current_target==target) 액션
    in_target = (
      (correct[0] == 'OPEN' and correct[1] == target)
      or state.current_target == target
    )
    if in_target:
      rows.append((correct, named_feats(spec, state, correct)))
  return rows


def fmt(s):
  vals = sorted(s)
  return '{' + ', '.join(f'{v:g}' for v in vals) + '}'


if __name__ == '__main__':
  train_tasks = (
    make_tasks() + make_chain_curriculum(8, K=2) + make_chain_curriculum(8, K=3)
  )
  seq4 = make_seq4()
  sup = collect_support(train_tasks)

  print('학습 = 8 ref(+seq3) + K=2/3 변형 | 시험 = seq4 Y3(최상단) rung body')
  print('정답 라벨 액션이 받는 문맥 특징의 OOD 점검\n')

  for target in ('Y2', 'Y3'):  # Y2(전이 성공) 대조 + Y3(붕괴)
    print(f'=== seq4 {target} rung body (대조: Y2=전이성공 / Y3=붕괴) ===')
    rows = collect_target(seq4, target)
    print(f'{"act":<22}{"feat":<14}{"값":<10}{"학습 support":<28}OOD')
    print('-' * 88)
    for act, feats in rows:
      astr = str(act)
      first = True
      for k, v in feats.items():
        ood = v not in sup.get(k, set())
        a_disp = astr if first else ''
        flag = '  <-- OOD' if ood else ''
        print(f'{a_disp:<22}{k:<14}{v:<10g}{fmt(sup.get(k, set())):<28}{flag}')
        first = False
      print()
