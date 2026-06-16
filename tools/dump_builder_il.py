"""dump_builder_il.py — 실전 빌더 모티프 IL 추출 (백엔드 env 전용).

synex_web 백엔드의 ladder_util 모티프 프리미티브를 합성 주소로 구동 →
LadderBlock._items 를 (instr, *operands) IL 로 변환 → JSON 을 stdout 으로.
경로 결합이 없어 어디서 실행하든 stdout 만 rnd-ml fixture 로 저장하면 됨.

이 스크립트는 `app.mitsubishi` 를 import 하므로 **백엔드 uv env** 에서만 돈다:

  uv run --directory <synex_web>/backend python <이 파일 절대경로> \\
    > ~/rnd-ml/data/builder_motifs.json

사람용 로그는 stderr 로 분리 (stdout = 순수 JSON).
"""

import json
import os
import sys

# 스크립트가 rnd-ml 에 있어도 백엔드 env 에서 돌 수 있게 — `--directory backend`
# 로 잡힌 cwd(=backend) 를 import 경로에 넣어 `app` 패키지를 찾게 한다.
sys.path.insert(0, os.getcwd())

from app.mitsubishi.export._ladder.create_actuator_ladder import (
  _build_action_condition_rows,
  _build_action_lamp_rows,
  _build_arrival_timeover_rows,
  _build_auto_condition_rows,
  _build_departure_timeover_rows,
  _build_input_rows,
  _build_interlock_rows,
  _build_solenoid_output_rows,
)
from app.mitsubishi.export._ladder.ladder_util import (
  LadderBlock,
  build_addrs_chain_block,
  build_chain_block,
  build_interlock_block,
  build_step_chain_block,
)


def il_of(lb: LadderBlock) -> list[list[str]]:
  out = []
  for item in lb._items:
    if item[0] == 'INSTR':
      _, instr, io = item
      out.append([instr, *io.split()] if io else [instr])
    else:
      out.append(['STMT', item[1]])
  return out


def closed(lb: LadderBlock, coil: str = 'Y_il') -> LadderBlock:
  """조건 블록에 테스트 코일 OUT 을 붙여 완결 rung 으로."""
  lb.out(coil)
  return lb


cases = {
  'interlock/single_and': closed(
    build_interlock_block(
      [{'_row': 0, 'addresses': ['M_A']}, {'_row': 0, 'addresses': ['M_B']}]
    )
  ),
  'interlock/or_group': closed(
    build_interlock_block([{'_row': 0, 'addresses': ['M_A', 'M_B', 'M_C']}])
  ),
  'interlock/multi_row': closed(
    build_interlock_block(
      [
        {'_row': 0, 'addresses': ['M_A', 'M_B']},
        {'_row': 1, 'addresses': ['M_C']},
      ]
    )
  ),
  'interlock/none_fallback': closed(build_interlock_block(None)),
  'chain/ani_series': closed(build_chain_block(['M_A', 'M_B', 'M_C'], 'ANI')),
  'chain/and_series': closed(build_chain_block(['X0', 'X1', 'X2'], 'AND')),
  'chain/or_parallel': closed(build_chain_block(['X0', 'X1'], 'OR')),
  'chain/addrs_ani': closed(
    build_addrs_chain_block(
      [{'k': 'M_A'}, {'k': 'M_B'}], 'k', op='ANI', close='ANB'
    )
  ),
  'chain/step_inv': closed(build_step_chain_block(['M_S1', 'M_S2'], 'ANI')),
}

# ── actuator 8개 섹션 — 실전 멀티-rung 조합 모티프 (1 action 합성 구동) ──
# 서브함수는 plain dict/list 만 받음 (MemoryLookup 우회). 1 control module,
# 1 action 으로 구동 → 각 섹션의 코어 모티프 (코일은 함수가 자체 emit).
_A = {
  'Input': 'X_in', 'M_Command': 'M_cmd', 'M_Detect': 'M_det',
  'T_Action_Time': 'T_at', 'D_Action_V': 'D_av', 'M_Int': 'M_int',
  'M_Auto': 'M_auto', 'L_Action_LFS': 'L_lfs', 'L_Action_LFC': 'L_lfc',
  'Output': 'Y_out', 'T_Action_FS': 'T_fs', 'T_Action_FC': 'T_fc',
}
_acts = [{'id': 'a1'}]


def _sec(fn):
  """섹션 함수를 1-action 으로 구동 → LadderBlock."""
  lb = LadderBlock()
  fn(lb)
  return lb


cases.update({
  'actuator/sensor_input': _sec(
    lambda lb: _build_input_rows(lb, 'M_simul', 'M_auto_run', [_A])
  ),
  'actuator/interlock': _sec(
    lambda lb: _build_interlock_rows(lb, {}, _acts, [_A])  # 인터락 없음 → SM400
  ),
  'actuator/auto_cond': _sec(
    lambda lb: _build_auto_condition_rows(lb, [], _acts, [_A])  # step 없음
  ),
  'actuator/action_cond': _sec(
    lambda lb: _build_action_condition_rows(lb, 'M_auto_run', 'M_manual', [_A])
  ),
  'actuator/solenoid': _sec(lambda lb: _build_solenoid_output_rows(lb, [_A])),
  'actuator/action_lamp': _sec(lambda lb: _build_action_lamp_rows(lb, [_A])),
  'actuator/arrival_timeover': _sec(
    lambda lb: _build_arrival_timeover_rows(lb, [_A])
  ),
  'actuator/departure_timeover': _sec(
    lambda lb: _build_departure_timeover_rows(lb, [_A])
  ),
})

out = {name: il_of(lb) for name, lb in cases.items()}
print(json.dumps(out, ensure_ascii=False, indent=2))  # stdout = JSON
print(f'덤프 {len(out)} 모티프', file=sys.stderr)
