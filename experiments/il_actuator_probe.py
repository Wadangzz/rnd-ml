"""
il_actuator_probe.py — 실전 빌더 IL → R&D IR 역복원 검증 (프로토타입)

[ 왜 ]
  "실전 LadderBlock 빌더를 R&D 로 분석 가능한가?" 의 첫 실증. 실제
  `create_actuator_ladder._build_sensor_rows` 가 뱉는 니모닉을 그대로 옮겨
  il_to_program 으로 R&D IR 로 복원 → (1) 자기유지 관용구가 보이는지
  (2) 시뮬레이터에서 래치 동작하는지 (3) coverage 확인.

[ 실행 ]
  uv run experiments/il_actuator_probe.py
"""

from ladder.il_parse import il_to_program
from ladder.search import program_str
from ladder.sim import Memory, scan_once

# create_actuator_ladder._build_sensor_rows 의 실제 emit 순서 (operand 실명 유지).
# lb.ldi(Em_Sim) / and_(T_Action) / ld(Em_Sim) / ld(M_Command) / or_(M_Detect)
# / anb() / orb() / out(M_Detect)  — siblings 체인은 없다고 가정(단일 action).
ACTUATOR_SENSOR_IL = [
  ('STMT', '[Title]Sensor Input'),
  ('LDI', 'Em_Sim'),
  ('AND', 'T_Action_Time'),
  ('LD', 'Em_Sim'),
  ('LD', 'M_Command'),
  ('OR', 'M_Detect'),
  ('ANB',),
  ('ORB',),
  ('OUT', 'M_Detect'),
]

# 타이머 구동 + 읽기가 섞인 케이스 (input rows: 'OUT T_Action_Time D_Action_V'
# 는 타이머 코일). 별도 rung 에서 T 를 읽는 시나리오를 합쳐 인라인 검증.
TIMER_CASE_IL = [
  ('LD', 'X0'),
  ('AND', 'X1'),
  ('OUT', 'T_Action_Time', 'K30'),  # 타이머 코일 (구동 = X0*X1, preset 30)
  ('LD', 'T_Action_Time'),  # 타이머 done 비트 읽기 → Timer 노드로 인라인
  ('OUT', 'Y0'),
]

# (C) 비교 명령 — create_vision_ladder 의 P↔V 핸드셰이크 패턴 (LD= K1 busy / AND=)
VISION_COMPARE_IL = [
  ('LD=', 'K1', 'D_busy_v2p'),  # busy 워드 == 1 → 불 원자
  ('AND=', 'K0', 'D_trigger_p2v'),  # AND 비교
  ('OUT', 'M_vision_ready'),
]

# (A) 데이터 액션 — 조건 게이트 후 MOV (vision 위치값 전송 류). 조건 motif 보존.
DATA_MOVE_IL = [
  ('LD', 'M_Detect'),
  ('AND', 'M_vision_ready'),
  ('MOV', 'D_pos_src', 'D_pos_dst'),  # 데이터 액션 → 불투명 코일
  ('LD', 'M_Detect'),
  ('DMOV', 'D_v_src', 'D_v_dst'),
]


def show(title, il):
  print(f'\n=== {title} ===')
  print('IL (니모닉):')
  for it in il:
    print('   ' + ' '.join(it))
  res = il_to_program(il)
  print('\n복원된 R&D IR:')
  print(program_str(res.program))
  print(
    f'\ncoverage = {res.coverage:.0%} | logic_share = {res.logic_share:.0%}'
    f' | skipped = {res.skipped or "없음"}'
  )
  if res.timers:
    print(f'timers = {list(res.timers)}')
  return res


if __name__ == '__main__':
  res = show('ACTUATOR Sensor Input (실전 IL)', ACTUATOR_SENSOR_IL)

  # 자기유지 동작 확인 — Em_Sim=1(실모드)에서 M_Command 1펄스 후 M_Detect 래치?
  print('\n--- 래치 동작 시뮬 (Em_Sim=1 고정) ---')
  mem = Memory()
  mem.set('Em_Sim', True)
  mem.set('T_Action_Time', False)
  trace_in = [
    {'M_Command': 0},  # 0: idle
    {'M_Command': 1},  # 1: 지령 펄스
    {'M_Command': 0},  # 2: 지령 해제 → 래치 유지되어야
    {'M_Command': 0},  # 3
  ]
  for i, step in enumerate(trace_in):
    for k, v in step.items():
      mem.set(k, bool(v))
    mem.set('Em_Sim', True)
    scan_once(res.program, mem)
    print(
      f'  scan{i}: M_Command={mem.get("M_Command"):d} '
      f'-> M_Detect={mem.get("M_Detect"):d}'
    )

  show('타이머 인라인 (OUT T K → LD T)', TIMER_CASE_IL)
  show('(C) 비교 명령 → 불 원자 (VISION 핸드셰이크)', VISION_COMPARE_IL)
  show('(A) 데이터 액션 → 불투명 코일 (조건 motif 보존)', DATA_MOVE_IL)
