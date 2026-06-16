"""il_parse.py — GX Works 니모닉 IL → R&D IR 파서 (LadderBlock.emit 의 역방향).

[ 왜 ]
  실전 빌더(`backend/.../export/_ladder/manual/create_*.py`)가 `LadderBlock`
  으로 뱉는 명령어 리스트(LD/AND/OR/ANB/ORB/OUT/SET/RST/타이머)는 R&D IR
  (Contact/And/Or/Timer + Rung)의 스택-머신 직렬화와 정확히 같은 표현이다.
  이 파서로 실전 회로를 IR 로 역복원하면 → 손으로 추측한 합성 모티프 대신
  **실제 모티프 분포**로 변형 커리큘럼을 채울 수 있다 (decompose 의 역방향).

[ 범위 ]
  불 논리 부분만 (Contact/And/Or/Timer/Pulse). MOV/DMOV/연산/비교 같은
  데이터 명령은 현재 IR(불 논리 전용) 밖이라 스킵하고 coverage 로 보고한다.

[ IL 스택 의미론 ]
  LD/LDI/LDP        : 새 결과를 메인 스택에 push (LDI=B접점, LDP=상승펄스)
  AND/ANI/OR/ORI..  : 스택 top 을 접점과 결합 (직렬=And / 병렬=Or)
  ANB/ORB           : top 2개를 pop 해 블록 결합 (중첩 구조)
  MPS/MRD/MPP       : 분기점 저장/재독/복원 (한 논리에서 다중 코일)
  OUT/SET/RST dev   : 현재 top 논리로 코일 emit → Rung
  OUT T K100        : 타이머 코일 (구동 논리 + preset) → 이후 LD T 는 Timer 노드
"""

import copy
from dataclasses import dataclass, field

from ladder.sim import And, Coil, Contact, Logic, Or, Program, Pulse, Rung, Timer

# 데이터/미지원 명령 — 불 논리 IR 밖. 스킵하고 coverage 에 집계.
_DATA_OPS = {'MOV', 'MOVP', 'DMOV', 'DMOVP', 'BMOV', 'FMOV', 'INC', 'DEC'}
_NOP_OPS = {'END', 'MEP', 'MEF', 'INV', 'NOP'}


@dataclass
class ParseResult:
  program: Program
  skipped: list[tuple] = field(default_factory=list)  # 미지원(데이터 등) 명령
  timers: dict = field(default_factory=dict)  # name → (preset, drive_logic)

  @property
  def coverage(self) -> float:
    """emit 된 rung / (rung + 스킵된 데이터 명령) — 1.0 이면 순수 불 논리."""
    emitted = len(self.program.rungs)
    denom = emitted + len(self.skipped)
    return emitted / denom if denom else 1.0


def _merge(node: Logic, contact: Logic, cls) -> Logic:
  """top 을 contact 와 결합 — 같은 종류면 args 에 흡수(좌측 폴딩 평탄화)."""
  if isinstance(node, cls):
    return cls(node.args + [contact])
  return cls([node, contact])


def _is_timer_coil(ops: list[str]) -> bool:
  """`OUT <dev> K100` / `OUT <dev> D5` 처럼 preset operand 가 따라오면 타이머."""
  return len(ops) >= 2 and ops[1][:1] in ('K', 'D', 'H')


def _preset(tok: str) -> int:
  """K100 → 100. D/H 레지스터 preset 은 동적이라 0 (구조만 보존)."""
  return int(tok[1:]) if tok[:1] == 'K' and tok[1:].isdigit() else 0


def il_to_program(items: list[tuple]) -> ParseResult:
  """니모닉 (instr, *operands) 리스트 → ParseResult.

  items 예: [('LDI','Em_Sim'), ('AND','T_Act'), ..., ('OUT','M_Detect')]
  STMT/주석은 ('STMT', text) 로 와도 무시.
  """
  main: list[Logic] = []  # 결과 스택
  aux: list[Logic] = []  # MPS/MRD/MPP 분기점
  rungs: list[Rung] = []
  skipped: list[tuple] = []
  timers: dict = {}

  for item in items:
    instr = item[0]
    ops = list(item[1:])
    dev = ops[0] if ops else None

    if instr in ('LD', 'LDI', 'LDP', 'LDF'):
      mode = 'NC' if instr == 'LDI' else 'NO'
      node: Logic = Contact(dev, mode)
      if instr in ('LDP', 'LDF'):  # 엣지 → Pulse 래핑
        node = Pulse(f'P_{dev}', Contact(dev, 'NO'))
      main.append(node)
    elif instr in ('AND', 'ANI', 'ANDP', 'ANDF'):
      mode = 'NC' if instr == 'ANI' else 'NO'
      main[-1] = _merge(main[-1], Contact(dev, mode), And)
    elif instr in ('OR', 'ORI', 'ORP', 'ORF'):
      mode = 'NC' if instr == 'ORI' else 'NO'
      main[-1] = _merge(main[-1], Contact(dev, mode), Or)
    elif instr == 'ANB':
      b, a = main.pop(), main.pop()
      main.append(_merge(a, b, And))
    elif instr == 'ORB':
      b, a = main.pop(), main.pop()
      main.append(_merge(a, b, Or))
    elif instr == 'MPS':
      aux.append(copy.deepcopy(main[-1]))
    elif instr == 'MRD':
      main[-1] = copy.deepcopy(aux[-1])
    elif instr == 'MPP':
      main[-1] = aux.pop()
    elif instr in ('OUT', 'SET', 'RST'):
      if instr == 'OUT' and _is_timer_coil(ops):
        timers[dev] = (_preset(ops[1]), main[-1])  # 타이머 코일: 구동+preset
      else:
        rungs.append(Rung(Coil(dev, instr), main[-1]))
    elif instr in _DATA_OPS:
      skipped.append(item)  # 데이터 명령 — IR 밖
    elif instr in _NOP_OPS or instr == 'STMT':
      pass  # 무시 (구조 무관)
    else:
      skipped.append(item)  # 미지원 — coverage 에 반영

  prog = _inline_timers(Program(rungs), timers)
  return ParseResult(program=prog, skipped=skipped, timers=timers)


def _inline_timers(prog: Program, timers: dict) -> Program:
  """`OUT T K` 로 정의된 타이머를, 그 T 를 읽는 Contact 자리에 Timer 노드로 치환.

  실전 IL 은 타이머를 코일로 구동(`OUT T0 K100`)하고 별도로 접점(`LD T0`)으로
  읽는다. R&D IR 의 Timer 는 입력 논리를 감싸는 Logic 노드라, 읽는 자리에
  구동 논리를 인라인해 의미를 맞춘다.
  """
  if not timers:
    return prog

  def sub(node: Logic) -> Logic:
    if isinstance(node, Contact) and node.device in timers:
      preset, drive = timers[node.device]
      return Timer(node.device, preset, sub(drive))
    if isinstance(node, (And, Or)):
      return type(node)([sub(a) for a in node.args])
    if isinstance(node, (Timer, Pulse)):
      node.input = sub(node.input)
      return node
    return node

  return Program([Rung(r.coil, sub(r.logic)) for r in prog.rungs])
