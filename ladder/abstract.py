"""abstract.py — 복원된 실명 IR → 역할 정규화 + 스펙 도출.

[ 왜 ]
  il_parse 가 복원한 회로는 프로젝트 실명(M_Detect, Em_Sim...). R&D 학습기는
  이름이 아니라 역할(입력/출력/타이머)로 본다 (featurizer 가 역할 기반). 실명을
  표준 역할(X0../Y0../T0..)로 바꿔야 합성 모티프와 같은 형식이 되어 변형
  커리큘럼·발견 시험에 흘릴 수 있다.

[ 무엇 ]
  abstract_roles(prog) → (역할 정규화 prog, 실명→역할 맵)
    - 출력 Y: 코일 타깃 (OUT/SET/RST), 코일 첫 등장 순
    - 타이머 T: Timer 노드 이름
    - 입력 X: 코일·타이머·특수릴레이 아닌 접점 리프 (등장 순). 비교 원자도 X.
    - 특수릴레이(SM*): 상수라 그대로 유지 (always-on)
  derive_spec(prog) → Spec (추상 회로를 레퍼런스로 시뮬 → 기대 trace).
    추출 모티프를 커리큘럼/발견 시험에 넣기 위한 최소 스펙.
"""

from ladder.il_parse import is_compare_device
from ladder.search import Scenario, Spec
from ladder.sim import (
  And,
  Coil,
  Contact,
  Logic,
  Or,
  Program,
  Pulse,
  Rung,
  Timer,
  simulate,
)


def _is_special(dev: str) -> bool:
  """SM400/SM401 등 특수 릴레이 — 상수(always-on)라 역할 부여 제외."""
  return dev.startswith('SM')


def _classify(prog: Program):
  """prog → (outputs, inputs, timers) 역할별 디바이스 (등장 순, 유니크)."""
  outputs: list[str] = []
  for r in prog.rungs:
    if r.coil.op in ('OUT', 'SET', 'RST') and r.coil.device not in outputs:
      outputs.append(r.coil.device)

  timers: list[str] = []
  inputs: list[str] = []
  out_set = set(outputs)

  def visit(node: Logic):
    if isinstance(node, Contact):
      d = node.device
      if d in out_set or _is_special(d) or d in inputs:
        return
      inputs.append(d)  # 비교 원자([a op b]) 포함 — 회로가 읽는 불값
    elif isinstance(node, Timer):
      if node.name not in timers:
        timers.append(node.name)
      visit(node.input)
    elif isinstance(node, Pulse):
      visit(node.input)
    else:  # And / Or
      for a in node.args:
        visit(a)

  for r in prog.rungs:
    visit(r.logic)
  return outputs, inputs, timers


def abstract_roles(prog: Program) -> tuple[Program, dict[str, str]]:
  """실명 IR → 역할 정규화 IR + {실명: 역할} 맵."""
  outputs, inputs, timers = _classify(prog)
  role: dict[str, str] = {}
  for i, d in enumerate(outputs):
    role[d] = f'Y{i}'
  for i, d in enumerate(inputs):
    role[d] = f'X{i}'
  for i, t in enumerate(timers):
    role[t] = f'T{i}'
  # 특수 릴레이는 자기 자신으로 (상수 유지)

  def rn(d: str) -> str:
    return role.get(d, d)

  def sub(node: Logic) -> Logic:
    if isinstance(node, Contact):
      return Contact(rn(node.device), node.mode)
    if isinstance(node, Timer):
      return Timer(rn(node.name), node.preset, sub(node.input))
    if isinstance(node, Pulse):
      return Pulse(rn(node.name), sub(node.input))
    return type(node)([sub(a) for a in node.args])

  rungs = [
    Rung(Coil(rn(r.coil.device), r.coil.op, list(r.coil.operands)), sub(r.logic))
    for r in prog.rungs
  ]
  return Program(rungs), role


def _input_patterns(inputs: list[str], n_scans: int) -> list[list[dict]]:
  """입력별 펄스(0→1→0) + 전체 ON + 전체 OFF 패턴 — 각 입력 자극."""
  patterns = []
  # 각 입력 단독 펄스 (래치/엣지 자극)
  for k in inputs:
    trace = []
    for s in range(n_scans):
      trace.append({i: (1 if (i == k and s in (1, 2)) else 0) for i in inputs})
    patterns.append(trace)
  # 전체 ON 유지 / 전체 OFF
  patterns.append([{i: 1 for i in inputs} for _ in range(n_scans)])
  patterns.append([{i: 0 for i in inputs} for _ in range(n_scans)])
  return patterns


def derive_spec(prog: Program, n_scans: int = 6) -> Spec:
  """역할 정규화 prog 를 레퍼런스로 시뮬 → 스펙 (입력 자극별 기대 출력).

  특수 릴레이(SM*)는 상수 ON 으로 매 스캔 주입 (production 의미). 추출
  모티프가 커리큘럼/발견 시험에 들어갈 최소 스펙 — 레퍼런스는 정의상 통과.
  """
  outputs, inputs, _ = _classify(prog)
  specials = sorted(
    {
      c.device
      for r in prog.rungs
      for c in _contacts(r.logic)
      if _is_special(c.device)
    }
  )

  scenarios = []
  for pat in _input_patterns(inputs, n_scans):
    trace = [{**step, **{sm: 1 for sm in specials}} for step in pat]
    expected = simulate(prog, trace, outputs)
    scenarios.append(Scenario(input_trace=trace, expected=expected))
  return Spec(inputs=inputs, outputs=outputs, internals=[], scenarios=scenarios)


def _contacts(node: Logic):
  if isinstance(node, Contact):
    yield node
  elif isinstance(node, (Timer, Pulse)):
    yield from _contacts(node.input)
  else:
    for a in node.args:
      yield from _contacts(a)
