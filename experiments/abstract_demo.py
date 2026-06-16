"""
abstract_demo.py — 실명 IR → 역할 정규화 + 스펙 도출 시연/검증

[ 무엇 ]
  ① actuator 센서 모티프(실전 IL) 를 역할 정규화 → 합성 모티프 형식 확인
  ② 카탈로그 모티프(builder_motifs.json) 일괄 역할 정규화
  ③ derive_spec → 레퍼런스(추상 회로)가 자기 스펙 통과하는지 (acc==1.0 sanity)

[ 실행 ]
  uv run experiments/abstract_demo.py
"""

import json
from pathlib import Path

from ladder.abstract import abstract_roles, derive_spec
from ladder.il_parse import il_to_program
from ladder.search import evaluate, program_str

ROOT = Path(__file__).resolve().parent.parent

# il_actuator_probe 와 동일한 실전 actuator 센서 IL
ACTUATOR_SENSOR_IL = [
  ('LDI', 'Em_Sim'),
  ('AND', 'T_Action_Time'),
  ('LD', 'Em_Sim'),
  ('LD', 'M_Command'),
  ('OR', 'M_Detect'),
  ('ANB',),
  ('ORB',),
  ('OUT', 'M_Detect'),
]


def main():
  print('=== ① actuator 센서 모티프 — 역할 정규화 ===')
  res = il_to_program([tuple(x) for x in ACTUATOR_SENSOR_IL])
  print('실명 IR:')
  print(program_str(res.program))
  abs_prog, role = abstract_roles(res.program)
  print('\n역할 정규화 IR:')
  print(program_str(abs_prog))
  print(f'\n매핑: {role}')

  print('\n=== ③ derive_spec → 레퍼런스 자기 스펙 통과 (sanity) ===')
  spec = derive_spec(abs_prog)
  acc, viol = evaluate(abs_prog, spec)
  print(
    f'inputs={spec.inputs} outputs={spec.outputs} '
    f'scenarios={len(spec.scenarios)}'
  )
  print(f'레퍼런스 evaluate: acc={acc:.3f} viol={viol}  '
        f'{"✅ 통과" if acc == 1.0 and viol == 0 else "❌ 실패"}')

  print('\n=== ② 카탈로그 모티프 일괄 역할 정규화 ===')
  jpath = ROOT / 'data' / 'builder_motifs.json'
  if jpath.exists():
    motifs = json.loads(jpath.read_text(encoding='utf-8-sig'))
    for name, il in motifs.items():
      r = il_to_program([tuple(x) for x in il])
      ap, rmap = abstract_roles(r.program)
      print(f'  {name:<26} {program_str(ap).strip()}')
  else:
    print('  (builder_motifs.json 없음 — motif_catalog 먼저)')


if __name__ == '__main__':
  main()
