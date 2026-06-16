"""
motif_catalog.py — 실전 빌더 모티프 IL → R&D IR 카탈로그

[ 왜 ]
  실전 ladder_util 모티프 프리미티브(_motif_dump.py 가 덤프)를 il_parse 로
  R&D IR 로 복원 → 모티프 어휘 목록화 + coverage/logic_share 측정. 손제작
  합성 모티프 대신 실전 분포로 변형 커리큘럼을 채우기 위한 1차 채굴.

[ 입력 ]
  tools/dump_builder_il.py 가 백엔드 env 에서 추출한 fixture
  (data/builder_motifs.json). 다른 경로는 argv[1] 로.
  재추출: uv run --directory <synex_web>/backend python \\
            ~/rnd-ml/tools/dump_builder_il.py > data/builder_motifs.json

[ 실행 ]
  uv run experiments/motif_catalog.py [json_path]
"""

import json
import sys
from pathlib import Path

from ladder.il_parse import il_to_program
from ladder.search import program_str

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = str(ROOT / 'data' / 'builder_motifs.json')


def main():
  path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON)
  if not path.exists():
    print(f'JSON 없음: {path}\n(backend 에서 _motif_dump.py 먼저 실행)')
    return
  # utf-8-sig: PowerShell 로 재추출 시 붙는 BOM 도 견디게 (CRLF 는 json 무관)
  motifs = json.loads(path.read_text(encoding='utf-8-sig'))
  print(f'실전 모티프 {len(motifs)}종 — IL → R&D IR 카탈로그\n')

  rows = []
  for name, il in motifs.items():
    items = [tuple(x) for x in il]
    res = il_to_program(items)
    print(f'━━ {name} ━━')
    print(program_str(res.program) or '  (빈 프로그램)')
    flags = []
    if res.skipped:
      flags.append(f'미표현 {[i[0] for i in res.skipped]}')
    if res.timers:
      flags.append(f'timers {list(res.timers)}')
    print(
      f'  coverage={res.coverage:.0%} logic_share={res.logic_share:.0%}'
      + (f' | {" | ".join(flags)}' if flags else '')
    )
    print()
    rows.append((name, res.coverage, res.logic_share, len(res.skipped)))

  # 요약 테이블
  print('=' * 56)
  print(f'{"motif":<26}{"cov":>6}{"logic":>8}{"미표현":>8}')
  print('-' * 56)
  for name, cov, logic, nskip in rows:
    print(f'{name:<26}{cov:>6.0%}{logic:>8.0%}{nskip:>8}')


if __name__ == '__main__':
  main()
