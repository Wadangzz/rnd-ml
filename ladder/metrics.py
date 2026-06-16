"""metrics.py — 실험 성능 지표 누적 기록 (append-only JSONL).

각 측정 1건(= experiment × variant × seed)이 한 레코드. 한 번의 프로세스
실행은 동일 RUN_ID 로 묶인다 (plot 에서 시점 단위 집계용). 정본 history 는
`metrics/history.jsonl`, 차트는 tools/plot_metrics.py 가 그린다.

왜 JSONL append-only:
  - 실험은 시간 축으로 누적된다 (featurizer 수술 → 재측정 반복). 덮어쓰기가
    아니라 누적이라야 "어제 0.971 → 오늘 1.0" 같은 개선 궤적이 남는다.
  - 스키마 free-form (extra 로 임의 필드) — 실험마다 추가 지표가 다르다.
"""

import datetime
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / 'metrics'
HISTORY = METRICS_DIR / 'history.jsonl'

# 한 프로세스 실행 = 한 RUN_ID (초 단위 타임스탬프). 같은 invocation 의
# 여러 시드/variant 를 plot 에서 한 시점으로 묶는다.
RUN_ID = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def log_run(
  experiment: str,
  variant: str,
  seed: int,
  found: int,
  best_acc: float,
  *,
  ref_size: int | None = None,
  prog_size: int | None = None,
  note: str = '',
  extra: dict | None = None,
) -> dict:
  """측정 1건 기록 → history.jsonl 한 줄 append.

  found  : 발견 횟수 (0/None = 미발견/고원). discovered 는 found>0.
  best_acc: 미발견 시 도달 최고 accuracy (고원 높이).
  note   : featurizer/그래머 버전 같은 자유 태그 (개선 궤적 구분용).
  """
  METRICS_DIR.mkdir(exist_ok=True)
  n_found = int(found or 0)  # 미발견은 runner 가 0 또는 None 으로 반환
  rec = {
    'run_id': RUN_ID,
    'ts': datetime.datetime.now().isoformat(timespec='seconds'),
    'experiment': experiment,
    'variant': variant,
    'seed': int(seed),
    'found': n_found,
    'discovered': n_found > 0,
    'best_acc': round(float(best_acc), 4),
    'ref_size': ref_size,
    'prog_size': prog_size,
    'note': note or os.environ.get('METRICS_NOTE', ''),
  }
  if extra:
    rec.update(extra)
  with HISTORY.open('a', encoding='utf-8') as f:
    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
  return rec


def load_history(path: Path | None = None) -> list[dict]:
  """history.jsonl 전체 로드 (없으면 빈 리스트)."""
  p = path or HISTORY
  if not p.exists():
    return []
  out = []
  for line in p.read_text(encoding='utf-8').splitlines():
    line = line.strip()
    if line:
      out.append(json.loads(line))
  return out
