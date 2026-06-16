"""
motif_discover.py — 추출·추상화한 실전 모티프를 합성기가 재발견하는지 시험

[ 왜 ]
  il_parse → abstract 로 실전 actuator 센서 모티프를 역할 정규화 회로
  `((X0/*X1)+(X0*(X2+Y0)))->Y0` 로 추출했다. 이게 derive_spec 스펙을 타깃으로
  실제로 합성(발견)되는지 — 베이스라인(random/mcts_w/gp) + 학습 prior
  (net-rollout/puct+net) 양쪽으로. end-to-end (IL→IR→추상→스펙→탐색→발견).

[ 실행 ]
  uv run experiments/motif_discover.py                      # action_cond (기본)
  uv run experiments/motif_discover.py --motif actuator/sensor_input
  uv run experiments/motif_discover.py --skip-prior         # 베이스라인만
  uv run experiments/motif_discover.py --budget 20000       # 예산 축소
"""

import json
import sys
from pathlib import Path

from ladder.abstract import abstract_roles, derive_spec
from ladder.benchmark import (
  GenCfg,
  Task,
  make_tasks,
  run_gp,
  run_mcts_w,
  run_random,
)
from ladder.curriculum import make_chain_curriculum, make_motif_curriculum
from ladder.il_parse import il_to_program
from ladder.metrics import log_run
from ladder.parallel import parallel_search
from ladder.policy import build_samples, extract_weights, train
from ladder.search import evaluate, program_size, program_str

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (0, 1, 2)


def make_motif_task(motif_name: str) -> Task:
  """builder_motifs.json 의 실전 모티프 IL → 역할 정규화 Task."""
  motifs = json.loads(
    (ROOT / 'data' / 'builder_motifs.json').read_text(encoding='utf-8-sig')
  )
  if motif_name not in motifs:
    raise SystemExit(f'모티프 없음: {motif_name}\n선택지: {list(motifs)}')
  res = il_to_program([tuple(x) for x in motifs[motif_name]])
  abs_prog, _role = abstract_roles(res.program)
  spec = derive_spec(abs_prog)
  n_out = max(len(spec.outputs), 1)
  gen = GenCfg(max_rungs=n_out, max_depth=5, setrst_p=0.0)
  mk = dict(max_actions=40, max_stack=4, max_rungs=n_out, allow_setrst=False)
  return Task(motif_name, f'추출 모티프 {motif_name}', spec, abs_prog, gen, mk)


def _arg(flag, default):
  if flag in sys.argv:
    return int(sys.argv[sys.argv.index(flag) + 1])
  return default


def _arg_str(flag, default):
  if flag in sys.argv:
    return sys.argv[sys.argv.index(flag) + 1]
  return default


if __name__ == '__main__':
  budget = _arg('--budget', 200_000)
  skip_prior = '--skip-prior' in sys.argv
  skip_base = '--skip-base' in sys.argv
  motif_name = _arg_str('--motif', 'actuator/action_cond')

  task = make_motif_task(motif_name)
  ref_sz = program_size(task.reference)
  a, v = evaluate(task.reference, task.spec)
  exp_name = 'motif_' + motif_name.replace('/', '_')
  print(f'타깃 = 추출 모티프 {motif_name} (ref 크기 {ref_sz})')
  print(program_str(task.reference))
  print(f'레퍼런스 자기 스펙: acc={a:.3f} viol={v}  '
        f'inputs={task.spec.inputs} scenarios={len(task.spec.scenarios)}\n')

  if not skip_base:
    print(f'=== 베이스라인 (예산 {budget:,} × 시드 {SEEDS}) ===')
    for name, runner in [('random', run_random), ('mcts_w', run_mcts_w),
                         ('gp', run_gp)]:
      for s in SEEDS:
        found, acc, prog = runner(task, budget, s)
        stat = f'{found:,}회 발견' if found else f'실패 (acc {acc:.3f})'
        print(f'  {name:<7} seed{s}: {stat}', flush=True)
        log_run(exp_name, name, s, found, acc, ref_size=ref_sz,
                prog_size=program_size(prog) if prog else None, note='extracted')

  if skip_prior:
    sys.exit(0)

  with_motif = '--no-motif-cur' not in sys.argv
  motif_n = _arg('--motif-n', 12)
  label = '8 ref + 체인 K2/3' + (f' + 모티프 변형×{motif_n}' if with_motif else '')
  print(f'\n=== 학습 prior — {label} ===')
  train_tasks = (
    make_tasks() + make_chain_curriculum(8, K=2) + make_chain_curriculum(8, K=3)
  )
  if with_motif:
    # 추출 모티프(=task.reference, 역할 정규화)의 배정 순열 변형. canonical
    # (held-out 타깃)은 생성기가 제외 → 시험 오염 없음.
    train_tasks += make_motif_curriculum(task.reference, n_variants=motif_n)
  samples, _ = build_samples(train_tasks)
  print(f'라벨 {len(samples)}')
  model, _ = train(samples)
  w = extract_weights(model)

  print(f'\n=== 학습 prior 측정 (예산 {budget:,} × 시드 {SEEDS}, 병렬) ===')
  jobs, labels = [], []
  for name, use_prior in [('net-rollout', False), ('puct+net', True)]:
    for s in SEEDS:
      jobs.append((task.spec, task.mcts_kwargs, budget, s, w, use_prior))
      labels.append((name, s))
  prior_note = 'motif_cur' if with_motif else 'chains_only'
  for (name, s), (found, acc, prog) in zip(labels, parallel_search(jobs)):
    stat = f'{found:,}회 발견' if found else f'실패 (acc {acc:.3f})'
    print(f'  {name:<12} seed{s}: {stat}', flush=True)
    log_run(exp_name, name, s, found, acc, ref_size=ref_sz,
            prog_size=program_size(prog) if prog else None, note=prior_note)
    if prog and found:
      print(program_str(prog))
