"""
smoke_test.py — 벤치마크 인프라 자가 점검 (예산 소액, 수 초 내 완료)

  - make_tasks: intent assert + 퇴화 가드 + ref acc 1.0 + 마스킹 비율
  - 네 방법(random/mcts/mcts_w/gp)이 크래시 없이 도는지
  - SET/RST EMIT 액션이 실제로 합법 액션에 등장하는지 (seq2)
  - 단순화 패스 동작 보존 (무작위 프로그램 property 검사)

실행:  python smoke_test.py  (폴더 안에서)
"""

import random
import tempfile
from pathlib import Path

from ladder.abstract import abstract_roles, derive_spec
from ladder.benchmark import METHODS, make_tasks, random_program_ext
from ladder.il_parse import il_to_program
from ladder.mcts import BuildState
from ladder.search import (
    accuracy,
    evaluate,
    find_coil_conflicts,
    program_size,
    program_str,
)
from ladder.simplify import shrink_program, simplify_program
from ladder.sim import Memory, scan_once

tasks = make_tasks()

for t in tasks:
    masked = sum(
        1
        for sc in t.spec.scenarios
        for w in sc.expected
        for v in w.values()
        if v is None
    )
    total = sum(len(w) for sc in t.spec.scenarios for w in sc.expected)
    acc = accuracy(t.reference, t.spec)
    assert acc >= 1.0, f'{t.name} ref acc={acc}'
    assert not find_coil_conflicts(t.reference), f'{t.name} ref 이중 코일!'
    print(f'[spec OK] {t.name:<12} masked {masked}/{total}')

seq2 = next(t for t in tasks if t.name == 'seq2')
st = BuildState(seq2.spec, **seq2.mcts_kwargs)
# 새 그래머: rung 닫힘 국면(fresh)에 OPEN(coil, op) 이 합법. SET/RST 는
# 타깃 선언 시점(OPEN)에 op 로 붙는다 (구 EMIT 의 op 가 앞으로 이동).
ops = {a[2] for a in st.legal_actions() if a[0] == 'OPEN'}
assert ops == {'OUT', 'SET', 'RST'}, f'OPEN ops: {ops}'
print(f'[OPEN OK] seq2 ops={sorted(ops)}')

for t in tasks:
    for m, runner in METHODS:
        found, acc, prog = runner(t, 300, 0)
        assert prog is not None, f'{t.name}/{m}: prog None'
        assert not find_coil_conflicts(prog), (
            f'{t.name}/{m} 이중 코일:\n{program_str(prog)}'
        )
        print(f'[run OK] {t.name:<12} {m:<7} acc={acc:.3f}')

# 단순화 패스(동작 보존) + 생성기 이중 코일 금지 property 검사
N_PROPERTY = 300
for t in tasks:
    rng = random.Random(7)
    for _ in range(N_PROPERTY):
        p = random_program_ext(t.spec, rng, t.gen_cfg)
        assert not find_coil_conflicts(p), (
            f'{t.name} 생성기 이중 코일:\n{program_str(p)}'
        )
        s = simplify_program(p, t.spec)
        ev_p, ev_s = evaluate(p, t.spec), evaluate(s, t.spec)
        assert ev_p == ev_s, (
            f'{t.name} 단순화가 동작을 바꿈: {ev_p} → {ev_s}\n'
            f'[원본]\n{program_str(p)}\n[단순화]\n{program_str(s)}'
        )
    print(f'[simplify OK] {t.name:<12} {N_PROPERTY}개 동작 보존 + 코일 규칙')

# 스펙 축소: evaluate 보존이 구조상 보장되지만 IR 조작 버그(크래시/경로
# 꼬임)를 잡기 위해 소량 property 검사
N_SHRINK = 40
for t in tasks:
    rng = random.Random(11)
    for _ in range(N_SHRINK):
        p = random_program_ext(t.spec, rng, t.gen_cfg)
        s = shrink_program(p, t.spec)
        assert evaluate(p, t.spec) == evaluate(s, t.spec), (
            f'{t.name} 스펙 축소가 evaluate 를 바꿈'
        )
    print(f'[shrink OK] {t.name:<12} {N_SHRINK}개 스펙 보존')

# ── 오늘 작업 (2026-06-16): IL 파서 → 역할 추상화 → 지표 기록 ──

# 실전 actuator 센서 IL (il_actuator_probe 와 동일)
ACT_IL = [
    ('LDI', 'Em_Sim'), ('AND', 'T_Act'), ('LD', 'Em_Sim'),
    ('LD', 'M_Cmd'), ('OR', 'M_Det'), ('ANB',), ('ORB',), ('OUT', 'M_Det'),
]

# (1) il_parse: 자기유지 복원 — Em_Sim=1 에서 M_Cmd 1펄스 후 M_Det 래치
res = il_to_program([tuple(x) for x in ACT_IL])
assert len(res.program.rungs) == 1, f'rung 수 {len(res.program.rungs)}'
assert res.program.rungs[0].coil.device == 'M_Det'
mem = Memory()
for s, cmd in enumerate([0, 1, 0, 0]):
    mem.set('Em_Sim', True)
    mem.set('T_Act', False)
    mem.set('M_Cmd', bool(cmd))
    scan_once(res.program, mem)
    if s >= 1:
        assert mem.get('M_Det'), f'scan{s} 래치 실패'
print('[il OK] actuator 래치 복원 + 시뮬')

# (1b) 견고성: 단편/불완전 IL 이 크래시 대신 skipped 집계
frag = il_to_program([('ANB',), ('AND', 'X0'), ('ORB',)])
assert len(frag.skipped) == 3, f'단편 skipped={frag.skipped}'
print(f'[il OK] 단편 스택언더플로우 방어 (skipped {len(frag.skipped)})')

# (2) abstract: 구조 보존 + derive_spec 레퍼런스 자기 통과
abs_prog, role = abstract_roles(res.program)
assert program_size(abs_prog) == program_size(res.program), '추상화가 구조 변경'
assert role['M_Det'] == 'Y0', f'코일 역할 {role.get("M_Det")}'
spec = derive_spec(abs_prog)
a, v = evaluate(abs_prog, spec)
assert a == 1.0 and v == 0, f'derive_spec 레퍼런스 실패 acc={a} viol={v}'
print(f'[abstract OK] 역할정규화 {program_str(abs_prog).strip()} + 스펙 통과')

# (3) metrics: log/load 라운드트립 + found=None 안전 (임시 경로, 실 history 무오염)
import ladder.metrics as _M

_tmp = Path(tempfile.mkdtemp())
_M.METRICS_DIR, _M.HISTORY = _tmp, _tmp / 'h.jsonl'
rec = _M.log_run('smoke', 'v', 0, None, 0.5)  # found=None → 0
assert rec['found'] == 0 and rec['discovered'] is False, rec
hist = _M.load_history(_M.HISTORY)
assert len(hist) == 1 and hist[0]['best_acc'] == 0.5, hist
print('[metrics OK] log/load 라운드트립 + None 안전')

print('smoke test 통과')
