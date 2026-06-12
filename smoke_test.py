"""
smoke_test.py — 벤치마크 인프라 자가 점검 (예산 소액, 수 초 내 완료)

  - make_tasks: intent assert + 퇴화 가드 + ref acc 1.0 + 마스킹 비율
  - 네 방법(random/mcts/mcts_w/gp)이 크래시 없이 도는지
  - SET/RST EMIT 액션이 실제로 합법 액션에 등장하는지 (seq2)
  - 단순화 패스 동작 보존 (무작위 프로그램 property 검사)

실행:  python smoke_test.py  (폴더 안에서)
"""

import random

from ladder.benchmark import METHODS, make_tasks, random_program_ext
from ladder.mcts import BuildState
from ladder.search import (
    accuracy,
    evaluate,
    find_coil_conflicts,
    program_str,
)
from ladder.simplify import shrink_program, simplify_program

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
st.apply(('PUSH', 'X0', 'NO'))
ops = {a[-1] for a in st.legal_actions() if a[0] == 'EMIT'}
assert ops == {'OUT', 'SET', 'RST'}, f'EMIT ops: {ops}'
print(f'[EMIT OK] seq2 ops={sorted(ops)}')

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

print('smoke test 통과')
