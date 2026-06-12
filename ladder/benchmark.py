"""
ladder_benchmark.py — 난이도 사다리 벤치마크: random / mcts / mcts_w / gp 비교

[ 목적 ]
  과제 난이도를 단계적으로 올리면서 네 탐색기의 '완벽해 발견까지
  시뮬레이터 호출 수'를 측정한다.
    - 무작위가 무너지는 지점  = MCTS/가중 롤아웃이 밥값을 시작하는 지점
    - MCTS도 무너지는 지점    = rung 교차(GP)가 정당화되는 지점
    - GP도 무너지는 지점      = 정책망(신경망)이 필요해지는 지점 (seq3)

[ 난이도 사다리 ]
  1. self_hold    자기유지 (기동/정지)                    — 조합+피드백
  2. interlock    전후진 인터로크 (동시 ON 금지, 선입력 우선)— 출력 2개 상호참조
  3. one_shot     원샷 (버튼 길이 무관 1스캔 펄스)          — PLS 필요
  4. delayed_off  지연 정지 (OFF 후 preset 스캔 뒤 꺼짐)    — TON + 자기유지
  5. flicker      플리커 (ON 2스캔 / OFF 2스캔 발진)        — TON 2개 교차결합
  6. seq2         2단 시퀀스 (기동→단계1→센서→단계2→센서→완료)— 래치 체인
  7. actuator     전후진 실린더 (지령 latch+센서 해제+인터로크)— self_hold×interlock
  8. seq3         3단 시퀀스 (seq2 +1단)                    — GP 한계 탐침

[ 방법론 ]
  - 과제마다 사람이 짠 '레퍼런스 해'를 두고, 기대 trace는 레퍼런스를
    simulate해서 도출 → 스펙 충족 가능성이 구조적으로 보장됨.
  - 레퍼런스가 의도대로 움직이는지는 과제별 intent assert로 별도 검증.
  - intent assert 통과 후 전이 스캔 don't-care 마스킹 적용
    (interlock/delayed_off/seq2 — 레퍼런스 타이밍 부산물 박제 방지.
     one_shot은 펄스가 전이 스캔 자체라 제외, 2026-06-11).
  - 완벽해 판정은 accuracy==1.0 (크기 페널티 제외 — 큰 정답 회로 누락 방지).
  - 예산 = score/accuracy 호출 수. 두 탐색기 동일 조건.

[ 실행 ]
  python -m ladder_benchmark            # 전체
  python -m ladder_benchmark interlock seq2
"""

import random
import sys
from dataclasses import dataclass
from itertools import count

from ladder.gp import gp_search
from ladder.mcts import (
    BuildState,
    mcts_search,
    weighted_rollout,
)
from ladder.search import (
    Scenario,
    Spec,
    accuracy,
    coil_allowed,
    evaluate,
    make_self_hold_spec,
    program_size,
    program_str,
)
from ladder.sim import (
    And,
    Coil,
    Contact,
    Or,
    Program,
    Pulse,
    Rung,
    Timer,
    simulate,
)

# ---------- 과제 정의 ----------


@dataclass
class GenCfg:
    """무작위 생성기 설정 (과제별 재료/한도)"""

    max_rungs: int = 3
    max_depth: int = 3
    timer_presets: tuple[int, ...] = ()
    allow_pulse: bool = False
    wrap_p: float = 0.15  # 서브트리를 TON/PLS로 감쌀 확률
    setrst_p: float = 0.0  # rung 코일을 SET/RST로 만들 확률


@dataclass
class Task:
    name: str
    desc: str
    spec: Spec
    reference: Program
    gen_cfg: GenCfg
    mcts_kwargs: dict  # BuildState 생성자 인자


def derive_spec(inputs, outputs, internals, reference, input_traces) -> Spec:
    """기대 trace를 레퍼런스 시뮬레이션으로 도출"""
    scenarios = [
        Scenario(t, simulate(reference, t, outputs)) for t in input_traces
    ]
    return Spec(
        inputs=inputs, outputs=outputs, internals=internals, scenarios=scenarios
    )


def mask_transition_scans(spec: Spec, width: int = 1) -> Spec:
    """입력이 바뀐 스캔(부터 width스캔)의 기대값을 don't-care(None)로 치환.

    레퍼런스 simulate로 도출한 기대 trace에는 레퍼런스의 스캔 타이밍
    부산물(rung 순서, 중간 릴레이 유무에 따른 1스캔 차이)까지 박제된다.
    의도는 같고 전이 타이밍만 다른 대안 회로를 감점하지 않도록
    전이 스캔을 채점에서 제외한다 (accuracy()는 None을 skip).

    ** intent assert는 반드시 마스킹 전 spec에서 실행할 것. **
    ** one_shot처럼 출력이 전이 스캔 자체에 실리는 과제에는 쓰면 안 됨
       (스펙의 신호가 통째로 마스킹돼 빈 회로가 통과한다). **
    """
    new_scenarios = []
    for sc in spec.scenarios:
        cur: dict[str, int] = {}
        trans = set()
        for t, upd in enumerate(sc.input_trace):
            if any(cur.get(k, 0) != v for k, v in upd.items()):
                trans.add(t)
            cur.update(upd)
        expected = []
        for t, want in enumerate(sc.expected):
            if any((t - d) in trans for d in range(width)):
                expected.append({k: None for k in want})
            else:
                expected.append(dict(want))
        new_scenarios.append(Scenario(sc.input_trace, expected))
    masked = Spec(
        inputs=spec.inputs,
        outputs=spec.outputs,
        internals=spec.internals,
        scenarios=new_scenarios,
        invariants=spec.invariants,
    )
    assert_spec_not_degenerate(masked)
    return masked


def assert_spec_not_degenerate(spec: Spec):
    """마스킹 과다로 자명한 회로가 통과하는 퇴화 스펙을 차단.

    시나리오가 전이 스캔 위주로 압축돼 있으면 마스킹 후 남는 채점
    스캔이 전부 휴지 상태(0)가 돼 '항상 꺼진 회로'가 acc 1.0이 된다.
    (실측: 초기 seq2 시나리오가 정확히 이 함정 — dwell 스캔으로 해소)
    """
    x = spec.inputs[0]
    for logic, label in [
        (And([Contact(x), Contact(x, 'NC')]), '항상0'),
        (Or([Contact(x), Contact(x, 'NC')]), '항상1'),
    ]:
        prog = Program([Rung(Coil(o), logic) for o in spec.outputs])
        acc = accuracy(prog, spec)
        assert acc < 1.0, (
            f"don't-care 과다: {label} 회로가 통과 (acc={acc:.3f})"
        )


def make_tasks() -> list[Task]:
    tasks = []

    # ---- 1. self_hold: 기존 손제작 스펙 그대로 (가장 강한 형태) ----
    ref = Program(
        [
            Rung(
                Coil('Y0'),
                Or(
                    [
                        Contact('X0'),
                        And([Contact('Y0'), Contact('X1', 'NC')]),
                    ]
                ),
            )
        ]
    )
    tasks.append(
        Task(
            'self_hold',
            '자기유지 (기동/정지)',
            make_self_hold_spec(),
            ref,
            GenCfg(max_rungs=3, max_depth=3),
            dict(max_actions=14, max_stack=3, max_rungs=2),
        )
    )

    # ---- 2. interlock: Y0=X0·NOT Y1, Y1=X1·NOT Y0 (rung 순서=우선순위) ----
    ref = Program(
        [
            Rung(Coil('Y0'), And([Contact('X0'), Contact('Y1', 'NC')])),
            Rung(Coil('Y1'), And([Contact('X1'), Contact('Y0', 'NC')])),
        ]
    )
    # 선점 차단 구간에 dwell을 길게 둬야 한다 — 차단 의도가 채점 1점에만
    # 실리면 'Y0=X0' 같은 인터로크 없는 회로가 그 1점만 버리고 통과한다
    # (실측: 마스킹 도입 직후 6런 전부 0.967 고원이 정확히 이 함정)
    spec = derive_spec(
        ['X0', 'X1'],
        ['Y0', 'Y1'],
        ['M0'],
        ref,
        [
            # A: 교대로 누름
            [
                {'X0': 0, 'X1': 0},
                {'X0': 1},
                {},
                {'X0': 0},
                {},
                {'X1': 1},
                {},
                {'X1': 0},
                {},
                {},
            ],
            # B: 동시 누름 → 윗 rung(Y0) 우선
            [
                {'X0': 0, 'X1': 0},
                {'X0': 1, 'X1': 1},
                {},
                {},
                {'X0': 0, 'X1': 0},
                {},
            ],
            # C: X1 선점 중 X0 추가 → Y1 유지, Y0 차단 (둘 다 누른 채 4스캔)
            [
                {'X0': 0, 'X1': 0},
                {'X1': 1},
                {},
                {'X0': 1},
                {},
                {},
                {},
                {},
                {'X1': 0},
                {},
                {},
                {'X0': 0},
                {},
                {},
            ],
            # D: C의 대칭 — X0 선점 중 X1 추가 → Y0 유지, Y1 차단
            [
                {'X0': 0, 'X1': 0},
                {'X0': 1},
                {},
                {'X1': 1},
                {},
                {},
                {},
                {},
                {'X0': 0},
                {},
                {},
                {'X1': 0},
                {},
                {},
            ],
        ],
    )
    # intent: 어떤 시나리오에서도 동시 ON 금지
    for sc in spec.scenarios:
        for want in sc.expected:
            assert not (want['Y0'] and want['Y1']), 'interlock 동시 ON!'
    assert spec.scenarios[0].expected[1]['Y0'] == 1
    assert spec.scenarios[0].expected[5]['Y1'] == 1
    assert sum(w['Y1'] for w in spec.scenarios[1].expected) == 0, (
        '동시 누름에서 Y0 우선이어야'
    )
    c = spec.scenarios[2].expected
    assert c[7]['Y1'] == 1 and c[7]['Y0'] == 0, 'C: 선점 유지/차단 실패'
    assert c[9]['Y0'] == 1, 'C: X1 뗀 후 Y0 양보 실패'
    d = spec.scenarios[3].expected
    assert d[7]['Y0'] == 1 and d[7]['Y1'] == 0, 'D: 선점 유지/차단 실패'
    assert d[9]['Y1'] == 1, 'D: X0 뗀 후 Y1 양보 실패'
    # 동시 ON 금지는 전역 invariant로 강제 — 타임차트 점수만으로는
    # Y0=X0 꼼수가 채점 4점만 희생하고 0.933 고원을 형성 (9런 실측)
    spec.invariants = [lambda out: not (out['Y0'] and out['Y1'])]
    spec = mask_transition_scans(spec)  # 전이 스캔 don't-care
    tasks.append(
        Task(
            'interlock',
            '전후진 인터로크 (선입력 우선)',
            spec,
            ref,
            GenCfg(max_rungs=3, max_depth=3),
            dict(max_actions=12, max_stack=3, max_rungs=2),
        )
    )

    # ---- 3. one_shot: Y0 = PLS(X0) ----
    ref = Program([Rung(Coil('Y0'), Pulse('P0', Contact('X0')))])
    spec = derive_spec(
        ['X0'],
        ['Y0'],
        ['M0'],
        ref,
        [
            # A: 길게/짧게 섞어 3번 누름 → 펄스 정확히 3개
            [
                {'X0': 0},
                {'X0': 1},
                {},
                {},
                {'X0': 0},
                {},
                {'X0': 1},
                {'X0': 0},
                {'X0': 1},
                {},
                {'X0': 0},
            ],
            # B: 안 누름
            [{'X0': 0}, {}, {}, {}],
            # C: 시작부터 누른 채
            [{'X0': 1}, {}, {}, {}],
        ],
    )
    a = [w['Y0'] for w in spec.scenarios[0].expected]
    assert sum(a) == 3, f'엣지 3개여야: {a}'
    assert all(not (a[i] and a[i + 1]) for i in range(len(a) - 1)), (
        '펄스는 1스캔만'
    )
    assert [w['Y0'] for w in spec.scenarios[2].expected] == [1, 0, 0, 0]
    tasks.append(
        Task(
            'one_shot',
            '원샷 (1스캔 펄스)',
            spec,
            ref,
            GenCfg(max_rungs=2, max_depth=2, allow_pulse=True),
            dict(max_actions=8, max_stack=2, max_rungs=2, allow_pulse=True),
        )
    )

    # ---- 4. delayed_off: M0=TON(3,NOT X0), Y0=(X0+Y0)·NOT M0 ----
    ref = Program(
        [
            Rung(Coil('M0'), Timer('T0', 3, Contact('X0', 'NC'))),
            Rung(
                Coil('Y0'),
                And(
                    [
                        Or([Contact('X0'), Contact('Y0')]),
                        Contact('M0', 'NC'),
                    ]
                ),
            ),
        ]
    )
    spec = derive_spec(
        ['X0'],
        ['Y0'],
        ['M0'],
        ref,
        [
            # A: 3스캔 운전 후 정지 → 2스캔 더 유지, 3스캔째 꺼짐
            [{'X0': 0}, {}, {'X0': 1}, {}, {}, {'X0': 0}, {}, {}, {}, {}],
            # B: 안 누름
            [{'X0': 0}, {}, {}, {}, {}],
            # C: 1스캔 블립
            [{'X0': 0}, {'X0': 1}, {'X0': 0}, {}, {}, {}, {}],
            # D: 지연 중 재기동 → 타이머 리셋
            [
                {'X0': 0},
                {'X0': 1},
                {'X0': 0},
                {},
                {'X0': 1},
                {'X0': 0},
                {},
                {},
                {},
                {},
            ],
        ],
    )
    a = [w['Y0'] for w in spec.scenarios[0].expected]
    assert a == [0, 0, 1, 1, 1, 1, 1, 0, 0, 0], f'지연 정지 trace 이상: {a}'
    assert all(w['Y0'] == 0 for w in spec.scenarios[1].expected)
    d = [w['Y0'] for w in spec.scenarios[3].expected]
    assert d[6] == 1 and d[7] == 0, f'재기동 리셋 이상: {d}'
    spec = mask_transition_scans(spec)  # 전이 스캔 don't-care
    tasks.append(
        Task(
            'delayed_off',
            '지연 정지 (TON+자기유지)',
            spec,
            ref,
            GenCfg(max_rungs=3, max_depth=3, timer_presets=(3,)),
            dict(
                max_actions=14,
                max_stack=3,
                max_rungs=2,
                timer_presets=(3,),
                max_timers=2,
            ),
        )
    )

    # ---- 5. flicker: Y0=TON(2, X0·NOT M0), M0=TON(2, Y0) ----
    ref = Program(
        [
            Rung(
                Coil('Y0'),
                Timer(
                    'T0',
                    2,
                    And(
                        [
                            Contact('X0'),
                            Contact('M0', 'NC'),
                        ]
                    ),
                ),
            ),
            Rung(Coil('M0'), Timer('T1', 2, Contact('Y0'))),
        ]
    )
    spec = derive_spec(
        ['X0'],
        ['Y0'],
        ['M0'],
        ref,
        [
            # A: 12스캔 운전 → ON 2 / OFF 2 발진
            [{'X0': 0}, {'X0': 1}] + [{}] * 11 + [{'X0': 0}, {}, {}],
            # B: 안 누름
            [{'X0': 0}, {}, {}, {}],
        ],
    )
    a = [w['Y0'] for w in spec.scenarios[0].expected]
    assert a[2:10] == [1, 1, 0, 0, 1, 1, 0, 0], f'발진 패턴 이상: {a}'
    assert a[-1] == 0 and a[-2] == 0, '정지 후 꺼져야'
    tasks.append(
        Task(
            'flicker',
            '플리커 (TON 2개 교차결합)',
            spec,
            ref,
            GenCfg(max_rungs=3, max_depth=3, timer_presets=(2,)),
            dict(
                max_actions=12,
                max_stack=3,
                max_rungs=2,
                timer_presets=(2,),
                max_timers=2,
            ),
        )
    )

    # ---- 6. seq2: Y1=((X1·Y0)+Y1)·NOT X2, Y0=(X0+Y0)·NOT X1 ----
    #      (단계2 rung이 먼저 와야 같은 스캔에 X1이 Y0를 죽이기 전 래치됨)
    ref = Program(
        [
            Rung(
                Coil('Y1'),
                And(
                    [
                        Or(
                            [And([Contact('X1'), Contact('Y0')]), Contact('Y1')]
                        ),
                        Contact('X2', 'NC'),
                    ]
                ),
            ),
            Rung(
                Coil('Y0'),
                And(
                    [
                        Or([Contact('X0'), Contact('Y0')]),
                        Contact('X1', 'NC'),
                    ]
                ),
            ),
        ]
    )
    # 전이 스캔 사이에 dwell(유지) 스캔을 둬야 마스킹 후에도 래치 상태가
    # 채점된다 (전이만 빽빽하면 항상0 회로가 통과 — assert_spec_not_degenerate)
    cycle = [
        {'X0': 0, 'X1': 0, 'X2': 0},  # 0: 대기
        {'X0': 1},
        {},  # 1-2: 기동
        {'X0': 0},
        {},
        {},  # 3-5: 버튼 뗌 (Y0 유지)
        {'X1': 1},
        {},  # 6-7: 센서1 (Y0→Y1 전이)
        {'X1': 0},
        {},
        {},  # 8-10: (Y1 유지)
        {'X2': 1},
        {},  # 11-12: 센서2 (완료)
        {'X2': 0},
        {},
        {},  # 13-15: 대기 복귀
    ]
    spec = derive_spec(
        ['X0', 'X1', 'X2'],
        ['Y0', 'Y1'],
        ['M0'],
        ref,
        [
            cycle,  # A: 1사이클
            [{'X0': 0, 'X1': 0, 'X2': 0}, {}, {}, {}],  # B: 대기
            # C: 기동 없이 센서만 → 아무 일도 없어야
            [
                {'X0': 0, 'X1': 0, 'X2': 0},
                {'X1': 1},
                {},
                {'X1': 0},
                {},
                {'X2': 1},
                {},
                {'X2': 0},
                {},
                {},
            ],
            cycle + cycle[1:],  # D: 2사이클 (재기동)
        ],
    )
    a = spec.scenarios[0].expected
    assert [w['Y0'] for w in a] == [0] + [1] * 5 + [0] * 10
    assert [w['Y1'] for w in a] == [0] * 6 + [1] * 5 + [0] * 5
    assert all(
        w['Y0'] == 0 and w['Y1'] == 0 for w in spec.scenarios[2].expected
    ), '기동 없이 동작!'
    d = spec.scenarios[3].expected
    assert d[16]['Y0'] == 1 and d[16 + 6]['Y1'] == 1, '재기동 실패'
    spec = mask_transition_scans(spec)  # 전이 스캔 don't-care
    # SET/RST 허용 — 래치 체인의 자연형은 SET/RST 4 rung
    # (SET Y0←X0 / RST Y0←X1 / SET Y1←X1·Y0 / RST Y1←X2)이라 rung 한도 4
    tasks.append(
        Task(
            'seq2',
            '2단 시퀀스 (래치 체인)',
            spec,
            ref,
            GenCfg(max_rungs=4, max_depth=4, setrst_p=0.3),
            dict(max_actions=18, max_stack=3, max_rungs=4, allow_setrst=True),
        )
    )

    # ---- 7. actuator: 전후진 실린더 (운영표준 Actuator 모티프) ----
    #      지령 latch + 도달 센서 해제 + 전후진 인터로크 (self_hold × interlock)
    #      X0/X1=전·후진 지령, X2/X3=전·후진단 센서, Y0/Y1=전·후진 솔
    ref = Program(
        [
            Rung(
                Coil('Y0'),
                And(
                    [
                        Or([Contact('X0'), Contact('Y0')]),
                        Contact('X2', 'NC'),
                        Contact('Y1', 'NC'),
                    ]
                ),
            ),
            Rung(
                Coil('Y1'),
                And(
                    [
                        Or([Contact('X1'), Contact('Y1')]),
                        Contact('X3', 'NC'),
                        Contact('Y0', 'NC'),
                    ]
                ),
            ),
        ]
    )
    spec = derive_spec(
        ['X0', 'X1', 'X2', 'X3'],
        ['Y0', 'Y1'],
        ['M0'],
        ref,
        [
            # A: 전진 → 도달 → 후진 → 도달
            [
                {'X0': 0, 'X1': 0, 'X2': 0, 'X3': 0},
                {'X0': 1},
                {},  # 1-2 전진 지령
                {'X0': 0},
                {},
                {},  # 3-5 유지
                {'X2': 1},
                {},  # 6-7 전진단 → off
                {'X2': 0},
                {},  # 8-9
                {'X1': 1},
                {},  # 10-11 후진 지령
                {'X1': 0},
                {},
                {},  # 12-14 유지
                {'X3': 1},
                {},  # 15-16 후진단 → off
                {'X3': 0},
                {},  # 17-18
            ],
            # B: 대기
            [{'X0': 0, 'X1': 0, 'X2': 0, 'X3': 0}, {}, {}, {}],
            # C: 동시 지령 → 윗 rung(Y0) 우선
            [
                {'X0': 0, 'X1': 0, 'X2': 0, 'X3': 0},
                {'X0': 1, 'X1': 1},
                {},
                {},
                {'X0': 0, 'X1': 0},
                {},
                {'X2': 1},
                {'X2': 0},
                {},
            ],
            # D: 전진 동작 중 후진 지령 → 인터로크로 무시
            [
                {'X0': 0, 'X1': 0, 'X2': 0, 'X3': 0},
                {'X0': 1},
                {'X0': 0},
                {},
                {'X1': 1},
                {},
                {},
                {'X1': 0},
                {},
                {'X2': 1},
                {'X2': 0},
                {},
                {},
            ],
        ],
    )
    a = spec.scenarios[0].expected
    assert [w['Y0'] for w in a] == [0] + [1] * 5 + [0] * 13
    assert [w['Y1'] for w in a] == [0] * 10 + [1] * 5 + [0] * 4
    c = spec.scenarios[2].expected
    assert c[2]['Y0'] == 1 and sum(w['Y1'] for w in c) == 0, '동시 지령 Y0 우선'
    d = spec.scenarios[3].expected
    assert d[5]['Y0'] == 1 and sum(w['Y1'] for w in d) == 0, '동작 중 후진 차단'
    for sc in spec.scenarios:
        for want in sc.expected:
            assert not (want['Y0'] and want['Y1']), 'actuator 동시 ON!'
    spec.invariants = [lambda out: not (out['Y0'] and out['Y1'])]
    spec = mask_transition_scans(spec)
    tasks.append(
        Task(
            'actuator',
            '전후진 실린더 (latch+센서 해제+인터로크)',
            spec,
            ref,
            GenCfg(max_rungs=4, max_depth=3, setrst_p=0.3),
            dict(max_actions=18, max_stack=3, max_rungs=4, allow_setrst=True),
        )
    )

    # ---- 8. seq3: 3단 래치 체인 (seq2 +1단 — GP 한계 탐침) ----
    ref = Program(
        [
            Rung(
                Coil('Y2'),
                And(
                    [
                        Or(
                            [And([Contact('X2'), Contact('Y1')]), Contact('Y2')]
                        ),
                        Contact('X3', 'NC'),
                    ]
                ),
            ),
            Rung(
                Coil('Y1'),
                And(
                    [
                        Or(
                            [And([Contact('X1'), Contact('Y0')]), Contact('Y1')]
                        ),
                        Contact('X2', 'NC'),
                    ]
                ),
            ),
            Rung(
                Coil('Y0'),
                And(
                    [
                        Or([Contact('X0'), Contact('Y0')]),
                        Contact('X1', 'NC'),
                    ]
                ),
            ),
        ]
    )
    cycle = [
        {'X0': 0, 'X1': 0, 'X2': 0, 'X3': 0},  # 0: 대기
        {'X0': 1},
        {},  # 1-2: 기동
        {'X0': 0},
        {},
        {},  # 3-5: Y0 유지
        {'X1': 1},
        {},  # 6-7: 센서1 (Y0→Y1)
        {'X1': 0},
        {},
        {},  # 8-10: Y1 유지
        {'X2': 1},
        {},  # 11-12: 센서2 (Y1→Y2)
        {'X2': 0},
        {},
        {},  # 13-15: Y2 유지
        {'X3': 1},
        {},  # 16-17: 센서3 (완료)
        {'X3': 0},
        {},
        {},  # 18-20: 대기 복귀
    ]
    spec = derive_spec(
        ['X0', 'X1', 'X2', 'X3'],
        ['Y0', 'Y1', 'Y2'],
        ['M0'],
        ref,
        [
            cycle,  # A: 1사이클
            [{'X0': 0, 'X1': 0, 'X2': 0, 'X3': 0}, {}, {}, {}],  # B: 대기
            # C: 기동 없이 센서만 → 아무 일도 없어야
            [
                {'X0': 0, 'X1': 0, 'X2': 0, 'X3': 0},
                {'X1': 1},
                {},
                {'X1': 0},
                {},
                {'X2': 1},
                {},
                {'X2': 0},
                {},
                {'X3': 1},
                {},
                {'X3': 0},
                {},
                {},
            ],
            cycle + cycle[1:],  # D: 2사이클 (재기동)
        ],
    )
    a = spec.scenarios[0].expected
    assert [w['Y0'] for w in a] == [0] + [1] * 5 + [0] * 15
    assert [w['Y1'] for w in a] == [0] * 6 + [1] * 5 + [0] * 10
    assert [w['Y2'] for w in a] == [0] * 11 + [1] * 5 + [0] * 5
    assert all(
        w['Y0'] == 0 and w['Y1'] == 0 and w['Y2'] == 0
        for w in spec.scenarios[2].expected
    ), '기동 없이 동작!'
    d = spec.scenarios[3].expected
    assert d[21]['Y0'] == 1 and d[21 + 10]['Y2'] == 1, '재기동 실패'
    # 단계는 동시에 하나만 (PackML step 의미)
    spec.invariants = [lambda out: out['Y0'] + out['Y1'] + out['Y2'] <= 1]
    spec = mask_transition_scans(spec)
    # SET/RST 자연형은 6 rung (SET/RST × 3단계)
    tasks.append(
        Task(
            'seq3',
            '3단 시퀀스 (래치 체인 ×3)',
            spec,
            ref,
            GenCfg(max_rungs=6, max_depth=4, setrst_p=0.3),
            dict(max_actions=26, max_stack=3, max_rungs=6, allow_setrst=True),
        )
    )

    return tasks


# ---------- 무작위 생성기 (TON/PLS 지원 확장판) ----------


def random_logic_ext(devices, depth, rng, cfg: GenCfg, name_counter):
    if depth <= 0 or rng.random() < 0.4:
        node = Contact(rng.choice(devices), rng.choice(['NO', 'NC']))
    else:
        op = rng.choice([And, Or])
        node = op(
            [
                random_logic_ext(devices, depth - 1, rng, cfg, name_counter)
                for _ in range(rng.randint(2, 3))
            ]
        )
    if cfg.timer_presets and rng.random() < cfg.wrap_p:
        node = Timer(
            f'T{next(name_counter)}', rng.choice(cfg.timer_presets), node
        )
    if cfg.allow_pulse and rng.random() < cfg.wrap_p:
        node = Pulse(f'P{next(name_counter)}', node)
    return node


def random_program_ext(spec: Spec, rng, cfg: GenCfg) -> Program:
    contact_pool = spec.inputs + spec.internals + spec.outputs
    coil_pool = spec.outputs + spec.internals
    name_counter = count()
    rungs = []
    used: dict[str, set] = {}
    for _ in range(rng.randint(1, cfg.max_rungs)):
        op = 'OUT'
        if cfg.setrst_p and rng.random() < cfg.setrst_p:
            op = rng.choice(['SET', 'RST'])
        cands = [d for d in coil_pool if coil_allowed(used, d, op)]
        if not cands:
            break  # 이중 코일 금지로 더 둘 코일이 없음
        dev = rng.choice(cands)
        used.setdefault(dev, set()).add(op)
        rungs.append(
            Rung(
                Coil(dev, op),
                random_logic_ext(
                    contact_pool, cfg.max_depth, rng, cfg, name_counter
                ),
            )
        )
    if not any(r.coil.device in spec.outputs for r in rungs):
        # 출력이 전혀 안 쓰였으면 출력은 전부 미작성 → OUT 항상 허용
        rungs.append(
            Rung(
                Coil(rng.choice(spec.outputs)),
                random_logic_ext(
                    contact_pool, cfg.max_depth, rng, cfg, name_counter
                ),
            )
        )
    return Program(rungs)


# ---------- 러너 ----------


def run_random(task: Task, budget: int, seed: int):
    rng = random.Random(seed)
    best_acc, best_prog, found = 0.0, None, None
    for i in range(1, budget + 1):
        prog = random_program_ext(task.spec, rng, task.gen_cfg)
        acc, viol = evaluate(prog, task.spec)
        if acc > best_acc:
            best_acc, best_prog = acc, prog
        if acc >= 1.0 and viol == 0:
            found = i
            break
    return found, best_acc, best_prog


def run_mcts(task: Task, budget: int, seed: int):
    ev = mcts_search(
        task.spec,
        budget,
        seed,
        state_factory=lambda: BuildState(task.spec, **task.mcts_kwargs),
    )
    return ev.found_at, ev.best_acc, ev.best_prog


def run_mcts_w(task: Task, budget: int, seed: int):
    """롤아웃만 종류 가중 샘플로 교체한 MCTS (PUSH 지배 해소 가설 검증)"""
    ev = mcts_search(
        task.spec,
        budget,
        seed,
        state_factory=lambda: BuildState(task.spec, **task.mcts_kwargs),
        rollout_policy=weighted_rollout,
    )
    return ev.found_at, ev.best_acc, ev.best_prog


def run_gp(task: Task, budget: int, seed: int):
    """rung 단위 교차 GP — '두 rung 동시 정합' 과제(interlock) 정조준"""
    cfg = task.gen_cfg
    contact_pool = task.spec.inputs + task.spec.internals + task.spec.outputs

    def new_logic(rng):
        # 변이용 국소 서브트리 (이름은 renumber_stateful이 재부여)
        return random_logic_ext(contact_pool, 2, rng, cfg, count(100))

    return gp_search(
        task.spec,
        budget,
        seed,
        new_program=lambda rng: random_program_ext(task.spec, rng, cfg),
        new_logic=new_logic,
        coil_pool=task.spec.outputs + task.spec.internals,
        max_rungs=cfg.max_rungs,
        allow_setrst=cfg.setrst_p > 0,
    )


METHODS = [
    ('random', run_random),
    ('mcts', run_mcts),
    ('mcts_w', run_mcts_w),
    ('gp', run_gp),
]


# ---------- 메인 ----------

if __name__ == '__main__':
    BUDGET = 200_000
    SEEDS = [0, 1, 2]
    tasks = make_tasks()
    if len(sys.argv) > 1:  # 과제/방법 이름 혼합으로 부분 실행
        names = set(sys.argv[1:])
        method_names = {m for m, _ in METHODS}
        sel_methods = names & method_names
        sel_tasks = names - method_names
        unknown = sel_tasks - {t.name for t in tasks}
        assert not unknown, f'없는 과제: {unknown}'
        if sel_tasks:
            tasks = [t for t in tasks if t.name in sel_tasks]
        if sel_methods:
            METHODS = [(m, r) for m, r in METHODS if m in sel_methods]

    print('난이도 사다리 벤치마크 — random / mcts / mcts_w / gp')
    print(f'예산: 평가 {BUDGET:,}회/시드  |  시드: {SEEDS}')
    print('=' * 70)

    # 레퍼런스 검증
    for t in tasks:
        acc, viol = evaluate(t.reference, t.spec)
        assert acc >= 1.0 and viol == 0, (
            f'{t.name} 레퍼런스가 스펙 미달! acc={acc} viol={viol}'
        )
        print(
            f'[ref OK] {t.name:<12} 크기 {program_size(t.reference):>2}  ({t.desc})'
        )
    print('=' * 70)

    results = []  # (task, method, seed, found, best_acc, prog)
    for t in tasks:
        for method, runner in METHODS:
            for seed in SEEDS:
                found, best_acc, prog = runner(t, BUDGET, seed)
                results.append((t.name, method, seed, found, best_acc, prog))
                stat = f'{found:,}회' if found else f'실패 (acc {best_acc:.3f})'
                print(
                    f'  {t.name:<12} {method:<7} seed{seed}: {stat}', flush=True
                )

    # 요약 표
    width = 13 + sum(3 + 26 for _ in METHODS)
    print()
    print('=' * width)
    print(
        f'{"과제":<13}'
        + ''.join(f' | {m + " (3시드)":>26}' for m, _ in METHODS)
    )
    print('-' * width)
    by = {}
    for name, method, seed, found, best_acc, _ in results:
        by.setdefault((name, method), []).append((found, best_acc))
    for t in tasks:

        def fmt(method):
            cells = []
            for found, acc in by[(t.name, method)]:
                cells.append(
                    f'{found:,}' if found else f'x(.{int(acc * 1000):03d})'
                )
            return ' / '.join(f'{c:>8}' for c in cells)

        print(f'{t.name:<13}' + ''.join(f' | {fmt(m)}' for m, _ in METHODS))
    print('-' * width)
    print('x(.NNN) = 실패, 괄호는 최고 accuracy')

    # 발견된 해 출력 (과제×방법별 첫 성공 시드)
    print()
    for t in tasks:
        for m, _ in METHODS:
            for name, method, seed, found, _, prog in results:
                if name == t.name and method == m and found and prog:
                    print(f'--- {t.name} ({m} seed{seed}, {found:,}회) ---')
                    print(program_str(prog))
                    break
