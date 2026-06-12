"""
ladder_search.py — 보상 함수 + 무작위 탐색 베이스라인

[ 구조 ]

  Spec   = 입력 trace + 기대 출력 trace 묶음 (여러 시나리오 가능)
  score()= 후보 프로그램을 simulate해서 기대 trace와 비교 -> 0.0 ~ 1.0
           (스캔×출력 단위 일치율. 부분 점수 -> sparse 보상 완화)
           + 크기 페널티 (같은 점수면 작은 회로 선호)

  random_program() = IR 문법 위에서 무작위 래더 생성
                     -> RL 없이 '눈 감고 던지기'가 얼마나 먹히는지 측정

  이 실험의 목적:
    - 보상 함수 파이프라인이 end-to-end로 도는 것 확인
    - 탐색 공간의 험준함 체감 -> MCTS/정책망의 필요성 정량화
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List

from ladder_sim import (
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

# ---------- 스펙 정의 ----------


@dataclass
class Scenario:
    input_trace: List[Dict[str, int]]
    expected: List[Dict[str, int]]  # 스캔별 기대 출력


@dataclass
class Spec:
    inputs: List[str]  # 사용 가능한 입력 디바이스
    outputs: List[str]  # 채워야 하는 출력 디바이스
    internals: List[str]  # 사용 가능한 내부 릴레이
    scenarios: List[Scenario]
    # 전역 속성 (스캔별 출력 dict -> bool). 타임차트 점수에 과소반영되는
    # 의도(예: interlock 동시 ON 금지)를 마스킹 스캔 포함 전 스캔에 강제.
    invariants: List = field(default_factory=list)


# ---------- 보상 함수 ----------


def program_size(prog: Program) -> int:
    """노드 수 세기 (크기 페널티용)"""

    def logic_size(node):
        if isinstance(node, Contact):
            return 1
        if isinstance(node, (Timer, Pulse)):
            return 1 + logic_size(node.input)
        return 1 + sum(logic_size(a) for a in node.args)

    return sum(1 + logic_size(r.logic) for r in prog.rungs)


def evaluate(prog: Program, spec: Spec):
    """(순수 일치율, invariant 위반 스캔 수) — 시나리오당 simulate 1회.

    일치율은 스캔 x 출력 단위, don't-care(None) 스캔은 채점 제외.
    invariant는 don't-care 여부와 무관하게 모든 스캔에서 검사한다
    (마스킹은 타이밍 관용이지 전역 속성의 면제가 아님).

    '완벽해 발견' 판정은 acc==1.0 AND 위반 0 —
    score()는 페널티가 섞여 있어 큰 회로는 정답이어도 0.99에 못 미친다.
    """
    total, correct, viol = 0, 0, 0
    for sc in spec.scenarios:
        assert len(sc.input_trace) == len(sc.expected), (
            f"trace 길이 불일치: input {len(sc.input_trace)} vs expected {len(sc.expected)}"
        )
        trace = simulate(prog, sc.input_trace, spec.outputs)
        for got, want in zip(trace, sc.expected):
            for dev, want_val in want.items():
                if want_val is None:  # don't-care 스캔은 채점 제외
                    continue
                total += 1
                if got.get(dev, 0) == want_val:
                    correct += 1
            if any(not inv(got) for inv in spec.invariants):
                viol += 1
    return (correct / total if total else 0.0), viol


def accuracy(prog: Program, spec: Spec) -> float:
    return evaluate(prog, spec)[0]


def score(
    prog: Program, spec: Spec, size_weight: float = 0.001, inv_weight: float = 0.05
) -> float:
    """
    보상 = 일치율 - invariant 위반 페널티 - 크기 페널티
    (위반 페널티가 커야 '점수 좋은 꼼수 고원' 자체가 무너진다)
    """
    acc, viol = evaluate(prog, spec)
    return acc - inv_weight * viol - size_weight * program_size(prog)


# ---------- 이중 코일 금지 (GX Works3 이중 코일 경고와 동일 기준) ----------


def coil_usage(rungs) -> Dict[str, set]:
    """디바이스별 사용된 코일 op 집합"""
    used: Dict[str, set] = {}
    for r in rungs:
        used.setdefault(r.coil.device, set()).add(r.coil.op)
    return used


def coil_allowed(used: Dict[str, set], dev: str, op: str) -> bool:
    """OUT은 디바이스당 유일 작성자. SET/RST는 OUT과 혼합 금지
    (SET+RST 래치 쌍, 다중 SET은 PLC 정상 패턴이라 허용)"""
    ops = used.get(dev, set())
    if op == "OUT":
        return not ops
    return "OUT" not in ops


def find_coil_conflicts(prog: Program) -> List[str]:
    """이중 코일 위반 디바이스 목록 (검증용)"""
    used: Dict[str, set] = {}
    bad = []
    for r in prog.rungs:
        if not coil_allowed(used, r.coil.device, r.coil.op):
            bad.append(r.coil.device)
        used.setdefault(r.coil.device, set()).add(r.coil.op)
    return bad


# ---------- 무작위 프로그램 생성기 ----------


def random_logic(devices: List[str], depth: int, rng: random.Random):
    """깊이 제한 재귀로 무작위 논리 트리 생성"""
    if depth <= 0 or rng.random() < 0.4:
        return Contact(rng.choice(devices), rng.choice(["NO", "NC"]))
    op = rng.choice([And, Or])
    n_args = rng.randint(2, 3)
    return op([random_logic(devices, depth - 1, rng) for _ in range(n_args)])


def random_program(
    spec: Spec, rng: random.Random, max_rungs: int = 3, max_depth: int = 3
) -> Program:
    """
    무작위 래더 생성.
    접점으로 쓸 수 있는 것: 입력 + 내부릴레이 + 출력(자기유지용 피드백)
    코일로 쓸 수 있는 것: 출력 + 내부릴레이
    """
    contact_pool = spec.inputs + spec.internals + spec.outputs
    coil_pool = spec.outputs + spec.internals

    rungs = []
    n_rungs = rng.randint(1, max_rungs)
    for _ in range(n_rungs):
        coil_dev = rng.choice(coil_pool)
        logic = random_logic(contact_pool, max_depth, rng)
        rungs.append(Rung(Coil(coil_dev), logic))

    # 출력 코일이 하나도 없으면 강제로 하나 추가 (완전 무의미한 후보 방지)
    if not any(r.coil.device in spec.outputs for r in rungs):
        rungs.append(
            Rung(
                Coil(rng.choice(spec.outputs)),
                random_logic(contact_pool, max_depth, rng),
            )
        )
    return Program(rungs)


# ---------- IR -> 텍스트 (사람 확인용) ----------


def logic_str(node) -> str:
    if isinstance(node, Contact):
        return f"{node.device}{'' if node.mode == 'NO' else '/'}"  # /=B접점
    if isinstance(node, Timer):
        return f"TON({node.name},K{node.preset},{logic_str(node.input)})"
    if isinstance(node, Pulse):
        return f"PLS({logic_str(node.input)})"
    sym = " * " if isinstance(node, And) else " + "  # *=직렬 +=병렬
    return "(" + sym.join(logic_str(a) for a in node.args) + ")"


def program_str(prog: Program) -> str:
    lines = []
    for r in prog.rungs:
        op = "" if r.coil.op == "OUT" else f"{r.coil.op} "
        lines.append(f"  {logic_str(r.logic)}  ->  {op}{r.coil.device}")
    return "\n".join(lines)


# ---------- 자기유지 스펙 ----------


def make_self_hold_spec() -> Spec:
    """
    목표: X0(기동, 모멘터리) / X1(정지, 모멘터리) -> Y0 자기유지
    시나리오 3개로 스펙을 조여서 '꼼수 회로'가 못 통과하게 함
    """
    # 시나리오 A: 기동 -> 유지 -> 정지 -> 꺼짐 유지
    sA = Scenario(
        input_trace=[
            {"X0": 0, "X1": 0},
            {"X0": 1},
            {"X0": 0},
            {},
            {},
            {"X1": 1},
            {"X1": 0},
            {},
        ],
        expected=[
            {"Y0": 0},
            {"Y0": 1},
            {"Y0": 1},
            {"Y0": 1},
            {"Y0": 1},
            {"Y0": 0},
            {"Y0": 0},
            {"Y0": 0},
        ],
    )
    # 시나리오 B: 아무것도 안 누름 -> 계속 꺼져 있어야 함 (상시 ON 회로 차단)
    sB = Scenario(
        input_trace=[{"X0": 0, "X1": 0}, {}, {}, {}],
        expected=[{"Y0": 0}] * 4,
    )
    # 시나리오 C: 재기동 가능해야 함 (한 번 쓰고 죽는 회로 차단)
    sC = Scenario(
        input_trace=[
            {"X0": 0, "X1": 0},
            {"X0": 1},
            {"X0": 0},
            {"X1": 1},
            {"X1": 0},
            {"X0": 1},
            {"X0": 0},
            {},
        ],
        expected=[
            {"Y0": 0},
            {"Y0": 1},
            {"Y0": 1},
            {"Y0": 0},
            {"Y0": 0},
            {"Y0": 1},
            {"Y0": 1},
            {"Y0": 1},
        ],
    )
    return Spec(
        inputs=["X0", "X1"], outputs=["Y0"], internals=["M0"], scenarios=[sA, sB, sC]
    )


# ---------- 무작위 탐색 실험 ----------


def random_search(spec: Spec, budget: int, seed: int = 0):
    rng = random.Random(seed)
    best_prog, best_score = None, -1.0
    found_at = None
    history = []  # (시도번호, best_score) 기록

    for i in range(1, budget + 1):
        prog = random_program(spec, rng)
        s = score(prog, spec)
        if s > best_score:
            best_score, best_prog = s, prog
            history.append((i, s))
            if s >= 0.99 and found_at is None:  # 사실상 완벽 (페널티 감안)
                found_at = i
    return best_prog, best_score, found_at, history


if __name__ == "__main__":
    spec = make_self_hold_spec()
    BUDGET = 200_000

    print(f"목표: 기동/정지 자기유지 회로  (예산: {BUDGET:,}회 무작위 시도)")
    print("=" * 60)

    for seed in [0, 1, 2]:
        prog, s, found_at, hist = random_search(spec, BUDGET, seed)
        status = f"발견! ({found_at:,}번째)" if found_at else "실패"
        print(f"\n[seed {seed}]  최고 점수: {s:.4f}   완벽해답: {status}")
        print(f"  점수 갱신 이력: {[(i, round(v, 3)) for i, v in hist[-5:]]}")
        print("  최고 후보 회로:")
        print(program_str(prog))

    print()
    print("=" * 60)
    print("표기: X0  = A접점,  X0/ = B접점,  * = 직렬(AND),  + = 병렬(OR)")
