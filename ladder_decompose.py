"""
ladder_decompose.py — Program → BuildState 액션열 분해 (정책망 라벨 공장)

[ 왜 ]
  신경망 prior 의 정책망 학습 데이터는 (스펙, 상태, 정답 다음수) 쌍이다.
  '정답 다음수' 라벨은 사람이 찍는 게 아니라 **이미 가진 정답 회로
  (reference / GP 해) 를 BuildState 액션열로 펴면 매 단계가 그대로 라벨**이
  된다. 이 파일이 그 분해기 — 라벨 공장의 첫 기계.

      정답회로  ((X0) + (Y0 * X1/)) -> Y0
        →  PUSH X0 / PUSH Y0 / PUSH X1/ / AND / OR / EMIT Y0 / DONE
        →  (∅→PUSH X0), ([X0]→PUSH Y0), ... 7개 (상태→정답수) 라벨

[ 설계: BuildState 게임 규칙에 맞춘 후위순회 ]
  - AND/OR 은 게임에서 **이항**이다 (스택 top 2개 묶음). IR 의 n-항
    And([a,b,c]) 는 ((a AND b) AND c) 로 **좌측 폴딩**해서 편다.
    구조는 중첩되지만 의미(evaluate)는 동일.
  - Timer/Pulse 는 입력 서브트리를 먼저 스택에 쌓은 뒤 ("TON",p)/("PLS",)
    로 감싼다 (스택 top 1개를 소비).
  - 한 rung 의 로직은 스택 +1 (트리 전체가 1개로 환원) → EMIT 이 그 1개를
    꺼내 rung 확정. rung 사이엔 스택이 비므로 EMIT 합법성이 자연히 성립.
  - 게임이 Timer/Pulse 이름을 T{n}/P{n} 로 자동 재부여하므로 원본 이름은
    버려진다 (이름 유일성 전제 하 evaluate 불변 — renumber_stateful 과 동일).

[ 안전 규칙 ]
  - 분해는 BuildState 의 탐색용 한도(max_stack/max_rungs/...)를 따르지
    않는다. 한도는 '탐색'을 가두는 장치지 '라벨링'을 가두는 게 아니다.
    재생(replay)은 apply() 를 직접 호출해 legal_actions 게이트를 우회한다.
  - 라벨 유효성은 구조 동일이 아니라 evaluate() 동일로 검증한다
    (n-항 폴딩·타이머 재명명 때문에 구조는 달라질 수 있음).
  - 한 회로의 분해 순서는 유일하지 않다 (And/Or 인자 순서, rung 순서).
    여기선 인자·rung 을 원본 좌→우 순서로 펴는 **정규(canonical) 분해**
    하나만 낸다. 다중 순서 증강은 학습 단계에서 별도로.

[ 실행 ]
  python ladder_decompose.py            # 8과제 reference 분해 + 라운드트립 검증
  python ladder_decompose.py interlock  # 특정 과제 액션열 출력
"""

import sys
import textwrap
from typing import List, Tuple

from ladder_mcts import BuildState
from ladder_search import evaluate, program_size, program_str
from render_ladder import ladder_str
from ladder_sim import (
    And,
    Contact,
    Or,
    Program,
    Pulse,
    Rung,
    Timer,
)

Action = Tuple


# ---------- 분해: Program → 액션열 ----------


def emit_logic(node, out: List[Action]):
    """로직 트리를 후위순회로 펴서 out 에 액션 추가 (스택 net +1)"""
    if isinstance(node, Contact):
        out.append(("PUSH", node.device, node.mode))
    elif isinstance(node, (And, Or)):
        op = "AND" if isinstance(node, And) else "OR"
        args = node.args
        if not args:
            raise ValueError(f"빈 {op} 노드는 분해 불가")
        emit_logic(args[0], out)  # 첫 인자 (단항이면 op 없이 그대로)
        for a in args[1:]:
            emit_logic(a, out)
            out.append((op,))  # 좌측 폴딩: 직전 결과 + 새 인자
    elif isinstance(node, Timer):
        emit_logic(node.input, out)
        out.append(("TON", node.preset))
    elif isinstance(node, Pulse):
        emit_logic(node.input, out)
        out.append(("PLS",))
    else:
        raise TypeError(f"분해 불가 노드: {node!r}")


def program_to_actions(prog: Program) -> List[Action]:
    """정답 회로 → 정규 액션열 (마지막은 항상 DONE)"""
    acts: List[Action] = []
    for rung in prog.rungs:
        emit_logic(rung.logic, acts)
        acts.append(("EMIT", rung.coil.device, rung.coil.op))
    acts.append(("DONE",))
    return acts


# ---------- 재생: 액션열 → (상태, 액션) 라벨 / 복원 ----------


def _fresh_state(spec) -> BuildState:
    """한도 없는 재생용 BuildState (라벨링은 탐색 한도에 안 묶임)"""
    big = 10**9
    return BuildState(
        spec,
        max_actions=big,
        max_stack=big,
        max_rungs=big,
        allow_pulse=True,
        allow_setrst=True,
        max_timers=big,
        max_pulses=big,
    )


def decompose_with_states(prog: Program, spec) -> List[Tuple[BuildState, Action]]:
    """정답 회로 → [(상태 스냅샷, 정답 다음수), ...] 학습 라벨.

    상태 스냅샷은 '그 액션을 적용하기 직전' 의 BuildState (stack/rungs/
    카운터 보유). 스펙은 state.spec 으로 접근. 정책망 입력 featurize 는
    하류에서 — 여기선 원형 그대로 낸다.
    """
    actions = program_to_actions(prog)
    st = _fresh_state(spec)
    pairs = []
    for a in actions:
        pairs.append((st.clone(), a))  # 액션 '전' 상태
        st.apply(a)
    return pairs


def actions_to_program(actions: List[Action], spec) -> Program:
    """액션열 → 복원 Program (라운드트립 검증용, legal 게이트 우회)"""
    st = _fresh_state(spec)
    for a in actions:
        st.apply(a)
    return st.to_program()


def verify_roundtrip(prog: Program, spec) -> bool:
    """분해→복원 후 evaluate 동일성 (라벨 유효성의 절대 기준)"""
    actions = program_to_actions(prog)
    back = actions_to_program(actions, spec)
    assert back is not None, "복원 결과가 빈 프로그램"
    return evaluate(back, spec) == evaluate(prog, spec)


# ---------- 디버그 출력 ----------


def action_str(a: Action) -> str:
    kind = a[0]
    if kind == "PUSH":
        return f"PUSH {a[1]}{'/' if a[2] == 'NC' else ''}"
    if kind == "EMIT":
        op = a[2] if len(a) > 2 else "OUT"
        return f"EMIT {a[1]}" + ("" if op == "OUT" else f" [{op}]")
    if kind == "TON":
        return f"TON K{a[1]}"
    return kind  # AND / OR / PLS / DONE


def iter_training_pairs(programs, spec):
    """여러 정답 회로 → 라벨 스트림 (reference + GP 해 등을 섞어 공급)"""
    for prog in programs:
        yield from decompose_with_states(prog, spec)


# ---------- 자가 점검 ----------

if __name__ == "__main__":
    from ladder_benchmark import make_tasks

    tasks = {t.name: t for t in make_tasks()}
    names = sys.argv[1:] or list(tasks)
    unknown = set(names) - set(tasks)
    assert not unknown, f"없는 과제: {unknown}"

    detail = len(sys.argv) > 1  # 과제 지정 시 액션열까지 출력
    total = 0
    for name in names:
        t = tasks[name]
        actions = program_to_actions(t.reference)
        ok = verify_roundtrip(t.reference, t.spec)
        pairs = decompose_with_states(t.reference, t.spec)
        total += len(pairs)
        mark = "OK " if ok else "FAIL"
        print(
            f"[{mark}] {name:<12} ref 크기 {program_size(t.reference):>2}"
            f"  →  {len(actions):>2}수 / {len(pairs):>2} 라벨"
        )
        assert ok, f"{name}: 라운드트립 evaluate 불일치"
        if detail:
            print(f"  회로: {program_str(t.reference).strip()}")
            print(textwrap.indent(ladder_str(t.reference), "  "))
            print("  액션열: " + "  ".join(action_str(a) for a in actions))
            print()

    print("-" * 52)
    print(f"reference {len(names)}과제에서 (상태→수) 라벨 {total}개 추출")
    print("라운드트립 검증 전부 통과 (evaluate 동일)")
