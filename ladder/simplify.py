"""
ladder_simplify.py — 발견 해 단순화 패스 (동작 보존)

GP 발견 해의 비대함은 무작위 쓰레기가 아니라 기계적으로 제거 가능한
패턴들이다 (2026-06-11 출력 분석 실측):
  1. 상수 전파      — 어디서도 코일로 안 쓰인 디바이스는 항상 0
  2. 죽은 rung      — 출력에 도달 못 하는 rung / 읽히기 전에 덮어쓰이는 OUT
  3. 불 대수        — 평탄화 · 중복 제거 · 보원(X * X/ = 0) · 흡수(A * (A+B) = A)

[ 상태(Timer/Pulse) 안전 규칙 ]
  - 이름이 프로그램 내 유일하다는 전제 (GP renumber_stateful / 생성기 counter
    가 보장). 그러면 삭제되는 서브트리 안의 타이머 상태는 외부에서 관측
    불가능 → 통째 폴딩·삭제가 관측상 동등하다.
  - 단 cross-arg 규칙(중복/보원/흡수)은 순수(무상태) 항끼리만 비교한다 —
    구조가 같아도 이름이 다른 Timer 둘은 별개 상태라 합치면 안 됨.
  - 인자 순서는 절대 바꾸지 않는다 (and/or 단락 평가가 타이머 평가 횟수를
    바꾸지만, 위 유일성 전제로 결과는 동등 — 그래도 보수적으로 유지).

[ 절대 기준 ]
  simplify_program 전후로 evaluate(prog, spec) 가 완전히 동일해야 한다.
  (smoke_test 의 property 검사로 상시 검증)
"""

from copy import deepcopy

from ladder.search import evaluate, logic_str, program_size, program_str
from ladder.sim import And, Coil, Contact, Or, Program, Pulse, Rung, Timer

# ---------- 판별 유틸 ----------


def is_pure(node) -> bool:
    """Timer/Pulse 가 없는 순수 조합 논리인가"""
    if isinstance(node, Contact):
        return True
    if isinstance(node, (Timer, Pulse)):
        return False
    return all(is_pure(a) for a in node.args)


def contact_devices(node) -> set:
    """논리 트리가 읽는 디바이스 집합 (Timer/Pulse input 포함)"""
    if isinstance(node, Contact):
        return {node.device}
    if isinstance(node, (Timer, Pulse)):
        return contact_devices(node.input)
    out = set()
    for a in node.args:
        out |= contact_devices(a)
    return out


def taut(dev):
    """항상 참 (IR 에 상수 노드가 없어 표준형으로 표현)"""
    return Or([Contact(dev, 'NO'), Contact(dev, 'NC')])


def contradiction(dev):
    """항상 거짓"""
    return And([Contact(dev, 'NO'), Contact(dev, 'NC')])


# ---------- 논리 단순화 (노드 → 노드 | True | False) ----------


def simp(node, zero: set, taut_dev: str):
    if isinstance(node, Contact):
        if node.device in zero:
            return node.mode == 'NC'  # 항상0 디바이스: NO→False, NC→True
        return node

    if isinstance(node, Timer):
        inp = simp(node.input, zero, taut_dev)
        if inp is False:
            return False  # 입력 영원히 거짓 → 도달 불가
        if inp is True:
            inp = taut(taut_dev)  # 시간 의존이라 상수 아님 — 유지
        return Timer(node.name, node.preset, inp)

    if isinstance(node, Pulse):
        inp = simp(node.input, zero, taut_dev)
        if inp is False:
            return False  # 엣지 없음
        if inp is True:
            inp = taut(taut_dev)  # 첫 스캔 1펄스 — 보존해야 함
        return Pulse(node.name, inp)

    is_and = isinstance(node, And)

    # 자식 단순화 + 상수 폴딩 + 동종 평탄화 (인자 순서 보존)
    args = []
    for a in node.args:
        s = simp(a, zero, taut_dev)
        if s is True:
            if not is_and:
                return True
            continue
        if s is False:
            if is_and:
                return False
            continue
        if type(s) is type(node):
            args.extend(s.args)
        else:
            args.append(s)
    if not args:
        return is_and  # 전부 소거: And→True, Or→False

    # 중복 제거 (순수 항끼리만)
    seen, out = set(), []
    for a in args:
        if is_pure(a):
            k = logic_str(a)
            if k in seen:
                continue
            seen.add(k)
        out.append(a)
    args = out

    # 보원: 평접점 X 와 X/ 동시 존재
    plain = {(a.device, a.mode) for a in args if isinstance(a, Contact)}
    for d, m in plain:
        if (d, 'NC' if m == 'NO' else 'NO') in plain:
            return False if is_and else True

    # 흡수: And 안의 bare A 가 Or 형제의 disjunct 면 그 Or 제거 (쌍대 동일)
    bare_keys = {logic_str(a) for a in args if is_pure(a)}
    dual = Or if is_and else And
    kept = []
    for a in args:
        if (
            isinstance(a, dual)
            and is_pure(a)
            and any(logic_str(x) in bare_keys for x in a.args)
        ):
            continue
        kept.append(a)
    args = kept

    if len(args) == 1:
        return args[0]
    return (And if is_and else Or)(args)


# ---------- 프로그램 단순화 (fixpoint) ----------


def simplify_program(prog: Program, spec, max_iter: int = 20) -> Program:
    prog = deepcopy(prog)
    inputs = set(spec.inputs)
    outputs = set(spec.outputs)
    taut_dev = spec.inputs[0]

    for _ in range(max_iter):
        before = program_str(prog)

        # 1) 상수 디바이스: 어디서도 코일로 안 쓰임 + 입력 아님 → 항상 0
        written = {r.coil.device for r in prog.rungs}
        zero = set()
        for r in prog.rungs:
            zero |= {
                d
                for d in contact_devices(r.logic)
                if d not in written and d not in inputs
            }

        # 2) rung 별 논리 단순화 + 상수 rung 처리
        rungs = []
        for r in prog.rungs:
            s = simp(r.logic, zero, taut_dev)
            if s is False:
                if r.coil.op in ('SET', 'RST'):
                    continue  # 조건 영원히 거짓 → 무효 rung
                writers = sum(
                    1 for x in prog.rungs if x.coil.device == r.coil.device
                )
                if writers == 1:
                    continue  # 유일 작성자의 OUT 0 = 미작성과 동일
                s = contradiction(taut_dev)  # 다른 작성자 있음 → 0 쓰기 유지
            elif s is True:
                s = taut(taut_dev)
            rungs.append(Rung(Coil(r.coil.device, r.coil.op), s))
        prog = Program(rungs)

        # 3) liveness: 출력에서 역추적해 도달 불가능한 rung 제거
        live = set(outputs)
        changed = True
        while changed:
            changed = False
            for r in prog.rungs:
                if r.coil.device in live:
                    ds = contact_devices(r.logic)
                    if not ds <= live:
                        live |= ds
                        changed = True
        prog = Program([r for r in prog.rungs if r.coil.device in live])

        # 4) 덮어쓰기 죽은 rung: OUT 이후 읽히기 전에 같은 디바이스 OUT
        keep = []
        n = len(prog.rungs)
        for i, ri in enumerate(prog.rungs):
            dead = False
            if ri.coil.op == 'OUT':
                dev = ri.coil.device
                for j in range(i + 1, n):
                    rj = prog.rungs[j]
                    if dev in contact_devices(rj.logic):
                        break  # 그 전에 읽힘 → 필요
                    if rj.coil.device == dev:
                        dead = rj.coil.op == 'OUT'
                        break  # SET/RST 면 조건부라 유지
            if not dead:
                keep.append(ri)
        prog = Program(keep)

        if program_str(prog) == before:
            break
    return prog


# ---------- 스펙 보존 축소 (greedy ablation) ----------
#
# simplify_program 과 기준이 다르다:
#   simplify = 모든 입력에서 동작 동일 (구문적 보편 법칙만)
#   shrink   = 스펙(evaluate 결과)만 보존 — 스펙이 oracle.
#              런타임 상관관계 덕에 무의미해진 항을 스펙 검사로 제거하므로
#              스펙 밖 입력에서의 동작은 달라질 수 있다 (delta debugging 원리).


def _iter_paths(rung):
    """(path, node) 나열 — path 는 rung.logic 에서의 내비게이션 단계"""

    def walk(node, path):
        yield path, node
        if isinstance(node, (And, Or)):
            for i, a in enumerate(node.args):
                yield from walk(a, path + [i])
        elif isinstance(node, (Timer, Pulse)):
            yield from walk(node.input, path + ['input'])

    yield from walk(rung.logic, [])


def _get(rung, path):
    node = rung.logic
    for step in path:
        node = node.input if step == 'input' else node.args[step]
    return node


def _replace(rung, path, new):
    if not path:
        rung.logic = new
        return
    parent = _get(rung, path[:-1])
    if path[-1] == 'input':
        parent.input = new
    else:
        parent.args[path[-1]] = new


def _shrink_candidates(prog: Program):
    """한 걸음 작아진 변형들 (각각 독립 deepcopy)"""
    for i in range(len(prog.rungs)):  # rung 삭제
        c = deepcopy(prog)
        del c.rungs[i]
        yield c
    for ri, rung in enumerate(prog.rungs):
        for path, node in _iter_paths(rung):
            if isinstance(node, (And, Or)):  # 인자 삭제 (2개면 unwrap)
                for k in range(len(node.args)):
                    c = deepcopy(prog)
                    cn = _get(c.rungs[ri], path)
                    if len(cn.args) == 2:
                        _replace(c.rungs[ri], path, cn.args[1 - k])
                    else:
                        del cn.args[k]
                    yield c
            elif isinstance(node, (Timer, Pulse)):  # 타이머/펄스 벗기기
                c = deepcopy(prog)
                _replace(c.rungs[ri], path, _get(c.rungs[ri], path).input)
                yield c


def shrink_program(prog: Program, spec) -> Program:
    """evaluate(spec) 가 변하지 않는 한 탐욕적으로 제거. 크기 단조 감소로
    종료 보장. 채택 조건이 oracle 검사라 스펙 보존은 구조상 깨질 수 없다."""
    target = evaluate(prog, spec)
    prog = deepcopy(prog)
    changed = True
    while changed:
        changed = False
        for cand in _shrink_candidates(prog):
            if (
                program_size(cand) < program_size(prog)
                and evaluate(cand, spec) == target
            ):
                prog = cand
                changed = True
                break
    return prog


def polish_program(prog: Program, spec) -> Program:
    """발견 해 정리 표준 절차: 단순화 → 스펙 축소 → 마무리 단순화"""
    out = simplify_program(prog, spec)
    out = shrink_program(out, spec)
    return simplify_program(out, spec)
