"""
ladder_gp.py — 유전 프로그래밍 (rung 단위 교차 + 서브트리 변이)

[ 왜 GP인가 ]
  interlock의 난점은 "좋은 Y0 rung과 좋은 Y1 rung이 동시에 있어야
  점수가 난다"는 것 — 샘플링(무작위/MCTS 롤아웃)은 둘을 한 번에 뽑아야
  하지만, rung 단위 교차는 서로 다른 후보가 반쪽씩 찾은 정답을 합칠 수
  있다. 2026-06-11 invariant 반증 후의 1순위 후보.

[ 설계 ]
  개체   = Program (rung 목록)
  적합도 = score 동일식 (일치율 - invariant 페널티 - 크기 페널티)
  선택   = 토너먼트 k=3 + 엘리트 2 (적합도 캐시, 재평가 안 함) + 새 피 5%
  교차   = rung 목록 단일점 스플라이스 (rung 순서 = 우선순위 보존)
  변이   = 서브트리 교체 / 접점 모드 반전 / 코일 교체 / rung 추가·삭제
  예산   = evaluate() 호출 수 (다른 방법과 동일 조건)

  교차·변이 후 Timer/Pulse 이름을 재부여한다 — 부모가 다른 rung끼리
  같은 T0를 공유하면 의도치 않은 타이머 상태 결합이 생긴다.

  무작위 개체/서브트리 생성기는 주입받는다 (ladder_benchmark의
  random_program_ext / random_logic_ext 재사용, 순환 import 회피).
"""

import random
from copy import deepcopy

from ladder.search import (
    coil_allowed,
    coil_usage,
    evaluate,
    program_size,
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
)

# ---------- 트리 유틸 ----------


def renumber_stateful(prog: Program):
    """Timer/Pulse 이름 재부여 — 교차로 합쳐진 rung 간 상태 공유 차단"""
    n = 0

    def walk(node):
        nonlocal n
        if isinstance(node, Timer):
            node.name = f'T{n}'
            n += 1
            walk(node.input)
        elif isinstance(node, Pulse):
            node.name = f'P{n}'
            n += 1
            walk(node.input)
        elif isinstance(node, (And, Or)):
            for a in node.args:
                walk(a)

    for r in prog.rungs:
        walk(r.logic)


def logic_positions(rung: Rung):
    """교체 가능한 모든 자리 — (컨테이너, 'logic'/'input' attr 또는 args 인덱스)"""
    out = [(rung, 'logic')]

    def walk(node):
        if isinstance(node, (And, Or)):
            for i, a in enumerate(node.args):
                out.append((node, i))
                walk(a)
        elif isinstance(node, (Timer, Pulse)):
            out.append((node, 'input'))
            walk(node.input)

    walk(rung.logic)
    return out


def set_at(container, slot, node):
    if isinstance(slot, int):
        container.args[slot] = node
    else:
        setattr(container, slot, node)


def contacts_of(rung: Rung):
    cs = []

    def walk(node):
        if isinstance(node, Contact):
            cs.append(node)
        elif isinstance(node, (And, Or)):
            for a in node.args:
                walk(a)
        elif isinstance(node, (Timer, Pulse)):
            walk(node.input)

    walk(rung.logic)
    return cs


# ---------- 유전 연산자 ----------


def crossover(
    a: Program, b: Program, rng: random.Random, max_rungs: int
) -> Program:
    """rung 목록 단일점 스플라이스 (rung 순서 보존)"""
    i = rng.randint(0, len(a.rungs))
    j = rng.randint(0, len(b.rungs))
    rungs = (a.rungs[:i] + b.rungs[j:])[:max_rungs]
    if not rungs:
        rungs = [rng.choice(a.rungs + b.rungs)]
    return deepcopy(Program(list(rungs)))


def dedupe_coils(prog: Program):
    """이중 코일 수선 — 위반하는 뒤쪽 rung 제거 (교차 직후 호출)"""
    used, keep = {}, []
    for r in prog.rungs:
        if coil_allowed(used, r.coil.device, r.coil.op):
            used.setdefault(r.coil.device, set()).add(r.coil.op)
            keep.append(r)
    prog.rungs = keep


def mutate(
    prog: Program,
    rng: random.Random,
    new_logic,
    coil_pool,
    coil_ops,
    max_rungs: int,
):
    kinds = ['subtree', 'contact', 'coil']
    weights = [0.45, 0.25, 0.15]
    if len(prog.rungs) < max_rungs:
        kinds.append('add')
        weights.append(0.08)
    if len(prog.rungs) > 1:
        kinds.append('del')
        weights.append(0.07)
    kind = rng.choices(kinds, weights)[0]

    if kind == 'add':
        used = coil_usage(prog.rungs)
        cands = [
            (d, op)
            for d in coil_pool
            for op in coil_ops
            if coil_allowed(used, d, op)
        ]
        if cands:
            d, op = rng.choice(cands)
            prog.rungs.append(Rung(Coil(d, op), new_logic(rng)))
        return
    if kind == 'del':
        prog.rungs.pop(rng.randrange(len(prog.rungs)))
        return
    rung = rng.choice(prog.rungs)
    if kind == 'coil':
        others = coil_usage([x for x in prog.rungs if x is not rung])
        cands = [
            (d, op)
            for d in coil_pool
            for op in coil_ops
            if coil_allowed(others, d, op)
        ]
        if cands:
            d, op = rng.choice(cands)
            rung.coil = Coil(d, op)
    elif kind == 'contact':
        cs = contacts_of(rung)
        if cs:
            c = rng.choice(cs)
            c.mode = 'NC' if c.mode == 'NO' else 'NO'
    else:
        container, slot = rng.choice(logic_positions(rung))
        set_at(container, slot, new_logic(rng))


# ---------- 메인 루프 ----------


def gp_search(
    spec,
    budget: int,
    seed: int,
    new_program,
    new_logic,
    coil_pool,
    max_rungs: int,
    allow_setrst: bool = False,
    pop_size: int = 200,
    tourney: int = 3,
    cross_p: float = 0.7,
    immigrant_p: float = 0.05,
    elite: int = 2,
):
    """반환: (found_at, best_acc, best_prog) — 벤치마크 러너 호환"""
    rng = random.Random(seed)
    coil_ops = ('OUT', 'SET', 'RST') if allow_setrst else ('OUT',)
    outputs = set(spec.outputs)

    evals = 0
    found_at = None
    best_score, best_acc, best_prog = -1.0, 0.0, None

    def fitness(prog):
        nonlocal evals, found_at, best_score, best_acc, best_prog
        evals += 1
        acc, viol = evaluate(prog, spec)
        s = acc - 0.05 * viol - 0.001 * program_size(prog)
        if acc > best_acc:
            best_acc = acc
        if s > best_score:
            best_score, best_prog = s, prog
        if acc >= 1.0 and viol == 0 and found_at is None:
            found_at = evals
            best_prog = prog
        return s

    def ensure_output_coil(prog):
        if not any(r.coil.device in outputs for r in prog.rungs):
            # 출력이 전혀 안 쓰인 상태 → 출력 OUT은 항상 coil_allowed
            r = rng.choice(prog.rungs)
            r.coil = Coil(rng.choice(spec.outputs))

    def make_child(pick):
        if rng.random() < immigrant_p:
            return new_program(rng)  # 새 피 (조기 수렴 완화)
        a = pick()
        if rng.random() < cross_p:
            child = crossover(a, pick(), rng, max_rungs)
            dedupe_coils(child)  # 스플라이스로 생긴 이중 코일 수선
        else:
            child = deepcopy(a)
        mutate(child, rng, new_logic, coil_pool, coil_ops, max_rungs)
        ensure_output_coil(child)
        renumber_stateful(child)
        return child

    entries = []  # (prog, fit) — 엘리트 캐시용
    for _ in range(pop_size):
        if evals >= budget or found_at:
            break
        p = new_program(rng)
        entries.append((p, fitness(p)))

    while evals < budget and found_at is None and entries:

        def pick():
            cand = [rng.randrange(len(entries)) for _ in range(tourney)]
            return entries[max(cand, key=lambda i: entries[i][1])][0]

        entries.sort(key=lambda e: e[1], reverse=True)
        nxt = entries[:elite]  # 엘리트는 적합도 캐시 그대로
        while len(nxt) < pop_size:
            if evals >= budget or found_at:
                break
            c = make_child(pick)
            nxt.append((c, fitness(c)))
        entries = nxt

    return found_at, best_acc, best_prog
