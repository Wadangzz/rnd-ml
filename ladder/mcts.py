"""
ladder_mcts.py — 래더 합성 MCTS (순수 UCT, 신경망 없음)

** 결론 (2026-06-10 벤치마크): 순수 UCT는 무작위 탐색에 전패. **
   delayed_off/flicker는 무작위만 성공, interlock/seq2는 양쪽 실패.
   원인: (a) 부분점수 높은 기만적 가지에 예산 집중
         (b) 롤아웃 분포가 재귀 생성기보다 나쁨 (PUSH 지배)
         (c) 분기 14~25 × 깊이 10~18에서 트리가 얕은 prefix만 기억.
   이 파일은 확장 본체가 아니라 '비교용 베이스라인'으로 유지.
   BuildState(스택 액션 게임 규칙)는 향후 정책망/롤아웃 개선의 기반.

[ 핵심 설계: 래더 만들기를 '한 수씩 두는 게임'으로 ]

  상태  = 지금까지의 액션 시퀀스 (= 부분 완성 프로그램)
  액션  = PUSH(dev,mode) : 접점을 스택에 올림
          AND / OR       : 스택 위 2개를 묶어 1개로
          EMIT(coil)     : 스택 top을 꺼내 rung으로 확정
          DONE           : 프로그램 완성 선언
  보상  = 완성된 프로그램의 score() (0~1, 시뮬레이터 기반)

  MCTS 4단계:
    1) 선택   : UCB1로 루트부터 유망한 자식 따라 내려감
    2) 확장   : 안 가본 액션 하나를 자식으로 추가
    3) 롤아웃 : 거기서부터 무작위로 끝까지 둬보고 점수 측정
    4) 역전파 : 점수를 경로 위 모든 노드에 반영

  무작위 탐색과의 차이: 좋은 부분 구조(높은 평균 점수의 가지)를
  '기억'하고 그 가지를 집중적으로 파고듦.
"""

import math
import random

from ladder.search import (
    Spec,
    coil_allowed,
    coil_usage,
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
)

# ---------- 게임 규칙 (상태/액션) ----------

MAX_ACTIONS = 14  # 한 게임 최대 수 (기본값)
MAX_STACK = 3  # 스택 깊이 제한 (기본값)
MAX_RUNGS = 2  # rung 수 제한 (기본값)


class BuildState:
    """부분 완성 프로그램 = 액션 시퀀스를 재생한 결과

    과제별 한도/재료는 생성자 인자로 조절:
      timer_presets=(2, 3)  → ("TON", preset) 액션 허용 (스택 top을 타이머로 감쌈)
      allow_pulse=True      → ("PLS",) 액션 허용 (스택 top을 상승엣지로 감쌈)
    """

    def __init__(
        self,
        spec: Spec,
        max_actions: int = MAX_ACTIONS,
        max_stack: int = MAX_STACK,
        max_rungs: int = MAX_RUNGS,
        timer_presets: tuple[int, ...] = (),
        allow_pulse: bool = False,
        max_timers: int = 2,
        max_pulses: int = 1,
        allow_setrst: bool = False,
    ):
        self.spec = spec
        self.max_actions = max_actions
        self.max_stack = max_stack
        self.max_rungs = max_rungs
        self.timer_presets = timer_presets
        self.allow_pulse = allow_pulse
        self.max_timers = max_timers
        self.max_pulses = max_pulses
        self.allow_setrst = allow_setrst
        self.stack: list = []  # 논리 노드 스택
        self.rungs: list[Rung] = []
        self.n_actions = 0
        self.n_timers = 0  # 이름 부여 + 개수 제한용
        self.n_pulses = 0
        self.done = False

    def clone(self) -> 'BuildState':
        s = BuildState.__new__(BuildState)
        s.spec = self.spec
        s.max_actions = self.max_actions
        s.max_stack = self.max_stack
        s.max_rungs = self.max_rungs
        s.timer_presets = self.timer_presets
        s.allow_pulse = self.allow_pulse
        s.max_timers = self.max_timers
        s.max_pulses = self.max_pulses
        s.allow_setrst = self.allow_setrst
        s.stack = list(self.stack)  # 노드는 불변 취급이라 얕은 복사 OK
        s.rungs = list(self.rungs)
        s.n_actions = self.n_actions
        s.n_timers = self.n_timers
        s.n_pulses = self.n_pulses
        s.done = self.done
        return s

    def legal_actions(self) -> list[tuple]:
        if self.done or self.n_actions >= self.max_actions:
            return []
        acts = []
        devices = self.spec.inputs + self.spec.internals + self.spec.outputs
        # PUSH
        if len(self.stack) < self.max_stack:
            for d in devices:
                acts.append(('PUSH', d, 'NO'))
                acts.append(('PUSH', d, 'NC'))
        # AND / OR
        if len(self.stack) >= 2:
            acts.append(('AND',))
            acts.append(('OR',))
        # TON / PLS: 스택 top을 감쌈
        if self.stack and self.n_timers < self.max_timers:
            for p in self.timer_presets:
                acts.append(('TON', p))
        if self.stack and self.allow_pulse and self.n_pulses < self.max_pulses:
            acts.append(('PLS',))
        # EMIT (allow_setrst면 SET/RST 코일도 — 래치 체인의 자연형)
        # 이중 코일 금지: 이미 쓴 코일과 충돌하는 EMIT은 액션에서 제외
        if len(self.stack) == 1 and len(self.rungs) < self.max_rungs:
            ops = ('OUT', 'SET', 'RST') if self.allow_setrst else ('OUT',)
            used = coil_usage(self.rungs)
            for c in self.spec.outputs + self.spec.internals:
                for op in ops:
                    if coil_allowed(used, c, op):
                        acts.append(('EMIT', c, op))
        # DONE: 출력 코일이 최소 1개 있고 스택이 비었을 때만
        if not self.stack and any(
            r.coil.device in self.spec.outputs for r in self.rungs
        ):
            acts.append(('DONE',))
        return acts

    def apply(self, act: tuple):
        kind = act[0]
        if kind == 'PUSH':
            self.stack.append(Contact(act[1], act[2]))
        elif kind == 'AND':
            b, a = self.stack.pop(), self.stack.pop()
            self.stack.append(And([a, b]))
        elif kind == 'OR':
            b, a = self.stack.pop(), self.stack.pop()
            self.stack.append(Or([a, b]))
        elif kind == 'TON':
            self.stack.append(
                Timer(f'T{self.n_timers}', act[1], self.stack.pop())
            )
            self.n_timers += 1
        elif kind == 'PLS':
            self.stack.append(Pulse(f'P{self.n_pulses}', self.stack.pop()))
            self.n_pulses += 1
        elif kind == 'EMIT':
            op = act[2] if len(act) > 2 else 'OUT'
            self.rungs.append(Rung(Coil(act[1], op), self.stack.pop()))
        elif kind == 'DONE':
            self.done = True
        self.n_actions += 1

    def is_terminal(self) -> bool:
        return self.done or not self.legal_actions()

    def to_program(self) -> Program | None:
        if not self.rungs:
            return None
        return Program(self.rungs)


# ---------- 평가 (롤아웃 끝에서 호출, 호출 횟수 = 예산) ----------


class Evaluator:
    """score() 호출을 세는 래퍼 — 무작위 탐색과 공정 비교용

    완벽해 판정은 accuracy==1.0 AND invariant 위반 0 기준
    (score는 페널티가 섞여 큰 정답 회로가 0.99 문턱을 영영 못 넘음)
    """

    def __init__(self, spec: Spec):
        self.spec = spec
        self.calls = 0
        self.best_score = -1.0
        self.best_acc = 0.0
        self.best_prog = None
        self.found_at = None  # 완벽해 최초 발견 시점

    def __call__(self, state: BuildState) -> float:
        prog = state.to_program()
        if prog is None:
            return 0.0
        self.calls += 1
        acc, viol = evaluate(prog, self.spec)
        # score()와 동일식, 시뮬 1회로
        s = acc - 0.05 * viol - 0.001 * program_size(prog)
        if acc > self.best_acc:
            self.best_acc = acc
        if s > self.best_score:
            self.best_score, self.best_prog = s, prog
        if acc >= 1.0 and viol == 0 and self.found_at is None:
            self.found_at = self.calls
            self.best_prog = prog
        return s


# ---------- 롤아웃 정책 ----------

# 균등 샘플은 PUSH가 지배한다 (디바이스 5종 × 2모드 = 10개 vs 구조 액션
# 1~2개) → 롤아웃이 스택만 쌓다 max_actions를 소진. 액션 '종류' 단위로
# 먼저 가중 샘플해 구조 액션(AND/OR/EMIT/DONE)의 발현 빈도를 복원한다.
KIND_WEIGHTS = {
    'PUSH': 1.0,
    'AND': 3.0,
    'OR': 3.0,
    'TON': 2.0,
    'PLS': 2.0,
    'EMIT': 4.0,
    'DONE': 3.0,
}


def weighted_rollout(
    state: BuildState, acts: list[tuple], rng: random.Random
) -> tuple:
    """종류 우선 가중 샘플 → 종류 내 균등 샘플"""
    kinds = {}
    for a in acts:
        kinds.setdefault(a[0], []).append(a)
    names = list(kinds)
    name = rng.choices(names, [KIND_WEIGHTS.get(n, 1.0) for n in names])[0]
    group = kinds[name]
    return group[rng.randrange(len(group))]


# ---------- MCTS (UCT) ----------


class Node:
    __slots__ = (
        'state',
        'parent',
        'children',
        'untried',
        'visits',
        'value_sum',
        'value_max',
        'action',
        'prior',
        'priors',
    )

    def __init__(self, state: BuildState, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: list[Node] = []
        self.untried = state.legal_actions()
        self.visits = 0
        self.value_sum = 0.0
        self.value_max = 0.0  # 이 가지 아래서 본 최고 점수
        self.prior = prior  # 부모가 본 P(이 액션) — PUCT 용
        self.priors = None  # 이 노드의 untried 액션별 prior 캐시

    def ucb1(self, c: float, mix: float = 0.5) -> float:
        if self.visits == 0:
            return float('inf')
        mean = self.value_sum / self.visits
        # 평균과 최대를 섞음: 단일 플레이어 결정론 문제의 정석.
        # 평균만 쓰면 '드물지만 완벽한 해'가 있는 가지가 묻힌다.
        exploit = (1 - mix) * mean + mix * self.value_max
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def puct(self, c: float, mix: float = 0.5) -> float:
        """AlphaZero식 선택: explore 항에 prior 를 곱함.
        미방문 자식도 inf 가 아니라 prior 크기 순으로 줄 세워진다 —
        prior 가 '탐색 방향'을 바꾸는 지점 (rollout 주입과의 본질 차이)."""
        if self.visits == 0:
            exploit = 0.0
        else:
            mean = self.value_sum / self.visits
            exploit = (1 - mix) * mean + mix * self.value_max
        explore = (
            c
            * self.prior
            * math.sqrt(self.parent.visits + 1)
            / (1 + self.visits)
        )
        return exploit + explore


def mcts_search(
    spec: Spec,
    budget: int,
    seed: int = 0,
    c_uct: float = 1.0,  # 스윕 결과: 0.7은 갇히고 2.0은 산만. 1.0이 최적
    state_factory=None,  # 과제별 BuildState 설정 주입용
    rollout_policy=None,  # None=균등, weighted_rollout 등 주입 가능
    prior_fn=None,  # (state, acts) -> probs. 주면 선택=PUCT, 확장=prior 내림차순
):
    rng = random.Random(seed)
    ev = Evaluator(spec)
    root = Node(state_factory() if state_factory else BuildState(spec))

    while ev.calls < budget and ev.found_at is None:
        # 1) 선택
        node = root
        while not node.untried and node.children:
            if prior_fn:
                node = max(node.children, key=lambda n: n.puct(c_uct))
            else:
                node = max(node.children, key=lambda n: n.ucb1(c_uct))

        # 2) 확장
        if node.untried and not node.state.is_terminal():
            if prior_fn:
                if node.priors is None:
                    probs = prior_fn(node.state, node.untried)
                    node.priors = {a: p for a, p in zip(node.untried, probs)}
                # prior 큰 액션부터 트리에 들어옴 (AlphaZero 확장 순서)
                act = max(node.untried, key=lambda a: node.priors[a])
                node.untried.remove(act)
                prior = node.priors[act]
            else:
                act = node.untried.pop(rng.randrange(len(node.untried)))
                prior = 0.0
            child_state = node.state.clone()
            child_state.apply(act)
            child = Node(child_state, parent=node, action=act, prior=prior)
            node.children.append(child)
            node = child

        # 3) 롤아웃: 정책 따라 끝까지 (기본 균등)
        ro = node.state.clone()
        while not ro.is_terminal():
            acts = ro.legal_actions()
            if rollout_policy:
                ro.apply(rollout_policy(ro, acts, rng))
            else:
                ro.apply(acts[rng.randrange(len(acts))])
        reward = ev(ro)

        # 4) 역전파 (평균용 합계 + 최대값 동시 갱신)
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            if reward > node.value_max:
                node.value_max = reward
            node = node.parent

    return ev


# ---------- 실험: MCTS vs 무작위 ----------

if __name__ == '__main__':
    spec = make_self_hold_spec()
    BUDGET = 200_000

    print('문제: 기동/정지 자기유지 회로 합성')
    print(f'예산: 시뮬레이터 호출 {BUDGET:,}회  |  완벽해 찾으면 조기 종료')
    print('=' * 62)

    print('\n--- MCTS (순수 UCT) ---')
    mcts_results = []
    for seed in [0, 1, 2]:
        ev = mcts_search(spec, BUDGET, seed)
        mcts_results.append(ev.found_at)
        status = f'{ev.found_at:,}회 만에 발견' if ev.found_at else '실패'
        print(f'[seed {seed}] {status}  (최고점수 {ev.best_score:.4f})')
        if ev.best_prog:
            print(program_str(ev.best_prog))

    # 무작위 탐색 결과 (ladder_search.py 동일 조건 실행값)
    random_results = [16203, 16721, 43108]

    print('\n' + '=' * 62)
    print('비교: 완벽해 발견까지 시뮬레이터 호출 수')
    print('-' * 62)
    print(f'{"seed":>6} | {"무작위 탐색":>12} | {"MCTS":>10} | {"배율":>8}')
    print('-' * 62)
    for i, (r, m) in enumerate(zip(random_results, mcts_results)):
        if m:
            print(f'{i:>6} | {r:>12,} | {m:>10,} | {r / m:>7.1f}x')
        else:
            print(f'{i:>6} | {r:>12,} | {"실패":>10} | {"-":>8}')
