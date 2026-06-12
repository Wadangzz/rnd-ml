"""
ladder_policy.py — 정책망 prior (모방학습) + numpy 추론 롤아웃

[ 왜 ]
  seq3 가 GP 까지 무너지는 경계로 확정됐다 (ladder_benchmark). 샘플링 3방법은
  0.880~0.893 고원, GP 는 0.940~0.973 으로 근접하나 22크기 3단 체인을 완성
  조립 못 한다. 이 building-block 한계를 **학습된 prior** 가 메울 수 있는가가
  다음 검증 대상. 그 첫 코드 — 정답 회로의 (상태→다음수) 라벨로 정책망을
  모방학습(behavior cloning)하고, MCTS 롤아웃에 prior 로 꽂는다.

[ 설계 핵심 두 가지 ]
  1) per-action 스코어링: 고정 액션 사전이 아니라 (상태특징 ⊕ 액션특징) → 점수.
     과제마다 디바이스 집합이 달라도 같은 망이 동작한다 (스펙 요약만 입력).
  2) 학습은 torch(GPU), 추론은 numpy: MCTS 롤아웃은 스텝마다 합법 액션을
     점수매겨야 하는데(수백만 회) 그 핫루프의 torch 호출 오버헤드가 stdlib
     weighted_rollout 보다 수십 배 느리다. 학습 후 가중치를 numpy 로 빼서
     롤아웃은 numpy 순전파로 돌린다 (작은 MLP 는 numpy 가 microsecond 급).

[ 라벨 ]
  ladder_decompose.decompose_with_states(prog, spec) = (상태 스냅샷, 정답 다음수).
  라벨링 무비용 — 정답회로 자신이 다음수 오라클. reference 8과제 = 94 라벨,
  GP 해를 섞으면 증가 (--gp).

[ 주의: candidate 보강 ]
  decompose 의 _fresh_state 는 timer_presets=() 라 legal_actions() 에 ("TON",p)
  가 없다 (apply 로 직접 재생하므로 분해엔 문제없음). 학습 시 정답이
  candidate 집합에 반드시 있어야 하므로 candidate = legal ∪ {정답} 으로 둔다.
  추론(MCTS) 시점엔 과제별 BuildState 가 TON 을 정상 legal 로 내므로 무관.

[ 실행 ]
  python ladder_policy.py                 # reference 로 학습 + top-1 정확도 + 가중치 저장
  python ladder_policy.py --gp            # GP 해도 섞어 학습 (느림)
  python ladder_policy.py --mcts seq3     # 학습 후 3구성 MCTS 로 seq3 측정
  python ladder_policy.py --holdout seq3  # seq3 라벨 제외 학습 → seq3 측정 (전이 시험)
  python ladder_policy.py --holdout seq3 --curriculum 12   # + 2단 체인 변형 12개
"""

import sys

import numpy as np
import torch
import torch.nn as nn

from ladder.benchmark import make_tasks, run_gp
from ladder.decompose import iter_training_pairs
from ladder.mcts import BuildState, mcts_search, weighted_rollout
from ladder.search import program_str
from ladder.sim import And, Contact, Or
from ladder.simplify import polish_program

Action = tuple

# ---------- featurize ----------

KINDS = ['PUSH', 'AND', 'OR', 'TON', 'PLS', 'EMIT', 'DONE']
OPS = ['OUT', 'SET', 'RST']

STATE_DIM = 16
# kind(7) + role(3) + mode(2) + op(3) + preset(1) + 문맥(5) + 스펙역할(9) = 30
ACTION_DIM = len(KINDS) + 3 + 2 + 3 + 1 + 5 + 9
FEAT_DIM = STATE_DIM + ACTION_DIM


# ---------- 스펙 기능 역할 (featurizer v3) ----------
#
# 6차 실측: 배정 변형 커리큘럼이 무승부 — featurizer 가 스펙에서 풀
# 크기만 읽어 "어느 X 가 기동인지"를 원리적으로 구분 못 함 (같은 풀
# 크기 + 다른 배정 = 같은 특징 + 다른 정답 라벨). 여기서 시나리오
# trace 로부터 디바이스별 기능 역할을 기계적으로 도출한다 — 이름
# 공간이 아니라 역할 공간에서 관용구를 학습하게 하는 재인덱싱.
#
#   입력 d: 기동성 (출력이 발화하는 시나리오에서 처음 눌림) /
#           d 의 press 직후 점등·소등하는 출력과 그 점등 순서
#   출력 y: 시나리오에서 몇 번째로 점등하나 (체인 단계 순서)
#
# don't-care 마스킹(전이 스캔 None)을 견디도록 press 전후 값은
# 가장 가까운 비 None 스캔에서 읽는다. 스펙은 불변 → id 캐시.

_ROLE_CACHE: dict = {}


def spec_roles(spec) -> dict:
    cached = _ROLE_CACHE.get(id(spec))
    if cached is not None:
        return cached
    on_votes: dict = {}
    off_votes: dict = {}
    start: set = set()
    first_on: dict = {}
    for sc in spec.scenarios:
        # press 이벤트 수집 (0→1 전이)
        cur: dict = {}
        presses = []
        for t, upd in enumerate(sc.input_trace):
            for k, v in upd.items():
                if cur.get(k, 0) == 0 and v == 1:
                    presses.append((t, k))
            cur.update(upd)

        def val(y, t):
            if 0 <= t < len(sc.expected):
                return sc.expected[t].get(y)
            return None

        def nearest_before(y, t):
            for u in range(t - 1, -1, -1):
                v = val(y, u)
                if v is not None:
                    return v
            return 0  # 기록 이전 = 전원 투입 직후 0

        def first_change(y, t, before, horizon=5):
            """press 후 출력이 처음 '변한' 스캔까지의 지연 (없으면 None).

            모티프 정체성의 핵심 — 래치는 관측 지연 ~1 (전이 스캔이 마스킹돼
            한 칸 밀림), preset 3 타이머는 ~2. tchain 1차 실측: 지연을 버리는
            추출은 래치/타이머 스펙이 동일 역할 구조로 보여 prior 가 우세
            관용구(래치)를 자신 있게 오답으로 냄 (0.962 고원 ×6런).
            """
            for u in range(t, min(t + horizon, len(sc.expected))):
                v = val(y, u)
                if v is not None and v != before:
                    return u - t
            return None

        fired = False
        for y in spec.outputs:
            for t, w in enumerate(sc.expected):
                if w.get(y) == 1:
                    first_on[y] = min(first_on.get(y, 10**9), t)
                    fired = True
                    break
        for t, d in presses:
            for y in spec.outputs:
                b = nearest_before(y, t)
                delay = first_change(y, t, b)
                if delay is None:
                    continue
                tgt = on_votes if b == 0 else off_votes
                tgt.setdefault(d, {}).setdefault(y, []).append(delay)
        if presses and fired:
            t0 = presses[0][0]  # 출력이 발화한 시나리오의 첫 press = 기동 후보
            start.update(d for t, d in presses if t == t0)

    # 출력 점등 순서 rank (점등 안 하는 distractor 는 None)
    lit = sorted(first_on, key=lambda y: first_on[y])
    rank = {y: i for i, y in enumerate(lit)}

    def dominant(votes):
        """입력별 최다 득표 출력 → (rank, 평균 지연)"""
        out = {}
        for d, ys in votes.items():
            y = max(ys, key=lambda k: len(ys[k]))
            out[d] = (rank.get(y, 0) or 0, sum(ys[y]) / len(ys[y]))
        return out

    roles = {
        'start': start,
        'on_rank': dominant(on_votes),  # 입력 → (점등 출력 rank, 지연)
        'off_rank': dominant(off_votes),  # 입력 → (소등 출력 rank, 지연)
        'out_rank': rank,  # 출력 → 점등 순서 rank
    }
    _ROLE_CACHE[id(spec)] = roles
    return roles


def role_feats(dev: str, roles: dict) -> list[float]:
    """디바이스 1개 → 스펙 역할 특징 9차원 (해당 없으면 0).

    지연 2차원이 모티프 판별자 — 래치(~1) vs 타이머(~2+)."""
    on_r = roles['on_rank'].get(dev)  # (rank, delay) | None
    off_r = roles['off_rank'].get(dev)
    self_r = roles['out_rank'].get(dev)
    return [
        1.0 if dev in roles['start'] else 0.0,
        1.0 if on_r is not None else 0.0,
        (on_r[0] if on_r else 0) / 5.0,
        (on_r[1] if on_r else 0) / 4.0,  # 점등 지연
        1.0 if off_r is not None else 0.0,
        (off_r[0] if off_r else 0) / 5.0,
        (off_r[1] if off_r else 0) / 4.0,  # 소등 지연
        1.0 if self_r is not None else 0.0,
        (self_r or 0) / 5.0,
    ]


def dev_role(spec, dev: str) -> int:
    """input=0 / internal=1 / output=2"""
    if dev in spec.inputs:
        return 0
    if dev in spec.internals:
        return 1
    return 2


def dev_idx(spec, dev: str) -> int:
    """역할 풀 내 인덱스 (X1→1, Y2→2). 순서 패턴 학습용"""
    for pool in (spec.inputs, spec.internals, spec.outputs):
        if dev in pool:
            return pool.index(dev)
    return 0


def iter_contacts(node):
    if isinstance(node, Contact):
        yield node
    elif isinstance(node, (And, Or)):
        for a in node.args:
            yield from iter_contacts(a)
    else:  # Timer / Pulse
        yield from iter_contacts(node.input)


class Ctx:
    """featurize 공유 문맥 — 상태당 1회 계산 (후보 수만큼 재계산 방지)

    v2 핵심: v1 의 카운트 요약은 '스택에 뭐가 있는지 / 어느 출력을
    채웠는지'를 못 봐 seq3 top-1 이 0.609 에서 천장 (동일 특징 → 다른
    정답 라벨이 원리적으로 구분 불가). 진행 포인터 + 스택 내용물 +
    상대 인덱스로 그 정보를 공급한다.
    """

    def __init__(self, state: BuildState):
        spec = state.spec
        self.coiled = {r.coil.device for r in state.rungs}
        # 진행 포인터: 아직 코일을 안 쓴 첫 출력 (전부 썼으면 n_out)
        self.next_out = next(
            (i for i, d in enumerate(spec.outputs) if d not in self.coiled),
            len(spec.outputs),
        )
        self.stack_devs = set()
        for node in state.stack:
            for c in iter_contacts(node):
                self.stack_devs.add(c.device)
        self.roles = spec_roles(spec)  # v3: 스펙 기능 역할 (id 캐시라 비용 0)


def featurize_state(state: BuildState, ctx: Ctx) -> list[float]:
    """BuildState + 스펙 → 고정 길이 상태 벡터 (디바이스 수 무관)"""
    spec = state.spec
    outs = set(spec.outputs)
    emitted = len({r.coil.device for r in state.rungs if r.coil.device in outs})
    n_out = len(spec.outputs) or 1
    top = state.stack[-1] if state.stack else None
    top_contacts = list(iter_contacts(top)) if top is not None else []
    return [
        len(state.stack) / 5.0,
        len(state.rungs) / 5.0,
        state.n_timers / 3.0,
        state.n_pulses / 2.0,
        state.n_actions / 20.0,
        emitted / n_out,  # 이미 채운 출력 비율
        1.0 if not state.stack else 0.0,  # 스택 빔 (EMIT/DONE 국면)
        len(spec.inputs) / 5.0,
        len(spec.outputs) / 5.0,
        len(spec.internals) / 5.0,
        # --- v2: 진행 포인터 + 스택 top 내용물 요약 ---
        ctx.next_out / n_out,
        1.0 if top is not None else 0.0,
        min(len(top_contacts), 5) / 5.0,
        1.0 if any(c.device in spec.inputs for c in top_contacts) else 0.0,
        1.0 if any(c.device in outs for c in top_contacts) else 0.0,
        1.0 if any(c.mode == 'NC' for c in top_contacts) else 0.0,
    ]


def featurize_action(state: BuildState, act: Action, ctx: Ctx) -> list[float]:
    """액션 → 고정 길이 벡터 (종류 one-hot + 역할/모드/op/preset + 문맥 5)"""
    spec = state.spec
    kind = act[0]
    v = [0.0] * len(KINDS)
    v[KINDS.index(kind)] = 1.0
    role = [0.0, 0.0, 0.0]
    mode = [0.0, 0.0]  # NO, NC
    op = [0.0, 0.0, 0.0]  # OUT, SET, RST
    preset = 0.0
    # v2 문맥: 인덱스/상대위치/코일기왕/다음출력여부/스택기왕
    c_idx, c_rel, c_coiled, c_isnext, c_instack = 0.0, 0.0, 0.0, 0.0, 0.0
    dev = None
    if kind == 'PUSH':
        dev = act[1]
        role[dev_role(spec, dev)] = 1.0
        mode[0 if act[2] == 'NO' else 1] = 1.0
    elif kind == 'EMIT':
        dev = act[1]
        role[dev_role(spec, dev)] = 1.0
        op[OPS.index(act[2] if len(act) > 2 else 'OUT')] = 1.0
    elif kind == 'TON':
        preset = act[1] / 10.0
    rf = [0.0] * 9
    if dev is not None:
        idx = dev_idx(spec, dev)
        c_idx = idx / 5.0
        c_rel = (idx - ctx.next_out) / 5.0  # 진행 포인터 대비 상대 위치
        c_coiled = 1.0 if dev in ctx.coiled else 0.0
        n_out = len(spec.outputs)
        if ctx.next_out < n_out and dev == spec.outputs[ctx.next_out]:
            c_isnext = 1.0
        c_instack = 1.0 if dev in ctx.stack_devs else 0.0
        rf = role_feats(dev, ctx.roles)  # v3: 이름이 아니라 역할로 식별
    return (
        v
        + role
        + mode
        + op
        + [preset, c_idx, c_rel, c_coiled, c_isnext, c_instack]
        + rf
    )


def featurize_pairs(state: BuildState, acts: list[Action]) -> np.ndarray:
    """상태 1개 × 후보 액션들 → 특징 행렬 [n, FEAT_DIM] (문맥 1회 계산)"""
    ctx = Ctx(state)
    sf = featurize_state(state, ctx)
    return np.array(
        [sf + featurize_action(state, a, ctx) for a in acts], dtype=np.float32
    )


def candidate_actions(state: BuildState, correct: Action) -> list[Action]:
    """학습용 후보 = 합법 액션 ∪ {정답} (timer_presets=() 로 빠진 TON 보강)"""
    cands = list(state.legal_actions())
    if correct not in cands:
        cands.append(correct)
    return cands


# ---------- 데이터셋 ----------


def build_samples(tasks, include_gp=False, gp_budget=80_000, gp_seeds=(0, 1)):
    """각 과제 정답 회로 → [(후보특징행렬 [n,FEAT], 정답인덱스), ...]"""
    samples = []
    per_task = {}
    for t in tasks:
        progs = [t.reference]
        if include_gp:
            for s in gp_seeds:
                found, _, prog = run_gp(t, gp_budget, s)
                if found and prog is not None:
                    # 날것 GP 해는 비대한 노이즈 (4차 실측: 그대로 분해하면 관용구
                    # 라벨이 7:1 로 압도돼 전이 이득이 사라짐 — '분포가 관건' 역실증).
                    # polish (단순화→스펙축소) 로 관용구 크기로 정리한 뒤 분해한다.
                    progs.append(polish_program(prog, t.spec))
        n0 = len(samples)
        for state, act in iter_training_pairs(progs, t.spec):
            cands = candidate_actions(state, act)
            tgt = cands.index(act)
            samples.append((featurize_pairs(state, cands), tgt))
        per_task[t.name] = (n0, len(samples))
    return samples, per_task


# ---------- 정책망 (torch) ----------


class PolicyNet(nn.Module):
    def __init__(self, d_in: int = FEAT_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # x [n, d_in] → [n]
        return self.net(x).squeeze(-1)


def train(samples, epochs=1500, lr=1e-3, seed=0, device=None):
    """모방학습: 각 상태의 후보 위 cross-entropy (정답수를 1등으로)"""
    torch.manual_seed(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = PolicyNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # 샘플 텐서 미리 device 로
    data = [
        (torch.from_numpy(x).to(device), torch.tensor([t], device=device))
        for x, t in samples
    ]
    ce = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        total = torch.zeros((), device=device)
        for x, tgt in data:
            logits = model(x).unsqueeze(0)  # [1, n_cand]
            total = total + ce(logits, tgt)
        total = total / len(data)
        total.backward()
        opt.step()
        if ep % 100 == 0 or ep == 1:
            print(f'  epoch {ep:>4}  loss {total.item():.4f}', flush=True)
    return model, device


# ---------- numpy 추론 (롤아웃 핫루프용) ----------


def extract_weights(model: PolicyNet):
    lins = [m for m in model.net if isinstance(m, nn.Linear)]
    w = {}
    for i, lin in enumerate(lins, 1):
        w[f'W{i}'] = lin.weight.detach().cpu().numpy().astype(np.float32)
        w[f'b{i}'] = lin.bias.detach().cpu().numpy().astype(np.float32)
    return w


def np_forward(w, X: np.ndarray) -> np.ndarray:
    h = np.maximum(X @ w['W1'].T + w['b1'], 0.0)
    h = np.maximum(h @ w['W2'].T + w['b2'], 0.0)
    return (h @ w['W3'].T + w['b3']).reshape(-1)


def _softmax(logits: np.ndarray, temp: float) -> np.ndarray:
    z = (logits - logits.max()) / temp
    p = np.exp(z)
    return p / p.sum()


def make_net_rollout(w, temp: float = 1.0):
    """학습 가중치 → mcts_search(rollout_policy=) 어댑터.
    rng(random.Random) 로 샘플해 시드 재현성 유지."""

    def policy(state, acts, rng):
        p = _softmax(np_forward(w, featurize_pairs(state, acts)), temp)
        r = rng.random()
        c = 0.0
        for i, pi in enumerate(p):
            c += pi
            if r <= c:
                return acts[i]
        return acts[-1]

    return policy


def make_prior_fn(w, temp: float = 1.0):
    """학습 가중치 → mcts_search(prior_fn=) 어댑터 (PUCT 선택단계용)"""

    def prior(state, acts):
        return _softmax(
            np_forward(w, featurize_pairs(state, acts)), temp
        ).tolist()

    return prior


# ---------- 평가 (모방 품질) ----------


def top1_accuracy(w, samples) -> float:
    """후보 중 argmax 점수가 정답인 비율 (모방이 구조를 배웠나)"""
    hit = 0
    for x, tgt in samples:
        if int(np.argmax(np_forward(w, x))) == tgt:
            hit += 1
    return hit / len(samples)


# ---------- 메인 ----------

WEIGHTS_PATH = 'policy_weights.npz'


def main():
    args = sys.argv[1:]
    include_gp = '--gp' in args
    mcts_task = None
    if '--mcts' in args:
        i = args.index('--mcts')
        mcts_task = args[i + 1] if i + 1 < len(args) else 'seq3'
    # held-out 시험: 해당 과제 라벨을 학습에서 빼고 그 과제를 측정 —
    # 암기 재생이 아니라 관용구 '전이'가 일어나는지 (오염 없는 일반화 시험)
    holdout = None
    if '--holdout' in args:
        i = args.index('--holdout')
        holdout = args[i + 1] if i + 1 < len(args) else 'seq3'
        mcts_task = holdout
    # 모티프 변형 커리큘럼 (5차 교훈: 증강 축 = 같은 모티프의 다른 과제)
    n_var = 0
    if '--curriculum' in args:
        i = args.index('--curriculum')
        has_n = i + 1 < len(args) and args[i + 1].isdigit()
        n_var = int(args[i + 1]) if has_n else 12

    tasks = make_tasks()
    train_tasks = [t for t in tasks if t.name != holdout]
    if n_var:
        from ladder_curriculum import make_chain_curriculum

        # 2단 + 3단 혼합. 3단 변형이 rank=2 특징값을 분포 안으로 들여온다
        # (7차 OOD 외삽 진단). canonical 3단 = seq3 는 생성기가 제외.
        n2 = (n_var + 1) // 2
        n3 = n_var - n2
        variants = make_chain_curriculum(n2, K=2)
        if n3:
            variants += make_chain_curriculum(n3, K=3)
        train_tasks = train_tasks + variants
        print(f'curriculum: 체인 변형 2단 {n2} + 3단 {n3} 추가')
    print(
        f'featurize: state {STATE_DIM} + action {ACTION_DIM} = {FEAT_DIM} dim'
    )
    if holdout:
        print(f'held-out: {holdout} 라벨 제외 학습 → {holdout} 측정')
    print(f'학습 데이터 구성 중 (include_gp={include_gp})...')
    samples, per_task = build_samples(train_tasks, include_gp=include_gp)
    print(f'총 {len(samples)} 라벨 ({len(train_tasks)}과제)')

    print('\n모방학습 시작')
    model, device = train(samples)
    print(f'  device={device}')
    w = extract_weights(model)
    np.savez(WEIGHTS_PATH, **w)
    print(f'  가중치 저장 → {WEIGHTS_PATH}')

    print('\ntop-1 정확도 (후보 중 정답을 1등으로 뽑나)')
    print(f'  전체: {top1_accuracy(w, samples):.3f}')
    for t in train_tasks:
        lo, hi = per_task[t.name]
        if hi > lo:
            acc = top1_accuracy(w, samples[lo:hi])
            print(f'  {t.name:<12} {acc:.3f}  ({hi - lo} 라벨)')

    if mcts_task:
        t = next(t for t in tasks if t.name == mcts_task)
        budget, seeds = 200_000, [0, 1, 2]
        rollout = make_net_rollout(w)
        prior = make_prior_fn(w)
        # prior 를 꽂는 두 위치를 분리 측정:
        #   net-rollout = 롤아웃만 (1차에서 무승부였던 구성 — featurizer v2 재검)
        #   puct+w      = 선택단계만 (PUCT 단독 효과)
        #   puct+net    = 양쪽 다
        configs = [
            ('net-rollout', dict(rollout_policy=rollout)),
            ('puct+w', dict(rollout_policy=weighted_rollout, prior_fn=prior)),
            ('puct+net', dict(rollout_policy=rollout, prior_fn=prior)),
        ]
        print(f'\nMCTS 실험 — {mcts_task} (예산 {budget:,} / 시드 {seeds})')
        print('  기준: mcts_w 0.880~0.893 고원 / gp 0.940~0.973, 둘 다 미발견')
        for name, kw in configs:
            for seed in seeds:
                ev = mcts_search(
                    t.spec,
                    budget,
                    seed,
                    state_factory=lambda: BuildState(t.spec, **t.mcts_kwargs),
                    **kw,
                )
                stat = (
                    f'{ev.found_at:,}회 발견'
                    if ev.found_at
                    else f'실패 (acc {ev.best_acc:.3f})'
                )
                print(f'  {name:<12} seed{seed}: {stat}', flush=True)
                if ev.found_at and ev.best_prog:
                    print(program_str(ev.best_prog))


if __name__ == '__main__':
    main()
