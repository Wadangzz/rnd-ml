"""
diag_rank_ood.py — step 2 진단: 길이 외삽이 깨지는 지점을 '눈으로' 확인

[ 왜 ]
  어제 "길이 외삽(K<=3 학습 -> K=4)이 리트머스지"라고 결론냈고, 7차에서
  깨지는 원인을 'OOD rank'로 추정만 했지 직접 본 적은 없다. 코드(인코딩)를
  고치기 전에, K=4 시험에서 prior 가 마지막 단(rank=2 너머)에서 정확히
  분포 밖 특징값을 받는지를 featurize 결과로 확인한다.

[ 무엇 ]
  학습 불필요 — OOD 는 입력 특징값의 문제다. 학습 분포(8과제 ref + K=2/3
  변형, seq4 미공급) vs 시험(seq4 canonical) 에서 위치 특징이 갖는 값
  집합을 비교:
    - c_idx      : 절대 dev_idx /5      (의심 1)
    - on/off/out_rank : v3 절대 스펙-rank (의심 2 — 7차가 지목, 상대 대응물 없음)
    - c_rel      : (idx - next_out) /5  (이미 상대 — OOD 안 나면 처방의 존재증명)

[ 판정 ]
  절대 특징이 seq4 에서 학습 support 밖 값(rank=3 등)을 받고, c_rel 은 안
  받으면 -> "절대 rank 를 c_rel 처럼 상대로 바꾼다"(step 3 수술)가 정당화됨.

[ 실행 ]
  uv run experiments/diag_rank_ood.py
"""

from collections import defaultdict

from k4_probe import make_seq4

from ladder.benchmark import make_tasks
from ladder.curriculum import make_chain_curriculum
from ladder.decompose import decompose_with_states
from ladder.policy import (
    Ctx,
    candidate_actions,
    dev_idx,
    spec_roles,
)

FEATS = [
    'c_idx',
    'c_rel',
    'on_rank',
    'off_rank',
    'out_rank',
    # 후보 상대 인코딩 (수술 전 coverage 측정) — out_rank 를 대표로
    'out_rank_rel_next',  # A: rank - rank(진행 포인터 출력)
    'out_rank_rel_emit',  # 대안: rank - 코일 완료 출력 수
]


def device_of(act):
    return act[1] if act[0] in ('PUSH', 'EMIT') else None


def collect(tasks):
    """task 목록 -> {특징: 등장한 raw 값 집합}. net 이 평가하는 모든
    디바이스 후보(legal ∪ 정답)의 featurize 입력을 그대로 훑는다."""
    vals = defaultdict(set)
    for t in tasks:
        spec = t.spec
        roles = spec_roles(spec)
        n_out = len(spec.outputs)
        n_lit = len(roles['out_rank'])  # 점등(rank 부여) 출력 수
        for state, correct in decompose_with_states(t.reference, spec):
            ctx = Ctx(state)
            emitted = sum(1 for d in ctx.coiled if d in spec.outputs)
            # A 기준점: 진행 포인터가 가리키는 출력의 rank
            ref = 0
            if ctx.next_out < n_out:
                ref = roles['out_rank'].get(spec.outputs[ctx.next_out], 0)
            for act in candidate_actions(state, correct):
                dev = device_of(act)
                if dev is None:
                    continue
                idx = dev_idx(spec, dev)
                vals['c_idx'].add(idx)
                vals['c_rel'].add(idx - ctx.next_out)
                on_r = roles['on_rank'].get(dev)
                off_r = roles['off_rank'].get(dev)
                self_r = roles['out_rank'].get(dev)
                if on_r is not None:
                    vals['on_rank'].add(on_r[0])
                if off_r is not None:
                    vals['off_rank'].add(off_r[0])
                if self_r is not None:
                    vals['out_rank'].add(self_r)
                    vals['out_rank_rel_next'].add(self_r - ref)
                    vals['out_rank_rel_emit'].add(self_r - emitted)
                    # B: 연속 정규화 (외삽 X 보간 — 끝점 K 무관 일치)
                    if n_lit > 1:
                        vals['out_rank_norm'].add(round(self_r / (n_lit - 1), 3))
                    vals['out_rank_norm2'].add(round(self_r / n_lit, 3))
    return vals


def seq4_correct_trace(seq4):
    """seq4 reference 를 조립하는 정답 액션열에서, 출력 EMIT 마다 그
    출력의 out_rank 를 찍는다 — '몇 단째에서 분포 밖으로 나가는가'."""
    spec = seq4.spec
    roles = spec_roles(spec)
    rows = []
    for state, correct in decompose_with_states(seq4.reference, spec):
        dev = device_of(correct)
        if dev is None or dev not in spec.outputs:
            continue
        ctx = Ctx(state)
        rows.append(
            (dev, roles['out_rank'].get(dev), dev_idx(spec, dev) - ctx.next_out)
        )
    return rows


def fmt(s):
    return '{' + ', '.join(str(v) for v in sorted(s)) + '}'


if __name__ == '__main__':
    train_tasks = (
        make_tasks()  # 8 ref (seq3=K3 포함, seq4 미포함)
        + make_chain_curriculum(8, K=2)
        + make_chain_curriculum(8, K=3)
    )
    seq4 = make_seq4()

    print('학습 = 8 ref(+seq3) + K=2/3 변형 | 시험 = seq4 (held-out)\n')
    tr = collect(train_tasks)
    te = collect([seq4])

    print(f'{"feat":<10}{"학습 support":<22}{"seq4":<18}OOD (seq4 not in 학습)')
    print('-' * 78)
    for f in FEATS:
        ood = te[f] - tr[f]
        flag = fmt(ood) if ood else '-'
        print(f'{f:<20}{fmt(tr[f]):<26}{fmt(te[f]):<22}{flag}')

    print('\n연속 정규화 후보 (B) — seq4 range 가 학습 range 안이면 보간(OOD 아님):')
    print(f'  {"feat":<18}{"학습 range":<18}{"seq4 range":<18}판정')
    for f in ('out_rank_norm', 'out_rank_norm2'):
        a, b = sorted(tr[f]), sorted(te[f])
        ok = b[0] >= a[0] and b[-1] <= a[-1]
        verdict = '보간 OK' if ok else '외삽'
        print(
            f'  {f:<18}[{a[0]}, {a[-1]}]{"":<8}[{b[0]}, {b[-1]}]{"":<8}{verdict}'
        )

    print('\nseq4 정답 조립 — 출력 EMIT 단계별 (절대 out_rank vs 상대 c_rel):')
    print(f'  {"output":<8}{"out_rank(절대)":<16}{"c_rel(상대)":<12}')
    tr_out = tr['out_rank']
    for dev, r, rel in seq4_correct_trace(seq4):
        mark = '  <- OOD' if r not in tr_out else ''
        print(f'  {dev:<8}{str(r):<16}{str(rel):<12}{mark}')
