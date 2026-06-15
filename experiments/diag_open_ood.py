"""
diag_open_ood.py — OPEN-선택 OOD 진단: Y3(최상단) 붕괴의 근인 확인

[ 왜 ]
  그래머 재설계(OPEN/CLOSE + 타깃-상대 rank)로 seq4 전이 프런티어가 Y1→Y2
  로 전진했으나 최상단 Y3 만 붕괴 (puct+net 0.971, 미발견). 가설: c_tgtrel
  은 '열린 타깃 대비 접점'을 K-불변으로 고쳤지만, '어느 코일을 먼저 열까'
  (OPEN 선택, top-down 이라 최고 rank 부터)는 아직 절대 인덱스 특징(c_idx/
  c_rel)에 키잉 → K<=3 학습에선 '먼저 여는 코일 = rank 2'만 봐서 seq4 의
  rank 3 OPEN 이 분포 밖.

[ 무엇 ]
  학습 불필요 — OPEN 정답 라벨의 디바이스가 받는 featurize 입력값을
  학습(8 ref + K=2/3 변형, seq4 미공급) vs 시험(seq4)에서 비교:
    - c_idx       : 절대 dev_idx /5            (의심)
    - c_rel       : (idx - next_out) /5        (의심 — next_out 은 top-down 과 역행)
    - role_rank   : role_feats self_rank (B 정규화 rank/(n_lit-1))  (혐의 대상)
    - open_gap    : max(미코일 rank) - rank(coil)  (상대 후보 — top-down 픽 = 0)

[ 판정 ]
  c_idx/c_rel 가 seq4 OPEN 에서 support 밖 값을 받고 open_gap 은 안 받으면
  → "OPEN 선택을 (최상단 미코일 대비) 상대로 바꾼다"가 정당화됨.

[ 실행 ]
  uv run experiments/diag_open_ood.py
"""

from collections import defaultdict

from k4_probe import make_seq4

from ladder.benchmark import make_tasks
from ladder.curriculum import make_chain_curriculum
from ladder.decompose import decompose_with_states
from ladder.policy import Ctx, dev_idx, role_feats, spec_roles

FEATS = ['c_idx', 'c_rel', 'role_rank', 'open_gap']


def collect(tasks):
    """task 목록 → {특징: OPEN 정답 라벨 디바이스가 받는 raw 값 집합}"""
    vals = defaultdict(set)
    for t in tasks:
        spec = t.spec
        roles = spec_roles(spec)
        out_rank = roles['out_rank']
        for state, correct in decompose_with_states(t.reference, spec):
            if correct[0] != 'OPEN':
                continue
            dev = correct[1]
            ctx = Ctx(state)
            idx = dev_idx(spec, dev)
            vals['c_idx'].add(round(idx / 5.0, 3))
            vals['c_rel'].add(round((idx - ctx.next_out) / 5.0, 3))
            vals['role_rank'].add(round(role_feats(dev, roles)[8], 3))
            # 상대 후보: 아직 안 코일된 출력 중 최고 rank 대비 gap (top-down
            # 정답 픽 = 최고 rank 미코일 = gap 0, K 무관)
            uncoiled = [
                y for y in spec.outputs if y not in ctx.coiled and y in out_rank
            ]
            if uncoiled and dev in out_rank:
                mx = max(out_rank[y] for y in uncoiled)
                vals['open_gap'].add(mx - out_rank[dev])
    return vals


def fmt(s):
    return '{' + ', '.join(str(v) for v in sorted(s)) + '}'


if __name__ == '__main__':
    train_tasks = (
        make_tasks()  # 8 ref (seq3=K3 포함, seq4 미포함)
        + make_chain_curriculum(8, K=2)
        + make_chain_curriculum(8, K=3)
    )
    seq4 = make_seq4()

    print('학습 = 8 ref(+seq3) + K=2/3 변형 | 시험 = seq4 (held-out)')
    print('대상 = OPEN 정답 라벨 디바이스의 featurize 입력값\n')
    tr = collect(train_tasks)
    te = collect([seq4])

    print(f'{"feat":<12}{"학습 support":<24}{"seq4":<20}OOD')
    print('-' * 76)
    for f in FEATS:
        ood = te[f] - tr[f]
        print(f'{f:<12}{fmt(tr[f]):<24}{fmt(te[f]):<20}{fmt(ood) if ood else "-"}')

    print(
        '\n판정: c_idx/c_rel 가 OOD 이고 open_gap 이 보간이면 → OPEN 선택을'
        '\n      (최상단 미코일 대비) 상대 특징으로 교체가 정당화됨.'
    )
