"""
exit_loop.py — Expert Iteration 다회전 자기개선 루프 (범용 타깃)

[ 왜 ]
  1회전(seq3 seed2 1,492→96)은 "최악 분산 축소"만 봤다. 다회전 + 범용
  타깃으로 일반화 — 환류가 라운드마다 발견 비용/분산을 더 줄이고 수렴하나가
  진짜 질문. 이게 좁은 의미 RL(자기개선 루프) 진입 (value network 는 다음).

[ 순수성 규칙 ]
  타깃의 사람 레퍼런스 라벨은 학습에서 제외. 환류되는 것은 탐색이 스스로
  발견한 검증 해(acc==1.0 & viol==0 & 이중코일 0)뿐 — polish(단순화→스펙축소)
  후 분해. 라운드 간 중복 해는 dedup(program_str). 사람 개입 0.

[ 실행 ]
  uv run experiments/exit_loop.py                       # seq3, 4 라운드
  uv run experiments/exit_loop.py --target seq4 --rounds 5
"""

import argparse

from ladder.benchmark import make_tasks
from ladder.curriculum import make_chain_curriculum
from ladder.decompose import decompose_with_states
from ladder.parallel import parallel_search
from ladder.policy import (
    build_samples,
    candidate_actions,
    extract_weights,
    featurize_pairs,
    train,
)
from ladder.search import evaluate, find_coil_conflicts, program_str
from ladder.simplify import polish_program

SEEDS = (0, 1, 2)


def resolve_target(name):
    """타깃 이름 → (target_task, train_tasks).

    train = 타깃 제외 ref + 길이 맞는 변형 커리큘럼 (타깃이 발견 가능한 레짐 —
    8차/k4 에서 확인). 타깃의 canonical 배정은 커리큘럼 생성기가 자동 제외."""
    refs = make_tasks()
    if name == 'seq4':
        from k4_probe import make_seq4

        target = make_seq4()
        ks = (2, 3, 4)
    else:
        target = next(t for t in refs if t.name == name)
        L = int(name[-1]) if name[-1:].isdigit() else 3
        ks = tuple(range(2, L + 1))
    train = [t for t in refs if t.name != name]
    for k in ks:
        train += make_chain_curriculum(8, K=k)
    return target, train


def measure(target, w, budget):
    """puct+net × SEEDS 병렬 → (발견비용 리스트, 발견 Program 리스트)"""
    jobs = [(target.spec, target.mcts_kwargs, budget, s, w, True) for s in SEEDS]
    res = parallel_search(jobs)
    costs = [found_at for found_at, _, _ in res]
    progs = [p for found_at, _, p in res if found_at and p is not None]
    return costs, progs


def expert_labels(progs, spec, seen):
    """발견 해 → 검증 게이트 → polish → 분해 라벨 (seen 으로 라운드 간 dedup)"""
    samples = []
    for prog in progs:
        acc, viol = evaluate(prog, spec)
        if not (acc >= 1.0 and viol == 0 and not find_coil_conflicts(prog)):
            continue
        p = polish_program(prog, spec)
        key = program_str(p)
        if key in seen:
            continue
        seen.add(key)
        for state, act in decompose_with_states(p, spec):
            cands = candidate_actions(state, act)
            samples.append((featurize_pairs(state, cands), cands.index(act)))
    return samples


def fmt(costs):
    return ' | '.join(f'{c:,}' if c else '실패' for c in costs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='seq3')
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--budget', type=int, default=200_000)
    args = ap.parse_args()

    target, base_tasks = resolve_target(args.target)
    base_samples, _ = build_samples(base_tasks)
    print(
        f'타깃 {args.target} (held-out, 사람 라벨 제외) | base {len(base_samples)} '
        f'라벨 | {args.rounds} 라운드 × seed{SEEDS} (예산 {args.budget:,})\n'
    )

    seen, extra, history = set(), [], []
    for r in range(args.rounds):
        model, _ = train(base_samples + extra)
        w = extract_weights(model)
        costs, progs = measure(target, w, args.budget)
        new = expert_labels(progs, target.spec, seen)
        extra += new
        history.append(costs)
        print(f'round {r}: [{fmt(costs)}]  +{len(new)} 라벨 (누적 {len(extra)})')
        if not new and r > 0:  # 환류 고갈 = fixpoint, 더 돌려도 동일 데이터 재학습
            print('환류 고갈 — early stop')
            break

    print(f'\n{"round":>6} | ' + ' | '.join(f'seed{s}' for s in SEEDS))
    print('-' * 34)
    for r, costs in enumerate(history):
        print(f'{r:>6} | {fmt(costs)}')


if __name__ == '__main__':
    main()
