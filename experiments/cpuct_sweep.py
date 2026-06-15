"""
cpuct_sweep.py — seq4 길이 외삽 고원: 탐색상수(c_uct) 스윕

[ 왜 ]
  해부(k4_probe --lenext) 결과 puct+net seed2 가 0.954 — Y3 가 ref 의
  `X3*Y2` 대신 `X3+Y2` (AND/OR 한 번 차이) + latch/clear 정확. 딥 노드의
  1-edit 갭이라 탐색을 더 펼치면(c_uct↑) 대안 액션을 시도해 닫힐 수 있다.
  mcts_search 기존 노트: 0.7 갇힘 / 1.0 최적 / 2.0 산만 (쉬운 과제 기준) —
  '못 푸는' 이 과제선 더 높은 c 가 정당화되는지 확인.

[ 측정 ]
  lenext prior(K<=3, B 인코딩) 1회 학습 → seq4 puct+net 을 c_uct 스윕 ×
  3시드 병렬. 발견(1.0) 나오면 c_uct 가 길이 외삽의 마지막 1-edit 을 닫는다.

[ 실행 ]
  uv run experiments/cpuct_sweep.py
"""

from k4_probe import make_seq4

from ladder.benchmark import make_tasks
from ladder.curriculum import make_chain_curriculum
from ladder.parallel import parallel_search
from ladder.policy import build_samples, extract_weights, train
from ladder.search import program_str

BUDGET = 200_000
SEEDS = (0, 1, 2)
CS = [1.0, 1.5, 2.0, 3.0, 4.0]

if __name__ == '__main__':
    seq4 = make_seq4()
    train_tasks = (
        make_tasks()
        + make_chain_curriculum(8, K=2)
        + make_chain_curriculum(8, K=3)
    )
    samples, _ = build_samples(train_tasks)
    print(f'라벨 {len(samples)} | lenext prior(B 인코딩) 학습...')
    model, _ = train(samples)
    w = extract_weights(model)

    jobs, labels = [], []
    for c in CS:
        for s in SEEDS:
            jobs.append((seq4.spec, seq4.mcts_kwargs, BUDGET, s, w, True, c))
            labels.append((c, s))
    print(f'puct+net × c_uct{CS} × seed{SEEDS} — {len(jobs)} jobs 병렬\n')

    results = parallel_search(jobs)

    cell = {}
    for (c, s), (found, acc, prog) in zip(labels, results):
        cell[(c, s)] = f'{found:,}!' if found else f'{acc:.3f}'
        if found:
            print(f'  *** 발견 c_uct={c} seed{s}: {found:,}회 ***')
            print(program_str(prog))

    print(f'\n{"c_uct":>6} | ' + ' | '.join(f'seed{s}' for s in SEEDS))
    print('-' * 34)
    for c in CS:
        print(f'{c:>6} | ' + ' | '.join(f'{cell[(c, s)]:>7}' for s in SEEDS))
