"""test_parallel.py — parallel_search 정합 + 가속 확인 (작은 예산, 수십 초)

병렬 결과 == 순차 결과(시드 고정) + 벽시계 가속을 한 번에 본다.
"""

import time

from ladder.benchmark import make_tasks
from ladder.curriculum import make_chain_curriculum
from ladder.mcts import BuildState, mcts_search
from ladder.parallel import parallel_search
from ladder.policy import (
    build_samples,
    extract_weights,
    make_net_rollout,
    make_prior_fn,
    train,
)

if __name__ == '__main__':
    tasks = (
        make_tasks()
        + make_chain_curriculum(8, K=2)
        + make_chain_curriculum(8, K=3)
    )
    samples, _ = build_samples(tasks)
    model, _ = train(samples, epochs=200)
    w = extract_weights(model)
    seq3 = next(t for t in make_tasks() if t.name == 'seq3')

    BUDGET = 15000
    jobs = [
        (seq3.spec, seq3.mcts_kwargs, BUDGET, sd, w, up)
        for up in (False, True)
        for sd in (0, 1, 2)
    ]

    t0 = time.perf_counter()
    par = parallel_search(jobs)
    tp = time.perf_counter() - t0

    t0 = time.perf_counter()
    seq = []
    for spec, kw, bud, sd, ww, up in jobs:
        ro = make_net_rollout(ww)
        pr = make_prior_fn(ww) if up else None
        ev = mcts_search(
            spec,
            bud,
            sd,
            state_factory=lambda spec=spec, kw=kw: BuildState(spec, **kw),
            rollout_policy=ro,
            prior_fn=pr,
        )
        seq.append((ev.found_at, ev.best_acc))
    ts = time.perf_counter() - t0

    match = all(abs(p[1] - s[1]) < 1e-9 for p, s in zip(par, seq))
    print(f'\n병렬 {tp:.1f}s | 순차 {ts:.1f}s | 가속 {ts / tp:.1f}x')
    print(f'정합(병렬==순차): {match}')
