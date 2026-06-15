"""
parallel.py — 독립 MCTS 탐색을 프로세스로 병렬화 (측정 회전율 가속)

[ 왜 ]
  측정은 시드×구성 독립 탐색의 모음이고 전부 CPU 단일스레드 numpy
  (bench_inference.py: 이 배치 크기에선 GPU 가 ~20배 느림). 코어 수만큼
  프로세스로 쪼개면 거의 선형 가속 — GPU 리워크보다 싸고 위험 0.

[ 규칙 ]
  - spec 에 make_tasks 의 lambda 가 박혀 있어 pickle 불가 → spawn 으로
    job 을 워커에 못 보낸다. 대신 **fork + 전역 인덱스**: job(spec 포함)을
    모듈 전역에 두고 워커엔 int 인덱스만 매핑 → fork 상속(copy-on-write)으로
    워커가 spec 을 직렬화 없이 읽는다.
  - 워커는 numpy 추론(np_forward)만 하므로 CUDA 미접촉 → 부모가 학습으로
    CUDA 컨텍스트를 띄운 뒤 fork 해도 자식이 CUDA 를 안 건드려 안전.
  - best_prog(Program)는 그대로 반환 — Program 은 lambda 없어 pickle 안전
    (pickle 불가는 spec 뿐). ExIt 가 발견 해로 라벨을 만들 수 있게.
"""

import multiprocessing as mp

from ladder.mcts import BuildState, mcts_search
from ladder.policy import make_net_rollout, make_prior_fn

_JOBS: list = []  # fork 로 워커에 상속되는 job 목록 (직렬화 우회)


def _worker(i):
    spec, kwargs, budget, seed, w, use_prior, *rest = _JOBS[i]
    c_uct = rest[0] if rest else 1.0  # 옵션 7번째 = 탐색상수 (기본 1.0)
    rollout = make_net_rollout(w)
    prior = make_prior_fn(w) if use_prior else None
    ev = mcts_search(
        spec,
        budget,
        seed,
        c_uct=c_uct,
        state_factory=lambda: BuildState(spec, **kwargs),
        rollout_policy=rollout,
        prior_fn=prior,
    )
    return ev.found_at, ev.best_acc, ev.best_prog


def parallel_search(jobs, procs=None):
    """독립 탐색을 프로세스로 동시 실행.

    jobs : [(spec, mcts_kwargs, budget, seed, w, use_prior[, c_uct]), ...]
    반환  : [(found_at, best_acc, best_prog|None), ...] (입력 순서 보존)
    """
    global _JOBS
    _JOBS = jobs
    if procs is None:
        procs = min(len(jobs), mp.cpu_count())
    ctx = mp.get_context('fork')
    with ctx.Pool(procs) as pool:
        return pool.map(_worker, range(len(jobs)))
