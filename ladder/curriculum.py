"""
ladder_curriculum.py — 모티프 변형 과제 생성기 (데이터 분포 설계)

[ 왜 ]
  5차까지의 데이터 처방 서열: ref-only(0.953) > polish GP(0.927) > 날것
  GP(0.893). '같은 과제의 다른 해' 증강은 역효과 — 확률 질량이 관용구에서
  분산된다. 올바른 축은 '같은 모티프의 다른 과제': 래치 체인 관용구의
  학습 예시가 사실상 seq2 하나뿐인데 3단 외삽(seq3)을 기대한 것이 본질
  한계였다. 여기서 2단 래치 체인의 디바이스 배정 변형을 프로그램적으로
  생성해 "체인을 한 단 늘리는 법" 자체를 분포로 가르친다.

[ 설계 ]
  - 변형 축: 입력 배정 순열 (기동/센서가 어느 X인지) × 출력 배정 순열
    (단계가 어느 Y인지) × 디바이스 풀 크기 (미사용 distractor 포함)
  - 관용구는 seq2/seq3 레퍼런스와 동일 (역순 rung + OUT 자기유지 체인)
    — 스타일 일관성이 5차 교훈의 핵심. 변형은 배정만, 구조는 불변
  - 스펙은 benchmark 와 동일 절차: derive_spec → intent assert →
    invariant → 전이 마스킹 (퇴화 가드 포함)
  - 변형 과제는 **라벨 생성 전용** — 탐색 벤치마크 대상이 아님

[ 실행 ]
  python ladder_curriculum.py        # 변형 12개 생성 + 분해 라운드트립 검증
  python ladder_curriculum.py 20     # 개수 지정
"""

import random
import sys

from ladder.benchmark import GenCfg, Task, derive_spec, mask_transition_scans
from ladder.search import evaluate
from ladder.sim import And, Coil, Contact, Or, Program, Rung, Timer


def make_chain_reference(K: int, xs: list, ys: list) -> Program:
    """K단 래치 체인 레퍼런스 — seq2/seq3 와 동일 관용구.

    xs = [기동, 센서1, ..., 센서K] (K+1개), ys = [단계1, ..., 단계K].
    rung 은 역순 (마지막 단계 먼저) — 같은 스캔에 센서가 이전 단계를
    죽이기 전에 다음 단계가 래치되도록 (seq2 주석과 동일한 이유).
    """
    rungs = []
    for k in range(K, 1, -1):
        rungs.append(
            Rung(
                Coil(ys[k - 1]),
                And(
                    [
                        Or(
                            [
                                And([Contact(xs[k - 1]), Contact(ys[k - 2])]),
                                Contact(ys[k - 1]),
                            ]
                        ),
                        Contact(xs[k], 'NC'),
                    ]
                ),
            )
        )
    rungs.append(
        Rung(
            Coil(ys[0]),
            And(
                [
                    Or([Contact(xs[0]), Contact(ys[0])]),
                    Contact(xs[1], 'NC'),
                ]
            ),
        )
    )
    return Program(rungs)


def make_chain_task(
    K: int, xs: list, ys: list, x_pool: list, y_pool: list, name: str
) -> Task:
    """배정 1개 → 검증된 Task. 배정이 스펙 가드에 걸리면 AssertionError."""
    ref = make_chain_reference(K, xs, ys)

    # 사이클: seq3 와 동일 리듬 (기동 2스캔 + 해제 dwell 3스캔, 센서도 동일)
    cycle = [{x: 0 for x in x_pool}]
    cycle += [{xs[0]: 1}, {}, {xs[0]: 0}, {}, {}]
    for k in range(1, K + 1):
        cycle += [{xs[k]: 1}, {}, {xs[k]: 0}, {}, {}]
    idle = [{x: 0 for x in x_pool}, {}, {}, {}]
    no_start = [{x: 0 for x in x_pool}]
    for k in range(1, K + 1):
        no_start += [{xs[k]: 1}, {}, {xs[k]: 0}, {}]

    spec = derive_spec(
        x_pool, y_pool, ['M0'], ref, [cycle, idle, no_start, cycle + cycle[1:]]
    )
    # intent: 각 단계가 사이클에서 실제로 점등 + 기동 없이는 전부 침묵
    a = spec.scenarios[0].expected
    for y in ys:
        assert sum(w[y] for w in a) >= 2, f'{name}: {y} 미발화'
    assert all(
        all(w[y] == 0 for y in y_pool) for w in spec.scenarios[2].expected
    ), f'{name}: 기동 없이 동작'
    chain = list(ys)
    spec.invariants = [lambda out, chain=chain: sum(out[y] for y in chain) <= 1]
    spec = mask_transition_scans(spec)
    acc, viol = evaluate(ref, spec)
    assert acc >= 1.0 and viol == 0, f'{name}: ref 미달 acc={acc} viol={viol}'
    return Task(name, f'{K}단 체인 변형 {xs}->{ys}', spec, ref, GenCfg(), {})


def make_chain_curriculum(
    n_variants: int = 12, K: int = 2, seed: int = 0
) -> list:
    """배정 순열 샘플링으로 변형 n개 생성 (가드 실패 배정은 건너뜀).

    canonical 배정 (X0..XK → Y0..Y{K-1}) 은 항상 제외 — K=2 면 seq2 와
    중복이고, K=3 이면 holdout 대상인 seq3 그 자체라 시험이 오염된다.
    """
    rng = random.Random(seed)
    canonical = (
        tuple(f'X{i}' for i in range(K + 1)),
        tuple(f'Y{i}' for i in range(K)),
    )
    tasks, seen, attempts = [], set(), 0
    while len(tasks) < n_variants:
        attempts += 1
        assert attempts < n_variants * 50, '유효 배정 고갈 — 변형 축 점검 필요'
        n_in = rng.choice(range(K + 1, 6))  # K+1 ~ 5 (featurizer idx/5 한계)
        # K≤3 은 기존 분포 보존 (K~3), K=4 는 출력 4개 고정 (Y0~Y3)
        n_out = rng.choice(range(K, max(K + 1, 4)))
        x_pool = [f'X{i}' for i in range(n_in)]
        y_pool = [f'Y{i}' for i in range(n_out)]
        xs = rng.sample(x_pool, K + 1)
        ys = rng.sample(y_pool, K)
        if (tuple(xs), tuple(ys)) == canonical:
            continue  # holdout 오염 방지
        key = (tuple(xs), tuple(ys), n_in, n_out)
        if key in seen:
            continue
        seen.add(key)
        try:
            tasks.append(
                make_chain_task(
                    K, xs, ys, x_pool, y_pool, f'chain{K}_v{len(tasks)}'
                )
            )
        except AssertionError:
            continue  # 가드에 걸린 배정은 폐기 (퇴화 스펙 등)
    return tasks


# ---------- 타이머 체인 (지연 핸드오프 — 운영표준 step chain 풍) ----------
#
# 래치 체인과 질적으로 다른 관용구 2요소:
#   ① on-delay 전진 — 센서가 preset 스캔 유지돼야 다음 단계 진입
#     (TON(p, 센서·이전단계), 채터링 필터)
#   ② 핸드오프 클리어 — 이전 단계는 센서가 아니라 **다음 단계 출력**이
#     끈다. 센서 즉시 클리어면 타이머가 여물기 전에 게이트(이전단계)가
#     죽어 전진 불가 — 지연이 강제하는 구조적 제약. 역순 rung 덕에
#     같은 스캔 핸드오프 (다음 단계 점등 → 같은 스캔 이전 단계 소등,
#     invariant ≤1 무위반)
#
# 레시피(역할 featurizer + 변형 커리큘럼)가 래치 전용인지 모티프
# 일반인지 가르는 시험용.

# preset 3 이어야 마스킹 후에도 래치와 구별 가능 — 래치는 전이 스캔(t,
# 마스킹됨) 점등이라 관측 지연 1, preset 2 타이머는 t+1 점등이라 역시
# 관측 지연 1 로 동률. preset 3 → t+2 점등 = 관측 지연 2 로 분리.
TIMER_PRESET = 3


def make_timer_chain_reference(K: int, xs: list, ys: list) -> Program:
    """K단 지연 핸드오프 체인.

    stage 1     : ys[0] = (xs[0] + ys[0]) · NOT ys[1]
    stage 2..K-1: ys[k] = (TON(p, xs[k]·ys[k-1]) + ys[k]) · NOT ys[k+1]
    stage K     : 마지막만 최종 센서 xs[K] 가 클리어
    """
    rungs = []
    for k in range(K, 1, -1):
        advance = Timer(
            f'TC{k}',
            TIMER_PRESET,
            And([Contact(xs[k - 1]), Contact(ys[k - 2])]),
        )
        clear = Contact(xs[K], 'NC') if k == K else Contact(ys[k], 'NC')
        rungs.append(
            Rung(
                Coil(ys[k - 1]),
                And([Or([advance, Contact(ys[k - 1])]), clear]),
            )
        )
    rungs.append(
        Rung(
            Coil(ys[0]),
            And(
                [
                    Or([Contact(xs[0]), Contact(ys[0])]),
                    Contact(ys[1], 'NC') if K >= 2 else Contact(xs[1], 'NC'),
                ]
            ),
        )
    )
    return Program(rungs)


def make_timer_chain_task(
    K: int, xs: list, ys: list, x_pool: list, y_pool: list, name: str
) -> Task:
    ref = make_timer_chain_reference(K, xs, ys)

    # 센서를 preset+2 스캔 유지 (타이머 여물 시간) 후 해제 + dwell
    cycle = [{x: 0 for x in x_pool}]
    cycle += [{xs[0]: 1}, {}, {xs[0]: 0}, {}, {}]
    for k in range(1, K + 1):
        cycle += [{xs[k]: 1}, {}, {}, {}, {xs[k]: 0}, {}, {}]
    idle = [{x: 0 for x in x_pool}, {}, {}, {}]
    no_start = [{x: 0 for x in x_pool}]
    for k in range(1, K + 1):
        no_start += [{xs[k]: 1}, {}, {}, {xs[k]: 0}, {}]

    spec = derive_spec(
        x_pool, y_pool, ['M0'], ref, [cycle, idle, no_start, cycle + cycle[1:]]
    )
    a = spec.scenarios[0].expected
    for y in ys:
        assert sum(w[y] for w in a) >= 2, f'{name}: {y} 미발화'
    assert all(
        all(w[y] == 0 for y in y_pool) for w in spec.scenarios[2].expected
    ), f'{name}: 기동 없이 동작'
    chain = list(ys)
    spec.invariants = [lambda out, chain=chain: sum(out[y] for y in chain) <= 1]
    spec = mask_transition_scans(spec)
    acc, viol = evaluate(ref, spec)
    assert acc >= 1.0 and viol == 0, f'{name}: ref 미달 acc={acc} viol={viol}'
    return Task(
        name, f'{K}단 타이머체인 변형 {xs}->{ys}', spec, ref, GenCfg(), {}
    )


def make_timer_chain_curriculum(
    n_variants: int = 8, K: int = 3, seed: int = 0
) -> list:
    """배정 순열 변형 (canonical 제외 — K=3 canonical 은 held-out tchain3)"""
    rng = random.Random(seed)
    canonical = (
        tuple(f'X{i}' for i in range(K + 1)),
        tuple(f'Y{i}' for i in range(K)),
    )
    tasks, seen, attempts = [], set(), 0
    while len(tasks) < n_variants:
        attempts += 1
        assert attempts < n_variants * 50, '유효 배정 고갈'
        n_in = rng.choice(range(K + 1, 6))
        n_out = rng.choice(range(K, max(K + 1, 4)))
        x_pool = [f'X{i}' for i in range(n_in)]
        y_pool = [f'Y{i}' for i in range(n_out)]
        xs = rng.sample(x_pool, K + 1)
        ys = rng.sample(y_pool, K)
        if (tuple(xs), tuple(ys)) == canonical:
            continue
        key = (tuple(xs), tuple(ys), n_in, n_out)
        if key in seen:
            continue
        seen.add(key)
        try:
            tasks.append(
                make_timer_chain_task(
                    K, xs, ys, x_pool, y_pool, f'tchain{K}_v{len(tasks)}'
                )
            )
        except AssertionError:
            continue
    return tasks


# ---------- 자가 점검 ----------

if __name__ == '__main__':
    from ladder.decompose import decompose_with_states, verify_roundtrip

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    tasks = make_chain_curriculum(n)
    total = 0
    for t in tasks:
        ok = verify_roundtrip(t.reference, t.spec)
        pairs = decompose_with_states(t.reference, t.spec)
        total += len(pairs)
        mark = 'OK ' if ok else 'FAIL'
        print(f'[{mark}] {t.name:<12} {t.desc}  →  {len(pairs)} 라벨')
        assert ok, f'{t.name}: 라운드트립 불일치'
    print('-' * 60)
    print(f'변형 {len(tasks)}개에서 라벨 {total}개 — 라운드트립 전부 통과')
