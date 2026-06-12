# 내일 이어서 — 다음 작업 (핸드오프)

> 작성 2026-06-12 밤. 진행 전반은 `README.md` 실험 일지 (이날 1~6차), 이론은 `theory.html`.

## 한 줄 현재 위치

**Phase 2(학습 prior) 1일차 완료 — 전이 실증(0.893 고원→0.953), seq3 발견은
미달. 6차에 걸쳐 데이터 처방 서열 확정, 병목은 데이터가 아니라 featurizer
(스펙의 기능 역할을 못 읽음)로 이동.**

환경: WSL `~/rnd-ml` (uv 프로젝트, torch 2.12.0+cu130, RTX 5070 Ti 정상).
코드 정본도 여기 (repo `docs/` 사본은 stale, 폐기 방침).

## 오늘 확정된 것 (요약)

| 시도 | 처방 | holdout seq3 best |
|---|---|---|
| 3차 | **ref-only (71라벨)** | **0.953** ← 현 최선 |
| 6차 | ref + 체인 변형 12 (251) | 0.940 |
| 5차 | ref + polish GP (189) | 0.927 |
| 4차 | ref + 날것 GP (507) | 0.893 (무학습 고원) |

- 전이는 실재 (mcts_w 고원 0.893 → 0.953, seq3 무노출 prior)
- 같은 과제의 다른 해 증강 = 역효과 / 같은 모티프의 다른 과제 증강 =
  무승부 (배정 모호성 때문에 못 배움)
- PUCT 는 암기 prior 에선 압도(1회), 전이 prior 에선 net-rollout 보다 약함
  (prior 과신 — 디리클레 노이즈 미구현)
- 2차에서 배관 전부 검증됨: featurizer v2(37dim) / PUCT / numpy 추론 어댑터

## ① 다음 본 작업 — featurizer v3: 스펙 기능 역할 (spec-role features)

6차의 배정 모호성이 표적. 시나리오 trace 에서 **디바이스별 역할 힌트**를
기계적으로 도출해 액션 특징에 결합:

- [ ] 입력 d 에 대해: 시나리오에서 처음 눌리는 입력인가 / d 가 1 이 된
      스캔 직후 어느 출력이 점등·소등하나 (출력 인덱스와의 상관 — "d 는
      Y_k 를 점등시키는 입력" / "d 는 Y_k 를 끄는 입력" 힌트)
- [ ] 출력 y 에 대해: 시나리오에서 몇 번째로 점등하나 (체인 순서)
- [ ] 구현 위치: `ladder_policy.py` 의 `Ctx` 에 스펙 단위 사전 계산 추가
      (스펙은 불변이라 과제당 1회 — 캐시), PUSH/EMIT 액션 특징에 결합
- [ ] 검증 ①: 6차 커리큘럼 재학습 → 비표준 배정 변형 top-1 이
      0.73~0.93 → 0.95+ 로 오르는가 (배정 모호성 해소 확인)
- [ ] 검증 ②: holdout seq3 재측정 → 0.953 돌파 + 발견 여부
      (커리큘럼과 결합해야 비로소 변형이 학습 가능 — 6차의 가설)

## ② 차순위 (①이 0.953 을 넘기면)

- [ ] PUCT 루트 디리클레 노이즈 — 전이 prior 과신 완화 (AlphaZero 표준)
- [ ] rollout softmax 온도 스윕 (현재 temp=1.0 고정)
- [ ] 3단 변형 커리큘럼 (`make_chain_curriculum(K=3)`) — seq3 와 같은
      길이의 비표준 배정 학습 (외삽이 아니라 인스턴스 일반화 시험으로 완화)

## ③ 더 뒤 (방향만)

- ExIt 루프 — 탐색이 찾은 검증된 해(acc 1.0)를 라벨로 환류. seq3 첫
  발견이 나오는 순간부터 가동 가능
- `evaluate()` 가속 — net-rollout 실패 시드가 수십 분 (numba/배치화).
  실험 회전율의 직접 병목
- 정책망 구조 2차 — 스펙 인코더 (역할 힌트의 학습형 일반화), MLP→Transformer
- CEGIS / 타임차트 UI (장기, 변동 없음)

## 실행 메모

```bash
cd ~/rnd-ml
uv run ladder_policy.py --holdout seq3                  # ref-only 기준 재현 (0.953)
uv run ladder_policy.py --holdout seq3 --curriculum 12  # 커리큘럼 (6차 재현)
uv run ladder_curriculum.py                             # 변형 생성기 자가 점검
uv run smoke_test.py                                    # 코드 수정 후 필수
```

- 한글 출력 `PYTHONUTF8=1` (직접 python 호출 시 필요)
- MCTS 실패 시드는 200k 소진까지 수 분~수십 분 — 발견 시드만 빨리 끝남
- 모든 시드 고정 — 같은 명령 = 같은 결과
