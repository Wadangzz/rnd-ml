# 다음 작업 — 핸드오프

> 갱신 2026-06-16. 진행 전반은 `README.md` 실험 일지 (06-12 정책망 prior 1~8차
> + ExIt + K=4 + tchain, 06-15~16 길이 외삽/그래머/탐색 벽), 이론은 `theory.html`,
> 메모리 스냅샷은 `MEMO.md`.

## 한 줄 현재 위치

**Phase 2 (학습 prior) 결착 + 길이 외삽 트랙 종결.** held-out 발견 3종
(seq3/seq4/tchain3) + 통합 회귀 9/9. **순수 길이 외삽(K≤3→K=4, K=4 데이터 0)은
인코딩 천장 + 탐색 벽으로 미해결 확정** (2026-06-16): c_tgtrel+c_opengap 이
프런티어를 Y2 까지 밀었으나 Y3 만 붕괴, 진단상 인코딩 OOD 아님(`diag_y3_body`)
+ c_uct 스윕 무효(`cpuct_sweep`) = 기만적 부분점수 고원. **남은 길 = value
network/reward shaping or 변형 공급.** featurizer 추가 수술은 근거 없음.

→ **다음은 길이 외삽을 더 파지 말고 ② 실전 모티프(제품 본선) 또는 value
network 로 피벗.** 길이 외삽은 변형 공급(인스턴스 일반화)으로 이미 풀리는
영역이라 실전 ROI 낮음.

환경: WSL `~/rnd-ml` (uv 프로젝트, torch 2.12.0+cu130, RTX 5070 Ti).
git 정본 여기 — GitHub 원격 연결은 미완 (private repo 생성 + push 인증 필요).
성능 지표: `tools/plot_metrics.py` (측정마다 `metrics/history.jsonl` 자동 박제).

## 확정된 레시피 (오늘의 산출)

1. **featurizer v3.1** — 스펙 역할 (시나리오 trace → 기동성 / 입력→출력
   점·소등 rank / 출력 점등 순서) + **지연** (래치~1 vs 타이머~2, 모티프
   판별자). 이름 공간 → 역할 공간 재인덱싱
2. **변형 커리큘럼** — 같은 모티프의 배정 순열 변형 (canonical 상시 제외).
   '같은 과제의 다른 해' 증강은 역효과 (4·5차 실측)
3. **학습/추론 후보 정합** — TON 이 학습 후보에서 상시 경쟁해야 보정됨
   (`_fresh_state` timer_presets)
4. **prior 는 롤아웃에 (핵심 운반체), PUCT 는 좋은 prior 의 증폭기**
5. 검증 규율: 동일 예산 200k / accuracy==1.0 / 3시드 / holdout + canonical 제외

## ① 다음 본 작업 — 통합 회귀 프로브 ("누적 증명")

지금까지 프로브가 과제별 개별 학습이었음 — "건바이건이 아니라 누적"을
데이터로 박제할 차례:

- [ ] **단일 prior** (8과제 ref + 래치 변형 2/3/4단 + 타이머 변형 2/3단,
      전 held-out canonical 제외) 한 번 학습 → **seq3 + seq4 + tchain3
      세 held-out 동시 측정** (puct+net 3시드씩)
- [ ] 전부 발견되면 = 누적 증명 완결. 어느 하나 깨지면 = 모티프 간 간섭
      발견 (그것대로 다음 연구 대상)

## ② 실전 모티프 (제품 본선)

- [ ] actuator 패밀리 변형 커리큘럼 (지령 latch + 센서 해제 + 인터로크 —
      이미 benchmark 과제, 변형 생성기만)
- [ ] 알람 패턴 / `manual/create_*` 모티프 조사 → 커리큘럼화 우선순위
- [ ] 모티프 늘수록 혼합 분산 증가 (tchain 발견 비용 16~37k) — ExIt
      환류로 좁히기 + 분포 비중 설계

## ③ 그 외 (순서 낮음)

- [ ] 길이 외삽 (K≤3→K=4) 미증명 — 상대/순환 rank 인코딩 후보
- [ ] 스케일 캡 — 생성기 입력≤5 풀 캡 + idx/5 정규화, K=5 부터 걸림
- [ ] `evaluate()` 가속 (numba/배치) — 실패 시드 수십 분이 회전율 병목
- [ ] PUCT 루트 디리클레 노이즈
- [ ] 장기: 학습형 스펙 인코더 (손제작 역할 특징의 일반화) → CEGIS →
      타임차트 UI

## 실행 메모

```bash
cd ~/rnd-ml
uv run smoke_test.py                                    # 코드 수정 후 필수 (수 초)
uv run python -m ladder.policy --holdout seq3 --curriculum 16   # 8차 재현
uv run experiments/unified_probe.py                     # 통합 회귀 (누적 증명)
uv run experiments/k4_probe.py --skip-base              # seq4 재현
uv run experiments/tchain_probe.py --skip-base          # tchain3 재현
uv run experiments/exit_loop.py                         # ExIt 1회전 재현
```

- 모든 시드 고정 — 같은 명령 = 같은 결과. 실패 시드는 200k 소진 (수십 분)
- 집에서: clone 후 `uv sync` → torch 는 GPU 에 맞는 인덱스로 `uv add`
  재실행 필요할 수 있음 (uv.lock 은 cu130 기준)
