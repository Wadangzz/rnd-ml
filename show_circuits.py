"""
show_circuits.py — 과제별 레퍼런스 해 + GP 발견 해를 텍스트로 출력

표기:  X0 = A접점(NO)   X0/ = B접점(NC)   * = 직렬(AND)   + = 병렬(OR)
       TON(T0,K3,...) = 온딜레이 타이머   PLS(...) = 상승엣지
       -> Y0 = 보통 코일   -> SET Y0 / RST Y0 = 래치/리셋 코일

실행:
  python show_circuits.py                  # 빠른 과제 둘 (self_hold, interlock)
  python show_circuits.py seq2             # 무거운 과제는 수십 초 걸릴 수 있음
"""

import sys
import textwrap

from ladder_benchmark import make_tasks, run_gp
from ladder_search import evaluate, program_size, program_str
from ladder_simplify import polish_program, simplify_program
from render_ladder import ladder_str

SEED = 1
BUDGET = 200_000


def stage(title: str, prog):
    """단계 제목 + 대수식 + ASCII 래더"""
    print(title)
    print("  " + program_str(prog).strip().replace("\n", "\n  "))
    print(textwrap.indent(ladder_str(prog), "  "))

if __name__ == "__main__":
    tasks = {t.name: t for t in make_tasks()}
    names = sys.argv[1:] or ["self_hold", "interlock"]
    unknown = set(names) - set(tasks)
    assert not unknown, f"없는 과제: {unknown}"

    for name in names:
        t = tasks[name]
        print(f"=== {t.name} — {t.desc} ===")
        stage(f"[레퍼런스 해]  (사람이 작성, 크기 {program_size(t.reference)})", t.reference)
        found, acc, prog = run_gp(t, BUDGET, SEED)
        if found:
            stage(f"[GP 발견 해]  ({found:,}회 평가, 크기 {program_size(prog)})", prog)
        else:
            stage(f"[GP 최고 후보]  (미발견, acc {acc:.3f})", prog)
        simple = simplify_program(prog, t.spec)
        assert evaluate(simple, t.spec) == evaluate(prog, t.spec), \
            f"{name}: 단순화가 동작을 바꿈"
        stage(f"[단순화 후]  (동작 보존, 크기 {program_size(prog)} → "
              f"{program_size(simple)})", simple)
        final = polish_program(prog, t.spec)
        assert evaluate(final, t.spec) == evaluate(prog, t.spec), \
            f"{name}: 스펙 축소가 스펙을 바꿈"
        stage(f"[스펙 축소 후]  (스펙만 보존, 크기 "
              f"{program_size(simple)} → {program_size(final)})", final)
        print()
