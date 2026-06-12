"""
ladder_sim.py — 래더 IR v1: 순차 로직 + 다중 스캔 시뮬레이션

[ v0 대비 추가된 것 ]

  출력 노드 확장:
    Coil(device)             보통 코일. 매 스캔 logic 결과를 그대로 기록 (OUT)
    Coil(device, op="SET")   logic이 True인 순간 1로 래치 (SET)
    Coil(device, op="RST")   logic이 True인 순간 0으로 리셋 (RST)

  새 논리 노드:
    Timer(name, preset)      온딜레이 타이머(TON).
                             입력이 True로 유지된 스캔 수가 preset에 도달하면 True.
                             입력이 False로 떨어지면 카운트 리셋.
                             (실제 PLC의 T0 K30 같은 것. 여기선 시간단위=스캔)
    Pulse(name, input)       상승엣지 검출(PLS). 입력이 False->True로 바뀐
                             그 스캔에만 True. (원샷/플리커 회로의 재료)

  시뮬레이션:
    simulate(prog, input_trace, watch)
      input_trace: 스캔별 입력 딕셔너리 목록 [{X0:1}, {X0:0}, ...]
                   (해당 스캔에 명시된 키만 갱신, 나머진 유지 → 실제 입력처럼)
      watch:       기록할 디바이스 목록
      반환:        스캔별 watch 값 기록 (trace)

  ** 보상 함수는 이렇게 됨:
     스펙 = (input_trace, 기대 output_trace)
     보상 = simulate 결과와 기대 trace의 일치율
"""

from dataclasses import dataclass, field
from typing import Dict, List, Union

# ---------- IR 노드 ----------


@dataclass
class Contact:
    device: str
    mode: str = "NO"  # "NO"=A접점, "NC"=B접점


@dataclass
class Timer:
    name: str  # 타이머 이름 ("T0" 등) — 상태가 메모리에 저장됨
    preset: int  # 도달해야 하는 스캔 수
    input: "Logic"  # 타이머를 구동하는 논리


@dataclass
class Pulse:
    name: str  # 상태 키 ("P0" 등) — 직전 스캔의 입력값 저장에 사용
    input: "Logic"  # 엣지를 검출할 논리


@dataclass
class And:
    args: List["Logic"]


@dataclass
class Or:
    args: List["Logic"]


Logic = Union[Contact, And, Or, Timer, Pulse]


@dataclass
class Coil:
    device: str
    op: str = "OUT"  # "OUT" | "SET" | "RST"


@dataclass
class Rung:
    coil: Coil
    logic: Logic


@dataclass
class Program:
    rungs: List[Rung] = field(default_factory=list)


# ---------- 메모리 (비트 + 타이머 상태) ----------


class Memory:
    def __init__(self):
        self.bits: Dict[str, bool] = {}
        self.timer_acc: Dict[str, int] = {}  # 타이머 누적 스캔 수
        self.pulse_prev: Dict[str, bool] = {}  # PLS 직전 스캔 입력값

    def get(self, dev: str) -> bool:
        return self.bits.get(dev, False)

    def set(self, dev: str, val: bool):
        self.bits[dev] = bool(val)


# ---------- 평가기 ----------


def eval_logic(node: Logic, mem: Memory) -> bool:
    if isinstance(node, Contact):
        v = mem.get(node.device)
        return v if node.mode == "NO" else not v
    if isinstance(node, And):
        return all(eval_logic(a, mem) for a in node.args)
    if isinstance(node, Or):
        return any(eval_logic(a, mem) for a in node.args)
    if isinstance(node, Timer):
        # 온딜레이: 입력 True면 누적+1, False면 리셋
        if eval_logic(node.input, mem):
            acc = mem.timer_acc.get(node.name, 0) + 1
            mem.timer_acc[node.name] = min(acc, node.preset)  # 포화
        else:
            mem.timer_acc[node.name] = 0
        return mem.timer_acc[node.name] >= node.preset
    if isinstance(node, Pulse):
        cur = eval_logic(node.input, mem)
        prev = mem.pulse_prev.get(node.name, False)
        mem.pulse_prev[node.name] = cur
        return cur and not prev
    raise TypeError(f"알 수 없는 논리 노드: {node}")


def scan_once(prog: Program, mem: Memory):
    for rung in prog.rungs:
        result = eval_logic(rung.logic, mem)
        if rung.coil.op == "OUT":
            mem.set(rung.coil.device, result)
        elif rung.coil.op == "SET":
            if result:
                mem.set(rung.coil.device, True)
        elif rung.coil.op == "RST":
            if result:
                mem.set(rung.coil.device, False)


# ---------- 시뮬레이션 = scan_once를 시간축으로 반복 ----------


def simulate(
    prog: Program, input_trace: List[Dict[str, int]], watch: List[str]
) -> List[Dict[str, int]]:
    """
    input_trace[t] : 스캔 t에서 갱신할 입력 (명시 안 한 입력은 이전 값 유지)
    watch          : 매 스캔 끝에 기록할 디바이스
    반환           : 스캔별 {디바이스: 값} 목록
    """
    mem = Memory()
    trace = []
    for t, inputs in enumerate(input_trace):
        for dev, val in inputs.items():
            mem.set(dev, val)
        scan_once(prog, mem)
        trace.append({d: int(mem.get(d)) for d in watch})
    return trace


def print_trace(input_trace, trace, in_devs, out_devs):
    """trace를 타임차트처럼 출력"""
    header = (
        "scan | "
        + " ".join(f"{d:>3}" for d in in_devs)
        + " | "
        + " ".join(f"{d:>3}" for d in out_devs)
    )
    print(header)
    print("-" * len(header))
    cur_in = {d: 0 for d in in_devs}
    for t, (inp, out) in enumerate(zip(input_trace, trace)):
        cur_in.update(inp)
        ins = " ".join(f"{cur_in.get(d, 0):>3}" for d in in_devs)
        outs = " ".join(f"{out.get(d, 0):>3}" for d in out_devs)
        print(f"{t:>4} | {ins} | {outs}")


# ---------- 데모 ----------

if __name__ == "__main__":
    # ============================================================
    # 데모 1: 자기유지 (전형적인 기동/정지 회로)
    #   rung: ( X0(기동,A접점) OR M0(자기유지) ) AND X1(정지,B접점) -> M0
    #         M0 -> Y0
    # ============================================================
    self_hold = Program(
        [
            Rung(
                Coil("M0"),
                And(
                    [
                        Or([Contact("X0", "NO"), Contact("M0", "NO")]),
                        Contact("X1", "NC"),
                    ]
                ),
            ),
            Rung(Coil("Y0"), Contact("M0", "NO")),
        ]
    )

    # 입력 시나리오: 기동버튼 1스캔만 누름 -> 유지 -> 정지버튼 누름
    inputs1 = [
        {"X0": 0, "X1": 0},  # 0: 대기
        {"X0": 1},  # 1: 기동 버튼 ON (한 스캔만)
        {"X0": 0},  # 2: 버튼 뗌    <- 그래도 유지돼야 함
        {},  # 3:
        {},  # 4:
        {"X1": 1},  # 5: 정지 버튼 ON
        {"X1": 0},  # 6: 정지 버튼 뗌 <- 꺼진 상태 유지돼야 함
        {},  # 7:
    ]
    print("=" * 50)
    print("데모 1: 자기유지 (X0=기동, X1=정지, Y0=출력)")
    print("=" * 50)
    trace1 = simulate(self_hold, inputs1, ["M0", "Y0"])
    print_trace(inputs1, trace1, ["X0", "X1"], ["M0", "Y0"])
    print("→ 버튼(X0)을 뗐는데도 Y0가 유지되다가, 정지(X1)에 꺼짐")
    print("→ 진리표로는 절대 표현 못 하는 동작. 이게 '순차'다.")
    print()

    # ============================================================
    # 데모 2: 타이머 (X0 ON 후 3스캔 지나면 Y0 ON)
    #   rung: TON(T0, preset=3, input=X0) -> Y0
    # ============================================================
    timer_prog = Program(
        [
            Rung(Coil("Y0"), Timer("T0", 3, Contact("X0", "NO"))),
        ]
    )

    inputs2 = [
        {"X0": 0},  # 0
        {"X0": 1},  # 1: 입력 ON  (누적 1)
        {},  # 2:          (누적 2)
        {},  # 3:          (누적 3 -> Y0 ON)
        {},  # 4:          (유지)
        {"X0": 0},  # 5: 입력 OFF (리셋 -> Y0 OFF)
        {"X0": 1},  # 6: 다시 ON  (누적 1부터 다시)
        {},  # 7:          (누적 2 — 아직 OFF)
    ]
    print("=" * 50)
    print("데모 2: 온딜레이 타이머 (preset=3스캔)")
    print("=" * 50)
    trace2 = simulate(timer_prog, inputs2, ["Y0"])
    print_trace(inputs2, trace2, ["X0"], ["Y0"])
    print("→ ON 유지 3스캔째에 켜지고, 입력 끊기면 즉시 리셋")
