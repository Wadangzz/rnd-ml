"""
render_ladder.py — IR Program 을 ASCII 래더 다이어그램으로 그리기

대수식(`program_str`)은 정확하지만 '사다리' 로 안 보인다. 발견 해를
사람 눈으로 검수하려면 접점/코일 기호로 그려야 한다.

표기:
  --] [--   A접점 (NO)        --]/[--   B접점 (NC)
  -[TON K3]-  온딜레이 타이머   -[PLS]-   상승엣지
  -( )-  코일(OUT)   -(S)-  SET   -(R)-  RST
  직렬(AND)=가로 연결, 병렬(OR)=세로 분기(+ 레일), 디바이스명은 기호 위.

[ 구현: 2D 박스 합성 ]
  각 서브트리를 (lines, mid) 박스로 렌더 — mid 는 전류가 흐르는 가로줄.
  series 는 mid 를 맞춰 가로로 잇고, parallel 은 세로로 쌓아 좌우 레일(+/|)
  로 묶는다. 표준 라이브러리만 사용.
"""

from ladder.sim import And, Contact, Or, Program, Pulse, Rung, Timer


class Box:
    """ASCII 조각 — 모든 줄 길이 동일, mid 는 통전 가로줄 인덱스"""

    __slots__ = ('lines', 'mid')

    def __init__(self, lines: list[str], mid: int):
        self.lines = lines
        self.mid = mid

    @property
    def w(self) -> int:
        return len(self.lines[0]) if self.lines else 0

    @property
    def h(self) -> int:
        return len(self.lines)


# ---------- 리프/단말 박스 ----------


def _labeled(body: str, name: str) -> Box:
    """기호줄 body + 그 위에 디바이스명 (가운데 정렬), mid=아래줄"""
    return Box([name.center(len(body)), body], 1)


def _contact(c: Contact) -> Box:
    sym = '] [' if c.mode == 'NO' else ']/['
    return _labeled(f'-{sym}-', c.device)


def _block(text: str) -> Box:
    body = f'-[{text}]-'
    return Box([' ' * len(body), body], 1)


def _coil_box(coil) -> Box:
    inner = ' ' if coil.op == 'OUT' else coil.op[0]  # S / R
    return _labeled(f'-({inner})-', coil.device)


# ---------- 합성: series(AND) / parallel(OR) ----------


def _series(boxes: list[Box]) -> Box:
    boxes = [b for b in boxes if b is not None]
    if len(boxes) == 1:
        return boxes[0]
    above = max(b.mid for b in boxes)
    below = max(b.h - b.mid for b in boxes)
    H = above + below
    padded = []
    for b in boxes:
        top = above - b.mid
        bot = H - b.h - top
        padded.append([' ' * b.w] * top + list(b.lines) + [' ' * b.w] * bot)
    out = [''.join(p[r] for p in padded) for r in range(H)]
    return Box(out, above)


def _parallel(boxes: list[Box]) -> Box:
    boxes = [b for b in boxes if b is not None]
    if len(boxes) == 1:
        return boxes[0]
    W = max(b.w for b in boxes)
    stacked: list[str] = []
    mids: list[int] = []
    for b in boxes:
        base = len(stacked)
        mids.append(base + b.mid)
        for r, ln in enumerate(b.lines):
            fill = '-' if r == b.mid else ' '
            stacked.append(ln + fill * (W - len(ln)))
    lo, hi = mids[0], mids[-1]
    out = []
    for r, ln in enumerate(stacked):
        if r in mids:
            left = right = '+'
        elif lo < r < hi:
            left = right = '|'
        else:
            left = right = ' '
        out.append(left + ln + right)
    return Box(out, mids[0])


def _render(node) -> Box:
    if isinstance(node, Contact):
        return _contact(node)
    if isinstance(node, And):
        return _series([_render(a) for a in node.args])
    if isinstance(node, Or):
        return _parallel([_render(a) for a in node.args])
    if isinstance(node, Timer):
        return _series([_render(node.input), _block(f'TON K{node.preset}')])
    if isinstance(node, Pulse):
        return _series([_render(node.input), _block('PLS')])
    raise TypeError(f'렌더 불가 노드: {node!r}')


# ---------- rung / program ----------


def _rung_lines(rung: Rung) -> list[str]:
    b = _series([_render(rung.logic), _coil_box(rung.coil)])
    out = []
    for r, ln in enumerate(b.lines):
        if r == b.mid:
            out.append('|-' + ln + '|')  # 좌 전원레일 + 통전, 우 레일
        else:
            out.append('| ' + ln + ' ')
    return out


def ladder_str(prog: Program) -> str:
    """Program → 여러 rung ASCII (rung 사이 빈 줄)"""
    blocks = [_rung_lines(r) for r in prog.rungs]
    return '\n\n'.join('\n'.join(b) for b in blocks)


# ---------- 데모 ----------

if __name__ == '__main__':
    import sys

    from benchmark import make_tasks

    tasks = {t.name: t for t in make_tasks()}
    names = sys.argv[1:] or ['self_hold', 'actuator', 'delayed_off', 'one_shot']
    for name in names:
        t = tasks[name]
        print(f'=== {name} — {t.desc} ===')
        print(ladder_str(t.reference))
        print()
