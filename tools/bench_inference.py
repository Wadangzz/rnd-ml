"""
bench_inference.py — 추론 핫루프: numpy(CPU) vs torch(GPU) 마이크로벤치

[ 왜 ]
  "측정 때 GPU 같이 쓰면 더 빠르지 않나?" 를 말이 아니라 숫자로 가른다.
  policy.py 설계 = 학습 GPU / 추론 numpy(CPU). MCTS 롤아웃은 한 스텝에
  후보 ~10~30개짜리 작은 행렬을 수백만 번 순차로 친다 — GPU 가 이기는지
  지는지는 '배치 크기'가 결정한다.

[ 공정 규칙 ]
  - 실제 가중치 형상(policy_weights.npz) 사용
  - GPU 콜은 정직하게: featurize 결과(numpy CPU) -> GPU 전송 -> forward ->
    argmax 위해 CPU 회수, 전 비용 포함 (= 실사용 1콜)
  - 'GPU resident'(전송 제외)도 따로 재서 전송 vs 런치 오버헤드 분해
  - warmup 후 동일 iters, cuda.synchronize 로 async 보정

[ 실행 ]
  uv run tools/bench_inference.py
"""

import time

import numpy as np
import torch

from ladder.policy import WEIGHTS_PATH, np_forward

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
# 실사용 per-step 배치(8~32) + GPU 가 이기기 시작하는 큰 배치(512~4096)
BATCHES = [8, 16, 32, 256, 512, 4096]


def torch_fwd_from_w(w, device):
    """np_forward 와 동일 연산(matmul + 사이 ReLU, 마지막 ReLU 없음)을 torch 로"""
    mats, i = [], 1
    while f'W{i}' in w:
        W = torch.tensor(w[f'W{i}'], device=device)
        b = torch.tensor(w[f'b{i}'], device=device)
        mats.append((W, b))
        i += 1

    def fwd(x):
        for j, (W, b) in enumerate(mats):
            x = x @ W.T + b
            if j < len(mats) - 1:
                x = torch.relu(x)
        return x

    return fwd


def bench(fn, iters, sync):
    for _ in range(20):  # warmup
        fn()
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # μs/콜


if __name__ == '__main__':
    w = dict(np.load(WEIGHTS_PATH))
    in_dim = w['W1'].shape[1]
    print(f'device={DEV} | 입력차원={in_dim} | 층={sum(1 for k in w if k[0] == "W")}')
    print(f'(net 파라미터 ~{sum(v.size for v in w.values()):,})\n')

    gpu_fwd = torch_fwd_from_w(w, DEV)

    hdr = f'{"batch":>6} | {"numpy CPU":>12} | {"GPU 전송포함":>14} | {"GPU 전송제외":>14} | 승자'
    print(hdr)
    print('-' * len(hdr))
    for n in BATCHES:
        iters = 3000 if n <= 64 else 300
        X = np.random.randn(n, in_dim).astype(np.float32)
        Xg = torch.from_numpy(X).to(DEV)  # GPU 상주본 (전송 제외용)

        t_np = bench(lambda: np_forward(w, X), iters, sync=False)

        def gpu_full():  # 정직한 1콜: numpy -> GPU -> forward -> CPU 회수
            return gpu_fwd(torch.from_numpy(X).to(DEV)).cpu().numpy()

        t_gpu = bench(gpu_full, iters, sync=True)
        t_gpu_res = bench(lambda: gpu_fwd(Xg), iters, sync=True)

        win = 'numpy' if t_np < t_gpu else 'GPU'
        print(
            f'{n:>6} | {t_np:>9.1f}μs | {t_gpu:>11.1f}μs | '
            f'{t_gpu_res:>11.1f}μs | {win}'
        )
