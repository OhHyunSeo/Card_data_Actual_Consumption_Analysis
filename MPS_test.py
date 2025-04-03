#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:22:22 2025

@author: oh
"""

import torch
import time
from tqdm import tqdm

x = torch.rand(5, 3)
print(x)


print(torch.__version__) # 설치된 PyTorch 버전을 확인합니다. 1.12 이상이어야 합니다.

print(torch.backends.mps.is_built()) # MPS 장치를 지원하도록 빌드되어있는지 확인합니다. True여야 합니다.

print(torch.backends.mps.is_available())

# 사용 가능한 디바이스 설정: MPS 사용 가능하면 MPS, 아니면 CPU 사용
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# 행렬 크기와 반복 횟수 설정
matrix_size = 20000    # 2000 x 2000 행렬
outer_iterations = 50  # 외부 반복 횟수
inner_iterations = 10  # 각 외부 반복당 내부 반복 횟수

# 디바이스에 큰 랜덤 행렬 두 개를 생성합니다.
A = torch.randn(matrix_size, matrix_size, device=device)
B = torch.randn(matrix_size, matrix_size, device=device)

print("Starting heavy computation...")

# 시작 시간 기록
start_time = time.time()

# 외부 반복문: 진행 상태바(tqdm)로 감쌉니다.
for i in tqdm(range(outer_iterations), desc="Outer Loop"):
    temp = A
    # 내부 반복문: 같은 행렬 곱셈을 여러 번 수행
    for j in range(inner_iterations):
        temp = torch.mm(temp, B)
    # 각 외부 반복마다 결과값의 총합을 계산해 연산을 강제 실행합니다.
    _ = temp.sum().item()

# 종료 시간 기록 및 전체 소요 시간 계산
end_time = time.time()
elapsed_time = end_time - start_time

print("총 걸린 시간: {:.2f}초".format(elapsed_time))























