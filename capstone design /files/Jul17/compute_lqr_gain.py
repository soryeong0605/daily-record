#!/usr/bin/env python3
import numpy as np
from scipy.linalg import solve_continuous_are

# ===== 물리 파라미터 (URDF 값) =====
M = 1.0       # 카트 질량 (kg)
m = 0.1       # 진자 질량 (kg)
l = 0.25      # 회전축~무게중심 거리 (m)
I = 0.0084    # 진자 관성모멘트 (kg*m^2)
g = 9.81

# ===== 중간 계산값 =====
p = I + m * l**2
D = (M + m) * p - (m * l)**2

# ===== A, B 행렬 (선형화된 상태공간 모델) =====
A = np.array([
    [0, 1, 0, 0],
    [0, 0, -(m**2 * g * l**2) / D, 0],
    [0, 0, 0, 1],
    [0, 0, (M + m) * m * g * l / D, 0]
])

B = np.array([
    [0],
    [p / D],
    [0],
    [-(m * l) / D]
])

# ===== Q, R 가중치 행렬 (튜닝 손잡이) =====
# Q: 상태 오차에 대한 페널티 (대각선 = [x위치, x속도, 각도, 각속도] 순서)
Q = np.diag([1.0, 1.0, 10.0, 1.0])   # 각도 오차를 더 중요하게 (10.0)
R = np.array([[0.1]])                # 힘 사용에 대한 페널티 (작을수록 힘을 아낌없이 씀)

# ===== 리카티 방정식 풀어서 K 계산 =====
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

print("A =\n", A)
print("B =\n", B)
print("K (LQR 게인) =\n", K)