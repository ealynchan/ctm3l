# 计算张量的奇异值分解，TSVD
# 导入需要的包
import math
import torch
import numpy as np
import copy as cp
from scipy.linalg import svd


def t_svd(A):
    if not isinstance(A, torch.Tensor):
        A = torch.as_tensor(A)
    if A.is_complex():
        raise ValueError("Input must be real-valued.")

    device = A.device
    dtype = A.dtype
    n3, n1, n2 = A.shape

    # FFT along slice dim (dim=0)
    A_bar = torch.fft.fft(A.to(torch.complex64), dim=0)

    # Pre-allocate full-size tensors
    U_bar = torch.zeros(n3, n1, n1, dtype=torch.complex64, device=device)
    S_bar = torch.zeros(n3, n1, n2, dtype=torch.complex64, device=device)
    Vh_bar = torch.zeros(n3, n2, n2, dtype=torch.complex64, device=device)

    t = math.ceil((n3 + 1) / 2) - 1

    for i in range(t + 1):
        # ✅ CRITICAL: use some=False for full SVD
        U_i, s_i, Vh_i = torch.svd(A_bar[i], some=False)

        # Build full diagonal matrix
        S_i = torch.zeros(n1, n2, dtype=torch.complex64, device=device)
        k = min(n1, n2)
        S_i[:k, :k] = torch.diag(s_i[:k])

        U_bar[i] = U_i  # now (n1, n1) ← OK
        S_bar[i] = S_i  # (n1, n2) ← OK
        Vh_bar[i] = Vh_i  # (n2, n2) ← OK

    # Hermitian symmetry for remaining slices
    for i in range(t + 1, n3):
        j = n3 - i
        U_bar[i] = torch.conj(U_bar[j])
        S_bar[i] = S_bar[j]
        Vh_bar[i] = torch.conj(Vh_bar[j])

    # IFFT + real
    U = torch.fft.ifft(U_bar, dim=0).real.to(dtype)
    S = torch.fft.ifft(S_bar, dim=0).real.to(dtype)
    Vh = torch.fft.ifft(Vh_bar, dim=0).real.to(dtype)

    return U, S, Vh

'''
def t_svd(A):
    # 输入：张量A(维度n1 x n2 x n3)
    # 输出：张量A的T-SVD分量 张量U, S, V
    # 第 1 步：计算张量A的离散傅立叶变换A_bar
    n3, n1, n2 = A.shape
    A_bar = t_operation.ifft_3d(A)
    # 第 2 步：计算A_bar的SVD分量U_bar, S_bar, V_bar的每个正面切片
    U_bar, S_bar, Vh_bar = [], [], []
    t = math.ceil((n3 + 1) / 2) - 1
    for i in range(0, n3):
        if i <= t:
            U_bar_i, S_bar_i, Vh_bar_i = svd(A_bar[i])
            if n1 == n2:
                S_bar_i = np.diag(S_bar_i)
            elif n1 < n2:
                S_bar_i = np.diag(S_bar_i)
                S_bar_i = np.pad(S_bar_i, ((0, 0), (0, n2 - S_bar_i.shape[1])), 'constant')
            else:
                S_bar_i = np.diag(S_bar_i)
                S_bar_i = np.pad(S_bar_i, ((0, n1 - S_bar_i.shape[0]), (0, 0)), 'constant')
        else:
            U_bar_i = np.conj(U_bar[n3 - i])
            S_bar_i = S_bar[n3 - i]
            Vh_bar_i = np.conj(Vh_bar[n3 - i])
        U_bar.append(U_bar_i)
        S_bar.append(S_bar_i)
        Vh_bar.append(Vh_bar_i)
    # 第 3 步：计算U_bar, S_bar, V_bar的逆离散傅立叶变换得到U, S, V, 所得到的V已经是转置之后的结果
    U, S, Vh = t_operation.ifft_3d(U_bar), t_operation.ifft_3d(S_bar), t_operation.ifft_3d(Vh_bar)

    # B = t_operation.t_product(t_operation.t_product(U, S), Vh)
    return U, S, Vh
'''
# # 测试代码
# n1, n2, n3 = 10, 3, 4
# np.random.seed(10086)  # 随机数生成函数，用于生成指定随机数
# A = np.random.rand(n3, n1, n2)
# U, S, Vh = t_svd(A)
# # U[U<0] = 0
# # Vh[Vh<0] = 0
# S[S<0] = 0
# E = cp.deepcopy(S)
# E[E>0] = E[E>0] + 0.6
# B = t_operation.t_product(t_operation.t_product(U, S), Vh)
# C = t_operation.t_product(t_operation.t_product(U, E), Vh)
# print('U:', U, U.shape)
# print('S:', S, S.shape)
# print('E:', E, E.shape)
# print('Vh:', Vh, Vh.shape)
#
# dot = 2
# A = np.around(A, decimals=dot)
# B = np.around(B, decimals=dot)
# C = np.around(C, decimals=dot)
# dist1 = np.linalg.norm(A - B)
# dist2 = np.linalg.norm(A - C)
#
# print('A:', A)
# print('B:', B)
# print('C:', C)
# print(dist1, dist2)

# # 检验张量的SVT, 两种方法结果不一样
# A = np.random.rand(3, 2, 4)
# print('A:', A, A.shape)
# A_svt = t_operation.t_svt(A, tao=1)
# print('A_svt:', A_svt, A_svt.shape)
#
# U, S, Vh = t_svd(A)
# S_bar = t_operation.fft_3d(S)
# S_bar = S_bar-1
# S_bar[S_bar<0]=0
# S_mu = t_operation.ifft_3d(S_bar)
# L = t_operation.t_product(t_operation.t_product(U, S_mu), Vh)
# print('L:', L)