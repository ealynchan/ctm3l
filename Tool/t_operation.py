# 计算两个张量之间的乘积，t-product
# 导入需要的包
import math
import numpy as np
import scipy.linalg as la
import copy as cp
import torch
import scipy.fftpack as fft

# # 使用tile函数将单位矩阵沿第一维重复3次，得到一个形状为(3, 3, 3)的三维数组
# I = np.tile(np.eye(4), (3, 1, 1))

# 在 Python 中，np.fft.fft(A, axis=2) 表示对 A 进行快速傅里叶变换，
# 其中 axis=2 表示沿着第三个维度进行变换。
def fft_3d(A):
    return np.fft.fft(A, axis=0)  # np.fft.fft(A, axis=0)
def ifft_3d(A):
    return np.fft.ifftn(A, axes=(0,))  # np.fft.ifftn(A, axes=(0,))

# # 验证对角矩阵的fft不一定是对角矩阵，但是f-对角张量的fft一定还是f-对角张量
# S = np.array([[[1.0277581 , 0.        , 0.        ],
#         [0.        , 0.33039653, 0.        ],
#         [0.        , 0.        , 0.18196222],
#         [0.        , 0.        , 0.        ]],
#        [[0.51365315, 0.        , 0.        ],
#         [0.        , 0.14452752, 0.        ],
#         [0.        , 0.        , 0.04285943],
#         [0.        , 0.        , 0.        ]]])
# print('f-对角张量S:\n', S, S.shape)
# print('fft(S):\n', fft_3d(S))
# B = S[0]
# print('对角矩阵B:\n', B, B.shape)
# print('fft(B):\n', np.fft.fft2(B))

def I_tensor(n, n3):  # 表示单位张量
    I = np.zeros(shape=(n3, n, n))
    I[0] = np.eye(n)
    return I

# c, q, m = 5, 4, 3
# I = I_tensor(q, m)
# print('I:\n', I)
# W_bar = []
# for i in range(0, m):
#     # 生成一个3x3的复数随机矩阵
#     A = np.random.rand(c, q) + 1j * np.random.rand(c, q)
#
#     # 使用QR分解来得到正交矩阵Q
#     Q0, R = np.linalg.qr(A)
#
#     # # 确保Q的列是单位向量
#     # Q = Q / np.sqrt(np.sum(np.abs(Q)**2, axis=0))
#     Q = Q0[:, :q]
#     W_bar.append(Q)
#     print("正交复数矩阵Q:")
#     print(Q, Q.shape)
#     print('ifft(Q):\n', np.fft.ifft2(Q))
#
#     # 验证正交性和单位长度
#     print("验证正交性 Q^T * Q:")
#     print(np.round(np.dot(Q.conj().T, Q), 10), np.dot(Q.conj().T, Q).shape)
#
#     print("验证每个向量的模长:")
#     for i in range(q):
#         print(f"向量 {i + 1} 的模长: {np.linalg.norm(Q[:, i])}")
# W_bar = np.array(W_bar)
# print('W_bar:\n', W_bar, W_bar.shape)
# W = ifft_3d(W_bar)
# print('W:\n', W, W.shape)


# # 生成一个每一列都是单位向量且相互正交的正交实矩阵
# # 生成一个10x3的随机矩阵
# A = np.random.rand(c, q)
# # 使用QR分解来得到正交矩阵Q
# Q, R = np.linalg.qr(A)
# # Q矩阵的前q列就是我们需要的q个单位正交向量
# W = Q[:, :q]
# # 打印结果
# print("R^c中的q个单位正交向量W:")
# print(W, W.shape, '\n检验列正交性W.T @ W=\n', np.round(W.T @ W, 10))
# W_bar = np.fft.fft2(W)
# print('fft(W):\n', W_bar)
# print('W_bar^T * W_bar:\n', np.round(W_bar.conj().T @ W_bar, 10))
# # 计算每一列向量的模长
# column_norms = np.linalg.norm(W, axis=0)
# print("每一列向量的模长为：", column_norms)
# print('I:', np.eye(3), np.round(np.fft.fft2(np.eye(3)), 4))
# print('I:', np.zeros((3, 3)), np.round(np.fft.fft2(np.zeros((3, 3))), 14))
def block_diag_matrix(matrices):
    # 定义一个函数block_diag_matrix，它接受一个列表作为参数，返回一个分块对角矩阵
    # 初始化一个空矩阵
    result = matrices[0]
    # 遍历列表中的每个矩阵
    for i in range(1, len(matrices)):
        # 使用block_diag函数将当前矩阵和结果矩阵拼接起来
        result = la.block_diag(result, matrices[i])
    # 返回结果矩阵
    return result


import torch
import math

def t_product(A, B):
    """
    Compute the t-product of two real-valued tensors A and B.

    Input layout: (n3, n1, n2) — tube fibers along dim=0
        A: (n3, n1, n2)
        B: (n3, n2, l)   ← must have A.shape[2] == B.shape[1]

    Output:
        C: (n3, n1, l)

    Supports CPU, CUDA, and MPS devices.
    """
    # Ensure inputs are real torch.Tensor
    if not isinstance(A, torch.Tensor):
        A = torch.as_tensor(A)
    if not isinstance(B, torch.Tensor):
        B = torch.as_tensor(B)

    if A.is_complex() or B.is_complex():
        raise ValueError("t_product only supports real-valued inputs.")

    device = A.device
    dtype = A.dtype
    n3, n1, n2 = A.shape
    _, n2_B, l = B.shape

    if n2 != n2_B:
        raise ValueError(f"t-product requires A.shape[2] == B.shape[1], got {n2} vs {n2_B}")

    # Step 1: FFT along tube dimension (dim=0)
    A_bar = torch.fft.fft(A.to(torch.complex64), dim=0)  # (n3, n1, n2)
    B_bar = torch.fft.fft(B.to(torch.complex64), dim=0)  # (n3, n2, l)

    # Pre-allocate output in frequency domain
    C_bar = torch.zeros(n3, n1, l, dtype=torch.complex64, device=device)

    # Hermitian symmetry: only compute first t+1 slices
    t = math.ceil((n3 + 1) / 2) - 1  # index of Nyquist frequency

    # Compute first half (including Nyquist)
    for i in range(t + 1):
        C_bar[i] = torch.matmul(A_bar[i], B_bar[i])  # (n1, n2) × (n2, l) → (n1, l)

    # Fill second half using conjugate symmetry
    for i in range(t + 1, n3):
        j = n3 - i
        C_bar[i] = torch.conj(C_bar[j])

    # Step 2: Inverse FFT to time domain
    C = torch.fft.ifft(C_bar, dim=0).real.to(dtype)

    return C

'''
def t_product(A, B):
    # 输入：张量A(维度n1 x n2 x n3)和张量B(维度n2 x l x n3)
    # 输出：张量A和张量B的乘积C=A*B(维度n1 x l x n3)
    # 第 1 步：将张量A和B分别进行离散傅里叶变换，得到复数域上的块对角矩阵A_bar和B_bar
    n3 = A.shape[0]
    # A, B = np.random.rand(n3, 6, 3), np.random.rand(n3, 3, 4)
    A_bar, B_bar = fft_3d(A), fft_3d(B)
    # print('A', A, A.shape)
    # print('A_bar:', A_bar, A_bar.shape)
    # print('ifft_3d(A_bar):', ifft_3d(A_bar), ifft_3d(A_bar).shape)
    # print(A == np.around(ifft_3d(A_bar), decimals=dot))
    # 第 2 步：计算C_bar的每个正面切片，利用公式
    # 当i=1,2,...,t,其中t=大于等于(n3+1)/2的最小整数时，C_bar_i=A_bar_i x B_bar_i;
    # 当i=t+1,...,n3时，C_bar_i=C_bar_{n3-i+2}的共轭
    C_bar = []
    t = math.ceil((n3 + 1) / 2) - 1  # math.ceil(-1.8)表示将-1.8向上取整为-1，math.floor(-1.8)表示将-1.8向下取整为-2。
    for i in range(0, n3):
        if i <= t:
            C_bar_i = np.dot(np.array(A_bar[i]), np.array(B_bar[i]))
        else:
            C_bar_i = np.conj(C_bar[n3 - i])
        C_bar.append(C_bar_i)
    C_bar = np.array(C_bar)
    # 第 3 步：计算张量A和B的乘积C=A*B，通过对C_bar进行逆傅立叶变换
    C = ifft_3d(C_bar)
    return C
'''
# # 测试代码
# A, B = np.random.rand(3, 2, 3), np.random.rand(3, 3, 4)
# C = t_product(A, B)
# print('C:', C, C.shape)

def t_transpose(V):
    """
    Compute t-transpose assuming V has shape (n3, n1, n2),
    where n3 is the number of frontal slices.

    Output: V_t of shape (n3, n2, n1), where:
        V_t[0] = V[0].T
        V_t[j] = V[n3 - j].T  for j = 1, 2, ..., n3-1
    """
    V = torch.as_tensor(V)  # supports numpy input
    if V.is_complex():
        raise ValueError("Input must be real-valued.")

    n3, n1, n2 = V.shape

    if n3 == 1:
        return V.transpose(1, 2).contiguous()  # (1, n1, n2) -> (1, n2, n1)

    # Step 1: Transpose all slices: (n3, n1, n2) -> (n3, n2, n1)
    V_t_all = V.transpose(1, 2)  # vectorized transpose

    # Step 2: Create index mapping: [0, n3-1, n3-2, ..., 1]
    indices = torch.cat([
        torch.tensor([0], device=V.device),
        torch.arange(n3 - 1, 0, -1, device=V.device)  # [n3-1, n3-2, ..., 1]
    ])

    # Step 3: Reorder slices using advanced indexing
    V_t = V_t_all[indices]

    return V_t.contiguous()
'''
def t_transpose(V):
    # 计算张量V的转置
    n3 = V.shape[0]
    V_t = []
    V_temp = cp.deepcopy(V)
    for j in range(0, n3):
        if j == 0:
            V_t_i = np.transpose(V_temp[j])
        else:
            V_t_i = np.transpose(V_temp[n3 - j])
        V_t.append(V_t_i)
    V_t = np.array(V_t)
    return V_t
# print('W^T * W:\n', t_product(t_transpose(W), W))
'''

def t_conjTranspose(V):
    # 计算张量V的转置
    n3 = V.shape[0]
    V_t = []
    V_temp = cp.deepcopy(V)
    for i in range(0, n3):
        V_temp[i] = np.conjugate(V_temp[i])

    for j in range(0, n3):
        if j == 0:
            V_t_i = np.transpose(V_temp[j])
        else:
            V_t_i = np.transpose(V_temp[n3 - j])
        V_t.append(V_t_i)
    V_t = np.array(V_t)
    return V_t

def t_inverse(A):
    # 计算张量A的逆, 要求A的维度为n x n x n3
    n3, n, n = A.shape
    B_bar = []
    I = np.zeros((n3, n, n))
    I[0] = np.eye(n)
    A_bar, I_bar = fft_3d(A), fft_3d(I)
    for i in range(0, n3):
        A_bar_i_inv = np.linalg.inv(A_bar[i])
        B_bar_i = np.dot(np.array(A_bar_i_inv), np.array(I_bar[i]))
        B_bar.append(B_bar_i)
    B_bar = np.array(B_bar)
    B = ifft_3d(B_bar)
    return B

def t_plus(S, tao):
    # 输入：张量S, 参数tao
    # 输出：S-tao的正数部分, (S-tao)_+=max(S-tao, 0)
    S = S - tao
    S[S<0] = 0
    return S

def t_svt(Y, tao):
    # 输入: 张量Y，参数tao>0
    # 输出：张量奇异值阈值D_tao(Y)
    # step 1: 计算张量Y的离散傅立叶变换
    Y_bar = fft_3d(Y)
    # step 2: 对张量Y_bar的每一个正面切片作矩阵SVT
    n3, n1, n2 = Y.shape
    t = math.ceil((n3 + 1) / 2) - 1
    W_bar = []
    for i in range(0, t+1):
        U, S, Vh = np.linalg.svd(Y_bar[i])
        if n1 == n2:
            S = np.diag(S)
        elif n1 < n2:
            S = np.diag(S)
            S = np.pad(S, ((0, 0), (0, n2 - S.shape[1])), 'constant')
        else:
            S = np.diag(S)
            S = np.pad(S, ((0, n1 - S.shape[0]), (0, 0)), 'constant')
        S_tao = t_plus(S, tao)
        W_bar_i = U @ S_tao @ Vh
        W_bar.append(W_bar_i)
    for j in range(t+1, n3):
        W_bar_j = np.conj(W_bar[n3 - j])
        W_bar.append(W_bar_j)
    # 第 3 步：计算张量Y的svt，通过对W_bar进行逆傅立叶变换
    Y_svt = ifft_3d(W_bar)
    return Y_svt

# # 测试代码
# n3, n = 4, 1000
# A = np.random.rand(n3, n, n)
# B = t_inverse(A)
# C = t_product(A, B)
#
# dot = 2
# # 使用around函数对A中的每个元素保留1位小数，得到一个新的三维数组B
# A = np.around(A, decimals=dot)
# B = np.around(B, decimals=dot)
# C = np.around(C, decimals=dot)
# print('A:', A, A.shape)
# print('B:', B, B.shape)
# print('C', C, C.shape)
#
# I = np.tile(np.eye(n), (n3, 1, 1))
# # 使用norm函数计算A和B之间的距离
# dist = np.linalg.norm(I - C)
# print(I == C, dist)

# 检验离散傅里叶变换算子和张量逆算子的可交换性--------------------------------------------------------------------
# B_fft = fft_3d(B)
# B_fft= np.around(B_fft, decimals=dot)
# print('B_fft:', B_fft)
# A_bar_inv = np.linalg.inv(fft_3d(A))
# A_bar_inv= np.around(A_bar_inv, decimals=dot)
# print('A_bar_inv:', A_bar_inv)
# print(B_fft == A_bar_inv)

# # 验证矩阵和它的奇异值分解之间的关系----------------------------------------------------------------------------
# A = np.random.rand(3, 4)
# h, l = A.shape
# U, S, V = np.linalg.svd(A)
#
# # # 将S转换为一个与A形状相同的矩阵
# # S = np.pad(np.diag(S), ((0, A.shape[0] - S.size), (0, 0)))
#
# if h == l:
#     S = np.diag(S)
# elif h < l:
#     S = np.diag(S)
#     S = np.pad(S, ((0, 0), (0, l-S.shape[1])), 'constant')
# else:
#     S = np.diag(S)
#     S = np.pad(S, ((0, h - S.shape[0]), (0, 0)), 'constant')
#
# print('U:', U, U.shape)
# print('S:', S, S.shape)
# print('V:', V, V.shape)
#
# B = U @ S @ V
#
# dot = 2
# # 使用around函数对A中的每个元素保留1位小数，得到一个新的三维数组B
# A = np.around(A, decimals=dot)
# B = np.around(B, decimals=dot)
# print('A:', A, A.shape)
# print('B:', B, B.shape)
# dist = np.linalg.norm(A - B)
# print(A == B, dist)
# # 检查u,sigma,v的乘积是否等于A
# print(np.allclose(B, A)) # 输出True

# from time import time
#
# start_time = time()
# a = np.random.rand(20000, 100)
# U, s, V = np.linalg.svd(a)
# print(time()-start_time)
#
# import mars.tensor as mt
# start_time = time()
# a = mt.random.rand(20000, 100, chunk_size=100)
# U, s, V = mt.linalg.svd(a).execute()
# print(time()-start_time)