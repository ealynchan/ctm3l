import numpy as np

def fft_3d(A):
    return np.fft.fft(A, axis=0)  # np.fft.fft(A, axis=0)
def ifft_3d(A):
    return np.fft.ifftn(A, axes=(0,))  # np.fft.ifftn(A, axes=(0,))

import torch
import math

def t_product(A, B):
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

def t_plus(S, tao):
    # 输入：张量S, 参数tao
    # 输出：S-tao的正数部分, (S-tao)_+=max(S-tao, 0)
    S = S - tao
    S[S<0] = 0
    return S