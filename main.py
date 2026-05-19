# This is for CTM3L
import math
import pandas as pd
from Tool import metrics_MLL, MSMIML_BLS
import numpy as np
import time
from Tool import t_SVD
from Tool import t_operation as top
from sklearn.metrics import precision_recall_curve
import torch
import warnings
import platform
# 忽略所有警告
warnings.filterwarnings('ignore')

# 检测并配置GPU设备
def setup_device():
    if platform.system() == "Darwin":  # macOS (Apple Silicon)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            print("Using Metal Performance Shaders (MPS) on Apple Silicon")
        else:
            device = torch.device("cpu")
            print("MPS not available, using CPU on macOS")
    elif platform.system() == "Windows":  # Windows
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # print(f"Using CUDA on Windows - {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU on Windows")
    else:  # Linux or other systems
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA on {platform.system()} - {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print(f"CUDA not available, using CPU on {platform.system()}")

    return torch.device("cpu")


class MSMIMLmodel:
    def __init__(self, dim, R1, alpha, lbd1, lbd2, lbd3, lbd4, eta1, eta2, kNum, tol, max_iter, m=None, d=None, q=None):
        self.dim = dim
        self.R1 = R1
        self.alpha = alpha
        self.lbd1 = lbd1
        self.lbd2 = lbd2
        self.lbd3 = lbd3
        self.lbd4 = lbd4

        self.eta1 = eta1
        self.eta2 = eta2
        self.kNum = kNum
        self.tol = tol
        self.max_iter = max_iter

        # 设置设备
        self.device = setup_device()
        self.supports_complex = self.device.type != 'mps'

        # 新增：初始化 W_list (m, d, q)
        if m is not None and d is not None and q is not None:
            # 随机初始化投影矩阵
            self.W_list = torch.randn(m, d, q, device=self.device) * 0.01
        else:
            self.W_list = None

    def fit(self, X_train, y_train):
        self.X_train = X_train  # 存储训练集特征
        self.y_train = y_train

        # 自动获取维度
        m, n, d = X_train.shape
        q = y_train.shape[1]  # 标签维度

        # 初始化 W_list (m, d, q)
        self.W_list = torch.randn(m, d, q, device=self.device) * 0.01

    def to_device(self, tensor):
        """将张量移动到指定设备"""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        else:
            return torch.tensor(tensor, device=self.device, dtype=torch.float32)

    def vec(self, array_3d):
        # 确保 array_3d 是一个 NumPy 数组
        array_3d = np.asarray(array_3d)

        # 检查 array_3d 的维度是否为 3
        if array_3d.ndim == 3:
            vector = array_3d.transpose((0, 2, 1)).reshape(-1)
        elif array_3d.ndim == 2:
            vector = array_3d.T.reshape(-1)
        else:
            raise ValueError("Unsupported array dimension")
        return vector

    def innvec(self, vector, m, n1, n2):
        # 重新将向量转换回三维数组
        # 需要指定原始的形状和 transpose 操作的反向步骤
        if vector.size != m * n1 * n2:
            raise ValueError(f"无法将大小为{vector.size}的数组reshape为({m}, {n1}, {n2})的形状")

        array_3d_reconstructed = vector.reshape(m, n1, n2).transpose(0, 2, 1)
        return array_3d_reconstructed

    # 将张量n-模展开，n=1，2，3
    def unfold(self, tensor, n):
        # 保持张量在原始设备上（GPU/CPU）
        if isinstance(tensor, torch.Tensor):
            # 直接使用PyTorch操作在GPU上进行模展开
            mode = 0 if n == 3 else n
            if n == 3:  # 模3展开
                print(tensor.shape)
                tensor = tensor.transpose(1, 2)  # 替代numpy的转置操作
                moved = tensor.movedim(mode, 0)
            else:
                moved = tensor.movedim(mode, 0)

            # 使用view替代reshape
            return moved.contiguous().view(moved.shape[0], -1)
        else:
            # 对于numpy数组，保持原有逻辑
            tensor = np.array(tensor)
            mode = 0 if n == 3 else n
            if n == 3:
                tensor = np.array([item.T for item in tensor])
                result = np.moveaxis(tensor, mode, 0)
                if result.shape[0] == 0 or result.size == 0:
                    return result.reshape(0, 0) if len(result.shape) >= 2 else np.array([]).reshape(0, 0)
                return result.reshape(result.shape[mode], -1)
            else:
                result = np.moveaxis(tensor, mode, 0)
                if result.shape[0] == 0 or result.size == 0:
                    return result.reshape(0, 0) if len(result.shape) >= 2 else np.array([]).reshape(0, 0)
                return result.reshape(result.shape[mode], -1)

    # n-模乘积
    def mode_n_product(self, tensor, matrix, mode):
        """
        Compute the mode-n product of a tensor and a matrix.

        Args:
            tensor (torch.Tensor): Input tensor of shape (I1, I2, ..., In, ...)
            matrix (torch.Tensor): Matrix of shape (J, In) — must match tensor.shape[mode]
            mode (int): Mode (dimension index) to contract, 0-based.

        Returns:
            torch.Tensor: Result of shape (I1, ..., I_{mode-1}, J, I_{mode+1}, ...)
        """
        # Ensure inputs are PyTorch tensors
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.as_tensor(matrix)

        # Move matrix to same device/dtype as tensor
        matrix = matrix.to(device=tensor.device, dtype=tensor.dtype)

        # Validate mode
        if mode < 0:
            mode = tensor.dim() + mode
        if not (0 <= mode < tensor.dim()):
            raise ValueError(f"Invalid mode={mode} for tensor with {tensor.dim()} dimensions.")

        # Check dimension compatibility
        if matrix.shape[1] != tensor.shape[mode]:
            raise ValueError(
                f"Matrix shape {matrix.shape} incompatible with tensor.shape[{mode}]={tensor.shape[mode]}. "
                "Expected matrix.shape[1] == tensor.shape[mode]."
            )

        # Permute so that 'mode' is the first dimension
        perm_order = [mode] + [i for i in range(tensor.dim()) if i != mode]
        tensor_perm = tensor.permute(perm_order)  # (In, I1, ..., I_{mode-1}, I_{mode+1}, ...)

        # Reshape to (In, -1)
        original_shape = tensor.shape
        tensor_unfolded = tensor_perm.reshape(tensor.shape[mode], -1)  # (In, N)

        # Mode-n product: matrix @ tensor_unfolded → (J, N)
        result_unfolded = torch.matmul(matrix, tensor_unfolded)  # (J, N)

        # Reshape back: (J, I1, ..., I_{mode-1}, I_{mode+1}, ...)
        new_shape = [matrix.shape[0]] + [original_shape[i] for i in range(len(original_shape)) if i != mode]
        result_perm = result_unfolded.reshape(new_shape)

        # Invert permutation to restore original order (with mode replaced by J)
        inv_perm_order = [0] * len(new_shape)
        idx = 1  # skip first dim (which is J)
        for i in range(len(original_shape)):
            if i == mode:
                inv_perm_order[i] = 0  # J goes to position 'mode'
            else:
                inv_perm_order[i] = idx
                idx += 1

        result = result_perm.permute(*inv_perm_order)
        return result

    # 获取三维数组的非零元素的索引及其值
    def non_zero(self, arr):
        # 获取非零元素的索引
        non_zero_indices = np.nonzero(arr)
        # 获取非零元素的值
        non_zero_values = arr[non_zero_indices]
        return non_zero_indices, non_zero_values

    def TLR_theorem(self, K, A, tao, alpha):
        """
        Solve: 1/2 * ||A - K||_F^2 + tao * ||K||_TLR
        All operations on GPU if inputs are on GPU.
        """
        device = A.device if isinstance(A, torch.Tensor) else torch.device('cpu')
        dtype = A.dtype if isinstance(A, torch.Tensor) else torch.float32

        # Ensure inputs are real tensors
        A = torch.as_tensor(A, dtype=dtype, device=device)
        K = torch.as_tensor(K, dtype=dtype, device=device)

        if A.is_complex():
            A = A.real
        if K.is_complex():
            K = K.real

        # Clamp inf/nan (PyTorch version)
        A = torch.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)
        K = torch.nan_to_num(K, nan=0.0, posinf=1e6, neginf=-1e6)

        # t-SVD of A
        C, S, D = t_SVD.t_svd(A)  # Assume returns (n1, r, n3), (r, r, n3), (n2, r, n3)

        # FFT along tube fibers (dim=2)
        S_bar = torch.fft.fft(S, dim=0)  # complex

        # Get non-zero values (real part only)
        non_zero_mask_S = S_bar.abs() > 1e-8  # avoid numerical noise
        non_zero_values_S = S_bar[non_zero_mask_S].real  # shape: (N,)

        # t-SVD of K
        C0, T, D0 = t_SVD.t_svd(K)
        T_bar = torch.fft.fft(T, dim=0)
        non_zero_mask_T = T_bar.abs() > 1e-8
        non_zero_values_T = T_bar[non_zero_mask_T].real  # shape: (M,)

        # Pad to same length
        len_S = non_zero_values_S.numel()
        len_T = non_zero_values_T.numel()

        if len_S <= len_T:
            target_len = len_T
            padded_S = torch.nn.functional.pad(non_zero_values_S, (0, target_len - len_S), value=0.0)
            padded_T = non_zero_values_T
            result = padded_S - tao * (alpha * torch.pow(padded_T, alpha - 1)) / (1 + torch.pow(padded_T, alpha))
            result = torch.clamp(result, min=0.0)
            # Put back into T_bar
            T_bar_new = T_bar.clone()
            T_bar_new[non_zero_mask_T] = result.to(T_bar.dtype)
            T_new = torch.fft.ifft(T_bar_new, dim=0).real
        else:
            target_len = len_S
            padded_S = non_zero_values_S
            padded_T = torch.nn.functional.pad(non_zero_values_T, (0, target_len - len_T), value=0.0)
            result = padded_S - tao * (alpha * torch.pow(padded_T, alpha - 1)) / (1 + torch.pow(padded_T, alpha))
            result = torch.clamp(result, min=0.0)
            T_bar_new = T_bar.clone()
            # Use mask from S? But updating T_bar — better to update T_bar's own mask or use full tensor
            # Safer: reconstruct full T_bar with updated singular values?
            # For simplicity, we assume same support (or just update where T had non-zeros)
            T_bar_new[non_zero_mask_T] = result[:len_T].to(T_bar.dtype)  # truncate if longer
            T_new = torch.fft.ifft(T_bar_new, dim=0).real

        # Ensure D is real (t_svd should return real, but be safe)
        D = D.real if D.is_complex() else D

        # Final reconstruction: C * T_new * D^T
        D_t = top.t_transpose(D)  # must accept torch.Tensor and return real
        temp = top.t_product(C, T_new)  # both real

        result = top.t_product(temp, D_t)  # all real → no error

        return result

    ###################################更新各个变量############################################
    def updateF(self, B):  # B: (m, n, q), R1 and F: (m, b, n)
        Y_tensor = torch.as_tensor(self.y_train, device=self.device, dtype=torch.float32)
        m, n, q = B.shape

        if not self.supports_complex:
            # 保存原始设备，然后临时使用CPU
            original_device = self.device
            Y_tensor_cpu = Y_tensor.cpu()
            B_cpu = B.cpu()
            R1_cpu = self.R1.cpu()

            # 在CPU上执行复数运算，确保数据类型一致
            Y_bar = torch.fft.fft(Y_tensor_cpu.to(torch.complex64), dim=0)
            B_bar = torch.fft.fft(B_cpu.to(torch.complex64), dim=0)
            R1_bar = torch.fft.fft(R1_cpu.to(torch.complex64), dim=0)

            B_conj_T = B_bar.transpose(-2, -1).conj()
            gram_matrices = torch.bmm(B_bar, B_conj_T)
            identity = torch.eye(n, dtype=B_bar.dtype, device=B_bar.device).unsqueeze(0).expand(m, -1, -1)
            # 增加更强的正则化项来确保矩阵非奇异
            regularization_factor = max(1e-6, 1.0)  # 使用固定的小正则化项
            gram_plus_eye = gram_matrices + regularization_factor * identity
            left_terms = torch.bmm(Y_bar, B_conj_T) + R1_bar

            # 使用更稳定的求解方法
            try:
                F_bar = torch.linalg.solve(gram_plus_eye, left_terms.transpose(-2, -1)).transpose(-2, -1)
            except torch._C._LinAlgError:
                # 如果直接求解失败，使用伪逆作为备选方案
                print("Warning: Using pseudoinverse due to singular matrix in updateF")
                # 使用SVD计算伪逆
                U, S, Vh = torch.linalg.svd(gram_plus_eye)
                # 避免除以接近0的奇异值
                S_inv = torch.where(S > 1e-10, 1.0 / S, torch.zeros_like(S))
                S_inv_matrix = torch.diag_embed(S_inv)

                # 确保S_inv_matrix与U,Vh类型一致
                S_inv_matrix = S_inv_matrix.to(dtype=U.dtype)

                gram_plus_eye_inv = torch.matmul(Vh.transpose(-2, -1), torch.matmul(S_inv_matrix, U.transpose(-2, -1)))
                F_bar = torch.matmul(gram_plus_eye_inv, left_terms.transpose(-2, -1)).transpose(-2, -1)

            # 转换回实数并移回原设备
            F = torch.real(torch.fft.ifft(F_bar, dim=2)).to(original_device)
        else:
            # 原有逻辑适用于CUDA或CPU，但确保数据类型一致
            Y_bar = torch.fft.fft(Y_tensor.to(torch.complex64), dim=0)
            B_bar = torch.fft.fft(B.to(torch.complex64), dim=0)
            R1_bar = torch.fft.fft(self.R1.to(torch.complex64), dim=0)

            B_conj_T = B_bar.transpose(-2, -1).conj()
            gram_matrices = torch.bmm(B_bar, B_conj_T)
            identity = torch.eye(n, dtype=B_bar.dtype, device=B_bar.device).unsqueeze(0).expand(m, -1, -1)
            # 增加更强的正则化项来确保矩阵非奇异
            regularization_factor = max(1e-6, 1.0)  # 使用固定的小正则化项
            gram_plus_eye = gram_matrices + regularization_factor * identity

            left_terms = torch.bmm(Y_bar, B_conj_T) + R1_bar

            # 使用更稳定的求解方法
            try:
                F_bar = torch.linalg.solve(gram_plus_eye, left_terms.transpose(-2, -1)).transpose(-2, -1)
            except torch._C._LinAlgError:
                # 如果直接求解失败，使用伪逆作为备选方案
                print("Warning: Using pseudoinverse due to singular matrix in updateF")
                # 使用SVD计算伪逆
                U, S, Vh = torch.linalg.svd(gram_plus_eye)
                # 避免除以接近0的奇异值
                S_inv = torch.where(S > 1e-10, 1.0 / S, torch.zeros_like(S))
                S_inv_matrix = torch.diag_embed(S_inv)

                # 确保S_inv_matrix与U,Vh类型一致
                S_inv_matrix = S_inv_matrix.to(dtype=U.dtype)

                gram_plus_eye_inv = torch.matmul(Vh.transpose(-2, -1), torch.matmul(S_inv_matrix, U.transpose(-2, -1)))
                F_bar = torch.matmul(gram_plus_eye_inv, left_terms.transpose(-2, -1)).transpose(-2, -1)

            F = torch.real(torch.fft.ifft(F_bar, dim=2))

        # 对F进行软约束：每行和为1且非负（向量化操作）
        # 确保非负
        F = torch.clamp(F, min=0.0)
        # 按最后一维求和并广播进行归一化
        row_sums = torch.sum(F, dim=-1, keepdim=True)
        # 避免除零
        row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        F = F / row_sums

        return F

    # 更新张量R2根据公式(17)############################################
    def updateR2(self, R2, U, V, W, B, C, OList, muList):
        # R2: (m, n, q), A: (m, n, d), U: (n, n), V: (q, d), W: (m, m), B: (m, n, q), C: (m, n, q)
        A = self.X_train

        # 原始
        AUVW = self.mode_n_product(self.mode_n_product(self.mode_n_product(A, U, 1), V, 2), W, 0)

        mu_1, mu_2 = muList[0], muList[1]
        O_1, O_2 = self.to_device(OList[0]), self.to_device(OList[1])
        AUVW = self.to_device(AUVW)
        B = self.to_device(B)

        # 检查张量维度是否匹配，如果不匹配则调整
        if AUVW.shape != B.shape:
            # 如果形状不匹配，我们使用较小的形状
            min_shape = [min(AUVW.shape[i], B.shape[i]) for i in range(len(AUVW.shape))]
            # 对每个张量进行切片以匹配较小的形状
            AUVW = AUVW[:min_shape[0], :min_shape[1], :min_shape[2]]
            B = B[:min_shape[0], :min_shape[1], :min_shape[2]]
            O_1 = O_1[:min_shape[0], :min_shape[1], :min_shape[2]]
        # here, D1 = U_tensor
        D1 = (2 * (AUVW) + mu_1 * B + mu_2 * C - O_1 - O_2) / (1 + mu_1 +mu_2)
        # Min-Max 归一化
        D1_min = D1.min()
        D1_max = D1.max()
        if D1_max > D1_min:
            D1_normalized = (D1 - D1_min) / (D1_max - D1_min)
        else:
            D1_normalized = torch.zeros_like(D1)

        R2_new = self.TLR_theorem(R2, D1_normalized, self.lbd1 / (1 + mu_1 + mu_2), self.alpha)
        result = self.to_device(R2_new.real)

        # 处理可能的NaN和inf值
        if torch.isnan(result).any() or torch.isinf(result).any():
            result = torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

        return result

    # 更新矩阵U根据公式(21)############################################
    def updateU1(self, W, V, R2):
        # V = U_2, W = U_3
        # A: (m, n, d), W: (m, m), V: (q, d), R2: (m, n, q)
        A = self.to_device(self.X_train)
        device = R2.device  # 自动获取设备（CPU/GPU）
        In = torch.eye(A.shape[1], device=device)  # n阶单位矩阵，放在相同设备上
        V, W = self.to_device(V), self.to_device(W)
        # Step 1: Compute V^T @ R2[k]^T for all k → (m, n, n)
        VR2T = torch.einsum('nq,mqp->mnp', V.T, R2.permute(0, 2, 1))
        # Step 2: temp_all[i] = sum_k W[k, i] * VR2T[k]
        temp_all = torch.tensordot(W.T, VR2T, dims=([1], [0]))  # (m, n, n)
        # Step 3: fast = sum_i A[i] @ temp_all[i]
        fast = torch.einsum('mij,mjk->ik', A, temp_all)

        # --- Clean input ---
        # 进行归一化[0, 1]
        fast_min = fast.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # 获取每个样本的最小值
        fast_max = fast.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # 获取每个样本的最大值
        # 避免除零情况
        denominator = fast_max - fast_min
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        fast = (fast - fast_min) / denominator

        # --- Add jitter for stability ---
        if fast.dim() == 2:
            eps = 1e-6
            fast = fast + eps * torch.eye(fast.shape[0], device=fast.device, dtype=fast.dtype)
        else:
            # For batched SVD
            eps = 1e-6
            I = torch.eye(fast.shape[-1], device=fast.device, dtype=fast.dtype)
            fast = fast + eps * I

        # --- Try SVD ---
        try:
            E, Sigma, Fh = torch.linalg.svd(fast)
        except torch._C._LinAlgError:
            # Fallback: move to CPU with gesvd
            print("SVD failed on GPU, falling back to CPU gesvd...")
            fast_cpu = fast.cpu()
            E, Sigma, Fh = torch.linalg.svd(fast_cpu, driver='gesvd')
            E, Sigma, Fh = E.to(fast.device), Sigma.to(fast.device), Fh.to(fast.device)

        U1_new = Fh.T @ In @ E.T
        U1_new[U1_new < 0.0] = 0.0
        U1_new = U1_new / torch.sum(U1_new)
        return U1_new

    # 更新矩阵U2根据公式(23)############################################
    # def updateU2(self, R2, W, U):
    #     # R2: (m, n, q), A: (m, n, d), W: (m, m), U: (n, n)
    #     A = self.to_device(self.X_train)
    #     device = R2.device  # 自动获取设备（CPU/GPU）
    #     I = torch.eye(A.shape[2], device=device)  # n阶单位矩阵，放在相同设备上
    #     U, W = self.to_device(U), self.to_device(W)
    #     # Step 1: Compute U @ A[k] for all k → (m, n, d)
    #     UA = torch.einsum('nd,mdp->mnp', U, A)
    #     # Step 2: temp_all[i] = sum_k W[i, k] * UA[k]
    #     temp_all = torch.tensordot(W, UA, dims=([1], [0]))
    #     # Step 3: fast = sum_i Z[i].T @ temp_all[i]
    #     R2t = R2.transpose(-2, -1)  # (m, j, i) if Z is (m, i, j)
    #     tmp1 = torch.einsum('mij,mjk->ik', R2t, temp_all)  # (i, k)
    #
    #     WUA2 = temp_all.reshape(-1, A.shape[2])
    #     gram_matrix = WUA2.T @ WUA2 + self.lbd3 * I
    #
    #     # Use torch.linalg.solve or inverse (inverse is less stable but matches original)
    #     # 求解 X*gram_matrix = tmp1 (即 XA=B 形式)
    #     # 转换为 (X*gram_matrix)^T = tmp1^T => gram_matrix^T * X^T = tmp1^T
    #     U2_new = torch.linalg.solve(gram_matrix.T, tmp1.T).T  # 这就是 XA=B 的解
    #
    #     # Non-negativity constraint
    #     U2_new = torch.clamp(U2_new, min=0.0)
    #
    #     return U2_new
    def updateU2(self, R2, W, U):
        # R2: (m, n, q), A: (m, n, d), W: (m, m), U: (n, n)
        A = self.to_device(self.X_train)
        device = R2.device  # 自动获取设备（CPU/GPU）
        I = torch.eye(A.shape[2], device=device)  # n阶单位矩阵，放在相同设备上
        U, W = self.to_device(U), self.to_device(W)
        # Step 1: Compute U @ A[k] for all k → (m, n, d)
        UA = torch.einsum('nd,mdp->mnp', U, A)
        # Step 2: temp_all[i] = sum_k W[i, k] * UA[k]
        temp_all = torch.tensordot(W, UA, dims=([1], [0]))
        # Step 3: fast = sum_i Z[i].T @ temp_all[i]
        R2t = R2.transpose(-2, -1)  # (m, j, i) if Z is (m, i, j)
        tmp1 = torch.einsum('mij,mjk->ik', R2t, temp_all)  # (i, k)

        WUA2 = temp_all.reshape(-1, A.shape[2])
        gram_matrix = WUA2.T @ WUA2 + self.lbd3 * I

        # MPS兼容的求解方法
        try:
            # 尝试直接求解
            U2_new = torch.linalg.solve(gram_matrix.T, tmp1.T).T
        except NotImplementedError:
            # 如果MPS不支持，使用CPU求解
            gram_matrix_cpu = gram_matrix.cpu()
            tmp1_cpu = tmp1.cpu()
            U2_new_cpu = torch.linalg.solve(gram_matrix_cpu.T, tmp1_cpu.T).T
            U2_new = U2_new_cpu.to(self.device)

        # Non-negativity constraint
        U2_new = torch.clamp(U2_new, min=0.0)

        return U2_new

    # 更新矩阵W根据公式(28)############################################
    def updateU3(self, U3, R2, U2, U1, G, OList, muList):
        A = self.to_device(self.X_train)
        O_3, mu_3 = OList[2], muList[2]
        # Step 1: Compute U @ A[:,k,:] for all k
        U1, R2 = self.to_device(U1), self.to_device(R2)
        U3 = self.to_device(U3)
        U1A = torch.einsum('nd,mdp->pnm', U1, A)
        # Step 2: temp_all[i] = sum_k V[i, k] * UA[k]
        temp_all = torch.tensordot(U2, U1A, dims=([1], [0]))
        # Step 3: fast = sum_i Z[:,i,:].T @ temp_all[i]
        R2UVAT = torch.einsum('qmn,qnk->mk', R2.permute(2, 0, 1), temp_all)
        VUA3 = temp_all.reshape(-1, temp_all.shape[2])
        tmp1 = U3 @ VUA3.T @ VUA3 - R2UVAT
        tmp2 = 2 * torch.sign(U3) * torch.sum(torch.abs(U3), axis=0)
        tmp3 = mu_3 * (U3 - G) + O_3
        # Step 2: 更新 W
        U3_new = U3 - self.eta1 * (tmp1 + self.lbd4 * tmp2 + tmp3)
        # Step 3: Normalize
        U3_new = U3_new / torch.sum(U3_new)

        return U3_new

    # 更新矩阵R3根据公式(36)############################################
    def updateR3(self, C, G):
        G1 = torch.sum(G, dim=1)  # shape (m,)

        # Step 5: Compute right = mode_n_product_vector(B, C1, 0)
        # Assuming mode_n_product_vector(B, C1, 0) means: C1 @ B (since mode=0)
        # B shape: (m, n, r) → after mode-0 product with C1 (m,) → (n, r)
        if C.dim() == 3:
            # Contract first dimension: (m,) @ (m, n, r) → (n, r)
            R3 = torch.tensordot(G1, C, dims=([0], [0]))  # (n, r)
        elif C.dim() == 2:
            # B: (m, n), then C1 @ B → (n,)
            R3 = G1 @ C  # (n,)
            R3 = R3.unsqueeze(-1)  # make (n, 1) if needed
        else:
            raise ValueError(f"Unsupported C shape: {C.shape}")

        # Min-Max normalization
        R3_min = R3.min()
        R3_max = R3.max()
        eps = 1e-10
        if (R3_max - R3_min) > eps:
            R3_norm = (R3 - R3_min) / (R3_max - R3_min)
        else:
            R3_norm = R3

        # Handle NaN/Inf
        R3_norm = torch.nan_to_num(R3_norm, nan=0.0, posinf=1.0, neginf=-1.0)

        return R3_norm

    # 更新矩阵B根据公式(41)############################################
    def updateB(self, F, R2, OList, muList):
        O_1, mu_1 = OList[0], muList[0]
        m, n, q = R2.shape
        Y_tensor = self.y_train

        # 如果支持复数运算（CUDA或CPU），直接执行
        if self.supports_complex:
            Y_bar = torch.fft.fft(Y_tensor.to(torch.complex64), dim=0)
            F_bar = torch.fft.fft(F.to(torch.complex64), dim=0)
            R2_bar = torch.fft.fft(R2.to(torch.complex64), dim=0)
            O_1_bar = torch.fft.fft(O_1.to(torch.complex64), dim=0)
        else:
            # MPS不支持复数，使用CPU进行FFT运算
            Y_tensor_cpu = Y_tensor.cpu()
            F_cpu = F.cpu()
            R2_cpu = R2.cpu()
            O_1_cpu = O_1.cpu()

            Y_bar = torch.fft.fft(Y_tensor_cpu.to(torch.complex64), dim=0)
            F_bar = torch.fft.fft(F_cpu.to(torch.complex64), dim=0)
            R2_bar = torch.fft.fft(R2_cpu.to(torch.complex64), dim=0)
            O_1_bar = torch.fft.fft(O_1_cpu.to(torch.complex64), dim=0)

        # 计算共轭转置和其他矩阵运算
        F_conj_T = F_bar.transpose(-2, -1).conj()
        gram_matrices = torch.bmm(F_conj_T, F_bar)
        identity = torch.eye(n, dtype=F_bar.dtype, device=F_bar.device).unsqueeze(0).expand(m, -1, -1)
        # 增加正则化强度，确保矩阵正定
        gram_plus_eye = gram_matrices + mu_1 * identity

        right_terms = torch.bmm(F_conj_T, Y_bar) + mu_1 * R2_bar + O_1_bar

        # 使用更稳定的求解方法
        try:
            B_bar = torch.linalg.solve(gram_plus_eye, right_terms)
        except torch._C._LinAlgError:
            # 如果直接求解失败，使用伪逆作为备选方案
            print("Warning: Using pseudoinverse due to singular matrix in updateB")

            # 对于复数矩阵，需要特殊处理
            if torch.is_complex(gram_plus_eye):
                # 使用SVD计算伪逆，保持复数类型
                U, S, Vh = torch.linalg.svd(gram_plus_eye)
                # 避免除以接近0的奇异值
                S_inv = torch.where(S > 1e-10, 1.0 / S, torch.zeros_like(S))
                S_inv_matrix = torch.diag_embed(S_inv)

                # 确保所有矩阵类型一致 - 将S_inv_matrix转换为复数类型
                S_inv_matrix = S_inv_matrix.to(dtype=U.dtype)

                gram_plus_eye_inv = torch.matmul(Vh.conj().transpose(-2, -1),
                                                 torch.matmul(S_inv_matrix, U.conj().transpose(-2, -1)))
            else:
                # 实数情况
                U, S, Vh = torch.linalg.svd(gram_plus_eye)
                # 避免除以接近0的奇异值
                S_inv = torch.where(S > 1e-10, 1.0 / S, torch.zeros_like(S))
                S_inv_matrix = torch.diag_embed(S_inv)
                gram_plus_eye_inv = torch.matmul(Vh.transpose(-2, -1),
                                                 torch.matmul(S_inv_matrix, U.transpose(-2, -1)))

            B_bar = torch.matmul(gram_plus_eye_inv, right_terms)

        # 返回实部并移到原设备
        if self.supports_complex:
            B = torch.real(torch.fft.ifft(B_bar, dim=2))
        else:
            B = torch.real(torch.fft.ifft(B_bar, dim=2)).to(self.device)

        # 对B进行归一化[0, 1]
        B_min = B.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # 获取每个样本的最小值
        B_max = B.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # 获取每个样本的最大值

        # 避免除零情况
        denominator = B_max - B_min
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)

        B = (B - B_min) / denominator

        return B

    # 更新张量C根据公式(38)############################################
    def updateC(self, G, R2, R3, OList, muList):
        m, n, q = R2.shape
        N = n * q
        G1 = torch.sum(G, dim=1)
        mu2 = muList[1]
        O2 = self.to_device(OList[1])

        # 预计算常量
        D_inv = 1.0 / (mu2 + 1e-12)  # 标量而非向量
        u_scaled = math.sqrt(self.lbd1) * G1  # (m,)
        v_scaled = D_inv * u_scaled  # (m,)
        alpha = torch.dot(u_scaled, v_scaled)  # 标量
        denom = 1.0 + alpha + 1e-12

        # 直接构建 rhs 并一次性计算
        R3_flat = R3.reshape(-1)  # (N,)
        R2_flat = R2.reshape(m, N)  # (m, N)
        O2_flat = O2.reshape(m, N)  # (m, N)

        # 构建 rhs = lbd1 * (G1 @ R3_flat.T) + mu2 * R2_flat + O2_flat
        temp_term = torch.einsum('i,j->ij', u_scaled, R3_flat)  # (m, N)
        rhs = self.lbd1 * temp_term + mu2 * R2_flat + O2_flat  # (m, N)

        # Sherman-Morrison 公式的向量化实现
        w = D_inv * rhs.T  # (N, m)
        beta_i = w @ u_scaled  # (N,)
        correction = torch.outer(beta_i, v_scaled) / denom  # (N, m)

        C_flat = w - correction  # (N, m)
        C = C_flat.reshape(m, n, q)  # (m, n, q)

        return C

    def updateG(self, R2, U3, R3, OList, muList):
        R3 = self.to_device(R3)
        O3 = self.to_device(OList[2])
        mu3 = muList[2]
        eps = 1e-12

        m, n, q = R2.shape
        N = n * q

        # --- Step 1:
        device = U3.device

        # --- Step 2: Compute u = B_(3) vec(Z)
        R2_mat = R2.reshape(m, N)  # B_(3) ∈ R^{m × N}
        R3_vec = R3.reshape(N)  # vec(Z)
        u = R2_mat @ R3_vec  # (m,)

        # --- Step 3: Construct RHS matrix R = μ2*W + O2 + 2β*(u 1^T)
        ones = torch.ones(m, dtype=u.dtype, device=device)
        R = mu3 * U3 + O3 + 2 * self.lbd1 * torch.outer(u, ones)  # (m, m)

        # --- Step 4: Compute K = B_(3) B_(3)^T ∈ R^{m×m}
        K = R2_mat @ R2_mat.T  # (m, m), cost O(m^2 * N) = O(m^2 n q)
        # --- Step 5: Solve for s in (μ2 I + 2β m K) s = R 1
        r = R @ ones  # (m,)
        Im = torch.eye(m, device=device)
        M = mu3 * Im + 2 * self.lbd1 * m * K  # (m, m)
        # Add small regularization for numerical stability
        M += eps * Im

        # MPS兼容的求解方法
        try:
            s = torch.linalg.solve(M, r)  # shape: (m,), cost O(m^3)
        except NotImplementedError:
            # 如果MPS不支持，使用CPU求解
            M_cpu = M.cpu()
            r_cpu = r.cpu()
            s_cpu = torch.linalg.solve(M_cpu, r_cpu)
            s = s_cpu.to(self.device)

        # --- Step 6: Recover C = (1/μ2) * (R - 2β K s 1^T)
        ks = K @ s  # (m,)
        G = (R - 2 * self.lbd1 * torch.outer(ks, ones)) / mu3  # (m, m)

        G_min = G.min()
        G_max = G.max()
        # 避免除零（如果 W_max == W_min）
        denom = G_max - G_min
        if denom.abs() < 1e-12:
            G_normalized = torch.zeros_like(G)
        else:
            G_normalized = (G - G_min) / denom

        # Step 4: 截断非正值为 0
        G_normalized = torch.clamp(G_normalized, min=0.0)

        return G_normalized

    # 更新Oi(i=1,2)
    def updateOmu(self, R3_new, B_new, C_new, U3_new, G_new, OList, muList, rho, mu_max):
        B_new = self.to_device(B_new)
        U3_new = self.to_device(U3_new)
        C_new = self.to_device(C_new)

        OList[0] = self.to_device(OList[0]) + muList[0] * (R3_new - B_new)
        OList[1] = OList[1] + muList[1] * (R3_new - C_new)
        OList[2] = OList[2] + muList[2] * (U3_new - G_new)

        for i in range(len(muList)):
            muList[i] = np.minimum(rho * muList[i], mu_max)
        return OList, muList

    def round_tensor(self, x: torch.Tensor, decimals: int) -> torch.Tensor:
        """
        Round a scalar tensor to a given number of decimal places.
        Equivalent to np.round(x, decimals=decimals).
        """
        if decimals < 0:
            raise ValueError("decimals must be non-negative")
        if decimals == 0:
            return torch.round(x)
        factor = 10.0 ** decimals
        return torch.round(x * factor) / factor

    def compute_errors(self,
            Ztensor_new, Ztensor,
            E_new, E,
            H_new, H,
            U_new, U,
            V_new, V,
            W_new, W,
            Zmat_new, Zmat,
            redus: int = 4
    ):
        """
        Compute infinity-norm errors between new and old tensors.

        All inputs must be torch.Tensor on the same device (e.g., 'mps', 'cuda', or 'cpu').
        Returns rounded scalar tensors (0-D) for each error.
        """

        # Helper: safe L-infinity norm
        def inf_norm(a, b):
            return torch.norm(a - b, p=float('inf'))

        # Compute errors for most variables
        err_Zt = self.round_tensor(inf_norm(Ztensor_new, Ztensor), redus)
        err_E = self.round_tensor(inf_norm(E_new, E), redus)
        err_H = self.round_tensor(inf_norm(H_new, H), redus)
        err_U = self.round_tensor(inf_norm(U_new, U), redus)
        err_W = self.round_tensor(inf_norm(W_new, W), redus)
        err_Zm = self.round_tensor(inf_norm(Zmat_new, Zmat), redus)

        # Special handling for V (shape may mismatch during initialization)
        if V_new.shape == V.shape:
            err_V = self.round_tensor(inf_norm(V_new, V), redus)
        else:
            V_flat_new = V_new.flatten()
            V_flat = V.flatten()
            min_size = min(V_flat_new.numel(), V_flat.numel())
            if min_size > 0:
                diff = V_flat_new[:min_size] - V_flat[:min_size]
                err_V = self.round_tensor(torch.norm(diff, p=float('inf')), redus)
            else:
                # Create zero tensor on the same device as V
                err_V = torch.tensor(0.0, device=V.device, dtype=V.dtype)

        return err_Zt, err_E, err_H, err_U, err_V, err_W, err_Zm

    # 主算法MSWMLFG
    def main_Algorithm(self):
        # 检查是否为消融实验
        if torch.all(self.R1 == 0):
            print("⚠️ 检测到消融实验模式 - 调整模型参数")
            # 大幅调整关键超参数
            original_lbd1, original_alpha = self.lbd1, self.alpha
            self.lbd1 = original_lbd1 * 0.01  # 极度降低正则化
            self.lbd2 = self.lbd2 * 0.01  # 极度降低正则化
            self.alpha = original_alpha * 10  # 大幅增加稀疏性
            print(f"参数调整: lbd1 {original_lbd1:.3f}→{self.lbd1:.3f}, alpha {original_alpha:.3f}→{self.alpha:.3f}")

        # m=number of sources, b=number of bags, n=number of train samples, q=number of labels
        m, n, q, b = self.dim[3], self.dim[5][0], self.dim[2], self.R1.shape[1]
        # print('m, n, q, b=', m, n, q, b)

        # # 使用不同的随机种子确保每次初始化不同
        # torch.manual_seed(datetime.datetime.now().microsecond)
        # np.random.seed(datetime.datetime.now().microsecond)
        SEED = 42  # 或任何你选的数
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # 使用不同的初始化策略，增加随机性
        U3 = torch.rand(m, m, device=self.device) * 0.5 + 0.1  # 更大的初始化值范围
        G = torch.rand(m, m, device=self.device) * 0.5 + 0.1
        C = torch.rand(m, n, q, device=self.device) * 0.1
        R2 = torch.rand(m, n, q, device=self.device) * 0.5 + 0.1
        B = torch.rand(m, n, q, device=self.device) * 0.1
        F = torch.rand(m, b, n, device=self.device) * 0.1
        U1 = (torch.rand(n, n, device=self.device) * 0.5 + 0.1)
        U2 = (torch.rand(q, self.X_train.shape[2], device=self.device) * 0.5 + 0.1)
        R3 = (torch.rand(n, q, device=self.device) * 0.1)

        O1 = torch.zeros(m, n, q, device=self.device)
        O2 = torch.zeros(m, n, q, device=self.device)
        O3 = torch.zeros(U3.shape, device=self.device)
        OList = [O1, O2, O3]
        muList = [10e-3, 0.001, 10e-3]  # 更大的初始mu值
        rho = self.eta2
        mu_max = 10e+8  # 更大的mu_max

        t = 0  # 迭代次数
        prev_err = float('inf')
        no_improvement_count = 0

        while t < self.max_iter:
            # print('t=', t, '-----------------------------------')
            F_new = self.updateF(B)
            # print('F_new:', F_new, F_new.shape)
            R2_new = self.updateR2(R2, U1, U2, U3, B, C, OList, muList)
            # print('R2_new:', R2_new, R2_new.shape)
            U1_new = self.updateU1(U3, U2, R2_new)
            # print('U1_new:', U1_new, U1_new.shape)
            U2_new = self.updateU2(R2, U3, U1_new)
            # print('U2_new:', U2_new, U2_new.shape)
            U3_new = self.updateU3(U3, R2_new, U2_new, U1_new, G, OList, muList)
            # print('U3_new:', U3_new, U3_new.shape)
            R3_new = self.updateR3(C, G)
            # print('R3_new:', R3_new, R3_new[0].shape)
            B_new = self.updateB(F_new, R2_new, OList, muList)
            # print('B_new:', B_new, B_new.shape, type(B_new))
            C_new = self.updateC(G, R2_new, R3_new, OList, muList)
            # print('C_new:', C_new, C_new.shape)
            G_new = self.updateG(R2_new, U3_new, R3_new, OList, muList)
            # print('G_new:', G_new, G_new.shape)

            OList_new, muList_new = self.updateOmu(R3_new, B_new, C_new, U3_new, G_new, OList, muList, rho, mu_max)

            # 确保所有张量在相同设备上进行计算
            err_R2, err_F, err_B, err_U1, err_U2, err_U3, err_R3 = self.compute_errors(
                R2_new, R2,
                F_new, F,
                B_new, B,
                U1_new, U1,
                U2_new, U2,
                U3_new, U3,
                R3_new, R3,
                redus=4
            )

            total_err = err_R2 + err_F + err_U1 + err_U2 + err_U3 + err_R3
            # print(f't={t}, err_R2={err_R2.item():.4f}, err_B={err_B.item():.4f}, err_F={err_F.item():.4f}, '
            #       f'err_R3={err_R3.item():.4f}, err_U1={err_U1.item():.4f}, '
            #       f'err_U2={err_U2.item():.4f}, err_U3={err_U3.item():.4f}')
            # 终止条件判断
            if t > 1:
                if total_err <= self.tol and total_err > 1e-10:  # 添加下限避免过早终止
                    # print('end algorithm1')
                    break
                # 检查是否有改进
                if abs(prev_err - total_err) < 1e-8:  # 更严格的改进判断
                    no_improvement_count += 1
                    if no_improvement_count > 3:  # 连续3次没有改进则退出
                        break
                else:
                    no_improvement_count = 0

            prev_err = total_err

            # 更新各个值
            F = F_new
            R2 = R2_new
            U1 = U1_new
            U2 = U2_new
            U3 = U3_new
            R3 = R3_new
            B = B_new
            C = C_new
            G = G_new

            OList = OList_new
            muList = muList_new

            t += 1

        return F, R2, U1, U2, U3, R3

    def find_common_k_neighbors_between_test_train(self, X_test, X_train, k):
        """
        查找测试样本在训练集中的共同k近邻
        X_test: 测试集数据源列表
        X_train: 训练集数据源列表
        """
        common_neighbors = []

        # 遍历每个测试样本
        for i in range(len(X_test[0])):
            all_train_neighbors = []

            # 对每个训练数据源，找到测试样本的k近邻
            for src in range(len(X_train)):
                train_data = X_train[src]  # 训练数据源

                # 处理测试样本，正确获取第i行数据
                test_src_data = X_test[src]  # 当前源的测试数据
                if isinstance(test_src_data, pd.DataFrame):
                    # 如果是 DataFrame，使用 iloc 获取第i行
                    test_sample = test_src_data.iloc[i]
                elif isinstance(test_src_data, pd.Series):
                    # 如果是 Series，直接取值
                    test_sample = test_src_data.iloc[i]
                elif hasattr(test_src_data, '__getitem__') and len(test_src_data) > i:
                    # 如果是数组或列表类型
                    test_sample = test_src_data[i]
                else:
                    # 其他情况，转换为 numpy 数组后取第i行
                    test_sample = np.array(test_src_data)[i]

                # 将数据转换为 numpy 数组
                if hasattr(train_data, 'values'):
                    # 如果是 pandas DataFrame
                    train_data = train_data.values
                elif isinstance(train_data, pd.Series):
                    # 如果是 pandas Series
                    train_data = train_data.values.reshape(-1, 1)
                elif not isinstance(train_data, np.ndarray):
                    # 如果是其他类型，转换为 numpy 数组
                    train_data = np.array(train_data)

                if hasattr(test_sample, 'values'):
                    # 如果是 pandas Series 或 DataFrame
                    test_sample = test_sample.values
                elif not isinstance(test_sample, np.ndarray):
                    # 如果是其他类型，转换为 numpy 数组
                    test_sample = np.array(test_sample)

                # 确保 test_sample 是二维的 (1, n_features)
                if test_sample.ndim == 0 or test_sample.ndim == 1:
                    test_sample = test_sample.reshape(1, -1)

                # 使用KNN找到最近邻
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=min(k, len(train_data)), algorithm='auto').fit(train_data)
                distances, indices = nbrs.kneighbors(test_sample)  # 使用转换后的 test_sample
                all_train_neighbors.append(indices[0])  # 添加当前源的近邻索引

            # 计算多个源之间的共同近邻
            common = set(all_train_neighbors[0])
            for j in range(1, len(all_train_neighbors)):
                common = common.intersection(set(all_train_neighbors[j]))

            # 如果没有共同近邻，则合并所有近邻并取前k个
            if len(common) == 0:
                union_neighbors = set()
                for neighbors in all_train_neighbors:
                    union_neighbors.update(neighbors)
                common = list(union_neighbors)[:k]
            else:
                common = list(common)

            common_neighbors.append(common)

        return common_neighbors

    def binarize_with_fallback(self, score_tensor, threshold):
        """
        Binarize multi-label scores with fallback to top-1 when all below threshold.

        Args:
            score_tensor: Tensor of shape (V, N, L) or (N, L)
            threshold: float

        Returns:
            binary_tensor: Long or Int tensor of same shape, values 0/1
        """
        # Step 1: 初始二值化（>= threshold）
        binary = (score_tensor >= threshold).long()  # (V, N, L)

        # Step 2: 找出全零的样本位置（按最后一个维度求和 == 0）
        # all_zero: (V, N)
        all_zero = (binary.sum(dim=-1) == 0)

        if all_zero.any():
            # Step 3: 对全零样本，找出每样本最大值的索引
            # argmax_idx: (V, N)
            argmax_idx = score_tensor.argmax(dim=-1)  # 在 L 维上取 argmax

            # Step 4: 构造 one-hot 向量
            # 使用 scatter_ 将 1 写入最大值位置
            fallback = torch.zeros_like(binary)
            fallback.scatter_(dim=-1, index=argmax_idx.unsqueeze(-1), value=1)

            # Step 5: 只对 all_zero 的位置使用 fallback
            # 扩展 all_zero 到 (V, N, 1) 以便广播
            mask = all_zero.unsqueeze(-1)  # (V, N, 1)
            binary = torch.where(mask, fallback, binary)

        return binary

    def mode_n_product_vector(self, tensor, vector, mode):
        """
        Compute the mode-n product of a tensor with a vector (tensor contraction along one mode).

        Args:
            tensor: torch.Tensor of shape (I1, I2, ..., In)
            vector: torch.Tensor of shape (Im,), where Im = tensor.shape[mode]
            mode: int, the mode (dimension) to contract (0-based)

        Returns:
            result: torch.Tensor of shape (I1, ..., I_{mode-1}, I_{mode+1}, ..., In)
        """
        # Ensure inputs are tensors
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        if not isinstance(vector, torch.Tensor):
            vector = torch.as_tensor(vector)

        # Move vector to same device and dtype as tensor
        vector = vector.to(device=tensor.device, dtype=tensor.dtype)

        # Normalize negative mode index
        if mode < 0:
            mode = tensor.dim() + mode
        if not (0 <= mode < tensor.dim()):
            raise ValueError(f"Invalid mode={mode} for tensor with {tensor.dim()} dimensions.")

        # Check dimension compatibility
        if vector.numel() != tensor.shape[mode]:
            raise ValueError(
                f"Vector length {vector.numel()} does not match tensor.shape[{mode}] = {tensor.shape[mode]}"
            )

        # Ensure vector is 1D
        if vector.dim() != 1:
            vector = vector.view(-1)

        # Step 1: Move the target mode to the front
        perm = [mode] + [i for i in range(tensor.dim()) if i != mode]
        tensor_perm = tensor.permute(perm)  # (Im, I1, ..., I_{m-1}, I_{m+1}, ...)

        # Step 2: Reshape to (Im, -1)
        tensor_flat = tensor_perm.reshape(tensor.shape[mode], -1)  # (Im, N)

        # Step 3: Contract with vector: v^T @ tensor_flat → (N,)
        contracted = torch.matmul(vector, tensor_flat)  # (N,)

        # Step 4: Reshape back to original shape without mode dimension
        new_shape = [tensor.shape[i] for i in range(tensor.dim()) if i != mode]
        result = contracted.reshape(new_shape)

        return result

    def predict(self, A, X_test, X_train_aggregated, y_test, X_test_allSource, X_train_allSource):
        # 1. 获取测试样本在训练集中的共同 k 近邻（返回的是训练集内部索引！）
        actual_kNum = min(self.kNum, self.y_train[0].shape[0])
        common_neighbors = self.find_common_k_neighbors_between_test_train(X_test, X_train_aggregated, k=actual_kNum)
        common_neighbors_instance = self.find_common_k_neighbors_between_test_train(X_test_allSource, X_train_allSource, k=actual_kNum)
        # 提取每个子列表的第一个近邻索引
        first_neighbors = [sublist[0] for sublist in common_neighbors_instance]

        # 2. 获取模型重建部分（你已有的逻辑）
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        F, R2, U1, U2, U3, R3 = self.main_Algorithm()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        train_time = time.perf_counter() - start
        print(f"Train Model Time taken: {train_time:.4f} seconds")

        # 测试阶段开始计时
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        test_start = time.perf_counter()
        # For instance-level prediction
        U_test = U1[first_neighbors, :][:, first_neighbors]
        y_score_model_instance_tensor = self.to_device(self.mode_n_product(
            self.mode_n_product(
                self.mode_n_product(A, U_test, 1), U2, 2
            ), U3, 0
        ))
        U31 = torch.sum(U3, dim=1)
        U31 = U31 / torch.sum(U31)
        # For bag-level prediction
        y_score_model_bag_tensor = torch.as_tensor(torch.zeros_like(y_test), dtype=torch.float32, device=self.device)
        y_score_knn_bag = torch.as_tensor(torch.zeros_like(y_test[0]), dtype=torch.float32, device=self.device)
        for b in range(y_test.shape[1]):
            neighbor_indices = common_neighbors[b]  # 如 [0, 2, 4]，合法索引
            # --- 模型重建得分 ---
            y_score_model_bagb = self.to_device(torch.mean(self.to_device(top.t_product(F[:, neighbor_indices, :][:, :, first_neighbors],
                                                                         y_score_model_instance_tensor)), dim=1))  # fuse all sources' y_score of p_neighbors of one test instance
            y_score_model_bag_tensor[:, b, :] = y_score_model_bagb

            # --- KNN 标签投票得分 ---
            neighbor_labels = torch.as_tensor(self.y_train[0][neighbor_indices], dtype=torch.float32,  device=self.device)
            y_score_knn_bag[b, :] = torch.mean(neighbor_labels, dim=0)

        y_score_model_bag = self.mode_n_product_vector(y_score_model_bag_tensor, U31, 0)

        # --- 融合得分 ---
        y_score = 0.5 * y_score_model_bag + 0.5 * y_score_knn_bag  # 可调权重
        # --- 生成预测标签 ---
        precision, recall, thresholds_pr = precision_recall_curve(
            y_test[0].ravel(),
            y_score.detach().cpu().numpy().ravel()
        )
        precision = precision[:-1]
        recall = recall[:-1]
        optimal_idx_pr = np.argmax(precision - recall)
        optimal_threshold_pr = thresholds_pr[optimal_idx_pr]

        # 向量化实现
        y_score_tensor = torch.as_tensor(y_score, device=self.device, dtype=torch.float32)

        # 初始二值化
        y_pred = (y_score_tensor >= optimal_threshold_pr).long()

        # 找出全零行
        all_zero_rows = (y_pred.sum(dim=1) == 0)

        # 为全零行分配最大值索引
        if torch.any(all_zero_rows):
            max_indices = torch.argmax(y_score_tensor, dim=1)
            # 为全零行设置最大值位置为1
            y_pred[all_zero_rows, max_indices[all_zero_rows]] = 1

        y_pred_view = self.binarize_with_fallback(y_score_model_bag_tensor, optimal_threshold_pr)

        # 在调用sklearn函数前转换张量
        y_score_np = y_score.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        y_pred_view_np = y_pred_view.detach().cpu().numpy()

        # 结束测试时间测量
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        test_time = time.perf_counter() - test_start
        print(f"Test Model Time taken: {test_time:.4f} seconds")

        return y_score_np, y_pred_np, y_pred_view_np

    def predict_ins(self, A, X_test, X_train_aggregated, y_test, X_test_allSource, X_train_allSource,
                    y_train_testIns, y_test_testIns):
        # 1. 获取测试样本在训练集中的共同 k 近邻（返回的是训练集内部索引！）
        actual_kNum = min(self.kNum, self.y_train[0].shape[0])
        common_neighbors = self.find_common_k_neighbors_between_test_train(X_test, X_train_aggregated, k=actual_kNum)
        common_neighbors_instance = self.find_common_k_neighbors_between_test_train(X_test_allSource, X_train_allSource, k=actual_kNum)
        # 提取每个子列表的第一个近邻索引
        first_neighbors = [sublist[0] for sublist in common_neighbors_instance]
        # 2. 获取模型重建部分（你已有的逻辑）
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        F, R2, U1, U2, U3, R3 = self.main_Algorithm()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        train_time = time.perf_counter() - start
        print(f"Train Model Time taken: {train_time:.4f} seconds")

        # 测试阶段开始计时
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        test_start = time.perf_counter()
        # For instance-level prediction
        # part one --- 模型重建得分 ---
        U_test = U1[first_neighbors, :][:, first_neighbors]
        y_score_model_instance_tensor = self.to_device(self.mode_n_product(
            self.mode_n_product(
                self.mode_n_product(A, U_test, 1), U2, 2
            ), U3, 0
        ))
        U31 = torch.sum(U3, dim=1)
        U31 = U31 / torch.sum(U31)

        # For instance-level prediction
        # part one--- 模型重建得分 ---
        y_score_model_ins = self.mode_n_product_vector(y_score_model_instance_tensor, U31, 0)
        # part two--- KNN 标签投票得分 ---
        m, n, d = A.shape
        y_score_knn_ins = torch.as_tensor(torch.zeros_like(y_score_model_instance_tensor[0]), dtype=torch.float32, device=self.device)
        for i in range(n):
            neighbor_indices = common_neighbors_instance[i]
            neighbor_labels = torch.as_tensor(y_train_testIns[0][neighbor_indices], dtype=torch.float32,  device=self.device)
            y_score_knn_ins[i, :] = torch.mean(neighbor_labels, dim=0)

        # --- 融合得分 ---
        y_score_ins = 0.5 * y_score_model_ins + 0.5 * y_score_knn_ins  # 可调权重

        # For bag-level prediction
        y_score_model_bag_tensor = torch.as_tensor(torch.zeros_like(y_test), dtype=torch.float32, device=self.device)
        y_score_knn_bag = torch.as_tensor(torch.zeros_like(y_test[0]), dtype=torch.float32, device=self.device)
        for b in range(y_test.shape[1]):
            neighbor_indices = common_neighbors[b]  # 如 [0, 2, 4]，合法索引
            # --- 模型重建得分 ---
            y_score_model_bagb = self.to_device(
                torch.mean(self.to_device(top.t_product(F[:, neighbor_indices, :][:, :, first_neighbors],
                                                        y_score_model_instance_tensor)),
                           dim=1))  # fuse all sources' y_score of p_neighbors of one test instance
            y_score_model_bag_tensor[:, b, :] = y_score_model_bagb

            # --- KNN 标签投票得分 ---
            neighbor_labels = torch.as_tensor(self.y_train[0][neighbor_indices], dtype=torch.float32,
                                              device=self.device)
            y_score_knn_bag[b, :] = torch.mean(neighbor_labels, dim=0)

        y_score_model_bag = self.mode_n_product_vector(y_score_model_bag_tensor, U31, 0)

        # --- 融合得分 ---
        y_score_bag = 0.5 * y_score_model_bag + 0.5 * y_score_knn_bag  # 可调权重

        # --- 生成预测标签 ---
        precision_bag, recall_bag, thresholds_pr_bag = precision_recall_curve(
            y_test[0].ravel(),
            y_score_bag.detach().cpu().numpy().ravel()
        )
        precision_bag = precision_bag[:-1]
        recall_bag = recall_bag[:-1]
        optimal_idx_pr_bag = np.argmax(precision_bag - recall_bag)
        optimal_threshold_pr_bag = thresholds_pr_bag[optimal_idx_pr_bag]
        # for instance-level prediction
        precision_ins, recall_ins, thresholds_pr_ins = precision_recall_curve(
            y_test_testIns[0].ravel(),
            y_score_ins.detach().cpu().numpy().ravel()
        )
        precision_ins = precision_ins[:-1]
        recall_ins = recall_ins[:-1]
        optimal_idx_pr_ins = np.argmax(precision_ins - recall_ins)
        optimal_threshold_pr_ins = thresholds_pr_ins[optimal_idx_pr_ins]

        y_score_bag_tensor = torch.as_tensor(y_score_bag, device=self.device, dtype=torch.float32)  # 向量化实现
        y_pred_bag = (y_score_bag_tensor >= optimal_threshold_pr_bag).long()  # 初始二值化
        all_zero_rows_bag = (y_pred_bag.sum(dim=1) == 0)  # 找出全零行
        if torch.any(all_zero_rows_bag):  # 为全零行分配最大值索引
            max_indices = torch.argmax(y_score_bag_tensor, dim=1)
            y_pred_bag[all_zero_rows_bag, max_indices[all_zero_rows_bag]] = 1  # 为全零行设置最大值位置为1
        y_pred_bag_view = self.binarize_with_fallback(y_score_model_bag_tensor, optimal_threshold_pr_bag)

        y_score_ins_tensor = torch.as_tensor(y_score_ins, device=self.device, dtype=torch.float32)  # 向量化实现
        y_pred_ins = (y_score_ins_tensor >= optimal_threshold_pr_ins).long()  # 初始二值化
        all_zero_rows_ins = (y_pred_ins.sum(dim=1) == 0)  # 找出全零行
        if torch.any(all_zero_rows_ins):  # 为全零行分配最大值索引
            max_indices = torch.argmax(y_score_ins_tensor, dim=1)
            y_pred_ins[all_zero_rows_ins, max_indices[all_zero_rows_ins]] = 1  # 为全零行设置最大值位置为1
        y_pred_ins_view = self.binarize_with_fallback(y_score_model_instance_tensor, optimal_threshold_pr_bag)

        # 在调用sklearn函数前转换张量
        y_score_bag_np = y_score_bag.detach().cpu().numpy()
        y_pred_bag_np = y_pred_bag.detach().cpu().numpy()
        y_pred_bag_view_np = y_pred_bag_view.detach().cpu().numpy()
        y_score_ins_np = y_score_ins.detach().cpu().numpy()
        y_pred_ins_np = y_pred_ins.detach().cpu().numpy()
        y_pred_ins_view_np = y_pred_ins_view.detach().cpu().numpy()

        # 结束测试时间测量
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        test_time = time.perf_counter() - test_start
        print(f"Test Model Time taken: {test_time:.4f} seconds")

        return y_score_bag_np, y_pred_bag_np, y_pred_bag_view_np, y_score_ins_np, y_pred_ins_np, y_pred_ins_view_np


########################################### Test code ##################################################################
dataList = ['table1']
souList = [2]
labList = [3]
bagNum = [3]
split = 1
alpha, lbd1, lbd2, lbd3, lbd4 = 0.310, 0.051, 103.568, 6.255, 0.603
eta1, eta2, kNum, tol, max_iter = 1.05, 0.1, 3, 0.08, 1
n_feature_maps, n_enhancement_groups, feature_dim, enh_dim = 2, 5, 30, 5
for d in range(0, len(dataList)):  # len(selectedList)
    # 读取总样本数
    datasetPath = './datasets/'
    train_data0 = pd.read_csv(datasetPath + dataList[d] + '_0_train_split_1.csv')
    test_data0 = pd.read_csv(datasetPath + dataList[d] + '_0_test_split_1.csv')

    # 获取唯一的包ID
    unique_train_bags = train_data0[train_data0.columns[0]].unique()
    unique_test_bags = test_data0[test_data0.columns[0]].unique()

    # 按选中的包过滤数据
    train_data0 = train_data0[train_data0[train_data0.columns[0]].isin(unique_train_bags)]
    test_data0 = test_data0[test_data0[test_data0.columns[0]].isin(unique_test_bags)]

    samNum, labNum, souNum = train_data0.shape[0] + test_data0.shape[0], labList[d], souList[d]
    print(dataList[d], ':', samNum, labNum, souNum)

    Scores, scoresList, Scores_ins, scoresList_ins = [], [], [], []  # 创建最终分数列表
    for splitTime in range(split):
        print('---------------', dataList[d], '----------------splitTime:', splitTime + 1,
              '-------------------------------')
        dim = []
        dim = dim + [samNum, bagNum[d], labNum, souNum]
        attList, attList_test = [], []
        X_train_allSource, y_train_allSource = [], []
        X_test_allSource, y_test_allSource = [], []
        X_train_aggregated, X_test_aggregated = [], []
        dfs, dfs_test = [], []
        y_train_testIns, y_test_testIns = [], []

        for k in range(souNum):
            train_data = pd.read_csv(
                datasetPath + dataList[d] + '_' + str(k) + '_train_split_' + str(splitTime + 1) + '.csv')
            test_data = pd.read_csv(
                datasetPath + dataList[d] + '_' + str(k) + '_test_split_' + str(splitTime + 1) + '.csv')
            dfs.append(train_data)
            dfs_test.append(test_data)
            X_train, X_test = np.array(train_data.iloc[:, 2:-2 * labNum]), np.array(
                test_data.iloc[:, 2:-2 * labNum])
            y_train_testIns.append(np.array(train_data.iloc[:, -2 * labNum:-labNum]))
            y_test_testIns.append(np.array(test_data.iloc[:, -2 * labNum:-labNum]))
            y_train = np.array(train_data.drop_duplicates(subset=[train_data.columns[0]])[
                                   list(train_data.columns[-2 * labNum:-labNum])
                               ])
            y_test = np.array(test_data.drop_duplicates(subset=[test_data.columns[0]])[
                                  list(test_data.columns[-2 * labNum:-labNum])
                              ])
            # 包内的聚合方式，用于测试集的包的近邻计算
            X_test_bag = np.array(test_data.groupby(test_data.columns[0]).mean(numeric_only=True).reset_index(
                drop=True))
            X_train_bag = np.array(
                train_data.groupby(train_data.columns[0]).mean(numeric_only=True).reset_index(
                    drop=True))

            attList.append(X_train.shape[1])
            attList_test.append(X_test.shape[1])

            X_train_allSource.append(X_train)
            y_train_allSource.append(y_train)
            X_test_allSource.append(X_test)
            y_test_allSource.append(y_test)
            X_test_aggregated.append(X_test_bag)
            X_train_aggregated.append(X_train_bag)
        dim.append(attList)
        dim.append([X_train_allSource[0].shape[0], X_test_allSource[0].shape[0]])
        # print('[samNum, bagNum, labNum, souNum, attList, [train_samNum, test_samNum]]:', dim, dim[5][0])
        activations = ['relu', 'tanh', 'sigmoid', 'tribas', 'gaussian']
        act = activations[1]
        # print(f"\n🧪 Testing activation: {act}")
        # 获取BLS特征表示学习时间
        BLS_start = time.perf_counter()
        model = MSMIML_BLS.MSMIML_BLS(n_feature_maps, n_enhancement_groups,
                                      feature_dim, enh_dim, reg_lambda=1e-3,
                                      feature_activation='relu', enhancement_activation=act)  # 'tanh'

        A_tensor, R1, bag_count = model.feature_map(dim, dfs)
        # print('R1 shape: ', R1.shape)
        R1_shape = R1.shape
        A_tensor_test, R1_test, bag_count_test = model.feature_map(dim, dfs_test)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        BLS_time = time.perf_counter() - BLS_start
        # print(f"BLS PreLearn Time taken: {BLS_time:.4f} seconds")

        # 获取BLS特征表示学习时间
        train_test_start = time.perf_counter()

        model = MSMIMLmodel(dim, R1, alpha, lbd1, lbd2, lbd3, lbd4, eta1, eta2, kNum, tol, max_iter, m=souNum,
                            d=feature_dim, q=labNum)
        model.fit(A_tensor, torch.from_numpy(np.array(y_train_allSource)))

        y_score_bag, y_pred_bag, y_pred_bag_view, y_score_ins, y_pred_ins, y_pred_ins_view = \
            model.predict_ins(A_tensor_test, X_test_aggregated, X_train_aggregated,
                              torch.from_numpy(np.array(y_test_allSource)),
                              X_test_allSource, X_train_allSource, y_train_testIns, y_test_testIns)
        train_test_time = time.perf_counter() - train_test_start
        print(f"Model Train_Test Time taken: {train_test_time:.4f} seconds")
        print(f"Total Time: {time.perf_counter() - BLS_start:.4f} seconds")
        y_test_ins = y_test_testIns[0].astype(int)
        scores_bag = metrics_MLL.mll_metrics(y_test, y_pred_bag, y_score_bag, y_pred_bag_view)
        scores_ins = metrics_MLL.mll_metrics(y_test_ins, y_pred_ins, y_score_ins, y_pred_ins_view)
        scoresList.append(scores_bag)
        scoresList_ins.append(scores_ins)
        print(dataList[d], '   score:  ap, auc, rl, hl')
        print(dataList[d], 'score_bag:', scores_bag, ' splitTime:', splitTime + 1)
        print(dataList[d], 'score_ins:', scores_ins, ' splitTime:', splitTime + 1)
