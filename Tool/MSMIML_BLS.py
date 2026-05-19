import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
class MSMIML_BLS:
    def __init__(self, n_feature_maps=3, n_enhancement_groups=2,
                 feature_dim=100, enh_dim=50, reg_lambda=1e-3,
                 feature_activation='relu', enhancement_activation='tanh'):
        """
        Parameters:
            n_feature_maps: m1 is number of feature map groups
            n_enhancement_groups: m2 is number of enhancement node groups
            feature_dim: p1 is dimension of each Z_i
            enh_dim: p2 is dimension of each H_j
            feature_activation: φ_i (usually fixed as ReLU)
            enhancement_activation: ξ_j (choose from Table I)
        """
        self.m1 = n_feature_maps
        self.m2 = n_enhancement_groups
        self.p1 = feature_dim
        self.p2 = enh_dim
        self.lam = reg_lambda
        self.phi = feature_activation
        self.xi = enhancement_activation

        # 存储所有随机参数（用于 predict）
        self.W_ei_list = []
        self.beta_ei_list = []
        self.W_hj_list = []
        self.beta_hj_list = []

    def to_device(self, tensor):
        """将张量移动到指定设备"""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        else:
            return torch.tensor(tensor, device=self.device, dtype=torch.float32)

    def _apply_activation_torch(self, x, name):
        """Apply activation function to torch tensors"""
        if name == 'relu':
            return torch.relu(x)
        elif name == 'sigmoid':
            return torch.sigmoid(x)
        elif name == 'tanh':
            return torch.tanh(x)
        elif name == 'tansigmoid':
            # Custom tansigmoid implementation
            x_clipped = torch.clamp(x, -500, 500)
            return (1 - torch.exp(-2 * x_clipped)) / (1 + torch.exp(-2 * x_clipped))
        elif name == 'tribas':
            return torch.max(torch.tensor(1.0) - torch.abs(x), torch.tensor(0.0))
        elif name == 'gaussian':
            # Calculate L2 norm squared for each sample
            norm_sq = torch.sum(x ** 2, dim=1, keepdim=True)
            return torch.exp(-norm_sq)
        else:
            raise ValueError(f"Unknown activation: {name}")

    def extract_bag_samples0(self, dfs, bag_col, dim):
        """
        提取所有数据源中每个bag的样本

        Args:
            dfs: 包含所有数据源的DataFrame列表
            bag_col: bag列的名称

        Returns:
            bag_samples_dict: 字典，键为数据源索引，值为另一个字典，其键为bag_id，值为对应样本的DataFrame
            bag_counts_list: 每个数据源中每个bag的样本数列表
        """
        bag_samples_dict = {}
        bag_counts_list = []
        samNum, bagNum, souNum = dim[0], dim[1], dim[3]
        bag_instance_tensor = torch.zeros(souNum, len(dfs[0][bag_col].unique()), samNum)

        for k, df in enumerate(dfs):
            bag_samples_dict[k] = {}
            bag_counts_k = []

            # 获取当前数据源的所有唯一bag
            unique_bags = df[bag_col].unique()
            # print(f"Data source {k}: {len(unique_bags)} bags, {unique_bags}")

            flag = 0
            for bag_id in sorted(unique_bags):
                # print(f"Processing bag {sorted(unique_bags)}")
                # 提取特定bag的所有样本
                bag_samples = df[df[bag_col] == bag_id].copy()
                sample_count = len(bag_samples)

                bag_samples_dict[k][bag_id] = bag_samples
                bag_counts_k.append(sample_count)

                sample_id = bag_samples.iloc[:, 1]
                # print(f"bag_id: {bag_id}, sample_id.values: {len(sample_id.values)}, "
                #       f"bag_instance_tensor: {bag_instance_tensor.shape}")
                # print(k, bag_id, sample_id.values)
                bag_instance_tensor[k, flag, sample_id.values] = 1
                flag += 1

            bag_counts_list.append(bag_counts_k)
        filtered_tensor = bag_instance_tensor[:, :, torch.any(torch.sum(bag_instance_tensor, dim=1) > 0, dim=0)]

        return bag_samples_dict, bag_counts_list, filtered_tensor

    def extract_bag_samples(self, B, bag_col):
        """
        Extract bag samples information
        """
        bag_samples_dict = {}
        bag_counts_list = []

        # 获取所有bag IDs的集合，然后确定张量尺寸
        all_bag_ids = set()
        max_row_index = 0

        for k in range(len(B)):
            bag_data = B[k]
            # 收集所有bag ID
            unique_bags = bag_data[bag_col].unique()
            all_bag_ids.update(unique_bags)

            # 记录最大的行索引
            current_max_row_idx = len(bag_data) - 1
            if current_max_row_idx > max_row_index:
                max_row_index = current_max_row_idx

        # 创建映射：bag_id -> bag索引
        bag_id_to_idx = {bag_id: idx for idx, bag_id in enumerate(sorted(all_bag_ids))}

        # 创建适当大小的张量
        num_sources = len(B)
        num_bags = len(all_bag_ids)
        num_instances = max_row_index + 1  # 使用行索引的最大值

        bag_instance_tensor = torch.zeros(num_sources, num_bags, num_instances, dtype=torch.float32)

        for k in range(len(B)):
            bag_data = B[k]
            bag_ids = bag_data[bag_col].unique()
            bag_samples_dict[k] = {}

            for bag_id in bag_ids:
                # 获取当前包的实例（行）索引
                sample_mask = (bag_data[bag_col] == bag_id)
                sample_indices = bag_data[sample_mask].index.tolist()
                bag_samples_dict[k][bag_id] = sample_indices

                # 获取bag的索引
                bag_idx = bag_id_to_idx[bag_id]

                # 安全地填充 bag_instance_tensor，检查索引边界
                valid_sample_indices = [idx for idx in sample_indices if idx < num_instances]
                if valid_sample_indices and bag_idx < num_bags:
                    bag_instance_tensor[k, bag_idx, valid_sample_indices] = 1

            bag_counts_list.append(len(bag_ids))

        return bag_samples_dict, bag_counts_list, bag_instance_tensor

    def feature_map0(self, dim, B):
        """
        Input: {B_k^l (nl×dk)} for m sources and b bag, list, one entry is dataframe of all bag under source k
                Y (n×c)
        Output: self.Wout (trained output weights) and A
        MSMIML_BLS_way1是BLS适应多源多示例多标签数据集的第一种方式：
    一个视图k下，一个bag l 独立生成Z_k^l (n_l x p_1m) 和H_k^l (n_l x p_2m) ，得到A_k^l (n_l x (p_1m+p_2m) )
    -->一个视图下，所有bag的A_k^1, A_k^2, ..., A_k^b组合成A_k (n x (p_1m+p_2m) )
    -->所有视图的A_k, k=1,2,...,m 组合成张量A (n x (p_1m+p_2m) x m)
        """
        samNum, bagNum, labNum, souNum, attList = dim[0], dim[1], dim[2], dim[3], dim[4]
        # 使用函数提取不同包的样本
        bag_col = B[0].columns[0]
        bag_samples_dict, bag_counts_list, bag_instance_tensor = self.extract_bag_samples(B, bag_col)
        # 计算数据源0的所有行数总和，也就是测试样本的数量
        total_rows_source_0 = 0
        for bag_id in bag_samples_dict[0].keys():
            total_rows_source_0 += bag_samples_dict[0][bag_id].shape[0]
        # print(f"数据源0的总行数: {total_rows_source_0}")

        A_tensor = torch.zeros(souNum, total_rows_source_0, self.p1 * self.m1 + self.p2 * self.m2)
        for k in range(souNum):
            dk = attList[k]
            W_ei_feature_maps, beta_ei_feature_maps = [], []
            W_hj_enhancement_nodes, beta_hj_enhancement_nodes = [], []
            # 访问特定bag的样本
            A_allBag_List = []
            for l in sorted(bag_samples_dict[k].keys()):
                samples = bag_samples_dict[k][l]
                Bk_l_np = samples.iloc[:, 2:dk + 2].values  # 转换为numpy array
                # 标准化（重要！）
                scaler = StandardScaler()
                Bk_l_np = scaler.fit_transform(Bk_l_np)
                Bk_l = torch.from_numpy(Bk_l_np).float()  # 转换为tensor

                # Generate p1 groups of feature mappings
                Z_list = []
                for i in range(self.m1):  # Line 1: for i = 1 to m1
                    W_ei_np = np.random.randn(dk, self.p1) * 0.1  # Random W_ei ∈ ℝ^{dk×p1}
                    beta_ei_np = np.random.randn(1, self.p1) * 0.1  # Random β_ei ∈ ℝ^{1×p1}

                    # 转换为tensor
                    W_ei = torch.from_numpy(W_ei_np).float()
                    beta_ei = torch.from_numpy(beta_ei_np).float()

                    Zi_input = torch.matmul(Bk_l, W_ei) + beta_ei  # Compute X W_ei + β_ei
                    # print('Zi_input', Zi_input.shape)
                    Zi = self._apply_activation_torch(Zi_input, self.phi)  # Apply activation
                    Z_list.append(Zi)
                    # Save for prediction (convert back to numpy if needed for prediction)
                    if l == 0:
                        W_ei_feature_maps.append(W_ei.numpy())
                        beta_ei_feature_maps.append(beta_ei.numpy())

                # ─────────── Algorithm 1: Line 5 ───────────
                Zn = torch.cat(Z_list, dim=1)  # Concatenate along feature dimension

                # ─────────── Algorithm 1: Lines 6–9 ───────────
                H_list = []
                for j in range(self.m2):  # Line 6: for j = 1 to m
                    W_hj_np = np.random.randn(Zn.shape[1], self.p2) * 0.1  # W_hj ∈ ℝ^{(∑p)×q}
                    beta_hj_np = np.random.randn(1, self.p2) * 0.1  # β_hj ∈ ℝ^{1×q}

                    # 转换为tensor
                    W_hj = torch.from_numpy(W_hj_np).float()
                    beta_hj = torch.from_numpy(beta_hj_np).float()

                    Hj_input = torch.matmul(Zn, W_hj) + beta_hj  # Compute Z^n W_hj + β_hj
                    Hj = self._apply_activation_torch(Hj_input, self.xi)  # Apply activation
                    H_list.append(Hj)
                    if l == 0:
                        W_hj_enhancement_nodes.append(W_hj.numpy())
                        beta_hj_enhancement_nodes.append(beta_hj.numpy())

                # ─────────── Algorithm 1: Line 10 ───────────
                Hm = torch.cat(H_list, dim=1)  # Concatenate enhancement nodes

                # ─────────── Algorithm 1: Line 11 ───────────
                A_bag = torch.cat([Zn, Hm], dim=1)  # Concatenate feature and enhancement
                A_allBag_List.append(A_bag)


            A_allBag = torch.cat(A_allBag_List, dim=0)  # Vertically stack all bags
            A_tensor[k] = A_allBag
            # print(f"source {k} 特征组合后的特征形状: {A_allBag.shape}")
            # Save for prediction
            self.W_hj_list.append(W_hj_enhancement_nodes)
            self.beta_hj_list.append(beta_hj_enhancement_nodes)
            self.W_ei_list.append(np.array(W_ei_feature_maps))
            self.beta_ei_list.append(np.array(beta_ei_feature_maps))

        # print(f"所有数据源特征组合后的特征形状: {A_tensor.shape}")

        return A_tensor, bag_instance_tensor, bag_counts_list

    def feature_map(self, dim, B):
        """
        Input: {B_k^l (nl×dk)} for m sources and b bag, list, one entry is dataframe of all bag under source k
                Y (n×c)
        Output: self.Wout (trained output weights) and A
        MSMIML_BLS_way1是BLS适应多源多示例多标签数据集的第一种方式：
    一个视图k下，一个bag l 独立生成Z_k^l (n_l x p_1m) 和H_k^l (n_l x p_2m) ，得到A_k^l (n_l x (p_1m+p_2m) )
    -->一个视图下，所有bag的A_k^1, A_k^2, ..., A_k^b组合成A_k (n x (p_1m+p_2m) )
    -->所有视图的A_k, k=1,2,...,m 组合成张量A (n x (p_1m+p_2m) x m)
        """
        samNum, bagNum, labNum, souNum, attList = dim[0], dim[1], dim[2], dim[3], dim[4]
        # 使用函数提取不同包的样本
        bag_col = B[0].columns[0]
        bag_samples_dict, bag_counts_list, bag_instance_tensor = self.extract_bag_samples(B, bag_col)

        # 计算数据源0的所有行数总和，也就是测试样本的数量
        total_rows_source_0 = 0
        for bag_id in bag_samples_dict[0].keys():
            # 获取对应的数据frame
            bag_data = B[0][B[0][bag_col] == bag_id]  # 从原始数据中获取对应bag的数据
            total_rows_source_0 += bag_data.shape[0]

        A_tensor = torch.zeros(souNum, total_rows_source_0, self.p1 * self.m1 + self.p2 * self.m2)
        # print(attList, souNum)
        for k in range(souNum):
            dk = attList[k]
            W_ei_feature_maps, beta_ei_feature_maps = [], []
            W_hj_enhancement_nodes, beta_hj_enhancement_nodes = [], []
            # 访问特定bag的样本
            A_allBag_List = []
            for l in sorted(bag_samples_dict[k].keys()):
                # 使用原始数据帧来获取对应bag的数据
                samples = B[k][B[k][bag_col] == l]  # 从原始数据B[k]中获取对应bag的数据
                Bk_l_np = samples.iloc[:, 2:dk + 2].values  # 转换为numpy array
                # 标准化（重要！）
                scaler = StandardScaler()
                Bk_l_np = scaler.fit_transform(Bk_l_np)
                Bk_l = torch.from_numpy(Bk_l_np).float()  # 转换为tensor

                # Generate p1 groups of feature mappings
                Z_list = []
                for i in range(self.m1):  # Line 1: for i = 1 to m1
                    W_ei_np = np.random.randn(dk, self.p1) * 0.1  # Random W_ei ∈ ℝ^{dk×p1}
                    beta_ei_np = np.random.randn(1, self.p1) * 0.1  # Random β_ei ∈ ℝ^{1×p1}

                    # 转换为tensor
                    W_ei = torch.from_numpy(W_ei_np).float()
                    beta_ei = torch.from_numpy(beta_ei_np).float()

                    Zi_input = torch.matmul(Bk_l, W_ei) + beta_ei  # Compute X W_ei + β_ei
                    # print('Zi_input', Zi_input.shape)
                    Zi = self._apply_activation_torch(Zi_input, self.phi)  # Apply activation
                    Z_list.append(Zi)
                    # Save for prediction (convert back to numpy if needed for prediction)
                    if l == sorted(bag_samples_dict[k].keys())[0]:  # 只在第一个bag时保存
                        W_ei_feature_maps.append(W_ei.numpy())
                        beta_ei_feature_maps.append(beta_ei.numpy())

                # ─────────── Algorithm 1: Line 5 ───────────
                Zn = torch.cat(Z_list, dim=1)  # Concatenate along feature dimension

                # ─────────── Algorithm 1: Lines 6–9 ───────────
                H_list = []
                for j in range(self.m2):  # Line 6: for j = 1 to m
                    W_hj_np = np.random.randn(Zn.shape[1], self.p2) * 0.1  # W_hj ∈ ℝ^{(∑p)×q}
                    beta_hj_np = np.random.randn(1, self.p2) * 0.1  # β_hj ∈ ℝ^{1×q}

                    # 转换为tensor
                    W_hj = torch.from_numpy(W_hj_np).float()
                    beta_hj = torch.from_numpy(beta_hj_np).float()

                    Hj_input = torch.matmul(Zn, W_hj) + beta_hj  # Compute Z^n W_hj + β_hj
                    Hj = self._apply_activation_torch(Hj_input, self.xi)  # Apply activation
                    H_list.append(Hj)
                    if l == sorted(bag_samples_dict[k].keys())[0]:  # 只在第一个bag时保存
                        W_hj_enhancement_nodes.append(W_hj.numpy())
                        beta_hj_enhancement_nodes.append(beta_hj.numpy())

                # ─────────── Algorithm 1: Line 10 ───────────
                Hm = torch.cat(H_list, dim=1)  # Concatenate enhancement nodes

                # ─────────── Algorithm 1: Line 11 ───────────
                A_bag = torch.cat([Zn, Hm], dim=1)  # Concatenate feature and enhancement
                A_allBag_List.append(A_bag)

            A_allBag = torch.cat(A_allBag_List, dim=0)  # Vertically stack all bags
            A_tensor[k] = A_allBag
            # print(f"source {k} 特征组合后的特征形状: {A_allBag.shape}")
            # Save for prediction
            self.W_hj_list.append(W_hj_enhancement_nodes)
            self.beta_hj_list.append(beta_hj_enhancement_nodes)
            self.W_ei_list.append(np.array(W_ei_feature_maps))
            self.beta_ei_list.append(np.array(beta_ei_feature_maps))

        # print(f"所有数据源特征组合后的特征形状: {A_tensor.shape}")
        print(f"PCA feature_map - A_tensor final shape V3: {A_tensor.shape}")
        return A_tensor, bag_instance_tensor, bag_counts_list

    def train(self, dim, B, Y):
        # 将Y转换为tensor - 修复类型检查逻辑
        if isinstance(Y, list):
            Y = np.array(Y)
        Y_tensor = torch.from_numpy(Y).float() if isinstance(Y, np.ndarray) else Y.float()
        A_tensor, bag_instance_tensor, bag_counts_list = self.feature_map(dim, B)
        A_ins = torch.sum(A_tensor, dim=0)
        # 对A进行包内聚合
        start_idx = 0
        aggregated_rows = []
        for count in bag_counts_list[0]:  # 取第一个数据源的包计数
            if count > 0:
                aggregated_rows.append(
                    torch.mean(A_ins[start_idx:start_idx + count], dim=0)
                )
            start_idx += count
        A = torch.stack(aggregated_rows, dim=0) if aggregated_rows else \
            torch.empty(0, A_ins.shape[1], device=A_ins.device, dtype=A_ins.dtype)
        # print('bag_counts_list:', bag_counts_list)
        # print(f"特征组合后的特征形状: {A.shape}")

        # ─────────── Algorithm 1: Lines 12–13 (Eq.6 & Eq.7) ───────────
        # Use regularized least squares: W = (A^T A + λI)^{-1} A^T Y
        AT = A.t()  # Transpose
        ATA = torch.matmul(AT, A)  # A^T A

        # Add regularization term (λI)
        I = torch.eye(ATA.shape[0]).to(ATA.device)
        ATA_reg = ATA + self.lam * I

        # print(AT.shape, Y_tensor.shape)

        ATY = torch.matmul(AT, Y_tensor[0])  # A^T Y

        # Solve the linear system (PyTorch doesn't have direct solve like numpy)
        # We need to convert to numpy for solving, then back to tensor
        ATA_np = ATA_reg.detach().cpu().numpy()
        ATY_np = ATY.detach().cpu().numpy()

        Wout_np = np.linalg.solve(ATA_np, ATY_np)
        self.Wout = torch.from_numpy(Wout_np).float()
        return self.Wout, A_tensor

    def predict(self, X, dim_test):
        """Use stored random parameters to compute prediction with torch tensors."""
        souNum, attList = dim_test[3], dim_test[4]
        # 使用函数提取不同包的样本
        bag_col = X[0].columns[0]
        bag_samples_dict, bag_counts_list, bag_insctance_tensor = self.extract_bag_samples(X, bag_col, dim_test)
        # print("bag_counts_list:", bag_counts_list)
        # 计算数据源0的所有行数总和，也就是测试样本的数量
        total_rows_source_0 = 0
        for bag_id in bag_samples_dict[0].keys():
            total_rows_source_0 += bag_samples_dict[0][bag_id].shape[0]

        A_tensor = torch.zeros(souNum, total_rows_source_0, self.p1 * self.m1 + self.p2 * self.m2)
        for k in range(souNum):
            dk = attList[k]
            # print(f"数据源 {k} 特征数量: {dk}")
            # 访问特定bag的样本
            A_allBag_List = []
            for l in sorted(bag_samples_dict[k].keys()):
                samples = bag_samples_dict[k][l]
                Xk_l_np = samples.iloc[:, 2:dk + 2].values  # 转换为numpy array
                # 标准化（重要！）
                scaler = StandardScaler()
                Xk_l_np = scaler.fit_transform(Xk_l_np)
                Xk_l = torch.from_numpy(Xk_l_np).float()  # 转换为tensor

                # Reconstruct Z^n
                Z_list = []
                for i in range(self.m1):
                    W_tensor = torch.from_numpy(np.array(self.W_ei_list[k][i])).float()
                    beta_tensor = torch.from_numpy(np.array(self.beta_ei_list[k][i])).float()
                    Zi = self._apply_activation_torch(
                        torch.matmul(Xk_l, W_tensor) + beta_tensor, self.phi
                    )
                    Z_list.append(Zi)
                Zn = torch.cat(Z_list, dim=1)

                # Reconstruct H^m
                H_list = []
                for j in range(self.m2):
                    W_tensor = torch.from_numpy(self.W_hj_list[k][j]).float()
                    beta_tensor = torch.from_numpy(self.beta_hj_list[k][j]).float()

                    Hj = self._apply_activation_torch(
                        torch.matmul(Zn, W_tensor) + beta_tensor, self.xi
                    )
                    H_list.append(Hj)
                Hm = torch.cat(H_list, dim=1)

                A_bag = torch.cat([Zn, Hm], dim=1)  # Concatenate feature and enhancement
                A_allBag_List.append(A_bag)

            A_allBag = torch.cat(A_allBag_List, dim=0)  # Vertically stack all bags
            A_tensor[k] = A_allBag

        A = torch.sum(A_tensor, dim=0)

        return torch.matmul(A, self.Wout)