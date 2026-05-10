from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

class MSMIML_PCA:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca_list = []

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

    def feature_map(self, dim, B):
        if len(dim) >= 5:
            samNum, bagNum, labNum, souNum, attList = dim[:5]
        else:
            raise ValueError("dim 至少需要包含5个元素")

        # 使用实际的数据行数而不是dim[0]
        actual_rows = B[0].shape[0]  # 当前数据源的样本数
        # print(f"PCA feature_map - actual_rows: {actual_rows}, n_components: {self.n_components}")

        A_tensor = torch.zeros(souNum, actual_rows, self.n_components)

        bag_counts_list = []
        # # 修正bag_instance_tensor的构造方式
        # bag_instance_tensor = torch.zeros(souNum, bagNum, actual_rows)
        bag_col = B[0].columns[0]
        bag_samples_dict, bag_counts_list, bag_instance_tensor = self.extract_bag_samples(B, bag_col)

        for k in range(souNum):
            dk = attList[k]
            data_k = B[k].iloc[:, 2:dk + 2].values
            # print(f"Source {k} - data_k shape: {data_k.shape}")

            scaler = StandardScaler()
            data_k = scaler.fit_transform(data_k)

            max_comp = min(data_k.shape[0], data_k.shape[1])
            n_comp = min(self.n_components, max_comp)
            # print(f"Source {k} - max_comp: {max_comp}, n_comp: {n_comp}")

            pca = PCA(n_components=n_comp)
            data_k_pca = pca.fit_transform(data_k)
            # print(f"Source {k} - data_k_pca shape: {data_k_pca.shape}")

            if n_comp < self.n_components:
                pad_width = self.n_components - n_comp
                data_k_pca = np.pad(data_k_pca, ((0, 0), (0, pad_width)), mode='constant')
                # print(f"Source {k} - padded data_k_pca shape: {data_k_pca.shape}")

            self.pca_list.append(pca)
            A_tensor[k] = torch.from_numpy(data_k_pca).float()

            # # 记录 bag 数量
            # bag_counts_list.append(len(B[k][B[k].columns[0]].unique()))
            #
            # # 填充 bag_instance_tensor
            # bag_ids = B[k][B[k].columns[0]].values
            # unique_bags = np.unique(bag_ids)
            # for i, bag_id in enumerate(bag_ids):
            #     bag_idx = np.where(unique_bags == bag_id)[0][0]
            #     bag_instance_tensor[k, i, bag_idx] = 1

        # print(f"PCA feature_map - A_tensor final shape V4: {A_tensor.shape}")
        # print(f"PCA feature_map - bag_instance_tensor final shape: {bag_instance_tensor.shape}")
        return A_tensor, bag_instance_tensor, bag_counts_list

    def train(self, dim, B, Y):
        if isinstance(Y, list):
            Y = np.array(Y)
        Y_tensor = torch.from_numpy(Y).float() if isinstance(Y, np.ndarray) else Y.float()

        # 正确接收三个返回值
        A_tensor, R1, bag_count = self.feature_map(dim, B)
        A_ins = torch.sum(A_tensor, dim=0)  # 聚合多源特征

        AT = A_ins.t()
        ATA = torch.matmul(AT, A_ins)
        I = torch.eye(ATA.shape[0]).to(ATA.device)
        ATA_reg = ATA + 1e-3 * I
        ATY = torch.matmul(AT, Y_tensor[0])

        ATA_np = ATA_reg.detach().cpu().numpy()
        ATY_np = ATY.detach().cpu().numpy()
        Wout_np = np.linalg.solve(ATA_np, ATY_np)
        self.Wout = torch.from_numpy(Wout_np).float()

        return self.Wout, A_tensor

    def predict(self, X, dim_test):
        souNum, attList = dim_test[3], dim_test[4]
        # 使用测试数据的实际行数
        actual_rows = X[0].shape[0]
        A_tensor = torch.zeros(souNum, actual_rows, self.n_components)

        for k in range(souNum):
            dk = attList[k]
            data_k = X[k].iloc[:, 2:dk+2].values
            scaler = StandardScaler()
            data_k = scaler.fit_transform(data_k)

            pca = self.pca_list[k]
            data_k_pca = pca.transform(data_k)
            A_tensor[k] = torch.from_numpy(data_k_pca).float()

        A = torch.sum(A_tensor, dim=0)
        return torch.matmul(A, self.Wout)
