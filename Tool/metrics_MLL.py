# 多标签分类的评价指标
# Zhang M L, Zhou Z H. A review on multi-label learning algorithms[J]. IEEE transactions on knowledge and data engineering, 2013, 26(8): 1819-1837.
# Wu X Z, Zhou Z H. A unified view of multi-label performance measures[C]//international conference on machine learning. PMLR, 2017: 3780-3788.
# Mining Multi-label Data

import numpy as np
import scipy.io
from sklearn.metrics import coverage_error, label_ranking_average_precision_score, label_ranking_loss, hamming_loss, zero_one_loss, roc_auc_score, f1_score

# 多标签评价指标, 基于实例的指标
# 指标1：hamming_loss(HL)--hamming_loss, 指标2：One-Error(OE)--zero_one_loss
# 指标3：label_ranking_average_precision_score(LRAP or AP)--label_ranking_average_precision_score,
# 指标4：coverage_error(CV)--coverage_error, 指标5：label_ranking_loss(RL)--label_ranking_loss

from sklearn.metrics import jaccard_score

def compute_consistency_score(pred_views, average='samples'):
    """
    pred_views: list of [N x L] binary arrays, length = V
    Returns: scalar consistency score (higher is better)
    """
    V = len(pred_views)
    N, L = pred_views[0].shape
    total_score = 0.0
    count = 0

    for i in range(N):
        view_preds = [pred[i] for pred in pred_views]  # list of (L,) vectors
        pairwise_scores = []
        for a in range(V):
            for b in range(a + 1, V):
                if average == 'samples':
                    # Handle all-zero case: if both zero, Jaccard = 1
                    if np.sum(view_preds[a]) == 0 and np.sum(view_preds[b]) == 0:
                        jac = 1.0
                    else:
                        jac = jaccard_score(view_preds[a], view_preds[b])
                    pairwise_scores.append(jac)
        if pairwise_scores:
            total_score += np.mean(pairwise_scores)
            count += 1

    return total_score / count if count > 0 else 1.0

def safe_multiclass_auc(y_true, y_score, average='macro', handle_case='warn'):
    """
    安全计算多分类 ROC AUC，跳过只含一个类别的二分类子问题。

    Parameters:
    - y_true: (n_samples,) 真实标签
    - y_score: (n_samples, n_classes) 预测概率或置信度分数
    - average: 'macro', 'weighted', or None
    - handle_case: 'warn', 'ignore', 'raise'

    Returns:
    - auc_score: float (平均 AUC) 或 list (每类 AUC)
    """
    n_classes = y_score.shape[1]
    aucs = []

    for c in range(n_classes):
        y_true_c = (y_true == c).astype(int)  # One-vs-Rest 编码
        y_score_c = y_score[:, c]

        # 检查是否至少有两个类别
        if len(np.unique(y_true_c)) < 2:
            msg = f"类别 {c}: y_true 中只包含一个类别（全是 {np.unique(y_true_c)[0]}），跳过 AUC 计算。"
            if handle_case == 'warn':
                print("⚠️  " + msg)
            elif handle_case == 'raise':
                raise ValueError(msg)
            aucs.append(np.nan)
        else:
            auc = roc_auc_score(y_true_c, y_score_c)
            aucs.append(auc)

    # 返回每类 AUC 或平均值
    if average is None:
        return aucs

    if average == 'macro':
        return np.nanmean(aucs)
    elif average == 'weighted':
        class_counts = np.bincount(y_true, minlength=n_classes)
        weights = class_counts / len(y_true)
        return np.nansum(np.array(aucs) * weights)
    else:
        raise ValueError("average must be 'macro', 'weighted', or None")


# === 使用示例 ===
# y_true = np.array([0, 1, 2, 1, 0])
# y_score = np.random.rand(5, 3)  # 模拟预测分数

# auc = safe_multiclass_auc(y_true, y_score, average='macro')
# print(f"Macro AUC: {auc:.3f}")

from sklearn.metrics import jaccard_score


def compute_consistency_score_binary(y_pred, y_pred_view):
    """
    Compute Consistency Score (CS) from binary predictions.

    Parameters:
    ----------
    y_pred : array-like, shape (N, L)
        Final fused binary prediction (0/1).
    y_pred_view : list of array-like, each (N, L)
        Binary predictions from each view (0/1).

    Returns:
    -------
    cs : float
        Consistency Score in [0, 1]. Higher is better.
    """
    # 转为 NumPy（兼容 PyTorch CPU tensor）
    y_pred = np.asarray(y_pred)
    y_pred_views = [np.asarray(pred) for pred in y_pred_view]

    N, L = y_pred.shape
    V = len(y_pred_views)

    # 添加边界检查
    if N == 0 or V == 0:
        return 0.0

    total_jac = 0.0
    total_pairs = N * V

    # 添加额外的安全检查
    if total_pairs == 0:
        return 0.0

    for i in range(N):
        for v in range(V):
            a = y_pred[i]
            b = y_pred_views[v][i]

            # 处理全零情况：两者都无标签 → 完全一致
            if np.sum(a) == 0 and np.sum(b) == 0:
                jac = 1.0
            else:
                jac = jaccard_score(a, b)

            total_jac += jac

    return total_jac / total_pairs


def mll_metrics(y_test, y_pred, y_score, y_pred_view):
    # 计算多标签分类的评价指标：label ranking loss, hamming loss, one error, coverage error, average precision
    # 输入：测试样本的真实标签y_test, 测试样本的预测标签y_pred
    # 输出：测试样本的各项指标
    scorce = []
    # print('y_test:', y_test, y_test.shape)
    # print('y_score:', y_score, y_score.shape)
    oe = zero_one_loss(y_test, y_pred)
    cv = coverage_error(y_test, y_score)
    # print('cv: %.4f' % cv, end=', ')
    ap = label_ranking_average_precision_score(y_test, y_score)
    # print('ap: %.4f' % ap, end=', ')
    f1 = f1_score(y_test, y_pred, average='macro')  # 是精确率和召回率的调和平均值，用于综合考虑模型的准确性和覆盖率。取值范围在 0 到 1 之间，越接近 1 表示模型的性能越好。
    # print('f1: %.4f' % f1, end=', ')
    auc = roc_auc_score(y_test, y_score, average='micro')
    # auc = safe_multiclass_auc(y_test, y_score, average='macro')
    # print('AUC: %.4f' % auc, end=', ')
    rl = label_ranking_loss(y_test, y_score)
    # print('rl: %.4f' % rl, end=', ')
    hl = hamming_loss(y_test, y_pred)
    # print('hl: %.4f' % hl)
    cs = compute_consistency_score_binary(y_pred, y_pred_view)

    scorce.append(ap)
    scorce.append(f1)
    scorce.append(auc)
    scorce.append(rl)
    scorce.append(hl)
    scorce.append(cs)
    scorce.append(oe)
    scorce.append(cv)
    # ap, f1, auc, rl, hl, cs, oe, cv

    return np.round(scorce, 4)


def myHamming_loss(y_test, y_pred):
    # Hamming_loss: 它评估被错误分类的标签对之间的差异, 或者在每个实例中，基础真值标签和预测标签之间不一致的平均比率
    # HL = 1/tq * \sum_{i=1}^t |y_i Δ h_i|,
    # 集合的异或运算: AΔB = (A-B)∪(B-A) = (A∪B)-(A∩B), 对称差Δ用于计算两个向量的不一致
    # 其中 t:测试样本数量, q: 标签数量, h_i: 第i个测试样本的预测标签向量, y_i: 第i个测试样本的实际标签向量
    # 输入：测试样本的真实标签y_test, 测试样本的预测标签y_pred
    # 输出：汉明损失hamLos
    sym_diff = np.count_nonzero(y_test - y_pred)  # 计算\sum_{i=1}^t |y_i Δ h_i|
    hamLos = sym_diff / (len(y_test) * len(y_test[0]))
    return hamLos

def myOne_error():
    # One_error: 它计算其基础真值标签yi不包含预测标签hi的顶级标签的实例的比例
    # OE = 1/t * \sum_{i=1}^t <argmin_{l∈L} rank_f(x_i, l)∉yi>
    # OE =
    # <x> = 1 if the predicate x holds, and 0 otherwise
    # 排序函数rank_f(·,·): if f(xi, lj)<f(xi, lk), then rank_f(xi, lj)>rank_f(xi, lk)
    g = 0

# def myRanking_loss(y_test, y_pred):
#     # Ranking_loss: 它计算逆序标签对的分数, RL = 1/t * \sum_{i=1}^t 1/D_i



# y_test = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
# y_pred = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 1]])
# y_out = np.array([[0.5, 1, 1], [1, 0, 0], [0, 1, 1]])
# scores = mll_metrics(y_test, y_pred)
# print('rl, hl, oe, cv, ap:', scores)
#
# sym_diff = np.count_nonzero(y_test-y_pred)
# hamLos = sym_diff / (len(y_test) * len(y_test[0]))
# print(sym_diff, hamLos)
