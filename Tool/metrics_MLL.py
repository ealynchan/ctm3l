import numpy as np
from sklearn.metrics import label_ranking_average_precision_score, label_ranking_loss, hamming_loss, roc_auc_score

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
    scorce = []
    ap = label_ranking_average_precision_score(y_test, y_score)
    # print('ap: %.4f' % ap, end=', ')
    auc = roc_auc_score(y_test, y_score, average='micro')
    # auc = safe_multiclass_auc(y_test, y_score, average='macro')
    rl = label_ranking_loss(y_test, y_score)
    # print('rl: %.4f' % rl, end=', ')
    hl = hamming_loss(y_test, y_pred)
    # print('hl: %.4f' % hl)

    scorce.append(ap)
    scorce.append(auc)
    scorce.append(rl)
    scorce.append(hl)
    return np.round(scorce, 4)