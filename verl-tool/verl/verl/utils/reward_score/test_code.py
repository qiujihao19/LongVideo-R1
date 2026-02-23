
def interval_length(iv):
    return max(0.0, iv[1] - iv[0])

def intersection(a, b):
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)

def total_intersection_with_gt(visited_intervals, gt_intervals):
    # 计算 visited 与 GT 的总交集长度
    total = 0.0
    for v in visited_intervals:
        for g in gt_intervals:
            total += intersection(v, g)
    return total

def union_visited_length(visited_intervals):
    # 合并 visited_intervals 的长度（防重复计数）
    if not visited_intervals:
        return 0.0
    segs = sorted(visited_intervals, key=lambda x: x[0])
    merged = []
    cur_s, cur_e = segs[0]
    for s,e in segs[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return sum(interval_length(x) for x in merged)

def gt_total_length(gt_intervals):
    return sum(interval_length(g) for g in gt_intervals)


def compute_reward(
    gt_intervals,
    visited_intervals,
    steps,
    answer_correct,
    hierarchy_score=1.0,
    confidence=1.0,
    repeat_penalty_factor=0.1,
    weights=None
):
    # 默认权重
    if weights is None:
        weights = {
            'w_acc': 4.0,
            'w_f1': 1.5,
            'w_eff': 0.5,
            'w_hier': 0.3,
            'w_rep': -0.2
        }

    # 基础量
    gt_len = gt_total_length(gt_intervals)  # 可能为0（保护）
    inter_len = total_intersection_with_gt(visited_intervals, gt_intervals)
    visited_len = union_visited_length(visited_intervals)

    # Coverage
    cov = (inter_len / gt_len) if gt_len > 0 else 0.0
    # Precision
    prec = (inter_len / visited_len) if visited_len > 0 else 0.0
    # F1-style
    if cov + prec > 0:
        f1 = 2 * cov * prec / (cov + prec)
    else:
        f1 = 0.0

    # Efficiency
    eff = 1.0 / (1.0 + float(steps))  # e.g., steps=0 -> 1.0 ; steps=3 -> 0.25

    # Hierarchy score assumed in [0,1]
    hier = max(0.0, min(1.0, float(hierarchy_score)))

    # Repeat penalty: we approximate by counting extra visited length outside GT
    extra_len = max(0.0, visited_len - inter_len)
    # normalize repeat penalty by visited_len to get 0~1
    rep = - (extra_len / visited_len) * repeat_penalty_factor if visited_len > 0 else 0.0

    # Accuracy terminal reward (可以乘以置信度)
    acc = 1.0 * (1.0 if answer_correct else 0.0) * confidence

    # 线性组合
    R = (weights['w_acc'] * acc +
         weights['w_f1'] * f1 +
         weights['w_eff'] * eff +
         weights['w_hier'] * hier +
         weights['w_rep'] * rep)

    # 可选：归一化 / 裁剪到某区间
    # 例如裁剪到 [ -1, +6 ] 或者 [0,1]（若需要）
    return {
        'reward': R,
        'components': {
            'acc': acc,
            'f1': f1,
            'coverage': cov,
            'precision': prec,
            'efficiency': eff,
            'hier': hier,
            'repeat_penalty': rep,
            'visited_len': visited_len,
            'intersection_len': inter_len
        }
    }