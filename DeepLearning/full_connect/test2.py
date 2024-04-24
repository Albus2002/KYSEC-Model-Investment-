import numpy as np
from sklearn.metrics import r2_score
from concurrent.futures import ThreadPoolExecutor, as_completed

def compute_r2_score(pred, true):
    """计算单个标签组的 r2 score."""
    if len(pred) > 0 and len(true) > 0:
        return r2_score(true, pred)
    else:
        return None

def grouped_r2_score_multithreaded(output, target, labels, max_workers=32):
    # 按标签分组
    groups = {}
    for i, label in enumerate(labels):
        if np.isnan(output[i]) == False:
            if label not in groups:
                groups[label] = {'pred': [], 'true': []}
            groups[label]['pred'].append(output[i])
            groups[label]['true'].append(target[i])
    
    # 使用 ThreadPoolExecutor 来并行计算每个组的 r2 score
    scores = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_label = {executor.submit(compute_r2_score, groups[label]['pred'], groups[label]['true']): label for label in groups}
        for future in as_completed(future_to_label):
            score = future.result()
            if score is not None:
                scores.append(score)

    # 计算平均 r2 score
    if len(scores) > 0:
        return sum(scores) / len(scores)
    else:
        return None

# 示例数据
output = [1, 2, 3, 4]
target = [1.1, 1.9, 3.1, 4.1]
labels = [1,2,1,2]

# 调用函数
r2 = grouped_r2_score_multithreaded(output, target, labels)
print(f"平均 R2 Score: {r2}")
