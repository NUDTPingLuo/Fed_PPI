import os
import numpy as np

# ==================================================
# 1. 自动化提取数据集名称与动态目录创建
# ==================================================
original_file = 'diabetes-bmi.npz'  # 👈 以后换数据集只需要改这一个文件名即可！

dataset_name = os.path.splitext(original_file)[0]
os.makedirs(dataset_name, exist_ok=True)

# 2. 动态载入并解析所有内部键值
data = np.load(original_file)
all_keys = data.files
print(f"成功载入数据集 [{original_file}]，内部包含的键为: {all_keys}")

# 自动对齐 Y 和 Yhat / Y_hat 的键名
y_key = 'Y'
yhat_key = 'Y_hat' if 'Y_hat' in all_keys else 'Yhat'

if y_key not in all_keys or yhat_key not in all_keys:
    raise KeyError(f"数据集中必须包含标签 Y 和预测值(Yhat 或 Y_hat)！当前键: {all_keys}")

Y = data[y_key]
Y_hat = data[yhat_key]

# 💡 核心修复：动态收集所有“其他”伴随矩阵（如 X, phosphorylated 等），不进行硬编码
other_data = {key: data[key] for key in all_keys if key not in [y_key, yhat_key]}

# ==================================================
# 3. 自适应精准准确率控制算法（通用多分类/二分类/概率）
# ==================================================
unique_classes = np.unique(Y)
is_binary = set(unique_classes).issubset({0, 1, 0.0, 1.0})
# 判断是否为连续的概率输出（如 alphafold）还是离散整型标签（如 stellar, salary）
is_prob = is_binary and np.any((Y_hat > 0) & (Y_hat < 1))

# 计算原始情况下的对错掩码
if is_prob:
    is_correct = (Y == (Y_hat >= 0.5))
else:
    is_correct = (Y == Y_hat)


def set_exact_accuracy_universal(Y, Y_hat, target_acc, is_prob, is_binary, unique_classes):
    n = len(Y)
    target_correct = int(round(target_acc * n))

    indices = np.arange(n)
    np.random.seed(42)  # 固定随机种子保证结果可重复
    np.random.shuffle(indices)

    correct_indices = indices[:target_correct]
    incorrect_indices = indices[target_correct:]

    new_Y_hat = Y_hat.copy()

    # 1. 确保 correct_indices 内部全对
    to_fix_correct = correct_indices[~is_correct[correct_indices]]
    if is_prob:
        new_Y_hat[to_fix_correct] = 1.0 - new_Y_hat[to_fix_correct]
    else:
        new_Y_hat[to_fix_correct] = Y[to_fix_correct]

    # 2. 确保 incorrect_indices 内部全错
    to_fix_incorrect = incorrect_indices[is_correct[incorrect_indices]]
    if is_prob:
        new_Y_hat[to_fix_incorrect] = 1.0 - new_Y_hat[to_fix_incorrect]
    else:
        if is_binary:
            new_Y_hat[to_fix_incorrect] = 1 - Y[to_fix_incorrect]
        else:
            num_classes = int(unique_classes.max() + 1)
            new_Y_hat[to_fix_incorrect] = (Y[to_fix_incorrect] + 1) % num_classes

    return new_Y_hat


# ==================================================
# 4. 循环生成 10 个文件并安全存入目标文件夹
# ==================================================
print(f"正在为 [{dataset_name}] 生成 0.1~1.0 准确率文件...")
for acc in np.arange(0.1, 1.1, 0.1):
    acc_str = f"{acc:.1f}"
    filename = f"{dataset_name}_acc_{acc_str}.npz"
    filepath = os.path.join(dataset_name, filename)

    # 计算调整后的预测值
    adjusted_Y_hat = set_exact_accuracy_universal(Y, Y_hat, acc, is_prob, is_binary, unique_classes)

    # 动态组装保存字典
    save_dict = {y_key: Y, yhat_key: adjusted_Y_hat}
    save_dict.update(other_data)  # 把其他伴随矩阵自动塞回去

    # 原生打包保存
    np.savez(filepath, **save_dict)

    # 验证最终生成的实际准确率
    if is_prob:
        actual_acc = np.mean(Y == (adjusted_Y_hat >= 0.5))
    else:
        actual_acc = np.mean(Y == adjusted_Y_hat)

    print(f"  -> 成功生成: {filepath} | 实际验证准确率: {actual_acc:.4f}")

print(f"🎉 全部结束！所有文件已整齐保存在 ./{dataset_name}/ 文件夹下。")