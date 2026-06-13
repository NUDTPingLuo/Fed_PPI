import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_summary_from_csv(dataset_name, distribution='IID', alpha_dir=None, fixed_acc=None):
    """
    全自动自适应 PPI-FL 置信区间演进趋势绘图系统
    """
    # ==================================================
    # 1. 自动路由数据路径与读取
    # ==================================================
    csv_root = os.path.join('.', 'result', dataset_name, 'csv')

    # 🛠️ 核心修改：重新校准 X 轴含义与标签名称
    if fixed_acc is not None:
        csv_path = os.path.join(csv_root, f"{dataset_name}_{distribution}_ratio_summary.csv")
        x_col = 'labeled_ratio'
        x_label = r'$\lambda$'  # 👈 标签比例采用 \lambda 符号
        acc_col = 'fixed_accuracy'
        title_suffix = f"(Fixed Acc = {fixed_acc})"
    else:
        csv_path = os.path.join(csv_root, f"{dataset_name}_{distribution}_summary.csv")
        if not os.path.exists(csv_path) and os.path.exists(f"{dataset_name}_{distribution}_summary.csv"):
            csv_path = f"{dataset_name}_{distribution}_summary.csv"

        x_col = 'target_accuracy'
        x_label = 'Acc'  # 👈 模型准确率采用 Acc 单词
        acc_col = 'target_accuracy'
        title_suffix = r"(Fixed $\lambda = 0.3$)"  # 👈 固定的比率用 \lambda 表示

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 找不到对应的数据汇总总表，请检查路径: {csv_path}")

    df = pd.read_csv(csv_path)

    # ==================================================
    # 2. 动态清洗与条件过滤
    # ==================================================
    df_filtered = df[df['distribution'] == distribution].copy()

    # 🛠️ 核心修改：全面剔除 "Dirichlet" 单词，改用纯数学符号表达
    if distribution == 'Non-IID':
        if alpha_dir is None:
            raise ValueError("❌ 运行 Non-IID 绘图时，必须指定 alpha_dir 参数！")
        df_filtered = df_filtered[df_filtered['alpha_dir'].astype(str) == str(alpha_dir)]
        title_dist = f"$\\alpha_{{Dir}} = {alpha_dir}$"
    else:
        title_dist = "IID Distribution"

    if fixed_acc is not None:
        df_filtered = df_filtered[df_filtered[acc_col].astype(str) == str(fixed_acc)]

    if df_filtered.empty:
        print(f"⚠️ [警告] 按照当前过滤条件，表格内未筛选出任何有效数据，请检查输入！")
        return

    df_filtered[x_col] = df_filtered[x_col].astype(float)
    df_filtered = df_filtered.sort_values(by=x_col)

    x_values = df_filtered[x_col].values
    centralized_low = df_filtered['ppi_ci_lower'].values
    centralized_upper = df_filtered['ppi_ci_upper'].values
    fl_low = df_filtered['mean_cpp_lower'].values
    fl_upper = df_filtered['mean_cpp_upper'].values

    # ==================================================
    # 3. 置信区间趋势图渲染
    # ==================================================
    fig, ax = plt.subplots(figsize=(5, 4.5))

    ax.plot(x_values, centralized_upper, label='Centralized Upper', color='#F3D266', marker='o')
    ax.plot(x_values, centralized_low, label='Centralized Lower', color='#F3D266', linestyle='-', marker='o',
            markerfacecolor='w')

    ax.plot(x_values, fl_upper, label='FL Upper', color='#96C37D', marker='s')
    ax.plot(x_values, fl_low, label='FL Lower', color='#96C37D', linestyle='-', marker='s', markerfacecolor='w')

    ax.fill_between(x_values, fl_upper, centralized_upper, color='#F3D266', alpha=0.3)
    ax.fill_between(x_values, fl_low, centralized_low, color='#96C37D', alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 🛠️ 核心修改：大幅放宽 labelpad=18，防止总结图中的科学记数法标志发生交叉重叠
    ax.set_xlabel(x_label, fontsize=12, labelpad=18)
    ax.set_ylabel('Confidence Interval Bounds', fontsize=12, labelpad=10)
    ax.set_title(f"{dataset_name.capitalize()} ({title_dist})\n{title_suffix}", fontsize=13, pad=12)

    all_y_values = np.concatenate([centralized_low, centralized_upper, fl_low, fl_upper])
    y_min, y_max = all_y_values.min(), all_y_values.max()
    y_pad = (y_max - y_min) * 0.1 if y_max != y_min else 0.01
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.legend(frameon=False, loc='best')
    plt.tight_layout()

    # ==================================================
    # 4. 分类分流落盘管理
    # ==================================================
    pdf_save_dir = os.path.join('.', 'result', dataset_name, 'pdf', 'summary_plots')
    os.makedirs(pdf_save_dir, exist_ok=True)

    if fixed_acc is not None:
        fig_name = f"Summary-Ratio_Curve-{distribution}-alpha_{alpha_dir if alpha_dir else 'None'}.pdf"
    else:
        fig_name = f"Summary-Accuracy_Curve-{distribution}-alpha_{alpha_dir if alpha_dir else 'None'}.pdf"

    full_pdf_path = os.path.join(pdf_save_dir, fig_name)
    plt.savefig(full_pdf_path, dpi=600)
    print(f"🎉 [绘图成功] 总结趋势图已完美输出至: {full_pdf_path}")
    plt.close()

# ==================================================
# 顺序调用全部三个实验场景
# ==================================================
if __name__ == "__main__":
    # 🎯 场景 A：看 IID 下准确率从 0.1~1.0 变化的曲线
    plot_summary_from_csv(dataset_name='diabetes-bmi', distribution='IID')

    # 🎯 场景 B：看 Non-IID (alpha=0.01) 下准确率从 0.1~1.0 变化的曲线
    plot_summary_from_csv(dataset_name='diabetes-bmi', distribution='Non-IID', alpha_dir=0.01)

    # 🎯 场景 C：看固定 50% 准确率和IID下，有标签比例 0.1~0.5 演进的趋势曲线
    plot_summary_from_csv(dataset_name='diabetes-bmi', distribution='IID', fixed_acc=0.5)

    # 🎯 场景 D：看固定 50% 准确率和Non-IID(alpha=0.01)下，有标签比例 0.1~0.5 演进的趋势曲线
    plot_summary_from_csv(dataset_name='diabetes-bmi', distribution='Non-IID', alpha_dir=0.01, fixed_acc=0.5)