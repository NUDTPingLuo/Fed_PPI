from ppi_py import *
from statsmodels.stats.weightstats import _zconfint_generic
import pandas as pd
import numpy as np
from examples import utils
import os

# 为各节点上分配不同比例的样本
def split_array_by_ratio(arr, ratios):
    total = sum(ratios)
    lengths = [int(len(arr) * ratio / total) for ratio in ratios]

    # Adjust the lengths to make sure the sum matches the original array length
    diff = len(arr) - sum(lengths)
    for i in range(diff):
        lengths[i % len(lengths)] += 1

    return np.array_split(arr, np.cumsum(lengths)[:-1])

# 将数组索性分为两份
def split_array(arr):
    length = len(arr)
    mid = length // 2
    if length % 2 != 0:
        mid += 1
    first_half_indices = arr[0:mid]
    second_half_indices = arr[mid: length]
    return mid, first_half_indices, second_half_indices

def analyze_dataset(alpha, X_total, Y_total, Yhat_total, dataset_dist, num_ratio, method, grid):
    # 为了确保结果的可重复性
    np.random.seed(100)
    is_round = False   # 是否把lambda参数的变化制表
    # 下载并加载数据集

    if dataset_dist == 'IID':

        is_iid = True
    else:
        is_iid = False

    if is_round:
        labeled_ratio_arr = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    else:
        labeled_ratio_arr = [0.0018]  # 每份有标签数据点的数量占比

    # 设置推断问题的参数
    coordinate = 0  # Choose between 0, 1
    alternative = "two-sided"
    q = 0.5  # 中位数
    for j in range(len(labeled_ratio_arr)):
        labeled_ratio = labeled_ratio_arr[j]
        if is_iid:
            # 随机打乱数据索引
            rand_idx = np.random.permutation(Y_total.shape[0])
            split_indices = split_array_by_ratio(rand_idx, num_ratio)
        else:
            # 将Yhat平均分成两份
            mid, Yhat_total0, Yhat_total1 = split_array(Yhat_total)
            # 前一半iid后一半non-iid
            rand_idx = np.random.permutation(Yhat_total0.shape[0])
            sorted_idx = mid + np.argsort(Yhat_total1)
            # 决定是case2和case3的开关
            # all_idx = np.concatenate((rand_idx, sorted_idx))  # case3
            all_idx = np.argsort(Yhat_total)  # case2

            split_indices = split_array_by_ratio(all_idx, num_ratio)
            # 输出每个分割的Y值范围
            for i, indices in enumerate(split_indices):
                min_Y = Yhat_total[indices[0]]
                max_Y = Yhat_total[indices[-1]]
                print(f"Split {i+1}: Y range from {min_Y} to {max_Y}")

        Yhat_labeled_all = []
        Y_labeled_all = []
        Yhat_unlabeled_all = []
        X_labeled_all = []
        X_unlabeled_all = []
        Yhat_unlabeled_mean_all = []

        theta_values = []
        cpp_intervals = []

        # 存储预估点和方差
        ppi_pointest_all = []
        ppi_pointest_all_q = []
        imputed_var_all = []
        rectifier_var_all = []
        datasize_i_all = []
        Sigma_hat_all = []
        var_unlabeled_all = []
        var_labeled_all = []
        inv_hessian_all = []

        # 循环处理每一份数据
        for i, indices in enumerate(split_indices):
            # 在当前份中随机选择有标签数据
            labeled_num = int(np.floor(labeled_ratio * len(indices)))
            labeled_indices = np.random.choice(indices, labeled_num, replace=False)
            unlabeled_indices = np.setdiff1d(indices, labeled_indices)

            # 有标签数据
            Yhat_labeled = Yhat_total[labeled_indices]
            Y_labeled = Y_total[labeled_indices]
            if method == 'logistic' or method == 'linear':
                X_labeled = X_total[labeled_indices]
            print('分组：', i+1)
            print('带标签的样本量：', len(Y_labeled))

            # 无标签数据
            Yhat_unlabeled = np.array(Yhat_total[unlabeled_indices])
            if method == 'logistic' or method == 'linear':
                X_unlabeled = np.array(X_total[unlabeled_indices])
            print('不带标签的样本量：', len(Yhat_unlabeled))

            # 存储每份数据
            Yhat_labeled_all.append(Yhat_labeled)
            Y_labeled_all.append(Y_labeled)
            Yhat_unlabeled_all.append(Yhat_unlabeled)
            Yhat_unlabeled_mean_all.append(Yhat_unlabeled.mean())
            if method == 'logistic' or method == 'linear':
                X_labeled_all.append(X_labeled)
                X_unlabeled_all.append(X_unlabeled)

            # 存储样本大小
            datasize_i = len(Y_labeled) + len(Yhat_unlabeled)
            datasize_i_all.append(datasize_i)

            # 计算预测支持的置信区间
            if method == 'mean':
                ppi_ci = np.array(ppi_mean_ci(Y_labeled, Yhat_labeled, Yhat_unlabeled, alpha=alpha))
                ppi_ci = ppi_ci.reshape(1, 2)[0]
                ppi_pointest, imputed_var, rectifier_var = ppi_mean_ci_FL(Y_labeled, Yhat_labeled, Yhat_unlabeled, alpha=alpha)
                ppi_pointest_q = [0]
            elif method == 'quantile':
                ppi_ci = np.array(ppi_quantile_ci(Y_labeled, Yhat_labeled, Yhat_unlabeled, q, alpha=alpha))
                ppi_ci = ppi_ci.reshape(1, 2)[0]
                ppi_pointest, ppi_pointest_q, imputed_var, rectifier_var = ppi_quantile_ci_FL(Y_labeled, Yhat_labeled, Yhat_unlabeled, q, grid, alpha=alpha)
            elif method == 'logistic':
                optimizer_options = {
                    "ftol": 1e-5,
                    "gtol": 1e-5,
                    "maxls": 10000,
                    "maxiter": 10000,
                }
                ppi_ci = ppi_logistic_ci(
                    X_labeled,
                    Y_labeled,
                    Yhat_labeled,
                    X_unlabeled,
                    Yhat_unlabeled,
                    alpha=alpha,
                    optimizer_options=optimizer_options,
                )
                ppi_ci = np.array(ppi_ci)[:, coordinate]
                ppi_pointest, var_unlabeled, var_labeled, inv_hessian = ppi_logistic_ci_FL(
                    X_labeled,
                    Y_labeled,
                    Yhat_labeled,
                    X_unlabeled,
                    Yhat_unlabeled,
                    alpha=alpha,
                    optimizer_options=optimizer_options,
                )
                ppi_pointest = [ppi_pointest[coordinate]]
                ppi_pointest_q = [0]
                imputed_var = [0]
                rectifier_var = [0]
            elif method == 'linear':
                ppi_ci = ppi_ols_ci(X_labeled,
                    Y_labeled,
                    Yhat_labeled,
                    X_unlabeled,
                    Yhat_unlabeled,
                    alpha=alpha)
                ppi_ci = np.array(ppi_ci)[:, coordinate]
                ppi_pointest, var_unlabeled, var_labeled, inv_hessian = ppi_ols_ci_FL(X_labeled,
                    Y_labeled,
                    Yhat_labeled,
                    X_unlabeled,
                    Yhat_unlabeled,
                    alpha=alpha)
                ppi_pointest = [ppi_pointest[coordinate]]
                ppi_pointest_q = [0]
                imputed_var = [0]
                rectifier_var = [0]

            # 存储每份预估点和方差
            ppi_pointest_all.append(ppi_pointest)
            ppi_pointest_all_q.append(ppi_pointest_q)
            imputed_var_all.append(imputed_var)
            rectifier_var_all.append(rectifier_var)
            if method == 'logistic' or method == 'linear':
                var_unlabeled_all.append(var_unlabeled)
                var_labeled_all.append(var_labeled)
                inv_hessian_all.append(inv_hessian)

            # 存储每份数据的均值和置信区间
            theta_values.append(ppi_pointest)
            cpp_intervals.append(ppi_ci)

            # 打印每份数据的结果
            # print(f"\n分组 {i+1}: 预测 theta={ppi_pointest}, CPP={ppi_ci}")
            # print(f"\n估算方差 var={imputed_var}, 调整方差 var={rectifier_var}")

        total_datasize = np.sum(datasize_i_all)
        proportions = datasize_i_all / total_datasize

        # 计算平均theta值
        ppi_pointest_all = np.array(ppi_pointest_all)
        ppi_pointest_all_q = np.array(ppi_pointest_all_q)
        imputed_var_all = np.array(imputed_var_all)
        rectifier_var_all = np.array(rectifier_var_all)

        ppi_pointest_mean = np.dot(proportions, ppi_pointest_all)
        ppi_pointest_mean_q = np.dot(proportions, ppi_pointest_all_q)

        # 计算样本平均值
        Yhat_labeled_mean_all = [np.mean(sub_array) for sub_array in Yhat_labeled_all]
        Y_labeled_mean_all = [np.mean(sub_array) for sub_array in Y_labeled_all]
        n = labeled_ratio * len(Y_total)
        N = (1 - labeled_ratio) * len(Y_total)

        rectifier_mean_all = np.subtract(Yhat_labeled_mean_all, Y_labeled_mean_all)
        # print("各节点上纠正偏差平均值:", rectifier_mean_all)
        if method == 'mean' or method == 'quantile':
            _, imputed_var_mean_all = combine_var(Yhat_unlabeled_mean_all, datasize_i_all, imputed_var_all)
            imputed_var_mean_all = imputed_var_mean_all / ((1-labeled_ratio) * len(Y_total))
            print("imputed var:", imputed_var_mean_all)
            _, rectifier_var_mean_all = combine_var(rectifier_mean_all, datasize_i_all, rectifier_var_all)
            rectifier_var_mean_all = rectifier_var_mean_all / (labeled_ratio * len(Y_total))
            print('rectifier var', rectifier_var_mean_all)
        if method == 'logistic' or method == 'linear':
            _, var_labeled_mean_all = combine_var(rectifier_mean_all, datasize_i_all, var_labeled_all)
            _, var_unlabeled_mean_all = combine_var(Yhat_unlabeled_mean_all, datasize_i_all, var_unlabeled_all)
            inv_hessian_mean_all, _ = combine_var(inv_hessian_all, datasize_i_all, var_unlabeled_all)
            Sigma_hat = inv_hessian_mean_all @ (n / N * var_unlabeled_mean_all + var_labeled_mean_all) @ inv_hessian_mean_all

        # 计算五份数据置信区间的FL聚合
        if method == 'mean':
            mean_cpp = _zconfint_generic(
                ppi_pointest_mean,
                np.sqrt(imputed_var_mean_all + rectifier_var_mean_all),
                alpha,
                alternative,
            )
            mean_cpp = np.array(mean_cpp).reshape(1, 2)[0]
        elif method == 'quantile':
            rec_p_value = rectified_p_value(
                ppi_pointest_mean_q,
                np.sqrt(rectifier_var_mean_all),
                ppi_pointest_mean,
                np.sqrt(imputed_var_mean_all),
                null=q,
                alternative="two-sided",
            )
            mean_cpp = grid[rec_p_value > alpha][[0, -1]]
        elif method == 'logistic' or method == 'linear':
            if method == 'logistic':
                ppi_pointest_mean = ppi_logistic_pointestimate_FL(
                    X_labeled_all,
                    Y_labeled_all,
                    Yhat_labeled_all,
                    X_unlabeled_all,
                    Yhat_unlabeled_all,
                )
            mean_cpp = _zconfint_generic(
                ppi_pointest_mean,
                np.sqrt(np.diag(Sigma_hat) / n),
                alpha=alpha,
                alternative=alternative,
            )
            mean_cpp = np.array(mean_cpp)[:, coordinate]
        # 打印不直接组合数据的theta值和置信区间
        # print(f"\n联邦学习后 theta={ppi_pointest_mean}, 平均 CPP={mean_cpp}")
        # print(f"\n联邦学习后估算方差 var={imputed_var_mean_all}, 调整方差 var={rectifier_var_mean_all}")

        Yhat_labeled_combined = np.concatenate(Yhat_labeled_all)
        Y_labeled_combined = np.concatenate(Y_labeled_all)
        Yhat_unlabeled_combined = np.concatenate(Yhat_unlabeled_all)
        if method == 'logistic' or method == 'linear':
            X_labeled_combined = np.concatenate(X_labeled_all)
            X_unlabeled_combined = np.concatenate(X_unlabeled_all)

        print('带标签的样本量：', len(Y_labeled_combined))
        print('不带标签的样本量：', len(Yhat_unlabeled_combined))

        # 计算新的组合数据集的预测支持的置信区间
        if method == 'mean':
            ppi_ci_combined = ppi_mean_ci(Y_labeled_combined, Yhat_labeled_combined, Yhat_unlabeled_combined, alpha=alpha)
            ppi_ci_combined = np.array(ppi_ci_combined).reshape(1, 2)[0]
            ppi_pointest_combined, imputed_var_combined, rectifier_var_combined = ppi_mean_ci_FL(Y_labeled_combined, Yhat_labeled_combined, Yhat_unlabeled_combined, alpha=alpha)
        if method == 'quantile':
            ppi_ci_combined = ppi_quantile_ci(Y_labeled_combined, Yhat_labeled_combined, Yhat_unlabeled_combined, q, alpha=alpha)
            ppi_ci_combined = np.array(ppi_ci_combined).reshape(1, 2)[0]
        if method == 'logistic':
            ppi_ci_combined = ppi_logistic_ci(
                X_labeled_combined,
                Y_labeled_combined,
                Yhat_labeled_combined,
                X_unlabeled_combined,
                Yhat_unlabeled_combined,
                alpha=alpha,
                optimizer_options=optimizer_options,
            )
            ppi_ci_combined = np.array(ppi_ci_combined)[:, coordinate]
        if method == 'linear':
            ppi_ci_combined = ppi_ols_ci(X_labeled_combined,
                Y_labeled_combined,
                Yhat_labeled_combined,
                X_unlabeled_combined,
                Yhat_unlabeled_combined,
                alpha=alpha)
            ppi_ci_combined = np.array(ppi_ci_combined)[:, coordinate]
        # 打印新的组合数据集的平均theta值和置信区间
        # print(f"\n组合数据后 theta={ppi_pointest_combined}, CPP={ppi_ci_combined}")
        # print(f"\n组合数据后估算方差 var={imputed_var_combined}, 调整方差 var={rectifier_var_combined}")

        # 计算真实值
        if method == 'mean':
            true_theta = Y_total.mean()
        elif method == 'quantile':
            true_theta = np.quantile(Y_total, q)
        elif method == 'logistic':
            true_theta = (
                LogisticRegression(
                    penalty="none",
                    solver="lbfgs",
                    max_iter=10000,
                    tol=1e-15,
                    fit_intercept=False,
                )
                .fit(X_total, Y_total)
                .coef_.squeeze()
            )[coordinate]
        elif method == 'linear':
            # true_theta = OLS(Y_total, exog=X_total).fit().params[coordinate]
            true_theta = ppi_ols_true_point(X_total, Y_total)[coordinate]
        # print(f"\n真实 theta={true_theta}")
        print("\n最终结果：")
        print(f"真实 theta: {true_theta}")
        print(f"CPP intervals: {cpp_intervals}")
        print(f"组合数据的置信区间: {ppi_ci_combined}")
        print(f"联邦聚合后的置信区间: {mean_cpp}")

        # 文件名
        filename = 'parameters.csv'

        # 新的数组数据
        new_data = {
            'centralized_lower': [ppi_ci_combined[0]],
            'centralized_upper': [ppi_ci_combined[1]],
            'fl_lower': [mean_cpp[0]],
            'fl_upper': [mean_cpp[1]]
        }

        # 检查参数文件是否存在
        if os.path.exists(filename):
            # 如果文件存在，读取现有数据
            df = pd.read_csv(filename)
        else:
            # 如果文件不存在，创建一个空的 DataFrame
            df = pd.DataFrame(columns=['centralized_lower', 'centralized_upper', 'fl_lower', 'fl_upper'])

        # 将新数据添加到 DataFrame
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df, new_df], ignore_index=True)

        # 保存更新后的数据到CSV文件
        df.to_csv(filename, index=False)

        print(f"Updated CSV file '{filename}' successfully.")

    return true_theta, cpp_intervals, ppi_ci_combined, mean_cpp

def plot_cpp(true_theta, cpp_intervals, ppi_ci_combined, mean_cpp, file_name, xlim, ylim, title):
    num_splits = 5
    # 创建一个包含结果的 DataFrame
    results = []

    # 各个分组的置信区间
    for i, cpp in enumerate(cpp_intervals):
        # cpp = np.array(cpp)
        results.append(pd.DataFrame([{
            'method': 'Node',
            'n': i,
            'lower': cpp[0],
            'upper': cpp[1],
            'trial': 0
        }]))

    # 组合数据的置信区间
    results.append(pd.DataFrame([{
        'method': 'Combined',
        'n': num_splits,
        'lower': ppi_ci_combined[0],
        'upper': ppi_ci_combined[1],
        'trial': 0
    }]))

    # 五份数据的平均置信区间
    results.append(pd.DataFrame([{
        'method': 'FL Aggregation',
        'n': num_splits + 1,
        'lower': mean_cpp[0],
        'upper': mean_cpp[1],
        'trial': 0
    }]))

    # 合并所有结果
    df = pd.concat(results, axis=0, ignore_index=True)
    df['width'] = df['upper'] - df['lower']

    # 绘制图形
    utils.make_plots_fl(
        df,
        file_name,  # 保存路径
        xlim,
        ylim,
        n_idx=num_splits,  # 对应的分组数量
        intervals_xlabel=title,
        true_theta=true_theta
    )

