import os
import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import seaborn as sns


def plot_interval(
        ax,
        lower,
        upper,
        height,
        color_face,
        color_stroke,
        linewidth=2,
        linewidth_modifier=1.1,
        offset=0.25,
        label=None,
):
    label = label if label is None else " " + label
    ax.plot(
        [lower, upper],
        [height, height],
        linewidth=linewidth,
        color=color_face,
        path_effects=[
            pe.Stroke(
                linewidth=linewidth * linewidth_modifier,
                offset=(-offset, 0),
                foreground=color_stroke,
            ),
            pe.Stroke(
                linewidth=linewidth * linewidth_modifier,
                offset=(offset, 0),
                foreground=color_stroke,
            ),
            pe.Normal(),
        ],
        label=label,
        solid_capstyle="butt",
    )


# 将数据集路由参数提升为显式必备形参，斩断一切全局变量依赖
def make_plots_fl(
        df,
        dataset_name,
        dataset_dist,
        acc,
        alpha_dir,
        xlim,
        ylim,
        n_idx=-1,
        true_theta=None,
        true_label="Ground truth",
        intervals_xlabel="x",
        plot_classical=True,
        ppi_facecolor="#2F7FC1",
        ppi_strokecolor="b",
        classical_facecolor="#96C37D",
        classical_strokecolor="g",
        imputation_facecolor="#F3D266",
        imputation_strokecolor="y",
        empty_panel=False,
        sub_folder=None,
):
    ns = df.n.unique()
    ns = ns[~np.isnan(ns)].astype(int)
    n = ns[n_idx]

    ppi_intervals = df[df.method == "Node"]
    num_intervals = len(ppi_intervals)
    if plot_classical:
        classical_interval = df[df.method == "FL Aggregation"]
    imputation_interval = df[df.method == "Combined"]

    if empty_panel:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    else:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3, 2.5))

    sns.set_theme(style="white", font_scale=1, font="DejaVu Sans")
    if true_theta is not None:
        axs.axvline(
            true_theta,
            ymin=0.0,
            ymax=1.0,
            linestyle="dotted",
            linewidth=2,
            label=true_label,
            color="#A9B8C6",
        )

    for i in range(num_intervals):
        ppi_interval = ppi_intervals.iloc[i]

        if i == 0:
            plot_interval(axs, ppi_interval.lower, ppi_interval.upper, 0.95, ppi_facecolor, ppi_strokecolor, label=r"Client 1-20")
            if plot_classical:
                plot_interval(axs, classical_interval.lower, classical_interval.upper, 0.15, classical_facecolor, classical_strokecolor, label=r"FL aggregation")
            plot_interval(axs, imputation_interval.lower, imputation_interval.upper, 0.075, imputation_facecolor, imputation_strokecolor, label=r"Centralized data")
        else:
            lighten_factor = 1.0 / np.sqrt(i)
            yshift = i * 0.04
            plot_interval(axs, ppi_interval.lower, ppi_interval.upper, 0.95 - yshift, lighten_color(ppi_facecolor, lighten_factor), lighten_color(ppi_strokecolor, lighten_factor))

    # 🛠️ 核心修改：将 labelpad 大幅提升至 18，彻底阻断 X 轴标签与科学记数法标志(如 1e-3)的物理重叠
    axs.set_xlabel(intervals_xlabel, fontsize=12, labelpad=18)
    axs.set_yticks([])
    axs.set_yticklabels([])
    axs.set_ylim(ylim)
    axs.set_xlim(xlim)

    # 强行启动 X 轴科学记数法
    axs.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    sns.despine(ax=axs, top=True, right=True, left=True)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.tight_layout()

    # ==================================================
    # 🌟 绝对物理安全分流：强约束拼装路径
    # ==================================================
    if sub_folder is not None:
        target_dir = os.path.join('.', 'result', dataset_name, 'pdf', str(sub_folder))
    else:
        target_dir = os.path.join('.', 'result', dataset_name, 'pdf', f"acc_{acc}")

    os.makedirs(target_dir, exist_ok=True)

    if dataset_dist == 'Non-IID' and alpha_dir is not None:
        final_filename = f"{dataset_dist}-{dataset_name}-alpha_{alpha_dir}.pdf"
    else:
        final_filename = f"{dataset_dist}-{dataset_name}.pdf"

    full_save_path = os.path.join(target_dir, final_filename)
    plt.savefig(full_save_path, dpi=600)
    plt.close()


def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])