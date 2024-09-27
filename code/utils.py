import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import seaborn as sns
import pdb


def plot_interval(
    ax,
    lower,
    upper,
    height,
    color_face,
    color_stroke,
    linewidth=5,  # 5个节点:5, 20个节点：2
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

def make_plots_fl(
    df,
    plot_savename,
    xlim,
    ylim,
    n_idx=-1,
    true_theta=None,
    # true_label=r"$\theta^*$",
    true_label= "Ground truth",
    intervals_xlabel="x",
    plot_classical=True,
    ppi_facecolor="#2F7FC1",  # 填充颜色
    ppi_strokecolor="b",  # 边框颜色
    classical_facecolor="#96C37D",
    classical_strokecolor="g",
    imputation_facecolor="#F3D266",
    imputation_strokecolor="y",
    empty_panel=False,
):
    # Make plot
    ns = df.n.unique()
    ns = ns[~np.isnan(ns)].astype(int)
    n = ns[n_idx]

    ppi_intervals = df[df.method == "Node"]
    num_intervals = len(ppi_intervals)
    if plot_classical:
        classical_interval = df[df.method == "FL Aggregation"]
    imputation_interval = df[df.method == "Combined"]

    # xlim = [0, 1.0]
    # ylim = [0, 1.0]

    if empty_panel:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    else:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3, 2.5))  # 原（3，2.5） 加标题则为(3,3)，获取legend为（12，3）
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
            plot_interval(
                axs,
                ppi_interval.lower,
                ppi_interval.upper,
                0.90,  # 5个节点：0.90；20个节点：0.95
                ppi_facecolor,
                ppi_strokecolor,
                label=r"Client 1-20",
            )
            if plot_classical:
                plot_interval(
                    axs,
                    classical_interval.lower,
                    classical_interval.upper,
                    0.20,  # 5个节点：0.20；20个节点：0.15
                    classical_facecolor,
                    classical_strokecolor,
                    label=r"FL aggregation",
                )
            plot_interval(
                axs,
                imputation_interval.lower,
                imputation_interval.upper,
                0.075,  # 0.075
                imputation_facecolor,
                imputation_strokecolor,
                label=r"Centralized data",
            )
        else:
            lighten_factor = 1.0 / np.sqrt(i)
            yshift = i * 0.15  # 5个节点：0.15；20个节点：0.04
            plot_interval(
                axs,
                ppi_interval.lower,
                ppi_interval.upper,
                0.90 - yshift,  # 5个节点：0.90；20个节点：0.95
                lighten_color(ppi_facecolor, lighten_factor),
                lighten_color(ppi_strokecolor, lighten_factor),
            )
    # axs.set_title('Case 3', fontsize=18, pad=20)
    axs.set_xlabel(intervals_xlabel, fontsize=16, labelpad=10)
    axs.set_yticks([])
    axs.set_yticklabels([])
    axs.set_ylim(ylim)
    axs.set_xlim(xlim)
    # axs.legend(fontsize=10, ncol=4)

    sns.despine(ax=axs, top=True, right=True, left=True)

    if empty_panel:
        sns.despine(ax=axs, top=True, right=True, left=True, bottom=True)
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_xticklabels([])
        axs.set_yticklabels([])
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.tight_layout()
    plt.show()
    plt.savefig('./result/'+ plot_savename, dpi=600)

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
