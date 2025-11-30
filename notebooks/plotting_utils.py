import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def discretize_variable(values, window, min_val=None, max_val=None, labels=True, include_above_max=False):
    """
    Discretize a continuous variable into bins.

    Parameters
    ----------
    values : array-like or pd.Series
        Continuous variable values.
    window : float
        Bin width.
    min_val : float, optional
        Minimum value for binning. Defaults to values.min().
    max_val : float, optional
        Maximum value for binning. Defaults to values.max().
    labels : bool, default=True
        If True, return string labels like '0.0-0.5'.
        If False, return integer bin indices.
    include_above_max : bool, default=False
        If True, create an additional bin for values > max_val.

    Returns
    -------
    pd.Series
        Discretized categories corresponding to each value.
    """
    values = pd.Series(values)

    if min_val is None:
        min_val = values.min()
    if max_val is None:
        max_val = values.max()

    # Construct bin edges
    bins = np.arange(min_val, max_val + window, window)
    if include_above_max:
        bins = np.append(bins, np.inf)

    # Create labels if requested
    if labels:
        bin_labels = [f"{round(bins[i], 3)}-{round(bins[i+1], 3)}"
                     for i in range(len(bins) - 2)]
        if include_above_max:
            bin_labels.append(f"{round(max_val, 3)}+")
    else:
        bin_labels = False  # pd.cut will return integers

    return pd.cut(values, bins=bins, labels=bin_labels, include_lowest=True, right=True)


def mfe_discretize_variable(values, window, min_val=None, max_val=None, labels=True, include_below_min=False,
                            include_above_max=False):
    """
    Discretize a continuous variable into bins.

    Parameters
    ----------
    values : array-like or pd.Series
        Continuous variable values.
    window : float
        Bin width.
    min_val : float, optional
        Minimum value for binning. Defaults to values.min().
    max_val : float, optional
        Maximum value for binning. Defaults to values.max().
    labels : bool, default=True
        If True, return string labels like '0.0-0.5'.
        If False, return integer bin indices.
    include_below_min : bool, default=False
        If True, create an additional bin for values < min_val.
    include_above_max : bool, default=False
        If True, create an additional bin for values > max_val.

    Returns
    -------
    pd.Series
        Discretized categories corresponding to each value.
    """
    values = pd.Series(values)

    if min_val is None:
        min_val = values.min()
    if max_val is None:
        max_val = values.max()

    # Construct bin edges
    bins = np.arange(min_val, max_val + window, window)

    # Create labels if requested
    if labels:
        bin_labels = [f"({round(bins[i], 3)}, {round(bins[i+1], 3)}]"
                     for i in range(len(bins) - 1)]
        if include_below_min:
            bin_labels.insert(0, f"-{round(min_val, 3)}")
        if include_above_max:
            bin_labels.append(f"{round(max_val, 3)}+")
    else:
        bin_labels = False  # pd.cut will return integers

    if include_below_min:
        bins = np.insert(bins, 0, [-999999999]) # Use big enough integer
    if include_above_max:
        bins = np.append(bins, np.inf)

    # Create labels if requested
    if labels:
        bin_labels = [f"({round(bins[i])}, {round(bins[i+1])}]"
                     for i in range(1, len(bins) - 2)]
        if include_below_min:
            bin_labels.insert(0, f"< {round(min_val, 3)}")
        if include_above_max:
            bin_labels.append(f"{round(max_val, 3)}+")
    else:
        bin_labels = False  # pd.cut will return integers

    return pd.cut(values, bins=bins, labels=bin_labels, include_lowest=True, right=True)


def plot_grouped_violin_analysis(
    df, predictor_col, target_col, group_col,
    predictor_label="Predictor", target_label="Target", group_label="",
    target_bins=10, ceiling_count=8, show_sample_counts=False,
    min_samples=1,
    title=None, cmap='viridis', custom_palette=None, ylim=None,
    legend=True, legend_loc='lower right', legend_ncols=2,
    bbox_to_anchor=(0.5, -0.05), figsize=(8, 5), dpi=600,
    fontsize_title=12, fontsize_axis_labels=10, fontsize_xtick=9, fontsize_ytick=9,
    fontsize_legend=10, fontsize_sample_counts=8,
    group_orders=None, numeric_binning=False, numeric_bin_size=2, legend_row_major=True
):
    """
    Plot violin/box plots of predictor, binned by target and grouped by features.

    Parameters
    ----------
    df : DataFrame
        Input data
    predictor_col : str
        Column name for y-axis variable
    target_col : str
        Column name for x-axis. Can be numeric (will be binned) or categorical (used as-is)
    group_col : str
        Column name for grouping (hue)
    predictor_label : str
        Y-axis label
    target_label : str
        X-axis label
    group_label : str
        Legend title for group column
    target_bins : int or array-like
        Number of bins or bin edges for target variable (ignored if target_col is categorical)
    ceiling_count : int
        Maximum number of groups to show
    show_sample_counts : bool
        Whether to annotate sample counts on plot
    min_samples : int
        Minimum samples required per group×bin combination
    title : str, optional
        Plot title
    cmap : str
        Colormap name
    custom_palette : list, optional
        Custom color palette
    ylim : tuple, optional
        Y-axis limits
    legend : bool
        Whether to show legend
    legend_loc : str
        Legend location
    legend_ncols : int
        Number of legend columns
    bbox_to_anchor : tuple
        Legend anchor position
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Figure DPI
    fontsize_* : int
        Font sizes for various elements
    group_orders : list, optional
        Custom ordering for groups: [ordered_values]
    numeric_binning : bool
        If True, applies numeric binning for group column (0, 1-2, 3-4, ..., N+)
    numeric_bin_size: int
        Numeric bin size
    legend_row_major : bool
        If True, applies row-major reordering to legend

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    df_plot = df.copy()

    from pandas import CategoricalDtype
    is_target_categorical = isinstance(df[target_col].dtype, CategoricalDtype) or \
                            pd.api.types.is_object_dtype(df[target_col])

    if is_target_categorical:
        # Use categorical values as-is
        df_plot['target_bin_str'] = df_plot[target_col].astype(str)
    else:
        # Bin numeric target variable
        if isinstance(target_bins, int):
            df_plot['target_bin'] = pd.cut(df_plot[target_col], bins=target_bins, include_lowest=True)
        else:
            df_plot['target_bin'] = pd.cut(df_plot[target_col], bins=target_bins, include_lowest=True)

        # Convert intervals to readable strings
        def format_interval(interval):
            if pd.isna(interval):
                return 'NaN'
            left = 0 if round(interval.left) == 0 else f"{interval.left:.0f}"
            return f"{left}-{interval.right:.0f}"

        df_plot['target_bin_str'] = df_plot['target_bin'].apply(format_interval)

    if numeric_binning:
        # Convert to numeric first (handles categorical with numeric values)
        step = numeric_bin_size
        cutoff = ceiling_count - 1
        bins = [(0, 0)]
        labels = ["0"]
        for start in range(1, cutoff, step):
            end = min(start + step - 1, cutoff - 1)
            bins.append((start, end))
            labels.append(f"{start}-{end}" if start != end else f"{start}")
        cutoff_label = f"{cutoff}+"
        labels.append(cutoff_label)

        def assign_bin(val):
            if pd.isna(val):
                return np.nan
            if val >= cutoff:
                return cutoff_label
            for (start, end), label in zip(bins, labels[:-1]):
                if start <= val <= end:
                    return label
            return np.nan

        df_plot[group_col] = df_plot[group_col].apply(assign_bin)
        group_counts = df_plot.groupby(group_col, observed=False).size()
        top_groups = group_counts[group_counts >= min_samples].index.tolist()
        df_plot = df_plot[df_plot[group_col].notna()].copy()
    elif group_orders is not None:
        # Custom ordering mode
        available_groups = set(df_plot[group_col].unique())
        top_groups = [g for g in group_orders if g in available_groups][:ceiling_count]
        df_plot = df_plot[df_plot[group_col].isin(top_groups)].copy()
    else:
        if isinstance(df_plot[group_col].dtype, CategoricalDtype):
            # For categorical, get counts but preserve original values
            group_counts = df_plot[group_col].value_counts()
            top_groups = group_counts.index[:ceiling_count].tolist()
            df_plot = df_plot[df_plot[group_col].isin(top_groups)].copy()
            # Keep original categorical values, just filter categories
            df_plot[group_col] = df_plot[group_col].cat.remove_unused_categories()
        else:
            group_counts = df_plot[group_col].value_counts()
            top_groups = group_counts.index[:ceiling_count].tolist()
            df_plot = df_plot[df_plot[group_col].isin(top_groups)].copy()

    # Convert to categorical with proper ordering
    # Always convert to string for consistency
    if isinstance(df_plot[group_col].dtype, CategoricalDtype):
        visible_groups = [str(c) for c in df_plot[group_col].cat.categories]
    else:
        visible_groups = [str(g) for g in top_groups]

    df_plot[group_col] = pd.Categorical(
        df_plot[group_col].astype(str),
        categories=visible_groups,
        ordered=True
    )

    # Filter by min_samples
    group_bin_counts = df_plot.groupby([group_col, 'target_bin_str'], observed=False).size().reset_index(name='count')
    valid_bins = group_bin_counts[group_bin_counts['count'] >= min_samples]
    df_plot_filtered = df_plot.merge(
        valid_bins[[group_col, 'target_bin_str']],
        on=[group_col, 'target_bin_str'],
        how='inner'
    )

    # Update visible groups based on filtered data
    if isinstance(df_plot_filtered[group_col].dtype, CategoricalDtype):
        # Preserve categorical ordering
        df_plot_filtered[group_col] = df_plot_filtered[group_col].cat.remove_unused_categories()
        visible_groups = [str(c) for c in df_plot_filtered[group_col].cat.categories]
        df_plot_filtered[group_col] = pd.Categorical(
            df_plot_filtered[group_col].astype(str),
            categories=visible_groups,
            ordered=True
        )
    else:
        visible_groups = sorted([str(g) for g in df_plot_filtered[group_col].unique()])
        df_plot_filtered[group_col] = pd.Categorical(
            df_plot_filtered[group_col].astype(str),
            categories=visible_groups,
            ordered=True
        )

    # Preserve bin order, keep only visible bins
    if is_target_categorical:
        if isinstance(df[target_col].dtype, CategoricalDtype):
            all_bins = [str(c) for c in df[target_col].cat.categories]
        else:
            all_bins = sorted(df_plot['target_bin_str'].unique())
    else:
        all_bins = df_plot['target_bin_str'].cat.categories

    visible_bins = [b for b in all_bins if b in df_plot_filtered['target_bin_str'].values]
    df_plot_filtered['target_bin_str'] = pd.Categorical(
        df_plot_filtered['target_bin_str'],
        categories=visible_bins,
        ordered=True
    )

    if custom_palette is None:
        color_palette = sns.color_palette(cmap, len(visible_groups))
    else:
        color_palette = custom_palette

    if not df_plot_filtered.empty:
        sns.boxplot(
            data=df_plot_filtered,
            x='target_bin_str',
            y=predictor_col,
            hue=group_col,
            hue_order=visible_groups,
            ax=ax,
            palette=color_palette,
            fliersize=0
        )

    ax.set_xlabel(target_label, fontsize=fontsize_axis_labels)
    ax.set_ylabel(predictor_label, fontsize=fontsize_axis_labels)
    ax.grid(axis='y', alpha=0.7, linestyle="dashed", color="lightgrey")
    ax.axhline(y=0, xmin=0, xmax=1, linestyle="solid", color="lightgrey", linewidth=1, zorder=0)
    ax.tick_params(axis='x', rotation=0, labelsize=fontsize_xtick)
    ax.tick_params(axis='y', labelsize=fontsize_ytick)
    for label in ax.get_xticklabels():
        label.set_ha('center')

    # Force ticks to match visible bins
    ax.set_xticks(range(len(visible_bins)))
    ax.set_xticklabels(visible_bins)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(ax.get_ylim()[0], 1.25)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels,
            title=group_label,
            loc=legend_loc,
            framealpha=0.9,
            ncol=legend_ncols,
            bbox_to_anchor=bbox_to_anchor,
            fontsize=fontsize_legend,
            title_fontsize=fontsize_legend
        )

    if title:
        ax.set_title(title, fontsize=fontsize_title)

    # Sample size
    ax.text(
        0.975, 0.975, f"n={len(df)}",
        transform=ax.transAxes,
        va="top", ha="right",
        fontsize=fontsize_axis_labels
    )

    return fig


def plot_boxplot_two_dfs(
    df1, df2, columns, labels,
    fontsize=8, dpi=300, figsize=(6,3), bbox_to_anchor=(0.5, -0.45)
):
    df1_long = df1[columns].melt(var_name="Type", value_name="Score")
    df1_long["Dataset"] = labels[0]
    df2_long = df2[columns].melt(var_name="Type", value_name="Score")
    df2_long["Dataset"] = labels[1]    
    df_long = pd.concat([df1_long, df2_long], ignore_index=True)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    sns.boxplot(
        data=df_long,
        x="Type",
        y="Score",
        hue="Dataset",
        palette=["tab:blue", "tab:orange"],
        fliersize=0,
        ax=ax
    )

    legend_counts = df_long.groupby("Dataset").size().to_dict()
    new_labels = [f"{lab} (n={legend_counts[lab] // len(columns)})" for lab in labels]

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles, labels=new_labels, title="", ncol=1, 
        loc="lower center", bbox_to_anchor=bbox_to_anchor, 
        fontsize=fontsize, title_fontsize=fontsize
    )
    
    ax.tick_params(axis='x', rotation=45, labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_ylabel("")
    ax.set_xlabel("")

    return fig, ax


def plot_compare_two_dfs(
    df1, df2, label1, label2, score_col, group_col, bin_edges,
    score_label=None, group_label=None, title="",
    figsize=(5,4), bbox_to_anchor=None, dpi=300, ylim=None
):
    df1 = df1.copy()
    df1["Dataset"] = label1
    df2 = df2.copy()
    df2["Dataset"] = label2
    df = pd.concat([df1, df2], ignore_index=True)

    df[group_col] = df[group_col].apply(
        lambda v: v[0] if isinstance(v, (list, tuple)) and len(v)==1 else v
    )
    df[group_col] = pd.to_numeric(df[group_col], errors="coerce")

    bin_edges_ext = list(bin_edges) + [df[group_col].max() + 1]
    df["bin"] = pd.cut(df[group_col], bins=bin_edges_ext, include_lowest=True, right=False)

    def bin_label(interval, last_edge=bin_edges[-1]):
        if interval.left >= last_edge:
            return f"{int(interval.left)}+"
        else:
            return f"{int(interval.left)}–{int(interval.right)}"

    df["bin_str"] = df["bin"].apply(bin_label)
    categories_order = df["bin_str"].unique()
    df["bin_str"] = pd.Categorical(df["bin_str"], categories=categories_order, ordered=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.boxplot(
        data=df, x="bin_str", y=score_col, hue="Dataset",
        ax=ax, palette=["tab:blue", "tab:orange"], fliersize=0
    )

    if ylim is not None:
        ax.set_ylim(ylim)
        
    ax.set_xlabel(group_label if group_label else group_col, fontsize=7)
    ax.set_ylabel(score_label if score_label else score_col, fontsize=7)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_title(title, fontsize=8)
    ax.legend(
        title="", ncol=1, loc="lower center",
        bbox_to_anchor=bbox_to_anchor, fontsize=7, title_fontsize=7
    )

    return fig, ax