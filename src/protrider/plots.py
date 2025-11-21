import numpy as np
import pandas as pd
import plotnine as pn
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from .datasets import covariates


__all__ = ["plot_pvals", "plot_encoding_dim", "plot_aberrant_per_sample", "plot_aberrant_per_sample",
           "plot_training_loss", "plot_cv_loss", "plot_expected_vs_observed", "plot_correlation_heatmap"]

logger = logging.getLogger(__name__)


def plot_pvals(output_dir=None, distribution="t", plot_title="", fontsize=10, pvals_one_sided=None):
    """
    Plot p-value distributions.
    
    Args:
        output_dir: Optional directory to save plots. If None, plots are not saved.
        distribution: Distribution type ("t" or "normal")
        plot_title: Title for the plots
        fontsize: Font size for plot text
        pvals_one_sided: Optional DataFrame with p-values. If None, reads from output_dir/pvals_one_sided.csv
        
    Returns:
        tuple: (histogram_plot, qq_plot) - plotnine plot objects
    """
    if pvals_one_sided is None:
        if output_dir is None:
            raise ValueError("Either output_dir or pvals_one_sided must be provided")
        dt_pvals = pd.read_csv(os.path.join(output_dir, "pvals_one_sided.csv"))
    else:
        dt_pvals = pvals_one_sided.copy()
    
    dt_pvals = dt_pvals.melt(id_vars='proteinID')
    dt_pvals = dt_pvals.dropna(subset=['value'])
    
    if distribution == "t":
        dist = "Student's t-distribution"
        p_color = "#0047ab"
    else:
        dist = "Normal distribution"
        p_color = "#3cb371"

    dt_pvals['type'] = dist

    g_pvals = (
        pn.ggplot(dt_pvals, pn.aes(x='value')) +
        pn.geom_histogram(bins=100, boundary=0, fill=p_color) +
        pn.theme_bw(base_size=fontsize) +
        pn.facet_wrap('~type', ncol=1) +
        pn.labs(
            x='One-sided P-value',
            y='Count',
            title=plot_title
        ) 
    )
    
    if output_dir is not None:
        os.makedirs(f"{output_dir}/plots/", exist_ok=True)
        g_pvals.save(f"{output_dir}/plots/pvalues_dist.png", width=4, height=4, dpi=300)

    pv_t = dt_pvals['value'].dropna().values
    theoretical = -np.log10((np.arange(1, len(pv_t) + 1)) / (len(pv_t) + 1))
    sample = -np.log10(np.sort(pv_t))
    qq_data = pd.DataFrame({
        'Theoretical': theoretical,
        'Sample': sample,
        'type': dist
    })

    g_qq = (
        pn.ggplot(qq_data, pn.aes(x='Theoretical', y='Sample')) +
        pn.geom_point(color=p_color) +
        pn.geom_abline(intercept=0, slope=1, color='lightgrey', linetype='dashed') +
        pn.labs(
            title=plot_title,
            x='Theoretical -log10(P-value)',
            y='Observed -log10(P-value)'
        ) +
        pn.theme_bw(base_size=fontsize)
    )

    if output_dir is not None:
        g_qq.save(f"{output_dir}/plots/qqplots.png", width=4, height=4, dpi=300)
    
    return g_pvals, g_qq


def plot_encoding_dim(output_dir, find_q_method, plot_title="", oht_q=None, fontsize=10):
    os.makedirs(f"{output_dir}/plots/", exist_ok=True)
 
    if find_q_method != "gs":
        print("plot_encoding_dim is not implemented for OHT yet.")
        return
    
    encoding_dims = pd.read_csv(output_dir + "/grid_search.csv")
    additional_info = pd.read_csv(output_dir + "/additional_info.csv")
    bestQ = additional_info["q"].iloc[0]
    best_row = encoding_dims[encoding_dims["encod_dim"] == bestQ]

    p_out = (
        pn.ggplot(encoding_dims, pn.aes("encod_dim", "aucpr")) +
        pn.geom_point() +
        pn.scale_x_log10() +
        #pn.geom_smooth(method='loess') +
        # Optimum line & label
        pn.geom_vline(pn.aes(xintercept="encod_dim", color="'Grid search'"), 
                   data=best_row, show_legend=True) +
        pn.geom_text(pn.aes(x=bestQ - 0.5, y=0.0, label=str(bestQ)), data=best_row) +
        pn.labs(x="Encoding dimensions", 
                y="Evaluation loss (AUPRC)", 
                color="Optimal Q", 
                title="Search for best encoding dimension " + plot_title)
    )

    # Add OHT line/label if present
    if oht_q is not None:
        p_out += (
            pn.geom_vline(pn.aes(xintercept=oht_q, color="'OHT'"), show_legend=True) +
            pn.geom_text(pn.aes(x=oht_q - 0.5, y=0.0, label=str(oht_q)),
                      data=pd.DataFrame({"x": [oht_q], "y": [0.0]})) +
            pn.scale_color_manual(values={"Grid search": "red", "OHT": "lightblue"})
        )
    else:
        p_out += pn.scale_color_manual(values={"Grid search": "red"})
    
    p_out += pn.theme_bw(base_size=fontsize)

    p_out.save(f"{output_dir}/plots/encoding_dim_search.png",
               width=6, height=4, units='in', dpi=300)


def plot_aberrant_per_sample(output_dir=None, plot_title="", fontsize=10, protrider_summary=None):
    """
    Plot aberrant protein counts per sample.
    
    Args:
        output_dir: Optional directory to save plot. If None, plot is not saved.
        plot_title: Title for the plot
        fontsize: Font size for plot text
        protrider_summary: Optional DataFrame with summary. If None, reads from output_dir/protrider_summary.csv
        
    Returns:
        plotnine plot object
    """
    if protrider_summary is None:
        if output_dir is None:
            raise ValueError("Either output_dir or protrider_summary must be provided")
        res = pd.read_csv(f"{output_dir}/protrider_summary.csv")
    else:
        res = protrider_summary.copy()

    aberrants = res[res["PROTEIN_outlier"]]
    non_aberrant_ids = set(res["sampleID"].unique()) - set(aberrants["sampleID"])
    non_aberrant = pd.DataFrame({"sampleID": list(non_aberrant_ids), "outlier_count": 0})

    # Count aberrant outliers per sample
    aberrant_counts = aberrants["sampleID"].value_counts().reset_index()
    aberrant_counts.columns = ["sampleID", "outlier_count"]

    # Combine and sort
    aberrant_numbers = pd.concat([aberrant_counts, non_aberrant], ignore_index=True)
    aberrant_numbers = aberrant_numbers.sort_values("outlier_count").reset_index(drop=True)
    aberrant_numbers["Rank"] = aberrant_numbers.index + 1
    
    median_val = aberrant_numbers["outlier_count"].median()
    percentile_95 = aberrant_numbers["outlier_count"].quantile(0.95)

    yadjust = 1.2
    
    p_out = (
        pn.ggplot(aberrant_numbers, pn.aes(x="Rank", y="outlier_count + 1")) +
        pn.geom_line(color="blue") +
        pn.labs(x="Sample rank", y="Outliers per sample + 1", title=plot_title) +
        pn.scale_y_log10() +
        pn.geom_hline(yintercept=median_val, color="black") +
        pn.geom_hline(yintercept=percentile_95, color="black") +
        pn.geom_text(
            pn.aes(x=5, y=median_val*yadjust, label="'Median'"),
            size=fontsize*0.7,
            ha='left'
        ) +
        pn.geom_text(
            pn.aes(x=5, y=percentile_95*yadjust, label="'95th percentile'"),
            size=fontsize*0.7,
            ha='left'
        ) +
        pn.theme_bw(base_size=fontsize)
    )
    
    if output_dir is not None:
        os.makedirs(f"{output_dir}/plots/", exist_ok=True)
        p_out.save(f"{output_dir}/plots/aberrant_per_sample.png",
                   width=6, height=4, units='in', dpi=300)
    
    return p_out


def plot_training_loss(output_dir=None, plot_title="", fontsize=10, train_losses=None):
    """
    Plot training loss over epochs.
    
    Args:
        output_dir: Optional directory to save plot. If None, plot is not saved.
        plot_title: Title for the plot
        fontsize: Font size for plot text
        train_losses: Optional DataFrame with training losses. If None, reads from output_dir/train_losses.csv
        
    Returns:
        plotnine plot object
    """
    if train_losses is None:
        if output_dir is None:
            raise ValueError("Either output_dir or train_losses must be provided")
        loss_histoty = pd.read_csv(f"{output_dir}/train_losses.csv")
    else:
        loss_histoty = train_losses.copy()
    
    p_out = (
        pn.ggplot(loss_histoty, pn.aes(x="epoch", y="train_loss")) +
        pn.geom_line(color="blue") +
        pn.labs(x="Epoch", y="Loss", title=plot_title) +
        pn.theme_bw(base_size=fontsize) +
        pn.scale_y_log10()
    )

    if output_dir is not None:
        os.makedirs(f"{output_dir}/plots/", exist_ok=True)
        p_out.save(f"{output_dir}/plots/training_loss.png",
                   width=6, height=4, units='in', dpi=300)
    
    return p_out


def plot_correlation_heatmap(output_dir, sample_annotation_path: str, plot_title="", covariate_name=None):
    """
    Create a correlation heatmap plot for protein data colored by covariate values.
    
    Args:
        
    """
    output_dir = Path(output_dir)
    zscore_data = pd.read_csv(output_dir / 'zscores.csv').set_index('proteinID')

    row_colors = None
    if sample_annotation_path is not None:
        sample_annotation = covariates.read_annotation_file(sample_annotation_path)
        # Get covariate values for coloring
        if covariate_name is not None:
            if covariate_name not in sample_annotation.columns:
                raise ValueError(f"Covariate '{covariate_name}' not found in sample annotation.")
            covariate_values = sample_annotation[covariate_name]
            
            # Create a lookup table for coloring rows
            # Use a larger color palette to support more unique covariate values
            unique_vals = covariate_values.unique()
            if len(unique_vals) > 20:
                logger.warning(f"Covariate '{covariate_name}' has more than 20 unique values. Skipping color annotation.")
            else:
                palette = sns.color_palette("tab20", len(unique_vals))
                lut = dict(zip(unique_vals, palette))
                row_colors = [lut[label] for label in covariate_values]
        
    # Calculate correlation matrix
    corr_matrix = zscore_data.corr()
    
    # Create clustermap
    clustermap = sns.clustermap(
        corr_matrix,
        cmap='mako',
        row_colors=None if not row_colors else row_colors,
    )
    
    # Add legend for row colors if they exist
    if row_colors is not None and covariate_name is not None:
        # Create legend patches
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=str(val)) 
                          for val, color in lut.items()]
        clustermap.ax_col_dendrogram.legend(handles=legend_elements, 
                                           title=covariate_name,
                                           bbox_to_anchor=(1.15, 1), 
                                           loc='upper left',
                                           frameon=True)
    
    # Adjust layout
    plt.setp(clustermap.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.setp(clustermap.ax_heatmap.get_yticklabels(), rotation=0)
    plt.title(plot_title)
    plt.savefig(output_dir / 'plots' / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved correlation heatmap to {output_dir / 'plots' / 'correlation_heatmap.png'}")


def plot_cv_loss(train_losses, val_losses, fold, out_dir):
    # plot the loss history; stratified by fold
    plot_dir = Path(out_dir) / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_p = f'{plot_dir}/loss_history_fold{fold}.png'

    df = pd.concat([pd.DataFrame(dict(type='validation', loss=val_losses, epoch=np.arange(len(val_losses)))),
                    pd.DataFrame(dict(type='train', loss=train_losses, epoch=np.arange(len(train_losses))))])
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='epoch', y='loss', hue='type')
    plt.title(f'Loss history for fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(title=f'Fold {fold}')
    plt.savefig(out_p)
    plt.close()
    logger.info(f"Saved loss history plot for fold {fold} to {out_p}")


def plot_expected_vs_observed(protein_id, output_dir=None, plot_title="", fontsize=10, 
                             processed_input=None, output_data=None, protrider_summary=None):
    """
    Plot expected vs observed intensities for a specific protein.
    
    Args:
        protein_id: ID of the protein to plot
        output_dir: Optional directory to save plot. If None, plot is not saved.
        plot_title: Title for the plot
        fontsize: Font size for plot text
        processed_input: Optional DataFrame with processed input data. If None, reads from output_dir/processed_input.csv
        output_data: Optional DataFrame with output data. If None, reads from output_dir/output.csv
        protrider_summary: Optional DataFrame with summary. If None, reads from output_dir/protrider_summary.csv
        
    Returns:
        plotnine plot object
    """
    if processed_input is None or output_data is None or protrider_summary is None:
        if output_dir is None:
            raise ValueError("Either output_dir or all data DataFrames must be provided")
        data_in = pd.read_csv(f"{output_dir}/processed_input.csv")
        data_out = pd.read_csv(f"{output_dir}/output.csv")
        summary = pd.read_csv(f"{output_dir}/protrider_summary.csv")
    else:
        data_in = processed_input.copy()
        data_out = output_data.copy()
        summary = protrider_summary.copy()

    data_in_melted = data_in[data_in["proteinID"] == protein_id].melt(id_vars="proteinID", var_name="SAMPLE_ID", value_name="in_int")
    data_out_melted = data_out[data_out["proteinID"] == protein_id].melt(id_vars="proteinID", var_name="SAMPLE_ID", value_name="out_int")

    # Merge inputs and outputs
    df_plot = pd.merge(data_in_melted, data_out_melted, on="SAMPLE_ID")
    summary_filtered = summary[(summary["proteinID"] == protein_id) & summary["PROTEIN_outlier"]]
    is_outlier = df_plot["SAMPLE_ID"].isin(summary_filtered["sampleID"])
    df_plot["outlier"] = is_outlier.replace({True: "Outlier", False: None})
    df_plot["label"] = df_plot["SAMPLE_ID"].where(is_outlier, None)
    df_plot["outlier"] = pd.Categorical(df_plot["outlier"], categories=["Outlier"])

    p_out = (
        pn.ggplot(df_plot, pn.aes(x="out_int", y="in_int")) +
        pn.geom_point(pn.aes(color="outlier"), show_legend=False) +
        pn.geom_text(
            pn.aes(label="label"),
            size=4,
            va="bottom", ha="left",
            nudge_y=0.001  # slight vertical shift to avoid overlap
        ) +
        pn.scale_x_log10() +
        pn.scale_y_log10() +
        pn.labs(
            x="Expected log10-intensity",
            y="Observed log10-intensity",
            title=f"Protein: {protein_id}"
        ) +
        pn.theme_bw(base_size=fontsize)
    )

    if output_dir is not None:
        os.makedirs(f"{output_dir}/plots/", exist_ok=True)
        p_out.save(f"{output_dir}/plots/expected_vs_observed.png",
                   width=4, height=4, units='in', dpi=300)
    
    return p_out
