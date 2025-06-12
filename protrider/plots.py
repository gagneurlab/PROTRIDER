import numpy as np
import pandas as pd
import plotnine as pn
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging


__all__ = ["_plot_pvals", "_plot_encoding_dim", "_plot_aberrant_per_sample", "_plot_aberrant_per_sample",
           "_plot_training_loss", "_plot_cv_loss", "_plot_expected_vs_observed"]

logger = logging.getLogger(__name__)


def _plot_pvals(output_dir, distribution, plot_title="", fontsize=10):
    os.makedirs(f"{output_dir}/plots/", exist_ok=True)
  
    dt_pvals = pd.read_csv(os.path.join(output_dir, "pvals_one_sided.csv"))
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

    g_qq.save(f"{output_dir}/plots/qqplots.png", width=4, height=4, dpi=300)


def _plot_encoding_dim(output_dir, find_q_method, plot_title="", oht_q=None, fontsize=10):
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
            pn.geom_vline(aes(xintercept=oht_q, color="'OHT'"), show_legend=True) +
            pn.geom_text(aes(x=oht_q - 0.5, y=0.0, label=str(oht_q)),
                      data=pd.DataFrame({"x": [oht_q], "y": [0.0]})) +
            pn.scale_color_manual(values={"Grid search": "red", "OHT": "lightblue"})
        )
    else:
        p_out += pn.scale_color_manual(values={"Grid search": "red"})
    
    p_out += pn.theme_bw(base_size=fontsize)

    p_out.save(f"{output_dir}/plots/encoding_dim_search.png",
               width=6, height=4, units='in', dpi=300)


def _plot_aberrant_per_sample(output_dir, plot_title="", fontsize=10):
    os.makedirs(f"{output_dir}/plots/", exist_ok=True)

    res = pd.read_csv(f"{output_dir}/protrider_summary.csv")
    
    aberrants = res[res["PROTEIN_outlier"] == True]
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
    
    p_out.save(f"{output_dir}/plots/aberrant_per_sample.png",
               width=6, height=4, units='in', dpi=300)


def _plot_training_loss(output_dir, plot_title="", fontsize=10):
    os.makedirs(f"{output_dir}/plots/", exist_ok=True)
    loss_histoty = pd.read_csv(f"{output_dir}/train_losses.csv")
    p_out = (
        pn.ggplot(loss_histoty, pn.aes(x="epoch", y="train_loss")) +
        pn.geom_line(color="blue") +
        pn.labs(x="Epoch", y="Loss", title=plot_title) +
        pn.theme_bw(base_size=fontsize)
    )

    p_out.save(f"{output_dir}/plots/training_loss.png",
               width=6, height=4, units='in', dpi=300)


def _plot_cv_loss(train_losses, val_losses, fold, out_dir):
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


def _plot_expected_vs_observed(output_dir, protein_id, plot_title="", fontsize=10):
    os.makedirs(f"{output_dir}/plots/", exist_ok=True)
    data_in = pd.read_csv(f"{output_dir}/processed_input.csv")
    data_out = pd.read_csv(f"{output_dir}/output.csv")
    summary = pd.read_csv(f"{output_dir}/protrider_summary.csv")

    data_in_melted = data_in[data_in["proteinID"] == protein_id].melt(id_vars="proteinID", var_name="SAMPLE_ID", value_name="in_int")
    data_out_melted = data_out[data_out["proteinID"] == protein_id].melt(id_vars="proteinID", var_name="SAMPLE_ID", value_name="out_int")

    # Merge inputs and outputs
    df_plot = pd.merge(data_in_melted, data_out_melted, on="SAMPLE_ID")
    summary_filtered = summary[(summary["proteinID"] == protein_id) & (summary["PROTEIN_outlier"] == True)]
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

    p_out.save(f"{output_dir}/plots/expected_vs_observed.png",
               width=4, height=4, units='in', dpi=300)
