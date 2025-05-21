import numpy as np
import pandas as pd
import plotnine as pn
import os

import matplotlib.pyplot as plt

__all__ = ["_plot_pvals", "_plot_encoding_dim"]


def _plot_pvals(output_dir, distribution, plot_title=""):
    os.makedirs(output_dir + "/plots/", exist_ok=True)
  
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
   
    fontsize = 10
    
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
    
    g_pvals.save(output_dir + "/plots/pvalues_dist.png", width=4, height=4, dpi=300)

    
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

    # Save QQ plot
    g_qq.save(output_dir + "/plots/qqplots.png", width=4, height=4, dpi=300)


def _plot_encoding_dim(output_dir, find_q_method, plot_title="", oht_q=None):
    
    if find_q_method != "gs":
        print("plot_encoding_dim is not implemented for OHT yet.")
        return
    
    os.makedirs(output_dir + "/plots/", exist_ok=True)
    
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
    
    fontsize = 10
    p_out += pn.theme_bw(base_size=fontsize)

    # Save plot
    p_out.save(output_dir + "/plots/encoding_dim_search.png", 
               width=6, height=6, units='in', dpi=300)
