"""Build a sparse per-protein genotype representation from a long pQTL table.

PROTRIDER stays dataset-agnostic: it consumes a long table keyed by sample, protein and
SNP, and turns it into a **sparse** set of genetic terms rather than a dense
``(n_samples, n_proteins, K)`` tensor. Each distinct ``(protein, snp)`` pair is one
"term" with its own learnable effect; the per-protein genetic contribution is recovered
in the model by scatter-adding term contributions back into protein columns. This avoids
the K-padding (driven by the protein with the most SNPs) and the all-zero slices for
proteins without any pQTL.

The representation is:
    geno  : float64 ``(S, M)`` dosage matrix, one column per term
    pidx  : int64   ``(M,)``   protein-column index each term scatters into
    beta  : float64 ``(M,)``   warm-start effect sizes (0 where absent)
with ``M`` = number of distinct (protein, snp) associations matching the dataset.
The harmonisation of external SNP / sample / protein identifiers into the long schema is
left to dataset-specific preprocessing upstream.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS = ["sampleID", "proteinID", "snp_id", "dosage"]


def read_genotype_table(path: str) -> pd.DataFrame:
    """Read a long-format genotype table (CSV/TSV/Parquet) and validate its columns."""
    if str(path).endswith(".parquet"):
        tab = pd.read_parquet(path)
    else:
        sep = "\t" if str(path).endswith((".tsv", ".tab")) else ","
        tab = pd.read_csv(path, sep=sep)
    missing = [c for c in REQUIRED_COLS if c not in tab.columns]
    if missing:
        raise ValueError(
            f"Genotype table {path} is missing required columns {missing}; "
            f"expected at least {REQUIRED_COLS} (optional: 'effect_size')."
        )
    return tab


def build_genotype_sparse(tab: pd.DataFrame, samples, proteins):
    """Turn a long genotype table into a sparse list of (protein, snp) genetic terms.

    Args:
        tab: long table with columns ``sampleID, proteinID, snp_id, dosage`` and an
            optional ``effect_size`` column used to warm-start the per-term effects.
        samples: ordered sample index to align rows to (e.g. ``dataset.data.index``).
        proteins: ordered protein columns to align to (e.g. ``dataset.data.columns``).

    Returns:
        geno: float64 array ``(S, M)``, dosage of each term per sample (0 where absent).
        pidx: int64 array ``(M,)``, the protein-column index each term belongs to.
        beta_init: float64 array ``(M,)`` of warm-start effect sizes (0 where absent).
        term_protein: object array ``(M,)`` naming the protein of each term.
        term_snp: object array ``(M,)`` naming the SNP of each term.
    """
    samples = pd.Index(samples)
    proteins = pd.Index(proteins)
    s_pos = {s: i for i, s in enumerate(samples)}
    p_pos = {p: i for i, p in enumerate(proteins)}
    has_eff = "effect_size" in tab.columns

    tab = tab[tab["proteinID"].isin(p_pos) & tab["sampleID"].isin(s_pos)]

    S = len(samples)
    groups = list(tab.groupby(["proteinID", "snp_id"], sort=True))
    M = len(groups)

    geno = np.zeros((S, M), dtype=np.float64)
    pidx = np.zeros(M, dtype=np.int64)
    beta_init = np.zeros(M, dtype=np.float64)
    term_protein = np.empty(M, dtype=object)
    term_snp = np.empty(M, dtype=object)

    for m, ((prot, snp), sub) in enumerate(groups):
        pidx[m] = p_pos[prot]
        si = sub["sampleID"].map(s_pos).to_numpy()
        geno[si, m] = sub["dosage"].to_numpy(dtype=np.float64)
        if has_eff:
            beta_init[m] = float(sub["effect_size"].iloc[0])
        term_protein[m] = prot
        term_snp[m] = snp

    if M == 0:
        logger.warning("Genotype table has no entries matching the dataset's proteins/samples")
    else:
        logger.info(
            "Built sparse genotype: S=%d, M=%d (protein, snp) terms over %d proteins%s",
            S, M, len(np.unique(pidx)), " (effect-size warm-start)" if has_eff else "",
        )
    return geno, pidx, beta_init, term_protein, term_snp
