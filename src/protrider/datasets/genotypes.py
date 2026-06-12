"""Build a per-protein genotype tensor from a long-format pQTL genotype table.

PROTRIDER stays dataset-agnostic: it consumes a long table keyed by sample, protein
and SNP, and pivots it into a dense ``(n_samples, n_proteins, K)`` tensor aligned to the
dataset's own sample index and (post-filtering) protein columns. ``K`` is the maximum
number of cis-SNPs assigned to any single protein; proteins with fewer SNPs are
zero-padded. The harmonisation of external SNP / sample / protein identifiers into this
schema is left to dataset-specific preprocessing upstream.
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


def build_genotype_tensor(tab: pd.DataFrame, samples, proteins):
    """Pivot a long genotype table into a dense aligned tensor.

    Args:
        tab: long table with columns ``sampleID, proteinID, snp_id, dosage`` and an
            optional ``effect_size`` column used to warm-start the per-SNP effects.
        samples: ordered sample index to align rows to (e.g. ``dataset.data.index``).
        proteins: ordered protein columns to align to (e.g. ``dataset.data.columns``).

    Returns:
        geno: float64 array ``(S, P, K)``, zero-filled for absent (sample, protein, snp).
        snp_map: object array ``(P, K)`` naming the SNP in each slot (``''`` if empty).
        beta_init: float64 array ``(P, K)`` of warm-start effect sizes (0 where absent).
    """
    samples = pd.Index(samples)
    proteins = pd.Index(proteins)
    s_pos = {s: i for i, s in enumerate(samples)}
    p_pos = {p: i for i, p in enumerate(proteins)}
    has_eff = "effect_size" in tab.columns

    tab = tab[tab["proteinID"].isin(p_pos) & tab["sampleID"].isin(s_pos)]

    snps_per_prot = tab.groupby("proteinID")["snp_id"].unique()
    K = int(snps_per_prot.map(len).max()) if len(snps_per_prot) else 0

    S, P = len(samples), len(proteins)
    geno = np.zeros((S, P, K), dtype=np.float64)
    snp_map = np.empty((P, K), dtype=object)
    snp_map[:] = ""
    beta_init = np.zeros((P, K), dtype=np.float64)
    if K == 0:
        logger.warning("Genotype table has no entries matching the dataset's proteins/samples")
        return geno, snp_map, beta_init

    for prot, snp_ids in snps_per_prot.items():
        pi = p_pos[prot]
        snp_slot = {snp: k for k, snp in enumerate(snp_ids)}
        for k, snp in enumerate(snp_ids):
            snp_map[pi, k] = snp
        sub = tab[tab["proteinID"] == prot]
        si = sub["sampleID"].map(s_pos).to_numpy()
        ki = sub["snp_id"].map(snp_slot).to_numpy()
        geno[si, pi, ki] = sub["dosage"].to_numpy(dtype=np.float64)
        if has_eff:
            eff = sub.groupby("snp_id")["effect_size"].first()
            for snp, k in snp_slot.items():
                if snp in eff.index:
                    beta_init[pi, k] = eff.loc[snp]

    n_prot_with_snp = int((snp_map != "").any(axis=1).sum())
    logger.info(
        "Built genotype tensor: shape %s, K=%d, %d/%d proteins have >=1 cis-SNP%s",
        geno.shape, K, n_prot_with_snp, P, " (effect-size warm-start)" if has_eff else "",
    )
    return geno, snp_map, beta_init
