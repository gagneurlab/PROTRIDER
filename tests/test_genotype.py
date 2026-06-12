"""Tests for the pQTL genotype correction (additive per-protein genetic term).

Run with:
    pytest tests/test_genotype.py -v
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from protrider.datasets import ProtriderDataset
from protrider.datasets.genotypes import build_genotype_tensor
from protrider.model import ProtriderAutoencoder
from protrider.pipeline import save_model, load_model
from protrider import ProtriderConfig


INTENSITIES = 'sample_data/protrider_sample_dataset.tsv'
SA = 'sample_data/sample_annotations.tsv'


def _make_dataset(genotype=None):
    return ProtriderDataset(
        input_intensities=INTENSITIES,
        sa_file=SA,
        index_col='protein_ID',
        log_func=None,
        maxNA_filter=0.3,
        device=torch.device('cpu'),
        input_format='proteins_as_rows',
        genotype=genotype,
    )


@pytest.fixture
def synthetic_genotype_file():
    """Write a long genotype table for a handful of proteins/SNPs aligned to the data."""
    ds = _make_dataset()
    samples = list(ds.data.index)
    proteins = list(ds.data.columns)
    rng = np.random.default_rng(0)

    rows = []
    # proteins[0] gets 2 SNPs, proteins[1] gets 1 SNP, rest none
    plan = {proteins[0]: ['snpA', 'snpB'], proteins[1]: ['snpC']}
    for prot, snps in plan.items():
        for snp in snps:
            for s in samples:
                rows.append((s, prot, snp, int(rng.integers(0, 3)), 0.5))
    tab = pd.DataFrame(rows, columns=['sampleID', 'proteinID', 'snp_id', 'dosage', 'effect_size'])

    tmp = tempfile.NamedTemporaryFile(suffix='.tsv', delete=False, mode='w')
    tab.to_csv(tmp.name, sep='\t', index=False)
    yield tmp.name, plan
    Path(tmp.name).unlink(missing_ok=True)


def test_no_genotype_is_backward_compatible():
    ds = _make_dataset()
    assert ds.n_snps == 0
    assert ds.geno.shape == (ds.X.shape[0], ds.X.shape[1], 0)
    # __getitem__ now returns 5 elements; the 5th is an empty-K geno slice
    x, mask, cov, pm, geno = ds[0]
    assert geno.shape == (ds.X.shape[1], 0)


def test_build_genotype_tensor_alignment(synthetic_genotype_file):
    path, plan = synthetic_genotype_file
    ds = _make_dataset(genotype=path)
    P = ds.X.shape[1]
    assert ds.n_snps == 2  # max SNPs per protein
    assert ds.geno.shape == (ds.X.shape[0], P, 2)
    # protein 0 has 2 non-zero-mapped slots, protein 1 has 1, others empty
    assert (ds.geno_snp_map[0] != '').sum() == 2
    assert (ds.geno_snp_map[1] != '').sum() == 1
    assert (ds.geno_snp_map[2] == '').all()
    # effect-size warm start is carried into beta_init
    assert np.allclose(ds.geno_beta_init[0], [0.5, 0.5])
    assert np.allclose(ds.geno_beta_init[2], [0.0, 0.0])


def test_genetic_term_changes_output_and_matches_einsum(synthetic_genotype_file):
    path, _ = synthetic_genotype_file
    ds = _make_dataset(genotype=path)
    n_cov = ds.covariates.shape[1]
    n_prots = ds.X.shape[1]

    model = ProtriderAutoencoder(
        in_dim=n_prots, latent_dim=4, n_layers=1, n_cov=n_cov,
        prot_means=ds.prot_means_torch, presence_absence=False,
        n_snps=ds.n_snps, beta_init=ds.geno_beta_init,
    ).double()

    out_with = model(ds.X, ds.torch_mask, cond=ds.covariates, geno=ds.geno)
    out_without = model(ds.X, ds.torch_mask, cond=ds.covariates, geno=None)

    # warm-started non-zero beta => genetic term must shift the reconstruction
    assert not torch.allclose(out_with, out_without)

    # _genetic_term computes g[s,p] = sum_k geno[s,p,k] * beta[p,k]
    g = torch.einsum('spk,pk->sp', ds.geno, model.snp_effects)
    assert torch.allclose(model._genetic_term(ds.geno), g, atol=1e-12)
    # the added output term is exactly g on top of decoder(encoder(X - g))
    enc_in = ds.X - g
    expected = model.decoder(model.encoder(enc_in, cond=ds.covariates), cond=ds.covariates) + g
    assert torch.allclose(out_with, expected, atol=1e-9)


def test_genetic_term_has_gradient(synthetic_genotype_file):
    path, _ = synthetic_genotype_file
    ds = _make_dataset(genotype=path)
    model = ProtriderAutoencoder(
        in_dim=ds.X.shape[1], latent_dim=4, n_layers=1, n_cov=ds.covariates.shape[1],
        prot_means=ds.prot_means_torch, n_snps=ds.n_snps,
    ).double()
    out = model(ds.X, ds.torch_mask, cond=ds.covariates, geno=ds.geno)
    out.sum().backward()
    assert model.snp_effects.grad is not None
    assert torch.isfinite(model.snp_effects.grad).all()


def test_checkpoint_roundtrip_preserves_snp_effects(synthetic_genotype_file):
    path, _ = synthetic_genotype_file
    ds = _make_dataset(genotype=path)
    # prot_means=None matches the production save/load contract (models are saved after
    # PCA init, where the encoder carries a trainable bias rather than fixed prot_means).
    model = ProtriderAutoencoder(
        in_dim=ds.X.shape[1], latent_dim=4, n_layers=1, n_cov=ds.covariates.shape[1],
        prot_means=None, n_snps=ds.n_snps, beta_init=ds.geno_beta_init,
    ).double()
    with torch.no_grad():
        model.snp_effects.add_(0.123)

    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / 'model.pt'
        save_model(model, str(ckpt), q=4)
        cfg = ProtriderConfig(input_intensities=INTENSITIES, h_dim=None, device='cpu')
        loaded, q = load_model(ds, str(ckpt), cfg)

    assert loaded is not None and q == 4
    assert loaded.n_snps == ds.n_snps
    assert torch.allclose(loaded.snp_effects, model.snp_effects)
