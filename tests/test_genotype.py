"""Tests for the pQTL genotype correction (sparse additive per-protein genetic term).

The genotype is stored sparsely as M = (protein, snp) terms: a dosage matrix geno (S, M),
a learnable effect beta (M,), and a protein-index map (M,). The model recovers the
per-protein contribution g[s,p] = sum_{m: pidx[m]=p} geno[s,m]*beta[m] via index_add.

Run with:
    pytest tests/test_genotype.py -v
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import copy

from protrider.datasets import ProtriderDataset
from protrider.model import ProtriderAutoencoder, MSEBCELoss, train
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

    # proteins[0] gets 2 SNPs, proteins[1] gets 1 SNP -> M = 3 terms
    plan = {proteins[0]: ['snpA', 'snpB'], proteins[1]: ['snpC']}
    rows = []
    for prot, snps in plan.items():
        for snp in snps:
            for s in samples:
                rows.append((s, prot, snp, int(rng.integers(0, 3)), 0.5))
    tab = pd.DataFrame(rows, columns=['sampleID', 'proteinID', 'snp_id', 'dosage', 'effect_size'])

    tmp = tempfile.NamedTemporaryFile(suffix='.tsv', delete=False, mode='w')
    tab.to_csv(tmp.name, sep='\t', index=False)
    yield tmp.name, plan, proteins
    Path(tmp.name).unlink(missing_ok=True)


def test_no_genotype_is_backward_compatible():
    ds = _make_dataset()
    assert ds.n_geno_terms == 0
    assert ds.geno.shape == (ds.X.shape[0], 0)
    assert ds.geno_protein_index.shape == (0,)
    # __getitem__ returns 5 elements; the 5th is an empty per-term slice
    x, mask, cov, pm, geno = ds[0]
    assert geno.shape == (0,)


def test_build_sparse_alignment(synthetic_genotype_file):
    path, plan, proteins = synthetic_genotype_file
    ds = _make_dataset(genotype=path)
    assert ds.n_geno_terms == 3  # 2 + 1 (protein, snp) terms
    assert ds.geno.shape == (ds.X.shape[0], 3)
    assert ds.geno_protein_index.shape == (3,)

    # terms map onto the right protein columns (and only those)
    p0, p1 = proteins[0], proteins[1]
    expected_pidx = {proteins.index(p0): 2, proteins.index(p1): 1}  # protein col -> #terms
    counts = pd.Series(ds.geno_protein_index.numpy()).value_counts().to_dict()
    assert counts == expected_pidx
    assert set(ds.geno_term_protein) == {p0, p1}
    assert set(ds.geno_term_snp) == {'snpA', 'snpB', 'snpC'}
    # effect-size warm start carried into beta_init for every term
    assert np.allclose(ds.geno_beta_init, 0.5)


def test_genetic_term_matches_scatter_reference(synthetic_genotype_file):
    path, _, _ = synthetic_genotype_file
    ds = _make_dataset(genotype=path)
    model = ProtriderAutoencoder(
        in_dim=ds.X.shape[1], latent_dim=4, n_layers=1, n_cov=ds.covariates.shape[1],
        prot_means=ds.prot_means_torch, n_geno_terms=ds.n_geno_terms,
        beta_init=ds.geno_beta_init, geno_protein_index=ds.geno_protein_index,
    ).double()

    g = model._genetic_term(ds.geno)
    assert g.shape == (ds.X.shape[0], ds.X.shape[1])

    # independent reference: explicit per-term accumulation into protein columns
    g_ref = torch.zeros_like(g)
    for m in range(ds.n_geno_terms):
        p = int(ds.geno_protein_index[m])
        g_ref[:, p] += ds.geno[:, m] * model.snp_effects[m]
    assert torch.allclose(g, g_ref, atol=1e-12)

    # only proteins that have a term are non-zero
    nonzero_cols = torch.nonzero(g.abs().sum(0) > 0).flatten().tolist()
    assert set(nonzero_cols) == set(ds.geno_protein_index.tolist())


def test_forward_subtracts_at_encoder_adds_at_output(synthetic_genotype_file):
    path, _, _ = synthetic_genotype_file
    ds = _make_dataset(genotype=path)
    model = ProtriderAutoencoder(
        in_dim=ds.X.shape[1], latent_dim=4, n_layers=1, n_cov=ds.covariates.shape[1],
        prot_means=ds.prot_means_torch, n_geno_terms=ds.n_geno_terms,
        beta_init=ds.geno_beta_init, geno_protein_index=ds.geno_protein_index,
    ).double()

    out_with = model(ds.X, ds.torch_mask, cond=ds.covariates, geno=ds.geno)
    out_without = model(ds.X, ds.torch_mask, cond=ds.covariates, geno=None)
    assert not torch.allclose(out_with, out_without)  # warm-started beta shifts output

    g = model._genetic_term(ds.geno)
    expected = model.decoder(model.encoder(ds.X - g, cond=ds.covariates), cond=ds.covariates) + g
    assert torch.allclose(out_with, expected, atol=1e-9)


def test_genetic_term_has_gradient(synthetic_genotype_file):
    path, _, _ = synthetic_genotype_file
    ds = _make_dataset(genotype=path)
    model = ProtriderAutoencoder(
        in_dim=ds.X.shape[1], latent_dim=4, n_layers=1, n_cov=ds.covariates.shape[1],
        prot_means=ds.prot_means_torch, n_geno_terms=ds.n_geno_terms,
        geno_protein_index=ds.geno_protein_index,
    ).double()
    model(ds.X, ds.torch_mask, cond=ds.covariates, geno=ds.geno).sum().backward()
    assert model.snp_effects.grad is not None
    assert torch.isfinite(model.snp_effects.grad).all()


def test_genetic_l2_matches_formula(synthetic_genotype_file):
    path, _, _ = synthetic_genotype_file
    ds = _make_dataset(genotype=path)
    model = ProtriderAutoencoder(
        in_dim=ds.X.shape[1], latent_dim=4, n_layers=1, n_cov=ds.covariates.shape[1],
        prot_means=ds.prot_means_torch, n_geno_terms=ds.n_geno_terms,
        beta_init=ds.geno_beta_init, geno_protein_index=ds.geno_protein_index,
    ).double()
    # prior buffer equals the warm-start effect sizes
    assert torch.allclose(model.snp_effects_prior,
                          torch.as_tensor(ds.geno_beta_init, dtype=torch.double))
    # at init beta == prior, so penalty is 0
    assert model.genetic_l2().item() == pytest.approx(0.0)
    # perturb and check sum of squared deviations from the prior
    with torch.no_grad():
        model.snp_effects.add_(torch.tensor([0.1, -0.2, 0.3], dtype=torch.double))
    assert model.genetic_l2().item() == pytest.approx(0.1**2 + 0.2**2 + 0.3**2)


def test_geno_l2_shrinks_beta_toward_prior(synthetic_genotype_file):
    path, _, _ = synthetic_genotype_file
    ds = _make_dataset(genotype=path)
    base = ProtriderAutoencoder(
        in_dim=ds.X.shape[1], latent_dim=4, n_layers=1, n_cov=ds.covariates.shape[1],
        prot_means=None, n_geno_terms=ds.n_geno_terms,
        beta_init=ds.geno_beta_init, geno_protein_index=ds.geno_protein_index,
    ).double()
    crit = MSEBCELoss()
    prior = base.snp_effects_prior.detach().clone()

    # identical starting point for both runs
    free = copy.deepcopy(base)
    reg = copy.deepcopy(base)
    torch.manual_seed(0)
    train(ds, free, crit, n_epochs=60, learning_rate=1e-2, patience=100, geno_l2=0.0)
    torch.manual_seed(0)
    train(ds, reg, crit, n_epochs=60, learning_rate=1e-2, patience=100, geno_l2=1e3)

    free_dist = (free.snp_effects.detach() - prior).norm().item()
    reg_dist = (reg.snp_effects.detach() - prior).norm().item()
    # the strong penalty keeps beta near the prior; the free fit drifts away
    assert reg_dist < free_dist


def test_checkpoint_roundtrip_preserves_genetic_term(synthetic_genotype_file):
    path, _, _ = synthetic_genotype_file
    ds = _make_dataset(genotype=path)
    # prot_means=None matches the production save/load contract (encoder keeps a trainable
    # bias after PCA init rather than fixed prot_means).
    model = ProtriderAutoencoder(
        in_dim=ds.X.shape[1], latent_dim=4, n_layers=1, n_cov=ds.covariates.shape[1],
        prot_means=None, n_geno_terms=ds.n_geno_terms,
        beta_init=ds.geno_beta_init, geno_protein_index=ds.geno_protein_index,
    ).double()
    with torch.no_grad():
        model.snp_effects.add_(0.123)

    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / 'model.pt'
        save_model(model, str(ckpt), q=4)
        cfg = ProtriderConfig(input_intensities=INTENSITIES, h_dim=None, device='cpu')
        loaded, q = load_model(ds, str(ckpt), cfg)

    assert loaded is not None and q == 4
    assert loaded.n_geno_terms == ds.n_geno_terms
    assert torch.allclose(loaded.snp_effects, model.snp_effects)
    assert torch.equal(loaded.geno_protein_index, model.geno_protein_index)
