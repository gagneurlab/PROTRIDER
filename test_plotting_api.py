"""
Quick test to verify the new plotting API works.
"""
from protrider import ProtriderConfig, run_protrider

# Create a minimal config using sample data
config = ProtriderConfig(
    out_dir='output/',
    input_intensities='sample_data/protrider_sample_dataset.tsv',
    sample_annotation='sample_data/sample_annotations.tsv',
    index_col='protein_ID',
    cov_used=['AGE'],
    n_epochs=5,  # Just a few epochs for testing
    q=5
)

print("Running PROTRIDER...")
result, model_info = run_protrider(config)

print("\n=== Testing plotting with out_dir (saves files) ===")
print("Testing model_info.plot_training_loss()...")
plot = model_info.plot_training_loss(config.out_dir)
print(f"  Returned: {type(plot)}")

print("\nTesting result.plot_pvals()...")
hist, qq = result.plot_pvals(config.out_dir)
print(f"  Returned: {type(hist)}, {type(qq)}")

print("\nTesting result.plot_aberrant_per_sample()...")
plot = result.plot_aberrant_per_sample(config.out_dir)
print(f"  Returned: {type(plot)}")

# Get a protein ID from the results
protein_ids = result.df_pvals.columns.tolist()
if protein_ids:
    test_protein = protein_ids[0]
    print(f"\nTesting result.plot_expected_vs_observed('{test_protein}')...")
    plot = result.plot_expected_vs_observed(test_protein, config.out_dir)
    print(f"  Returned: {type(plot)}")

print("\n=== Testing plotting without out_dir (returns plot objects) ===")
print("Testing model_info.plot_training_loss() without out_dir...")
plot = model_info.plot_training_loss()
print(f"  Returned: {type(plot)}")

print("\nTesting result.plot_pvals() without out_dir...")
hist, qq = result.plot_pvals()
print(f"  Returned: {type(hist)}, {type(qq)}")

print("\nTesting result.plot_aberrant_per_sample() without out_dir...")
plot = result.plot_aberrant_per_sample()
print(f"  Returned: {type(plot)}")

if protein_ids:
    print(f"\nTesting result.plot_expected_vs_observed('{test_protein}') without out_dir...")
    plot = result.plot_expected_vs_observed(test_protein)
    print(f"  Returned: {type(plot)}")

print("\n✅ All plotting tests passed!")
