"""
Inference script for Kaggle submission.

This script is executed on Kaggle's environment.
It loads trained models and generates predictions.
"""

import config
import polars as pl

# =============================================================================
# Load Data
# =============================================================================
test_df = pl.read_csv(config.COMP_DATASET_DIR / "test.csv")
sub_df = pl.read_csv(config.COMP_DATASET_DIR / "sample_submission.csv")

# =============================================================================
# Load Model and Predict
# =============================================================================
# Example: Load model from artifact directory
# model_dir = config.ARTIFACT_EXP_DIR(config.EXP_NAME) / "models"
# model = joblib.load(model_dir / "model.joblib")
# predictions = model.predict(test_df[features].to_numpy())

# =============================================================================
# Create Submission
# =============================================================================
# Example: Update submission dataframe with predictions
# sub_df = sub_df.with_columns(pl.Series("target", predictions))

# Save submission file
sub_df.write_csv(config.OUTPUT_DIR / "submission.csv")

print(f"Submission saved to: {config.OUTPUT_DIR / 'submission.csv'}")
print(f"Submission shape: {sub_df.shape}")
