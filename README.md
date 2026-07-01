# TikTok Verified Creator Prediction — Full Data Science Pipeline

## Project Summary
This project builds a Logistic Regression model to predict whether a TikTok video's creator is Verified or Not Verified, using video-level metadata and engagement metrics. The primary business goal is to help TikTok's Operations team (Maika Abadi) prioritize a growing backlog of user reports by identifying high-risk content from verified creators — who are more likely to post Claims requiring moderation — allowing for more efficient resource allocation and reduced manual review time.

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook tiktok_verified_prediction.ipynb
```
Run all cells in order. The dataset `tiktok_dataset.csv` must be in the same folder as the notebook.

## Key Findings
- **Severe class imbalance**: Only ~8% of creators are Verified — addressed using upsampling (resample with replacement) on training data only, with the test set kept at the real-world distribution to prevent data leakage.
- **Top predictors**: Video engagement volume (views, likes, shares) is the strongest predictor of verified status. Transcription length (`text_length`) and author ban status are meaningful secondary signals.
- **Final model**: Logistic Regression tuned for verified-class Recall achieves **70.5% Recall** and **ROC-AUC = 0.853** — correctly catching 217 of 308 verified creators in the held-out test set.

## Final Performance (3,724-record test set)

| Metric | Score |
|--------|-------|
| Accuracy | 78.2% |
| Precision (verified) | 23.1% |
| **Recall (verified)** | **70.5%** |
| F1 Score | 34.8% |
| ROC-AUC | 0.853 |

*Recall is the primary metric — in content moderation, missing a verified-creator report (false negative) is costlier than an extra review (false positive).*

## Pipeline Steps
1. **Discovery & Cleaning** — shape, dtypes, null handling, outlier retention
2. **Feature Engineering** — `text_length` from transcription text; class imbalance documented
3. **Data Preparation** — 80/20 stratified split (random_state=42) → upsample minority in training only
4. **Technical Preprocessing** — correlation heatmap (drop |r|>0.85 pairs), One-Hot Encoding (drop_first=True), StandardScaler
5. **Modeling** — Baseline (`class_weight='balanced'`) vs Resampled Logistic Regression
6. **Evaluation** — Confusion Matrix, Classification Report, ROC-AUC
7. **Interpretability** — Top 5 positive/negative coefficients ranked; 200-word business memo

## Business Recommendation
Use the model's predicted probability as a **risk score** to sort the report backlog: predicted probability > 0.60 → priority queue for senior moderator review. Retrain quarterly as creator patterns evolve.

## Files
- `tiktok_verified_prediction.ipynb` — fully executed notebook (final cell shows Recall + business memo)
- `tiktok_dataset.csv` — dataset (19,382 TikTok video records)
- `requirements.txt` — Python dependencies
- `README.md` — this file
