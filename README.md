# Airline Passenger Satisfaction: Prediction & Analytics App

An interactive Streamlit application built from the *Airline Passenger Satisfaction Prediction* notebook. It predicts whether a passenger
is **satisfied** or **neutral/dissatisfied** with **XGBoost** or **LightGBM**, explains every individual prediction with SHAP, turns the
explanation into prioritised improvement notes, and offers a filterable analytics dashboard.

## Business problem
Passenger satisfaction depends on many interacting factors (service quality, schedule, delays, trip type). With hundreds of thousands of
records it is hard to see which matter. A model can estimate who is likely to be dissatisfied and show which factors drove each estimate, so
teams can decide where to investigate. *What is demonstrated:* predictive accuracy on held-out data and per-prediction explanations.
*What is not demonstrated:* any real-world effect (retention, revenue) of acting on the results.

## Dataset
`train.csv` from the notebook: 699,635 rows x 23 columns, target `satisfaction` (True = satisfied, 44.4% of rows). 292 rows (0.04%) have no
`Arrival Delay in Minutes` and are dropped as in the notebook; no duplicate rows. Service ratings are 0-5 (0 occurs in 12 of 13 columns and is kept as given).
The 21 predictors are Gender, Age, Customer Type, Type of Travel, Class, Flight Distance, 13 service ratings and 2 delays; `id` is excluded.

## Pipeline (reproduces the notebook)
1. Drop duplicates, then rows with missing values.
2. Encode Gender (Male 0/Female 1), Customer Type (Loyal 0/disloyal 1), Type of Travel (Business 0/Personal 1), Class (Eco 0/Eco Plus 1/Business 2).
3. Engineer `Total_Delay` (departure + arrival), `Delay_Diff` (arrival - departure), `Average_Service_Rating` (mean of 13 ratings).
4. 24 model features in a fixed order (`src/preprocessing.py`, shared by training and inference).
5. Stratified 80/20 split (`random_state=42`); both models use the notebook's hyperparameters
   (`n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8`). Nothing is tuned on the test set.

## Verified results
Held-out test set: 139,869 passengers, recomputed by `train_models.py` (positive class = Satisfied):

| Metric | XGBoost | LightGBM |
|---|---|---|
| Accuracy | 0.9244 | 0.9241 |
| Precision | 0.9259 | 0.9251 |
| Recall | 0.9016 | 0.9018 |
| F1 | 0.9136 | 0.9133 |
| ROC-AUC | 0.9570 | 0.9569 |
| Train - test accuracy gap | 0.0020 | 0.0016 |

The two models are effectively tied on quality (differences < 0.001). They agree on 99.56% of test passengers. These are consistent with the notebook's rounded values
(0.92 accuracy, 0.90 recall, 0.91 F1; precision 0.93/0.92). Small differences come from the stratified split; e.g. LightGBM precision is 0.925 here, which rounds to 0.93. Timing and size figures vary by machine and are shown in the app.
The notebook's final LightGBM was refit on 100% of the data for a Kaggle submission, which leaves no hold-out; this project evaluates models trained on the 80% split instead.

## Findings from reviewing the notebook
- **Noted:** the notebook also reads a `test.csv` that was not supplied (unlabeled, not needed for evaluation). The notebook's split was unstratified; this project stratifies (changed).
- **Changed:** the notebook's final model was fit on all rows, so it could not be evaluated honestly; the app evaluates held-out models.
- **Noted, left as-is:** `subsample=0.8` is passed to LightGBM without `subsample_freq`, so LightGBM ignores it (kept to match the notebook).
- **Noted:** engineered features did not improve accuracy in the notebook (0.92 before and after).
- **Noted:** the correlation heatmap in the notebook includes `id` and encoded categoricals; the app excludes `id` and warns about encoded categoricals.
- **Assumption:** `True` = satisfied. The notebook dictionary names the classes but does not state the boolean mapping; satisfied passengers rate services much higher, which supports it.
- **Not verified:** probability calibration (Brier scores are reported); the meaning of rating 0; the units of Flight Distance.

## Features
- **Project Overview:** business case, workflow, dataset facts computed from the data, feature engineering, model cards, metric comparison (tables, ROC, confusion matrices, generalisation, cost) with an explicit ranking criterion.
- **Prediction Center:** model selector, validated form, probability estimates, SHAP local contributions, ranked improvement notes (priority, direction), side-by-side model comparison, CSV report download.
- **Analytics & Insights:** KPIs (full vs filtered), target balance, service-rating analysis, segments, delays, correlations, engineered features, global importance (native gain, SHAP, permutation), sidebar filters with reset, automated insights.

## Project structure
```
app.py                       entry point (navigation)
pages/1_Project_Overview.py
pages/2_Prediction_Center.py
pages/3_Analytics_Insights.py
src/preprocessing.py         cleaning, encoding, feature engineering (shared by training & inference)
src/model_evaluation.py      reproducible training, evaluation, artifact saving
src/prediction.py            artifact loading, validation, prediction
src/explainability.py        SHAP contributions, improvement notes, report builder
src/analytics.py             dataset loading, filters, charts, automated insights
src/utils.py                 theme and UI helpers
models/                      xgboost_model.joblib, lightgbm_model.joblib, preprocessing_pipeline.joblib, training_report.json
data/                        train.zip (see data/README.md)
tests/test_app.py            pytest suite
train_models.py              CLI to (re)train and save artifacts
.streamlit/config.toml       theme
```

## Install and run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Preparing model artifacts
Artifacts are included. To regenerate them (e.g. after changing library versions):
```bash
python train_models.py --data data/train.zip     # or data/train.csv; ~1 minute
```
This rewrites everything in `models/`, including `training_report.json`, which the Overview and Analytics pages read.

## Testing
```bash
pytest -q
```
Covers preprocessing consistency, feature order, model loading, label mapping, probability output, SHAP additivity, recommendation rules, CSV export, filters, and page startup/interaction via Streamlit's `AppTest`.

## Deploy to Streamlit Community Cloud
1. Push the project to a GitHub repository (see below).
2. Go to share.streamlit.io, choose **New app**, select the repo and branch.
3. Set **Main file path** to `app.py`. Python 3.11 or 3.12 recommended.
4. Deploy. No secrets are required.
5. If a page reports missing artifacts, confirm `models/*.joblib` and `models/training_report.json` are committed; if Analytics reports a missing dataset, confirm `data/train.zip` is committed.

### Large model files
The current models are under 1 MB each. If you retrain much larger ones (GitHub limit: 100 MB/file), use Git LFS, a GitHub Release asset, or cloud storage
(download at startup into `models/` with credentials in `.streamlit/secrets.toml`, which is git-ignored; on Streamlit Cloud set them under **App settings > Secrets**).

## Upload to GitHub
```bash
git init
git add .
git commit -m "Airline passenger satisfaction Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-user>/airline-satisfaction-app.git
git push -u origin main
```
Check the dataset licence before committing `data/train.zip`.

## Limitations
- Predictions come from one dataset and describe associations; they do not show that changing a factor would change satisfaction.
- Class probabilities are model estimates; calibration is unverified.
- Improvement notes are rule-based on top of SHAP values, with fixed priority thresholds (25% / 10% of negative contribution); they are guidance for investigation, not prescriptions.
- Hyperparameters were not tuned. Timings are from the machine that trained the models.
- The app has not been deployed by the author of this README; deployment steps above are untested instructions.

## Screenshots
Not included yet. To add them: run the app, capture each page, save as `docs/overview.png`, `docs/prediction.png`, `docs/analytics.png`, and reference them here.
