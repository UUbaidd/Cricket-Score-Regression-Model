Cricket-Score-Prediction-Pipeline 🏏
Real-time First Innings Score Estimation using XGBoost

A machine learning pipeline designed to predict the final score of a cricket match (T20/ODI) based on real-time match dynamics. Unlike basic predictors that use only Run Rate, this model integrates wicket-loss acceleration and historical venue performance to provide a 90%+ accurate estimate.

Modeling: XGBoost Regressor (chosen for its ability to handle the non-linear relationship between wickets left and scoring acceleration).

Engineering: Scikit-Learn Pipelines, ColumnTransformers, and One-Hot Encoding for categorical features (batting team, city).

Data Processing: NumPy & Pandas for vectorized rolling window calculations.

Feature Engineering (The Secret Sauce): * I implemented a Rolling Window (last_five) sum to capture "current momentum," which standard static snapshots miss.

I calculated wickets_left and balls_left as inverse decay features to help the model learn the "acceleration phase" in the final overs.

Architecture Choice: I used a Pipeline (Step 84 in code) to bundle preprocessing and the model.

Reasoning: This prevents Data Leakage during cross-validation and ensures that the exact same transformations are applied during production inference.

Optimization Trade-off: I chose StandardScaler after One-Hot Encoding.

Reasoning: While XGBoost is tree-based and technically scale-invariant, scaling improves the stability of the pipeline if I decide to swap the regressor for a Neural Network in Phase 2.
