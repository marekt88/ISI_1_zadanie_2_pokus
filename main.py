"""
Regression Task Implementation using Ridge Model

This script implements a regression solution using Ridge model from scikit-learn.
The goal is to achieve R² score above 0.90 on the training data.

Data files:
- X_public.npy: Training features (NxF matrix)
- y_public.npy: Training targets (N-length vector)
- X_eval.npy: Evaluation features for prediction

Author: Cline
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

def get_all_categories(X_train, X_eval):
    """Get all unique categories from both training and evaluation data."""
    all_categories = []
    for i in range(10):  # First 10 columns are categorical
        unique_cats = set(np.concatenate([X_train[:, i], X_eval[:, i]]))
        all_categories.append(unique_cats)
    return all_categories

def analyze_feature_importance(X, y):
    """Analyze feature importance using mutual information."""
    mi_scores = mutual_info_regression(X, y)
    return mi_scores

def create_interaction_features(X_numerical, top_features_idx):
    """Create interaction features between top important features."""
    interactions = []
    for i in range(len(top_features_idx)):
        for j in range(i + 1, len(top_features_idx)):
            idx1, idx2 = top_features_idx[i], top_features_idx[j]
            interaction = X_numerical[:, idx1] * X_numerical[:, idx2]
            interactions.append(interaction)
    return np.column_stack(interactions) if interactions else np.array([]).reshape(X_numerical.shape[0], 0)

def preprocess_data(X, all_categories=None, is_training=True):
    """Preprocess the data by handling categorical variables and missing values."""
    # Split data into categorical and numerical columns
    categorical_cols = X[:, :10].copy()  # First 10 columns are categorical
    numerical_cols = X[:, 10:].astype(float)  # Rest are numerical
    
    # Handle categorical variables
    if is_training:
        label_encoders = []
        for i in range(10):
            le = LabelEncoder()
            le.fit(list(all_categories[i]))
            categorical_cols[:, i] = le.transform(categorical_cols[:, i])
            label_encoders.append(le)
        preprocess_data.label_encoders = label_encoders
    else:
        for i in range(10):
            categorical_cols[:, i] = preprocess_data.label_encoders[i].transform(categorical_cols[:, i])
    
    # Handle missing values in numerical columns
    if is_training:
        imputer = SimpleImputer(strategy='median')
        numerical_cols = imputer.fit_transform(numerical_cols)
        preprocess_data.imputer = imputer
        
        # Analyze feature importance
        mi_scores = analyze_feature_importance(numerical_cols, preprocess_data.y_train)
        # Select top 30 important features
        top_features_idx = np.argsort(mi_scores)[-30:]
        preprocess_data.top_features_idx = top_features_idx
        
        # Create interactions only between important features
        interactions = create_interaction_features(numerical_cols, top_features_idx)
        preprocess_data.n_interactions = interactions.shape[1]
        
    else:
        numerical_cols = preprocess_data.imputer.transform(numerical_cols)
        interactions = create_interaction_features(numerical_cols, preprocess_data.top_features_idx)
    
    # Combine all features
    if interactions.size > 0:
        X_processed = np.hstack([categorical_cols, numerical_cols, interactions])
    else:
        X_processed = np.hstack([categorical_cols, numerical_cols])
    
    return X_processed

def train_model(X_train, y_train):
    """Train the Ridge regression model with hyperparameter tuning."""
    # Initialize scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Define parameter grid
    param_grid = {
        'alpha': np.logspace(-2, 3, 30),
        'solver': ['auto', 'svd', 'cholesky', 'lsqr'],
        'fit_intercept': [True],
        'positive': [False]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(
        Ridge(max_iter=10000),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation R² score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, scaler

def evaluate_model(model, X_train, y_train, scaler):
    """Evaluate model performance using R² score."""
    X_train_scaled = scaler.transform(X_train)
    y_pred = model.predict(X_train_scaled)
    r2 = r2_score(y_train, y_pred)
    print(f"R² score on training data: {r2:.4f}")
    return r2

def generate_predictions(model, X_eval, scaler):
    """Generate predictions for evaluation data."""
    X_eval_scaled = scaler.transform(X_eval)
    y_pred = model.predict(X_eval_scaled)
    np.save('y_predikcia.npy', y_pred)
    print("Predictions saved to y_predikcia.npy")

def main():
    """Main execution function"""
    print("Loading data...")
    X_train = np.load('X_public.npy', allow_pickle=True)
    y_train = np.load('y_public.npy', allow_pickle=True)
    X_eval = np.load('X_eval.npy', allow_pickle=True)
    
    print("\nGathering all unique categories...")
    all_categories = get_all_categories(X_train, X_eval)
    
    # Store y_train for feature importance analysis
    preprocess_data.y_train = y_train
    
    print("\nPreprocessing training data...")
    X_train_processed = preprocess_data(X_train, all_categories, is_training=True)
    
    print("\nTraining model...")
    model, scaler = train_model(X_train_processed, y_train)
    
    print("\nEvaluating model...")
    r2 = evaluate_model(model, X_train_processed, y_train, scaler)
    
    print("\nPreprocessing evaluation data...")
    X_eval_processed = preprocess_data(X_eval, is_training=False)
    
    print("\nGenerating predictions...")
    generate_predictions(model, X_eval_processed, scaler)

if __name__ == "__main__":
    main()
