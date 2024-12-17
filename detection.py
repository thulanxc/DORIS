import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier
import joblib
import pickle

# Define constants
DATA_DIR = Path('results')
MODEL_OUTPUT_DIR = Path('models')
RESULTS_OUTPUT_DIR = Path('results')

def load_features(dataset_type: str) -> tuple:
    """
    Load features from all three extractors.
    
    Args:
        dataset_type: Either 'train' or 'test'
        
    Returns:
        tuple: (depression_features, mood_features, history_features, labels)
    """
    # Load depression features (F^DC)
    with open(DATA_DIR / f'{dataset_type}_symptom_features.pkl', 'rb') as f:
        depression_df = pickle.load(f)
    
    # Load mood course features (F^MC)
    with open(DATA_DIR / f'{dataset_type}_mood_course_features.pkl', 'rb') as f:
        mood_features = pickle.load(f)
        
    # Load post history features (F^PH)
    with open(DATA_DIR / f'{dataset_type}_post_history_features.pkl', 'rb') as f:
        history_features = pickle.load(f)
        
    return depression_df, mood_features, history_features

def fuse_features(depression_df: pd.DataFrame, 
                 mood_features: dict,
                 history_features: dict) -> tuple:
    """
    Fuse features according to the paper's methodology:
    F = Concat(F^MC + F^PH, F^DC)
    
    Args:
        depression_df: DataFrame with depression features and labels
        mood_features: Dictionary of mood course features by user_id
        history_features: Dictionary of post history features by user_id
        
    Returns:
        tuple: (fused_features, labels)
    """
    fused_features = []
    labels = []
    
    # Get depression feature columns (F^DC)
    depression_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    
    for user_id, row in depression_df.iterrows():
        if user_id in mood_features and user_id in history_features:
            # Get F^MC and F^PH features
            mood_feat = mood_features[user_id]
            history_feat = history_features[user_id]
            
            # Sum F^MC and F^PH as they share the same space
            combined_feat = mood_feat + history_feat
            
            # Get F^DC features
            depression_feat = np.array([row[col] for col in depression_cols])
            
            # Concatenate (F^MC + F^PH) with F^DC
            # F = Concat(F^MC + F^PH, F^DC)
            final_feature = np.concatenate([combined_feat, depression_feat])
            
            fused_features.append(final_feature)
            labels.append(row['is_positive'])
    
    return np.array(fused_features), np.array(labels)

def main():
    """Main execution function for the depression detection pipeline."""
    # Load features
    train_depression, train_mood, train_history = load_features('train')
    test_depression, test_mood, test_history = load_features('test')
    
    # Fuse features according to the paper's methodology
    X_train, y_train = fuse_features(train_depression, train_mood, train_history)
    X_test, y_test = fuse_features(test_depression, test_mood, test_history)
    
    # Feature standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train XGBoost classifier (Gradient Boosting Trees as specified)
    xgb_clf = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=200,    # Number of trees (M in the paper)
        max_depth=6,
        learning_rate=0.1,   # Learning rate (ν in the paper)
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_clf.fit(X_train_scaled, y_train)
    
    # Save trained model and scaler
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_clf, MODEL_OUTPUT_DIR / 'xgb_model.joblib')
    joblib.dump(scaler, MODEL_OUTPUT_DIR / 'scaler.joblib')
    
    # Generate predictions
    y_pred = xgb_clf.predict(X_test_scaled)
    y_pred_proba = xgb_clf.predict_proba(X_test_scaled)[:, 1]
    
    # Save predictions
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_predictions = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': y_pred,
        'Predicted_Probability': y_pred_proba
    })
    test_predictions.to_csv(RESULTS_OUTPUT_DIR / 'test_predictions.csv', index=False)
    
    # Evaluate and print metrics
    metrics = {
        'F1': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'AUROC': roc_auc_score(y_test, y_pred_proba),
        'AUPRC': average_precision_score(y_test, y_pred_proba)
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()