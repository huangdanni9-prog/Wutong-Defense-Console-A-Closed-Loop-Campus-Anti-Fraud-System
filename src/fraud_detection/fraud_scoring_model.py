"""
Fraud Scoring Model

Combines XGBoost (log loss) and Isolation Forest for fraud detection.
Focuses on feature importance for portrait generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, log_loss
import warnings

warnings.filterwarnings('ignore')

# Try to import XGBoost, fallback to RandomForest if not available
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGBOOST = False
    print("Warning: XGBoost not available, using RandomForest")


class FraudScoringModel:
    """
    Hybrid fraud scoring model combining:
    - XGBoost with log loss for probability estimation
    - Isolation Forest for anomaly detection
    
    Emphasizes feature importance for portrait generation.
    """
    
    def __init__(self, contamination: float = 0.05):
        """
        Initialize the scoring model.
        
        Args:
            contamination: Expected proportion of anomalies for Isolation Forest
        """
        self.contamination = contamination
        self.classifier = None
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importance = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FraudScoringModel':
        """
        Fit both models on the training data.
        
        Args:
            X: Feature DataFrame
            y: Target Series (0 = not fraud, 1 = fraud)
            
        Returns:
            Self for chaining
        """
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate class weights for imbalance handling
        n_fraud = y.sum()
        n_normal = len(y) - n_fraud
        scale_pos_weight = n_normal / n_fraud if n_fraud > 0 else 1
        
        print(f"Training on {len(y)} samples ({n_fraud} fraud, {n_normal} non-fraud)")
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Fit XGBoost with log loss
        if HAS_XGBOOST:
            self.classifier = xgb.XGBClassifier(
                objective='binary:logistic',  # Log loss
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                max_depth=5,
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
            self.classifier.fit(X_scaled, y)
            
            # Extract feature importance
            importance = self.classifier.feature_importances_
        else:
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                class_weight='balanced',
                random_state=42
            )
            self.classifier.fit(X_scaled, y)
            importance = self.classifier.feature_importances_
        
        # Store feature importance
        self.feature_importance = dict(zip(self.feature_names, importance))
        
        # Fit Isolation Forest for anomaly detection
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        self.isolation_forest.fit(X_scaled)
        
        print(f"Fitted classifier and Isolation Forest")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability for each sample.
        
        Returns probability of being fraud (0-1).
        """
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)[:, 1]
    
    def predict_anomaly(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly scores using Isolation Forest.
        
        Returns:
            Tuple of (is_anomaly, anomaly_score) where:
            - is_anomaly: 1 if anomaly, 0 if normal
            - anomaly_score: normalized 0-1 (higher = more anomalous)
        """
        X_scaled = self.scaler.transform(X)
        
        predictions = self.isolation_forest.predict(X_scaled)
        is_anomaly = (predictions == -1).astype(int)
        
        # Decision function returns negative for anomalies
        raw_scores = -self.isolation_forest.decision_function(X_scaled)
        # Normalize to 0-1
        normalized_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)
        
        return is_anomaly, normalized_scores
    
    def predict_combined_score(self, X: pd.DataFrame, 
                               classifier_weight: float = 0.7,
                               anomaly_weight: float = 0.3) -> np.ndarray:
        """
        Combine classifier probability and anomaly score.
        
        Args:
            X: Feature DataFrame
            classifier_weight: Weight for classifier probability
            anomaly_weight: Weight for anomaly score
            
        Returns:
            Combined fraud score (0-100)
        """
        fraud_proba = self.predict_proba(X)
        _, anomaly_score = self.predict_anomaly(X)
        
        combined = (fraud_proba * classifier_weight + anomaly_score * anomaly_weight) * 100
        return combined
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate model performance.
        
        Returns dict with AUC-ROC, precision, recall, F1, log loss.
        """
        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= 0.5).astype(int)
        
        metrics = {
            'auc_roc': roc_auc_score(y, y_proba),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'log_loss': log_loss(y, y_proba),
        }
        return metrics
    
    def get_feature_importance(self, top_n: int = None) -> pd.DataFrame:
        """
        Get feature importance sorted by importance.
        
        Args:
            top_n: Return only top N features (None for all)
            
        Returns:
            DataFrame with feature names and importance scores
        """
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importance.items()
        ])
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def generate_portrait_insights(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Generate portrait insights based on feature importance and statistics.
        
        Compares fraud vs non-fraud for each important feature.
        """
        insights = {}
        
        # Get top features
        top_features = self.get_feature_importance(top_n=10)
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            if feature in X.columns:
                fraud_mean = X.loc[y == 1, feature].mean()
                non_fraud_mean = X.loc[y == 0, feature].mean()
                
                if non_fraud_mean > 0:
                    lift = fraud_mean / non_fraud_mean
                else:
                    lift = fraud_mean if fraud_mean > 0 else 1
                
                insights[feature] = {
                    'importance': row['importance'],
                    'fraud_mean': fraud_mean,
                    'non_fraud_mean': non_fraud_mean,
                    'lift': lift,
                }
        
        return insights
    
    # ==============================================================
    # SHAP EXPLAINABILITY - "Why is this user flagged?"
    # ==============================================================
    
    def init_shap_explainer(self, X_background: pd.DataFrame = None):
        """
        Initialize SHAP TreeExplainer for model explanations.
        
        Args:
            X_background: Background data for SHAP (uses training sample if not provided)
        """
        try:
            import shap
            
            if X_background is not None:
                X_scaled = self.scaler.transform(X_background)
            else:
                X_scaled = None
            
            self.shap_explainer = shap.TreeExplainer(self.classifier)
            print("SHAP TreeExplainer initialized")
            return True
        except ImportError:
            print("Warning: SHAP not installed. Run: pip install shap")
            self.shap_explainer = None
            return False
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")
            self.shap_explainer = None
            return False
    
    def explain_prediction(self, X_sample: pd.DataFrame, plot: bool = False, 
                          save_path: str = None) -> dict:
        """
        Generate SHAP values to explain a specific prediction.
        
        Args:
            X_sample: A single row DataFrame (the user to explain)
            plot: If True, generates a waterfall plot
            save_path: If provided, saves the plot to this path
            
        Returns:
            Dict with 'shap_values', 'feature_contributions', and optionally 'plot_figure'
        """
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'shap_explainer') or self.shap_explainer is None:
            self.init_shap_explainer()
        
        if self.shap_explainer is None:
            return {'error': 'SHAP explainer not available'}
        
        try:
            import shap
            
            # Scale the sample
            X_scaled = self.scaler.transform(X_sample)
            
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(X_scaled)
            
            # For binary classification, get values for positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            # Create feature contribution dict
            contributions = {}
            for i, feature in enumerate(self.feature_names):
                contributions[feature] = float(shap_values[0][i])
            
            # Sort by absolute impact
            sorted_contributions = dict(sorted(
                contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            ))
            
            result = {
                'shap_values': shap_values[0].tolist(),
                'feature_contributions': sorted_contributions,
                'base_value': float(self.shap_explainer.expected_value[1] if isinstance(
                    self.shap_explainer.expected_value, list) else self.shap_explainer.expected_value),
                'prediction': float(self.predict_proba(X_sample)[0])
            }
            
            # Generate waterfall plot
            if plot:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create SHAP Explanation object for waterfall
                explanation = shap.Explanation(
                    values=shap_values[0],
                    base_values=result['base_value'],
                    data=X_scaled[0],
                    feature_names=self.feature_names
                )
                
                shap.plots.waterfall(explanation, show=False)
                
                if save_path:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"SHAP waterfall saved to: {save_path}")
                
                result['plot_figure'] = plt.gcf()
                
                if not save_path:
                    plt.show()
                plt.close()
            
            return result
            
        except Exception as e:
            return {'error': f'SHAP explanation failed: {str(e)}'}
    
    def generate_shap_summary(self, X: pd.DataFrame, save_path: str = None):
        """
        Generate SHAP summary plot showing global feature importance.
        
        Args:
            X: Feature DataFrame (sample of data for visualization)
            save_path: If provided, saves the plot to this path
        """
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'shap_explainer') or self.shap_explainer is None:
            self.init_shap_explainer()
        
        if self.shap_explainer is None:
            print("SHAP explainer not available")
            return None
        
        try:
            import shap
            
            X_scaled = self.scaler.transform(X)
            shap_values = self.shap_explainer.shap_values(X_scaled)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_scaled, feature_names=self.feature_names, show=False)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"SHAP summary saved to: {save_path}")
            else:
                plt.show()
            
            plt.close()
            return shap_values
            
        except Exception as e:
            print(f"SHAP summary generation failed: {e}")
            return None


def train_fraud_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[FraudScoringModel, Dict]:
    """
    Train and evaluate fraud scoring model.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion for test set
        
    Returns:
        Tuple of (trained model, evaluation metrics)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n=== Training Fraud Model ===")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    model = FraudScoringModel()
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n=== Evaluation ===")
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    print(f"Train - AUC: {train_metrics['auc_roc']:.4f}, Log Loss: {train_metrics['log_loss']:.4f}")
    print(f"Test  - AUC: {test_metrics['auc_roc']:.4f}, Log Loss: {test_metrics['log_loss']:.4f}")
    print(f"Test  - Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")
    
    # Feature importance
    print("\n=== Top 10 Features ===")
    print(model.get_feature_importance(top_n=10).to_string(index=False))
    
    return model, test_metrics


def main():
    """Test the fraud scoring model."""
    from fraud_feature_engineering import FraudFeatureEngineer
    
    # Load and engineer features
    engineer = FraudFeatureEngineer()
    X, y = engineer.get_train_data()
    
    # Train model
    model, metrics = train_fraud_model(X, y)
    
    # Generate portrait insights
    print("\n=== Portrait Insights ===")
    insights = model.generate_portrait_insights(X, y)
    for feature, stats in insights.items():
        print(f"{feature}: Fraud={stats['fraud_mean']:.2f}, Non-Fraud={stats['non_fraud_mean']:.2f}, Lift={stats['lift']:.2f}x")


if __name__ == "__main__":
    main()
