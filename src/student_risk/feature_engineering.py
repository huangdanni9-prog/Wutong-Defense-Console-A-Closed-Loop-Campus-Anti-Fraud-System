"""
Feature Engineering Module for High-Risk Student Identification

This module handles data loading, preprocessing, and feature engineering
for the wire fraud risk model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler, LabelEncoder


class StudentFeatureEngineer:
    """Feature engineering for student risk model."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize the feature engineer.
        
        Args:
            data_path: Path to the student_model.csv file
        """
        self.data_path = data_path or self._get_default_path()
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def _get_default_path(self) -> str:
        """Get default path to student data."""
        base = Path(__file__).parent.parent.parent
        return str(base / "Datasets" / "Student" / "Training and Testing Data" / "student_model.csv")
    
    def load_data(self) -> pd.DataFrame:
        """Load the student dataset."""
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} student records with {len(self.df.columns)} columns")
        return self.df
    
    def create_demographic_features(self) -> pd.DataFrame:
        """Create demographic-based risk features."""
        df = self.df.copy()
        
        # Mainland international student (uses 内地居民往来港澳通行证)
        df['is_mainland_student'] = (df['iden_type'] == '内地居民往来港澳通行证').astype(int)
        
        # Hong Kong 港漂 (new mainland arrival working/studying in HK)
        df['is_hk_piao'] = (df['hk_resident_type'] == '新来港内地人港漂').astype(int)
        
        # Other mainland-related resident types
        df['is_mainland_related'] = df['hk_resident_type'].isin([
            '新来港内地人港漂', '新来港内地人非港漂'
        ]).astype(int)
        
        # Age-based risk groups
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 22, 25, 30, 100],
            labels=['18-22', '23-25', '26-30', '31+']
        )
        df['is_young_student'] = (df['age'] <= 22).astype(int)
        
        # Has passport (potential international background)
        df['has_passport'] = (df['iden_type'] == 'Passport /CI').astype(int)
        
        return df
    
    def create_communication_features(self) -> pd.DataFrame:
        """Create communication behavior features."""
        df = self.df.copy()
        
        # Total overseas contact intensity
        df['total_overseas_contact'] = df['total_voice_cnt'] + df['total_msg_cnt']
        
        # Total local unknown contact
        df['total_local_contact'] = df['total_local_voice_cnt'] + df['total_local_msg_cnt']
        
        # Mainland operator call ratio
        total_calls = df['total_voice_cnt'] + df['total_local_voice_cnt'] + 1
        df['mainland_call_ratio'] = df['from_china_mobile_call_cnt'] / total_calls
        
        # Repeated contact indicator (same number calling multiple times)
        df['has_repeated_contact'] = (df['max_voice_cnt'] >= 3).astype(int)
        df['high_repeated_contact'] = (df['max_voice_cnt'] >= 5).astype(int)
        
        # High mainland call frequency
        df['high_mainland_calls'] = (df['from_china_mobile_call_cnt'] >= 10).astype(int)
        
        # High overseas contact frequency
        df['high_overseas_contact'] = (df['total_overseas_contact'] >= 20).astype(int)
        
        return df
    
    def create_fraud_response_features(self) -> pd.DataFrame:
        """Create features related to fraud contact and response patterns."""
        df = self.df.copy()
        
        # Has been contacted by fraud numbers
        df['has_fraud_contact'] = df['fraud_msisdn'].notna().astype(int)
        
        # Total fraud exposure (calls + messages received)
        df['fraud_exposure'] = df['voice_receive'] + df['msg_receive']
        
        # Total fraud response (calls + messages sent to fraud)
        df['fraud_response'] = df['voice_call'] + df['msg_call']
        
        # Fraud response rate (enthusiasm for feedback)
        # Higher rate = more likely to engage with fraud numbers
        df['fraud_response_rate'] = df['fraud_response'] / (df['fraud_exposure'] + 1)
        
        # Has responded to fraud (called or messaged back)
        df['has_responded_fraud'] = (df['fraud_response'] > 0).astype(int)
        
        # High fraud exposure
        df['high_fraud_exposure'] = (df['fraud_exposure'] >= 3).astype(int)
        
        return df
    
    def create_mobility_features(self) -> pd.DataFrame:
        """Create mobility and travel-related features."""
        df = self.df.copy()
        
        # Fill NaN mainland_cnt with 0
        df['mainland_cnt'] = df['mainland_cnt'].fillna(0)
        
        # Travels to mainland frequently
        df['frequent_mainland_travel'] = (df['mainland_to_hk_cnt'] >= 3).astype(int)
        
        # High mainland presence (days in mainland during observation period)
        df['high_mainland_presence'] = (df['mainland_cnt'] >= 30).astype(int)
        
        # Uses mainland apps
        df['uses_mainland_apps'] = (df['app_max_cnt'] > 0).astype(int)
        df['heavy_mainland_app_user'] = (df['app_max_cnt'] >= 14).astype(int)
        
        return df
    
    def engineer_all_features(self) -> pd.DataFrame:
        """Run all feature engineering steps and return enriched dataframe."""
        if self.df is None:
            self.load_data()
        
        # Create all feature sets
        df_demo = self.create_demographic_features()
        df_comm = self.create_communication_features()
        df_fraud = self.create_fraud_response_features()
        df_mobility = self.create_mobility_features()
        
        # Merge all features
        base_cols = ['user_id', 'msisdn']
        
        # Start with original dataframe
        result = self.df.copy()
        
        # Add demographic features
        demo_new_cols = [c for c in df_demo.columns if c not in result.columns]
        for col in demo_new_cols:
            result[col] = df_demo[col]
        
        # Add communication features
        comm_new_cols = [c for c in df_comm.columns if c not in result.columns]
        for col in comm_new_cols:
            result[col] = df_comm[col]
        
        # Add fraud response features
        fraud_new_cols = [c for c in df_fraud.columns if c not in result.columns]
        for col in fraud_new_cols:
            result[col] = df_fraud[col]
        
        # Add mobility features
        mobility_new_cols = [c for c in df_mobility.columns if c not in result.columns]
        for col in mobility_new_cols:
            result[col] = df_mobility[col]
        
        print(f"Engineered {len(result.columns)} total features")
        return result
    
    def get_features_for_clustering(self, df: pd.DataFrame = None) -> Tuple[np.ndarray, List[str]]:
        """
        Get normalized numeric features suitable for clustering.
        
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if df is None:
            df = self.engineer_all_features()
        
        # Select numeric features for clustering
        cluster_features = [
            'age',
            'total_voice_cnt', 'total_msg_cnt',
            'max_voice_cnt',
            'total_local_voice_cnt', 'total_local_msg_cnt',
            'from_china_mobile_call_cnt',
            'mainland_cnt', 'mainland_to_hk_cnt',
            'app_max_cnt',
            'voice_receive', 'voice_call',
            'msg_receive', 'msg_call',
            'total_overseas_contact', 'total_local_contact',
            'fraud_exposure', 'fraud_response', 'fraud_response_rate'
        ]
        
        # Filter to available columns
        available_features = [f for f in cluster_features if f in df.columns]
        
        # Extract and normalize
        X = df[available_features].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, available_features
    
    def get_features_for_rules(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Get binary/categorical features suitable for rule-based scoring.
        
        Returns:
            DataFrame with rule-compatible features
        """
        if df is None:
            df = self.engineer_all_features()
        
        rule_features = [
            'user_id',
            # Demographic
            'is_mainland_student', 'is_hk_piao', 'is_mainland_related',
            'is_young_student', 'has_passport',
            # Communication
            'has_repeated_contact', 'high_repeated_contact',
            'high_mainland_calls', 'high_overseas_contact',
            # Fraud response
            'has_fraud_contact', 'has_responded_fraud', 'high_fraud_exposure',
            'fraud_response_rate',
            # Mobility
            'frequent_mainland_travel', 'high_mainland_presence',
            'uses_mainland_apps', 'heavy_mainland_app_user',
            # Raw values for thresholds
            'age', 'from_china_mobile_call_cnt', 'total_overseas_contact',
            'fraud_exposure', 'fraud_response'
        ]
        
        available_features = [f for f in rule_features if f in df.columns]
        return df[available_features].copy()


def main():
    """Test the feature engineering module."""
    engineer = StudentFeatureEngineer()
    df = engineer.engineer_all_features()
    
    print("\n=== Feature Engineering Summary ===")
    print(f"Total records: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    
    # Show new features
    new_features = [
        'is_mainland_student', 'is_hk_piao', 'is_young_student',
        'total_overseas_contact', 'fraud_response_rate', 'has_responded_fraud'
    ]
    print("\n=== Sample Engineered Features ===")
    for feat in new_features:
        if feat in df.columns:
            print(f"{feat}: {df[feat].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
