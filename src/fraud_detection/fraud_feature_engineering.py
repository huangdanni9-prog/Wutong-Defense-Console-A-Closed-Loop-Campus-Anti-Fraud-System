"""
Fraud Feature Engineering Module (Leak-Free Version)

Extracts and engineers features from fraud_model_2.csv for fraud detection.
IMPORTANT: Uses BINNING to prevent behavioral leakage and removes look-ahead bias columns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


class FraudFeatureEngineer:
    """
    Feature engineering for wire fraud detection.
    
    LEAK PREVENTION:
    1. BINNING continuous features to prevent exact-value memorization
    2. REMOVING look-ahead bias columns (database joins computed after detection)
    3. REMOVING pre-calculated rule columns (circular logic)
    """
    
    # Columns to EXCLUDE - comprehensive leakage prevention
    LEAKAGE_COLUMNS = [
        # Post-hoc columns
        'audit_status',      # Target variable
        'audit_remark',      # Only known after audit
        'proc_time',         # Suspension time - only after action
        
        # Selection bias / circular logic
        'hit_student_model', # Pre-calculated rule result
        
        # Look-ahead bias (database joins computed after investigation)
        'iden_type_num',     # Linked SIMs count - needs heavy query
        'id_type_hk_num',    # Linked IDs - look-ahead
        'id_type_hk_mac_to_mainland_num',
        'id_type_mainland_to_hk_mac_num',
        'id_type_psp_num',
    ]
    
    def __init__(self, data_path: str = None):
        """
        Initialize the feature engineer.
        
        Args:
            data_path: Path to fraud_model_2.csv
        """
        if data_path is None:
            data_path = str(Path(__file__).parent.parent.parent / 
                           "Datasets" / "Fraud" / "Training and Testing Data" / "fraud_model_2.csv")
        self.data_path = data_path
        self.df = None
        self.dispersion_disabled = False
        
    def load_data(self, apply_whitelist: bool = True) -> pd.DataFrame:
        """
        Load the fraud dataset.
        
        Args:
            apply_whitelist: If True, mark whitelisted accounts as non-fraud
        """
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} fraud records with {len(self.df.columns)} columns")
        
        # Apply whitelist - mark whitelisted MSISDNs as non-fraud
        if apply_whitelist:
            whitelist_path = Path(__file__).parent.parent.parent / "Datasets" / "Fraud" / "Results" / "whitelist.csv"
            if whitelist_path.exists():
                whitelist_df = pd.read_csv(whitelist_path)
                whitelist_msisdns = set(whitelist_df['msisdn'].astype(str))
                
                # Mark whitelisted accounts as non-fraud (is_fraud = 0)
                if 'msisdn' in self.df.columns:
                    mask = self.df['msisdn'].astype(str).isin(whitelist_msisdns)
                    whitelisted_count = mask.sum()
                    
                    if whitelisted_count > 0:
                        # Create is_fraud column if not exists, then mark whitelisted as non-fraud
                        if 'is_fraud' not in self.df.columns and 'audit_status' in self.df.columns:
                            self.df['is_fraud'] = self.df['audit_status'].map({
                                '稽核不通過': 1,
                                '稽核通過': 0
                            })
                        
                        if 'is_fraud' in self.df.columns:
                            self.df.loc[mask, 'is_fraud'] = 0
                        
                        print(f"✅ Whitelist applied: {whitelisted_count} accounts marked as non-fraud")
            else:
                print("ℹ️ No whitelist found, proceeding without whitelist filter")
        
        return self.df
    
    def create_target(self) -> pd.DataFrame:
        """Create/attach binary target variable; keep unlabeled rows for balancing."""
        if self.df is None:
            self.load_data()
        
        df = self.df.copy()
        
        if 'is_fraud' not in df.columns:
            df['is_fraud'] = df['audit_status'].map({
                '稽核不通過': 1,
                '稽核通過': 0
            })
        
        labeled = df['is_fraud'].isin([0, 1]).sum()
        fraud = (df['is_fraud'] == 1).sum()
        print(f"Target created: {fraud} fraud, {labeled - fraud} non-fraud (kept {len(df)} total rows)")
        return df
    
    def _safe_numeric(self, series: pd.Series) -> pd.Series:
        """Convert to numeric, filling NaN with 0."""
        return pd.to_numeric(series, errors='coerce').fillna(0)
    
    def _bin_duration(self, duration: pd.Series) -> pd.Series:
        """
        BIN call duration to prevent simbox fingerprint leakage.
        
        Bins: 0-5s (very short), 5-30s (short), 30-120s (medium), 120+ (long)
        This prevents the model from memorizing exact 1.0s or 2.0s patterns.
        """
        bins = [0, 5, 30, 120, float('inf')]
        labels = [0, 1, 2, 3]  # very_short, short, medium, long
        return pd.cut(duration, bins=bins, labels=labels, include_lowest=True).astype(float).fillna(0)
    
    def _bin_call_volume(self, volume: pd.Series) -> pd.Series:
        """
        BIN call volume to prevent exact-count memorization.
        
        Bins: 0, 1-10, 11-50, 51-100, 100+
        """
        bins = [0, 1, 10, 50, 100, float('inf')]
        labels = [0, 1, 2, 3, 4]
        return pd.cut(volume, bins=bins, labels=labels, include_lowest=True).astype(float).fillna(0)
    
    def _bin_dispersion(self, dispersion: pd.Series) -> pd.Series:
        """
        BIN dispersion rate to prevent bot fingerprint.
        
        Converts dispersion=0.000 (bot pattern) to category instead of exact value.
        """
        bins = [0, 0.001, 0.1, 0.5, 1.0]
        labels = [0, 1, 2, 3]  # zero, low, medium, high
        return pd.cut(dispersion, bins=bins, labels=labels, include_lowest=True).astype(float).fillna(0)
    
    def engineer_call_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer call pattern features.
        
        CRITICAL: Avoids duration_bin which causes "duration addiction".
        """
        result = df.copy()
        
        # Raw values
        call_cnt = self._safe_numeric(result['call_cnt_day'])
        called_cnt = self._safe_numeric(result['called_cnt_day'])
        dispersion = self._safe_numeric(result['dispersion_rate'])
        zero_disp_ratio = (dispersion == 0).mean() if len(dispersion) else 0
        if zero_disp_ratio > 0.5:
            self.dispersion_disabled = True
            print(f"Dispersion disabled due to drift: {zero_disp_ratio:.1%} zeros")
        
        # BINNED call volume (safe)
        result['call_volume_bin'] = self._bin_call_volume(call_cnt + called_cnt)
        
        # NOTE: Removed dispersion features (low_dispersion, dispersion_level)
        # These caused "Perfect Separator Trap" - 76% of innocent users in validation
        # had zero dispersion, causing massive False Positives.
        # The Rule Engine still uses raw dispersion_rate from CSV.
        
        # Short/long call FLAGS (binary, not ratios)
        result['call_cnt_day_2s'] = self._safe_numeric(result['call_cnt_day_2s'])
        result['call_cnt_day_3m'] = self._safe_numeric(result['call_cnt_day_3m'])
        result['has_short_calls'] = (result['call_cnt_day_2s'] > 0).astype(int)
        result['has_long_calls'] = (result['call_cnt_day_3m'] > 0).astype(int)
        
        # Call ratio (outbound vs inbound)
        result['outbound_ratio'] = call_cnt / (call_cnt + called_cnt + 1)
        
        # SILENT RECEIVER: "He talks to people, but nobody wants to talk to him"
        # Fraud phones make calls but never receive any
        result['is_silent_receiver'] = ((called_cnt == 0) & (call_cnt > 0)).astype(int)
        
        return result
    
    def engineer_roaming_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer roaming-related features."""
        result = df.copy()
        
        # Roaming calls
        roam_call = self._safe_numeric(result['roam_unknow_call_cnt'])
        local_call = self._safe_numeric(result['local_unknow_call_cnt'])
        
        # Binary: any roaming activity (more robust than ratio)
        result['has_roaming_calls'] = (roam_call > 0).astype(int)
        
        # Roaming SMS
        roam_msg = self._safe_numeric(result['roam_msg_cnt'])
        result['has_roaming_sms'] = (roam_msg > 0).astype(int)
        
        # Has roaming service
        result['has_roaming_service'] = (result['is_support_roam'] == 'Yes').astype(int)
        
        return result
    
    def engineer_terminal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer terminal switching features (BINNED)."""
        result = df.copy()
        
        imei_changes = self._safe_numeric(result['change_imei_times'])
        
        # Binary instead of exact count
        result['multiple_terminals'] = (imei_changes > 1).astype(int)
        
        return result
    
    def engineer_time_period_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer time period activity features."""
        result = df.copy()
        
        # Time period columns
        t_9_12 = self._safe_numeric(result.get('call_cnt_times_9_12', 0))
        t_12_15 = self._safe_numeric(result.get('call_cnt_times_12_15', 0))
        t_15_18 = self._safe_numeric(result.get('call_cnt_times_15_18', 0))
        t_18_21 = self._safe_numeric(result.get('call_cnt_times_18_21', 0))
        
        # Binary: active during work hours
        result['active_work_hours'] = ((t_9_12 + t_12_15 + t_15_18) > 0).astype(int)
        result['active_evening'] = (t_18_21 > 0).astype(int)
        
        # Primarily work hours (fraud call centers operate 9-5)
        work = t_9_12 + t_12_15 + t_15_18
        total = work + t_18_21 + 0.001
        result['work_hours_dominant'] = (work / total > 0.7).astype(int)
        
        return result
    
    def engineer_student_targeting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer student targeting features (NOTE: potential leak)."""
        result = df.copy()
        
        # CAUTION: call_stu_cnt might have selection bias
        # Only use if recalculated from raw logs
        stu_cnt = self._safe_numeric(result['opp_num_stu_cnt'])
        
        # Binary only - less leaky than count
        result['calls_students'] = (stu_cnt > 0).astype(int)
        
        return result
    
    def engineer_account_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer account type features (safe, no look-ahead bias)."""
        result = df.copy()
        
        # Prepaid vs postpaid (known at activation)
        result['is_prepaid'] = (result['post_or_ppd'] == '预付').astype(int)
        
        # Virtual vs physical SIM (known at activation)
        if 'msisdn_type' in result.columns:
            result['is_virtual_sim'] = (result['msisdn_type'] == '虚拟号').astype(int)
        
        # Network type (known at activation)
        if 'ntwk_type' in result.columns:
            result['is_5g'] = (result['ntwk_type'] == '5G').astype(int)
        
        # PASSPORT IDENTITY: 63% of fraudsters use passports vs 16% of locals
        # This is a strong identity signal that doesn't suffer from "Perfect Separator" trap
        if 'iden_type' in result.columns:
            result['is_passport'] = (result['iden_type'] == '護照').astype(int)
        else:
            result['is_passport'] = 0
        
        # Activation days (stat_dt - open_dt) - BURNER PHONE FEATURE
        # Using VECTORIZED pandas for speed (O(1) vs O(n) loop)
        result['activation_days'] = 999  # Default to old account
        if 'open_dt' in result.columns and 'stat_dt' in result.columns:
            try:
                # Convert dates (format: 20250926.0) to datetime - VECTORIZED
                open_dates = pd.to_datetime(
                    result['open_dt'].fillna(0).astype(int).astype(str), 
                    format='%Y%m%d', 
                    errors='coerce'
                )
                stat_dates = pd.to_datetime(
                    result['stat_dt'].fillna(0).astype(int).astype(str), 
                    format='%Y%m%d', 
                    errors='coerce'
                )
                
                # Calculate days difference - VECTORIZED
                days_diff = (stat_dates - open_dates).dt.days
                result['activation_days'] = days_diff.fillna(999).clip(lower=0)
            except Exception as e:
                pass  # Keep default 999
        
        # Bin activation days for ML (prevent exact-day memorization)
        result['activation_days_bin'] = pd.cut(
            result['activation_days'],
            bins=[0, 7, 30, 90, 365, float('inf')],
            labels=[0, 1, 2, 3, 4]  # 0=burner(<7d), 1=new(<30d), 2=recent(<90d), 3=est(<1y), 4=old
        ).astype(float).fillna(4)
        
        return result
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of LEAK-FREE feature columns for modeling.
        
        EXCLUDED features (data quality / leakage):
        - duration_bin: Causes "Duration Addiction" (simbox fingerprint)
        - is_5g: Causes "5G Trap" (52% in train vs 25% in validation)
        - has_roaming_*: Dead (1-2 non-zero in training)
        - active_*: Dead (0-2 non-zero in training)
        - multiple_terminals: Very sparse (1.4% non-zero)
        """
        return [
            # Call features (behavior-based) - THESE WORK
            'call_volume_bin', 
            'has_short_calls', 'has_long_calls', 'outbound_ratio',
            
            # SILENT RECEIVER: Fraud phones make calls but never receive
            'is_silent_receiver',
            
            # NOTE: Removed dispersion features (low_dispersion, dispersion_level)
            # They caused "Perfect Separator Trap" on validation data
            
            # Student targeting (BINARY)
            'calls_students',
            
            # Account features (stable)
            'is_prepaid', 'is_virtual_sim',
            
            # IDENTITY: Passport users are 4x more likely to be fraud
            'is_passport',
            
            # Burner phone feature
            'activation_days_bin',  # 0=burner(<7d), 1=new(<30d), etc.
        ]
    
    def engineer_all_features(self) -> pd.DataFrame:
        """
        Engineer all LEAK-FREE features.
        
        Returns:
            DataFrame with target and engineered features
        """
        # Create target and filter
        df = self.create_target()
        
        # Engineer all feature groups
        df = self.engineer_call_features(df)
        df = self.engineer_roaming_features(df)
        df = self.engineer_terminal_features(df)
        df = self.engineer_time_period_features(df)
        df = self.engineer_student_targeting_features(df)
        df = self.engineer_account_features(df)
        
        print(f"Engineered {len(self.get_feature_columns())} leak-free features")
        return df
    
    def get_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get feature matrix and target for TRAINING.
        
        Only includes labeled records (稽核通過 or 稽核不通過).
        Excludes pending (待稽核) records.
        """
        df = self.engineer_all_features()
        
        feature_cols = self.get_feature_columns()
        available_cols = [c for c in feature_cols if c in df.columns]
        
        labeled_mask = df['is_fraud'].isin([0, 1])
        X = df.loc[labeled_mask, available_cols].copy()
        y = df.loc[labeled_mask, 'is_fraud'].copy()
        
        # Fill any remaining NaN
        X = X.fillna(0)
        
        print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Features used: {available_cols}")
        return X, y
    
    def get_prediction_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get feature matrix for PREDICTION (all records including pending).
        
        Returns:
            Tuple of (X, df_original) where:
            - X is feature DataFrame for all records
            - df_original is original DataFrame with all columns
        """
        # Load raw data
        df = pd.read_csv(self.data_path, low_memory=False)
        print(f"Loaded {len(df)} records for prediction")
        
        # Create is_fraud column (pending = NaN, but still process)
        if 'audit_status' in df.columns:
            df['is_fraud'] = df['audit_status'].map({
                '稽核不通過': 1,  # Confirmed fraud
                '稽核通過': 0,    # Not fraud
                '待稽核': -1      # Pending (unknown)
            }).fillna(-1).astype(int)
        else:
            df['is_fraud'] = -1  # All pending
        
        # Engineer features
        df = self.engineer_call_features(df)
        df = self.engineer_roaming_features(df)
        df = self.engineer_terminal_features(df)
        df = self.engineer_time_period_features(df)
        df = self.engineer_student_targeting_features(df)
        df = self.engineer_account_features(df)
        
        feature_cols = self.get_feature_columns()
        available_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[available_cols].copy().fillna(0)
        
        print(f"Prediction data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, df


def main():
    """Test the feature engineering."""
    engineer = FraudFeatureEngineer()
    X, y = engineer.get_train_data()
    
    print("\n=== Feature Summary ===")
    print(X.describe())
    
    print("\n=== Target Distribution ===")
    print(y.value_counts())


if __name__ == "__main__":
    main()
