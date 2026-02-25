"""
Fraud Detection Pipeline - Black Sample Identification

Hybrid Detection Strategy (Swiss Cheese Defense):
1. Rule Engine (Fast) - 6 rules catching "dumb" bots
2. ML Model (Precise) - XGBoost catching "smart" humans

Data Strategy:
- Train on: fraud_model_2.csv (labeled: Audit Passed + Failed)
- Predict on: fraud_model_1_2.csv (unlabeled) + Pending records

Output:
- blacklist.csv: High confidence (>85%) for BLOCKING
- greylist.csv: Medium confidence (60-85%) for WARNINGS

Usage:
    python run_fraud_model.py
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from fraud_feature_engineering import FraudFeatureEngineer
    from fraud_scoring_model import FraudScoringModel
    from fraud_rule_engine import FraudRuleEngine
except ImportError:
    from fraud_detection.fraud_feature_engineering import FraudFeatureEngineer
    from fraud_detection.fraud_scoring_model import FraudScoringModel
    from fraud_detection.fraud_rule_engine import FraudRuleEngine


def load_whitelist():
    """Load whitelist MSISDNs from whitelist.csv"""
    whitelist_path = Path(__file__).parent.parent.parent / "Datasets" / "Fraud" / "Results" / "whitelist.csv"
    if whitelist_path.exists():
        whitelist_df = pd.read_csv(whitelist_path)
        return set(whitelist_df['msisdn'].astype(str))
    return set()


def apply_whitelist_to_df(df, whitelist_msisdns):
    """Apply whitelist: mark whitelisted accounts as non-fraud (is_fraud=0)"""
    if len(whitelist_msisdns) == 0 or 'msisdn' not in df.columns:
        return df, 0
    
    mask = df['msisdn'].astype(str).isin(whitelist_msisdns)
    count = mask.sum()
    
    if count > 0 and 'is_fraud' in df.columns:
        df.loc[mask, 'is_fraud'] = 0
    
    return df, count


class FraudDetectionPipeline:
    """
    Black Sample Identification Pipeline.
    
    Phase 1: Train on labeled data (fraud_model_2)
    Phase 2: Mine on unlabeled data (fraud_model_1_2 + pending)
    """
    
    def __init__(self, output_dir: str = None):
        base_dir = Path(__file__).parent.parent.parent / "Datasets" / "Fraud"
        
        # Data paths
        self.train_path = str(base_dir / "Training and Testing Data" / "fraud_model_2.csv")
        self.mine_path_1 = str(base_dir / "Training and Testing Data" / "fraud_model_1_1.csv")
        self.mine_path_2 = str(base_dir / "Training and Testing Data" / "fraud_model_1_2.csv")
        
        self.output_dir = output_dir or str(base_dir / "Results")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_engineer = FraudFeatureEngineer(self.train_path)
        self.rule_engine = FraudRuleEngine()
        self.ml_model = FraudScoringModel()
    
    def run_full_pipeline(self) -> dict:
        """
        Run the complete Black Sample identification pipeline.
        """
        print("\n" + "=" * 60)
        print("BLACK SAMPLE IDENTIFICATION PIPELINE")
        print("=" * 60)
        print("\nWutong Defense Shield: Rules + ML → Blacklist\n")
        
        # ============================================================
        # PHASE 1: TRAIN ON BALANCED DATA
        # ============================================================
        print("=" * 60)
        print("PHASE 1: Prepare Balanced Training Data")
        print("=" * 60)
        
        # Load whitelist
        whitelist_msisdns = load_whitelist()
        if len(whitelist_msisdns) > 0:
            print(f"✅ Loaded whitelist: {len(whitelist_msisdns)} accounts")
        
        # Load fraud_model_2 (labeled data)
        train_df = pd.read_csv(self.train_path, low_memory=False)
        print(f"Loaded fraud_model_2: {len(train_df)} records")
        
        # Create labels for fraud_model_2; keep pending rows unlabeled to avoid noise
        train_df['is_fraud'] = np.nan
        train_df.loc[train_df['audit_status'] == '稽核不通過', 'is_fraud'] = 1
        train_df.loc[train_df['audit_status'] == '稽核通過', 'is_fraud'] = 0
        
        # Apply whitelist to fraud_model_2
        train_df, wl_count_2 = apply_whitelist_to_df(train_df, whitelist_msisdns)
        if wl_count_2 > 0:
            print(f"  ✅ Whitelist applied to fraud_model_2: {wl_count_2} accounts marked as non-fraud")
        
        confirmed_fraud = (train_df['is_fraud'] == 1).sum()
        confirmed_safe = (train_df['is_fraud'] == 0).sum()
        print(f"  Positives (Fraud): {confirmed_fraud}")
        print(f"  Hard Negatives (Safe): {confirmed_safe}")
        
        # Add easy negatives from fraud_model_1_1 (empty audit = normal users)
        try:
            df_1_1 = pd.read_csv(self.mine_path_1, low_memory=False)
            easy_negatives = df_1_1[df_1_1['audit_status'].isna() | (df_1_1['audit_status'] == '')].copy()
            easy_negatives['is_fraud'] = 0  # Normal users
            
            # Apply whitelist to fraud_model_1_1 easy negatives (already 0, but ensure consistency)
            easy_negatives, wl_count_1 = apply_whitelist_to_df(easy_negatives, whitelist_msisdns)
            if wl_count_1 > 0:
                print(f"  ✅ Whitelist matched in fraud_model_1_1: {wl_count_1} accounts")
            
            print(f"  Easy Negatives (Normal): {len(easy_negatives)} from fraud_model_1_1")
            
            # Combine for balanced training
            train_df = pd.concat([train_df, easy_negatives], ignore_index=True)
            print(f"\nTotal Training Data: {len(train_df)}")
        except Exception as e:
            print(f"  Warning: Could not load easy negatives: {e}")
        
        # Feature engineering on combined data
        print("\nEngineering features...")
        self.feature_engineer.df = train_df  # Override internal df
        df_featured = self.feature_engineer.engineer_all_features()
        
        # Get training data (only labeled rows)
        feature_cols = self.feature_engineer.get_feature_columns()
        available_cols = [c for c in feature_cols if c in df_featured.columns]
        
        # For training, use both fraud_model_2 labeled AND easy negatives
        train_mask = df_featured['is_fraud'].isin([0, 1])
        X = df_featured.loc[train_mask, available_cols].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        y = df_featured.loc[train_mask, 'is_fraud'].astype(int)
        
        print(f"\nTraining XGBoost on {len(X)} samples...")
        print(f"  Class distribution: {y.value_counts().to_dict()}")
        self.ml_model.fit(X, y)
        
        # ============================================================
        # PHASE 2: MINE UNLABELED DATA
        # ============================================================
        print("\n" + "=" * 60)
        print("PHASE 2: Mine Unlabeled Data")
        print("=" * 60)
        
        # Load mining targets
        mine_dfs = []
        
        # 1. Pending records from fraud_model_1_1
        try:
            df_1_1 = pd.read_csv(self.mine_path_1, low_memory=False)
            pending = df_1_1[df_1_1['audit_status'] == '待稽核'].copy()
            pending['source_file'] = 'fraud_model_1_1_pending'
            mine_dfs.append(pending)
            print(f"  fraud_model_1_1 (Pending): {len(pending)} records")
        except Exception as e:
            print(f"  Warning: Could not load fraud_model_1_1: {e}")
        
        # 2. All records from fraud_model_1_2 (unlabeled)
        try:
            df_1_2 = pd.read_csv(self.mine_path_2, low_memory=False)
            df_1_2['source_file'] = 'fraud_model_1_2'
            mine_dfs.append(df_1_2)
            print(f"  fraud_model_1_2 (Unlabeled): {len(df_1_2)} records")
        except Exception as e:
            print(f"  Warning: Could not load fraud_model_1_2: {e}")
        
        # 3. Also include the training data for full output
        train_df['source_file'] = 'fraud_model_2'
        mine_dfs.append(train_df)
        
        # Combine all
        all_df = pd.concat(mine_dfs, ignore_index=True)
        print(f"\nTotal records to scan: {len(all_df)}")

        # ============================================================
        # FEATURE ENGINEERING FOR PREDICTION
        # ============================================================
        # Apply the same leak-free feature set to mining data without filtering labels
        fe = self.feature_engineer
        all_df = fe.engineer_call_features(all_df)
        all_df = fe.engineer_roaming_features(all_df)
        all_df = fe.engineer_terminal_features(all_df)
        all_df = fe.engineer_time_period_features(all_df)
        all_df = fe.engineer_student_targeting_features(all_df)
        all_df = fe.engineer_account_features(all_df)
        
        # ============================================================
        # LAYER 1: APPLY 6-RULE ENGINE
        # ============================================================
        print("\n" + "=" * 60)
        print("LAYER 1: Apply 6-Rule Engine (Fast Detection)")
        print("=" * 60)
        
        # Ensure numeric types for rule columns to avoid comparison errors
        numeric_rule_cols = [
            'call_cnt_day', 'called_cnt_day', 'dispersion_rate',
            'opp_num_stu_cnt', 'change_imei_times', 'tot_msg_cnt',
            'roam_unknow_call_cnt', 'local_unknow_call_cnt'
        ]
        for col in numeric_rule_cols:
            if col in all_df.columns:
                all_df[col] = pd.to_numeric(all_df[col], errors='coerce').fillna(0)

        all_df = self.rule_engine.apply_all_rules(all_df)
        rule_hits = all_df['rule_hit'].sum()
        
        # ============================================================
        # LAYER 2: APPLY ML MODEL
        # ============================================================
        print("\n" + "=" * 60)
        print("LAYER 2: Apply ML Model (Precise Detection)")
        print("=" * 60)
        
        # Get feature columns - only numeric columns
        feature_cols = self.feature_engineer.get_feature_columns()
        available_cols = [c for c in feature_cols if c in all_df.columns]

        if not available_cols:
            print("Warning: No feature columns available for ML prediction; skipping ML scoring")
            all_df['ml_probability'] = 0.0
        else:
            # Select only numeric columns and convert properly
            X_predict = all_df[available_cols].copy()
            for col in X_predict.columns:
                X_predict[col] = pd.to_numeric(X_predict[col], errors='coerce').fillna(0)
            
            # Predict probabilities
            try:
                ml_proba = self.ml_model.predict_proba(X_predict)
                all_df['ml_probability'] = ml_proba
            except Exception as e:
                print(f"Warning: ML prediction failed: {e}")
                all_df['ml_probability'] = 0.0
        
        print(f"ML Probability Distribution:")
        print(f"  > 0.85 (Black): {(all_df['ml_probability'] > 0.85).sum():,}")
        print(f"  0.60-0.85 (Grey): {((all_df['ml_probability'] >= 0.60) & (all_df['ml_probability'] <= 0.85)).sum():,}")
        print(f"  < 0.60 (White): {(all_df['ml_probability'] < 0.60).sum():,}")
        
        # ============================================================
        # CLASSIFY INTO TIERS
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 3: Classify into Tiers")
        print("=" * 60)
        
        # Initialize
        all_df['risk_score'] = all_df['ml_probability']
        all_df['source'] = 'NONE'
        all_df['trigger_reason'] = ''
        all_df['risk_tier'] = 'WHITE'
        
        # Rule hits → BLACK tier (score = 1.0)
        rule_mask = all_df['rule_hit'] == True
        all_df.loc[rule_mask, 'risk_score'] = 1.0
        all_df.loc[rule_mask, 'source'] = 'RULE'
        all_df.loc[rule_mask, 'trigger_reason'] = all_df.loc[rule_mask, 'rule_reason']
        all_df.loc[rule_mask, 'risk_tier'] = 'BLACK'
        
        # ML > 0.85 → BLACK tier (if not already rule-hit)
        ml_black = (~rule_mask) & (all_df['ml_probability'] > 0.85)
        all_df.loc[ml_black, 'source'] = 'ML'
        all_df.loc[ml_black, 'trigger_reason'] = 'High_Prob_XGBoost'
        all_df.loc[ml_black, 'risk_tier'] = 'BLACK'
        
        # ML 0.60-0.85 → GREY tier
        ml_grey = (all_df['risk_tier'] == 'WHITE') & (all_df['ml_probability'] >= 0.60)
        all_df.loc[ml_grey, 'source'] = 'ML'
        all_df.loc[ml_grey, 'trigger_reason'] = 'Medium_Prob_XGBoost'
        all_df.loc[ml_grey, 'risk_tier'] = 'GREY'
        
        # ============================================================
        # FORMALIZED GREYLIST RULE: Passport + Silent
        # ============================================================
        # Users with passport identity AND silent receiver pattern are suspicious
        # Calls > 15 → BLACK (high activity fraud)
        # Calls <= 15 → GREY (low activity, needs monitoring)
        if 'is_passport' in all_df.columns and 'is_silent_receiver' in all_df.columns:
            passport_silent = (all_df['is_passport'] == 1) & (all_df['is_silent_receiver'] == 1)
            calls = all_df['call_cnt_day'].fillna(0)
            
            # Passport + Silent + Calls > 15 → BLACK
            ps_black = passport_silent & (calls > 15) & (all_df['risk_tier'] == 'WHITE')
            all_df.loc[ps_black, 'source'] = 'RULE_PS'
            all_df.loc[ps_black, 'trigger_reason'] = f'Passport+Silent: {calls[ps_black].astype(int)} calls'
            all_df.loc[ps_black, 'risk_tier'] = 'BLACK'
            
            # Passport + Silent + Calls <= 15 → GREY
            ps_grey = passport_silent & (calls <= 15) & (all_df['risk_tier'] == 'WHITE')
            all_df.loc[ps_grey, 'source'] = 'RULE_PS'
            all_df.loc[ps_grey, 'trigger_reason'] = f'Passport+Silent (Low): {calls[ps_grey].astype(int)} calls'
            all_df.loc[ps_grey, 'risk_tier'] = 'GREY'
            
            print(f"  Passport+Silent BLACK: {ps_black.sum()}")
            print(f"  Passport+Silent GREY: {ps_grey.sum()}")
        
        # ============================================================
        # CRITICAL: "TRUST THE AUDIT" FILTER
        # ============================================================
        # Users marked as 稽核通過 (Audit Passed) are NEVER blacklisted
        # Even if rules/ML flag them - human auditor already verified safe
        audit_passed_mask = all_df['audit_status'] == '稽核通過'
        overruled_count = ((audit_passed_mask) & (all_df['risk_tier'].isin(['BLACK', 'GREY']))).sum()
        all_df.loc[audit_passed_mask, 'risk_tier'] = 'WHITE'
        all_df.loc[audit_passed_mask, 'trigger_reason'] = 'AUDIT_PASSED (Protected)'
        print(f"\n=== Trust the Audit Filter ===")
        print(f"  Protected {audit_passed_mask.sum()} Audit-Passed users")
        print(f"  Overruled {overruled_count} false positives")
        
        print("\n=== Tier Distribution ===")
        print(all_df['risk_tier'].value_counts())
        
        # ============================================================
        # SAVE OUTPUTS
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 4: Save Outputs")
        print("=" * 60)
        
        output_cols = ['msisdn', 'risk_score', 'risk_tier', 'source', 'trigger_reason', 'source_file']
        
        # Blacklist (BLACK tier only - for BLOCKING)
        blacklist = all_df[all_df['risk_tier'] == 'BLACK'][output_cols].copy()
        
        # CRITICAL: Filter out whitelisted accounts from blacklist
        if len(whitelist_msisdns) > 0:
            blacklist_before = len(blacklist)
            blacklist = blacklist[~blacklist['msisdn'].astype(str).isin(whitelist_msisdns)]
            whitelist_filtered = blacklist_before - len(blacklist)
            if whitelist_filtered > 0:
                print(f"✅ Whitelist filter: Removed {whitelist_filtered} accounts from blacklist")
        
        blacklist_path = Path(self.output_dir) / "blacklist.csv"
        blacklist.to_csv(blacklist_path, index=False)
        print(f"Blacklist saved: {blacklist_path} ({len(blacklist)} records)")
        
        # Greylist (GREY tier - for WARNINGS)
        greylist = all_df[all_df['risk_tier'] == 'GREY'][output_cols].copy()
        
        # CRITICAL: Filter out whitelisted accounts from greylist
        if len(whitelist_msisdns) > 0:
            greylist_before = len(greylist)
            greylist = greylist[~greylist['msisdn'].astype(str).isin(whitelist_msisdns)]
            whitelist_filtered_grey = greylist_before - len(greylist)
            if whitelist_filtered_grey > 0:
                print(f"✅ Whitelist filter: Removed {whitelist_filtered_grey} accounts from greylist")
        
        greylist_path = Path(self.output_dir) / "greylist.csv"
        greylist.to_csv(greylist_path, index=False)
        print(f"Greylist saved: {greylist_path} ({len(greylist)} records)")
        
        # Summary by source file
        print("\n=== Black Samples by Source ===")
        black_by_source = blacklist.groupby('source_file').size()
        for src, cnt in black_by_source.items():
            print(f"  {src}: {cnt}")
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        
        return {
            'blacklist': str(blacklist_path),
            'greylist': str(greylist_path),
            'black_count': len(blacklist),
            'grey_count': len(greylist),
            'rule_hits': rule_hits,
            'rule_stats': self.rule_engine.get_rule_stats(),
        }


def main():
    parser = argparse.ArgumentParser(description='Fraud Detection Pipeline')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    pipeline = FraudDetectionPipeline(output_dir=args.output)
    results = pipeline.run_full_pipeline()
    
    print("\n=== Summary ===")
    print(f"Black Tier (BLOCK): {results['black_count']:,}")
    print(f"Grey Tier (WARN): {results['grey_count']:,}")
    print(f"\nRule Engine Breakdown:")
    for rule, count in results['rule_stats'].items():
        print(f"  {rule}: {count}")


if __name__ == "__main__":
    main()
