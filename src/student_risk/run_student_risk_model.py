"""
High-Risk Student Identification Pipeline

Workflow:
1. Risk Triangle Scoring → Identity + Exposure + Behavior
2. Portrait Generation → Analyze and report

Usage:
    python run_student_risk_model.py
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Handle imports
try:
    from feature_engineering import StudentFeatureEngineer
    from risk_triangle_scorer import RiskTriangleScorer
    from student_portrait import StudentPortraitGenerator
except ImportError:
    from student_risk.feature_engineering import StudentFeatureEngineer
    from student_risk.risk_triangle_scorer import RiskTriangleScorer
    from student_risk.student_portrait import StudentPortraitGenerator


class StudentRiskPipeline:
    """
    High-Risk Student Identification Pipeline.
    
    Using Risk Triangle approach:
    - Layer 1: Identity Vulnerability (Mainland student, age)
    - Layer 2: Threat Exposure (Unknown calls, fraud contact)
    - Layer 3: Risky Behavior (Answering/calling back fraud)
    """
    
    def __init__(self, data_path: str = None, output_dir: str = None):
        base_dir = Path(__file__).parent.parent.parent / "Datasets" / "Student"
        
        self.data_path = data_path
        self.output_dir = output_dir or str(base_dir / "Results")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_engineer = StudentFeatureEngineer(data_path)
        self.risk_scorer = RiskTriangleScorer()
        self.portrait_generator = StudentPortraitGenerator(self.output_dir)
    
    def run_full_pipeline(self) -> dict:
        """
        Run the complete pipeline:
        1. Risk Triangle Scoring → Identify high-risk
        2. Portrait → Analyze behavior
        """
        print("\n" + "=" * 60)
        print("HIGH-RISK STUDENT IDENTIFICATION PIPELINE")
        print("=" * 60)
        print("\nRisk Triangle: Identity → Exposure → Behavior\n")
        
        # ============================================================
        # STEP 1: Load Data & Feature Engineering
        # ============================================================
        print("=" * 60)
        print("STEP 1: Load Data & Engineer Features")
        print("=" * 60)
        
        df = self.feature_engineer.engineer_all_features()
        print(f"Loaded {len(df)} students")
        
        # Load fraud list for cross-reference (Killer Feature!)
        self.risk_scorer.load_fraud_list()
        
        # ============================================================
        # STEP 2: Risk Triangle Scoring
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 2: Risk Triangle Scoring")
        print("=" * 60)
        
        df = self.risk_scorer.score_dataframe(df)
        
        print(f"\nScore Distribution:")
        print(f"  Min: {df['risk_score'].min()}")
        print(f"  Max: {df['risk_score'].max()}")
        print(f"  Mean: {df['risk_score'].mean():.1f}")
        
        n_high = df['is_high_risk'].sum()
        print(f"\nHigh-Risk: {n_high:,} ({n_high/len(df)*100:.1f}%)")
        print(f"Low-Risk: {len(df) - n_high:,} ({(len(df)-n_high)/len(df)*100:.1f}%)")
        
        # Add placeholder cluster labels for compatibility
        df['cluster_label'] = -1
        df['cluster'] = -1
        
        # ============================================================
        # STEP 3: Generate Portrait with Risk Reasons
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 3: Generate Portrait Report")
        print("=" * 60)
        
        portrait_path = self.portrait_generator.generate_portrait_report(df)
        
        # ============================================================
        # STEP 4: Save Predictions with Risk Reasons
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 4: Save Predictions")
        print("=" * 60)
        
        # Add risk reason column for high-risk students
        df['risk_reason'] = ''
        for idx in df[df['is_high_risk'] == 1].index:
            reasons = self.risk_scorer.risk_reasons.get(idx, [])
            if reasons:
                # Take top 3 reasons
                df.loc[idx, 'risk_reason'] = ' | '.join(reasons[:3])
        
        output_cols = ['user_id', 'risk_score', 'identity_score', 'exposure_score', 'behavior_score',
                       'risk_tier', 'is_high_risk', 'risk_classification', 
                       'cluster_label', 'cluster', 'risk_reason']
        available = [c for c in output_cols if c in df.columns]
        
        pred_path = Path(self.output_dir) / "student_risk_predictions.csv"
        df[available].to_csv(pred_path, index=False)
        print(f"Saved: {pred_path}")
        
        # Show sample high-risk with reasons
        print("\n=== Sample High-Risk Students with Reasons ===")
        high_risk_sample = df[df['is_high_risk'] == 1].head(5)
        for _, row in high_risk_sample.iterrows():
            print(f"  Score={row['risk_score']}: {row['risk_reason'][:80]}...")
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        
        return {
            'predictions': str(pred_path),
            'portrait': portrait_path,
            'high_risk_count': n_high,
            'total_count': len(df)
        }


def main():
    parser = argparse.ArgumentParser(description='High-Risk Student Identification')
    parser.add_argument('--data', type=str, help='Path to student_model.csv')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    pipeline = StudentRiskPipeline(data_path=args.data, output_dir=args.output)
    results = pipeline.run_full_pipeline()
    
    print("\n=== Summary ===")
    print(f"High-Risk: {results['high_risk_count']:,} / {results['total_count']:,}")
    print(f"Portrait: {results['portrait']}")
    print(f"Predictions: {results['predictions']}")


if __name__ == "__main__":
    main()
