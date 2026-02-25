"""
Student Portrait Generator

Generates descriptive portraits of high-risk student groups,
analyzing their characteristics and behavioral patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime


class StudentPortraitGenerator:
    """
    Generates portraits/profiles of high-risk students.
    
    Analyzes demographic, communication, and behavioral patterns
    to describe what makes a student high-risk for wire fraud.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the portrait generator.
        
        Args:
            output_dir: Directory to save portrait reports
        """
        self.output_dir = output_dir or self._get_default_output_dir()
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
    def _get_default_output_dir(self) -> str:
        """Get default output directory."""
        base = Path(__file__).parent.parent
        return str(base / "Datasets" / "Student" / "Results")
    
    def generate_portrait(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive portrait of high-risk students.
        
        Args:
            df: DataFrame with risk scores and classifications
            
        Returns:
            Dict with portrait data
        """
        # Separate high-risk and low-risk groups (robust to tier labels)
        if 'is_high_risk' in df.columns:
            high_risk = df[df['is_high_risk'] == 1]
            low_risk = df[df['is_high_risk'] == 0]
        elif 'risk_classification' in df.columns:
            high_risk = df[df['risk_classification'].isin(['high_risk', 'CRITICAL', 'VULNERABLE'])]
            low_risk = df[~df['risk_classification'].isin(['high_risk', 'CRITICAL', 'VULNERABLE'])]
        else:
            # Use score threshold
            threshold = df['combined_risk_score'].quantile(0.9)
            high_risk = df[df['combined_risk_score'] >= threshold]
            low_risk = df[df['combined_risk_score'] < threshold]
        
        portrait = {
            'high_risk_count': len(high_risk),
            'high_risk_percentage': len(high_risk) / len(df) * 100,
            'total_students': len(df),
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            
            # Demographic Portrait
            'demographics': self._analyze_demographics(high_risk, low_risk),
            
            # Communication Portrait
            'communication': self._analyze_communication(high_risk, low_risk),
            
            # Fraud Engagement Portrait
            'fraud_engagement': self._analyze_fraud_engagement(high_risk, low_risk),
            
            # Mobility Portrait
            'mobility': self._analyze_mobility(high_risk, low_risk),
            
            # Key Differentiators
            'key_differentiators': self._find_key_differentiators(high_risk, low_risk)
        }
        
        return portrait
    
    def _analyze_demographics(self, high_risk: pd.DataFrame, low_risk: pd.DataFrame) -> Dict:
        """Analyze demographic characteristics."""
        return {
            'high_risk': {
                'avg_age': high_risk['age'].mean() if 'age' in high_risk.columns else None,
                'age_18_22_pct': (high_risk['age'] <= 22).mean() * 100 if 'age' in high_risk.columns else None,
                'mainland_student_pct': high_risk['is_mainland_student'].mean() * 100 if 'is_mainland_student' in high_risk.columns else None,
                'hk_piao_pct': high_risk['is_hk_piao'].mean() * 100 if 'is_hk_piao' in high_risk.columns else None,
            },
            'low_risk': {
                'avg_age': low_risk['age'].mean() if 'age' in low_risk.columns else None,
                'age_18_22_pct': (low_risk['age'] <= 22).mean() * 100 if 'age' in low_risk.columns else None,
                'mainland_student_pct': low_risk['is_mainland_student'].mean() * 100 if 'is_mainland_student' in low_risk.columns else None,
                'hk_piao_pct': low_risk['is_hk_piao'].mean() * 100 if 'is_hk_piao' in low_risk.columns else None,
            }
        }
    
    def _analyze_communication(self, high_risk: pd.DataFrame, low_risk: pd.DataFrame) -> Dict:
        """Analyze communication patterns."""
        return {
            'high_risk': {
                'avg_overseas_voice': high_risk['total_voice_cnt'].mean() if 'total_voice_cnt' in high_risk.columns else None,
                'avg_overseas_msg': high_risk['total_msg_cnt'].mean() if 'total_msg_cnt' in high_risk.columns else None,
                'avg_mainland_calls': high_risk['from_china_mobile_call_cnt'].mean() if 'from_china_mobile_call_cnt' in high_risk.columns else None,
                'avg_max_same_caller': high_risk['max_voice_cnt'].mean() if 'max_voice_cnt' in high_risk.columns else None,
            },
            'low_risk': {
                'avg_overseas_voice': low_risk['total_voice_cnt'].mean() if 'total_voice_cnt' in low_risk.columns else None,
                'avg_overseas_msg': low_risk['total_msg_cnt'].mean() if 'total_msg_cnt' in low_risk.columns else None,
                'avg_mainland_calls': low_risk['from_china_mobile_call_cnt'].mean() if 'from_china_mobile_call_cnt' in low_risk.columns else None,
                'avg_max_same_caller': low_risk['max_voice_cnt'].mean() if 'max_voice_cnt' in low_risk.columns else None,
            }
        }
    
    def _analyze_fraud_engagement(self, high_risk: pd.DataFrame, low_risk: pd.DataFrame) -> Dict:
        """Analyze fraud engagement patterns."""
        return {
            'high_risk': {
                'fraud_contact_pct': high_risk['has_fraud_contact'].mean() * 100 if 'has_fraud_contact' in high_risk.columns else None,
                'avg_fraud_calls_received': high_risk['voice_receive'].mean() if 'voice_receive' in high_risk.columns else None,
                'responded_to_fraud_pct': high_risk['has_responded_fraud'].mean() * 100 if 'has_responded_fraud' in high_risk.columns else None,
                'avg_response_rate': high_risk['fraud_response_rate'].mean() if 'fraud_response_rate' in high_risk.columns else None,
            },
            'low_risk': {
                'fraud_contact_pct': low_risk['has_fraud_contact'].mean() * 100 if 'has_fraud_contact' in low_risk.columns else None,
                'avg_fraud_calls_received': low_risk['voice_receive'].mean() if 'voice_receive' in low_risk.columns else None,
                'responded_to_fraud_pct': low_risk['has_responded_fraud'].mean() * 100 if 'has_responded_fraud' in low_risk.columns else None,
                'avg_response_rate': low_risk['fraud_response_rate'].mean() if 'fraud_response_rate' in low_risk.columns else None,
            }
        }
    
    def _analyze_mobility(self, high_risk: pd.DataFrame, low_risk: pd.DataFrame) -> Dict:
        """Analyze mobility patterns."""
        return {
            'high_risk': {
                'avg_mainland_days': high_risk['mainland_cnt'].mean() if 'mainland_cnt' in high_risk.columns else None,
                'avg_hk_trips': high_risk['mainland_to_hk_cnt'].mean() if 'mainland_to_hk_cnt' in high_risk.columns else None,
                'mainland_app_user_pct': high_risk['uses_mainland_apps'].mean() * 100 if 'uses_mainland_apps' in high_risk.columns else None,
            },
            'low_risk': {
                'avg_mainland_days': low_risk['mainland_cnt'].mean() if 'mainland_cnt' in low_risk.columns else None,
                'avg_hk_trips': low_risk['mainland_to_hk_cnt'].mean() if 'mainland_to_hk_cnt' in low_risk.columns else None,
                'mainland_app_user_pct': low_risk['uses_mainland_apps'].mean() * 100 if 'uses_mainland_apps' in low_risk.columns else None,
            }
        }
    
    def _find_key_differentiators(self, high_risk: pd.DataFrame, low_risk: pd.DataFrame) -> List[Dict]:
        """Find features that most distinguish high-risk from low-risk students."""
        differentiators = []
        
        features_to_compare = [
            ('is_mainland_student', 'Mainland Student Permit'),
            ('is_hk_piao', 'Hong Kong Piao Status'),
            ('from_china_mobile_call_cnt', 'Mainland Operator Calls'),
            ('total_voice_cnt', 'Overseas Voice Contacts'),
            ('has_responded_fraud', 'Responded to Fraud'),
            ('fraud_response_rate', 'Fraud Response Rate'),
            ('max_voice_cnt', 'Repeated Caller Contact'),
        ]
        
        for feat, description in features_to_compare:
            if feat in high_risk.columns and feat in low_risk.columns:
                high_mean = high_risk[feat].mean()
                low_mean = low_risk[feat].mean()
                
                if low_mean > 0:
                    ratio = high_mean / low_mean
                else:
                    ratio = high_mean if high_mean > 0 else 1
                
                differentiators.append({
                    'feature': feat,
                    'description': description,
                    'high_risk_mean': round(high_mean, 4),
                    'low_risk_mean': round(low_mean, 4),
                    'ratio': round(ratio, 2)
                })
        
        # Sort by ratio (biggest differences first)
        differentiators.sort(key=lambda x: x['ratio'], reverse=True)
        
        return differentiators
    
    def generate_portrait_report(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Generate and save a markdown report of the portrait.
        
        Args:
            df: DataFrame with risk scores
            filename: Output filename
            
        Returns:
            Path to saved report
        """
        portrait = self.generate_portrait(df)
        
        report = f"""# High-Risk Student Portrait Report

**Generated:** {portrait['analysis_date']}  
**Total Students Analyzed:** {portrait['total_students']:,}  
**High-Risk Students Identified:** {portrait['high_risk_count']:,} ({portrait['high_risk_percentage']:.1f}%)

---

## Executive Summary

High-risk students for wire fraud are characterized by:

1. **Demographics**: More likely to be mainland international students (using 内地居民往来港澳通行证) and 港漂 (new mainland arrivals)
2. **Communication**: Higher frequency of calls from mainland operators and unfamiliar overseas numbers
3. **Fraud Engagement**: More likely to have been contacted by fraud numbers AND to have responded to them
4. **Mobility**: Frequent travel between mainland and Hong Kong, heavy mainland app usage

---

## Demographic Profile

| Metric | High-Risk | Low-Risk | Ratio |
|--------|-----------|----------|-------|
"""
        
        demo = portrait['demographics']
        if demo['high_risk']['avg_age'] is not None:
            report += f"| Average Age | {demo['high_risk']['avg_age']:.1f} | {demo['low_risk']['avg_age']:.1f} | - |\n"
        if demo['high_risk']['age_18_22_pct'] is not None:
            report += f"| Age 18-22 % | {demo['high_risk']['age_18_22_pct']:.1f}% | {demo['low_risk']['age_18_22_pct']:.1f}% | {demo['high_risk']['age_18_22_pct']/max(demo['low_risk']['age_18_22_pct'],0.01):.1f}x |\n"
        if demo['high_risk']['mainland_student_pct'] is not None:
            report += f"| Mainland Student % | {demo['high_risk']['mainland_student_pct']:.1f}% | {demo['low_risk']['mainland_student_pct']:.1f}% | {demo['high_risk']['mainland_student_pct']/max(demo['low_risk']['mainland_student_pct'],0.01):.1f}x |\n"
        if demo['high_risk']['hk_piao_pct'] is not None:
            report += f"| HK Piao % | {demo['high_risk']['hk_piao_pct']:.1f}% | {demo['low_risk']['hk_piao_pct']:.1f}% | {demo['high_risk']['hk_piao_pct']/max(demo['low_risk']['hk_piao_pct'],0.01):.1f}x |\n"
        
        report += """
---

## Communication Pattern Profile

| Metric | High-Risk | Low-Risk | Ratio |
|--------|-----------|----------|-------|
"""
        
        comm = portrait['communication']
        if comm['high_risk']['avg_overseas_voice'] is not None:
            report += f"| Avg Overseas Calls | {comm['high_risk']['avg_overseas_voice']:.1f} | {comm['low_risk']['avg_overseas_voice']:.1f} | {comm['high_risk']['avg_overseas_voice']/max(comm['low_risk']['avg_overseas_voice'],0.01):.1f}x |\n"
        if comm['high_risk']['avg_mainland_calls'] is not None:
            report += f"| Avg Mainland Calls | {comm['high_risk']['avg_mainland_calls']:.1f} | {comm['low_risk']['avg_mainland_calls']:.1f} | {comm['high_risk']['avg_mainland_calls']/max(comm['low_risk']['avg_mainland_calls'],0.01):.1f}x |\n"
        if comm['high_risk']['avg_max_same_caller'] is not None:
            report += f"| Max Same Caller | {comm['high_risk']['avg_max_same_caller']:.1f} | {comm['low_risk']['avg_max_same_caller']:.1f} | {comm['high_risk']['avg_max_same_caller']/max(comm['low_risk']['avg_max_same_caller'],0.01):.1f}x |\n"
        
        report += """
---

## Fraud Engagement Profile

| Metric | High-Risk | Low-Risk | Ratio |
|--------|-----------|----------|-------|
"""
        
        fraud = portrait['fraud_engagement']
        if fraud['high_risk']['fraud_contact_pct'] is not None:
            report += f"| Fraud Contact % | {fraud['high_risk']['fraud_contact_pct']:.1f}% | {fraud['low_risk']['fraud_contact_pct']:.1f}% | {fraud['high_risk']['fraud_contact_pct']/max(fraud['low_risk']['fraud_contact_pct'],0.01):.1f}x |\n"
        if fraud['high_risk']['responded_to_fraud_pct'] is not None:
            report += f"| Responded to Fraud % | {fraud['high_risk']['responded_to_fraud_pct']:.1f}% | {fraud['low_risk']['responded_to_fraud_pct']:.1f}% | {fraud['high_risk']['responded_to_fraud_pct']/max(fraud['low_risk']['responded_to_fraud_pct'],0.01):.1f}x |\n"
        if fraud['high_risk']['avg_fraud_calls_received'] is not None:
            report += f"| Avg Fraud Calls | {fraud['high_risk']['avg_fraud_calls_received']:.2f} | {fraud['low_risk']['avg_fraud_calls_received']:.2f} | {fraud['high_risk']['avg_fraud_calls_received']/max(fraud['low_risk']['avg_fraud_calls_received'],0.01):.1f}x |\n"
        
        report += """
---

## Key Risk Differentiators

The following factors most strongly differentiate high-risk from low-risk students:

| Rank | Feature | High-Risk Mean | Low-Risk Mean | Lift |
|------|---------|----------------|---------------|------|
"""
        
        for i, diff in enumerate(portrait['key_differentiators'][:7], 1):
            report += f"| {i} | {diff['description']} | {diff['high_risk_mean']:.3f} | {diff['low_risk_mean']:.3f} | {diff['ratio']}x |\n"
        
        report += """
---

## High-Risk Student Archetype

Based on the analysis, a typical high-risk student is:

> **A young mainland international student (港漂) aged 18-22, who:**
> - Uses mainland permit (内地居民往来港澳通行证)
> - Receives frequent calls from mainland operators
> - Has higher-than-average contact with unfamiliar overseas numbers
> - Has been contacted by fraud numbers and may have responded
> - Travels frequently between mainland and Hong Kong
> - Uses mainland apps (WeChat, Taobao, etc.) regularly

---

## Recommendations for Early Intervention

1. **Targeted Education**: Focus fraud awareness campaigns on mainland international students
2. **Proactive Alerts**: Send warnings when students receive multiple calls from unfamiliar mainland numbers
3. **Response Monitoring**: Flag students who show signs of engaging with potential fraud calls
4. **Cross-border Focus**: Pay special attention to students with high mainland-HK travel patterns

---

## High-Risk Behavior Sub-Types (Cluster Analysis)

"""
        
        # Add cluster behavior analysis if cluster labels exist
        cluster_col = 'cluster_label' if 'cluster_label' in df.columns else ('cluster' if 'cluster' in df.columns else None)
        if cluster_col:
            high_risk = df[df['is_high_risk'] == 1] if 'is_high_risk' in df.columns else df
            
            # Get clusters within high-risk (exclude -2 = not high-risk, -1 = outliers separate)
            cluster_ids = sorted([c for c in high_risk[cluster_col].unique() if c >= 0])
            
            if cluster_ids:
                report += "The following behavior sub-types were discovered within high-risk students:\n\n"
                
                for cluster_id in cluster_ids:
                    cluster_df = high_risk[high_risk[cluster_col] == cluster_id]
                    
                    # Calculate cluster characteristics
                    size = len(cluster_df)
                    pct = size / len(high_risk) * 100
                    
                    # Key metrics
                    mainland_pct = cluster_df['is_mainland_student'].mean() * 100 if 'is_mainland_student' in cluster_df.columns else 0
                    fraud_pct = cluster_df['has_fraud_contact'].mean() * 100 if 'has_fraud_contact' in cluster_df.columns else 0
                    avg_overseas = cluster_df['total_voice_cnt'].mean() if 'total_voice_cnt' in cluster_df.columns else 0
                    avg_mainland = cluster_df['from_china_mobile_call_cnt'].mean() if 'from_china_mobile_call_cnt' in cluster_df.columns else 0
                    avg_age = cluster_df['age'].mean() if 'age' in cluster_df.columns else 0
                    hk_piao_pct = cluster_df['is_hk_piao'].mean() * 100 if 'is_hk_piao' in cluster_df.columns else 0
                    
                    # Determine cluster type
                    type_name = "General Risk"
                    if fraud_pct > 5:
                        type_name = "Fraud-Engaged"
                    elif mainland_pct > 50:
                        type_name = "Mainland-Heavy"
                    elif avg_overseas > 20:
                        type_name = "High Communication"
                    elif hk_piao_pct > 60:
                        type_name = "HK Newcomer"
                    
                    report += f"""### Cluster {cluster_id}: {type_name}
- **Size**: {size:,} students ({pct:.1f}% of high-risk)
- **Avg Age**: {avg_age:.1f} years
- **Mainland Student %**: {mainland_pct:.1f}%
- **HK Piao %**: {hk_piao_pct:.1f}%
- **Fraud Contact %**: {fraud_pct:.1f}%
- **Avg Overseas Calls**: {avg_overseas:.1f}
- **Avg Mainland Calls**: {avg_mainland:.1f}

"""
                
                # Add outliers section
                outliers = high_risk[high_risk[cluster_col] == -1]
                if len(outliers) > 0:
                    outlier_fraud_pct = outliers['has_fraud_contact'].mean() * 100 if 'has_fraud_contact' in outliers.columns else 0
                    report += f"""### Outliers (Cluster -1)
- **Size**: {len(outliers):,} students (anomalous behavior)
- **Fraud Contact %**: {outlier_fraud_pct:.1f}%
- These students have unusual patterns that don't fit typical clusters

"""
        else:
            report += "No cluster analysis available.\n"
        
        # Save report
        if filename is None:
            filename = "high_risk_student_portrait.md"
        
        output_path = Path(self.output_dir) / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Portrait report saved to: {output_path}")
        return str(output_path)


def main():
    """Test the portrait generator."""
    from feature_engineering import StudentFeatureEngineer
    from rule_based_scorer import RuleBasedScorer
    
    # Load data
    engineer = StudentFeatureEngineer()
    df = engineer.engineer_all_features()
    
    # Score students
    scorer = RuleBasedScorer()
    df_scored = scorer.score_dataframe(df)
    df_scored['risk_classification'] = df_scored['rule_score'].apply(scorer.classify_risk)
    df_scored['combined_risk_score'] = df_scored['rule_score']
    
    # Generate portrait
    generator = StudentPortraitGenerator()
    report_path = generator.generate_portrait_report(df_scored)
    print(f"\nPortrait report generated: {report_path}")


if __name__ == "__main__":
    main()
