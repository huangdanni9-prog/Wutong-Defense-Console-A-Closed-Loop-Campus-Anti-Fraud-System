"""
Fraud Portrait Generator

Generates a markdown report with wire fraud user characteristics
based on feature importance and statistical analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class FraudPortraitGenerator:
    """
    Generates fraud user portrait reports focusing on feature importance.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the portrait generator.
        
        Args:
            output_dir: Directory to save reports
        """
        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent.parent / 
                           "Datasets" / "Fraud" / "Results")
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_portrait_report(self, 
                                 feature_importance: pd.DataFrame,
                                 portrait_insights: Dict,
                                 metrics: Dict,
                                 df: pd.DataFrame,
                                 cluster_profiles: Dict = None,
                                 cluster_names: Dict = None) -> str:
        """
        Generate markdown portrait report with cluster analysis.
        """
        report = []
        
        # Header
        report.append("# Wire Fraud User Portrait Report")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}  ")
        report.append("**Methodology:** ML (XGBoost + Isolation Forest) → Clustering (K-Means) → Portrait")
        report.append("")
        
        # Summary stats
        is_fraud = df.get('is_fraud', pd.Series())
        n_fraud = is_fraud.sum() if len(is_fraud) > 0 else 0
        n_total = len(df)
        
        report.append(f"**Total Users Analyzed:** {n_total:,}  ")
        report.append(f"**Confirmed Fraud Users:** {n_fraud:,} ({n_fraud/n_total*100:.1f}%)")
        report.append("")
        report.append("---")
        report.append("")
        
        # Model Performance
        report.append("## Model Performance")
        report.append("")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| AUC-ROC | {metrics.get('auc_roc', 0):.4f} |")
        report.append(f"| Precision | {metrics.get('precision', 0):.4f} |")
        report.append(f"| Recall | {metrics.get('recall', 0):.4f} |")
        report.append(f"| F1 Score | {metrics.get('f1', 0):.4f} |")
        report.append(f"| Log Loss | {metrics.get('log_loss', 0):.4f} |")
        report.append("")
        report.append("---")
        report.append("")
        
        # Top Risk Indicators
        report.append("## Top Risk Indicators (Feature Importance)")
        report.append("")
        report.append("| Rank | Feature | Importance |")
        report.append("|------|---------|------------|")
        
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            feature = row['feature']
            importance = row['importance']
            report.append(f"| {i} | {feature} | {importance:.4f} |")
        
        report.append("")
        report.append("---")
        report.append("")
        
        # FRAUD CLUSTER ANALYSIS - NEW SECTION
        if cluster_profiles:
            report.append("## Fraud User Clusters (Sub-Types)")
            report.append("")
            report.append("K-Means clustering reveals distinct fraud tactics:")
            report.append("")
            
            for cluster_id, profile in cluster_profiles.items():
                name = cluster_names.get(cluster_id, f"Type {cluster_id}") if cluster_names else f"Cluster {cluster_id}"
                report.append(f"### Cluster {cluster_id}: {name}")
                report.append("")
                report.append(f"**Size:** {profile['size']} users ({profile['pct_of_fraud']:.1f}% of fraud)")
                report.append("")
                
                # Key characteristics
                report.append("| Feature | Mean Value |")
                report.append("|---------|------------|")
                for key, val in profile.items():
                    if key.startswith('mean_') and isinstance(val, (int, float)):
                        feature_name = key.replace('mean_', '')
                        report.append(f"| {feature_name} | {val:.3f} |")
                
                report.append("")
            
            report.append("---")
            report.append("")
        
        # Key Fraud Characteristics
        report.append("## Key Fraud Characteristics")
        report.append("")
        report.append("Based on feature importance analysis, typical fraud users exhibit:")
        report.append("")
        
        top_features = feature_importance.head(5)['feature'].tolist()
        
        if 'call_volume_bin' in top_features:
            report.append("- **Call volume patterns** - Specific daily call counts")
        if 'outbound_ratio' in top_features:
            report.append("- **Outbound-heavy calling** - Primarily making calls, not receiving")
        if 'has_short_calls' in top_features:
            report.append("- **Short call attempts** - Many calls under 2 seconds (robocalling)")
        if 'is_prepaid' in top_features:
            report.append("- **Prepaid accounts** - Using prepaid SIMs for anonymity")
        
        report.append("")
        report.append("---")
        report.append("")
        
        # Strategy Recommendations
        report.append("## Strategy Recommendations")
        report.append("")
        report.append("1. **Call volume monitoring**: Flag accounts with unusual call volume patterns")
        report.append("2. **Outbound ratio alerts**: Investigate accounts with >90% outbound calls")
        report.append("3. **Short call detection**: Alert on accounts with many sub-2-second calls")
        report.append("4. **Prepaid scrutiny**: Apply stricter verification for prepaid SIM accounts")
        report.append("5. **Cluster-based rules**: Create specific rules for each fraud cluster")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        output_path = Path(self.output_dir) / "fraud_user_portrait.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Portrait report saved to: {output_path}")
        return str(output_path)


def main():
    """Test portrait generation."""
    generator = FraudPortraitGenerator()
    
    # Create dummy data for testing
    import pandas as pd
    
    feature_importance = pd.DataFrame({
        'feature': ['dispersion_rate', 'call_volume', 'short_call_ratio'],
        'importance': [0.25, 0.20, 0.15]
    })
    
    portrait_insights = {
        'dispersion_rate': {'fraud_mean': 0.8, 'non_fraud_mean': 0.3, 'lift': 2.67}
    }
    
    metrics = {'auc_roc': 0.85, 'precision': 0.80, 'recall': 0.75, 'f1': 0.77, 'log_loss': 0.4}
    
    df = pd.DataFrame({'is_fraud': [1, 1, 0, 0], 'dispersion_rate': [0.9, 0.8, 0.2, 0.3]})
    
    generator.generate_portrait_report(feature_importance, portrait_insights, metrics, df)


if __name__ == "__main__":
    main()
