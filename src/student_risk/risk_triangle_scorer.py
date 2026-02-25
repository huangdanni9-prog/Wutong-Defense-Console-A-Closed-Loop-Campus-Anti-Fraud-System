"""
Risk Triangle Scorer for High-Risk Student Identification

Three-Layer Risk Assessment:
1. Identity Vulnerability - Who they are (Mainland student, Freshman, etc.)
2. Threat Exposure - What happens to them (Unknown calls, overseas calls)
3. Risky Behavior - How they react (Pick up, respond, long calls)

This approach enables "Precise Education" - telling students exactly WHY they're at risk.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from utils import load_threat_intelligence
except ImportError:
    load_threat_intelligence = None


MIN_CRITICAL_ENGAGEMENT = 1  # minimum answered or callback count to mark CRITICAL


class RiskTriangleScorer:
    """
    Risk Triangle scoring system for student fraud vulnerability.
    
    Layer 1: Identity Vulnerability (+5)
    Layer 2: Threat Exposure (+10/+5 mainland & overseas, +25 fraud contact)
    Layer 3: Risky Behavior (x2 multiplier, max score 100)
    """
    
    def __init__(self, fraud_msisdn_list: set = None):
        """
        Initialize with optional fraud MSISDN list for cross-reference.
        
        Args:
            fraud_msisdn_list: Set of known fraud MSISDNs from Task 2
        """
        self.fraud_msisdn_list = fraud_msisdn_list or set()
        self.risk_reasons = {}  # Store reasons for each user
    
    def load_fraud_list(self):
        """Load fraud MSISDNs using unified threat intelligence (Task 2 collaboration)."""
        if load_threat_intelligence:
            try:
                self.fraud_msisdn_list, _ = load_threat_intelligence()
                print(f"Loaded {len(self.fraud_msisdn_list)} threat MSISDNs (fraud_model_2 + blacklist.csv)")
            except Exception as e:
                print(f"Warning: Could not load threat intelligence: {e}")
                self.fraud_msisdn_list = set()
        else:
            # Fallback to old method
            fraud_data_path = str(Path(__file__).parent.parent.parent / 
                                  "Datasets" / "Fraud" / "Training and Testing Data" / "fraud_model_2.csv")
            try:
                fraud_df = pd.read_csv(fraud_data_path, low_memory=False)
                fraud_msisdns = fraud_df[fraud_df['audit_status'] == '稽核不通過']['msisdn'].dropna().unique()
                self.fraud_msisdn_list = set(fraud_msisdns)
                print(f"Loaded {len(self.fraud_msisdn_list)} confirmed fraud MSISDNs (fallback)")
            except Exception as e:
                print(f"Warning: Could not load fraud list: {e}")
                self.fraud_msisdn_list = set()
    
    def score_student(self, row: pd.Series) -> Tuple[int, List[str], Dict[str, int]]:
        """
        Score a single student using the Risk Triangle.
        
        Returns:
            Tuple of (risk_score, list_of_reasons, sub_scores_dict)
            sub_scores_dict contains identity_score, exposure_score, behavior_score (0-100 each)
        """
        score = 0
        reasons = []
        
        # ==================================================
        # LAYER 1: IDENTITY VULNERABILITY (GRADIENT 0-100)
        # ==================================================
        # Gradient scale based on age - younger = more vulnerable
        # Discovery Analysis: Cluster 4 avg age = 22, 59% are 18-22
        age = row.get('age', 25)
        
        # Gradient scoring: 18=100, 19=90, 20=80, 21=70, 22=60, 23=40, 24=20, 25+=0
        if age <= 18:
            identity_score_100 = 100
            identity_points = 5
        elif age == 19:
            identity_score_100 = 90
            identity_points = 5
        elif age == 20:
            identity_score_100 = 80
            identity_points = 5
        elif age == 21:
            identity_score_100 = 70
            identity_points = 4
        elif age == 22:
            identity_score_100 = 60
            identity_points = 4
        elif age == 23:
            identity_score_100 = 40
            identity_points = 2
        elif age == 24:
            identity_score_100 = 20
            identity_points = 1
        else:
            identity_score_100 = 0
            identity_points = 0
        
        if identity_points > 0:
            reasons.append(
                f"Age {age}: vulnerability score {identity_score_100} (younger students are more targeted) (+{identity_points})"
            )

        score += identity_points
        
        # ==================================================
        # LAYER 2: THREAT EXPOSURE (0-100 SCALE, CUMULATIVE)
        # ==================================================
        exposure_points = 0
        
        # Calls from Mainland China operators
        # REDUCED: Cluster analysis shows weak correlation (high-risk avg = 0.15)
        mainland_calls = row.get('from_china_mobile_call_cnt', 0) or 0
        if mainland_calls >= 5:
            exposure_points += 5  # Reduced from 10
            reasons.append(
                f"High Mainland call volume ({mainland_calls}); possible cross-border exposure (+5)"
            )
        elif mainland_calls >= 1:
            exposure_points += 2  # Reduced from 5
            reasons.append(
                f"Some Mainland calls ({mainland_calls}); minor exposure signal (+2)"
            )
        
        # Unknown overseas calls (total_voice_cnt) - STRONG signal
        overseas_calls = row.get('total_voice_cnt', 0) or 0
        if overseas_calls >= 10:
            exposure_points += 15  # Raised from 10 - strong correlation
            reasons.append(
                f"High overseas unknown calls ({overseas_calls}); often used for spoofed authority threats (+15)"
            )
        elif overseas_calls >= 5:
            exposure_points += 5
            reasons.append(
                f"Overseas unknown calls ({overseas_calls}); potential phishing probes (+5)"
            )
        
        # SMS received from fraud numbers - NEW RULE
        msg_receive = row.get('msg_receive', 0) or 0
        if msg_receive > 0:
            exposure_points += 5
            reasons.append(
                f"Received {msg_receive} SMS from suspected fraud numbers (+5)"
            )
        
        # ============================================================
        # Contacted by fraud MSISDN - CROSS-REFERENCE WITH BLACKLIST
        # ============================================================
        fraud_msisdn = row.get('fraud_msisdn')
        fraud_contact_confirmed = False
        
        if pd.notna(fraud_msisdn) and str(fraud_msisdn).strip():
            fraud_msisdn_str = str(fraud_msisdn).strip()
            
            # Check if this MSISDN is in our blacklist (from Task 2)
            if self.fraud_msisdn_list and fraud_msisdn_str in self.fraud_msisdn_list:
                # CONFIRMED: The number is in our blacklist
                exposure_points += 25
                fraud_contact_confirmed = True
                reasons.append(
                    f"Contacted by CONFIRMED fraud number ({fraud_msisdn_str[:4]}****); blacklist-verified (+25)"
                )
            else:
                # SUSPECTED: Contact exists but not in our blacklist (yet)
                exposure_points += 15
                reasons.append(
                    f"Contact from suspected fraud number ({fraud_msisdn_str[:4]}****); not yet blacklisted (+15)"
                )

        score += exposure_points
        exposure_score_100 = min(100, (exposure_points / 45) * 100 if exposure_points else 0)
        
        # ==================================================
        # LAYER 3: RISKY BEHAVIOR (0-100 SCALE, MAX EVENT)
        # ==================================================
        behavior_multiplier = 1.0
        behavior_reasons = []
        
        # Calculate behavior score (0-100) based on max risk event
        voice_call = row.get('voice_call', 0) or 0
        voice_receive = row.get('voice_receive', 0) or 0
        msg_call = row.get('msg_call', 0) or 0
        
        if voice_call > 0:
            behavior_score_100 = 100  # Callback = CRITICAL
            behavior_multiplier = 2.0
            behavior_reasons.append(
                f"Called back suspected fraud number {voice_call}x; active engagement doubles risk"
            )
        elif voice_receive > 0:
            behavior_score_100 = 60  # Answered = High
            behavior_multiplier = 1.5
            behavior_reasons.append(
                f"Answered suspected fraud calls {voice_receive}x; indicates partial engagement"
            )
        elif msg_call > 0:
            behavior_score_100 = 50  # SMS Reply = Medium
            behavior_multiplier = 1.5
            behavior_reasons.append(
                f"Replied to suspected fraud messages {msg_call}x; engagement increases exposure"
            )
        else:
            behavior_score_100 = 10  # No action = Safe
        
        # Apply multiplier to total score
        if behavior_multiplier > 1.0:
            score = int(score * behavior_multiplier)
            reasons.extend(behavior_reasons)
            reasons.append(f"Behavior Multiplier: x{behavior_multiplier}")

        # Clamp to 0-100 as new max scale
        score = min(100, score)
        
        # Package sub-scores for Radar Chart
        sub_scores = {
            'identity_score': identity_score_100,
            'exposure_score': exposure_score_100,
            'behavior_score': behavior_score_100
        }
        
        return score, reasons, sub_scores
    
    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all students and add risk columns including sub-scores for Radar Chart.
        """
        result = df.copy()
        
        print("\n=== Risk Triangle Scoring ===")
        print("Layer 1: Identity Vulnerability (0-100)")
        print("Layer 2: Threat Exposure (0-100)")
        print("Layer 3: Risky Behavior (0-100)")
        print()
        
        scores = []
        identity_scores = []
        exposure_scores = []
        behavior_scores = []
        all_reasons = {}
        
        for idx, row in result.iterrows():
            score, reasons, sub_scores = self.score_student(row)
            scores.append(score)
            identity_scores.append(sub_scores['identity_score'])
            exposure_scores.append(sub_scores['exposure_score'])
            behavior_scores.append(sub_scores['behavior_score'])
            if reasons:
                all_reasons[idx] = reasons
        
        result['risk_score'] = scores
        result['identity_score'] = identity_scores
        result['exposure_score'] = exposure_scores
        result['behavior_score'] = behavior_scores
        self.risk_reasons = all_reasons
        
        # ============================================================
        # TIERED CLASSIFICATION: CRITICAL / VULNERABLE / SAFE
        # ============================================================
        # CRITICAL: Active engagement (picked up OR called back fraud)
        # VULNERABLE: High exposure (score >= 50) but no engagement
        # SAFE: Everyone else
        
        # New sensitivity: expose-only students surface sooner (max non-fraud exposure = 25)
        VULNERABLE_THRESHOLD = 20
        
        # Default everyone to SAFE
        result['risk_tier'] = 'SAFE'
        result['is_high_risk'] = 0
        
        # CRITICAL: Meaningful active engagement (thresholded)
        voice_call = pd.to_numeric(result.get('voice_call', 0), errors='coerce').fillna(0)
        voice_receive = pd.to_numeric(result.get('voice_receive', 0), errors='coerce').fillna(0)
        engagement_mask = ((voice_call + voice_receive) >= MIN_CRITICAL_ENGAGEMENT)
        result.loc[engagement_mask, 'risk_tier'] = 'CRITICAL'
        result.loc[engagement_mask, 'is_high_risk'] = 1

        # ============================================================
        # FIX: CRITICAL students with risk_score=0 should have behavior contribute base points
        # This handles edge cases where identity + exposure = 0 but behavior exists
        # ============================================================
        critical_zero_mask = (result['risk_tier'] == 'CRITICAL') & (result['risk_score'] == 0)
        if critical_zero_mask.any():
            # For CRITICAL students with 0 score, use behavior_score as base (scaled to 0-50 range)
            # behavior_score is 0-100, we scale it to contribute 0-50 base points
            behavior_base = (result.loc[critical_zero_mask, 'behavior_score'] * 0.5).astype(int)
            result.loc[critical_zero_mask, 'risk_score'] = behavior_base
            print(f"  [FIX] Adjusted {critical_zero_mask.sum()} CRITICAL students with 0 risk_score")

        # VULNERABLE: high exposure or confirmed fraud contact without engagement
        has_fraud_contact = pd.to_numeric(result.get('has_fraud_contact', 0), errors='coerce').fillna(0) == 1
        vulnerable_mask = (~engagement_mask) & ((result['risk_score'] >= VULNERABLE_THRESHOLD) | has_fraud_contact)
        result.loc[vulnerable_mask, 'risk_tier'] = 'VULNERABLE'
        result.loc[vulnerable_mask, 'is_high_risk'] = 1
        
        # Binary label for downstream portraiting
        result['risk_classification'] = result['is_high_risk'].map({1: 'high_risk', 0: 'low_risk'})
        
        # Print summary
        print(f"\n=== Tier Distribution ===")
        print(f"  CRITICAL (Active Victims): {(result['risk_tier'] == 'CRITICAL').sum():,}")
        print(f"  VULNERABLE (High Exposure): {(result['risk_tier'] == 'VULNERABLE').sum():,}")
        print(f"  SAFE: {(result['risk_tier'] == 'SAFE').sum():,}")
        
        return result
    
    def get_risk_reasons(self, user_id: str = None) -> Dict:
        """Get risk reasons for a specific user or all users."""
        if user_id:
            return self.risk_reasons.get(user_id, [])
        return self.risk_reasons
    
    def get_intervention_message(self, row: pd.Series) -> str:
        """
        Generate a personalized intervention message for a student.
        This enables "Precise Education" - telling them exactly why they're at risk.
        """
        _, reasons = self.score_student(row)
        
        if not reasons:
            return "No specific risk factors identified."
        
        message = "You may be at risk because:\n"
        for reason in reasons:
            if "Mainland" in reason:
                message += "• You receive calls from unknown Mainland China numbers\n"
            elif "Overseas" in reason:
                message += "• You receive calls from unfamiliar overseas numbers\n"
            elif "Answered" in reason:
                message += "• You have answered calls from potential fraud numbers\n"
            elif "CALLED BACK" in reason:
                message += "⚠️ You have CALLED BACK potential fraud numbers - this is high risk!\n"
            elif "Fraud Number" in reason:
                message += "⚠️ You have been contacted by known fraud accounts\n"
        
        return message


def main():
    """Test the Risk Triangle Scorer."""
    from feature_engineering import StudentFeatureEngineer
    
    scorer = RiskTriangleScorer()
    scorer.load_fraud_list()
    
    engineer = StudentFeatureEngineer()
    df = engineer.load_data()
    
    df_scored = scorer.score_dataframe(df)
    
    print("\n=== Score Distribution ===")
    print(df_scored['risk_score'].describe())
    
    print("\n=== Sample High-Risk Reasons ===")
    high_risk = df_scored[df_scored['is_high_risk'] == 1].head(5)
    for idx in high_risk.index[:3]:
        reasons = scorer.risk_reasons.get(idx, [])
        print(f"\nStudent {idx}: Score={df_scored.loc[idx, 'risk_score']}")
        for r in reasons:
            print(f"  - {r}")


if __name__ == "__main__":
    main()
