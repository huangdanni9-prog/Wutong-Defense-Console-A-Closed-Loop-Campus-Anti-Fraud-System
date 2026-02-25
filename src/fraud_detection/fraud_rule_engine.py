"""
Fraud Rule Engine - Black Sample Identification

Seven-Rule Detection System (Multi-Vector Defense):
R1: Low Dispersion (Simbox) - Automated bots with low call diversity
R2: Silent Broadcaster (Wangiri) - Outbound-only spam
R3: Prepaid Burner - New disposable SIMs with high activity
R4: Student Hunter - Numbers targeting students specifically
R5: Device Hopper - Repeated IMEI changes to evade bans
R6: Smishing Bot - High SMS volume, zero voice calls
R7: Short Burst - Ultra-short calls + high volume (robocaller)

Swiss Cheese Defense: Rules catch "dumb" bots, ML catches "smart" humans.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime


class FraudRuleEngine:
    """
    7-Rule fraud detection engine.
    
    Each rule returns transparent reasoning for "Precise Intervention."
    """
    
    def __init__(self):
        self.rule_stats = {}
    
    def apply_all_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all 6 rules to the DataFrame.
        
        Returns DataFrame with columns:
        - rule_hit: bool (any rule triggered)
        - rule_id: str (which rule, e.g., "R1")
        - rule_reason: str (explanation)
        """
        result = df.copy()
        
        # Initialize columns
        result['rule_hit'] = False
        result['rule_id'] = ''
        result['rule_reason'] = ''
        
        # Pre-calculate activation_days for R3
        result = self._calculate_activation_days(result)
        
        # Apply each rule (vectorized for speed)
        r1_mask, r1_reason = self._r1_simbox(result)
        r2_mask, r2_reason = self._r2_silent_broadcaster(result)
        r3_mask, r3_reason = self._r3_prepaid_burner(result)
        r4_mask, r4_reason = self._r4_student_hunter(result)
        r5_mask, r5_reason = self._r5_device_hopper(result)
        r6_mask, r6_reason = self._r6_smishing_bot(result)
        r7_mask, r7_reason = self._r7_short_burst(result)
        
        # Store statistics
        self.rule_stats = {
            'R1_Simbox': r1_mask.sum(),
            'R2_Wangiri': r2_mask.sum(),
            'R3_Burner': r3_mask.sum(),
            'R4_StudentHunter': r4_mask.sum(),
            'R5_DeviceHopper': r5_mask.sum(),
            'R6_SmishingBot': r6_mask.sum(),
            'R7_ShortBurst': r7_mask.sum(),
        }
        
        # Apply rules in priority order (higher priority overwrites)
        # Priority: R4 > R7 > R5 > R6 > R1 > R2 > R3
        rules_priority = [
            ('R3', r3_mask, r3_reason),
            ('R2', r2_mask, r2_reason),
            ('R1', r1_mask, r1_reason),
            ('R6', r6_mask, r6_reason),
            ('R5', r5_mask, r5_reason),
            ('R7', r7_mask, r7_reason),
            ('R4', r4_mask, r4_reason),
        ]
        
        for rule_id, mask, reason_series in rules_priority:
            result.loc[mask, 'rule_hit'] = True
            result.loc[mask, 'rule_id'] = rule_id
            result.loc[mask, 'rule_reason'] = reason_series[mask]
        
        print("\n=== Fraud Rule Engine Results (7 Rules) ===")
        for rule_name, count in self.rule_stats.items():
            print(f"  {rule_name}: {count} hits")
        print(f"  Total Unique Hits: {result['rule_hit'].sum()}")
        
        return result
    
    def _calculate_activation_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate activation_days = observation_date - open_dt.
        Uses DATA_SNAPSHOT_DATE as fallback when stat_dt is missing.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        try:
            from config import DATA_SNAPSHOT_DATE
        except ImportError:
            # Fallback if config not found
            DATA_SNAPSHOT_DATE = 20251124
            print(f"  Warning: config.py not found, using default snapshot date: {DATA_SNAPSHOT_DATE}")
        
        result = df.copy()
        result['activation_days'] = 999  # Default to old
        
        if 'open_dt' not in result.columns:
            print("  Warning: open_dt column missing, R3 rule will not trigger")
            return result
        
        try:
            # Parse dates (format: 20250926.0)
            def parse_date(val):
                if pd.isna(val) or val == 0:
                    return None
                try:
                    return datetime.strptime(str(int(val)), '%Y%m%d')
                except:
                    return None
            
            open_dates = result['open_dt'].apply(parse_date)
            
            # Use stat_dt if available, otherwise use DATA_SNAPSHOT_DATE
            if 'stat_dt' in result.columns:
                stat_dates = result['stat_dt'].apply(parse_date)
                # Fill missing stat_dt with snapshot date
                snapshot = datetime.strptime(str(DATA_SNAPSHOT_DATE), '%Y%m%d')
                stat_dates = stat_dates.fillna(snapshot)
            else:
                # No stat_dt column - use snapshot date for all
                snapshot = datetime.strptime(str(DATA_SNAPSHOT_DATE), '%Y%m%d')
                stat_dates = pd.Series([snapshot] * len(result), index=result.index)
                print(f"  Using DATA_SNAPSHOT_DATE ({DATA_SNAPSHOT_DATE}) for activation_days")
            
            # Calculate days difference (vectorized)
            for i in range(len(result)):
                if open_dates.iloc[i] and stat_dates.iloc[i]:
                    delta = (stat_dates.iloc[i] - open_dates.iloc[i]).days
                    result.iloc[i, result.columns.get_loc('activation_days')] = max(0, delta)
            
            valid_count = (result['activation_days'] < 999).sum()
            print(f"  Calculated activation_days for {valid_count} records")
        except Exception as e:
            print(f"  Warning: Could not calculate activation_days: {e}")
        
        return result
    
    def _r1_simbox(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        R1: Low Dispersion (Simbox Trap)
        Logic: dispersion_rate < 0.04 AND call_count > adaptive_threshold
        
        ADAPTIVE: Threshold = max(50, 5 * avg_calls)
        This prevents false positives on busy network days.
        
        SAFETY SWITCH: If >50% of users have zero dispersion, disable rule.
        """
        disp = df['dispersion_rate'].fillna(0)
        calls = df['call_cnt_day'].fillna(0)
        
        # ADAPTIVE THRESHOLD: max(50, 5 * average)
        avg_calls = calls.mean()
        call_threshold = max(50, int(5 * avg_calls))
        
        # SAFETY SWITCH: Check if dataset is broken
        zero_disp_ratio = (disp == 0).sum() / len(disp) if len(disp) > 0 else 0
        if zero_disp_ratio > 0.50:
            print(f"  R1 SAFETY SWITCH: {zero_disp_ratio:.1%} zero dispersion - RULE DISABLED")
            return pd.Series(False, index=df.index), pd.Series('', index=df.index)
        
        # Fallback for 0 dispersion
        fallback = df['opp_num_stu_cnt'].fillna(0) / (calls + 1)
        disp = disp.where(disp > 0, fallback)
        
        mask = (disp < 0.04) & (calls > call_threshold)
        
        reason = pd.Series('', index=df.index)
        reason[mask] = 'R1_Simbox: Low dispersion (' + disp[mask].round(3).astype(str) + ') with ' + calls[mask].astype(int).astype(str) + ' calls (threshold: ' + str(call_threshold) + ')'
        
        return mask, reason
    
    def _r2_silent_broadcaster(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        R2: Silent Broadcaster (Wangiri)
        Logic: outgoing_calls > adaptive_threshold AND incoming_calls == 0
        
        ADAPTIVE: Threshold = max(50, 3 * avg_calls)
        """
        outgoing = df['call_cnt_day'].fillna(0)
        incoming = df['called_cnt_day'].fillna(0)
        
        # ADAPTIVE THRESHOLD
        avg_calls = outgoing.mean()
        call_threshold = max(50, int(3 * avg_calls))
        
        mask = (outgoing > call_threshold) & (incoming == 0)
        
        reason = pd.Series('', index=df.index)
        reason[mask] = 'R2_Wangiri: ' + outgoing[mask].astype(int).astype(str) + ' outgoing, 0 incoming (threshold: ' + str(call_threshold) + ')'
        
        return mask, reason
    
    def _r3_prepaid_burner(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        R3: Prepaid Burner Profile
        Logic: is_prepaid AND activation_days < 7 AND call_count > 10
        """
        if 'is_prepaid' in df.columns:
            is_prepaid = df['is_prepaid'] == 1
        else:
            mapped = df['post_or_ppd'].astype(str).str.strip().str.upper()
            is_prepaid = mapped.isin(['PPD', 'PREPAID', '预付', '预付卡'])
        activation = df['activation_days'].fillna(999)
        calls = df['call_cnt_day'].fillna(0)
        
        mask = is_prepaid & (activation < 7) & (calls > 10)
        
        reason = pd.Series('', index=df.index)
        reason[mask] = 'R3_Burner: Prepaid, ' + activation[mask].astype(int).astype(str) + ' days old, ' + calls[mask].astype(int).astype(str) + ' calls'
        
        return mask, reason
    
    def _r4_student_hunter(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        R4: Student Hunter (Predator)
        Logic: calls_to_students > 5 OR unique_students > 3
        """
        student_calls = df['call_stu_cnt'].fillna(0)
        unique_students = df['opp_num_stu_cnt'].fillna(0)
        
        mask = (student_calls > 5) | (unique_students > 3)
        
        reason = pd.Series('', index=df.index)
        # Clear explanation based on which condition triggered
        for idx in mask[mask].index:
            calls = int(student_calls.loc[idx])
            students = int(unique_students.loc[idx])
            if students > 0:
                reason.loc[idx] = f'R4_StudentHunter: {calls} calls to {students} unique students'
            else:
                reason.loc[idx] = f'R4_StudentHunter: High student call volume ({calls} calls)'
        
        return mask, reason
    
    def _r5_device_hopper(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        R5: Device Hopper
        Logic: change_imei_times > 1 (Normal people don't change phones twice in a month)
        """
        imei_changes = df['change_imei_times'].fillna(0) if 'change_imei_times' in df.columns else pd.Series(0, index=df.index)
        
        mask = imei_changes > 1
        
        reason = pd.Series('', index=df.index)
        reason[mask] = 'R5_DeviceHopper: Changed IMEI ' + imei_changes[mask].astype(int).astype(str) + ' times'
        
        return mask, reason
    
    def _r6_smishing_bot(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        R6: Smishing Bot (SMS Blaster)
        Logic: tot_msg_cnt > 50 AND call_cnt_day == 0 (High volume SMS, zero voice)
        """
        sms_count = df['tot_msg_cnt'].fillna(0) if 'tot_msg_cnt' in df.columns else pd.Series(0, index=df.index)
        voice_count = df['call_cnt_day'].fillna(0)
        
        mask = (sms_count > 50) & (voice_count == 0)
        
        reason = pd.Series('', index=df.index)
        reason[mask] = 'R6_SmishingBot: ' + sms_count[mask].astype(int).astype(str) + ' SMS, 0 voice calls'
        
        return mask, reason
    
    def _r7_short_burst(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        R7: Short Burst Robocaller
        Logic: avg_call_duration < 15 seconds AND call_count > 30
        
        Ultra-short calls with high volume = automated robocaller
        """
        duration = df['avg_actv_dur'].fillna(60) if 'avg_actv_dur' in df.columns else pd.Series(60, index=df.index)
        calls = df['call_cnt_day'].fillna(0)
        
        mask = (duration < 15) & (calls > 30)
        
        reason = pd.Series('', index=df.index)
        reason[mask] = 'R7_ShortBurst: ' + duration[mask].astype(int).astype(str) + 's avg duration, ' + calls[mask].astype(int).astype(str) + ' calls'
        
        return mask, reason
    
    def _r8_cross_border(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        R8: Cross-Border High Volume
        Logic: mainland_calls > 20 AND total_calls > 50
        
        High volume calls to/from mainland China = cross-border fraud network
        """
        # Try different column names for mainland calls
        mainland_cols = ['from_china_mobile_call_cnt', 'mainland_call_cnt', 'cn_call_cnt']
        mainland = pd.Series(0, index=df.index)
        for col in mainland_cols:
            if col in df.columns:
                mainland = df[col].fillna(0)
                break
        
        calls = df['call_cnt_day'].fillna(0)
        
        mask = (mainland > 20) & (calls > 50)
        
        reason = pd.Series('', index=df.index)
        reason[mask] = 'R8_CrossBorder: ' + mainland[mask].astype(int).astype(str) + ' mainland calls, ' + calls[mask].astype(int).astype(str) + ' total'
        
        return mask, reason
    
    def get_rule_stats(self) -> Dict[str, int]:
        """Get rule hit statistics."""
        return self.rule_stats


def main():
    """Test the 6-rule engine."""
    data_path = Path(__file__).parent.parent.parent / "Datasets" / "Fraud" / "Training and Testing Data" / "fraud_model_2.csv"
    df = pd.read_csv(data_path, low_memory=False)
    
    engine = FraudRuleEngine()
    df_ruled = engine.apply_all_rules(df)
    
    print("\n=== Sample Rule Hits ===")
    hits = df_ruled[df_ruled['rule_hit'] == True].head(10)
    for _, row in hits.iterrows():
        print(f"  {row['rule_id']}: {row['rule_reason'][:70]}")


if __name__ == "__main__":
    main()
