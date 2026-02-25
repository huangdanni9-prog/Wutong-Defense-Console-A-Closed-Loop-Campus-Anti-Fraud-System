"""
Wutong Cup Solution - Shared Utilities

Common functions used by both Task 1 (Student Risk) and Task 2 (Fraud Detection).
"""

import pandas as pd
from pathlib import Path
from typing import Set, Tuple

try:
    from config import (
        FRAUD_TRAIN_PATH, BLACKLIST_PATH, GREYLIST_PATH,
        FRAUD_RESULTS_DIR, STUDENT_RESULTS_DIR
    )
except ImportError:
    from src.config import (
        FRAUD_TRAIN_PATH, BLACKLIST_PATH, GREYLIST_PATH,
        FRAUD_RESULTS_DIR, STUDENT_RESULTS_DIR
    )


def load_threat_intelligence() -> Tuple[Set[str], pd.DataFrame]:
    """
    Load unified Threat Intelligence database.
    
    Combines:
    1. Old confirmed fraud from fraud_model_2.csv (audit_status == '稽核不通過')
    2. New detected fraud from blacklist.csv (Task 2 output)
    3. Suspicious numbers from greylist.csv (Grey-zone discovery)
    
    This ensures both Task 1 (Student Risk) and Task 2 (Fraud Detection)
    look at the exact same "Threat Database."
    
    Returns:
        Tuple of:
        - fraud_msisdns: Set of known fraud phone numbers
        - threat_df: DataFrame with full threat details
    """
    fraud_msisdns = set()
    threat_records = []
    
    # Source 1: Old confirmed fraud (fraud_model_2.csv)
    try:
        fraud_df = pd.read_csv(FRAUD_TRAIN_PATH, low_memory=False)
        confirmed_fraud = fraud_df[fraud_df['audit_status'] == '稽核不通過']
        old_msisdns = set(confirmed_fraud['msisdn'].dropna().astype(str))
        fraud_msisdns.update(old_msisdns)
        
        for msisdn in old_msisdns:
            threat_records.append({
                'msisdn': msisdn,
                'source': 'fraud_model_2',
                'confidence': 'CONFIRMED'
            })
        print(f"  Loaded {len(old_msisdns)} confirmed fraud MSISDNs from fraud_model_2")
    except Exception as e:
        print(f"  Warning: Could not load fraud_model_2: {e}")
    
    # Source 2: New detected fraud (blacklist.csv)
    try:
        if BLACKLIST_PATH.exists():
            blacklist_df = pd.read_csv(BLACKLIST_PATH)
            new_msisdns = set(blacklist_df['msisdn'].dropna().astype(str))
            fraud_msisdns.update(new_msisdns)
            
            for _, row in blacklist_df.iterrows():
                threat_records.append({
                    'msisdn': str(row['msisdn']),
                    'source': 'blacklist',
                    'confidence': 'DETECTED',
                    'trigger_reason': row.get('trigger_reason', '')
                })
            print(f"  Loaded {len(new_msisdns)} detected fraud MSISDNs from blacklist.csv")
        else:
            print(f"  Note: blacklist.csv not found - run Task 2 first")
    except Exception as e:
        print(f"  Warning: Could not load blacklist: {e}")
    
    # Source 3: Suspicious numbers (greylist.csv) - Grey-zone discovery
    try:
        if GREYLIST_PATH.exists():
            greylist_df = pd.read_csv(GREYLIST_PATH)
            grey_msisdns = set(greylist_df['msisdn'].dropna().astype(str))
            fraud_msisdns.update(grey_msisdns)
            
            for _, row in greylist_df.iterrows():
                threat_records.append({
                    'msisdn': str(row['msisdn']),
                    'source': 'greylist',
                    'confidence': 'SUSPICIOUS',
                    'trigger_reason': row.get('trigger_reason', '')
                })
            print(f"  Loaded {len(grey_msisdns)} suspicious MSISDNs from greylist.csv")
        else:
            print(f"  Note: greylist.csv not found")
    except Exception as e:
        print(f"  Warning: Could not load greylist: {e}")
    
    # Create consolidated DataFrame
    threat_df = pd.DataFrame(threat_records)
    
    print(f"  Total unique threat MSISDNs: {len(fraud_msisdns)}")
    
    return fraud_msisdns, threat_df


def ensure_directories():
    """Create all required output directories."""
    FRAUD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STUDENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_numeric(series: pd.Series, default: float = 0) -> pd.Series:
    """Convert series to numeric, filling NaN with default value."""
    return pd.to_numeric(series, errors='coerce').fillna(default)


def mask_pii(value: str, show_chars: int = 4) -> str:
    """
    Mask personally identifiable information for privacy-compliant display.
    Shows first and last characters with **** in the middle.
    
    Args:
        value: The PII string to mask (user_id, msisdn, etc.)
        show_chars: Number of characters to show at start and end
    
    Returns:
        Masked string like "m6Ki****ZA==" or "8521****5678"
    
    Examples:
        mask_pii("m6KiO9aD0AUOLEqEUHdSZA==") -> "m6Ki****ZA=="
        mask_pii("85212345678") -> "8521****5678"
    """
    if pd.isna(value):
        return ""
    s = str(value)
    if len(s) <= show_chars * 2:
        return s
    return s[:show_chars] + "****" + s[-show_chars:]


def mask_dataframe_pii(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Apply PII masking to specified columns in a DataFrame.
    
    Args:
        df: DataFrame to mask
        columns: List of column names to mask
    
    Returns:
        DataFrame with masked columns
    """
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = result[col].apply(mask_pii)
    return result


# ============================================================================
# Differential Privacy Utilities
# ============================================================================

def dp_count(true_count: int, epsilon: float = 1.0) -> int:
    """
    Apply Differential Privacy to a count using Laplace mechanism.
    
    This protects individual privacy in aggregate statistics by adding
    calibrated noise. Higher epsilon = less noise but less privacy.
    
    Args:
        true_count: The actual count to protect
        epsilon: Privacy budget (0.1 = high privacy, 1.0 = balanced, 10.0 = low privacy)
    
    Returns:
        Privacy-preserving noisy count (always >= 0)
    
    Example:
        critical_count = dp_count(627, epsilon=1.0)  # Returns ~627 ± some noise
    """
    try:
        from diffprivlib.mechanisms import Laplace
        mechanism = Laplace(epsilon=epsilon, sensitivity=1)
        noisy_count = true_count + mechanism.randomise(0)
        return max(0, int(noisy_count))
    except ImportError:
        # Fallback: use numpy for basic Laplace noise
        import numpy as np
        noise = np.random.laplace(0, 1/epsilon)
        return max(0, int(true_count + noise))


def dp_mean(values: list, epsilon: float = 1.0, bounds: tuple = None) -> float:
    """
    Apply Differential Privacy to a mean calculation.
    
    Args:
        values: List of numeric values
        epsilon: Privacy budget
        bounds: (min, max) bounds for the values. If None, estimated from data.
    
    Returns:
        Privacy-preserving mean
    """
    import numpy as np
    
    if len(values) == 0:
        return 0.0
    
    values = np.array(values)
    
    if bounds is None:
        bounds = (float(np.min(values)), float(np.max(values)))
    
    sensitivity = (bounds[1] - bounds[0]) / len(values)
    
    try:
        from diffprivlib.mechanisms import Laplace
        mechanism = Laplace(epsilon=epsilon, sensitivity=sensitivity)
        true_mean = float(np.mean(values))
        noisy_mean = true_mean + mechanism.randomise(0)
        return noisy_mean
    except ImportError:
        # Fallback
        true_mean = float(np.mean(values))
        noise = np.random.laplace(0, sensitivity/epsilon)
        return true_mean + noise


def dp_histogram(values: list, bins: int = 10, epsilon: float = 1.0) -> tuple:
    """
    Apply Differential Privacy to a histogram.
    
    Args:
        values: List of numeric values
        bins: Number of histogram bins
        epsilon: Privacy budget (split across bins)
    
    Returns:
        Tuple of (noisy_counts, bin_edges)
    """
    import numpy as np
    
    values = np.array(values)
    counts, bin_edges = np.histogram(values, bins=bins)
    
    # Apply DP to each bin count (split epsilon across bins)
    bin_epsilon = epsilon / bins
    noisy_counts = [dp_count(int(c), epsilon=bin_epsilon) for c in counts]
    
    return noisy_counts, bin_edges


# ============================================================================
# Whitelist Utilities
# ============================================================================

WHITELIST_PATH = FRAUD_RESULTS_DIR / "whitelist.csv"


def load_whitelist() -> pd.DataFrame:
    """
    Load the whitelist CSV file.
    
    Returns:
        pd.DataFrame: Whitelist data with columns:
            - msisdn: Account identifier
            - original_risk_score: Risk score before whitelist
            - original_risk_tier: Risk tier before whitelist (BLACK/GREY)
            - original_source: Detection source (RULE/ML)
            - original_trigger_reason: Reason for initial flag
            - original_source_file: Source data file
            - whitelist_reason: Reason for whitelist approval
            - reviewed_by: Staff member who approved
            - review_date: Date of approval
            - review_notes: Additional notes
    """
    if WHITELIST_PATH.exists():
        return pd.read_csv(WHITELIST_PATH)
    return pd.DataFrame(columns=[
        'msisdn', 'original_risk_score', 'original_risk_tier', 
        'original_source', 'original_trigger_reason', 'original_source_file',
        'whitelist_reason', 'reviewed_by', 'review_date', 'review_notes'
    ])


def save_whitelist(df: pd.DataFrame) -> None:
    """
    Save whitelist DataFrame to CSV.
    
    Args:
        df: Whitelist DataFrame to save
    """
    df.to_csv(WHITELIST_PATH, index=False)


def get_whitelisted_msisdns() -> Set[str]:
    """
    Get set of all whitelisted MSISDNs.
    
    Returns:
        Set[str]: Set of whitelisted account identifiers
    """
    whitelist_df = load_whitelist()
    if len(whitelist_df) > 0:
        return set(whitelist_df['msisdn'].tolist())
    return set()


def is_whitelisted(msisdn: str) -> bool:
    """
    Check if a specific MSISDN is whitelisted.
    
    Args:
        msisdn: Account identifier to check
        
    Returns:
        bool: True if whitelisted, False otherwise
    """
    return msisdn in get_whitelisted_msisdns()


def get_whitelist_for_training() -> pd.DataFrame:
    """
    Get whitelist data formatted for model training.
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - msisdn: Account identifier
            - is_whitelisted: Always 1 (for training labels)
            - whitelist_reason: Reason for approval
    """
    whitelist_df = load_whitelist()
    if len(whitelist_df) == 0:
        return pd.DataFrame(columns=['msisdn', 'is_whitelisted', 'whitelist_reason'])
    
    training_df = whitelist_df[['msisdn', 'whitelist_reason']].copy()
    training_df['is_whitelisted'] = 1
    return training_df


def filter_whitelisted_accounts(df: pd.DataFrame, msisdn_column: str = 'msisdn') -> pd.DataFrame:
    """
    Filter out whitelisted accounts from a DataFrame.
    
    Args:
        df: DataFrame containing flagged accounts
        msisdn_column: Name of the column containing account identifiers
        
    Returns:
        pd.DataFrame: DataFrame with whitelisted accounts removed
    """
    whitelisted = get_whitelisted_msisdns()
    if len(whitelisted) == 0:
        return df
    return df[~df[msisdn_column].isin(whitelisted)]


def get_whitelist_reasons() -> list:
    """
    Get list of unique whitelist reasons.
    
    Returns:
        List[str]: List of unique reasons used for whitelist approval
    """
    whitelist_df = load_whitelist()
    if len(whitelist_df) > 0:
        return whitelist_df['whitelist_reason'].unique().tolist()
    return []


def add_whitelist_labels_to_training_data(
    training_df: pd.DataFrame,
    label_column: str = 'is_fraud',
    msisdn_column: str = 'msisdn'
) -> pd.DataFrame:
    """
    Add whitelist information to training data.
    Marks whitelisted accounts as non-fraudulent (label = 0).
    
    Args:
        training_df: Training DataFrame
        label_column: Name of the fraud label column
        msisdn_column: Name of the MSISDN column
        
    Returns:
        pd.DataFrame: Training data with whitelist labels applied
    """
    whitelisted = get_whitelisted_msisdns()
    result_df = training_df.copy()
    
    if len(whitelisted) > 0 and label_column in result_df.columns:
        # Set whitelisted accounts as non-fraud
        result_df.loc[result_df[msisdn_column].isin(whitelisted), label_column] = 0
    
    return result_df


def export_whitelist_for_model(output_path: str = None) -> str:
    """
    Export whitelist in a format suitable for model training import.
    
    Args:
        output_path: Optional custom output path
        
    Returns:
        str: Path to exported file
    """
    if output_path is None:
        output_path = FRAUD_RESULTS_DIR / "whitelist_for_training.csv"
    
    training_df = get_whitelist_for_training()
    training_df.to_csv(output_path, index=False)
    
    return str(output_path)


def get_whitelist_stats() -> dict:
    """
    Get whitelist statistics.
    
    Returns:
        dict: Statistics including count, reasons breakdown, reviewers
    """
    whitelist_df = load_whitelist()
    
    stats = {
        'total_count': len(whitelist_df),
        'reasons': {},
        'reviewers': {},
        'by_original_tier': {}
    }
    
    if len(whitelist_df) > 0:
        stats['reasons'] = whitelist_df['whitelist_reason'].value_counts().to_dict()
        stats['reviewers'] = whitelist_df['reviewed_by'].value_counts().to_dict()
        stats['by_original_tier'] = whitelist_df['original_risk_tier'].value_counts().to_dict()
    
    return stats


if __name__ == "__main__":
    # Test the threat intelligence loader
    print("Loading Threat Intelligence...")
    msisdns, df = load_threat_intelligence()
    print(f"\nLoaded {len(msisdns)} total threat MSISDNs")
    print(f"DataFrame shape: {df.shape}")

