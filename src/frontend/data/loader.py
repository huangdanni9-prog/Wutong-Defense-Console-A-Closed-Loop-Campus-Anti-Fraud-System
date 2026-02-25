"""
Data loaders with Streamlit caching.
"""
import pandas as pd
import streamlit as st

# Use frontend config to avoid clashing with root config
from frontend.config import (
    STUDENT_PREDICTIONS_PATH,
    STUDENT_RAW_PATH,
    BLACKLIST_PATH,
    GREYLIST_PATH,
    CACHE_TTL,
)


@st.cache_data(ttl=CACHE_TTL)
def load_student_data() -> pd.DataFrame:
    """
    Load and merge student predictions with raw data.
    Returns complete student profiles with sub-scores.
    """
    # Load predictions (scores, tiers, reasons)
    predictions = pd.read_csv(STUDENT_PREDICTIONS_PATH)
    if 'user_id' in predictions.columns:
        predictions = predictions.drop_duplicates(subset=['user_id'])
    
    # Load RAW student_model.csv (age, voice_cnt, fraud_msisdn, etc.)
    raw_data = pd.read_csv(STUDENT_RAW_PATH, low_memory=False)
    if 'user_id' in raw_data.columns:
        raw_data = raw_data.drop_duplicates(subset=['user_id'])
    
    # Merge on user_id to get full profile
    merged = predictions.merge(raw_data, on='user_id', how='left')
    
    # Force numeric types for scores (avoid string conversion issues)
    score_cols = ['risk_score', 'identity_score', 'exposure_score', 'behavior_score']
    for c in score_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors='coerce').fillna(0).astype(int)
    
    # Guarantee one row per student after merge
    if 'user_id' in merged.columns:
        merged = merged.drop_duplicates(subset=['user_id'])
    return merged


@st.cache_data(ttl=CACHE_TTL)
def load_blacklist() -> pd.DataFrame:
    """Load confirmed fraud MSISDNs."""
    return pd.read_csv(BLACKLIST_PATH)


@st.cache_data(ttl=CACHE_TTL)
def load_greylist() -> pd.DataFrame:
    """Load suspicious MSISDNs (Grey-Zone Discovery)."""
    try:
        return pd.read_csv(GREYLIST_PATH)
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL)
def load_fraud_student_network() -> pd.DataFrame:
    """
    Load fraud-student network data for visualization.
    Returns students who have been contacted by fraud numbers.
    
    Columns: user_id, fraud_msisdn, risk_tier, risk_score, identity_score, 
             exposure_score, behavior_score, age, gndr, hk_resident_type
    """
    # Load merged student data (predictions + raw)
    student_df = load_student_data()
    
    # Filter to students with fraud_msisdn (contacted by fraud numbers)
    network_df = student_df[student_df['fraud_msisdn'].notna()].copy()
    
    # Also include frequently_opp_num if it matches blacklist
    blacklist = load_blacklist()
    blacklist_msisdns = set(blacklist['msisdn'].dropna().astype(str).tolist())
    
    # Check if frequently_opp_num is in blacklist
    if 'frequently_opp_num' in student_df.columns:
        additional_victims = student_df[
            (student_df['fraud_msisdn'].isna()) &  # Not already in network
            (student_df['frequently_opp_num'].astype(str).isin(blacklist_msisdns))
        ].copy()
        
        if len(additional_victims) > 0:
            # Use frequently_opp_num as fraud_msisdn for these
            additional_victims['fraud_msisdn'] = additional_victims['frequently_opp_num']
            network_df = pd.concat([network_df, additional_victims], ignore_index=True)
    
    # Select relevant columns
    columns_to_keep = [
        'user_id', 'fraud_msisdn', 'risk_tier', 'risk_score',
        'identity_score', 'exposure_score', 'behavior_score',
        'age', 'gndr', 'hk_resident_type', 'risk_reason'
    ]
    available_cols = [c for c in columns_to_keep if c in network_df.columns]
    network_df = network_df[available_cols].copy()
    
    # Remove duplicates (same student-fraud pair)
    network_df = network_df.drop_duplicates(subset=['user_id', 'fraud_msisdn'])
    
    return network_df


@st.cache_data(ttl=CACHE_TTL)
def load_fraud_profiles() -> pd.DataFrame:
    """Load fraud number profiles with detailed information."""
    try:
        from frontend.config import FRAUD_PROFILES_PATH
        return pd.read_csv(FRAUD_PROFILES_PATH)
    except (FileNotFoundError, ImportError):
        return pd.DataFrame()


def clear_all_caches():
    """Clear all cached data (for manual refresh button)."""
    st.cache_data.clear()
