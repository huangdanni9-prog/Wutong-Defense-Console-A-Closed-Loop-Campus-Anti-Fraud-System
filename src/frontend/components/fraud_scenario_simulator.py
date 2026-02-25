"""
Fraud Scenario Simulator - Hong Kong Campus Anti-Fraud Real-time Event Simulation

Simulates the complete fraud chain:
Contact → Detection → Alert → Intervention → Outcome

Based on real Hong Kong 2025 data, driven by real anonymized data
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta

# Import real data loader
try:
    from data.loader import load_student_data, load_blacklist, load_greylist, load_fraud_student_network
    from utils import mask_pii
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False

from pathlib import Path

# Import FraudRuleEngine for real-time detection testing
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from fraud_detection.fraud_rule_engine import FraudRuleEngine
    RULE_ENGINE_AVAILABLE = True
except (ImportError, NameError):
    RULE_ENGINE_AVAILABLE = False

# Paths to fraud data sources
FRAUD_DATA_DIR = Path(__file__).parent.parent.parent.parent / "Datasets" / "Fraud" / "Training and Testing Data"
FRAUD_MODEL_2_PATH = FRAUD_DATA_DIR / "fraud_model_2.csv"
VALIDATION_DATA_PATH = FRAUD_DATA_DIR / "validate_data.csv"


@st.cache_data(ttl=300)
def load_raw_fraud_data():
    """
    Load confirmed frauds (label=1) from fraud_model_2 and validate_data.
    This tests detection accuracy: how many TRUE frauds do our rules catch?
    """
    try:
        dfs = []
        
        # Load fraud_model_2 (has 'label' column)
        if FRAUD_MODEL_2_PATH.exists():
            df2 = pd.read_csv(FRAUD_MODEL_2_PATH)
            if 'label' in df2.columns:
                frauds_2 = df2[df2['label'] == 1]
                dfs.append(frauds_2)
        
        # Load validate_data (has 'label' column)
        if VALIDATION_DATA_PATH.exists():
            df_val = pd.read_csv(VALIDATION_DATA_PATH)
            if 'label' in df_val.columns:
                frauds_val = df_val[df_val['label'] == 1]
                dfs.append(frauds_val)
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            # Remove duplicates by msisdn if exists
            if 'msisdn' in combined.columns:
                combined = combined.drop_duplicates(subset=['msisdn'])
            return combined
        return None
    except Exception as e:
        st.warning(f"Failed to load confirmed fraud data: {e}")
        return None


def generate_synthetic_transaction(fraud_probability: float = 0.7, raw_fraud_df=None):
    """
    Generate a synthetic transaction for simulation (CTGAN-style approach).
    
    Args:
        fraud_probability: Probability that the generated transaction is fraudulent
        raw_fraud_df: Optional real fraud data to sample patterns from
        
    Returns:
        dict with transaction features and ground truth label
    """
    is_fraud = random.random() < fraud_probability
    
    if is_fraud:
        # Generate FRAUD-like transaction patterns
        # These should sometimes trigger rules, sometimes not
        
        # Randomize how "obvious" the fraud is (some will be caught, some missed)
        detection_difficulty = random.random()  # 0 = easy to detect, 1 = hard
        
        if detection_difficulty < 0.75:  # ~75% are detectable frauds
            # High-signal fraud (will likely trigger rules)
            call_cnt_day = random.randint(80, 300)
            outcl_cnt_day = random.randint(70, call_cnt_day)
            incl_cnt_day = random.randint(0, 10)  # Low incoming = Wangiri pattern
            opp_num_stu_cnt = random.randint(6, 50)  # High student targeting
            dispersion_rate = random.uniform(0.01, 0.05)  # Low dispersion = Simbox
        else:
            # Low-signal fraud (will likely evade rules)
            call_cnt_day = random.randint(10, 50)
            outcl_cnt_day = random.randint(5, 30)
            incl_cnt_day = random.randint(5, 20)  # Has incoming calls
            opp_num_stu_cnt = random.randint(1, 5)  # Low student count
            dispersion_rate = random.uniform(0.3, 0.8)  # Normal dispersion
        
        amount = random.lognormvariate(11, 1.5)  # Higher amounts for fraud
        fraud_type = random.choice(list(FRAUD_TYPES.keys()))
    else:
        # Generate LEGITIMATE transaction patterns
        call_cnt_day = random.randint(5, 30)
        outcl_cnt_day = random.randint(2, 15)
        incl_cnt_day = random.randint(3, 20)
        opp_num_stu_cnt = random.randint(0, 2)
        dispersion_rate = random.uniform(0.4, 0.9)
        amount = random.lognormvariate(9, 1.0)
        fraud_type = None
    
    # Create feature record matching fraud_model columns (all required by rule engine)
    record = {
        'msisdn': f"852{random.randint(10000000, 99999999)}",
        # Call counts
        'call_cnt_day': call_cnt_day,
        'outcl_cnt_day': outcl_cnt_day,
        'incl_cnt_day': incl_cnt_day,
        'called_cnt_day': incl_cnt_day,  # R2 uses this for incoming calls
        # Student targeting
        'opp_num_stu_cnt': opp_num_stu_cnt,
        'call_stu_cnt': opp_num_stu_cnt * 2 if is_fraud else 0,  # R4 uses this
        # Dispersion
        'dispersion_rate': dispersion_rate,
        # Device changes
        'change_imei_times': random.randint(2, 5) if (is_fraud and random.random() < 0.3) else 0,  # R5
        # SMS (smishing)
        'tot_msg_cnt': random.randint(60, 200) if (is_fraud and random.random() < 0.2) else random.randint(0, 20),  # R6
        # Prepaid status
        'post_or_ppd': 'PPD' if random.random() < 0.6 else 'POST',
        'open_dt': 20251120 if (is_fraud and random.random() < 0.4) else 20250101,  # R3 checks activation days
        # Call duration
        'call_dur_avg': random.uniform(30, 180) if is_fraud else random.uniform(60, 300),
        # Label
        'label': 1 if is_fraud else 0,
        # Metadata for display
        '_is_fraud_ground_truth': is_fraud,
        '_fraud_type': fraud_type,
        '_amount': amount
    }
    
    return record


# ============================================
# SIMULATION DATA CONFIGURATION
# ============================================

# Fraud types configuration (based on common Hong Kong fraud methods)
FRAUD_TYPES = {
    "Government Impersonation": {
        "icon": "👮",
        "description": "Impersonating mainland police/prosecutors, claiming involvement in money laundering cases",
        "avg_loss": 500000,
        "success_rate": 0.08,
        "target_profile": "Mainland Student",
        "call_pattern": "Overseas call → Transfer to 'police' → Request video statement",
        "risk_level": "CRITICAL"
    },
    "Investment Scam": {
        "icon": "📈",
        "description": "Fake investment platforms promising high returns with low risk",
        "avg_loss": 200000,
        "success_rate": 0.05,
        "target_profile": "Graduate/PhD Student",
        "call_pattern": "Social media recruitment → Small profits → Large investment",
        "risk_level": "HIGH"
    },
    "Shopping Refund": {
        "icon": "🛒",
        "description": "Impersonating e-commerce customer service, claiming order anomalies requiring refund",
        "avg_loss": 30000,
        "success_rate": 0.12,
        "target_profile": "Undergraduate",
        "call_pattern": "Precise order info → Guide to download APP → Steal bank info",
        "risk_level": "MEDIUM"
    },
    "Family Emergency": {
        "icon": "👨‍👩‍👧",
        "description": "Impersonating family/friends, claiming urgent need for money transfer",
        "avg_loss": 80000,
        "success_rate": 0.06,
        "target_profile": "Freshman",
        "call_pattern": "Obtain social info → Mimic tone → Emergency request",
        "risk_level": "HIGH"
    },
    "Part-time Job Scam": {
        "icon": "💼",
        "description": "Fake part-time jobs, small returns first then large scams",
        "avg_loss": 50000,
        "success_rate": 0.15,
        "target_profile": "Undergraduate",
        "call_pattern": "Social platform recruitment → Small task rewards → Large advance payment",
        "risk_level": "MEDIUM"
    },
    "Sextortion": {
        "icon": "⚠️",
        "description": "Inducing explicit video chat followed by extortion",
        "avg_loss": 100000,
        "success_rate": 0.04,
        "target_profile": "Male Student",
        "call_pattern": "Dating app contact → Video inducement → Threatening extortion",
        "risk_level": "CRITICAL"
    }
}

# Hong Kong universities list
HK_UNIVERSITIES = [
    "HKU", "CUHK", "HKUST", "PolyU",
    "CityU", "HKBU", "LingnanU", "EdUHK"
]

# Student identity mapping
RESIDENT_TYPE_MAP = {
    "新来港内地人港漂": "Mainland Student",
    "新来港港人": "Local Student",
    "香港人": "Local Student",
    "其他": "Exchange/Other"
}

# Fraud number patterns
FRAUD_NUMBER_PATTERNS = [
    {"prefix": "+86", "type": "Overseas Call", "risk": "HIGH"},
    {"prefix": "+852 5", "type": "New Prepaid SIM", "risk": "MEDIUM"},
    {"prefix": "+852 6", "type": "Local Mobile", "risk": "LOW"},
    {"prefix": "+1", "type": "Overseas Call", "risk": "HIGH"},
    {"prefix": "Unknown", "type": "Hidden Caller ID", "risk": "CRITICAL"},
]


def load_real_data():
    """
    Load real data for simulation.
    
    NEW APPROACH: Use BLACKLIST as the primary source of fraud numbers.
    This ensures the simulator shows events from numbers our system actually detected.
    """
    if not DATA_AVAILABLE:
        return None, None, None
    
    try:
        # Load blacklist - this is what our system detected
        blacklist_df = load_blacklist()
        # Load greylist
        greylist_df = load_greylist()
        # Load fraud-student network for student targets
        network_df = load_fraud_student_network()
        
        return blacklist_df, blacklist_df, greylist_df, network_df
    except Exception as e:
        st.warning(f"Unable to load real data: {e}")
        return None, None, None, None





def generate_synthetic_event(fraud_probability: float = 0.7, network_df=None):
    """
    Generate event using CTGAN-style synthetic transaction and run rule engine.
    
    This gives us:
    - True Positives: Fraud generated AND detected
    - False Negatives: Fraud generated but NOT detected
    - False Positives: Legit generated but flagged as fraud
    - True Negatives: Legit generated and NOT flagged
    """
    # Generate synthetic transaction
    record = generate_synthetic_transaction(fraud_probability)
    
    is_ground_truth_fraud = record['_is_fraud_ground_truth']
    fraud_type = record.get('_fraud_type')
    potential_loss = record.get('_amount', 50000)
    fraud_msisdn = record['msisdn']
    
    # Run the actual rule engine
    if RULE_ENGINE_AVAILABLE:
        engine = FraudRuleEngine()
        single_df = pd.DataFrame([record])
        result_df = engine.apply_all_rules(single_df)
        
        if len(result_df) > 0:
            rule_hit = result_df.iloc[0].get('rule_hit', False)
            rule_id = result_df.iloc[0].get('rule_id', '')
            rule_reason = result_df.iloc[0].get('rule_reason', '')
        else:
            rule_hit = False
            rule_id = ''
            rule_reason = ''
    else:
        # Simple heuristic fallback
        rule_hit = (record['opp_num_stu_cnt'] > 5 or 
                   record['incl_cnt_day'] == 0 or 
                   record['dispersion_rate'] < 0.05)
        rule_id = 'HEURISTIC' if rule_hit else ''
        rule_reason = 'Matched heuristic rules' if rule_hit else ''
    
    # Determine outcome category
    if is_ground_truth_fraud and rule_hit:
        outcome_type = "TRUE_POSITIVE"  # Fraud caught
    elif is_ground_truth_fraud and not rule_hit:
        outcome_type = "FALSE_NEGATIVE"  # Fraud missed!
    elif not is_ground_truth_fraud and rule_hit:
        outcome_type = "FALSE_POSITIVE"  # False alarm
    else:
        outcome_type = "TRUE_NEGATIVE"  # Correctly ignored
    
    # Use fraud type from generation or pick based on rule
    if fraud_type is None:
        if 'R4' in rule_id or 'Student' in rule_reason:
            fraud_type = "Government Impersonation"
        elif 'R2' in rule_id:
            fraud_type = "Wangiri (Callback) Scam"
        elif 'R1' in rule_id:
            fraud_type = "Simbox Fraud"
        else:
            fraud_type = "Legitimate Transaction" if not is_ground_truth_fraud else random.choice(list(FRAUD_TYPES.keys()))
    
    fraud_info = FRAUD_TYPES.get(fraud_type, FRAUD_TYPES["Government Impersonation"])
    number_type, number_risk = classify_phone_number(fraud_msisdn)
    
    # Generate student info
    student_info = {
        'user_id': f"STU{random.randint(100000, 999999)}",
        'age': random.randint(18, 30),
        'gender': random.choice(['M', 'F']),
        'resident_type': '新来港内地人港漂',
        'student_risk_tier': 'VULNERABLE' if is_ground_truth_fraud else 'SAFE'
    }
    
    if network_df is not None and len(network_df) > 0:
        student = network_df.sample(1).iloc[0]
        student_info = {
            'user_id': student.get('user_id', student_info['user_id']),
            'age': student.get('age', student_info['age']),
            'gender': student.get('gndr', student_info['gender']),
            'resident_type': student.get('hk_resident_type', student_info['resident_type']),
            # For legitimate transactions, always use SAFE tier
            'student_risk_tier': 'SAFE' if not is_ground_truth_fraud else student.get('risk_tier', student_info['student_risk_tier'])
        }
    
    student_type = RESIDENT_TYPE_MAP.get(student_info['resident_type'], "Other")
    
    event = {
        "timestamp": datetime.now(),
        "event_id": f"SYN{random.randint(10000, 99999)}",
        "fraud_type": fraud_type,
        "fraud_icon": fraud_info.get("icon", "📊"),
        "fraud_desc": fraud_info.get("description", "Synthetic transaction"),
        "caller_number": mask_pii(fraud_msisdn) if DATA_AVAILABLE else fraud_msisdn,
        "original_msisdn": fraud_msisdn,
        "number_type": number_type,
        "number_risk": number_risk,
        "target_student": mask_pii(str(student_info['user_id'])) if DATA_AVAILABLE else str(student_info['user_id']),
        "student_type": student_type,
        "university": random.choice(HK_UNIVERSITIES),
        "age": student_info['age'],
        "gender": "Male" if student_info['gender'] == "M" else "Female",
        "call_duration": record.get('call_dur_avg', 60),
        "risk_score": random.randint(70, 95) if is_ground_truth_fraud else random.randint(10, 30),
        "risk_tier": student_info['student_risk_tier'],
        "potential_loss": potential_loss if is_ground_truth_fraud else 0,
        "call_pattern": fraud_info.get("call_pattern", "Normal activity") if is_ground_truth_fraud else "Normal call pattern",
        "risk_level": fraud_info.get("risk_level", "MEDIUM") if is_ground_truth_fraud else "SAFE",
        "trigger_reason": rule_reason,
        "detection_source": rule_id if rule_hit else "NOT_DETECTED",
        "is_real_data": False,
        "is_synthetic": True,
        "is_detected": rule_hit,
        "is_ground_truth_fraud": is_ground_truth_fraud,
        "outcome_type": outcome_type,
        "rule_id": rule_id,
        "miss_reason": "" if rule_hit else analyze_miss_reason_raw(record)
    }
    
    return event


def analyze_miss_reason_raw(record) -> str:
    """Analyze why a raw fraud record was NOT detected by rules."""
    reasons = []
    
    # Check for low call volume (R1/R2 need higher volume)
    call_cnt = record.get('call_cnt_day', 0)
    if call_cnt < 50:
        reasons.append(f"Low call volume ({call_cnt} calls)")
    
    # Check for balanced call pattern (R2 needs 0 incoming)
    incl_cnt = record.get('incl_cnt_day', 1)
    if incl_cnt > 0:
        reasons.append("Has incoming calls (not Wangiri pattern)")
    
    # Check student targeting (R4 needs >5 students)
    stu_cnt = record.get('opp_num_stu_cnt', 0)
    if stu_cnt < 6:
        reasons.append(f"Low student targeting ({stu_cnt} students)")
    
    if not reasons:
        reasons.append("Does not match any rule thresholds")
    
    return "; ".join(reasons)




def infer_fraud_type(trigger_reason: str, risk_reason: str) -> str:
    """Infer fraud type from trigger reason"""
    combined = f"{trigger_reason} {risk_reason}".lower()
    
    if "wangiri" in combined or "callback" in combined:
        return "Shopping Refund"
    elif "studenthunter" in combined or "student" in combined:
        return random.choice(["Government Impersonation", "Investment Scam"])
    elif "high_volume" in combined:
        return "Part-time Job Scam"
    elif "roam" in combined:
        return "Government Impersonation"
    else:
        return random.choice(list(FRAUD_TYPES.keys()))


def classify_phone_number(msisdn: str) -> tuple:
    """Classify phone number type and risk level"""
    msisdn = str(msisdn)
    
    if msisdn.startswith("+86") or "86" in msisdn[:5]:
        return "Overseas (Mainland)", "HIGH"
    elif msisdn.startswith("+852 5") or msisdn.startswith("5"):
        return "New Prepaid SIM", "MEDIUM"
    elif msisdn.startswith("+852 6") or msisdn.startswith("6"):
        return "Local Mobile", "LOW"
    elif msisdn.startswith("+1") or msisdn.startswith("+44"):
        return "Overseas Call", "HIGH"
    elif "==" in msisdn:
        return "Known Fraud Number", "CRITICAL"
    else:
        return "Unknown Source", "MEDIUM"


def generate_fake_event():
    """Generate fake event when real data is unavailable"""
    fraud_type = random.choice(list(FRAUD_TYPES.keys()))
    fraud_info = FRAUD_TYPES[fraud_type]
    
    # Select target student based on fraud type
    if "Mainland" in fraud_info["target_profile"]:
        student_type = random.choice(["Mainland Undergraduate", "Mainland Graduate"])
    else:
        student_type = random.choice(["Local Undergraduate", "Local Graduate", "Mainland Undergraduate", "Mainland Graduate", "Exchange Student"])
    
    pattern = random.choice(FRAUD_NUMBER_PATTERNS)
    
    return {
        "timestamp": datetime.now(),
        "event_id": f"EVT{random.randint(10000, 99999)}",
        "fraud_type": fraud_type,
        "fraud_icon": fraud_info["icon"],
        "fraud_desc": fraud_info["description"],
        "caller_number": f"+852 5{random.randint(100,999)} {random.randint(1000,9999)}",
        "number_type": pattern["type"],
        "number_risk": pattern["risk"],
        "target_student": f"STU{random.randint(100000, 999999)}",
        "student_type": student_type,
        "university": random.choice(HK_UNIVERSITIES),
        "age": random.randint(18, 30),
        "gender": random.choice(["Male", "Female"]),
        "call_duration": random.randint(5, 300),
        "risk_score": random.randint(60, 100),
        "risk_tier": random.choice(["CRITICAL", "VULNERABLE", "SAFE"]),
        "potential_loss": fraud_info["avg_loss"] * random.uniform(0.3, 2.0),
        "call_pattern": fraud_info["call_pattern"],
        "risk_level": fraud_info["risk_level"],
        "trigger_reason": "",
        "risk_reason": ""
    }


def render_fraud_scenario_simulator():
    """Render the fraud scenario simulator main page"""
    
    # Page styles
    st.markdown("""
    <style>
        .simulator-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            border-left: 5px solid #ff4b4b;
        }
        .simulator-title {
            color: #ff4b4b;
            font-size: 2rem;
            font-weight: bold;
            margin: 0;
        }
        .simulator-subtitle {
            color: #a0a0a0;
            font-size: 1rem;
            margin-top: 8px;
        }
        .event-card {
            background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
            padding: 15px;
            border-radius: 12px;
            margin: 10px 0;
            border-left: 4px solid;
        }
        .event-critical { border-left-color: #e74c3c; }
        .event-high { border-left-color: #f39c12; }
        .event-medium { border-left-color: #3498db; }
        .event-detected { background: linear-gradient(135deg, #1a3a2e 0%, #2d4444 100%); }
        .stat-card {
            background: #1e1e2f;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: #ffffff;
        }
        .stat-label {
            color: #a0a0a0;
            font-size: 0.85rem;
        }
        .timeline-item {
            padding: 10px 15px;
            margin: 5px 0;
            background: #252540;
            border-radius: 8px;
            font-size: 0.9rem;
        }
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.75rem;
            font-weight: bold;
        }
        .badge-critical { background: #e74c3c; color: white; }
        .badge-high { background: #f39c12; color: white; }
        .badge-medium { background: #3498db; color: white; }
        .badge-low { background: #1abc9c; color: white; }
        .badge-safe { background: #2ecc71; color: white; }
        .badge-success { background: #2ecc71; color: white; }
        .badge-danger { background: #e74c3c; color: white; }
        .event-low { border-left-color: #1abc9c; }
        .event-safe { border-left-color: #2ecc71; }
    </style>
    """, unsafe_allow_html=True)
    
    # Page header
    st.markdown("""
    <div class="simulator-header">
        <p class="simulator-title">🎯 HK Campus Anti-Fraud Detection Demo</p>
        <p class="simulator-subtitle">
            Demonstrating the complete fraud chain: Contact → Detection → Alert → Intervention | 
            Powered by real anonymized data from Hong Kong 2025
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================
    # Load student network data (for realistic student info in synthetic events)
    # ============================================
    result = load_real_data()
    if result[0] is None:
        network_df = None
    else:
        _, _, _, network_df = result
    
    # Data status
    if network_df is not None and len(network_df) > 0:
        st.success(f"📊 Student Network loaded: **{len(network_df):,}** student-fraud connections available for realistic simulation")
    
    # ============================================
    # Hong Kong Fraud Data Background
    # ============================================
    with st.expander("📊 2025 Hong Kong Fraud Data Background", expanded=False):
        st.markdown("""
        ### Hong Kong Telecom Fraud Status (Jan-Aug 2025)
        
        | Metric | Data |
        |--------|------|
        | Total Fraud Cases | **28,379** |
        | Total Losses | **HKD 5.02 Billion** |
        | Student Phone Fraud | **~270 cases** |
        | Local Student Losses | **HKD 32 Million** |
        | Mainland Student Losses | **HKD 75 Million** |
        | Highest Single Case Loss | **Over HKD 10 Million** |
        
        ### Main Fraud Methods
        - 🔴 **Government Impersonation** - Pretending to be mainland police/prosecutors
        - 🟠 **Investment Scams** - Fake investment platforms
        - 🟡 **Shopping Refund Scams** - Impersonating e-commerce customer service
        - 🔵 **Part-time Job Scams** - Fake commission-based jobs
        
        ### Prevention Measures
        - Since December 2024: "New prepaid SIM incoming call voice alert" mechanism
        - Telecom operator data-driven precision anti-fraud system
        """)
    
    # ============================================
    # Simulation Control Panel
    # ============================================
    st.markdown("### ⚙️ Simulation Control Panel")
    
    # Synthetic simulation description
    st.info("**Synthetic Simulation**: Generates random transactions with configurable fraud probability. Measures True Positives (fraud caught), False Negatives (fraud missed), and False Positives (false alarms) using the 6-Rule Engine.")
    
    # Fraud probability slider
    prob_col1, prob_col2 = st.columns([1, 3])
    with prob_col1:
        fraud_probability = st.slider(
            "Fraud Probability",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Probability that each generated transaction is fraudulent"
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        simulation_speed = st.selectbox(
            "Speed",
            ["Slow (3s/event)", "Normal (1.5s/event)", "Fast (0.5s/event)"],
            index=1
        )
        speed_map = {"Slow (3s/event)": 3, "Normal (1.5s/event)": 1.5, "Fast (0.5s/event)": 0.5}
        speed = speed_map[simulation_speed]
    
    with col2:
        num_events = st.number_input("Number of Events", min_value=5, max_value=50, value=15)
    
    with col3:
        target_university = st.selectbox(
            "Target University",
            ["All Universities"] + HK_UNIVERSITIES
        )
    
    with col4:
        focus_fraud_type = st.selectbox(
            "Fraud Type",
            ["All Types"] + list(FRAUD_TYPES.keys())
        )
    
    # Start simulation button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start_simulation = st.button("🚀 Start Real-time Simulation", type="primary", use_container_width=True)
    with col3:
        if st.button("🗑️ Clear History", use_container_width=True):
            if 'simulation_stats' in st.session_state:
                del st.session_state.simulation_stats
            st.rerun()
    
    st.markdown("---")
    
    # ============================================
    # Real-time Statistics Panel
    # ============================================
    st.markdown("### 📈 Real-time Statistics")
    
    # Initialize session state
    if 'simulation_stats' not in st.session_state:
        st.session_state.simulation_stats = {
            'total_events': 0,
            'detected': 0,
            'blocked': 0,
            'missed': 0,
            'total_potential_loss': 0,
            'total_saved': 0,
            'events_history': []
        }
    
    stats = st.session_state.simulation_stats
    
    # Statistics cards
    stat_cols = st.columns(6)
    
    with stat_cols[0]:
        st.metric("📞 Total Events", f"{stats['total_events']}")
    with stat_cols[1]:
        st.metric("🔍 Detected", f"{stats['detected']}", 
                  delta=f"{stats['detected']/max(1,stats['total_events'])*100:.0f}%" if stats['total_events'] > 0 else None)
    with stat_cols[2]:
        st.metric("🛡️ Blocked", f"{stats['blocked']}")
    with stat_cols[3]:
        st.metric("⚠️ Missed", f"{stats['missed']}")
    with stat_cols[4]:
        st.metric("💰 Potential Loss", f"${stats['total_potential_loss']/10000:.1f}K")
    with stat_cols[5]:
        st.metric("✅ Saved", f"${stats['total_saved']/10000:.1f}K", 
                  delta=f"{stats['total_saved']/max(1,stats['total_potential_loss'])*100:.0f}% protected" if stats['total_potential_loss'] > 0 else None)
    
    st.markdown("---")
    
    # ============================================
    # Real-time Event Stream
    # ============================================
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔴 Live Event Stream")
        event_container = st.container()
    
    with col2:
        st.markdown("### 📊 Fraud Type Distribution")
        chart_container = st.empty()
    
    # ============================================
    # Run Simulation
    # ============================================
    if start_simulation:
        # Reset statistics
        st.session_state.simulation_stats = {
            'total_events': 0,
            'detected': 0,
            'blocked': 0,
            'missed': 0,
            'total_potential_loss': 0,
            'total_saved': 0,
            'events_history': [],
            'fraud_type_counts': {},
            'real_data_count': 0,
            # Confusion matrix for synthetic mode
            'true_positives': 0,
            'false_negatives': 0,
            'false_positives': 0,
            'true_negatives': 0
        }
        stats = st.session_state.simulation_stats
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Synthetic simulation only
        status_text.text(f"🧪 Generating synthetic transactions with {fraud_probability:.0%} fraud probability...")
        
        for i in range(num_events):
            # Generate synthetic event using CTGAN-style approach
            event = generate_synthetic_event(fraud_probability, network_df)
            
            # Set intervention and outcome based on detection status AND ground truth
            is_ground_truth_fraud = event.get('is_ground_truth_fraud', True)  # Default True for non-synthetic
            
            if event.get('is_detected', False):
                event["detected"] = True
                source = event.get('detection_source', 'RULE')
                trigger = event.get('trigger_reason', '')[:40]
                event["intervention"] = f"Detected by {source}: {trigger}..." if trigger else f"Detected by {source}"
                
                if is_ground_truth_fraud:
                    event["outcome"] = "Blocked"  # True Positive
                else:
                    event["outcome"] = "False Alarm"  # False Positive - flagged but not actually fraud
                event["saved_amount"] = event["potential_loss"] if is_ground_truth_fraud else 0
            else:
                event["detected"] = False
                miss_reason = event.get('miss_reason', 'Unknown')
                
                if is_ground_truth_fraud:
                    event["intervention"] = f"Not Detected: {miss_reason[:50]}"
                    event["outcome"] = "Fraud Successful"  # False Negative - real fraud missed!
                    event["saved_amount"] = 0
                else:
                    event["intervention"] = "Correctly Ignored (Not Fraud)"
                    event["outcome"] = "True Negative"  # Correctly ignored legitimate transaction
                    event["saved_amount"] = 0
            
            # Apply filter conditions
            if target_university != "All Universities":
                event["university"] = target_university
            if focus_fraud_type != "All Types":
                event["fraud_type"] = focus_fraud_type
                event["fraud_icon"] = FRAUD_TYPES[focus_fraud_type]["icon"]
                event["fraud_desc"] = FRAUD_TYPES[focus_fraud_type]["description"]
            
            # Update statistics
            stats['total_events'] += 1
            stats['total_potential_loss'] += event['potential_loss']
            stats['total_saved'] += event['saved_amount']
            
            if event.get('is_real_data', False):
                stats['real_data_count'] = stats.get('real_data_count', 0) + 1
            
            if event['detected']:
                stats['detected'] += 1
                if event['outcome'] == "Blocked":
                    stats['blocked'] += 1
            else:
                # Only count as "missed" if it was actual fraud (False Negative)
                # True Negatives (correctly ignored legit) should NOT count as missed
                if event.get('is_ground_truth_fraud', True):
                    stats['missed'] += 1
            
            # Update confusion matrix for synthetic mode
            outcome_type = event.get('outcome_type', '')
            if outcome_type == 'TRUE_POSITIVE':
                stats['true_positives'] += 1
            elif outcome_type == 'FALSE_NEGATIVE':
                stats['false_negatives'] += 1
            elif outcome_type == 'FALSE_POSITIVE':
                stats['false_positives'] += 1
            elif outcome_type == 'TRUE_NEGATIVE':
                stats['true_negatives'] += 1
            
            # Record fraud type
            fraud_type = event['fraud_type']
            stats['fraud_type_counts'][fraud_type] = stats['fraud_type_counts'].get(fraud_type, 0) + 1
            
            # Add to history
            stats['events_history'].append(event)
            
            # Update display
            with event_container:
                render_event_card(event)
            
            # Update chart
            with chart_container:
                if stats['fraud_type_counts']:
                    fig = create_fraud_type_pie(stats['fraud_type_counts'])
                    st.plotly_chart(fig, use_container_width=True, key=f"pie_{i}")
            
            # Update progress
            progress_bar.progress((i + 1) / num_events)
            status_text.text(f"Simulating... {i+1}/{num_events} events")
            
            time.sleep(speed)
        
        progress_bar.empty()
        status_text.success(f"✅ Simulation complete! Processed {num_events} events")
        st.rerun()
    
    # ============================================
    # Display History Events
    # ============================================
    if stats['events_history']:
        with event_container:
            for event in reversed(stats['events_history'][-10:]):
                render_event_card(event)
        
        with chart_container:
            if stats.get('fraud_type_counts'):
                fig = create_fraud_type_pie(stats['fraud_type_counts'])
                st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # Detailed Analysis Report
    # ============================================
    if stats['total_events'] > 0:
        st.markdown("---")
        render_simulation_report(stats)


def render_event_card(event):
    """Render a single event card"""
    risk_class = {
        "CRITICAL": "event-critical",
        "HIGH": "event-high",
        "MEDIUM": "event-medium",
        "LOW": "event-low",
        "SAFE": "event-safe"
    }.get(event['risk_level'], "event-medium")
    
    detected_class = "event-detected" if event['detected'] else ""
    
    # Include True Negative and False Alarm in success outcomes
    outcome_badge = "badge-success" if event['outcome'] in ["Blocked", "User Alerted", "User Self-Identified", "True Negative"] else "badge-danger"
    
    # Real data badge
    data_badge = '<span style="background: #9b59b6; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; margin-left: 5px;">Real Data</span>' if event.get('is_real_data', False) else ''
    
    # Risk tier and score
    risk_tier = event.get('risk_tier', '')
    risk_score = event.get('risk_score', 0)
    tier_color = {"CRITICAL": "#e74c3c", "VULNERABLE": "#f39c12", "SAFE": "#2ecc71"}.get(risk_tier, "#a0a0a0")
    
    # Extra info (age, gender)
    age = event.get('age', '')
    gender = event.get('gender', '')
    extra_info = " | " + str(age) + "yo " + str(gender) if age and gender else ""
    
    # Trigger reason
    trigger_info = ""
    if event.get('trigger_reason'):
        trigger_text = str(event["trigger_reason"])[:50]
        trigger_info = '<div style="margin-top: 5px; color: #a0a0a0; font-size: 0.8rem;">🔎 Trigger: ' + trigger_text + '...</div>'
    
    # Pre-calculate all dynamic content
    timestamp_str = event['timestamp'].strftime('%H:%M:%S')
    risk_level_lower = event['risk_level'].lower()
    potential_loss_str = "{:,.0f}".format(event['potential_loss'])
    
    # Detection status
    if event['detected']:
        detection_html = '🔍 <span style="color: #2ecc71;">Detected</span> → ' + str(event["intervention"])
    else:
        detection_html = '⚠️ <span style="color: #e74c3c;">Not Detected</span>'
    
    # Saved amount
    if event['saved_amount'] > 0:
        saved_html = ' | Saved: HKD ' + "{:,.0f}".format(event["saved_amount"])
    else:
        saved_html = ''
    
    # Build complete HTML string using concatenation to avoid f-string issues
    html_parts = []
    html_parts.append('<div class="event-card ' + risk_class + ' ' + detected_class + '">')
    html_parts.append('<div style="display: flex; justify-content: space-between; align-items: center;">')
    html_parts.append('<div>')
    html_parts.append('<span style="font-size: 1.5rem;">' + event['fraud_icon'] + '</span>')
    html_parts.append('<strong style="color: #ffffff; margin-left: 10px;">' + event['fraud_type'] + '</strong>')
    html_parts.append('<span class="badge badge-' + risk_level_lower + '" style="margin-left: 10px;">' + event['risk_level'] + '</span>')
    html_parts.append(data_badge)
    html_parts.append('</div>')
    html_parts.append('<div>')
    html_parts.append('<span style="color: ' + tier_color + '; font-weight: bold;">' + risk_tier + '</span>')
    html_parts.append('<span style="color: #a0a0a0; font-size: 0.8rem; margin-left: 10px;">' + timestamp_str + '</span>')
    html_parts.append('</div>')
    html_parts.append('</div>')
    html_parts.append('<div style="margin-top: 10px; color: #c0c0c0; font-size: 0.9rem;">')
    html_parts.append('<div>📞 Caller: <strong>' + str(event['caller_number']) + '</strong> (' + event['number_type'] + ')</div>')
    html_parts.append('<div>🎓 Target: ' + str(event['target_student']) + ' | ' + event['student_type'] + extra_info + ' | ' + event['university'] + '</div>')
    html_parts.append('<div>💰 Potential Loss: <strong style="color: #e74c3c;">HKD ' + potential_loss_str + '</strong> | Risk Score: <strong style="color: ' + tier_color + ';">' + str(risk_score) + '</strong></div>')
    html_parts.append(trigger_info)
    html_parts.append('</div>')
    html_parts.append('<div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #3d3d5c;">')
    html_parts.append('<div style="display: flex; justify-content: space-between;">')
    html_parts.append('<div>' + detection_html + '</div>')
    html_parts.append('<div><span class="badge ' + outcome_badge + '">' + event['outcome'] + '</span>' + saved_html + '</div>')
    html_parts.append('</div>')
    html_parts.append('</div>')
    html_parts.append('</div>')
    
    html_content = ''.join(html_parts)
    st.markdown(html_content, unsafe_allow_html=True)


def create_fraud_type_pie(fraud_type_counts):
    """Create fraud type pie chart"""
    labels = list(fraud_type_counts.keys())
    values = list(fraud_type_counts.values())
    
    # Add icons for each type (handle unknown types gracefully)
    labels_with_icons = []
    for t in labels:
        if t in FRAUD_TYPES:
            labels_with_icons.append(FRAUD_TYPES[t]['icon'] + " " + t)
        elif t == "Legitimate Transaction":
            labels_with_icons.append("✅ " + t)
        else:
            labels_with_icons.append("📊 " + t)  # Fallback icon
    
    fig = go.Figure(data=[go.Pie(
        labels=labels_with_icons,
        values=values,
        hole=0.4,
        marker_colors=['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6', '#1abc9c']
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(30,30,47,1)',
        font=dict(color='white', size=10),
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.05,
            font=dict(size=9)
        ),
        margin=dict(l=10, r=100, t=10, b=10),
        height=250
    )
    
    return fig


def render_simulation_report(stats):
    """Render simulation report"""
    st.markdown("### 📋 Simulation Analysis Report")
    
    # Data source statistics
    real_count = stats.get('real_data_count', 0)
    total = stats['total_events']
    if real_count > 0:
        st.info(f"📊 This simulation used **{real_count}/{total}** real anonymized data records ({real_count/total*100:.0f}%)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Detection Effectiveness Analysis")
        
        detection_rate = stats['detected'] / max(1, stats['total_events']) * 100
        block_rate = stats['blocked'] / max(1, stats['detected']) * 100 if stats['detected'] > 0 else 0
        save_rate = stats['total_saved'] / max(1, stats['total_potential_loss']) * 100
        
        # Detection rate gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=detection_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "System Detection Rate", 'font': {'size': 16, 'color': 'white'}},
            number={'suffix': '%', 'font': {'size': 36, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white'},
                'bar': {'color': '#2ecc71'},
                'bgcolor': 'rgba(50,50,50,0.5)',
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(231, 76, 60, 0.3)'},
                    {'range': [50, 75], 'color': 'rgba(241, 196, 15, 0.3)'},
                    {'range': [75, 100], 'color': 'rgba(46, 204, 113, 0.3)'}
                ]
            }
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(30,30,47,1)',
            height=200,
            margin=dict(l=30, r=30, t=50, b=30)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Detection Rate | **{detection_rate:.1f}%** |
        | Block Rate | **{block_rate:.1f}%** |
        | Fund Protection Rate | **{save_rate:.1f}%** |
        | Potential Loss | **HKD {stats['total_potential_loss']:,.0f}** |
        | Amount Saved | **HKD {stats['total_saved']:,.0f}** |
        """)
    
    with col2:
        st.markdown("#### 📊 Event Timeline")
        
        if stats['events_history']:
            # Create timeline data
            timeline_data = []
            for event in stats['events_history']:
                timeline_data.append({
                    'time': event['timestamp'],
                    'type': event['fraud_type'],
                    'detected': 1 if event['detected'] else 0,
                    'loss': event['potential_loss'],
                    'saved': event['saved_amount']
                })
            
            df = pd.DataFrame(timeline_data)
            
            # Cumulative chart
            df['cumulative_loss'] = df['loss'].cumsum()
            df['cumulative_saved'] = df['saved'].cumsum()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(len(df))),
                y=df['cumulative_loss'],
                name='Cumulative Potential Loss',
                line=dict(color='#e74c3c', width=2),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.2)'
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(len(df))),
                y=df['cumulative_saved'],
                name='Cumulative Saved',
                line=dict(color='#2ecc71', width=2),
                fill='tozeroy',
                fillcolor='rgba(46, 204, 113, 0.2)'
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(30,30,47,1)',
                plot_bgcolor='rgba(30,30,47,1)',
                font=dict(color='white'),
                legend=dict(orientation='h', y=1.1),
                margin=dict(l=50, r=20, t=30, b=50),
                height=250,
                xaxis=dict(title='Event #', showgrid=False),
                yaxis=dict(title='Amount (HKD)', showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("---")
    st.markdown("#### 💡 System Optimization Recommendations")
    
    suggestions = []
    if detection_rate < 70:
        suggestions.append("⚠️ Detection rate below 70%, recommend enhancing feature engineering and model training")
    if stats['missed'] > stats['detected'] * 0.5:
        suggestions.append("⚠️ High miss rate, recommend lowering detection threshold or adding rules")
    if save_rate < 60:
        suggestions.append("⚠️ Fund protection rate below 60%, recommend optimizing intervention response speed")
    
    if not suggestions:
        suggestions.append("✅ System running well, all metrics within normal range")
    
    for s in suggestions:
        st.markdown(f"- {s}")
    
    # FALSE NEGATIVE ANALYSIS - Key for model improvement
    missed_events = [e for e in stats['events_history'] if not e.get('detected', True)]
    
    if missed_events:
        st.markdown("---")
        st.markdown("### 🔍 False Negative Analysis (Missed Frauds)")
        st.warning(f"**{len(missed_events)} fraud events were NOT detected by the system.** Analyzing patterns for model improvement...")
        
        # Group by miss reason
        miss_reasons = {}
        for event in missed_events:
            reason = event.get('miss_reason', 'Unknown')
            if reason not in miss_reasons:
                miss_reasons[reason] = []
            miss_reasons[reason].append(event)
        
        # Display analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Why These Frauds Were Missed")
            
            for reason, events in miss_reasons.items():
                st.markdown(f"**{reason}** — {len(events)} case(s)")
                
            # Show sample missed events
            st.markdown("#### Sample Missed Events")
            for event in missed_events[:3]:
                st.markdown(f"""
                - **{event['fraud_type']}** | Number: `{event['caller_number']}` | 
                  Loss: HKD {event['potential_loss']:,.0f} | 
                  Reason: _{event.get('miss_reason', 'Unknown')}_
                """)
        
        with col2:
            st.markdown("#### 🛠️ Model Improvement Actions")
            st.markdown("""
            Based on false negative analysis:
            
            1. **Add new rules** for patterns not covered
            2. **Lower thresholds** if legitimate-looking numbers are used
            3. **Cross-reference** with external threat intelligence
            4. **Retrain ML model** with missed cases as positive samples
            """)
            
            # Calculate missed amount
            missed_amount = sum(e['potential_loss'] for e in missed_events)
            st.metric("💸 Total Missed Loss", f"HKD {missed_amount:,.0f}")


# Export function
__all__ = ['render_fraud_scenario_simulator']
