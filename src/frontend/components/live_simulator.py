"""
Live Risk Simulator Component

Interactive tool to explore how different parameters affect risk scores.
Allows users to manually adjust student characteristics and see real-time risk predictions.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Tuple, List


def calculate_simulated_risk(params: Dict) -> Tuple[int, List[str], Dict[str, int]]:
    """
    Calculate risk score based on simulated parameters.
    Uses the same logic as RiskTriangleScorer but with user inputs.
    
    Args:
        params: Dictionary of user-input parameters
        
    Returns:
        Tuple of (risk_score, reasons, sub_scores)
    """
    score = 0
    reasons = []
    
    # ==================================================
    # LAYER 1: IDENTITY VULNERABILITY (0-100 SCALE)
    # ==================================================
    # Gradient scale based on age - younger = more vulnerable
    # Matches RiskTriangleScorer: 18=100, 19=90, 20=80, 21=70, 22=60, 23=40, 24=20, 25+=0
    age = params.get('age', 25)
    
    if age <= 18:
        identity_score_100 = 100
        identity_points = 5
        reasons.append(f"Age {age}: vulnerability score 100 (youngest students most targeted) (+5)")
    elif age == 19:
        identity_score_100 = 90
        identity_points = 5
        reasons.append(f"Age {age}: vulnerability score 90 (very young student) (+5)")
    elif age == 20:
        identity_score_100 = 80
        identity_points = 5
        reasons.append(f"Age {age}: vulnerability score 80 (young student profile) (+5)")
    elif age == 21:
        identity_score_100 = 70
        identity_points = 4
        reasons.append(f"Age {age}: vulnerability score 70 (moderate youth risk) (+4)")
    elif age == 22:
        identity_score_100 = 60
        identity_points = 4
        reasons.append(f"Age {age}: vulnerability score 60 (slight youth risk) (+4)")
    elif age == 23:
        identity_score_100 = 40
        identity_points = 2
        reasons.append(f"Age {age}: vulnerability score 40 (reduced risk) (+2)")
    elif age == 24:
        identity_score_100 = 20
        identity_points = 1
        reasons.append(f"Age {age}: vulnerability score 20 (minimal risk) (+1)")
    else:
        identity_score_100 = 0
        identity_points = 0
    
    score += identity_points
    
    # ==================================================
    # LAYER 2: THREAT EXPOSURE (0-100 SCALE)
    # ==================================================
    exposure_points = 0
    
    # Calls from Mainland China operators
    mainland_calls = params.get('mainland_calls', 0)
    if mainland_calls >= 5:
        exposure_points += 10
        reasons.append(f"High Mainland call volume ({mainland_calls}); pattern matches telecom scam outreach (+10)")
    elif mainland_calls >= 1:
        exposure_points += 5
        reasons.append(f"Some Mainland calls ({mainland_calls}); early exposure to cross-border scam traffic (+5)")
    
    # Unknown overseas calls
    overseas_calls = params.get('overseas_calls', 0)
    if overseas_calls >= 10:
        exposure_points += 10
        reasons.append(f"High overseas unknown calls ({overseas_calls}); often used for spoofed authority threats (+10)")
    elif overseas_calls >= 5:
        exposure_points += 5
        reasons.append(f"Overseas unknown calls ({overseas_calls}); potential phishing probes (+5)")
    
    # Contacted by known fraud MSISDN
    has_fraud_contact = params.get('has_fraud_contact', False)
    if has_fraud_contact:
        exposure_points += 25
        reasons.append("Contact from confirmed fraud number; exposure is high even without engagement (+25)")
    
    score += exposure_points
    exposure_score_100 = min(100, (exposure_points / 45) * 100 if exposure_points else 0)
    
    # ==================================================
    # LAYER 3: RISKY BEHAVIOR (0-100 SCALE)
    # ==================================================
    behavior_multiplier = 1.0
    behavior_reasons = []
    
    callback_count = params.get('callback_count', 0)
    answered_count = params.get('answered_count', 0)
    sms_reply_count = params.get('sms_reply_count', 0)
    
    if callback_count > 0:
        behavior_score_100 = 100
        behavior_multiplier = 2.0
        behavior_reasons.append(f"Called back suspected fraud number {callback_count}x; active engagement doubles risk")
    elif answered_count > 0:
        behavior_score_100 = 60
        behavior_multiplier = 1.5
        behavior_reasons.append(f"Answered suspected fraud calls {answered_count}x; indicates partial engagement")
    elif sms_reply_count > 0:
        behavior_score_100 = 50
        behavior_multiplier = 1.5
        behavior_reasons.append(f"Replied to suspected fraud messages {sms_reply_count}x; engagement increases exposure")
    else:
        behavior_score_100 = 10
    
    # Apply multiplier
    if behavior_multiplier > 1.0:
        score = int(score * behavior_multiplier)
        reasons.extend(behavior_reasons)
        reasons.append(f"Behavior Multiplier: x{behavior_multiplier}")
    
    # Clamp to 0-100
    score = min(100, score)
    
    sub_scores = {
        'identity_score': identity_score_100,
        'exposure_score': exposure_score_100,
        'behavior_score': behavior_score_100
    }
    
    return score, reasons, sub_scores


def get_risk_tier(params: dict, score: int) -> str:
    """
    Determine risk tier based on BEHAVIOR (not score).
    
    Logic from risk_triangle_scorer.py:
    - CRITICAL: Active engagement (answered OR called back fraud numbers)
    - VULNERABLE: High exposure (score >= 20) OR fraud contact, but no engagement
    - SAFE: Everyone else
    """
    # Check for active engagement (CRITICAL trigger)
    callback_count = params.get('callback_count', 0)
    answered_count = params.get('answered_count', 0)
    
    # CRITICAL: Any meaningful engagement
    if (callback_count + answered_count) >= 1:
        return "CRITICAL"
    
    # VULNERABLE: High exposure or fraud contact without engagement
    has_fraud_contact = params.get('has_fraud_contact', False)
    VULNERABLE_THRESHOLD = 20
    
    if score >= VULNERABLE_THRESHOLD or has_fraud_contact:
        return "VULNERABLE"
    
    return "SAFE"


def render_live_simulator():
    """Render the Live Risk Simulator page."""
    st.title("🎮 Live Risk Simulator")
    st.markdown("""
    **Interactive tool to explore how different factors affect fraud risk scores.**
    
    Adjust the parameters below and watch the risk score change in real-time.
    This helps understand which factors contribute most to high-risk classifications.
    """)
    
    st.markdown("---")
    
    # Create two columns: inputs on left, results on right
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.subheader("📝 Input Parameters")
        
        # LAYER 1: Identity
        st.markdown("##### 👤 Layer 1: Identity Vulnerability")
        age = st.slider(
            "Age",
            min_value=16,
            max_value=40,
            value=20,
            help="Students aged 18-22 are at higher risk"
        )
        
        st.markdown("---")
        
        # LAYER 2: Threat Exposure
        st.markdown("##### 🌍 Layer 2: Threat Exposure")
        
        mainland_calls = st.slider(
            "Mainland China Calls Received",
            min_value=0,
            max_value=50,
            value=0,
            help="Number of calls from mainland operators (5+ = high risk)"
        )
        
        overseas_calls = st.slider(
            "Unknown Overseas Calls",
            min_value=0,
            max_value=50,
            value=0,
            help="Number of calls from unknown overseas numbers (10+ = high risk)"
        )
        
        has_fraud_contact = st.checkbox(
            "📱 Contacted by Known Fraud Number",
            value=False,
            help="Has received calls/messages from confirmed fraud MSISDNs (+25 points)"
        )
        
        st.markdown("---")
        
        # LAYER 3: Risky Behavior
        st.markdown("##### ⚠️ Layer 3: Risky Behavior")
        
        callback_count = st.slider(
            "Callback to Fraud Numbers",
            min_value=0,
            max_value=10,
            value=0,
            help="Times called back suspected fraud numbers (triggers 2x multiplier!)"
        )
        
        answered_count = st.slider(
            "Answered Fraud Calls",
            min_value=0,
            max_value=10,
            value=0,
            help="Times answered suspected fraud calls (triggers 1.5x multiplier)"
        )
        
        sms_reply_count = st.slider(
            "SMS Replies to Fraud",
            min_value=0,
            max_value=10,
            value=0,
            help="Times replied to suspected fraud SMS (triggers 1.5x multiplier)"
        )
    
    # Calculate risk based on inputs
    params = {
        'age': age,
        'mainland_calls': mainland_calls,
        'overseas_calls': overseas_calls,
        'has_fraud_contact': has_fraud_contact,
        'callback_count': callback_count,
        'answered_count': answered_count,
        'sms_reply_count': sms_reply_count
    }
    
    risk_score, reasons, sub_scores = calculate_simulated_risk(params)
    risk_tier = get_risk_tier(params, risk_score)  # Pass params for behavior check
    
    with col_result:
        st.subheader("📊 Risk Assessment Result")
        
        # Risk Tier Badge
        tier_colors = {'CRITICAL': '🔴', 'VULNERABLE': '🟡', 'SAFE': '🟢'}
        tier_color_css = {'CRITICAL': 'red', 'VULNERABLE': 'orange', 'SAFE': 'green'}
        
        st.markdown(f"### {tier_colors.get(risk_tier, '⚪')} Risk Tier: :{tier_color_css.get(risk_tier, 'gray')}[{risk_tier}]")
        
        # Main score metric
        st.metric(
            label="Total Risk Score",
            value=f"{risk_score}/100",
            delta=None
        )
        
        st.markdown("---")
        
        # Sub-scores with progress bars
        st.markdown("##### Sub-Scores (0-100)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Identity", sub_scores['identity_score'])
            st.progress(sub_scores['identity_score'] / 100)
        with col2:
            st.metric("Exposure", int(sub_scores['exposure_score']))
            st.progress(sub_scores['exposure_score'] / 100)
        with col3:
            st.metric("Behavior", sub_scores['behavior_score'])
            st.progress(sub_scores['behavior_score'] / 100)
        
        st.markdown("---")
        
        # Radar Chart
        st.markdown("##### Risk Radar")
        try:
            from components.risk_radar import create_risk_radar
            fig = create_risk_radar(
                sub_scores['identity_score'],
                int(sub_scores['exposure_score']),
                sub_scores['behavior_score']
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Radar chart component not available")
        
        st.markdown("---")
        
        # Risk Factors
        st.markdown("##### Risk Factors Triggered")
        if reasons:
            for reason in reasons:
                st.write(f"• {reason}")
        else:
            st.write("• No significant risk factors detected")
    
    # Educational Section
    st.markdown("---")
    st.subheader("📚 How Risk Scoring Works")
    
    with st.expander("Click to learn about the Risk Triangle Model"):
        st.markdown("""
        ### Risk Triangle Model
        
        The risk score is calculated using a **three-layer assessment**:
        
        | Layer | Name | Description | Max Points |
        |-------|------|-------------|------------|
        | 1️⃣ | **Identity Vulnerability** | Age-based risk (18-22 = higher risk) | +5 |
        | 2️⃣ | **Threat Exposure** | Calls from mainland, overseas, fraud contacts | +45 |
        | 3️⃣ | **Risky Behavior** | Callback, answered, SMS replies to fraud | **x2 multiplier** |
        
        ### ⚠️ Risk Tier Classification (NOT based on score!)
        
        | Risk Tier | Trigger Condition | Meaning |
        |-----------|-------------------|---------|
        | 🔴 CRITICAL | **Answered OR Called back** fraud number (≥1 time) | Active engagement - highest priority |
        | 🟡 VULNERABLE | Score ≥ 20 OR contacted by fraud number (no engagement) | High exposure, needs monitoring |
        | 🟢 SAFE | No engagement AND low exposure | Low risk |
        
        ### Key Insight
        **Risk Tier is determined by BEHAVIOR, not by score!** 
        
        A student with risk score as low as **1** can still be **CRITICAL** if they answered or called back a fraud number.
        This is because active engagement is the most dangerous indicator - it means the student is already interacting with scammers.
        
        **Score does NOT determine tier. Behavior does.**
        """)
    
    # Preset Scenarios
    st.markdown("---")
    st.subheader("🎯 Try These Scenarios")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Scenario 1: Safe Student**")
        st.markdown("""
        - Age: 25
        - Mainland calls: 0
        - Overseas calls: 2
        - No fraud contact
        - No callbacks/answered
        """)
        st.info("Expected: 🟢 SAFE")
    
    with col2:
        st.markdown("**Scenario 2: High Exposure**")
        st.markdown("""
        - Age: 20
        - Mainland calls: 10
        - Overseas calls: 15
        - ✅ Fraud contact
        - **No callbacks/answered**
        """)
        st.warning("Expected: 🟡 VULNERABLE")
    
    with col3:
        st.markdown("**Scenario 3: Low Score but CRITICAL**")
        st.markdown("""
        - Age: 25 (not vulnerable age)
        - Mainland calls: 0
        - Overseas calls: 0
        - No fraud contact
        - ✅ **Answered 1 call**
        """)
        st.error("Expected: 🔴 CRITICAL (score can be as low as 1!)")


if __name__ == "__main__":
    # For testing standalone
    render_live_simulator()
