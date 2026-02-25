"""
SHAP Explainer Component for Frontend

Displays feature contributions for student risk scoring.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def create_shap_bar_chart(contributions: dict, title: str = "Risk Factor Contributions"):
    """
    Create a horizontal bar chart showing feature contributions to risk score.
    
    Args:
        contributions: Dict of {feature_name: contribution_value}
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not contributions:
        return None
    
    # Sort by absolute value (most impactful first)
    sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    features = [item[0] for item in sorted_items[:10]]  # Top 10
    values = [item[1] for item in sorted_items[:10]]
    
    # Color based on positive/negative contribution
    colors = ['#ef4444' if v > 0 else '#22c55e' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"+{v:.1f}" if v > 0 else f"{v:.1f}" for v in values],
        textposition='outside',
        textfont=dict(size=12)
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Contribution to Risk Score",
        yaxis_title="",
        height=350,
        margin=dict(l=20, r=20, t=50, b=30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.3)',
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            autorange="reversed",
            gridcolor='rgba(255,255,255,0.1)'
        )
    )
    
    return fig


def calculate_rule_contributions(student: pd.Series) -> dict:
    """
    Calculate feature contributions based on the student's scoring rules.
    This simulates SHAP-like explanations using the actual scoring logic.
    
    Args:
        student: Series containing student data
        
    Returns:
        Dict of feature contributions
    """
    contributions = {}
    
    # Age contribution
    age = student.get('age', 25)
    if pd.notna(age):
        age = int(age)
        if age <= 18:
            contributions['Age (≤18)'] = 5
        elif age == 19:
            contributions['Age (19)'] = 5
        elif age == 20:
            contributions['Age (20)'] = 5
        elif age == 21:
            contributions['Age (21)'] = 4
        elif age == 22:
            contributions['Age (22)'] = 4
        elif age == 23:
            contributions['Age (23)'] = 2
        elif age == 24:
            contributions['Age (24)'] = 1
    
    # Mainland calls
    mainland = student.get('from_china_mobile_call_cnt', 0) or 0
    if mainland >= 5:
        contributions['Mainland Calls (5+)'] = 5
    elif mainland >= 1:
        contributions['Mainland Calls (1-4)'] = 2
    
    # Overseas calls
    overseas = student.get('total_voice_cnt', 0) or 0
    if overseas >= 10:
        contributions['Overseas Calls (10+)'] = 15
    elif overseas >= 5:
        contributions['Overseas Calls (5-9)'] = 5
    
    # Fraud contact
    fraud_msisdn = student.get('fraud_msisdn')
    if pd.notna(fraud_msisdn) and str(fraud_msisdn).strip():
        contributions['Fraud Contact'] = 20
    
    # Behavior multipliers
    voice_call = student.get('voice_call', 0) or 0
    voice_receive = student.get('voice_receive', 0) or 0
    msg_call = student.get('msg_call', 0) or 0
    
    if voice_call > 0:
        contributions['Callback (x2.0)'] = contributions.get('total', 0) * 1.0  # The multiplier effect
        contributions['Called Back Fraud'] = 10
    elif voice_receive > 0:
        contributions['Answered Fraud Call'] = 5
    elif msg_call > 0:
        contributions['SMS Reply'] = 3
    
    # msg_receive
    msg_receive = student.get('msg_receive', 0) or 0
    if msg_receive > 0:
        contributions['SMS from Fraud'] = 5
    
    return contributions


def render_shap_explanation(student: pd.Series):
    """
    Render SHAP-like explanation for a student's risk score.
    
    Args:
        student: Series containing student data
    """
    st.subheader("📊 Risk Score Breakdown")
    st.markdown("*Why this student received their risk score*")
    
    # Calculate contributions
    contributions = calculate_rule_contributions(student)
    
    if not contributions:
        st.info("No significant risk factors identified for this student.")
        return
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bar chart
        fig = create_shap_bar_chart(contributions, "Feature Contributions")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Summary stats
        total_positive = sum(v for v in contributions.values() if v > 0)
        
        st.markdown("### Contribution Summary")
        st.metric("Base Score", "0")
        st.metric("Added Points", f"+{total_positive:.0f}")
        st.metric("Final Score", f"{int(student.get('risk_score', 0))}")
        
        # Top factor
        if contributions:
            top_factor = max(contributions.items(), key=lambda x: x[1])
            st.markdown(f"**Top Factor:** {top_factor[0]}")
    
    # Explanation text
    st.markdown("---")
    st.markdown("**How to read this chart:**")
    st.markdown("""
    - 🔴 **Red bars** = Factors that **increase** risk
    - 🟢 **Green bars** = Factors that **decrease** risk (if any)
    - Longer bars = More significant impact on the final score
    """)
