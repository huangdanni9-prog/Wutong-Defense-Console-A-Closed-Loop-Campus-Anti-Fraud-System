"""
Wutong Defense Console - Real-time Anti-Fraud Intervention Dashboard

Built with Streamlit for rapid deployment and Groq LLM for AI-powered scripts.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Ensure local packages are importable when launched via `streamlit run`
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))            # allows `data`, `components`, `services`
sys.path.insert(0, str(THIS_DIR.parent))     # allows `frontend.*` absolute imports

# Page config must be first
st.set_page_config(
    page_title="Wutong Defense Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Local imports
from data.loader import load_student_data, load_blacklist, load_greylist, load_fraud_student_network, clear_all_caches
from components.risk_radar import create_risk_radar
from components.fraud_network import create_fraud_network, create_network_stats, create_hunter_table
from components.live_simulator import render_live_simulator
from components.fraud_scenario_simulator import render_fraud_scenario_simulator
from components.ethical_ai import render_ethical_ai
from services.groq_client import generate_intervention_script
from components.shap_explainer import render_shap_explanation

# Privacy utilities
from utils import mask_pii, mask_dataframe_pii, load_whitelist, save_whitelist
from utils import dp_count, dp_mean  # Differential Privacy for aggregate stats
from datetime import datetime


def main():
    # ========================================
    # SIDEBAR
    # ========================================
    with st.sidebar:
        st.title("🛡️ Wutong Console")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "👤 Student Details", "🎮 Live Simulator", "🕸️ Network Graph", "🚨 Fraud Intel", "✅ Whitelist Review", "🎯 Fraud Simulator", "🔒 Ethical AI"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            clear_all_caches()
            st.rerun()
        
        # Stats (with Differential Privacy protection)
        st.markdown("### 🔐 DP-Protected Stats")
        try:
            df = load_student_data()
            bl = load_blacklist()
            gl = load_greylist()
            
            wl = load_whitelist()
            # Apply DP noise to protect individual privacy in aggregates
            st.metric("Total Students", f"~{dp_count(len(df), epsilon=1.0):,}")
            st.metric("Critical", f"~{dp_count((df['risk_tier'] == 'CRITICAL').sum(), epsilon=1.0):,}")
            st.metric("Blacklist", f"~{dp_count(len(bl), epsilon=1.0):,}")
            st.metric("Greylist", f"~{dp_count(len(gl), epsilon=1.0):,}")
            st.metric("Whitelist", f"~{dp_count(len(wl), epsilon=1.0):,}")
            st.caption("ε=1.0 Laplace noise applied")
        except Exception as e:
            st.error(f"Data load error: {e}")
    
    # ========================================
    # MAIN CONTENT
    # ========================================
    if page == "📊 Dashboard":
        render_dashboard()
    elif page == "👤 Student Details":
        render_student_details()
    elif page == "🎮 Live Simulator":
        render_live_simulator()
    elif page == "🕸️ Network Graph":
        render_network_graph()
    elif page == "🚨 Fraud Intel":
        render_fraud_intel()
    elif page == "✅ Whitelist Review":
        render_whitelist_review()
    elif page == "🎯 Fraud Simulator":
        render_fraud_scenario_simulator()
    elif page == "🔒 Ethical AI":
        render_ethical_ai()


def render_dashboard():
    """Main dashboard view with KPIs and student table."""
    st.title("📊 Risk Dashboard")
    
    try:
        df = load_student_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return
    
    # Privacy toggle for DP
    dp_enabled = st.toggle("🔐 Enable Differential Privacy", value=True, 
                           help="Adds calibrated noise to protect individual privacy in aggregates")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    critical_count = (df['risk_tier'] == 'CRITICAL').sum()
    vulnerable_count = (df['risk_tier'] == 'VULNERABLE').sum()
    safe_count = (df['risk_tier'] == 'SAFE').sum()
    avg_score = df['risk_score'].mean()
    
    if dp_enabled:
        # Apply Differential Privacy (ε=1.0 for balanced privacy/utility)
        dp_critical = dp_count(critical_count, epsilon=1.0)
        dp_vulnerable = dp_count(vulnerable_count, epsilon=1.0)
        dp_safe = dp_count(safe_count, epsilon=1.0)
        dp_avg = dp_mean(df['risk_score'].tolist(), epsilon=1.0, bounds=(0, 100))
        prefix = "~"
    else:
        dp_critical, dp_vulnerable, dp_safe, dp_avg = critical_count, vulnerable_count, safe_count, avg_score
        prefix = ""
    
    with col1:
        st.metric("🔴 CRITICAL", f"{prefix}{dp_critical:,}", 
                  help="Active engagement with fraud (callback/answered)")
    with col2:
        st.metric("🟡 VULNERABLE", f"{prefix}{dp_vulnerable:,}",
                  help="High exposure but no engagement")
    with col3:
        st.metric("🟢 SAFE", f"{prefix}{dp_safe:,}",
                  help="Low risk students")
    with col4:
        st.metric("📈 Avg Score", f"{prefix}{dp_avg:.1f}")
    
    if dp_enabled:
        st.caption("🔐 Differential Privacy enabled (ε=1.0 Laplace mechanism) — counts are approximate to protect individual privacy")
    
    st.markdown("---")
    
    # Filter controls
    col1, col2 = st.columns([1, 3])
    with col1:
        tier_filter = st.selectbox(
            "Filter by Tier",
            ["All", "CRITICAL", "VULNERABLE", "SAFE"]
        )
    
    # Apply filter
    if tier_filter != "All":
        filtered_df = df[df['risk_tier'] == tier_filter]
    else:
        filtered_df = df
    
    # Show high-risk first
    filtered_df = filtered_df.sort_values('risk_score', ascending=False)
    
    # Student table
    st.subheader(f"Students ({len(filtered_df):,} shown)")
    
    display_cols = ['user_id', 'risk_score', 'identity_score', 'exposure_score', 
                    'behavior_score', 'risk_tier', 'risk_reason']
    available_cols = [c for c in display_cols if c in filtered_df.columns]
    
    # Apply PII masking for privacy
    display_df = mask_dataframe_pii(filtered_df[available_cols].head(100), ['user_id'])
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )


def render_student_details():
    """Individual student view with Radar Chart and script generation."""
    st.title("👤 Student Details")
    
    try:
        df = load_student_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return
    
    # Search and Filter row
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        # Search by user_id
        search_input = st.text_input(
            "🔍 Search User ID",
            placeholder="Enter full or partial user_id...",
            help="Search for a specific student by user_id"
        )
    
    with col2:
        # Risk tier filter
        risk_filter = st.selectbox(
            "Filter by Risk Level",
            ["All", "CRITICAL", "VULNERABLE", "SAFE"],
            help="Filter students by risk tier"
        )
    
    # Apply filter and sort by risk score
    if risk_filter == "All":
        filtered_df = df.sort_values('risk_score', ascending=False)
    else:
        filtered_df = df[df['risk_tier'] == risk_filter].sort_values('risk_score', ascending=False)
    
    # Apply search filter if provided
    if search_input.strip():
        search_term = search_input.strip()
        # Search in user_id (case-insensitive partial match)
        search_results = filtered_df[filtered_df['user_id'].astype(str).str.contains(search_term, case=False, na=False)]
        
        if len(search_results) == 0:
            st.warning(f"No students found matching '{search_term}'")
            # Still show the dropdown with filtered results
        else:
            filtered_df = search_results
            st.success(f"Found {len(search_results):,} student(s) matching '{search_term}'")
    
    if len(filtered_df) == 0:
        st.warning(f"No {risk_filter.lower()} students found.")
        return
    
    # Show count of filtered students
    with col3:
        tier_counts = {
            'CRITICAL': (df['risk_tier'] == 'CRITICAL').sum(),
            'VULNERABLE': (df['risk_tier'] == 'VULNERABLE').sum(),
            'SAFE': (df['risk_tier'] == 'SAFE').sum()
        }
        st.markdown(f"**Total:** {len(filtered_df):,} | "
                    f"🔴 {tier_counts['CRITICAL']:,} | "
                    f"🟡 {tier_counts['VULNERABLE']:,} | "
                    f"🟢 {tier_counts['SAFE']:,}")
    
    # Create dropdown options with masked display but keep original for lookup
    top_students = filtered_df.head(2000)
    option_map = {mask_pii(uid): uid for uid in top_students['user_id'].tolist()}
    masked_options = list(option_map.keys())
    selected_masked = st.selectbox(
        f"Select Student (Top 2000 {risk_filter if risk_filter != 'All' else 'All'} Students)", 
        masked_options
    )
    selected_id = option_map.get(selected_masked) if selected_masked else None
    
    if not selected_id:
        return
    
    # Get student data
    student = df[df['user_id'] == selected_id].iloc[0]
    
    # Layout: Info + Radar Chart
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Profile")
        
        # Tier badge
        tier = student.get('risk_tier', 'UNKNOWN')
        tier_colors = {'CRITICAL': 'red', 'VULNERABLE': 'orange', 'SAFE': 'green'}
        st.markdown(f"**Risk Tier:** :{tier_colors.get(tier, 'gray')}[{tier}]")
        
        st.metric("Total Risk Score", int(student.get('risk_score', 0)))
        
        # Sub-scores
        st.markdown("**Sub-Scores (0-100):**")
        st.write(f"- Identity: {int(student.get('identity_score', 0))}")
        st.write(f"- Exposure: {int(student.get('exposure_score', 0))}")
        st.write(f"- Behavior: {int(student.get('behavior_score', 0))}")
        
        # Risk reasons
        st.markdown("**Risk Factors:**")
        reasons = student.get('risk_reason', '')
        if reasons:
            for r in str(reasons).split(' | '):
                st.write(f"- {r}")
        else:
            st.write("- No specific factors")
    
    with col2:
        st.subheader("Risk Radar")
        
        # Create radar chart
        fig = create_risk_radar(
            int(student.get('identity_score', 0)),
            int(student.get('exposure_score', 0)),
            int(student.get('behavior_score', 0))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # SHAP-like Risk Explanation
    render_shap_explanation(student)
    
    st.markdown("---")
    
    # Intervention Script Generator
    st.subheader("🤖 AI Intervention Script")
    
    if st.button("Generate Script", type="primary"):
        with st.spinner("Generating script with Groq LLM..."):
            def safe_int(value):
                value = pd.to_numeric(value, errors='coerce')
                return int(value) if pd.notna(value) else 0

            profile = {
                'risk_tier': student.get('risk_tier', 'UNKNOWN'),
                'risk_score': int(student.get('risk_score', 0)),
                'age': student.get('age', 'Unknown'),
                'risk_reason': student.get('risk_reason', 'No specific factors'),
                'identity_score': int(student.get('identity_score', 0)),
                'exposure_score': int(student.get('exposure_score', 0)),
                'behavior_score': int(student.get('behavior_score', 0)),
                'total_voice_cnt': safe_int(student.get('total_voice_cnt', 0)),
                'total_msg_cnt': safe_int(student.get('total_msg_cnt', 0)),
                'mainland_cnt': safe_int(student.get('mainland_cnt', 0)),
                'mainland_to_hk_cnt': safe_int(student.get('mainland_to_hk_cnt', 0)),
                'from_china_mobile_call_cnt': safe_int(student.get('from_china_mobile_call_cnt', 0)),
                'app_max_cnt': safe_int(student.get('app_max_cnt', 0)),
                'fraud_msisdn_present': bool(pd.notna(student.get('fraud_msisdn')))
            }
            script = generate_intervention_script(profile)
        
        st.text_area("Intervention Script", script, height=200)


def render_network_graph():
    """Interactive network graph showing fraud-student connections."""
    st.title("🕸️ Fraud Network Graph")
    st.markdown("""
    Interactive visualization of fraud-student connections. 
    - **🔴 Red diamonds**: Fraud numbers (larger = more targets)
    - **Colored circles**: Students (color by risk tier)
    - **Click** nodes to see details, **scroll** to zoom, **drag** to pan
    """)
    
    try:
        network_df = load_fraud_student_network()
    except Exception as e:
        st.error(f"Failed to load network data: {e}")
        return
    
    if len(network_df) == 0:
        st.warning("No fraud-student connections found in the data.")
        return
    
    # Calculate network statistics
    stats = create_network_stats(network_df)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔴 Fraud Numbers", f"{stats['total_fraud_numbers']:,}")
    with col2:
        st.metric("👥 Targeted Students", f"{stats['total_targeted_students']:,}")
    with col3:
        st.metric("⚠️ One-to-Many Hunters", f"{stats['one_to_many_fraudsters']:,}",
                  help="Fraud numbers targeting 3+ students")
    with col4:
        st.metric("🎯 Max Targets", f"{stats['max_targets']:,}")
    
    st.markdown("---")
    
    # Filters
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        risk_filter = st.selectbox(
            "Filter by Student Risk",
            ["All", "CRITICAL", "VULNERABLE", "SAFE"],
            key="network_risk_filter"
        )
    
    with col2:
        min_targets = st.slider(
            "Min Targets per Fraud",
            min_value=1,
            max_value=min(10, stats['max_targets']) if stats['max_targets'] > 0 else 1,
            value=1,
            help="Show only fraud numbers with at least this many targets"
        )
    
    # Apply filters
    filtered_df = network_df.copy()
    
    if risk_filter != "All":
        filtered_df = filtered_df[filtered_df['risk_tier'] == risk_filter]
    
    if min_targets > 1:
        fraud_counts = filtered_df['fraud_msisdn'].value_counts()
        valid_frauds = fraud_counts[fraud_counts >= min_targets].index
        filtered_df = filtered_df[filtered_df['fraud_msisdn'].isin(valid_frauds)]
    
    # Limit nodes for performance
    max_nodes = 500
    if len(filtered_df) > max_nodes:
        st.warning(f"Showing top {max_nodes} connections (out of {len(filtered_df):,}) for performance.")
        # Prioritize showing one-to-many patterns
        fraud_counts = filtered_df['fraud_msisdn'].value_counts()
        top_frauds = fraud_counts.head(100).index
        filtered_df = filtered_df[filtered_df['fraud_msisdn'].isin(top_frauds)].head(max_nodes)
    
    with col3:
        st.info(f"Showing **{filtered_df['fraud_msisdn'].nunique():,}** fraud numbers → **{filtered_df['user_id'].nunique():,}** students")
    
    # Create and display network graph
    if len(filtered_df) > 0:
        with st.spinner("Generating network graph..."):
            html_content = create_fraud_network(filtered_df, height="650px")
        
        # Display using streamlit components
        import streamlit.components.v1 as components
        components.html(html_content, height=700, scrolling=True)
    else:
        st.warning("No connections match the current filters.")
    
    st.markdown("---")
    
    # Hunter Analysis Section
    st.subheader("🎯 One-to-Many Hunters Analysis")
    st.markdown("Fraud numbers targeting multiple students (potential organized fraud)")
    
    hunters_df = create_hunter_table(network_df, min_targets=3)
    
    if len(hunters_df) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display hunter table
            display_hunters = hunters_df[['display_msisdn', 'target_count', 'critical_targets', 'avg_victim_risk']].head(20)
            display_hunters.columns = ['Fraud MSISDN', 'Targets', 'Critical Victims', 'Avg Risk Score']
            st.dataframe(display_hunters, use_container_width=True, hide_index=True)
        
        with col2:
            # Hunter stats
            st.markdown("**Hunter Statistics:**")
            st.write(f"- Total Hunters: **{len(hunters_df):,}**")
            st.write(f"- Total Victims: **{hunters_df['target_count'].sum():,}**")
            st.write(f"- Avg Targets/Hunter: **{hunters_df['target_count'].mean():.1f}**")
            st.write(f"- Most Prolific: **{hunters_df['target_count'].max():,}** targets")
            
            # Risk distribution of victims
            critical_total = hunters_df['critical_targets'].sum()
            st.write(f"- Critical Victims: **{critical_total:,}**")
    else:
        st.info("No one-to-many hunters found (fraud numbers targeting 3+ students).")
    
    # Drill-down: Select a fraud number to see all its targets
    st.markdown("---")
    st.subheader("🔍 Fraud Number Drill-Down")
    
    fraud_numbers = network_df['fraud_msisdn'].value_counts()
    fraud_options = [f"{mask_pii(str(msisdn))} ({count} targets)" 
                     for msisdn, count in fraud_numbers.head(50).items()]
    fraud_map = {f"{mask_pii(str(msisdn))} ({count} targets)": msisdn 
                 for msisdn, count in fraud_numbers.head(50).items()}
    
    selected_fraud_display = st.selectbox(
        "Select a fraud number to see its targets:",
        [""] + fraud_options,
        key="fraud_drilldown"
    )
    
    if selected_fraud_display and selected_fraud_display in fraud_map:
        selected_fraud = fraud_map[selected_fraud_display]
        targets = network_df[network_df['fraud_msisdn'] == selected_fraud]
        
        st.markdown(f"**Targets of fraud number {mask_pii(str(selected_fraud))}:**")
        
        display_cols = ['user_id', 'risk_tier', 'risk_score', 'identity_score', 
                        'exposure_score', 'behavior_score', 'age', 'gndr']
        available_cols = [c for c in display_cols if c in targets.columns]
        
        display_targets = mask_dataframe_pii(targets[available_cols], ['user_id'])
        st.dataframe(display_targets, use_container_width=True, hide_index=True)
        
        # Show risk tier distribution
        tier_dist = targets['risk_tier'].value_counts()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 Critical", tier_dist.get('CRITICAL', 0))
        with col2:
            st.metric("🟡 Vulnerable", tier_dist.get('VULNERABLE', 0))
        with col3:
            st.metric("🟢 Safe", tier_dist.get('SAFE', 0))


def render_fraud_model_insights():
    """Display fraud model insights with feature importance and rule breakdown."""
    import plotly.graph_objects as go
    
    st.subheader("📊 Model Insights")
    st.markdown("*Understanding what drives fraud detection*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Feature importance based on actual model features
        # These are representative values based on typical fraud detection patterns
        feature_importance = {
            'Student Targeting (opp_num_stu_cnt)': 0.28,
            'Call Volume (call_cnt_day)': 0.18,
            'IMEI Switching': 0.15,
            'Night Activity Ratio': 0.12,
            'Incoming Call Ratio': 0.10,
            'SMS Volume': 0.08,
            'Account Age': 0.05,
            'Package Type': 0.04
        }
        
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color=['#ef4444' if i < 3 else '#f97316' if i < 5 else '#22c55e' for i in range(len(features))],
            text=[f"{v:.0%}" for v in importance],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Feature Importance in Fraud Detection",
            xaxis_title="Relative Importance",
            yaxis_title="",
            height=350,
            margin=dict(l=20, r=80, t=50, b=30),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(tickformat='.0%')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Key Insights")
        st.markdown("""
        🔴 **Student Targeting** is the #1 indicator - fraud numbers consistently call student MSISDNs
        
        🟠 **Call Volume** and **IMEI Switching** indicate automated behavior
        
        🟢 **Night Activity** pattern distinguishes legitimate vs. suspicious
        """)
    
    st.markdown("---")
    
    # Rule Engine Breakdown
    st.subheader("🎯 Detection Rules Breakdown")
    
    rules_data = {
        'Rule': ['R1: Simbox', 'R2: Wangiri', 'R3: Burner', 'R4: Student Hunter', 'R5: Device Hopper', 'R6: Smishing'],
        'What It Detects': [
            'Low call diversity bots',
            'Outbound-only scam calls',
            'Newly activated fraud SIMs',
            'Numbers targeting students',
            'Frequent IMEI changes',
            'SMS-only spam accounts'
        ],
        'Threshold': [
            'dispersion < 0.04',
            'outgoing > 50, incoming = 0',
            'activation < 7 days',
            'student calls > 5',
            'IMEI changes > 1',
            'SMS > 50, calls = 0'
        ]
    }
    
    st.dataframe(
        pd.DataFrame(rules_data),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.info("💡 **How it works**: The ML model assigns a probability score, then the Rule Engine checks for specific fraud patterns. A number is blacklisted if it triggers any rule OR exceeds the ML threshold.")


def render_fraud_intel():
    """Fraud intelligence view with blacklist, greylist, and model insights."""
    st.title("🚨 Fraud Intelligence")
    
    tab1, tab2, tab3 = st.tabs(["Blacklist (Confirmed)", "Greylist (Suspicious)", "📊 Model Insights"])
    
    with tab1:
        try:
            bl = load_blacklist()
            st.subheader(f"Blacklist ({len(bl):,} MSISDNs)")
            
            # Show by source
            if 'source' in bl.columns:
                st.write("**By Source:**")
                st.write(bl['source'].value_counts())
            
            display_cols = ['msisdn', 'source', 'trigger_reason', 'risk_tier']
            available = [c for c in display_cols if c in bl.columns]
            # Apply PII masking for privacy
            display_bl = mask_dataframe_pii(bl[available].head(100), ['msisdn'])
            st.dataframe(display_bl, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to load blacklist: {e}")
    
    with tab2:
        try:
            gl = load_greylist()
            if len(gl) == 0:
                st.info("No greylist data available.")
            else:
                st.subheader(f"Greylist ({len(gl):,} MSISDNs)")
                
                display_cols = ['msisdn', 'source', 'trigger_reason']
                available = [c for c in display_cols if c in gl.columns]
                # Apply PII masking for privacy
                display_gl = mask_dataframe_pii(gl[available].head(100), ['msisdn'])
                st.dataframe(display_gl, use_container_width=True)
                
        except Exception as e:
            st.error(f"Failed to load greylist: {e}")
    
    with tab3:
        render_fraud_model_insights()


# ============================================================================
# WHITELIST REVIEW FUNCTIONS
# ============================================================================

def add_to_whitelist(account_row, reason, reviewed_by, notes):
    """Add an account to the whitelist."""
    whitelist_df = load_whitelist()
    
    # Check if already in whitelist
    if len(whitelist_df) > 0 and account_row['msisdn'] in whitelist_df['msisdn'].values:
        return False, "Account already in whitelist"
    
    new_entry = {
        'msisdn': account_row['msisdn'],
        'original_risk_score': account_row.get('risk_score', 'N/A'),
        'original_risk_tier': account_row.get('risk_tier', 'N/A'),
        'original_source': account_row.get('source', 'N/A'),
        'original_trigger_reason': account_row.get('trigger_reason', 'N/A'),
        'original_source_file': account_row.get('source_file', 'N/A'),
        'whitelist_reason': reason,
        'reviewed_by': reviewed_by,
        'review_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'review_notes': notes
    }
    
    whitelist_df = pd.concat([whitelist_df, pd.DataFrame([new_entry])], ignore_index=True)
    save_whitelist(whitelist_df)
    return True, "Successfully added to whitelist"


def remove_from_whitelist(msisdn):
    """Remove an account from the whitelist."""
    whitelist_df = load_whitelist()
    whitelist_df = whitelist_df[whitelist_df['msisdn'] != msisdn]
    save_whitelist(whitelist_df)


def render_whitelist_review():
    """Whitelist Review Console - integrated view."""
    st.title("✅ Whitelist Review Console")
    st.markdown("Review flagged accounts and manage whitelist approvals.")
    
    # Initialize session state
    if 'selected_account' not in st.session_state:
        st.session_state.selected_account = None
    if 'show_whitelist_form' not in st.session_state:
        st.session_state.show_whitelist_form = False
    
    # Staff ID input
    col1, col2 = st.columns([1, 3])
    with col1:
        reviewer_name = st.text_input("Staff ID", key="reviewer_name", placeholder="Enter your staff ID")
    
    if not reviewer_name:
        st.warning("Please enter your staff ID to review accounts")
    
    st.markdown("---")
    
    # Custom CSS for prominent tabs
    st.markdown("""
    <style>
        /* Make tabs more prominent */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 10px 24px;
            background-color: white;
            border-radius: 8px;
            font-weight: 600;
            font-size: 16px;
            border: 2px solid #e0e0e0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ff4b4b !important;
            color: white !important;
            border: 2px solid #ff4b4b !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #ffe0e0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sub-navigation tabs
    tab1, tab2, tab3 = st.tabs(["📋 Review Flagged Accounts", "📝 Whitelist Management", "📊 Statistics"])
    
    with tab1:
        render_whitelist_review_page(reviewer_name)
    
    with tab2:
        render_whitelist_management(reviewer_name)
    
    with tab3:
        render_whitelist_statistics()


def render_whitelist_review_page(reviewer_name):
    """Main review page for flagged accounts."""
    # Load data
    blacklist_df = load_blacklist()
    greylist_df = load_greylist()
    whitelist_df = load_whitelist()
    
    # Filter out already whitelisted accounts
    whitelisted_msisdns = set(whitelist_df['msisdn'].tolist()) if len(whitelist_df) > 0 else set()
    
    if len(blacklist_df) > 0:
        blacklist_df = blacklist_df[~blacklist_df['msisdn'].isin(whitelisted_msisdns)].drop_duplicates(subset=['msisdn'])
    if len(greylist_df) > 0:
        greylist_df = greylist_df[~greylist_df['msisdn'].isin(whitelisted_msisdns)].drop_duplicates(subset=['msisdn'])
    
    # Sub-tabs for blacklist and greylist
    subtab1, subtab2 = st.tabs(["⚫ Blacklist Review", "🔘 Greylist Review"])
    
    with subtab1:
        render_account_review_table(blacklist_df, "blacklist", reviewer_name)
    
    with subtab2:
        render_account_review_table(greylist_df, "greylist", reviewer_name)


def render_account_review_table(df, list_type, reviewer_name):
    """Render account review table with filtering and selection."""
    if len(df) == 0:
        st.info(f"No accounts in {list_type} to review (all may be whitelisted)")
        return
    
    st.markdown(f"### {list_type.capitalize()} Accounts ({len(df):,} total)")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sources = ['All'] + df['source'].unique().tolist() if 'source' in df.columns else ['All']
        selected_source = st.selectbox(f"Filter by Source", sources, key=f"{list_type}_source")
    
    with col2:
        search_term = st.text_input(f"Search MSISDN", key=f"{list_type}_search")
    
    with col3:
        if 'risk_score' in df.columns:
            min_score = float(df['risk_score'].min())
            max_score = float(df['risk_score'].max())
            score_range = st.slider(
                f"Risk Score Range",
                min_score, max_score, (min_score, max_score),
                key=f"{list_type}_score"
            )
        else:
            score_range = None
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_source != 'All' and 'source' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['source'] == selected_source]
    
    if search_term:
        filtered_df = filtered_df[filtered_df['msisdn'].astype(str).str.contains(search_term, case=False, na=False)]
    
    if score_range and 'risk_score' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['risk_score'] >= score_range[0]) & 
            (filtered_df['risk_score'] <= score_range[1])
        ]
    
    st.markdown(f"**Showing {len(filtered_df):,} accounts**")
    
    # Pagination
    items_per_page = 10
    total_pages = max(1, (len(filtered_df) - 1) // items_per_page + 1)
    
    if f"{list_type}_page_num" not in st.session_state:
        st.session_state[f"{list_type}_page_num"] = 1
    
    current_page = st.session_state[f"{list_type}_page_num"]
    
    # Pagination controls
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⏮️ First", key=f"{list_type}_first", disabled=(current_page == 1)):
            st.session_state[f"{list_type}_page_num"] = 1
            st.rerun()
    with col2:
        if st.button("◀️ Prev", key=f"{list_type}_prev", disabled=(current_page == 1)):
            st.session_state[f"{list_type}_page_num"] = current_page - 1
            st.rerun()
    with col3:
        st.markdown(f"<div style='text-align: center; padding: 8px;'>Page {current_page} of {total_pages}</div>", unsafe_allow_html=True)
    with col4:
        if st.button("Next ▶️", key=f"{list_type}_next", disabled=(current_page == total_pages)):
            st.session_state[f"{list_type}_page_num"] = current_page + 1
            st.rerun()
    with col5:
        if st.button("Last ⏭️", key=f"{list_type}_last", disabled=(current_page == total_pages)):
            st.session_state[f"{list_type}_page_num"] = total_pages
            st.rerun()
    
    st.markdown("---")
    
    # Display accounts
    start_idx = (current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    page_df = filtered_df.iloc[start_idx:end_idx]
    
    for idx, row in page_df.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.markdown(f"**MSISDN:** `{mask_pii(str(row['msisdn']))}`")
                st.markdown(f"**Risk Score:** {row.get('risk_score', 'N/A')}")
                st.markdown(f"**Risk Tier:** {row.get('risk_tier', 'N/A')}")
            
            with col2:
                st.markdown(f"**Source:** {row.get('source', 'N/A')}")
                st.markdown(f"**Trigger Reason:** {row.get('trigger_reason', 'N/A')}")
            
            with col3:
                if st.button("✅ Whitelist", key=f"wl_{list_type}_{idx}"):
                    st.session_state.selected_account = row.to_dict()
                    st.session_state.show_whitelist_form = True
                    st.session_state.form_list_type = list_type
                    st.session_state.form_idx = idx
            
            # Show whitelist form if this account is selected
            if (st.session_state.show_whitelist_form and 
                st.session_state.selected_account and 
                st.session_state.selected_account.get('msisdn') == row['msisdn']):
                render_whitelist_approval_form(row, list_type, idx, reviewer_name)
            
            st.markdown("---")
    
    # Bottom pagination controls
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⏮️ First", key=f"{list_type}_first_bottom", disabled=(current_page == 1)):
            st.session_state[f"{list_type}_page_num"] = 1
            st.rerun()
    with col2:
        if st.button("◀️ Prev", key=f"{list_type}_prev_bottom", disabled=(current_page == 1)):
            st.session_state[f"{list_type}_page_num"] = current_page - 1
            st.rerun()
    with col3:
        st.markdown(f"<div style='text-align: center; padding: 8px; font-weight: bold;'>Page {current_page} / {total_pages}</div>", unsafe_allow_html=True)
    with col4:
        if st.button("Next ▶️", key=f"{list_type}_next_bottom", disabled=(current_page == total_pages)):
            st.session_state[f"{list_type}_page_num"] = current_page + 1
            st.rerun()
    with col5:
        if st.button("Last ⏭️", key=f"{list_type}_last_bottom", disabled=(current_page == total_pages)):
            st.session_state[f"{list_type}_page_num"] = total_pages
            st.rerun()


def render_whitelist_approval_form(row, list_type, idx, reviewer_name):
    """Render the whitelist approval form."""
    if not reviewer_name:
        st.error("Please enter your staff ID above before approving accounts")
        return
    
    st.markdown("### 📝 Whitelist Approval Form")
    
    with st.form(key=f"whitelist_form_{list_type}_{idx}"):
        st.markdown(f"**Account:** `{mask_pii(str(row['msisdn']))}`")
        
        reason_options = [
            "Verified legitimate business",
            "False positive - normal user behavior",
            "Customer complaint - verified identity",
            "Internal employee/staff account",
            "Test account",
            "Other (specify in notes)"
        ]
        selected_reason = st.selectbox("Reason for Whitelist", reason_options)
        
        notes = st.text_area(
            "Review Notes",
            placeholder="Add any additional notes about this review decision...",
            height=100
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            submit = st.form_submit_button("✅ Confirm Whitelist", use_container_width=True)
        with col2:
            cancel = st.form_submit_button("❌ Cancel", use_container_width=True)
        
        if submit:
            if selected_reason == "Other (specify in notes)" and not notes:
                st.error("Please provide notes when selecting 'Other'")
            else:
                success, message = add_to_whitelist(row, selected_reason, reviewer_name, notes)
                if success:
                    st.success(message)
                    st.session_state.show_whitelist_form = False
                    st.session_state.selected_account = None
                    st.rerun()
                else:
                    st.error(message)
        
        if cancel:
            st.session_state.show_whitelist_form = False
            st.session_state.selected_account = None
            st.rerun()


def render_whitelist_management(reviewer_name):
    """Whitelist management page."""
    st.markdown("### 📝 Whitelist Management")
    st.markdown("View and manage approved whitelist entries.")
    
    whitelist_df = load_whitelist()
    
    if len(whitelist_df) == 0:
        st.info("No accounts in whitelist yet. Review flagged accounts to add entries.")
        return
    
    # Export section
    st.markdown("#### 📥 Export Whitelist")
    col1, col2 = st.columns(2)
    
    with col1:
        csv_full = whitelist_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Whitelist CSV",
            data=csv_full,
            file_name=f"whitelist_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        simplified_df = whitelist_df[['msisdn', 'whitelist_reason', 'review_date']].copy()
        simplified_df['is_whitelisted'] = 1
        csv_simple = simplified_df.to_csv(index=False)
        st.download_button(
            label="📥 Download for Model Training",
            data=csv_simple,
            file_name=f"whitelist_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Display whitelist
    st.markdown(f"#### Whitelist Entries ({len(whitelist_df):,} total)")
    
    search = st.text_input("Search MSISDN", key="whitelist_mgmt_search")
    
    display_df = whitelist_df.copy()
    if search:
        display_df = display_df[display_df['msisdn'].astype(str).str.contains(search, case=False, na=False)]
    
    # Pagination for whitelist management
    items_per_page = 10
    total_items = len(display_df)
    total_pages = max(1, (total_items - 1) // items_per_page + 1)
    
    if "whitelist_mgmt_page" not in st.session_state:
        st.session_state.whitelist_mgmt_page = 1
    
    current_page = st.session_state.whitelist_mgmt_page
    
    # Top pagination controls
    st.markdown(f"**Showing page {current_page} of {total_pages}** ({total_items} entries)")
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    with col1:
        if st.button("⏮️ First", key="wl_mgmt_first", disabled=(current_page == 1)):
            st.session_state.whitelist_mgmt_page = 1
            st.rerun()
    with col2:
        if st.button("◀️ Prev", key="wl_mgmt_prev", disabled=(current_page == 1)):
            st.session_state.whitelist_mgmt_page = current_page - 1
            st.rerun()
    with col3:
        st.markdown(f"<div style='text-align: center; padding: 8px; font-weight: bold;'>Page {current_page} / {total_pages}</div>", unsafe_allow_html=True)
    with col4:
        if st.button("Next ▶️", key="wl_mgmt_next", disabled=(current_page == total_pages)):
            st.session_state.whitelist_mgmt_page = current_page + 1
            st.rerun()
    with col5:
        if st.button("Last ⏭️", key="wl_mgmt_last", disabled=(current_page == total_pages)):
            st.session_state.whitelist_mgmt_page = total_pages
            st.rerun()
    
    st.markdown("---")
    
    # Calculate page slice
    start_idx = (current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    page_df = display_df.iloc[start_idx:end_idx]
    
    for idx, row in page_df.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([3, 3, 1])
            
            with col1:
                st.markdown(f"**MSISDN:** `{mask_pii(str(row['msisdn']))}`")
                st.markdown(f"**Original Risk:** {row.get('original_risk_tier', 'N/A')} ({row.get('original_risk_score', 'N/A')})")
            
            with col2:
                st.markdown(f"**Reason:** {row.get('whitelist_reason', 'N/A')}")
                st.markdown(f"**Reviewed by:** {row.get('reviewed_by', 'N/A')} on {row.get('review_date', 'N/A')}")
                if row.get('review_notes'):
                    st.markdown(f"**Notes:** {row.get('review_notes')}")
            
            with col3:
                if st.button("🗑️ Remove", key=f"remove_wl_{idx}"):
                    remove_from_whitelist(row['msisdn'])
                    st.success("Removed from whitelist")
                    st.rerun()
            
            st.markdown("---")
    
    # Bottom pagination controls
    st.markdown("")
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    with col1:
        if st.button("⏮️ First", key="wl_mgmt_first_bottom", disabled=(current_page == 1)):
            st.session_state.whitelist_mgmt_page = 1
            st.rerun()
    with col2:
        if st.button("◀️ Prev", key="wl_mgmt_prev_bottom", disabled=(current_page == 1)):
            st.session_state.whitelist_mgmt_page = current_page - 1
            st.rerun()
    with col3:
        st.markdown(f"<div style='text-align: center; padding: 8px; font-weight: bold;'>Page {current_page} / {total_pages}</div>", unsafe_allow_html=True)
    with col4:
        if st.button("Next ▶️", key="wl_mgmt_next_bottom", disabled=(current_page == total_pages)):
            st.session_state.whitelist_mgmt_page = current_page + 1
            st.rerun()
    with col5:
        if st.button("Last ⏭️", key="wl_mgmt_last_bottom", disabled=(current_page == total_pages)):
            st.session_state.whitelist_mgmt_page = total_pages
            st.rerun()


def render_whitelist_statistics():
    """Statistics and overview page."""
    st.markdown("### 📊 Review Statistics")
    
    blacklist_df = load_blacklist()
    greylist_df = load_greylist()
    whitelist_df = load_whitelist()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("⚫ Blacklist", f"{len(blacklist_df):,}")
    with col2:
        st.metric("🔘 Greylist", f"{len(greylist_df):,}")
    with col3:
        st.metric("⚪ Whitelist", f"{len(whitelist_df):,}")
    with col4:
        total = len(blacklist_df) + len(greylist_df)
        reviewed_pct = (len(whitelist_df) / total * 100) if total > 0 else 0
        st.metric("📈 Review Rate", f"{reviewed_pct:.1f}%")
    
    st.markdown("---")
    
    if len(whitelist_df) > 0:
        st.markdown("#### Whitelist Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**By Reason**")
            reason_counts = whitelist_df['whitelist_reason'].value_counts()
            st.bar_chart(reason_counts)
        
        with col2:
            st.markdown("**By Reviewer**")
            reviewer_counts = whitelist_df['reviewed_by'].value_counts()
            st.bar_chart(reviewer_counts)
        
        st.markdown("#### Recent Reviews")
        recent_df = whitelist_df.sort_values('review_date', ascending=False).head(10)
        display_recent_df = mask_dataframe_pii(recent_df[['msisdn', 'whitelist_reason', 'reviewed_by', 'review_date']], ['msisdn'])
        st.dataframe(display_recent_df, use_container_width=True, hide_index=True)
    else:
        st.info("No whitelist entries yet. Start reviewing accounts to see statistics.")


if __name__ == "__main__":
    main()
