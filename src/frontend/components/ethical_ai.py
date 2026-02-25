"""
Ethical AI & Responsible AI Component

Clean, professional display of ethical AI principles implemented in the
Wutong Defense fraud detection system.
"""

import streamlit as st
import pandas as pd


def render_ethical_ai():
    """
    Render the Ethical AI & Responsible AI page.
    Clean, impactful design focusing on key principles.
    """
    # Custom CSS for premium dark theme
    st.markdown("""
    <style>
        .header-section {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 30px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .header-title {
            color: #ffffff;
            font-size: 2.8rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .header-subtitle {
            color: #94a3b8;
            font-size: 1.2rem;
            margin-top: 15px;
            font-weight: 300;
        }
        .principle-card {
            background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
            padding: 30px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 20px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .principle-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        .principle-icon {
            font-size: 3rem;
            margin-bottom: 15px;
        }
        .principle-title {
            color: #ffffff;
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 15px;
        }
        .principle-desc {
            color: #94a3b8;
            font-size: 1rem;
            line-height: 1.6;
        }
        .highlight {
            color: #60a5fa;
            font-weight: 500;
        }
        .commitment-badge {
            display: inline-block;
            background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 5px;
        }
        .footer-section {
            background: rgba(30, 41, 59, 0.5);
            padding: 25px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.05);
            margin-top: 30px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="header-section">
        <p class="header-title">🛡️ Responsible AI</p>
        <p class="header-subtitle">Building Trust Through Transparency, Fairness, and Privacy</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Core Principles - 2x2 Grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="principle-card">
            <div class="principle-icon">🔐</div>
            <div class="principle-title">Privacy by Design</div>
            <div class="principle-desc">
                All personal data is <span class="highlight">masked and protected</span> before display. 
                Phone numbers and IDs are never shown in full. Aggregate statistics use 
                <span class="highlight">differential privacy</span> to prevent re-identification attacks.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="principle-card">
            <div class="principle-icon">⚖️</div>
            <div class="principle-title">Fairness & Non-Discrimination</div>
            <div class="principle-desc">
                Fraud scoring uses <span class="highlight">behavioral patterns only</span>. 
                Age, gender, nationality, and other protected attributes are 
                <span class="highlight">explicitly excluded</span> from the ML model features.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="principle-card">
            <div class="principle-icon">💡</div>
            <div class="principle-title">Explainable Decisions</div>
            <div class="principle-desc">
                Every flagged account shows <span class="highlight">exactly why</span> it was flagged. 
                Our 6-rule engine provides clear trigger reasons, and 
                <span class="highlight">SHAP analysis</span> explains ML model predictions.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="principle-card">
            <div class="principle-icon">👥</div>
            <div class="principle-title">Human-in-the-Loop</div>
            <div class="principle-desc">
                No automated blocking. All flagged accounts are 
                <span class="highlight">reviewed by staff</span> before action. 
                A whitelist process allows appeals for false positives.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Compliance Badges
    st.markdown("### 📋 Compliance & Standards")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <span class="commitment-badge">✓ PCPD Guidelines</span>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 10px;">
                Hong Kong Privacy Commissioner for Personal Data
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <span class="commitment-badge">✓ Right to Explanation</span>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 10px;">
                Every decision includes human-readable justification
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <span class="commitment-badge">✓ Data Minimization</span>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 10px;">
                Only behavioral features; no call content stored
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Features Table
    st.markdown("### 🔧 Implementation Highlights")
    
    features = pd.DataFrame({
        'Principle': ['Privacy', 'Explainability', 'Fairness', 'Accountability'],
        'Implementation': [
            'PII masking, Differential Privacy for aggregate stats',
            'SHAP waterfall plots, Rule-based trigger reasons',
            'Protected attributes excluded, Balanced training data',
            'Whitelist appeals, Full audit trail'
        ],
        'Status': ['✅ Active', '✅ Active', '✅ Active', '✅ Active']
    })
    
    st.dataframe(
        features,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Principle': st.column_config.TextColumn('Principle', width='small'),
            'Implementation': st.column_config.TextColumn('How We Implement It', width='large'),
            'Status': st.column_config.TextColumn('Status', width='small')
        }
    )
    
    # Footer
    st.markdown("""
    <div class="footer-section">
        <h4 style="color: #ffffff; margin-bottom: 15px;">🎯 Our Commitment</h4>
        <p style="color: #94a3b8; margin: 0; line-height: 1.7;">
            We believe fraud detection should protect users without compromising their rights. 
            This system is designed to be <strong style="color: #60a5fa;">transparent</strong>, 
            <strong style="color: #60a5fa;">fair</strong>, and 
            <strong style="color: #60a5fa;">privacy-preserving</strong>. 
            Every decision can be explained, every flag can be appealed, and every user's privacy is protected.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    render_ethical_ai()
