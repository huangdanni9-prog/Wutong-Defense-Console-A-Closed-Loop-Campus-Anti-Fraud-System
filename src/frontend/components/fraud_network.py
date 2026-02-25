"""
Fraud Network Graph - Interactive visualization of fraud-student connections.
Shows one-to-many patterns where fraudsters target multiple students.
"""
import pandas as pd
import streamlit as st
from pyvis.network import Network
import tempfile
import os


def create_fraud_network(
    network_df: pd.DataFrame,
    height: str = "700px",
    width: str = "100%",
    show_physics_controls: bool = False
) -> str:
    """
    Create an interactive network graph showing fraud-student connections.
    
    Args:
        network_df: DataFrame with columns [fraud_msisdn, user_id, risk_tier, ...]
        height: Graph height
        width: Graph width
        show_physics_controls: Whether to show physics simulation controls
    
    Returns:
        HTML string for the network graph
    """
    # Initialize network
    net = Network(
        height=height,
        width=width,
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
        select_menu=False,
        filter_menu=False
    )
    
    # Physics settings for better layout
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.01,
        damping=0.09
    )
    
    if show_physics_controls:
        net.show_buttons(filter_=['physics'])
    
    # Track nodes to avoid duplicates
    added_fraud_nodes = set()
    added_student_nodes = set()
    
    # Count connections per fraud number for sizing
    fraud_connection_counts = network_df['fraud_msisdn'].value_counts().to_dict()
    
    # Risk tier colors for students
    tier_colors = {
        'CRITICAL': '#e74c3c',      # Red
        'VULNERABLE': '#f39c12',    # Orange
        'SAFE': '#2ecc71',          # Green
        'UNKNOWN': '#95a5a6'        # Gray
    }
    
    # Add nodes and edges
    for _, row in network_df.iterrows():
        fraud_id = str(row['fraud_msisdn'])
        student_id = str(row['user_id'])
        risk_tier = row.get('risk_tier', 'UNKNOWN')
        risk_score = row.get('risk_score', 0)
        
        # Skip if fraud_msisdn is empty/null
        if pd.isna(row['fraud_msisdn']) or fraud_id in ('', 'nan', 'None'):
            continue
        
        # Connection count for this fraud number
        conn_count = fraud_connection_counts.get(row['fraud_msisdn'], 1)
        
        # Add fraud node (if not already added)
        if fraud_id not in added_fraud_nodes:
            # Size based on number of connections (one-to-many indicator)
            fraud_size = min(15 + conn_count * 5, 60)
            
            # Determine if this is a "hunter" (targets many students)
            is_hunter = conn_count >= 3
            fraud_color = '#ff0000' if is_hunter else '#ff6b6b'
            
            fraud_label = f"📞 {_mask_msisdn(fraud_id)}"
            # Use plain text for tooltip (vis.js doesn't render HTML in tooltips by default)
            hunter_pattern = '⚠️ ONE-TO-MANY HUNTER' if is_hunter else 'Single target'
            fraud_title = f"🚨 Fraud Number\nMSISDN: {_mask_msisdn(fraud_id)}\nTargets: {conn_count} student(s)\nPattern: {hunter_pattern}"
            
            net.add_node(
                f"fraud_{fraud_id}",
                label=fraud_label,
                title=fraud_title,
                color=fraud_color,
                size=fraud_size,
                shape="diamond",
                borderWidth=3,
                borderWidthSelected=5,
                font={'size': 12, 'color': 'white'}
            )
            added_fraud_nodes.add(fraud_id)
        
        # Add student node (if not already added)
        if student_id not in added_student_nodes:
            student_color = tier_colors.get(risk_tier, tier_colors['UNKNOWN'])
            student_size = 20 + int(risk_score / 10)
            
            student_label = f"👤 {_mask_user_id(student_id)}"
            # Use plain text for tooltip (vis.js doesn't render HTML in tooltips by default)
            student_title = f"👤 Student\nID: {_mask_user_id(student_id)}\nRisk Tier: {risk_tier}\nRisk Score: {risk_score}/100\nIdentity: {row.get('identity_score', 'N/A')}\nExposure: {row.get('exposure_score', 'N/A')}\nBehavior: {row.get('behavior_score', 'N/A')}"
            
            net.add_node(
                f"student_{student_id}",
                label=student_label,
                title=student_title,
                color=student_color,
                size=student_size,
                shape="dot",
                borderWidth=2,
                borderWidthSelected=4,
                font={'size': 10, 'color': 'white'}
            )
            added_student_nodes.add(student_id)
        
        # Add edge (fraud -> student)
        edge_color = '#ff4444' if conn_count >= 3 else '#888888'
        net.add_edge(
            f"fraud_{fraud_id}",
            f"student_{student_id}",
            color=edge_color,
            width=2,
            arrows={'to': {'enabled': True, 'scaleFactor': 0.5}}
        )
    
    # Generate HTML
    try:
        # Create temp file for the graph - handle Windows file locking
        temp_path = None
        try:
            # Create a temp file and close it immediately to get the path
            fd, temp_path = tempfile.mkstemp(suffix='.html')
            os.close(fd)  # Close the file descriptor immediately
            
            # Save the graph to the temp file
            net.save_graph(temp_path)
            
            # Read the content back
            with open(temp_path, 'r', encoding='utf-8') as rf:
                html_content = rf.read()
            
            return html_content
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except (PermissionError, OSError):
                    pass  # Ignore cleanup errors on Windows
    except Exception as e:
        return f"<p>Error generating network graph: {e}</p>"


def create_network_stats(network_df: pd.DataFrame) -> dict:
    """
    Calculate network statistics for display.
    
    Returns:
        Dictionary with network metrics
    """
    if network_df.empty:
        return {
            'total_fraud_numbers': 0,
            'total_targeted_students': 0,
            'one_to_many_fraudsters': 0,
            'max_targets': 0,
            'avg_targets': 0,
            'hunter_list': []
        }
    
    # Count connections per fraud number
    fraud_counts = network_df['fraud_msisdn'].value_counts()
    
    # Identify "hunters" (fraud numbers targeting 3+ students)
    hunters = fraud_counts[fraud_counts >= 3]
    
    return {
        'total_fraud_numbers': network_df['fraud_msisdn'].nunique(),
        'total_targeted_students': network_df['user_id'].nunique(),
        'one_to_many_fraudsters': len(hunters),
        'max_targets': int(fraud_counts.max()) if len(fraud_counts) > 0 else 0,
        'avg_targets': round(fraud_counts.mean(), 1) if len(fraud_counts) > 0 else 0,
        'hunter_list': hunters.head(10).to_dict()  # Top 10 hunters
    }


def get_fraud_details(network_df: pd.DataFrame, fraud_msisdn: str) -> pd.DataFrame:
    """
    Get all students contacted by a specific fraud number.
    """
    return network_df[network_df['fraud_msisdn'] == fraud_msisdn]


def get_student_fraud_contacts(network_df: pd.DataFrame, user_id: str) -> pd.DataFrame:
    """
    Get all fraud numbers that contacted a specific student.
    """
    return network_df[network_df['user_id'] == user_id]


def _mask_msisdn(msisdn: str) -> str:
    """Mask phone number for privacy."""
    if not msisdn or len(msisdn) < 8:
        return msisdn
    return msisdn[:4] + "****" + msisdn[-4:]


def _mask_user_id(user_id: str) -> str:
    """Mask user ID for privacy."""
    if not user_id or len(user_id) < 8:
        return user_id
    return user_id[:4] + "..." + user_id[-4:]


def create_hunter_table(network_df: pd.DataFrame, min_targets: int = 3) -> pd.DataFrame:
    """
    Create a table of fraud numbers that target multiple students (hunters).
    
    Args:
        network_df: Network data
        min_targets: Minimum number of targets to be considered a hunter
    
    Returns:
        DataFrame with hunter information
    """
    if network_df.empty:
        return pd.DataFrame()
    
    # Group by fraud number and aggregate
    hunter_stats = network_df.groupby('fraud_msisdn').agg({
        'user_id': 'count',
        'risk_tier': lambda x: (x == 'CRITICAL').sum(),
        'risk_score': 'mean'
    }).reset_index()
    
    hunter_stats.columns = ['fraud_msisdn', 'target_count', 'critical_targets', 'avg_victim_risk']
    
    # Filter to hunters only
    hunters = hunter_stats[hunter_stats['target_count'] >= min_targets].copy()
    hunters = hunters.sort_values('target_count', ascending=False)
    
    # Add masked display
    hunters['display_msisdn'] = hunters['fraud_msisdn'].apply(_mask_msisdn)
    hunters['avg_victim_risk'] = hunters['avg_victim_risk'].round(1)
    
    return hunters
