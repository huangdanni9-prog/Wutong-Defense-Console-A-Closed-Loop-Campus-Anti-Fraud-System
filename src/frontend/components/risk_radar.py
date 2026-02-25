"""
Risk Radar Chart - Plotly Spider Plot for 3-axis risk visualization.
"""
import plotly.graph_objects as go


def create_risk_radar(identity_score: int, exposure_score: int, behavior_score: int) -> go.Figure:
    """
    Create Plotly Radar Chart for Risk Triangle visualization.
    
    Args:
        identity_score: Identity vulnerability (0-100)
        exposure_score: Threat exposure (0-100)
        behavior_score: Risky behavior (0-100)
    
    Returns:
        Plotly Figure object
    """
    categories = ['Identity', 'Exposure', 'Behavior']
    values = [identity_score, exposure_score, behavior_score]
    
    # Close the polygon by repeating first value
    r = values + [values[0]]
    theta = categories + [categories[0]]
    
    # Determine color based on max score
    max_score = max(values)
    if max_score >= 80:
        fill_color = 'rgba(231, 76, 60, 0.4)'  # Red
        line_color = '#e74c3c'
    elif max_score >= 50:
        fill_color = 'rgba(243, 156, 18, 0.4)'  # Orange
        line_color = '#f39c12'
    else:
        fill_color = 'rgba(46, 204, 113, 0.4)'  # Green
        line_color = '#2ecc71'
    
    fig = go.Figure(data=go.Scatterpolar(
        r=r,
        theta=theta,
        fill='toself',
        fillcolor=fill_color,
        line=dict(color=line_color, width=3),
        name='Risk Profile'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
            ),
            angularaxis=dict(
                tickfont=dict(size=12, weight='bold')
            )
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=40, b=40),
        height=350
    )
    
    return fig


def create_comparison_radar(student_scores: dict, avg_scores: dict) -> go.Figure:
    """
    Create comparison Radar Chart showing student vs average.
    """
    categories = ['Identity', 'Exposure', 'Behavior']
    
    student_values = [
        student_scores.get('identity_score', 0),
        student_scores.get('exposure_score', 0),
        student_scores.get('behavior_score', 0)
    ]
    avg_values = [
        avg_scores.get('identity_score', 0),
        avg_scores.get('exposure_score', 0),
        avg_scores.get('behavior_score', 0)
    ]
    
    fig = go.Figure()
    
    # Student
    fig.add_trace(go.Scatterpolar(
        r=student_values + [student_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(231, 76, 60, 0.3)',
        line=dict(color='#e74c3c', width=2),
        name='This Student'
    ))
    
    # Average
    fig.add_trace(go.Scatterpolar(
        r=avg_values + [avg_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.2)',
        line=dict(color='#3498db', width=2, dash='dash'),
        name='Population Average'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100])),
        showlegend=True,
        legend=dict(x=0.5, y=-0.1, xanchor='center', orientation='h'),
        margin=dict(l=60, r=60, t=40, b=60),
        height=400
    )
    
    return fig
