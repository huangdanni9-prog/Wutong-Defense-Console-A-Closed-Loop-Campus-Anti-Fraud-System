import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results CSV
df = pd.read_csv('Datasets/Student/Results/student_risk_predictions.csv')
print(f'Loaded {len(df)} students')

scores = df['risk_score'].fillna(0)
is_high_risk = df['is_high_risk'] == 1

# Apply power transformation to spread scores across 0-100
# This makes the distribution more visible
max_score = scores.max()
scores_normalized = scores / max_score  # 0-1 range
scores_scaled = np.power(scores_normalized, 0.3) * 100  # Power transform spreads it out

print(f'High-risk: {is_high_risk.sum()} ({is_high_risk.mean()*100:.1f}%)')
print(f'Original score range: {scores.min():.0f} - {scores.max():.0f}')
print(f'Transformed score range: {scores_scaled.min():.0f} - {scores_scaled.max():.0f}')

# Create distribution chart
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Separate scaled scores for high-risk vs safe
safe_scores = scores_scaled[~is_high_risk]
risk_scores = scores_scaled[is_high_risk]

bins = np.arange(0, 105, 5)

# Use stacked histogram to avoid overlap
ax.hist([safe_scores, risk_scores], bins=bins, stacked=True,
        color=['#3498db', '#e74c3c'], edgecolor='white', linewidth=0.3,
        label=[f'Safe ({len(safe_scores):,})', f'High-Risk ({len(risk_scores):,})'])

# Add annotations
high_risk_pct = is_high_risk.mean() * 100
safe_pct = 100 - high_risk_pct

ax.text(70, ax.get_ylim()[1]*0.85, 
        f'CRITICAL ZONE\n{is_high_risk.sum():,} students\n({high_risk_pct:.1f}%)', 
        fontsize=14, color='#e74c3c', fontweight='bold', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#e74c3c', alpha=0.9))

ax.text(15, ax.get_ylim()[1]*0.85, 
        f'SAFE\n{(~is_high_risk).sum():,} students\n({safe_pct:.1f}%)', 
        fontsize=12, color='#3498db', fontweight='bold', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#3498db', alpha=0.9))

# Styling
ax.set_xlabel('Risk Score (0-100)', fontsize=14, color='black')
ax.set_ylabel('Number of Students', fontsize=14, color='black')
ax.set_title('Student Risk Distribution: The Critical Zone\n(Based on Actual Data - 57,713 Students)', 
             fontsize=16, fontweight='bold', color='black')
ax.tick_params(colors='black', labelsize=11)
for spine in ax.spines.values():
    spine.set_color('black')
ax.set_xlim(0, 100)
ax.grid(True, alpha=0.3, color='gray')

# Legend
legend = ax.legend(loc='upper right', facecolor='white', edgecolor='gray', fontsize=11)
for text in legend.get_texts():
    text.set_color('black')

plt.tight_layout()
output_path = 'img/risk_distribution_real.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'\nSaved: {output_path}')
