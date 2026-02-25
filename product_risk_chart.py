import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load fraud data
data_path = Path('Datasets/Fraud/Training and Testing Data/fraud_model_2.csv')
df = pd.read_csv(data_path)
print('Total records:', len(df))

# Create fraud label
df['is_fraud'] = (df['audit_status'] == '稽核不通過').astype(int)

# Map Chinese to English for display
df['contract_type_en'] = df['post_or_ppd'].map({'预付': 'Prepaid', '后付': 'Contract'}).fillna('Unknown')

# Handle NaN in plan category
df['rt_plan_cat'] = df['rt_plan_cat'].fillna('Unknown Plan')

# Check distribution
print('\n=== Contract Type vs Fraud ===')
print(f"Prepaid: {len(df[df['contract_type_en']=='Prepaid'])} records, {df[df['contract_type_en']=='Prepaid']['is_fraud'].sum()} fraud")
print(f"Contract: {len(df[df['contract_type_en']=='Contract'])} records, {df[df['contract_type_en']=='Contract']['is_fraud'].sum()} fraud")

# Create combined product category: contract type + plan (English)
df['product'] = df['contract_type_en'] + ' - ' + df['rt_plan_cat'].astype(str).str[:15]

# Aggregate by product
plan_stats = df.groupby('product').agg({
    'contract_type_en': 'first',
    'call_cnt_day': 'mean',
    'is_fraud': ['sum', 'count', 'mean']
}).reset_index()
plan_stats.columns = ['product', 'contract_type', 'avg_calls', 'fraud_count', 'total', 'fraud_rate']
plan_stats = plan_stats[plan_stats['total'] >= 20]  # Filter small groups

# Create Anonymity Score (1 = Prepaid, 0 = Contract)
plan_stats['anonymity'] = (plan_stats['contract_type'] == 'Prepaid').astype(float)
# Add jitter for visibility
plan_stats['anonymity'] += np.random.uniform(-0.1, 0.1, len(plan_stats))

# Normalize voice ratio
max_calls = plan_stats['avg_calls'].quantile(0.95)
plan_stats['voice_ratio'] = np.clip(plan_stats['avg_calls'] / max_calls, 0, 1)

print('\n=== Top Fraud Products ===')
print(plan_stats.sort_values('fraud_rate', ascending=False).head(10))

# Create bubble chart
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Color by fraud rate
colors = []
for rate in plan_stats['fraud_rate']:
    if rate > 0.7:
        colors.append('#e74c3c')  # Red
    elif rate > 0.4:
        colors.append('#f39c12')  # Orange
    else:
        colors.append('#27ae60')  # Green

# Bubble size = fraud count (scaled)
sizes = plan_stats['fraud_count'] / plan_stats['fraud_count'].max() * 2000 + 100

scatter = ax.scatter(
    plan_stats['anonymity'], 
    plan_stats['voice_ratio'],
    s=sizes,
    c=colors,
    alpha=0.7,
    edgecolors='white',
    linewidth=2
)

# Add labels for all products
for idx, row in plan_stats.iterrows():
    label = row['product'][:20]
    fraud_pct = row['fraud_rate'] * 100
    color = '#e74c3c' if fraud_pct > 70 else '#f39c12' if fraud_pct > 40 else '#27ae60'
    ax.annotate(f"{label}\n{fraud_pct:.0f}% fraud\n(n={row['total']})",
                (row['anonymity'], row['voice_ratio']),
                textcoords='offset points', xytext=(10, 5),
                ha='left', fontsize=8, fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

# Add zone lines
ax.axhline(y=0.3, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=2)
ax.axvline(x=0.5, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=2)
ax.fill_between([0.5, 1.2], 0.3, 1.2, alpha=0.15, color='red')
ax.text(0.85, 0.9, 'HIGH RISK\nZONE', fontsize=16, color='#e74c3c', fontweight='bold', ha='center')
ax.text(0.15, 0.9, 'LOW RISK\n(Contract)', fontsize=12, color='#27ae60', fontweight='bold', ha='center')

# Styling
ax.set_xlabel('Anonymity Score (Contract → Prepaid)', fontsize=12, color='white')
ax.set_ylabel('Voice-to-Data Ratio (Avg Calls/Day)', fontsize=12, color='white')
ax.set_title('Product Risk Matrix: The Fraud Arsenal\n(Based on Actual CMHK Data)', fontsize=16, fontweight='bold', color='white')
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.1, 1.1)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color('white')
ax.grid(True, alpha=0.2, color='white')

# Legend
ax.scatter([], [], s=300, c='#e74c3c', label='High Fraud (>70%)', edgecolors='white')
ax.scatter([], [], s=300, c='#f39c12', label='Medium (40-70%)', edgecolors='white')
ax.scatter([], [], s=300, c='#27ae60', label='Low Fraud (<40%)', edgecolors='white')
legend = ax.legend(loc='lower left', title='Fraud Rate', facecolor='#2d2d44', edgecolor='white')
legend.get_title().set_color('white')
for text in legend.get_texts():
    text.set_color('white')

ax.text(0.02, 0.98, 'Bubble Size = Fraud Count', transform=ax.transAxes, 
        fontsize=10, color='white', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#2d2d44', alpha=0.8))

plt.tight_layout()
output_path = 'Datasets/Fraud/Results/product_risk_matrix.png'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f'\nSaved: {output_path}')
