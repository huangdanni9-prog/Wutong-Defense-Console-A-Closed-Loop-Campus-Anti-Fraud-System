# Rule Threshold Justification Report
*Generated using Unsupervised ML (K-Means Clustering)*

This report provides data-driven justification for all rule thresholds
based on cluster analysis of real data.

---

## Student Risk Model Thresholds

### Age 18-22: +5 points
- **Finding**: High-risk clusters: 8% are 18-22 vs 79% in low-risk
- **Threshold**: `18 <= age <= 22`

### Mainland Calls >= 1: +5 points
- **Finding**: High-risk avg: 0.1 vs Low-risk avg: 0.2
- **Threshold**: `from_china_mobile_call_cnt >= 1`

### Callback: x2.0 multiplier, Answered: x1.5 multiplier
- **Finding**: Active engagement (callback: 9.8750, answered: 2.4327) correlates with high fraud rate
- **Threshold**: `voice_call > 0 | voice_receive > 0`

---

## Fraud Detection Rule Thresholds

### R1_Simbox: calls > 325 AND dispersion < 0.04
- **Finding**: Fraud clusters avg calls: 49, dispersion: 0.0000
- **Threshold**: `call_cnt_day > 325`

### R2_Wangiri: calls > 50 AND incoming == 0
- **Finding**: Fraud clusters avg incoming: 0.4 (very low)
- **Threshold**: `called_cnt_day == 0`

---

## Methodology

1. **Data Loading**: Loaded student_model.csv and fraud_model_2.csv
2. **Feature Engineering**: Selected behavioral and exposure features
3. **Outlier Removal**: Used Isolation Forest (1% contamination)
4. **Clustering**: K-Means (k=5) to find natural behavioral groups
5. **Analysis**: Compared fraud rates across clusters
6. **Threshold Derivation**: Set thresholds at cluster boundaries

This is the **Gold Standard** workflow: Data → Discovery → Rules
