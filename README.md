# 🛡️ Wutong Defense Console

**AI-Powered Campus Telecom Fraud Detection System**

A comprehensive solution for identifying and preventing telecom fraud targeting students in Hong Kong.

---

## 📋 Project Overview

This project addresses three core tasks:

| Task       | Description                    | Approach                                   |
| ---------- | ------------------------------ | ------------------------------------------ |
| **Task 1** | High-Risk Student Portrait     | Risk Triangle Scoring + K-Means Clustering |
| **Task 2** | Wire Fraud User Portrait       | XGBoost + Rule Engine + Persona Clustering |
| **Task 3** | Product Vulnerability Analysis | Feature analysis of exploited products     |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Wutong Defense Console                        │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (Streamlit)                                            │
│  └── Dashboard, Student Details, Fraud Intel, Simulators        │
├─────────────────────────────────────────────────────────────────┤
│  Student Risk Module          │  Fraud Detection Module          │
│  ├── Feature Engineering      │  ├── Feature Engineering         │
│  ├── Risk Triangle Scorer     │  ├── 6-Rule Engine               │
│  ├── Clustering Model         │  ├── XGBoost + Isolation Forest  │
│  └── Portrait Generator       │  └── Fraud Clustering            │
├─────────────────────────────────────────────────────────────────┤
│  Privacy Stack: Differential Privacy (ε=5.0)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone and navigate
cd Solution

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run Application

```bash
# Start Streamlit frontend
cd src/frontend
python -m streamlit run app.py --server.port 8501
```

Open: http://localhost:8501

---

## 📁 Project Structure

```
Solution/
├── src/
│   ├── student_risk/           # Task 1: Student Risk Assessment
│   │   ├── feature_engineering.py
│   │   ├── risk_triangle_scorer.py
│   │   ├── clustering_model.py
│   │   └── student_portrait.py
│   │
│   ├── fraud_detection/        # Task 2: Fraud Detection
│   │   ├── fraud_feature_engineering.py
│   │   ├── fraud_rule_engine.py
│   │   ├── fraud_scoring_model.py
│   │   └── fraud_clustering.py
│   │
│   └── frontend/               # Streamlit UI
│       ├── app.py
│       └── components/
│
├── Datasets/
│   ├── Student/                # Student data & results
│   ├── Fraud/                  # Fraud data & results
│   └── Analysis/               # Cross-analysis outputs
│
├── models/                     # Saved ML models
├── img/                        # Screenshots & diagrams
└── requirements.txt
```

---

## 🎯 Key Features

### Student Risk Module (Task 1)

- **Risk Triangle Scoring**: Identity → Exposure → Behavior
- **Persona Clustering**: The Naive Freshman, The Connected Elder, etc.
- **Explainable Reasons**: Human-readable risk explanations

### Fraud Detection Module (Task 2)

- **6-Rule Engine**: Simbox, Wangiri, Student Hunter, etc.
- **Hybrid ML**: XGBoost + Isolation Forest
- **Persona Clustering**: The Robocall Factory, The Campus Predator, etc.

### Frontend Features

- 📊 Real-time Dashboard
- 👤 Student Detail Lookup
- 🎮 Live Risk Simulator
- 🕸️ Network Visualization
- ✅ Whitelist Review Workflow

---

## 📊 Results

### Task 1: Student Risk

- **5,240** HIGH-RISK students identified (9.1%)
- **4 Personas**: Naive Freshman, Connected Elder, Silent Victim, etc.

### Task 2: Fraud Detection

- **6-Rule Engine** catches known fraud patterns
- **ML Model** detects unknown fraud variants
- **3-Tier Classification**: BLACKLIST → GREYLIST → WHITELIST

---

## 🔒 Privacy & Ethics

- **Differential Privacy**: ε=5.0 noise injection
- **SHAP Explainability**: Transparent model decisions
- **Human-in-the-loop**: Greylist requires manual review

---

## 👥 Team

_[Add your team members here]_

---

## 📄 License

This project is for the CMHK AI Hackathon 2025.
