# 🛡️ Wutong Defense Console

**AI-Powered Campus Telecom Fraud Detection System**

A closed-loop, end-to-end solution for identifying, analyzing, and preventing telecom fraud targeting university students in Hong Kong. Built for the CHINA MOBILE WUTONG CUP 2025, the system combines rule-based detection, machine learning models, LLM-powered intervention, and an interactive Streamlit dashboard to create a comprehensive anti-fraud defense platform.

---

## 📋 Project Overview

Telecom fraud targeting campus students is a growing threat in Hong Kong — scammers exploit young, cross-border students through tactics such as fake government calls, Wangiri callbacks, and SIM-box robo-dialing. **Wutong Defense Console** tackles this problem across three interconnected tasks:

| Task       | Description                    | Approach                                   |
| ---------- | ------------------------------ | ------------------------------------------ |
| **Task 1** | High-Risk Student Portrait     | Risk Triangle Scoring + K-Means Clustering |
| **Task 2** | Wire Fraud User Portrait       | XGBoost + Rule Engine + Persona Clustering |
| **Task 3** | Product Vulnerability Analysis | Feature analysis of exploited products     |

### How the Tasks Connect (Closed-Loop Design)

The system forms a **closed loop** — outputs from one task feed into another:

1. **Task 2 (Fraud Detection)** identifies confirmed fraud MSISDNs and produces a Blacklist / Greylist.
2. **Task 1 (Student Risk)** cross-references students' call records against the Task 2 threat database to calculate a Risk Triangle Score (Identity × Exposure × Behavior).
3. **Task 3 (Product Analysis)** examines which telecom products and plans are most exploited by fraudsters, enabling proactive product-level hardening.
4. The **Frontend Dashboard** ties everything together — operators can review alerts, drill into student profiles, run live simulations, and generate AI-powered intervention scripts via Groq LLM.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Wutong Defense Console                            │
├──────────────────────────────────────────────────────────────────────────┤
│  Frontend (Streamlit)                                                    │
│  ├──  Real-time Dashboard           - KPI overview with DP-protected     │
│  │                                    aggregate statistics               │
│  ├──  Student Detail Lookup         - Individual risk profiles with      │
│  │                                    SHAP-style explanations            │
│  ├──  Live Risk Simulator           - Adjust parameters to see score     │
│  │                                    changes in real time               │
│  ├──  Network Visualization         - Interactive fraud-student graph    │
│  │                                     (vis.js / pyvis)                  │
│  ├──  Fraud Intelligence            - Blacklist & Greylist management    │
│  ├──  Whitelist Review Workflow     - Human-in-the-loop approval         │
│  ├──  Fraud Scenario Simulator      - Replay real fraud chains with      │
│  │                                    rule-engine live testing           │
│  └──  Ethical AI Dashboard          - Transparency & privacy controls    │
├──────────────────────────────────────────────────────────────────────────┤
│  Student Risk Module (Task 1)    │  Fraud Detection Module (Task 2)      │
│  ├── Feature Engineering         │  ├── Feature Engineering              │
│  ├── Risk Triangle Scorer        │  ├── 7-Rule Engine (Swiss Cheese)     │
│  │   (Identity × Exposure        │  │   R1 Simbox · R2 Wangiri ·         │
│  │    × Behavior)                │  │   R3 Burner · R4 Student Hunter ·  │
│  ├── K-Means Clustering          │  │   R5 Device Hopper · R6 Smishing · │
│  │   (4 Personas)                │  │   R7 Short Burst                   │
│  └── Portrait Generator          │  ├── XGBoost + Isolation Forest       │
│                                  │  └── Fraud Persona Clustering         │
├──────────────────────────────────────────────────────────────────────────┤
│  AI Services                                                             │
│  ├── Groq LLM — Personalized intervention script generation              │
│  └── SHAP-style Explainer — Feature contribution visualization           │
├──────────────────────────────────────────────────────────────────────────┤
│  Privacy & Ethics Stack                                                  │
│  ├── Differential Privacy (ε=5.0 / ε=1.0 Laplace noise on aggregates)    │
│  ├── PII Masking on all displayed data                                   │
│  └── Human-in-the-Loop for Greylist → Whitelist promotion                │
└──────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer            | Technologies                                                      |
| ---------------- | ----------------------------------------------------------------- |
| **Frontend**     | Streamlit, Plotly, pyvis (vis.js), tom-select                     |
| **ML / AI**      | XGBoost, Isolation Forest, scikit-learn, SHAP, K-Means            |
| **LLM**         | Groq API (intervention script generation)                         |
| **Privacy**      | IBM diffprivlib (Differential Privacy), custom PII masking        |
| **Data**         | pandas, NumPy                                                     |
| **Deployment**   | Docker, Docker Compose                                            |
| **Language**     | Python 3.11                                                       |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone and navigate
cd Wutong-Defense-Console-A-Closed-Loop-Campus-Anti-Fraud-System

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

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

- **Risk Triangle Scoring**: Three-layer assessment — Identity Vulnerability (age, student type) → Threat Exposure (unknown/overseas/mainland calls) → Risky Behavior (pick-up rate, callback, call duration). Each layer scores 0–100 and feeds into a composite risk score.
- **Cross-Task Threat Intelligence**: Cross-references every student's call history against the confirmed fraud MSISDN database produced by Task 2, creating a genuine closed-loop between detection and protection.
- **K-Means Persona Clustering (4 Personas)**: The Naive Freshman, The Connected Elder, The Silent Victim, and more — each persona drives targeted education strategies.
- **Explainable Reasons**: Every risk score comes with human-readable, per-student explanations (e.g., *"Age 19 + 12 mainland calls + answered 3 known-fraud numbers"*).

### Fraud Detection Module (Task 2)

- **7-Rule Swiss Cheese Engine**: Multiple overlapping rules ensure no single point of failure:
  - **R1 Simbox** — Low call diversity bots
  - **R2 Wangiri** — Outbound-only silent broadcasters
  - **R3 Prepaid Burner** — Disposable SIMs with burst activity
  - **R4 Student Hunter** — Numbers disproportionately targeting students
  - **R5 Device Hopper** — Repeated IMEI changes to evade bans
  - **R6 Smishing Bot** — High SMS volume with zero voice calls
  - **R7 Short Burst** — Ultra-short calls at high volume (robocallers)
- **Hybrid ML Layer**: XGBoost (supervised) + Isolation Forest (unsupervised) catch "smart" fraud that evades hand-crafted rules.
- **3-Tier Classification**: BLACKLIST (auto-block) → GREYLIST (human review) → WHITELIST (cleared).
- **Fraud Persona Clustering**: The Robocall Factory, The Campus Predator, etc. — actionable profiles for law enforcement.

### Frontend Dashboard Features

- 📊 **Real-time Dashboard** — KPI cards with Differential Privacy–protected aggregate counts
- 👤 **Student Detail Lookup** — Individual risk profiles, SHAP-style feature contribution charts, and one-click Groq LLM intervention script generation
- 🎮 **Live Risk Simulator** — Adjust student parameters (age, call counts, etc.) and watch the Risk Triangle Score change in real time
- 🕸️ **Network Visualization** — Interactive fraud ↔ student connection graph powered by vis.js / pyvis
- 🚨 **Fraud Intelligence Panel** — Browse the Blacklist & Greylist with rule-level justification
- ✅ **Whitelist Review Workflow** — Human-in-the-loop approval flow for Greylist numbers
- 🎯 **Fraud Scenario Simulator** — Replay real fraud event chains (Contact → Detection → Alert → Intervention → Outcome) against the live rule engine
- 🔒 **Ethical AI Dashboard** — Displays implemented privacy principles: Differential Privacy, PII masking, SHAP explainability, and human oversight

---

## 📊 Results

### Task 1: Student Risk

- **5,240** HIGH-RISK students identified (9.1% of the student population)
- **4 Personas** discovered via K-Means clustering: Naive Freshman, Connected Elder, Silent Victim, and more
- Each student receives a composite Risk Triangle Score (0–100) with sub-scores for Identity, Exposure, and Behavior

### Task 2: Fraud Detection

- **7-Rule Engine** catches known fraud patterns with transparent, per-number justifications
- **Hybrid ML Model** (XGBoost + Isolation Forest) detects previously unseen fraud variants
- **3-Tier Classification**: BLACKLIST (auto-block) → GREYLIST (manual review) → WHITELIST (cleared)
- Fraud persona clusters provide actionable intelligence for network-level takedowns

### Task 3: Product Vulnerability

- Feature-level analysis identifies which telecom products and plans are disproportionately exploited by fraudsters
- Risk distribution charts visualize product-level exposure for business decision-making

---

## 🔒 Privacy & Ethics

| Principle                | Implementation                                                                                       |
| ------------------------ | ---------------------------------------------------------------------------------------------------- |
| **Differential Privacy** | IBM diffprivlib with ε=5.0 for model training; ε=1.0 Laplace noise on all dashboard aggregate counts |
| **PII Masking**          | Phone numbers, names, and IDs are masked in all UI displays to prevent data leakage                  |
| **SHAP Explainability**  | Feature contribution bar charts explain every risk score — no black-box decisions                     |
| **Human-in-the-Loop**    | Greylist numbers require manual operator review before promotion to Whitelist or escalation           |
| **Audit Trail**          | Rule IDs and justification text are stored alongside every classification decision                    |

---

## 👥 Team
Daniel Koh Yu Hang,
Daniel Ng Tang Ni,
Ong Shun Yee,
Pang Yu Chen,
Jared Lee Chia Yang

---
