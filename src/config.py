"""
Wutong Cup Solution - Configuration

Central configuration for all paths and thresholds.
Edit this file to change locations or tuning parameters.
"""

from pathlib import Path

# ============================================================
# BASE PATHS
# ============================================================
BASE_DIR = Path(__file__).parent.parent  # Solution/

# Data directories
FRAUD_DATA_DIR = BASE_DIR / "Datasets" / "Fraud" / "Training and Testing Data"
STUDENT_DATA_DIR = BASE_DIR / "Datasets" / "Student" / "Training and Testing Data"

# Results directories
FRAUD_RESULTS_DIR = BASE_DIR / "Datasets" / "Fraud" / "Results"
STUDENT_RESULTS_DIR = BASE_DIR / "Datasets" / "Student" / "Results"

# ============================================================
# DATA FILES
# ============================================================
# Task 2: Fraud Detection
FRAUD_TRAIN_PATH = FRAUD_DATA_DIR / "fraud_model_2.csv"
FRAUD_MINE_1_PATH = FRAUD_DATA_DIR / "fraud_model_1_1.csv"
FRAUD_MINE_2_PATH = FRAUD_DATA_DIR / "fraud_model_1_2.csv"
FRAUD_VALIDATE_PATH = FRAUD_DATA_DIR / "validate_data.csv"

# Task 1: Student Risk
STUDENT_DATA_PATH = STUDENT_DATA_DIR / "student_model.csv"

# ============================================================
# OUTPUT FILES
# ============================================================
BLACKLIST_PATH = FRAUD_RESULTS_DIR / "blacklist.csv"
GREYLIST_PATH = FRAUD_RESULTS_DIR / "greylist.csv"
STUDENT_PREDICTIONS_PATH = STUDENT_RESULTS_DIR / "student_risk_predictions.csv"
STUDENT_PORTRAIT_PATH = STUDENT_RESULTS_DIR / "high_risk_student_portrait.md"

# ============================================================
# STUDENT RISK SCORING (Task 1)
# ============================================================
# Layer 1: Identity Vulnerability (Removed biased factors)
SCORE_YOUNG_AGE = 5              # Age 18-22

# Layer 2: Threat Exposure
SCORE_MAINLAND_CALLS = 10        # 1-4 calls
SCORE_HIGH_MAINLAND_CALLS = 20   # 5+ calls
SCORE_OVERSEAS_CALLS = 10        # 5-9 calls
SCORE_HIGH_OVERSEAS_CALLS = 20   # 10+ calls
SCORE_FRAUD_CONTACT = 30         # Contacted by known fraud

# Layer 3: Risky Behavior (Multipliers)
MULTIPLIER_ANSWERED = 1.5        # Picked up fraud call
MULTIPLIER_CALLED_BACK = 2.0     # Called back fraud number

# Thresholds
STUDENT_VULNERABLE_THRESHOLD = 40  # score >= 40 = VULNERABLE (lowered from 50)

# ============================================================
# FRAUD DETECTION (Task 2)
# ============================================================
# Rule Engine Thresholds
R1_SIMBOX_CALLS = 50             # Minimum calls for Simbox rule
R1_SIMBOX_DISPERSION = 0.04      # Maximum dispersion rate
R2_WANGIRI_CALLS = 50            # Minimum outgoing for Wangiri
R3_BURNER_DAYS = 7               # Activation days for burner
R4_STUDENT_CALLS = 5             # Minimum student calls

# ML Thresholds
FRAUD_BLACK_THRESHOLD = 0.85     # > 0.85 = BLACK tier
FRAUD_GREY_THRESHOLD = 0.60      # 0.60-0.85 = GREY tier

# Greylist Rule (Passport + Silent)
GREYLIST_CALL_THRESHOLD = 15     # Calls <= 15 = GREY, > 15 = BLACK

# Data Snapshot Date (for activation_days calculation when stat_dt is missing)
# Format: YYYYMMDD - represents the "observation date" for the dataset
DATA_SNAPSHOT_DATE = 20251124

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def ensure_results_dirs():
    """Create results directories if they don't exist."""
    FRAUD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STUDENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
