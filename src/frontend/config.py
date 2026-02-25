"""
Wutong Defense Console - Configuration
"""
import os
from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent.parent

# Try to load .env file from project root
try:
    from dotenv import load_dotenv
    env_path = BASE_DIR / ".env"
    load_dotenv(env_path)
    print(f"Loaded .env from: {env_path}")
except ImportError:
    pass  # dotenv not installed, use system env vars

# API Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Data Paths
DATA_DIR = BASE_DIR / "Datasets"

STUDENT_PREDICTIONS_PATH = DATA_DIR / "Student" / "Results" / "student_risk_predictions.csv"
STUDENT_RAW_PATH = DATA_DIR / "Student" / "Training and Testing Data" / "student_model.csv"
BLACKLIST_PATH = DATA_DIR / "Fraud" / "Results" / "blacklist.csv"
GREYLIST_PATH = DATA_DIR / "Fraud" / "Results" / "greylist.csv"
FRAUD_PROFILES_PATH = DATA_DIR / "Fraud" / "Results" / "fraud_profiles.csv"

# Cache settings
CACHE_TTL = 300  # 5 minutes
