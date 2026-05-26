"""
Configuration for AI Recommendation Module.
All paths, hyperparameters, and constants in one place.
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Online_Retail.xlsx")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
RULES_CSV_PATH = os.path.join(SAVED_MODELS_DIR, "association_rules.csv")
MAPPINGS_PATH = os.path.join(SAVED_MODELS_DIR, "mappings.pkl")
POPULAR_PATH = os.path.join(SAVED_MODELS_DIR, "popular_items.pkl")

# Ensure dirs exist
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Random seed
SEED = 42

# Data split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Apriori
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.5
MIN_LIFT = 1.5

# AutoEncoder
AE_EPOCHS = 80
AE_BATCH_SIZE = 64
AE_LR = 0.001
AE_DROPOUT = 0.1
AE_PATIENCE = 10

# NCF
NCF_EPOCHS = 25
NCF_BATCH_SIZE = 2048
NCF_LR = 0.001
NCF_EMBED_DIM = 128
NCF_NEG_RATIO = 2

# LSTM
LSTM_EPOCHS = 40
LSTM_BATCH_SIZE = 1024
LSTM_LR = 0.001
LSTM_SEQ_LEN = 10
LSTM_UNITS = 128

# Ensemble weights (sum to 1.0); AE downweighted until it improves
ENSEMBLE_WEIGHT_AE = 0.10
ENSEMBLE_WEIGHT_NCF = 0.35
ENSEMBLE_WEIGHT_LSTM = 0.55

# Evaluation
TOP_K_VALUES = [5, 10]

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_RECOMMENDATIONS = 10
