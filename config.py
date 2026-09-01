import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Dataset paths
ENRON_PATH = os.path.join(RAW_DIR, 'enron.csv')
PHISHING_PATH = os.path.join(RAW_DIR, 'phishing.csv')
SPAM_PATH = os.path.join(RAW_DIR, 'spam_email.csv')
SYNTHETIC_PATH = os.path.join(RAW_DIR, 'synthetic.csv')
MERGED_PATH = os.path.join(PROCESSED_DIR, 'merged_dataset.csv')
FINAL_PATH = os.path.join(PROCESSED_DIR, 'final_dataset.csv')

# TF-IDF parameters
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Stage 1 speed requirement
MAX_INFERENCE_MS = 5

# Labels
BENIGN_LABEL = 0
ATTACK_LABEL = 1

# Create directories if they don't exist
for directory in [RAW_DIR, PROCESSED_DIR, SPLITS_DIR,
                  MODELS_DIR, RESULTS_DIR,
                  os.path.join(RESULTS_DIR, 'confusion_matrices'),
                  os.path.join(RESULTS_DIR, 'roc_curves')]:
    os.makedirs(directory, exist_ok=True)

print("Config loaded. All directories verified.")