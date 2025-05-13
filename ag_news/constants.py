# Paths
DATA_PATH = "../data/ag_news_data"
MODELS_PATH = "../models"
VOCAB_PATH = "../data/vocab_data/vocab.json"

# Data info
NUM_CLASSES = 4

# Define fixed hyperparameters
VAL_PART = 0.95
LR = 0.001
EPOCHS = 3
BATCH_SIZE = 64
NUM_WORKERS = 4

# Column names
X_LABEL = "preprocessed_sentences"
X_INIT_LABEL = "Description"
Y_LABEL = "Class Index"
