# Paths
DATA_PATH = "../data"
MODELS_PATH = "../checkpoints"

# Define fixed hyperparameters
VAL_PART = 0.95
LR = 0.001
EPOCHS = 10
BATCH_SIZE = 64
NUM_WORKERS = 4

# Column names
X_LABEL = 'preprocessed_sentences'
X_INIT_LABEL = 'Description'
Y_LABEL = 'Class Index'