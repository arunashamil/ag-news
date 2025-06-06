DATA_PATH = "../../data/ag_news_data"
MODELS_PATH = "../../models"
VOCAB_PATH = "../../data/vocab_data/vocab.json"
ONNX_PATH = "../../triton/model_repository/text_classifier/1/model.onnx"


NUM_CLASSES = 4

VAL_PART = 0.05
LR = 0.001
BATCH_SIZE = 64
NUM_WORKERS = 4

X_LABEL = "preprocessed_sentences"
X_INIT_LABEL = "Description"
Y_LABEL = "Class Index"
MAX_PAD_LEN = 90
