import logging
import os

from fastapi import FastAPI, HTTPException

from triton import TextClassifierClient

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

VOCAB_PATH = os.getenv("VOCAB_PATH", "vocab.json")
MAX_PAD_LEN = int(os.getenv("MAX_PAD_LEN", "90"))
TRITON_URL = os.getenv("TRITON_URL", "triton-server:8000")

try:
    classifier = TextClassifierClient(
        vocab_path=VOCAB_PATH, max_pad_len=MAX_PAD_LEN, url=TRITON_URL
    )
    logger.info("Classifier initialized successfully")
except Exception as e:
    logger.error(f"Classifier init failed: {str(e)}")
    raise RuntimeError("Classifier initialization failed") from e


@app.get("/")
def home():
    return {"message": "Text Classification API", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "text-classification"}


@app.post("/classify")
async def classify_text(text: str):
    try:
        result = classifier.classify_text(text)
        return {
            "text": text,
            "predicted_class": result["predicted_class"],
            "confidence": max(result["class_probabilities"]),
        }
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/routes")
def list_routes():
    return [{"path": route.path, "method": route.methods} for route in app.routes]
