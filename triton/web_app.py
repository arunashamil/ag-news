import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from triton import TextClassifierClient

app = FastAPI()

VOCAB_PATH = "vocab.json"
MAX_PAD_LEN = 90
X_LABEL = "preprocessed_sentences"
UNK_TOKEN = "<unk>"

classifier = TextClassifierClient(
    vocab_path=VOCAB_PATH, max_pad_len=MAX_PAD_LEN, url=os.getenv("TRITON_URL")
)


class ClassificationRequest(BaseModel):
    text: str


@app.post("/classify")
async def classify_text(request: ClassificationRequest):
    try:
        result = classifier.classify_text(request.text)
        return {
            "text": result["text"],
            "predicted_class": result["predicted_class"],
            "probability": max(result["class_probabilities"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy"}
