import json
import re

import numpy as np
import tritonclient.http as httpclient

VOCAB_PATH = "vocab.json"
MAX_PAD_LEN = 90
X_LABEL = "preprocessed_sentences"
UNK_TOKEN = "<unk>"


class TextClassifierClient:
    def __init__(
        self,
        vocab_path,
        max_pad_len,
        model_name="text_classifier",
        url="localhost:8000",
    ):
        self.client = httpclient.InferenceServerClient(url=url)
        self.model_name = model_name
        self.max_pad_len = max_pad_len

        with open(vocab_path, "r") as f:
            self.vocab_dict = json.load(f)

        self.unk_index = self.vocab_dict.get(UNK_TOKEN, 0)

        self.token_to_index = lambda token: self.vocab_dict.get(token, self.unk_index)

    def preprocessing(self, text):
        """Preprocessing sentence"""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def tk(self, text):
        """Tokenization"""
        return text.split()

    def get_tokenized_sentences(self, sentences):
        """Get tokenized sentences"""
        return [self.tk(sentence) for sentence in sentences]

    def pad_num_sentences(self, sequence, max_len, pad_value=0):
        """Get padded sentences"""
        if len(sequence) >= max_len:
            return sequence[:max_len]
        else:
            return sequence + [pad_value] * (max_len - len(sequence))

    def preprocess_text(self, text: str) -> np.ndarray:
        """Preprocessing text"""

        cleaned_text = self.preprocessing(text)

        tokens = self.tk(cleaned_text)

        sequence = [self.token_to_index(token) for token in tokens]

        padded_sequence = self.pad_num_sentences(sequence, self.max_pad_len)

        return np.array([padded_sequence], dtype=np.int64)

    def classify_text(self, text: str) -> dict:
        """Classify text and return results"""

        input_array = self.preprocess_text(text)

        inputs = [
            httpclient.InferInput("PREPROCESSED_TEXT", input_array.shape, "INT64")
        ]
        inputs[0].set_data_from_numpy(input_array)

        outputs = [httpclient.InferRequestedOutput("LOGITS")]

        response = self.client.infer(
            model_name=self.model_name, inputs=inputs, outputs=outputs
        )

        logits = response.as_numpy("LOGITS")

        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()

        predicted_class = int(np.argmax(probabilities))

        return {
            "text": text,
            "preprocessed_text": " ".join(
                [token for token in self.tk(text) if token in self.vocab_dict]
            ),
            "predicted_class": predicted_class,
            "class_probabilities": probabilities.squeeze().tolist(),
            "logits": logits.squeeze().tolist(),
        }


if __name__ == "__main__":
    classifier = TextClassifierClient(vocab_path=VOCAB_PATH, max_pad_len=MAX_PAD_LEN)

    texts = [
        "I admire beauty of the world we live in.",
        "Sport is an absolute phenomena.",
        "Metrics are constantly growing, claims CEO of Apple.",
        "Quantum mechanics is fun for cats.",
    ]

    for text in texts:
        result = classifier.classify_text(text)
        print(f"\nOriginal Text: '{result['text']}'")
        print(f"Preprocessed: '{result['preprocessed_text']}'")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Probabilities: {result['class_probabilities']}")
        print("=" * 80)
