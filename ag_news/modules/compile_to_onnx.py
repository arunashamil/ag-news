import fire
import numpy as np
import onnxruntime
import torch

from ag_news.modules.constants import MODELS_PATH, ONNX_PATH
from ag_news.modules.trainer import TextClassifier


def main(checkpoint_name: str) -> None:
    module = TextClassifier.load_from_checkpoint(f"{MODELS_PATH}/{checkpoint_name}")
    print("Successfully loaded from checkpoints")
    module.eval()

    vocab_size = module.hparams.vocab_size
    input_array = torch.randint(0, vocab_size, (1, 128))

    module.to_onnx(ONNX_PATH, input_array, export_params=True)

    print("Model compiled to ONNX")

    ort_session = onnxruntime.InferenceSession(ONNX_PATH)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: input_array.numpy().astype(np.int64)}
    ort_outs = ort_session.run(None, ort_inputs)

    if ort_outs:
        print("ONNX model check passed")


if __name__ == "__main__":
    fire.Fire(main)
