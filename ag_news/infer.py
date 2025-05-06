import fire
import pandas as pd
import pytorch_lightning as pl

from trainer import TextClassificationModel
from dataloaders_preproc import get_test_dataloader_after_preprocess

from constants import (LR, DATA_PATH, MODELS_PATH, VOCAB_PATH)

def main(test_dir: str, checkpoint_name: str) -> None:

    test_csv = f"{DATA_PATH}/{test_dir}"
    test_df = pd.read_csv(test_csv)

    test_loader = get_test_dataloader_after_preprocess(test_df, VOCAB_PATH)
    module = TextClassificationModel.load_from_checkpoint(f"{MODELS_PATH}/{checkpoint_name}", lr=LR)

    trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10,
        )

    test_results = trainer.test(module, dataloaders=test_loader)
    print(f"Final Test Accuracy: {test_results[0]['test_acc']:.4f}")

if __name__ == "__main__":
    fire.Fire(main)