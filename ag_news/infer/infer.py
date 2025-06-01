import fire
import pandas as pd
import pytorch_lightning as pl

from ag_news.modules.constants import DATA_PATH, MODELS_PATH, VOCAB_PATH
from ag_news.modules.dataloaders import get_test_dataloader_after_preprocess
from ag_news.modules.trainer import TextClassifier


def main(test_dir: str, checkpoint_name: str) -> None:
    test_csv = f"{DATA_PATH}/{test_dir}"
    test_df = pd.read_csv(test_csv)

    vocab_size, test_loader = get_test_dataloader_after_preprocess(test_df, VOCAB_PATH)
    module = TextClassifier.load_from_checkpoint(f"{MODELS_PATH}/{checkpoint_name}")

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    test_results = trainer.test(module, dataloaders=test_loader)
    print(f"Final Test Accuracy: {test_results[0]['test_acc']:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
