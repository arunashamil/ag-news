import fire
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from trainer import TextClassificationModel
from load_data import load_test_data

from constants import (VAL_PART, LR, EPOCHS, BATCH_SIZE, DATA_PATH, MODELS_PATH)

def main(test_dir: str, checkpoint_name: str) -> None:
    test_csv = f"{DATA_PATH}/{test_dir}"
    test_dataset = load_test_data(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
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