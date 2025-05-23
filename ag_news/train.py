import hydra
import pandas as pd
import pytorch_lightning as pl
from dataloaders import get_dataloaders_after_preprocess
from download_data import download_and_unzip_from_gdrive
from logging_utils import get_git_commit, save_metrics
from model_selector import get_model
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from trainer import TextClassifier


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    url = config["data_load"]["url"]
    download_and_unzip_from_gdrive(url, output_dir=config["data_load"]["data_path"])

    train_df = pd.read_csv(config["data_load"]["train_data_path"])

    vocab, train_loader, val_loader = get_dataloaders_after_preprocess(
        train_df, config["data_load"]["vocab_path"]
    )

    vocab_size = len(vocab.stoi) + 1

    loggers = [
        pl.loggers.WandbLogger(
            project=config["logging"]["project"],
            name=config["logging"]["name"],
            save_dir=config["logging"]["save_dir"],
            config={
                "val_part": config["training"]["val_part"],
                "lr": config["training"]["lr"],
                "num_epochs": config["training"]["num_epochs"],
                "batch_size": config["training"]["batch_size"],
                "num_workers": config["training"]["num_workers"],
                "git_commit": get_git_commit(),
            },
        )
    ]

    model = get_model(vocab_size, config["model"])
    module = TextClassifier(model, lr=config["training"]["lr"])

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=2),
    ]

    callbacks.append(
        ModelCheckpoint(
            dirpath=config["model"]["model_local_path"],
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=config["model"]["save_top_k"],
            every_n_epochs=config["model"]["every_n_epochs"],
        )
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(module, train_loader, val_loader)

    save_metrics(config["logging"]["api_run"], config["logging"]["name"])


if __name__ == "__main__":
    main()
