import hydra
import pandas as pd
import pytorch_lightning as pl
from dataloaders import get_dataloaders_after_preprocess
from logging_utils import get_git_commit, save_metrics
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from trainer import TextClassificationModel


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    train_df = pd.read_csv(config["data_load"]["train_data_path"])

    vocab, train_loader, val_loader = get_dataloaders_after_preprocess(
        train_df, config["data_load"]["vocab_path"]
    )
    vocab_size = len(vocab.stoi) + 1

    model = TextClassificationModel(
        vocab_size=vocab_size,
        embed_dim=100,
        hidden_dim=128,
        output_dim=config["model"]["num_classes"],
        lr=config["training"]["lr"],
    )

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

    trainer.fit(model, train_loader, val_loader)

    save_metrics(config["logging"]["api_run"], config["logging"]["name"])


if __name__ == "__main__":
    main()
