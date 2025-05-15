import matplotlib.pyplot as plt
from wandb.apis.public import Api


def save_plot(epochs, metrics, label, savepath):
    plt.plot(epochs, metrics, label=label)
    plt.legend()
    plt.savefig(savepath)
    plt.close()


def save_metrics(api_run, exp_name):
    api = Api()
    run = api.runs(api_run)[0]
    history = run.scan_history()

    epochs = [row["epoch"] for row in history if "epoch" in row]
    global_step = [row["_step"] for row in history if "_step" in row]
    train_loss = [row["train_loss"] for row in history if "train_loss" in row]
    val_loss = [row["val_loss"] for row in history if "val_loss" in row]
    val_acc = [row.get("val_acc") for row in history if "val_acc" in row]

    save_plot(
        global_step,
        train_loss,
        "Train Loss",
        "../plots/" + exp_name + "_train_loss.png",
    )

    save_plot(epochs, val_loss, "Val Loss", "../plots/" + exp_name + "_val_loss.png")

    if val_acc:
        save_plot(
            epochs, val_acc, "Val Accuracy", "../plots/" + exp_name + "_val_acc.png"
        )


def get_git_commit():
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except (subprocess.CalledProcessError, UnicodeError):
        return "unknown"
