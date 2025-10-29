from ysautoml.network.oneshot import train_dynas

if __name__ == "__main__":
    train_dynas(
        log_dir="logs/spos_dynamic",
        file_name="spos_dynamic",
        method="dynas",
        seed=0,
        epochs=250,
        lr=0.025,
        save_path="./save_dir"
    )
