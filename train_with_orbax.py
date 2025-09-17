import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import joblib
import optax
import orbax.checkpoint as ocp
import wandb
import wandb_osh
from orbax.checkpoint.checkpoint_managers import LatestN
from sklearn.model_selection import train_test_split
from wandb_osh.hooks import TriggerWandbSyncHook

from two_q_sep_cp.neural_ode.model import (
    LossWeights,
    SeparateLosses,
    compute_loss_in_batches,
    get_params,
    make_model,
    simple_update_step,
    update_loss_weights,
)
from two_q_sep_cp.neural_ode.utils import (
    build_restored_model,
    get_abstract_structure,
    initialize_setup,
    make_ckpt_path,
    make_data_loader,
    make_scheduler,
)

# from scalene import scalene_profiler


# scalene_profiler.start()


wandb_osh.set_log_level("ERROR")


# --------------------- CLI and config loader --------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Train script: pass a JSON config via --config (required)"
    )
    p.add_argument(
        "--config", type=str, required=True, help="Path to JSON config file."
    )
    return p.parse_args()


def load_config(path_str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path_str}")
    with open(path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config JSON must be an object/dict at the top level.")
    return cfg


DEFAULT_CONFIG = {
    "depth": 3,
    "width": 128,
    "architecture": "simple_mlp",
    "spam_estimation_path": "results_spam_estimation/q15_q16_250ns_2025-08-05_SPAM_ESTIMATION.job",
    "dataset_path": "datasets/q15_q16_250ns_2025-08-05_FILTERED_DATASET.job",
    "n_epochs": 200,
    "batch_size": 512,
    "model_seed": 0,
    "dataset_window_mask": 0.4,
    "seed_train_test": 0,
    "comment": None,
}


def main():
    args = parse_args()
    config = load_config(args.config)

    # ---------------------------------------------------------------------------- #
    #                                 DATASET SETUP                                #
    # ---------------------------------------------------------------------------- #
    dataset_window_mask = config.get("dataset_window_mask")

    dataset = joblib.load(config.get("dataset_path"))

    X = dataset["X"]  # THIS IN THEORY DOES NOT INCLUDE THE 0 TIME
    Y = dataset["Y"]

    max_time = X[:, 2].max()
    X[:, 2] = X[:, 2] / max_time

    if dataset_window_mask:
        mask = X[:, 2] <= 0.4  # We are gonna learn only short times
        X = X[mask]

        Y = Y[mask]

    seed_train_test = config.get("seed_train_test", 0)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, random_state=seed_train_test
    )
    X_train, X_test, Y_train, Y_test = [
        jnp.array(arr) for arr in [X_train, X_test, Y_train, Y_test]
    ]

    # ---------------------------------------------------------------------------- #
    #                           VARIABLES INITIALIZATION                           #
    # ---------------------------------------------------------------------------- #

    setup, args_loss = initialize_setup(config)
    # ---------------------------------------------------------------------------- #
    #                                OPTIMIZER SETUP                               #
    # ---------------------------------------------------------------------------- #

    BATCH_SIZE = config["batch_size"]

    N_EPOCHS = config["n_epochs"]
    start_lr = config.get("start_learning_rate", 1e-3)
    clip_global_norm = config.get("clip_global_norm", 100)
    max_iterations_per_epoch = max(1, X_train.shape[0] // BATCH_SIZE)
    total_steps = int(
        N_EPOCHS * max_iterations_per_epoch
    )  #  ≈ (#epochs) × (#steps per epoch)
    warmup = total_steps * 0.025  # 2.5% of total steps 1000
    scheduler = make_scheduler(start_lr, warmup, total_steps)

    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_global_norm),
        optax.scale_by_adam(),
        optax.add_decayed_weights(1e-4),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0),
    )

    # ---------------------------------------------------------------------------- #
    #                                  MODEL SETUP                                 #
    # ---------------------------------------------------------------------------- #

    key, model = make_model(config)

    initial_opt_state = optimizer.init(get_params(model))

    opt_state = initial_opt_state

    loss_weights = LossWeights(jnp.array([1, 100], dtype=jnp.float64))

    # ---------------------------------------------------------------------------- #
    #                                  CKPT SETUP                                  #
    # ---------------------------------------------------------------------------- #

    # path = ocp.test_utils.erase_and_create_empty(
    #     "/home/antonio/dev/cesga/qpt/two_qubits_separable_povm/orbax_ckpt/tmp/my_checkpoints"
    # )
    ckpt_path = make_ckpt_path(config)

    abstract_structure = get_abstract_structure(model)

    policy = LatestN(10)

    options = ocp.CheckpointManagerOptions(
        best_fn=lambda m: m["loss_test"],  # pick the metric to rank by
        best_mode="min",
        preservation_policy=policy,
        enable_async_checkpointing=True,
    )

    checkpoint_manager = ocp.CheckpointManager(
        ckpt_path,
        options=options,
        metadata=config,  # we save the config in the metadata too
    )

    # ---------------------------------------------------------------------------- #
    #                                  WANDB INIT                                  #
    # ---------------------------------------------------------------------------- #

    trigger_sync = TriggerWandbSyncHook()

    run_name = Path(ckpt_path).name
    wandb.init(
        project="two_qubits_separable_cp",
        config=config,
        name=run_name,
        save_code=True,
        resume="allow",
        mode="offline",
    )
    # ---------------------------------------------------------------------------- #
    #                                TRAINING STAGE                                #
    # ---------------------------------------------------------------------------- #

    n_update_loss_weights = config.get("n_update_loss_weights", 250)
    key, subkey = jax.random.split(key)

    train_loader = make_data_loader(
        [X_train, Y_train], batch_size=BATCH_SIZE, subkey=subkey
    )

    train_loss_history = []
    test_loss_history = []
    global_step_counter = 0
    with checkpoint_manager as mngr:
        best_step = checkpoint_manager.best_step()
        start_epoch = best_step + 1 if best_step is not None else 0

        if best_step is not None:  # check if it exists a previous save
            restored = mngr.restore(
                step=best_step,
                args=ocp.args.Composite(
                    model_state=ocp.args.StandardRestore(abstract_structure),
                    opt_state=ocp.args.StandardRestore(opt_state),
                ),
            )
            model = build_restored_model(model, restored["model_state"])
            opt_state = restored["opt_state"]

        for epoch in range(start_epoch, N_EPOCHS):
            # epoch_train_loss = 0.0
            # epoch_test_loss = 0.0

            epoch_train_loss = jnp.array(
                0.0,
            )

            epoch_train_separate_losses = SeparateLosses(jnp.array(0.0), jnp.array(0.0))

            epoch_test_loss = jnp.array(0.0)

            for step in range(max_iterations_per_epoch):
                global_step_counter = global_step_counter + 1

                x, y = next(train_loader)

                model, opt_state, (step_train_loss, step_train_separate_losses) = (
                    simple_update_step(
                        model,
                        opt_state,
                        x,
                        y,
                        optimizer=optimizer,
                        loss_weights=loss_weights,
                        args_loss=args_loss,
                    )
                )

                epoch_train_loss = epoch_train_loss + step_train_loss
                epoch_train_separate_losses = (
                    epoch_train_separate_losses + step_train_separate_losses
                )

                if global_step_counter % n_update_loss_weights == 0:
                    loss_weights = update_loss_weights(
                        model, x, y, loss_weights, args_loss
                    )

            epoch_train_loss = epoch_train_loss / (step + 1)
            epoch_train_separate_losses = epoch_train_separate_losses / (step + 1)

            epoch_test_loss, epoch_test_separate_losses = compute_loss_in_batches(
                model, X_test, Y_test, loss_weights, args_loss, batch_size=BATCH_SIZE
            )

            train_loss_history.append(epoch_train_loss)
            test_loss_history.append(epoch_test_loss)

            metrics = {
                "loss_test": float(epoch_test_loss),
                "loss_train": float(epoch_train_loss),
                "nll_loss_test": float(epoch_test_separate_losses.loss_nll),
                "nll_loss_train": float(epoch_train_separate_losses.loss_nll),
                "choi_loss_test": float(epoch_test_separate_losses.loss_choi),
                "choi_loss_train": float(epoch_train_separate_losses.loss_choi),
            }

            wandb.log(
                {
                    "epoch": float(epoch),
                    "loss_test": float(epoch_test_loss),
                    "loss_train": float(epoch_train_loss),
                    "nll_loss_test": float(epoch_test_separate_losses.loss_nll),
                    "nll_loss_train": float(epoch_train_separate_losses.loss_nll),
                    "choi_loss_test": float(epoch_test_separate_losses.loss_choi),
                    "choi_loss_train": float(epoch_train_separate_losses.loss_choi),
                }
            )
            trigger_sync()

            mngr.save(
                step=epoch,
                args=ocp.args.Composite(
                    model_state=ocp.args.StandardSave(
                        eqx.partition(model, eqx.is_array)[0]
                    ),
                    opt_state=ocp.args.StandardSave(opt_state),
                ),
                metrics=metrics,
            )

            # print(
            #     f"Epoch {epoch + 1}/{N_EPOCHS} | Train Loss: {epoch_train_loss:.4e}| Test Loss: {epoch_test_loss:.4e}"
            # )
        mngr.wait_until_finished()

    wandb.finish()

    # scalene_profiler.stop()


# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        traceback.print_exc()
        import sys

        sys.exit(1)
