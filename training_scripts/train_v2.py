# training_scripts/train_v1.py
import argparse
import os
import datetime
import yaml
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.logger import configure

from env.env_v1 import make_env_from_csv
from callbacks.callback_v1 import TensorboardPnlCallback

def build_vec_env(csv_path, n_envs=1, seed=0, **env_kwargs):
    def _make():
        env = make_env_from_csv(csv_path, **env_kwargs)
        return env
    set_random_seed(seed)
    envs = DummyVecEnv([_make for _ in range(n_envs)])
    return envs

def main(args):
    # Load hyperparameters from YAML
    if args.config_yaml:
        with open(args.config_yaml, "r") as f:
            hp = yaml.safe_load(f)
    else:
        # fallback to CLI args
        hp = vars(args)

    os.makedirs(hp["save_dir"], exist_ok=True)

    # TensorBoard logging folder
    if hp.get("resume_tensorboard_log"):
        tensorboard_log = hp["resume_tensorboard_log"]
        if not os.path.exists(tensorboard_log):
            raise ValueError(f"Resume tensorboard log path does not exist: {tensorboard_log}")
    else:
        log_subfolder = hp.get("run_name") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_log = os.path.join(hp["save_dir"], "tensorboard", log_subfolder)
        os.makedirs(tensorboard_log, exist_ok=True)

    # Build environment
    env = build_vec_env(hp["train_csv"], n_envs=hp.get("n_envs", 1), seed=hp.get("seed", 0),
                        initial_amount=hp.get("initial_amount", 1000),
                        hmax=hp.get("hmax", 10),
                        transaction_cost_pct=hp.get("transaction_cost_pct", 0.001),
                        reward_scaling=hp.get("reward_scaling", 1.0))

    # enforce exclusive LSTM flags
    if hp.get("shared_lstm", True) and hp.get("enable_critic_lstm", False):
        raise ValueError("shared_lstm and enable_critic_lstm cannot both be True.")

    policy_kwargs = dict(
        lstm_hidden_size=hp.get("lstm_hidden_size", 256),
        n_lstm_layers=hp.get("n_lstm_layers", 1),
        shared_lstm=bool(hp.get("shared_lstm", True)),
        enable_critic_lstm=bool(hp.get("enable_critic_lstm", False)),
        net_arch=hp.get("net_arch")
    )

    # Configure SB3 logger
    new_logger = configure(tensorboard_log, ["stdout", "tensorboard"])

    # Load existing model or create new
    if hp.get("load_model"):
        print(f"Loading model from {hp['load_model']}")
        model = RecurrentPPO.load(hp["load_model"], env=env, tensorboard_log=tensorboard_log)
        model.set_logger(new_logger)
        reset_timesteps = False
    else:
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=env,
            verbose=1,
            n_steps=hp.get("n_steps", 2048),
            batch_size=hp.get("batch_size", 512),
            n_epochs=hp.get("n_epochs", 10),
            gamma=hp.get("gamma", 0.99),
            gae_lambda=hp.get("gae_lambda", 0.95),
            learning_rate=float(hp.get("learning_rate", 3e-5)),
            ent_coef=hp.get("ent_coef", 0.005),
            vf_coef=hp.get("vf_coef", 0.5),
            clip_range=hp.get("clip_range", 0.2),
            max_grad_norm=hp.get("max_grad_norm", 0.5),
            # lstm_dropout=hp.get("lstm_dropout", 0.2),
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log
        )
        model.set_logger(new_logger)
        reset_timesteps = True

    callback = TensorboardPnlCallback(
        verbose=1,
        log_freq=hp.get("log_freq", 500),
        initial_amount=hp.get("initial_amount", 1000)
    )

    print("Starting learn()")
    model.learn(total_timesteps=hp["total_timesteps"], callback=callback, reset_num_timesteps=reset_timesteps)
    model.save(os.path.join(hp["save_dir"], "final_model.zip"))
    print("Model saved to", os.path.join(hp["save_dir"], "final_model.zip"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()
    main(args)
