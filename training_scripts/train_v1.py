# train.py
import argparse
import os
import json
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
    # hyperparams (you can change here or pass JSON file)
    hp = {
        "total_timesteps": args.total_timesteps,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "clip_range": args.clip_range,
        "lstm_hidden_size": args.lstm_hidden_size,
        "n_lstm_layers": args.n_lstm_layers,
        "shared_lstm": args.shared_lstm,
        "net_arch": None,  # you may pass dict like {"pi":[64,64],"vf":[64,64]}
    }

    os.makedirs(args.save_dir, exist_ok=True)
    tensorboard_log = os.path.join(args.save_dir, "tensorboard")

    env = build_vec_env(args.train_csv, n_envs=args.n_envs, seed=args.seed,
                        initial_amount=args.initial_amount, hmax=args.hmax,
                        transaction_cost_pct=args.transaction_cost_pct,
                        reward_scaling=args.reward_scaling)

    # policy kwargs for Recurrent actor-critic policy
    policy_kwargs = dict(
        lstm_hidden_size=hp["lstm_hidden_size"],
        n_lstm_layers=hp["n_lstm_layers"],
        shared_lstm=hp["shared_lstm"],
        net_arch=hp["net_arch"],
    )

    # configure SB3 logger to also print to stdout + TB dir
    new_logger = configure(tensorboard_log, ["stdout", "tensorboard"])

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        verbose=1,
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        n_epochs=hp["n_epochs"],
        gamma=hp["gamma"],
        learning_rate=hp["learning_rate"],
        ent_coef=hp["ent_coef"],
        vf_coef=hp["vf_coef"],
        clip_range=hp["clip_range"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
    )
    model.set_logger(new_logger)

    callback = TensorboardPnlCallback(verbose=1, log_freq=args.log_freq)

    print("Starting learn()")
    model.learn(total_timesteps=hp["total_timesteps"], callback=callback)
    model.save(os.path.join(args.save_dir, "final_model.zip"))
    print("Model saved to", os.path.join(args.save_dir, "final_model.zip"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True, help="path to training CSV (15m candles)")
    p.add_argument("--save_dir", type=str, default="./models", help="where to save model + TB logs")
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--n_steps", type=int, default=64, help="rollout length; smaller values often used for RNNs")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--lstm_hidden_size", type=int, default=256)
    p.add_argument("--n_lstm_layers", type=int, default=1)
    p.add_argument("--shared_lstm", type=bool, default=True)
    p.add_argument("--n_envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--initial_amount", type=float, default=1000.0)
    p.add_argument("--hmax", type=int, default=10)
    p.add_argument("--transaction_cost_pct", type=float, default=0.001)
    p.add_argument("--reward_scaling", type=float, default=1.0)
    p.add_argument("--log_freq", type=int, default=500)
    args = p.parse_args()
    main(args)
