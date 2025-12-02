# train.py
import argparse
import os
import json
import numpy as np
import datetime

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
        "gae_lambda": args.gae_lambda,
        "max_grad_norm": args.max_grad_norm,
        "lstm_dropout": args.lstm_dropout,
        "net_arch": dict(pi=[64],vf=[64],lstm=[256])  # you may pass dict like {"pi":[64,64],"vf":[64,64]}
    }

    os.makedirs(args.save_dir, exist_ok=True)

    # determine tensorboard log folder
    if args.resume_tensorboard_log:
        # continue logging to previous folder
        tensorboard_log = args.resume_tensorboard_log
        if not os.path.exists(tensorboard_log):
            raise ValueError(f"Resume tensorboard log path does not exist: {tensorboard_log}")
    else:
        # new run
        log_subfolder = args.run_name if args.run_name else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_log = os.path.join(args.save_dir, "tensorboard", log_subfolder)
        os.makedirs(tensorboard_log, exist_ok=True)

    env = build_vec_env(args.train_csv, n_envs=args.n_envs, seed=args.seed,
                        initial_amount=args.initial_amount, hmax=args.hmax,
                        transaction_cost_pct=args.transaction_cost_pct,
                        reward_scaling=args.reward_scaling)


        # enforce exclusive choice for critic LSTM mode
    if args.shared_lstm and args.enable_critic_lstm:
        raise ValueError("shared_lstm and enable_critic_lstm cannot both be True. "
                         "Choose either shared LSTM (default) or a separate critic LSTM.")

    policy_kwargs = dict(
        lstm_hidden_size=hp["lstm_hidden_size"],
        n_lstm_layers=hp["n_lstm_layers"],
        # lstm_dropout=hp["lstm_dropout"],
        # explicit flags expected by sb3_contrib's recurrent policies
        shared_lstm = bool(args.shared_lstm),
        enable_critic_lstm = bool(args.enable_critic_lstm),
        net_arch = hp["net_arch"]
    )

    # configure SB3 logger to also print to stdout + TB dir
    new_logger = configure(tensorboard_log, ["stdout", "tensorboard"])

    # load existing model or create new
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        model = RecurrentPPO.load(
            args.load_model,
            env=env,
            tensorboard_log=tensorboard_log
        )
        model.set_logger(new_logger)
        reset_timesteps = False
    else:
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
            gae_lambda=hp["gae_lambda"],
            max_grad_norm=hp["max_grad_norm"],
            # lstm_dropout=hp["lstm_dropout"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log
        )
        model.set_logger(new_logger)
        reset_timesteps = True

    callback = TensorboardPnlCallback(
    verbose=1,
    log_freq=args.log_freq,
    initial_amount=args.initial_amount)

    print("Starting learn()")

    model.learn(total_timesteps=hp["total_timesteps"], callback=callback, reset_num_timesteps=reset_timesteps)

    model.save(os.path.join(args.save_dir, "final_model.zip"))
    print("Model saved to", os.path.join(args.save_dir, "final_model.zip"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True, help="path to training CSV (15m candles)")
    p.add_argument("--save_dir", type=str, default="./models", help="where to save model + TB logs")
    p.add_argument("--total_timesteps", type=int, default=200000)
    p.add_argument("--n_steps", type=int, default=2048, help="rollout length; smaller values often used for RNNs")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--lstm_dropout", type=float, default=0.2)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--ent_coef", type=float, default=0.005)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--lstm_hidden_size", type=int, default=256)
    p.add_argument("--n_lstm_layers", type=int, default=1)
    p.add_argument("--run_name", type=str, default=None, help="optional run name for tensorboard folder")
    p.add_argument("--load_model", type=str, default=None, help="Path to saved model to continue training")
    p.add_argument("--resume_tensorboard_log", type=str, default=None,
                   help="Path to previous tensorboard log folder to continue same run")

    # LSTM mode flags:
    # By default we use a shared LSTM (actor + critic share the same LSTM).
    # Use --disable_shared_lstm to disable that (actor-only LSTM), and
    # use --enable_critic_lstm to create a separate LSTM for the critic.

    p.add_argument("--shared_lstm", dest="shared_lstm", action="store_true",
                   help="Use a shared LSTM for actor + critic (default).")
    p.add_argument("--disable_shared_lstm", dest="shared_lstm", action="store_false",
                   help="Disable shared LSTM (use actor-only LSTM unless --enable_critic_lstm is set).")
    p.set_defaults(shared_lstm=True)

    p.add_argument("--enable_critic_lstm", action="store_true", default=False,
                   help="Create a separate LSTM for the critic (mutually exclusive with shared_lstm).")


    p.add_argument("--n_envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--initial_amount", type=float, default=1000.0)
    p.add_argument("--hmax", type=int, default=10)
    p.add_argument("--transaction_cost_pct", type=float, default=0.001)
    p.add_argument("--reward_scaling", type=float, default=1.0)
    p.add_argument("--log_freq", type=int, default=500)
    args = p.parse_args()
    main(args)
