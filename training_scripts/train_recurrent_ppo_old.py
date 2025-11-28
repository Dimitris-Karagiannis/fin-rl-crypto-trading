# train_recurrent_ppo.py
import os
import argparse
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn
from env.crypto_env_continuous_old import create_crypto_env
from callbacks.tb_callback_old import TensorboardBTCLogger

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, default="rppo_btc_15m", help="experiment name for logs/models")
    p.add_argument("--timesteps", type=int, default=200_000, help="total timesteps to train")
    p.add_argument("--initial_capital", type=float, default=1000.0, help="starting capital")
    p.add_argument("--lstm_hidden_size", type=int, default=256, help="LSTM hidden size")
    p.add_argument("--n_lstm_layers", type=int, default=1, help="Number of LSTM layers")
    p.add_argument("--net", nargs="+", type=int, default=[128,128], help="MLP net arch BEFORE LSTM, e.g. --net 128 128")
    p.add_argument("--n_steps", type=int, default=1024, help="n_steps (rollout length) — tune this")
    return p.parse_args()

def main():
    args = parse_args()
    exp = args.exp_name
    total_timesteps = args.timesteps

    # create dirs
    logdir = os.path.join("logs", exp)
    modeldir = os.path.join("models", exp)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(modeldir, exist_ok=True)

    # create vectorized env (single env wrapped)
    env = DummyVecEnv([lambda: create_crypto_env(csv_path="data/btc_usdt_15m_2017_2023.csv",
                                                 initial_capital=args.initial_capital)])

    # policy kwargs for RecurrentPPO LSTM policy
    policy_kwargs = dict(
        lstm_hidden_size=args.lstm_hidden_size,
        n_lstm_layers=args.n_lstm_layers,
        shared_lstm=False,                # actor/critic do not share LSTM; change if desired
        net_arch=list(args.net),          # small MLP before LSTM
        activation_fn=nn.Tanh
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=64,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=logdir,
        policy_kwargs=policy_kwargs
    )

    callback = TensorboardBTCLogger()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # save model
    save_path = os.path.join(modeldir, "recurrent_ppo_btc")
    model.save(save_path)
    print(f"Training finished. Model saved to: {save_path}")

if __name__ == "__main__":
    main()
