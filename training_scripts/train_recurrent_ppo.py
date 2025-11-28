import os
import argparse
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn
from env.crypto_env_continuous import create_crypto_env
from callbacks.tb_callback import TensorboardBTCLogger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='rppo_btc_15m', help='experiment name for logs/models')
    parser.add_argument('--timesteps', type=int, default=200_000, help='total timesteps to train')
    parser.add_argument('--initial_capital', type=float, default=1000.0, help='starting capital')
    parser.add_argument('--lstm_hidden_size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--n_lstm_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--net', nargs='+', type=int, default=[128,128], help='MLP net arch before LSTM')
    parser.add_argument('--n_steps', type=int, default=1024, help='rollout length n_steps')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=8, help='number of epochs per update')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_range', type=float, default=0.2, help='PPO clip range')
    parser.add_argument('--ent_coef', type=float, default=0.0, help='entropy coefficient')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='max grad norm')
    return parser.parse_args()

def main():
    args = parse_args()

    # Directories
    logdir = os.path.join('logs', args.exp_name)
    modeldir = os.path.join('models', args.exp_name)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(modeldir, exist_ok=True)

    # Environment
    env = DummyVecEnv([lambda: create_crypto_env(csv_path='data/btc_usdt_15m_2017_2023.csv',
                                                 initial_capital=args.initial_capital)])

    # Policy kwargs
    policy_kwargs = dict(
        lstm_hidden_size=args.lstm_hidden_size,
        n_lstm_layers=args.n_lstm_layers,
        shared_lstm=False,
        net_arch=list(args.net),
        activation_fn=nn.Tanh
    )

    # Recurrent PPO model
    model = RecurrentPPO(
        'MlpLstmPolicy',
        env,
        verbose=1,
        device="cuda",
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log=logdir,
        policy_kwargs=policy_kwargs
    )

    # Callback
    callback = TensorboardBTCLogger()

    # Train
    model.learn(total_timesteps=args.timesteps, callback=callback)

    # Save model
    save_path = os.path.join(modeldir, 'recurrent_ppo_btc')
    model.save(save_path)
    print(f'Training finished. Model saved to: {save_path}')

if __name__ == '__main__':
    main()
