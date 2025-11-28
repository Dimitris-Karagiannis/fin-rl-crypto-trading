# testing_scripts/test_v1.py
import os
import argparse
import pandas as pd
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from sb3_contrib import RecurrentPPO

from env.env_v1 import make_env_from_csv

def build_vec_env(csv_path, n_envs=1, **env_kwargs):
    def _make():
        env = make_env_from_csv(csv_path, **env_kwargs)
        return env
    return DummyVecEnv([_make for _ in range(n_envs)])

def main(args):
    # ensure output folders
    os.makedirs(args.model_dir, exist_ok=True)
    testing_results_dir = os.path.join(args.model_dir, "testing_results")
    os.makedirs(testing_results_dir, exist_ok=True)

    tensorboard_test_dir = os.path.join(args.model_dir, "tensorboard_test")
    os.makedirs(tensorboard_test_dir, exist_ok=True)

    # load environment
    env = build_vec_env(args.test_csv, n_envs=1,
                        initial_amount=args.initial_amount,
                        hmax=args.hmax,
                        transaction_cost_pct=args.transaction_cost_pct,
                        reward_scaling=args.reward_scaling)

    # configure logger
    tb_logger = configure(tensorboard_test_dir, ["stdout", "tensorboard"])

    # load trained model
    model = RecurrentPPO.load(args.model_path, env=env)
    model.set_logger(tb_logger)

    obs = env.reset()
    done = False
    step = 0
    records = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        info_data = info[0] if isinstance(info, list) else info

        # record data
        record = {
            "step": step,
            "timestamp": info_data.get("timestamp"),
            "account_value": info_data.get("account_value"),
            "cash": info_data.get("cash"),
            "shares_held": info_data.get("shares_held"),
            "reward": reward[0] if isinstance(reward, np.ndarray) else reward
        }
        records.append(record)

        # log to TensorBoard
        model.logger.record("test/account_value", record["account_value"])
        model.logger.record("test/cash", record["cash"])
        model.logger.record("test/shares_held", record["shares_held"])
        model.logger.record("test/reward", record["reward"])
        model.logger.dump(step)

        step += 1

    # save results CSV
    results_df = pd.DataFrame(records)
    csv_path = os.path.join(testing_results_dir, "test_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Test results saved to {csv_path}")

    # optional render of last step
    if args.render:
        env.envs[0].render()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .zip")
    parser.add_argument("--model_dir", type=str, required=True, help="Parent folder of model (used for saving test results and TB logs)")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--initial_amount", type=float, default=1000.0)
    parser.add_argument("--hmax", type=int, default=10)
    parser.add_argument("--transaction_cost_pct", type=float, default=0.001)
    parser.add_argument("--reward_scaling", type=float, default=1.0)
    parser.add_argument("--render", action="store_true", help="Render last step")
    args = parser.parse_args()

    main(args)
