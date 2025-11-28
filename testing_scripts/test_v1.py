# test.py
import argparse
import numpy as np
import pandas as pd
from env.env_v1 import make_env_from_csv
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

def build_single_env(csv_path, **env_kwargs):
    def _make():
        return make_env_from_csv(csv_path, **env_kwargs)
    return DummyVecEnv([_make])

def evaluate(model_path, test_csv, save_path=None, env_kwargs=None):
    env_kwargs = env_kwargs or {}
    env = build_single_env(test_csv, **env_kwargs)
    model = RecurrentPPO.load(model_path, env=env)

    obs = env.reset()
    lstm_states = None
    episode_account_values = []
    done = False
    while True:
        # RecurrentPPO's predict() expects lstm_states and episode_start for recurrent inference
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=None, deterministic=True)
        obs, reward, done_arr, infos = env.step(action)
        # env is VecEnv: done_arr is array
        if done_arr[0]:
            # finalize
            if infos and isinstance(infos, list) and len(infos) > 0:
                account = infos[0].get("account_value", None)
                if account is not None:
                    episode_account_values.append(account)
            break
        if infos and isinstance(infos, list) and len(infos) > 0:
            ac = infos[0].get("account_value", None)
            if ac is not None:
                episode_account_values.append(ac)

    df = pd.DataFrame({"account_value": episode_account_values})
    if save_path:
        df.to_csv(save_path, index=False)
        print("Saved evaluation account_value series to", save_path)
    print("Final account value:", df["account_value"].iloc[-1])
    print("PnL (final - initial):", df["account_value"].iloc[-1] - float(env_kwargs.get("initial_amount", 1000.0)))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--save_path", default="./test_account_values.csv")
    p.add_argument("--initial_amount", type=float, default=1000.0)
    p.add_argument("--hmax", type=int, default=10)
    args = p.parse_args()
    evaluate(args.model_path, args.test_csv, save_path=args.save_path, env_kwargs={"initial_amount": args.initial_amount, "hmax": args.hmax})
