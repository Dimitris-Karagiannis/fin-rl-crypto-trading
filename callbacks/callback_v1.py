# callback_v1.py
import os
from stable_baselines3.common.callbacks import BaseCallback
import datetime

class TensorboardPnlCallback(BaseCallback):
    """
    Logs custom metrics to TensorBoard:
    - account_value (average across VecEnv)
    - pnl = account_value - initial_amount
    - positions (if 'position' key exists in env info)
    - cash (if 'cash' key exists in env info)
    """
    def __init__(self, verbose=0, log_freq=1000, initial_amount=1000.0):
        super().__init__(verbose)
        self.log_freq = int(log_freq)
        self._step = 0
        self.initial_amount = initial_amount

    def _on_step(self) -> bool:
        self._step += 1
        if self._step % self.log_freq != 0:
            return True

        try:
            infos = self.locals.get("infos", None)
            if not infos:
                return True

            # handle VecEnv: infos is a list of dicts
            if isinstance(infos, list):
                # account_value
                avs = [info.get("account_value") for info in infos if info.get("account_value") is not None]
                if avs:
                    mean_av = sum(avs) / len(avs)
                    self.logger.record("train/account_value", float(mean_av))
                    self.logger.record("train/pnl", float(mean_av - self.initial_amount))

                # positions (if available)
                positions = [info.get("position") for info in infos if info.get("position") is not None]
                if positions:
                    mean_pos = sum(positions) / len(positions)
                    self.logger.record("train/position", float(mean_pos))

                # cash (if available)
                cashes = [info.get("cash") for info in infos if info.get("cash") is not None]
                if cashes:
                    mean_cash = sum(cashes) / len(cashes)
                    self.logger.record("train/cash", float(mean_cash))

            # flush logger (SB3 handles writer closing)
        except Exception as e:
            if self.verbose:
                print("TensorboardPnlCallback exception:", e)

        return True


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Save a model each time the mean reward over the last N episodes improves.
    Keeps track of best mean reward and saves model to log_dir.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -float("inf")
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        try:
            # retrieve last episode rewards if available
            ep_info_buf = self.locals.get("episode_rewards", None)
            if ep_info_buf:
                mean_reward = float(sum(ep_info_buf) / len(ep_info_buf))
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    path = os.path.join(self.log_dir, f"best_model_{self.num_timesteps}.zip")
                    self.model.save(path)
                    if self.verbose:
                        print(f"Saved new best model to {path} with mean_reward {mean_reward:.3f}")
        except Exception:
            pass

        return True
