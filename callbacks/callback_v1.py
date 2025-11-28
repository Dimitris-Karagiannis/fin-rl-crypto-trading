# callbacks.py
import os
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardPnlCallback(BaseCallback):
    """
    Logs custom pnl/account_value metrics to TensorBoard.
    Expects env info to contain 'account_value' key when step() returns info.
    """
    def __init__(self, verbose=0, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = int(log_freq)
        self._step = 0

    def _on_step(self) -> bool:
        self._step += 1
        if self._step % self.log_freq == 0:
            # Get latest env infos (vectorized env). We'll try to aggregate.
            try:
                infos = self.locals.get("infos", None)
                if infos:
                    # infos may be a list (for VecEnv) or a dict
                    if isinstance(infos, list):
                        # average account value across envs (if present)
                        avs = [info.get("account_value") for info in infos if info.get("account_value") is not None]
                        if len(avs) > 0:
                            mean_av = sum(avs) / len(avs)
                            self.logger.record("train/account_value", float(mean_av))
                    else:
                        av = infos.get("account_value")
                        if av is not None:
                            self.logger.record("train/account_value", float(av))

                    # If the model exposes recent episodic returns, also log them
                    if "episode" in self.locals:
                        pass

                # flush to TB (SB3 will handle writer on training end)
            except Exception as e:
                if self.verbose:
                    print("TensorboardPnlCallback exception:", e)
        return True


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Save a model each time the mean reward over the last N episodes improves.
    Simple example - keeps a rolling best and saves model.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -float("inf")
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Evaluate across env (if model has env)
            try:
                # retrieve last episode rewards if available in locals
                ep_info_buf = self.locals.get("episode_rewards", None)
                # we avoid heavy evaluation here - this is a lightweight check
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
