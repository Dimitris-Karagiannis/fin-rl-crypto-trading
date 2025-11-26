from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TensorboardBTCLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardBTCLogger, self).__init__(verbose)

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]

        # Portfolio total value
        portfolio_value = getattr(env, 'total_asset', None)
        if portfolio_value is not None:
            self.logger.record('portfolio/total_value', float(portfolio_value))

        # BTC holding
        btc_holding = getattr(env, 'stock_owned', None)
        if btc_holding is not None:
            val = float(btc_holding[0]) if isinstance(btc_holding, (list, np.ndarray)) else float(btc_holding)
            self.logger.record('portfolio/btc_holding', val)

        # Last action
        last_action = getattr(env, 'last_action', None)
        if last_action is not None:
            val = float(last_action[0]) if isinstance(last_action, (list, np.ndarray)) else float(last_action)
            self.logger.record('action/btc_fraction', val)

        # Reward per step
        last_reward = getattr(env, 'last_reward', None)
        if last_reward is not None:
            self.logger.record('reward/step', float(last_reward))

        # PnL per trade and cumulative
        last_trade_pnl = getattr(env, 'last_trade_pnl', None)
        if last_trade_pnl is not None:
            self.logger.record('trade/last_pnl', float(last_trade_pnl))

        cumulative_pnl = getattr(env, 'cumulative_pnl_value', None)
        if cumulative_pnl is not None:
            self.logger.record('trade/cumulative_pnl', float(cumulative_pnl))

        # Total trades and win ratio
        total_trades = getattr(env, 'total_trades', None)
        if total_trades is not None:
            self.logger.record('trade/total_trades', float(total_trades))

        win_ratio = getattr(env, 'win_ratio', None)
        if win_ratio is not None:
            self.logger.record('trade/win_ratio', float(win_ratio))

        return True
