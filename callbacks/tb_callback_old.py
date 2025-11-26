from stable_baselines3.common.callbacks import BaseCallback

class TensorboardBTCLogger(BaseCallback):
    """
    Custom callback to log:
      - portfolio total value
      - btc holding
      - last action fractional BTC
      - reward per step
      - last trade PnL and cumulative PnL
      - total trades and win ratio
    """

    def __init__(self, verbose=0):
        super(TensorboardBTCLogger, self).__init__(verbose)

    def _on_step(self) -> bool:
        # single env assumption: get first env
        env = self.training_env.envs[0]

        # -- portfolio total value --
        portfolio_value = getattr(env, 'total_asset', None)
        if portfolio_value is not None:
            self.logger.record('portfolio/total_value', float(portfolio_value))

        # -- BTC holding --
        btc_holding = getattr(env, 'stock_owned', None)
        if btc_holding is not None:
            # record first (and only) asset
            try:
                val = float(btc_holding[0])
            except Exception:
                val = float(btc_holding)
            self.logger.record('portfolio/btc_holding', val)

        # -- last action (fractional BTC) --
        last_action = getattr(env, 'last_action', None)
        if last_action is not None:
            try:
                self.logger.record('action/btc_fraction', float(last_action[0]))
            except Exception:
                self.logger.record('action/btc_fraction', float(last_action))

        # -- reward every step --
        last_reward = getattr(env, 'last_reward', None)
        if last_reward is not None:
            self.logger.record('reward/step', float(last_reward))

        # -- PnL per trade and cumulative PnL --
        last_trade_pnl = getattr(env, 'last_trade_pnl', None)
        if last_trade_pnl is not None:
            self.logger.record('trade/last_pnl', float(last_trade_pnl))

        cumulative_pnl = getattr(env, 'cumulative_pnl_value', None)
        if cumulative_pnl is not None:
            self.logger.record('trade/cumulative_pnl', float(cumulative_pnl))

        # -- trades and win ratio --
        total_trades = getattr(env, 'total_trades', None)
        if total_trades is not None:
            self.logger.record('trade/total_trades', float(total_trades))

        win_ratio = getattr(env, 'win_ratio', None)
        if win_ratio is not None:
            self.logger.record('trade/win_ratio', float(win_ratio))

        # ensure SB3 writes logs
        return True
