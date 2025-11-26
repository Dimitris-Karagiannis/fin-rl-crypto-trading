# env/crypto_env_continuous.py
import pandas as pd
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import numpy as np

class CryptoTradingEnvContinuous(StockTradingEnv):
    """
    Continuous action env built on top of FinRL's StockTradingEnv.
    Adds:
    - last_action (for logging)
    - last_reward
    - simple trade bookkeeping to compute realized PnL per sell
    - cumulative PnL, win/loss counters
    Notes:
    - For trade price we use the close price at the current environment day.
    - This implements a simple average-entry price for the currently held long position.
    """
    def __init__(self, *args, **kwargs):
        super(CryptoTradingEnvContinuous, self).__init__(*args, **kwargs)
        # bookkeeping
        self.last_action = None
        self.last_reward = None
        self.last_trade_pnl = 0.0
        self.cumulative_pnl = 0.0
        self.total_trades = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.avg_entry_price = 0.0  # weighted average entry price for current position (long)
        self.prev_stock_owned = None

    def reset(self, **kwargs):
        obs = super(CryptoTradingEnvContinuous, self).reset(**kwargs)
        self.last_action = None
        self.last_reward = None
        self.last_trade_pnl = 0.0
        self.cumulative_pnl = 0.0
        self.total_trades = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.avg_entry_price = 0.0
        # track previous holdings to compute deltas
        self.prev_stock_owned = self.stock_owned.copy() if hasattr(self, "stock_owned") else [0.0]
        return obs

    def _get_current_price(self):
        """
        Returns the close price for the current day index.
        StockTradingEnv usually keeps an index called `day`. We handle a fallback if not present.
        """
        # prefer attribute day used in FinRL StockTradingEnv
        if hasattr(self, "day"):
            idx = int(self.day)
        else:
            # fallback: use internal pointer name if available
            idx = int(getattr(self, "_current_step", 0))
        # ensure idx is within df range
        if idx < 0:
            idx = 0
        if idx >= len(self.df):
            idx = len(self.df) - 1
        return float(self.df.loc[idx, "close"])

    def step(self, action):
        """
        action: continuous vector (len = stock_dim). For single asset env, action[0] in [-1,1]
        We:
        - store last_action
        - call super().step(action)
        - compute changes in holdings and realize PnL on sells against avg_entry_price
        - update avg_entry_price on buys
        - store last_reward
        """
        # Ensure numpy array
        action = np.array(action).flatten()
        self.last_action = action.copy()

        # previous position
        prev_pos = self.stock_owned.copy() if hasattr(self, "stock_owned") else [0.0]

        # perform step (super will update stock_owned, cash, total_asset etc.)
        obs, reward, done, info = super(CryptoTradingEnvContinuous, self).step(action)

        # store last_reward for logging
        self.last_reward = float(reward)

        # current position after action
        cur_pos = self.stock_owned.copy() if hasattr(self, "stock_owned") else [0.0]

        # current price at this step (use close price)
        price = self._get_current_price()

        # For each asset (we have only 1 in this setup)
        delta = cur_pos[0] - prev_pos[0]

        # Buying (increasing position)
        if delta > 0:
            buy_amount = delta
            # update average entry price (weighted)
            old_qty = prev_pos[0]
            new_qty = cur_pos[0]
            if old_qty <= 0:
                # starting new long position
                self.avg_entry_price = price
            else:
                # weighted avg
                self.avg_entry_price = (self.avg_entry_price * old_qty + price * buy_amount) / new_qty

        # Selling (decreasing position) -> realize PnL on sold portion
        elif delta < 0:
            sell_amount = -delta
            # clamp sell amount to previous position
            sell_amount = min(sell_amount, prev_pos[0])
            if prev_pos[0] > 0:
                # realized PnL per unit = price - avg_entry_price
                realized_pnl = sell_amount * (price - self.avg_entry_price)
                # subtract transaction cost approx (env already applies it; we subtract here to be consistent)
                trade_value = sell_amount * price
                tx_cost = trade_value * getattr(self, "transaction_cost_pct", 0.0)
                realized_pnl -= tx_cost

                # record trade
                self.last_trade_pnl = float(realized_pnl)
                self.cumulative_pnl += float(realized_pnl)
                self.total_trades += 1
                if realized_pnl > 0:
                    self.win_trades += 1
                elif realized_pnl < 0:
                    self.loss_trades += 1

                # If we fully closed position, reset avg_entry_price
                remaining = prev_pos[0] - sell_amount
                if remaining <= 0:
                    self.avg_entry_price = 0.0
                # Otherwise avg_entry_price remains the same for remaining quantity
            else:
                # selling while no position (shouldn't happen): set last_trade_pnl zero
                self.last_trade_pnl = 0.0

        else:
            # no change
            self.last_trade_pnl = 0.0

        # update prev_stock_owned for next step
        self.prev_stock_owned = cur_pos.copy()

        return obs, reward, done, info

    # helper properties to expose metrics easily
    @property
    def win_ratio(self):
        if self.total_trades == 0:
            return 0.0
        return float(self.win_trades) / float(self.total_trades)

    @property
    def cumulative_pnl_value(self):
        return float(self.cumulative_pnl)
