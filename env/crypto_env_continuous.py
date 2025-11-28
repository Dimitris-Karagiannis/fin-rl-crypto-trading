import pandas as pd
import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

class CryptoTradingEnvContinuous(StockTradingEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Metrics
        self.last_action = None
        self.last_reward = None
        self.last_trade_pnl = 0.0
        self.cumulative_pnl = 0.0
        self.total_trades = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.avg_entry_price = 0.0

        # Make sure stock_owned exists before step() is called
        self._stock_owned = [0.0] * self.stock_dim

    @property
    def stock_owned(self):
        # Ensure stock_owned is always initialized
        if not hasattr(self, "_stock_owned"):
            self._stock_owned = [0.0] * self.stock_dim
        return self._stock_owned

    @stock_owned.setter
    def stock_owned(self, value):
        self._stock_owned = value

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.last_action = None
        self.last_reward = None
        self.last_trade_pnl = 0.0
        self.cumulative_pnl = 0.0
        self.total_trades = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.avg_entry_price = 0.0
        self._stock_owned = [0.0] * self.stock_dim
        return obs

    def _get_current_price(self):
        idx = int(getattr(self, "day", getattr(self, "_current_step", 0)))
        idx = max(0, min(idx, len(self.df) - 1))
        return float(self.df.loc[idx, "close"])

    def step(self, action):
        """Step function with fractional trades and metric updates"""
        action = np.array(action).flatten()
        self.last_action = action.copy()

        prev_pos = self.stock_owned.copy()
        price = self._get_current_price()

        # Apply fractional trade
        self._trade_stock(action)

        # Call parent step to get reward etc.
        result = super().step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result

        self.last_reward = float(reward)
        cur_pos = self.stock_owned.copy()
        delta = cur_pos[0] - prev_pos[0]

        # Update PnL metrics for sells
        if delta < 0 and prev_pos[0] > 0:
            sell_amount = min(-delta, prev_pos[0])
            realized_pnl = sell_amount * (price - self.avg_entry_price)
            tx_cost = sell_amount * price * getattr(self, "transaction_cost_pct", 0.0)
            realized_pnl -= tx_cost
            self.last_trade_pnl = float(realized_pnl)
            self.cumulative_pnl += float(realized_pnl)
            self.total_trades += 1
            if realized_pnl > 0:
                self.win_trades += 1
            elif realized_pnl < 0:
                self.loss_trades += 1

            remaining = prev_pos[0] - sell_amount
            self.avg_entry_price = 0.0 if remaining <= 0 else self.avg_entry_price
        elif delta > 0:
            buy_amount = delta
            old_qty = prev_pos[0]
            new_qty = cur_pos[0]
            self.avg_entry_price = price if old_qty <= 0 else (self.avg_entry_price * old_qty + price * buy_amount) / new_qty
            self.last_trade_pnl = 0.0
        else:
            self.last_trade_pnl = 0.0

        return obs, reward, terminated, truncated, info

    def _trade_stock(self, action):
        """Fractional trade execution"""
        for idx in range(self.stock_dim):
            if action[idx] > 0:  # buy
                buy_amount = action[idx]
                cost = buy_amount * self._get_current_price() * (1 + self.buy_cost_pct[idx])
                if cost <= self.cash:
                    self._stock_owned[idx] += buy_amount
                    self.cash -= cost
            elif action[idx] < 0:  # sell
                sell_amount = min(-action[idx], self._stock_owned[idx])
                proceeds = sell_amount * self._get_current_price() * (1 - self.sell_cost_pct[idx])
                self._stock_owned[idx] -= sell_amount
                self.cash += proceeds

    @property
    def win_ratio(self):
        return float(self.win_trades) / max(self.total_trades, 1)

    @property
    def cumulative_pnl_value(self):
        return float(self.cumulative_pnl)


def create_crypto_env(csv_path, initial_capital=1000.0, hmax=100, tech_indicator_list=None):
    # Load data
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df.rename(columns={"timestamp": "date"}, inplace=True)
    df['tic'] = 'BTC'

    stock_dim = 1
    if tech_indicator_list is None:
        tech_indicator_list = ["close"]

    env = CryptoTradingEnvContinuous(
        df=df,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=initial_capital,
        num_stock_shares=[0]*stock_dim,
        buy_cost_pct=[0.001]*stock_dim,
        sell_cost_pct=[0.001]*stock_dim,
        reward_scaling=1e-4,
        state_space=1 + 2*stock_dim + len(tech_indicator_list),
        action_space=stock_dim,
        tech_indicator_list=tech_indicator_list
    )

    return env
