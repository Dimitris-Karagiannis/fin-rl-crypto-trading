import pandas as pd
import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

class CryptoTradingEnvContinuous(StockTradingEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_metrics()

    def reset_metrics(self):
        self.last_action = None
        self.last_reward = None
        self.last_trade_pnl = 0.0
        self.cumulative_pnl = 0.0
        self.total_trades = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.avg_entry_price = 0.0
        self.prev_stock_owned = self.stock_owned.copy() if hasattr(self, "stock_owned") else [0.0]

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.reset_metrics()
        return obs

    def _get_current_price(self):
        idx = int(getattr(self, "day", getattr(self, "_current_step", 0)))
        idx = max(0, min(idx, len(self.df) - 1))
        return float(self.df.loc[idx, "close"])

    def step(self, action):
        """
        Continuous action in [-1, 1], scaled to max buy/sell fraction of hmax.
        Positive action = buy, negative action = sell.
        """
        action = np.array(action).flatten()
        self.last_action = action.copy()
        prev_pos = self.stock_owned.copy()

        # Scale action to fractional BTC quantity
        # Clip to allowable range [-hmax, +hmax]
        delta = np.clip(action[0] * self.hmax, -prev_pos[0], self.hmax)

        # Execute trade
        self.stock_owned[0] += delta
        price = self._get_current_price()

        # Calculate PnL if selling
        if delta < 0:  # sell
            sell_amount = -delta
            if prev_pos[0] > 0:
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
            else:
                self.last_trade_pnl = 0.0
        elif delta > 0:  # buy
            buy_amount = delta
            old_qty = prev_pos[0]
            new_qty = self.stock_owned[0]
            self.avg_entry_price = price if old_qty <= 0 else (self.avg_entry_price * old_qty + price * buy_amount) / new_qty
            self.last_trade_pnl = 0.0
        else:
            self.last_trade_pnl = 0.0

        self.prev_stock_owned = self.stock_owned.copy()

        # Call parent step to get reward, obs, etc.
        obs, reward, terminated, truncated, info = super().step([self.stock_owned[0]])
        self.last_reward = float(reward)
        return obs, reward, terminated, truncated, info

    @property
    def win_ratio(self):
        return float(self.win_trades) / max(self.total_trades, 1)

    @property
    def cumulative_pnl_value(self):
        return float(self.cumulative_pnl)

    def create_crypto_env(csv_path, initial_capital=1000.0, hmax=100, tech_indicator_list=None):
        # Load data
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df.rename(columns={"timestamp": "date"}, inplace=True)  # rename for env

        # Add 'tic' column for StockTradingEnv compatibility
        df['tic'] = 'BTC'  # single asset identifier

        stock_dim = 1  # only BTC
        if tech_indicator_list is None:
            tech_indicator_list = ["close"]  # simple default

        num_stock_shares = [0] * stock_dim  # start with zero BTC

        buy_cost_pct = [0.001] * stock_dim
        sell_cost_pct = [0.001] * stock_dim

        reward_scaling = 1e-4
        state_space = 1 + 2*stock_dim + len(tech_indicator_list)  # typical FinRL formula
        action_space = stock_dim

        env = CryptoTradingEnvContinuous(
            df=df,
            stock_dim=stock_dim,
            hmax=hmax,
            initial_amount=initial_capital,
            num_stock_shares=num_stock_shares,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            state_space=state_space,
            action_space=action_space,
            tech_indicator_list=tech_indicator_list
        )

        return env