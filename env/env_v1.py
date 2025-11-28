#28/11/2025 
# env.py
import gym
import numpy as np
import pandas as pd
from gym import spaces

class SolTradingEnv(gym.Env):
    """
    Minimal FinRL-style single-asset trading environment (15m candles).
    Action space: Discrete(2*hmax+1) -> action - hmax = signed integer shares to buy(+) / sell(-).
    Observation: [cash, shares_held, price, ema5, ema20, pct_change]
    Reward: delta portfolio value between steps.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, initial_amount=1000, hmax=10, transaction_cost_pct=0.001, reward_scaling=1.0):
        """
        df: pandas DataFrame with columns: timestamp (or date), open, high, low, close, volume
        initial_amount: starting cash (USD)
        hmax: maximum absolute shares the agent can buy/sell in one action
        transaction_cost_pct: per-trade commission fraction (applied to trade value)
        """
        super().__init__()
        self.df = df.copy().reset_index(drop=True)
        # enforce and rename date column
        if "timestamp" in self.df.columns:
            self.df.rename(columns={"timestamp": "date"}, inplace=True)
        if "date" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'date' or 'timestamp' column.")
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df.sort_values("date", inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # indicators
        self.df["pct_change"] = self.df["close"].pct_change().fillna(0.0)
        self.df["ema5"] = self.df["close"].ewm(span=5, adjust=False).mean()
        self.df["ema20"] = self.df["close"].ewm(span=20, adjust=False).mean()
        self.df.fillna(method="bfill", inplace=True)

        self.initial_amount = float(initial_amount)
        self.hmax = int(hmax)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.reward_scaling = float(reward_scaling)

        # action space: integer -hmax..hmax mapped to Discrete(2*hmax+1)
        self.action_space = spaces.Discrete(2 * self.hmax + 1)

        # observation: vector of 6 floats
        obs_low = np.array([0.0, 0.0, 0.0, -np.inf, -np.inf, -1.0], dtype=np.float32)
        obs_high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._reset_internal_state()

    def _reset_internal_state(self):
        self.current_step = 0
        self.cash = float(self.initial_amount)
        self.shares_held = 0
        self.trades = []
        self._update_account_value()  # set initial account value

    def _get_obs(self):
        row = self.df.loc[self.current_step]
        obs = np.array([
            float(self.cash),
            float(self.shares_held),
            float(row["close"]),
            float(row["ema5"]),
            float(row["ema20"]),
            float(row["pct_change"]),
        ], dtype=np.float32)
        return obs

    def _update_account_value(self):
        price = float(self.df.loc[self.current_step, "close"])
        self.account_value = self.cash + self.shares_held * price

    def reset(self):
        self._reset_internal_state()
        return self._get_obs()

    def step(self, action):
        """
        action: Discrete index; translate to signed integer num_shares = action - hmax
        """
        done = False
        info = {}

        num_shares = int(action) - self.hmax  # signed
        row = self.df.loc[self.current_step]
        price = float(row["close"])
        trade_value = price * abs(num_shares)
        cost = trade_value * self.transaction_cost_pct

        # Execute trade (buy or sell) constrained by cash / holdings
        if num_shares > 0:
            # buy up to what cash allows
            max_buyable = int(self.cash // (price * (1 + self.transaction_cost_pct)))
            units = min(num_shares, max_buyable)
            if units > 0:
                spent = units * price + units * price * self.transaction_cost_pct
                self.cash -= spent
                self.shares_held += units
                self.trades.append(("buy", units, price, self.current_step))
        elif num_shares < 0:
            # sell up to holdings
            units = min(abs(num_shares), self.shares_held)
            if units > 0:
                proceeds = units * price - units * price * self.transaction_cost_pct
                self.cash += proceeds
                self.shares_held -= units
                self.trades.append(("sell", units, price, self.current_step))
        # else 0 -> hold

        prev_value = self.account_value
        # advance
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        self._update_account_value()
        reward = (self.account_value - prev_value) * self.reward_scaling

        obs = self._get_obs()
        info["account_value"] = self.account_value
        info["cash"] = self.cash
        info["shares_held"] = self.shares_held
        info["timestamp"] = self.df.loc[self.current_step, "date"]

        return obs, float(reward), done, info

    def render(self, mode="human"):
        print(
            f"Step: {self.current_step} | Date: {self.df.loc[self.current_step,'date']} "
            f"| Price: {self.df.loc[self.current_step,'close']:.4f} | Cash: {self.cash:.2f} | "
            f"Shares: {self.shares_held} | Account value: {self.account_value:.2f}"
        )

    def close(self):
        pass


def make_env_from_csv(csv_path, initial_amount=1000, hmax=10, transaction_cost_pct=0.001, reward_scaling=1.0):
    df = pd.read_csv(csv_path)
    # ensure names: timestamp,open,high,low,close,volume
    if set(["open","high","low","close","volume"]).issubset(df.columns) and ("timestamp" in df.columns or "date" in df.columns):
        return SolTradingEnv(df, initial_amount=initial_amount, hmax=hmax, transaction_cost_pct=transaction_cost_pct, reward_scaling=reward_scaling)
    else:
        raise ValueError("CSV must include timestamp/date and open,high,low,close,volume columns.")
