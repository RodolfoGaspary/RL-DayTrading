import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Custom Environment for MTF Local S/R Trading.
    Focus: QQQ 1m, 5m, 15m, 4h features.
    """
    def __init__(self, df, initial_balance=10000, daily_target=0.01, 
                 reward_scaling=100.0, transaction_penalty=0.00005, trail_atr_mult=2.0,
                 multi_stage_stop=False, forced_exit_time=None, restrict_entry_hours=False):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.daily_target = daily_target
        self.reward_scaling = reward_scaling
        self.transaction_penalty = transaction_penalty
        self.trail_atr_mult = trail_atr_mult
        self.multi_stage_stop = multi_stage_stop
        self.forced_exit_time = forced_exit_time
        self.restrict_entry_hours = restrict_entry_hours
        
        # Action space: 0: Hold, 1: Buy (Long), 2: Sell/Close
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 
        # MTF features (scaled) + position + current_pnl + daily_pnl
        num_features = len([c for c in df.columns if not c.startswith('timestamp')])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features + 3,), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0 # 0: Flat, 1: Long
        self.entry_price = 0
        self.daily_pnl = 0
        self.trades = 0
        self.last_day = self.df.index[0].date()
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs = self.df.iloc[self.current_step].drop('timestamp', errors='ignore').values.astype(np.float32)
        
        # Calculate unrealized PnL %
        unrealized_pnl = 0
        if self.position == 1:
            current_price = self.df.iloc[self.current_step]['close']
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            
        # [features..., position, unrealized_pnl, daily_pnl]
        return np.append(obs, [float(self.position), float(unrealized_pnl), float(self.daily_pnl)]).astype(np.float32)

    def step(self, action):
        done = False
        truncated = False
        reward = 0
        
        current_price = self.df.iloc[self.current_step]['close']
        current_date = self.df.index[self.current_step].date()
        
        # Check if new day starts
        if current_date != self.last_day:
            self.daily_pnl = 0
            self.last_day = current_date
            
        # Forced Exit Logic (e.g. 15:55)
        if self.forced_exit_time and self.position == 1:
            current_time = self.df.index[self.current_step].time()
            # We want to exit if current time is >= forced_exit_time
            # Assuming forced_exit_time is a standard datetime.time object
            if current_time >= self.forced_exit_time:
                # Force Sell
                pnl = (current_price - self.entry_price) / self.entry_price
                self.balance *= (1 + pnl)
                self.daily_pnl += pnl
                self.position = 0
                self.entry_price = 0
                reward += pnl * self.reward_scaling
                # Maybe add a small penalty or bonus? For now just exit.
                return self._get_obs(), reward, done, truncated, {"reason": "Forced-Exit"}
            
        # Daily stop logic
        if self.daily_pnl >= self.daily_target:
            # We reached 1% for the day. Normally we stop.
            # But the user said: "if we go over with a trade... maybe nets us 43%... then we will stop".
            # So if we ARE in a trade, we keep going until it closes.
            # If we are NOT in a trade, we skip actions.
            if self.position == 0:
                action = 0 # Force hold
        
        # Entry Restriction (Sleep Logic)
        if self.restrict_entry_hours and action == 1 and self.position == 0:
            # Check time
            current_time_val = self.df.index[self.current_step].hour * 100 + self.df.index[self.current_step].minute
            # Market Open: 09:30 - 16:00
            if not (930 <= current_time_val < 1600):
                action = 0 # Deny entry
        
        # Execute Action
        if action == 1 and self.position == 0: # Buy
            self.position = 1
            self.entry_price = current_price
            self.max_price = current_price # For trailing stop
            self.trades += 1
            reward -= self.transaction_penalty # Use param
            
        elif action == 2 and self.position == 1: # Sell/Close
            pnl = (current_price - self.entry_price) / self.entry_price
            self.balance *= (1 + pnl)
            self.daily_pnl += pnl
            self.position = 0
            self.entry_price = 0
            reward += pnl * self.reward_scaling # Use param
            
            # Print for tracking
            if self.trades % 100 == 0:
                print(f"Trade {self.trades}: PnL {pnl:.2%}, Daily PnL: {self.daily_pnl:.2%}")

            # Bonus for hitting 1% daily target
            if self.daily_pnl >= self.daily_target:
                reward += 0.5
                
        # ATR-based Trailing Stop Loss
        if self.position == 1:
            atr = self.df.iloc[self.current_step]['1m_atr']
            if current_price > self.max_price:
                self.max_price = current_price
            
            # MULTI-STAGE STOP LOGIC
            current_trail_mult = self.trail_atr_mult
            if self.multi_stage_stop:
                current_pnl = (current_price - self.entry_price) / self.entry_price
                if current_pnl >= 0.001: # 0.1% profit hit
                    # Tighten stop to 0.5 ATR once in significant profit
                    current_trail_mult = 0.5 
                else:
                    # Start wider (e.g. 2.5) to survive the initial bounce noise
                    current_trail_mult = 2.5

            # Trailing stop at N * ATR from the high
            stop_loss = self.max_price - (current_trail_mult * atr)
            
            if current_price <= stop_loss:
                # Forced close
                pnl = (stop_loss - self.entry_price) / self.entry_price
                self.balance *= (1 + pnl)
                self.daily_pnl += pnl
                self.position = 0
                self.entry_price = 0
                reward += pnl * self.reward_scaling # Use param
                return self._get_obs(), reward, done, truncated, {"reason": f"MT-Stop ({current_trail_mult}x)"}
        
        # Move to next step
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
            
        return self._get_obs(), reward, done, truncated, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Daily PnL: {self.daily_pnl:.2%}, Trades: {self.trades}")
