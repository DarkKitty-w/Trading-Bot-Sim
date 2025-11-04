import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import requests
import os
from datetime import datetime, timedelta
import concurrent.futures
import scipy.optimize as sco # Import scipy.optimize for optimization

class EnhancedParallelCryptoSimulatorCG:
    def __init__(self, starting_cash=1000.0, update_interval=5, max_history_points=1000, max_drawdown_pct=0.20): # Added max_drawdown_pct
        self.starting_cash = starting_cash
        self.transaction_cost = 0.001
        self.update_interval = update_interval
        self.max_history_points = max_history_points
        self.max_drawdown_pct = max_drawdown_pct  # Maximum acceptable drawdown percentage
        self.live_data = {}
        self.is_running = False
        self.data_lock = threading.Lock()
        self.output_dir = "portfolio_logs"
        os.makedirs(self.output_dir, exist_ok=True)

        self.params = {
            "MA_Original": {"window": 5, "max_position_pct": 0.15, "stop_loss_pct": 0.05, "take_profit_pct": 0.10, "min_price_variation_pct": 0.00005, "confirmation_periods": 1, "timeframes": ['1m', '5m']},
            "MA_Fast": {"short_window": 3, "long_window": 10, "use_ema": True, "use_kama": False, "kama_fast_period": 2, "kama_slow_period": 30, "max_position_pct": 0.20, "stop_loss_pct": 0.08, "min_price_variation_pct": 0.00005, "confirmation_periods": 1, "timeframes": ['1m', '5m', '15m']},
            "MA_Enhanced": {"short_window": 5, "long_window": 20, "volatility_threshold": 0.0005, "max_position_pct": 0.12, "stop_loss_pct": 0.06, "take_profit_pct": 0.15, "min_price_variation_pct": 0.00005, "confirmation_periods": 1, "timeframes": ['1m', '5m']},
            "Momentum_Enhanced": {"period": 1, "threshold": 0.3, "smoothing": 1, "max_position_pct": 0.9, "stop_loss_pct": 0.7, "min_price_variation_pct": 0.00005, "confirmation_periods": 1, "timeframes": ['1m']},
            "Breakout": {"period": 5, "max_position_pct": 0.10, "stop_loss_pct": 0.04, "take_profit_pct": 0.12, "min_price_variation_pct": 0.00005, "confirmation_periods": 1, "timeframes": ['1m', '15m']},
            "MACD": {"fast": 12, "slow": 26, "signal": 9, "max_position_pct": 0.15, "stop_loss_pct": 0.06, "min_price_variation_pct": 0.00005, "confirmation_periods": 1, "timeframes": ['1m', '5m']},
            "ATR_Breakout": {"period": 14, "multiplier": 1.5, "max_position_pct": 0.12, "stop_loss_pct": 2.0, "take_profit_pct": 3.0, "min_price_variation_pct": 0.00005, "confirmation_periods": 1, "timeframes": ['1m', '15m']},
            "ADX_Trend": {"period": 14, "min_strength": 20, "max_position_pct": 0.10, "stop_loss_pct": 0.08, "min_price_variation_pct": 0.00005, "confirmation_periods": 1, "timeframes": ['1m', '5m']},
            "MeanReversion": {"period": 20, "buy_threshold": 0.99, "sell_threshold": 1.01, "max_position_pct": 0.08, "take_profit_pct": 1.015, "min_price_variation_pct": 0.00005, "confirmation_periods": 1, "limit_buy_offset_pct": -0.003, "trailing_stop_pct": 0.02, "timeframes": ['1m', '5m']}
        }

        self.coingecko_id_map = {
            "BTC-USD": "90", "ETH-USD": "80", "XRP-USD": "58", "BNB-USD": "2710",
            "SOL-USD": "48543", "DOGE-USD": "2", "ADA-USD": "257", "LINK-USD": "2751",
            "HBAR-USD": "48555", "AVAX-USD": "44883", "LTC-USD": "1", "SHIB-USD": "45088",
            "DOT-USD": "45219", "AAVE-USD": "46018", "NEAR-USD": "48563", "ICP-USD": "47311",
            "ATOM-USD": "33830", "SAND-USD": "45161", "AR-USD": "42441"
        }

        self.historical_data_dfs = {
            '1m': pd.DataFrame(columns=list(self.coingecko_id_map.keys())),
            '5m': pd.DataFrame(columns=list(self.coingecko_id_map.keys())),
            '15m': pd.DataFrame(columns=list(self.coingecko_id_map.keys()))
        }
        self._last_update_time = None
        self._last_timeframe_update = {tf: None for tf in self.historical_data_dfs.keys()}

    def start_real_time_data(self, coins):
        self.is_running = True
        self.coins = coins
        self.data_thread = threading.Thread(target=self._update_real_time_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        print(f"🔄 Started real-time CoinLore data for {len(coins)} coins")

    def stop_real_time_data(self):
        self.is_running = False
        print("🛑 Stopped real-time data collection")

    def _update_real_time_data(self):
        while self.is_running:
            try:
                coin_ids_for_api = []
                valid_coins = []

                for coin in self.coins:
                    if coin in self.coingecko_id_map:
                        coin_ids_for_api.append(self.coingecko_id_map[coin])
                        valid_coins.append(coin)
                    else:
                        print(f"⚠️ Warning: {coin} not found in CoinLore mapping")

                if not coin_ids_for_api:
                    print("❌ Error: No valid coins to fetch.")
                    time.sleep(60)
                    continue

                ids_param = ",".join(coin_ids_for_api)
                url = f"https://api.coinlore.net/api/ticker/?id={ids_param}"

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }

                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code != 200:
                    print(f"❌ API Error (CoinLore): Status code {response.status_code}")
                    time.sleep(60)
                    continue

                resp = response.json()
                current_time = datetime.now()

                with self.data_lock:
                    updated_count = 0
                    current_prices_dict = {}
                    for item in resp:
                        lore_id = item.get("id")
                        price_usd = item.get("price_usd")
                        if lore_id is None or price_usd is None:
                            continue

                        coin = None
                        for key, value in self.coingecko_id_map.items():
                            if value == lore_id:
                                coin = key
                                break

                        if coin is None:
                            print(f"⚠️ Unknown CoinLore ID received: {lore_id}")
                            continue

                        try:
                            price = float(price_usd)
                            current_prices_dict[coin] = price
                            if coin not in self.live_data:
                                self.live_data[coin] = {'price': price, 'timestamp': current_time}
                            else:
                                self.live_data[coin]['price'] = price
                                self.live_data[coin]['timestamp'] = current_time
                            updated_count += 1
                        except ValueError:
                            continue

                    if updated_count > 0:
                        for timeframe, data_df in self.historical_data_dfs.items():
                            interval_minutes = int(timeframe[:-1])
                            if self._last_timeframe_update.get(timeframe) is None or \
                               (current_time - self._last_timeframe_update[timeframe]).total_seconds() >= interval_minutes * 60:

                                new_row = pd.DataFrame([current_prices_dict], index=[current_time])
                                self.historical_data_dfs[timeframe] = pd.concat([data_df, new_row]).tail(self.max_history_points)
                                self._last_timeframe_update[timeframe] = current_time

                        self._last_update_time = current_time
                        print(f"✅ Updated {updated_count}/{len(valid_coins)} coins at {current_time.strftime('%H:%M:%S')}")

            except requests.exceptions.RequestException as e:
                print(f"❌ Network error fetching data: {e}")
                time.sleep(30)
            except Exception as e:
                print(f"❌ Unexpected error in data update: {e}")
                time.sleep(30)

            time.sleep(self.update_interval)


    def get_current_prices(self):
        with self.data_lock:
            return {coin: data['price'] for coin, data in self.live_data.items() if 'price' in data}

    def get_price_history(self, coin, timeframe='1m', limit=None) -> pd.Series:
        """
        Récupère l’historique des prix pour un coin à une granularité donnée, limité si besoin.
        Retourne une pd.Series.
        """
        with self.data_lock:
            if timeframe in self.historical_data_dfs and coin in self.historical_data_dfs[timeframe].columns:
                history_df = self.historical_data_dfs[timeframe][coin]
                if limit is not None:
                    return history_df.tail(limit)
                return history_df
            elif coin in self.live_data and 'history' in self.live_data[coin]:
                 print(f"⚠️ Requested timeframe '{timeframe}' not available in DataFrames for {coin}, using raw history.")
                 history = self.live_data[coin]['history']
                 if limit is not None:
                     return history.tail(limit)
                 return history
            return pd.Series([], dtype=float)


    ##############################
    # Technical Indicators (Vectorized)
    ##############################
    def calculate_technical_indicators(self, prices: pd.Series, strategy: str, lookback_period: int = 30):
        """
        Calculate additional technical indicators using vectorized operations.
        `prices` : pd.Series (index temporel) des prix.
        `strategy` : (non utilisé ici).
        `lookback_period`: The number of periods to look back for calculations.
        Retourne un dict d’indicateurs calculés.
        """
        if prices is None or len(prices) < 2:
            return {}

        indicators = {}

        # Volatility & momentum (vectorized)
        if len(prices) > 1:
            returns = prices.pct_change().dropna()
            indicators['volatility'] = returns.std() if not returns.empty else 0.0

            if len(prices) >= lookback_period and lookback_period > 0:
                try:
                    indicators['momentum'] = (prices.iloc[-1] / prices.iloc[-lookback_period] - 1) if prices.iloc[-lookback_period] != 0 else 0.0
                except IndexError:
                    indicators['momentum'] = 0.0
            elif len(prices) > 1:
                try:
                    indicators['momentum'] = (prices.iloc[-1] / prices.iloc[0] - 1) if prices.iloc[0] != 0 else 0.0
                except IndexError:
                    indicators['momentum'] = 0.0
            else:
                indicators['momentum'] = 0.0


        # RSI (Relative Strength Index) — vectorized
        rsi_period = 14
        if len(prices) > rsi_period:
            delta = prices.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)

            avg_gain = gain.ewm(com=rsi_period-1, adjust=False).mean()
            avg_loss = loss.ewm(com=rsi_period-1, adjust=False).mean()

            rs = avg_gain / (avg_loss + 1e-9)
            rsi_series = 100 - (100 / (1 + rs))
            indicators['rsi'] = rsi_series.iloc[-1] if not rsi_series.empty else 50
        else:
            indicators['rsi'] = 50

        return indicators

    ##############################
    # Trading Strategies (Vectorized)
    ##############################
    def generate_signal(self, prices: pd.Series, strategy: str, lookback_period: int = 30, min_price_variation_pct: float = 0.00005) -> int:
        """
        Génère un signal de trading (1 = achat, -1 = vente, 0 = neutre) selon la stratégie donnée.
        Retourne le signal scalaire final.
        """
        if prices is None or prices.empty or len(prices) < 2:
            return 0

        signals = pd.Series(0, index=prices.index)

        if len(prices) >= 2:
            short_var = prices.pct_change().fillna(0)
        else:
            short_var = pd.Series(0, index=prices.index)

        if len(prices) > 1:
            long_var_val = (prices.iloc[-1] / prices.iloc[0] - 1) if prices.iloc[0] != 0 else 0.0
        else:
            long_var_val = 0.0

        # Apply noise filter
        signals[abs(short_var) < min_price_variation_pct] = 0

        p = self.params.get(strategy, {})

        # --- STRATEGY LOGIC (VECTORIZED) ---
        if strategy == "MA_Original":
            window = min(p.get("window", 5), len(prices))
            if window >= 2:
                ma = prices.rolling(window=window).mean()
                signals[(prices > ma) & (signals != 0)] = 1
                signals[(prices < ma) & (signals != 0)] = -1

        elif strategy == "MA_Fast":
            short_window = min(p.get("short_window", 3), len(prices))
            long_window = min(p.get("long_window", 10), len(prices))
            if short_window >= 2 and long_window >= 2:
                if p.get("use_ema", True):
                    short_ma = prices.ewm(span=short_window, adjust=False).mean()
                    long_ma = prices.ewm(span=long_window, adjust=False).mean()
                else:
                    short_ma = prices.rolling(window=short_window).mean()
                    long_ma = prices.rolling(window=long_window).mean()
                signals[(short_ma > long_ma) & (signals != 0)] = 1
                signals[(short_ma < long_ma) & (signals != 0)] = -1

        elif strategy == "MA_Enhanced":
            short_window = min(p.get("short_window", 5), len(prices))
            long_window = min(p.get("long_window", 20), len(prices))
            if short_window >= 2 and long_window >= 2:
                vol_window = min(10, len(prices))
                vol = prices.pct_change().rolling(window=vol_window).std().fillna(0)
                short_ema = prices.ewm(span=short_window, adjust=False).mean()
                long_ema = prices.ewm(span=long_window, adjust=False).mean()
                volatility_threshold = p.get("volatility_threshold", 0.0003)

                buy_condition = (vol > volatility_threshold) & (short_ema > long_ema)
                sell_condition = (vol > volatility_threshold) & (short_ema < long_ema)
                signals[buy_condition & (signals != 0)] = 1
                signals[sell_condition & (signals != 0)] = -1

        elif strategy == "Momentum_Enhanced":
            period = min(p.get("period", 1), len(prices) - 1)
            if period >= 1:
                mom = prices.pct_change(period).fillna(0)
                smoothing = min(p.get("smoothing", 1), len(mom) if not mom.empty else 0)
                if smoothing > 1:
                    mom = mom.rolling(window=smoothing).mean().fillna(0)
                threshold = p.get("threshold", 0.2)
                signals[(mom > threshold) & (signals != 0)] = 1
                signals[(mom < -threshold) & (signals != 0)] = -1

        elif strategy == "Breakout":
            period = min(p.get("period", 5), len(prices) - 1)
            if period >= 1:
                res = prices.rolling(window=period).max().shift(1)
                sup = prices.rolling(window=period).min().shift(1)
                signals[(prices > res) & (signals != 0)] = 1
                signals[(prices < sup) & (signals != 0)] = -1

        elif strategy == "MACD":
            fast = min(p.get("fast", 12), len(prices))
            slow = min(p.get("slow", 26), len(prices))
            signal_win = min(p.get("signal", 9), len(prices))
            if fast >= 2 and slow >= 2 and signal_win >= 1:
                exp1 = prices.ewm(span=fast, adjust=False).mean()
                exp2 = prices.ewm(span=slow, adjust=False).mean()
                macd = exp1 - exp2
                signal_line = macd.ewm(span=signal_win, adjust=False).mean()
                signals[(macd > signal_line) & (signals != 0)] = 1
                signals[(macd < signal_line) & (signals != 0)] = -1

        elif strategy == "ATR_Breakout":
            period = min(p.get("period", 14), len(prices))
            if period >= 1:
                high_low = prices.diff().abs()
                high_close = (prices - prices.shift(1)).abs()
                low_close = (prices.shift(1) - prices).abs()
                tr = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
                atr = tr.rolling(window=period).mean().shift(1)

                if not atr.empty and len(prices) > period:
                    multiplier = p.get("multiplier", 1.5)
                    high_band = prices.rolling(window=period).max().shift(1) + atr * multiplier
                    low_band = prices.rolling(window=period).min().shift(1) - atr * multiplier
                    signals[(prices > high_band) & (signals != 0) & ~high_band.isna()] = 1
                    signals[(prices < low_band) & (signals != 0) & ~low_band.isna()] = -1

        elif strategy == "ADX_Trend":
            period = min(p.get("period", 14), len(prices))
            if period >= 2:
                plus_dm = (prices.diff().clip(lower=0)).ewm(com=period-1, adjust=False).mean()
                minus_dm = (-prices.diff().clip(upper=0)).ewm(com=period-1, adjust=False).mean()
                tr_adx = prices.diff().abs().ewm(com=period-1, adjust=False).mean()

                plus_di = 100 * (plus_dm / (tr_adx + 1e-9))
                minus_di = 100 * (minus_dm / (tr_adx + 1e-9))
                dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
                adx = dx.ewm(com=period-1, adjust=False).mean()
                min_strength = p.get("min_strength", 20)

                ma_adx = prices.rolling(window=period).mean()
                dir_sig = (prices > ma_adx).astype(int) - (prices < ma_adx).astype(int)

                signals[(adx > min_strength) & (signals != 0)] = dir_sig[(adx > min_strength) & (signals != 0)]

        elif strategy == "MeanReversion":
            period = min(p.get("period", 20), len(prices))
            if period >= 2:
                rm = prices.rolling(window=period).mean()
                buy_threshold = p.get("buy_threshold", 0.99)
                sell_threshold = p.get("sell_threshold", 1.01)
                rz = prices / (rm + 1e-9)
                signals[(rz < buy_threshold) & (signals != 0)] = 1
                signals[(rz > sell_threshold) & (signals != 0)] = -1

        # Long-term trend reinforcement
        last_signal = signals.iloc[-1] if not signals.empty else 0
        if long_var_val > 0.002 and (short_var.iloc[-1] if not short_var.empty else 0) > 0:
             signals.iloc[-1] = max(last_signal, 1)
        elif long_var_val < -0.002 and (short_var.iloc[-1] if not short_var.empty else 0) < 0:
             signals.iloc[-1] = min(last_signal, -1)

        return signals.iloc[-1] if not signals.empty else 0


    def enhanced_generate_signal(self, prices: pd.Series, strategy: str, indicators: dict = None, lookback_period: int = 30, min_price_variation_pct: float = 0.00005):
        if prices is None or prices.empty or len(prices) < 5:
            return 0

        signal_value = self.generate_signal(prices, strategy, lookback_period, min_price_variation_pct)

        if indicators is not None:
            rsi = indicators.get("rsi")
            if rsi is not None:
                if signal_value > 0 and rsi > 70:
                    return 0
                if signal_value < 0 and rsi < 30:
                    return 0

        return signal_value

    def calculate_position_size(self, strategy: str, cash: float) -> float:
        pct = self.params.get(strategy, {}).get("max_position_pct", 0.2)
        pct = max(0.2, pct)
        return min(cash * pct, cash * 0.95)

    def calculate_dynamic_position_size(self, strategy: str, cash: float, current_volatility: float = None) -> float:
        base_pct = self.params.get(strategy, {}).get("max_position_pct", 0.15)

        if current_volatility is not None:
            if current_volatility < 0.01:
                base_pct *= 1.5
            elif current_volatility > 0.03:
                base_pct *= 0.5

        return min(cash * base_pct, cash * 0.95)

    def calculate_kelly_position_size(self, strategy: str, cash: float, trade_log: list) -> float:
        """
        Calculate position size based on the Kelly Criterion.
        Requires trade history to estimate win rate and win/loss ratio.
        """
        if not trade_log:
            return self.calculate_dynamic_position_size(strategy, cash) # Fallback to dynamic sizing if no trades

        # Filter for executed trades (buys and sells)
        executed_trades = [t for t in trade_log if t.get('action') in ['BUY', 'SELL', 'LIMIT_BUY_FILLED', 'TRAILING_STOP_TRIGGERED']]

        if len(executed_trades) < 2:
             return self.calculate_dynamic_position_size(strategy, cash) # Need at least one buy and one sell to estimate win/loss

        # Calculate returns for executed trades (simplified: assumes buy/sell pairs)
        trade_returns = []
        buy_entry = None
        for trade in executed_trades:
            action = trade.get('action')
            price = trade.get('price')
            if action in ['BUY', 'LIMIT_BUY_FILLED']:
                buy_entry = trade # Store the full trade dict for potential future use
                buy_price = price
            elif action in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP_TRIGGERED'] and buy_price is not None:
                # Ensure the sell is for the same coin as the last buy entry
                if buy_entry and trade.get('coin') == buy_entry.get('coin'):
                    # Calculate return for the completed trade
                    trade_return = (price - buy_price) / buy_price
                    trade_returns.append(trade_return)
                    buy_price = None # Reset buy price after a sell
                    buy_entry = None # Reset buy entry

        if not trade_returns:
            return self.calculate_dynamic_position_size(strategy, cash)

        trade_returns_series = pd.Series(trade_returns)

        # Calculate win rate (p)
        winning_trades = trade_returns_series[trade_returns_series > 1e-9] # Consider return > 0 as a win
        p = len(winning_trades) / len(trade_returns_series) if len(trade_returns_series) > 0 else 0.0

        # Calculate average win (W) and average loss (L)
        average_win = winning_trades.mean() if not winning_trades.empty else 0.0
        losing_trades = trade_returns_series[trade_returns_series <= 1e-9] # Including zero returns as non-wins/losses
        average_loss = abs(losing_trades.mean()) if not losing_trades.empty else 0.0 # Use absolute value

        # Avoid division by zero for win/loss ratio
        win_loss_ratio = average_win / average_loss if average_loss > 1e-9 else 0.0

        # Calculate Kelly fraction (f)
        # Kelly formula: f = p - (1-p) / (W/L)
        # Ensure W/L is not zero or very small
        if win_loss_ratio < 1e-9:
             kelly_fraction = 0.0
        else:
             kelly_fraction = p - (1 - p) / win_loss_ratio

        # Apply a fraction of Kelly (e.g., half Kelly) to reduce risk
        kelly_fraction = max(0.0, kelly_fraction * 0.5) # Ensure fraction is non-negative and use half Kelly

        # Calculate position size as a fraction of current cash
        position_size_pct = kelly_fraction

        # Ensure position size is within reasonable bounds (e.g., not more than 10% of cash for any single trade)
        # This is a practical constraint to avoid over-betting even if Kelly suggests a large fraction
        max_single_trade_pct = self.params.get(strategy, {}).get("max_position_pct", 0.15) # Use strategy's max_position_pct as an upper bound
        position_size_pct = min(position_size_pct, max_single_trade_pct)

        # Also ensure a minimum position size to make trades worthwhile
        min_trade_value_pct = 0.005 # Example: minimum trade value is 0.5% of cash
        if kelly_fraction > 0 and position_size_pct < min_trade_value_pct:
             position_size_pct = min_trade_value_pct


        # Ensure position size doesn't exceed available cash (minus a small buffer)
        return min(cash * position_size_pct, cash * 0.95)



    def calculate_portfolio_volatility(self, portfolio_history: list) -> float:
        if portfolio_history is None or len(portfolio_history) < 5:
            return 0.0

        values = [p.get('value', 0) for p in portfolio_history[-10:] if 'value' in p]
        if len(values) < 2:
            return 0.0

        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        return float(np.std(returns)) if returns else 0.0

    def calculate_win_rate(self, trade_log: list) -> float:
        if not trade_log:
            return 0.0

        completed_trades = []
        buy_entry = None
        for trade in trade_log:
            action = trade.get('action')
            if action in ['BUY', 'LIMIT_BUY_FILLED']:
                buy_entry = trade
            elif action in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP_TRIGGERED'] and buy_entry is not None:
                if trade.get('coin') == buy_entry.get('coin'): # Ensure it's a sell for the same coin
                    completed_trades.append({'buy': buy_entry, 'sell': trade})
                    buy_entry = None

        if not completed_trades:
            return 0.0

        winning_trades = 0
        for trade_pair in completed_trades:
            buy_price = trade_pair['buy'].get('price', 0)
            sell_price = trade_pair['sell'].get('price', 0)
            # Consider a win if sell price is strictly greater than buy price
            if sell_price > buy_price + (buy_price * self.transaction_cost * 2): # Account for transaction costs
                winning_trades += 1

        total_completed_trades = len(completed_trades)
        return winning_trades / total_completed_trades if total_completed_trades > 0 else 0.0


    def calculate_limit_buy_price(self, prices: pd.Series, strategy_params: dict) -> float or None:
        limit_offset_pct = strategy_params.get("limit_buy_offset_pct", -0.002)
        if prices is not None and not prices.empty:
            return prices.iloc[-1] * (1 + limit_offset_pct)
        return None

    def calculate_trailing_stop_price(self, peak_price: float, trailing_pct: float) -> float:
        trailing_pct = max(trailing_pct * 0.7, 0.005)
        return peak_price * (1 - trailing_pct)

    def calculate_var_es(self, portfolio_history: list, confidence_level=0.99):
        """
        Calculate Value at Risk (VaR) and Expected Shortfall (ES).
        portfolio_history: list of dicts with 'value' key.
        confidence_level: e.g., 0.99 for 99% confidence.
        Returns a dict with 'VaR' and 'ES'.
        """
        if not portfolio_history or len(portfolio_history) < 2:
            return {'VaR': 0.0, 'ES': 0.0}

        # Extract portfolio values and calculate returns
        values = [p.get('value', 0) for p in portfolio_history if 'value' in p]
        if len(values) < 2:
             return {'VaR': 0.0, 'ES': 0.0}

        returns = pd.Series(values).pct_change().dropna()

        if returns.empty:
             return {'VaR': 0.0, 'ES': 0.0}

        # Calculate VaR (Historical Method)
        # Sort returns and find the return at the desired percentile
        sorted_returns = returns.sort_values(ascending=True)
        var_index = int(len(sorted_returns) * (1 - confidence_level))
        # Ensure index is within bounds
        var_index = max(0, min(var_index, len(sorted_returns) - 1))

        # VaR is the negative of the return at the VaR index
        var = -sorted_returns.iloc[var_index]

        # Calculate Expected Shortfall (ES)
        # ES is the average of returns below the VaR return
        es_returns = sorted_returns[sorted_returns <= -var] # Returns equal to or worse than VaR
        es = -es_returns.mean() if not es_returns.empty else 0.0

        return {'VaR': var, 'ES': es}

    def calculate_covariance_matrix(self, historical_prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the covariance matrix of cryptocurrency returns.

        Args:
            historical_prices_df: DataFrame of historical prices with time index and coin columns.

        Returns:
            The covariance matrix of returns.
        """
        if historical_prices_df is None or historical_prices_df.empty:
            return pd.DataFrame()

        returns_df = historical_prices_df.pct_change().dropna()
        covariance_matrix = returns_df.cov()

        return covariance_matrix

    def calculate_risk_parity_weights(self, cov_matrix: pd.DataFrame, coins: list) -> pd.Series:
        """
        Calculates risk-parity weights for a portfolio of assets.

        Args:
            cov_matrix: The covariance matrix of asset returns (pandas DataFrame).
            coins: A list of asset symbols (strings) corresponding to the columns/index of the covariance matrix.

        Returns:
            A pandas Series where keys are coin symbols and values are their risk-parity weights.
            Returns equal weights if optimization fails or covariance matrix is invalid.
        """
        num_assets = len(coins)
        if cov_matrix.empty or cov_matrix.shape != (num_assets, num_assets):
            print("⚠️ Invalid covariance matrix for risk parity, returning equal weights.")
            return pd.Series(1.0 / num_assets, index=coins)

        # Ensure the covariance matrix columns match the coins list
        try:
            cov_matrix = cov_matrix.loc[coins, coins]
        except KeyError:
             print("⚠️ Covariance matrix columns do not match coin list, returning equal weights.")
             return pd.Series(1.0 / num_assets, index=coins)


        # Objective function for Risk Parity: minimize the sum of squared differences between risk contributions
        # We want each asset's risk contribution to be equal, so we minimize the difference
        # between the actual risk contribution and the average risk contribution.
        def risk_contribution_objective(weights, covariance_matrix):
            # Portfolio variance
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            # Handle potential division by zero or very small variance
            if portfolio_variance < 1e-9:
                 return np.sum(weights**2) # Return sum of squared weights if variance is zero or near zero

            # Portfolio volatility
            portfolio_volatility = np.sqrt(portfolio_variance)

            # Marginal Contribution to Risk (MCR)
            # MCR_i = (Cov * weights)_i / portfolio_volatility
            mcr = np.dot(covariance_matrix, weights) / portfolio_volatility

            # Asset Contribution to Risk (ACR)
            # ACR_i = weights_i * MCR_i
            acr = weights * mcr

            # We want ACR_i to be equal for all i.
            # Objective: minimize the sum of squared differences between ACR_i and the average ACR
            average_acr = np.mean(acr)
            objective = np.sum((acr - average_acr)**2)
            return objective


        # Constraints:
        # 1. Weights must sum to 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

        # Bounds:
        # Weights must be non-negative (no shorting)
        bounds = tuple((0, 1) for _ in range(num_assets))

        # Initial guess: equal weights
        initial_weights = np.array([1.0 / num_assets] * num_assets)

        # Perform the optimization
        try:
            result = sco.minimize(risk_contribution_objective, initial_weights, args=(cov_matrix,),
                                  method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                optimized_weights = pd.Series(result.x, index=coins)
                # Normalize weights to sum to 1 in case of minor numerical inaccuracies
                optimized_weights /= optimized_weights.sum()
                return optimized_weights
            else:
                print(f"⚠️ Risk parity optimization failed: {result.message}, returning equal weights.")
                return pd.Series(1.0 / num_assets, index=coins)

        except Exception as e:
            print(f"❌ Error during risk parity optimization: {e}, returning equal weights.")
            return pd.Series(1.0 / num_assets, index=coins)


    def run_single_strategy(self, coins, strategy, duration_minutes=2, lookback_period=30):
        cash = self.starting_cash
        holdings = {coin: 0.0 for coin in coins}
        entry_prices = {}
        portfolio_history = []
        trade_log = []
        end_time = datetime.now() + timedelta(minutes=duration_minutes)

        open_orders = {coin: [] for coin in coins}
        trailing_stops = {coin: None for coin in coins}

        consecutive_losses = 0

        strategy_params = self.params.get(strategy, {})
        strategy_timeframes = strategy_params.get("timeframes", ['1m'])
        min_price_variation_pct = strategy_params.get("min_price_variation_pct", 0.00005)

        peak_portfolio_value = self.starting_cash # Track peak portfolio value for drawdown calculation


        # The main simulation loop within the single strategy run
        while datetime.now() < end_time:
            prices = self.get_current_prices()
            if prices is None or len(prices) < len(coins) // 2:
                time.sleep(1)
                continue

            current_signals = {}

            # Process limit orders
            orders_to_execute = []
            for coin in coins:
                if coin in prices:
                    current_price = prices[coin]
                    indices_to_remove = []
                    for i, order in enumerate(open_orders.get(coin, [])):
                        if order['type'] == 'LIMIT_BUY' and current_price <= order['price']:
                            orders_to_execute.append((coin, order))
                            indices_to_remove.append(i)
                    for i in sorted(indices_to_remove, reverse=True):
                        del open_orders[coin][i]

            # Execute filled orders
            for coin, order in orders_to_execute:
                if holdings.get(coin, 0) == 0:
                    amount_to_buy = order['amount']
                    cost = amount_to_buy * order['price'] * (1 + self.transaction_cost)

                    if cash >= cost:
                        holdings[coin] += amount_to_buy
                        cash -= cost
                        entry_prices[coin] = order['price']
                        trade_log.append({
                            'timestamp': datetime.now(),
                            'action': 'LIMIT_BUY_FILLED',
                            'coin': coin,
                            'amount': amount_to_buy,
                            'price': order['price']
                        })
                        print(f"✅ [{strategy}] LIMIT BUY FILLED {coin}: {amount_to_buy:.4f} at ${order['price']:.2f}")

                        trailing_stop_pct = strategy_params.get("trailing_stop_pct")
                        if trailing_stop_pct is not None:
                            trailing_stops[coin] = {
                                'stop_price': entry_prices[coin] * (1 - trailing_stop_pct),
                                'trailing_pct': trailing_stop_pct,
                                'peak_price': entry_prices[coin]
                            }

            # Process trailing stops
            stops_to_trigger = []
            for coin in coins:
                if holdings.get(coin, 0) > 0 and coin in prices and trailing_stops.get(coin) is not None:
                    current_price = prices[coin]
                    stop_info = trailing_stops[coin]
                    peak_price = stop_info.get('peak_price', current_price)

                    if current_price > peak_price:
                        stop_info['peak_price'] = current_price
                        new_stop_price = peak_price * (1 - stop_info['trailing_pct'])
                        if new_stop_price > stop_info['stop_price']:
                            stop_info['stop_price'] = new_stop_price

                    if current_price <= stop_info['stop_price']:
                        stops_to_trigger.append(coin)

            # Execute trailing stops
            for coin in stops_to_trigger:
                if holdings.get(coin, 0) > 0 and coin in prices:
                    current_price = prices[coin]
                    cash += holdings[coin] * current_price * (1 - self.transaction_cost)
                    trade_log.append({
                        'timestamp': datetime.now(),
                        'action': 'TRAILING_STOP_TRIGGERED',
                        'coin': coin,
                        'amount': holdings[coin],
                        'price': current_price,
                        'stop_price': trailing_stops[coin]['stop_price']
                    })
                    print(f"🛑 [{strategy}] TRAILING STOP TRIGGERED {coin} at ${current_price:.2f}")
                    holdings[coin] = 0
                    entry_prices.pop(coin, None)
                    trailing_stops.pop(coin, None)
                    consecutive_losses += 1

            # Generate signals in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(coins)) as executor:
                future_to_coin = {}
                for coin in coins:
                    if coin in prices:
                        future_to_coin[executor.submit(self._calculate_signals_for_coin, coin, prices, strategy, lookback_period, min_price_variation_pct, strategy_timeframes)] = coin

                for future in concurrent.futures.as_completed(future_to_coin):
                    coin = future_to_coin[future]
                    try:
                        current_signals[coin] = future.result()
                    except Exception as e:
                        print(f"❌ Error calculating signals for {coin} in strategy {strategy}: {e}")
                        current_signals[coin] = 0 # Default to no signal on error


            # Add logging for current signals
            if current_signals:
                 print(f"[{strategy}] Signals at {datetime.now().strftime('%H:%M:%S')}: {current_signals}")


            current_volatility = self.calculate_portfolio_volatility(portfolio_history)

            # Execute trades based on signals
            for coin in coins:
                if coin in current_signals and coin in prices:
                    sig = current_signals[coin]
                    current_price = prices[coin]
                    strategy_params = self.params.get(strategy, {})

                    # Sell logic
                    if sig < 0 and holdings.get(coin, 0) > 0 and coin not in stops_to_trigger:
                        if open_orders.get(coin):
                            for order in open_orders[coin]:
                                trade_log.append({
                                    'timestamp': datetime.now(),
                                    'action': f"CANCELLED_{order['type']}_BEFORE_SELL",
                                    'coin': order['coin'],
                                    'amount': order['amount'],
                                    'price': order['price']
                                })
                            open_orders[coin] = []

                        if trailing_stops.get(coin):
                            trade_log.append({
                                'timestamp': datetime.now(),
                                'action': 'CANCELLED_TRAILING_STOP_BEFORE_SELL',
                                'coin': coin,
                                'stop_price': trailing_stops[coin]['stop_price']
                            })
                            trailing_stops[coin] = None

                        cash += holdings[coin] * current_price * (1 - self.transaction_cost)
                        trade_log.append({
                            'timestamp': datetime.now(),
                            'action': 'SELL',
                            'coin': coin,
                            'amount': holdings[coin],
                            'price': current_price
                        })
                        print(f"🔴 [{strategy}] SELL {coin}: {holdings[coin]:.4f} at ${current_price:.2f}")
                        holdings[coin] = 0
                        entry_prices.pop(coin, None)

                    # Buy logic
                    elif sig > 0 and cash > 10 and holdings.get(coin, 0) == 0 and not open_orders.get(coin):
                        if cash < 1:
                            break

                        # Use Kelly Criterion for position sizing
                        size = self.calculate_kelly_position_size(strategy, cash, trade_log)

                        # If Kelly size is too small or zero, potentially use dynamic or fixed size
                        if size < cash * 0.01: # Example threshold: if Kelly size is less than 1% of cash
                            # Decide on a fallback: dynamic, fixed, or skip trade
                            size = self.calculate_dynamic_position_size(strategy, cash, current_volatility) # Fallback to dynamic sizing


                        amount = size / current_price

                        limit_buy_price = current_price * (1 + strategy_params.get("limit_buy_offset_pct", -0.002))

                        if strategy_params.get("limit_buy_offset_pct") is not None:
                            open_orders[coin].append({
                                'type': 'LIMIT_BUY',
                                'coin': coin,
                                'price': limit_buy_price,
                                'amount': amount,
                                'timestamp': datetime.now()
                            })
                            print(f"🅿️ [{strategy}] PLACING LIMIT BUY {coin}: {amount:.4f} at ${limit_buy_price:.2f}")
                        else:
                            current_positions = sum(1 for h in holdings.values() if h > 0)
                            if current_positions >= 5:
                                break

                            holdings[coin] += amount
                            cash -= size
                            entry_prices[coin] = current_price
                            trade_log.append({
                                'timestamp': datetime.now(),
                                'action': 'BUY',
                                'coin': coin,
                                'amount': amount,
                                'price': current_price
                            })
                            print(f"🟢 [{strategy}] BUY {coin}: {amount:.4f} at ${current_price:.2f}")

                            trailing_stop_pct = strategy_params.get("trailing_stop_pct")
                            if trailing_stop_pct is not None:
                                trailing_stops[coin] = {
                                    'stop_price': current_price * (1 - trailing_stop_pct),
                                    'trailing_pct': trailing_stop_pct,
                                    'peak_price': current_price
                                }

            # Update portfolio value
            port_value = cash + sum(holdings.get(c, 0) * prices.get(c, 0) for c in coins)
            portfolio_history.append({
                'timestamp': datetime.now(),
                'value': port_value,
                'cash': cash,
                'holdings_value': port_value - cash
            })

            # Update peak portfolio value and check for drawdown
            peak_portfolio_value = max(peak_portfolio_value, port_value)
            if port_value < peak_portfolio_value * (1 - self.max_drawdown_pct):
                 print(f"💥 [{strategy}] Maximum drawdown ({self.max_drawdown_pct*100:.1f}%) reached at {datetime.now().strftime('%H:%M:%S')}. Stopping simulation for this strategy.")
                 # Add a log entry for the stop
                 trade_log.append({
                     'timestamp': datetime.now(),
                     'action': 'MAX_DRAWDOWN_STOP',
                     'coin': 'PORTFOLIO', # Indicate portfolio level stop
                     'amount': port_value, # Log the value at stop
                     'price': peak_portfolio_value # Log the peak value
                 })
                 break # Exit the simulation loop for this strategy


            time.sleep(self.update_interval)

        # Clean up remaining orders (if simulation stopped due to drawdown or end time)
        for coin in coins:
            if open_orders.get(coin):
                for order in open_orders[coin]:
                    trade_log.append({
                        'timestamp': datetime.now(),
                        'action': f"CANCELLED_{order['type']}",
                        'coin': order['coin'],
                        'amount': order['amount'],
                        'price': order['price']
                    })
                open_orders[coin] = []

            if trailing_stops.get(coin):
                trade_log.append({
                    'timestamp': datetime.now(),
                    'action': 'CANCELLED_TRAILING_STOP',
                    'coin': coin,
                    'stop_price': trailing_stops[coin]['stop_price']
                })
                trailing_stops[coin] = None

        # Calculate VaR and ES at the end of the simulation
        risk_metrics = self.calculate_var_es(portfolio_history)

        final_value = portfolio_history[-1]['value'] if portfolio_history else self.starting_cash
        result = {
            'Strategy': strategy,
            'Final Value': final_value,
            'Return': (final_value - self.starting_cash) / self.starting_cash * 100,
            'Trades': len([t for t in trade_log if t['action'] in ['BUY', 'SELL', 'STOP_LOSS', 'TAKE_PROFIT', 'LIMIT_BUY_FILLED', 'TRAILING_STOP_TRIGGERED']]),
            'Win Rate': self.calculate_win_rate(trade_log),
            'VaR (99%)': risk_metrics['VaR'],
            'ES (99%)': risk_metrics['ES'],
            'Max Drawdown (%)': (1 - (min([p.get('value', self.starting_cash) for p in portfolio_history] + [self.starting_cash]) / max([p.get('value', self.starting_cash) for p in portfolio_history] + [self.starting_cash]))) * 100 # Calculate actual max drawdown
        }
        return result, portfolio_history, trade_log

    def _calculate_signals_for_coin(self, coin, prices, strategy, lookback_period, min_price_variation_pct, strategy_timeframes):
         """Helper function to calculate signals for a single coin across timeframes."""
         multi_timeframe_signals = {}
         for timeframe in strategy_timeframes:
             ph = self.get_price_history(coin, timeframe=timeframe)
             indicators = self.calculate_technical_indicators(ph, strategy, lookback_period)
             signal = self.enhanced_generate_signal(ph, strategy, indicators, lookback_period, min_price_variation_pct)
             multi_timeframe_signals[timeframe] = signal

         # Determine the final signal based on multi-timeframe confirmation
         final_signal = 0
         if strategy_timeframes:
             largest_timeframe = strategy_timeframes[-1]
             final_signal = multi_timeframe_signals.get(largest_timeframe, 0)

         return final_signal


    def run_parallel_strategies(self, coins, strategies, duration_minutes=2, lookback_period=30):
        self.start_real_time_data(coins)
        time.sleep(5)

        results_all = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            future_to_strategy = {executor.submit(self.run_single_strategy, coins, strat, duration_minutes, lookback_period): strat for strat in strategies}

            for future in concurrent.futures.as_completed(future_to_strategy):
                strat = future_to_strategy[future]
                try:
                    res, ph, log = future.result()
                    results_all[strat] = {
                        'results': res,
                        'portfolio_history': ph,
                        'trade_log': log
                    }
                    print(f"✅ Strategy {strat} completed.")
                except Exception as e:
                    print(f"❌ Error running strategy {strat}: {e}")
                    import traceback
                    traceback.print_exc()
                    results_all[strat] = {
                        'results': None,
                        'portfolio_history': [],
                        'trade_log': []
                    }

        self.stop_real_time_data()

        try:
            self.enhanced_plot_comparison(results_all)
        except Exception as e:
            print("⚠️ Error in enhanced_plot_comparison:", e)
            import traceback
            traceback.print_exc()

        return results_all

    ##############################
    # Plotting
    ##############################
    def enhanced_plot_comparison(self, all_results):
        fig, ax = plt.subplots(figsize=(12,6))
        for strat,data in all_results.items():
            vals = [p['value'] for p in data['portfolio_history']]
            ax.plot(vals,label=strat)
        ax.set_title("Portfolio Value Comparison")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(True,alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'portfolio_comparison.png'))
        print("📈 Plot saved as portfolio_comparison.png")

    ##############################
    # Enhanced Export Methods
    ##############################
    def export_results(self, all_results, filename=None):
        if filename is None:
            filename = f"trading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_data=[]
        for strategy,data in all_results.items():
            results_data.append({'Strategy':strategy,'Final_Value':data['results']['Final Value'],'Return_Pct':data['results']['Return'],'Total_Trades':data['results']['Trades']})
        df = pd.DataFrame(results_data)
        df.to_csv(os.path.join(self.output_dir,filename),index=False)
        print(f"📊 Results exported to {filename}")
        return df

    def enhanced_export_all_data(self, all_results):
        """Export all simulation data including detailed trade logs"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export summary results
        summary_file = f"simulation_summary_{timestamp}.csv"
        self.export_results(all_results, summary_file)
        
        # Export detailed trade logs for each strategy
        for strategy, data in all_results.items():
            trades_file = f"trades_{strategy}_{timestamp}.csv"
            if data['trade_log']:
                trades_df = pd.DataFrame(data['trade_log'])
                trades_df.to_csv(os.path.join(self.output_dir, trades_file), index=False)
                print(f"📋 Trade log exported for {strategy}")
        
        # Export portfolio history for each strategy
        for strategy, data in all_results.items():
            portfolio_file = f"portfolio_{strategy}_{timestamp}.csv"
            if data['portfolio_history']:
                portfolio_df = pd.DataFrame(data['portfolio_history'])
                portfolio_df.to_csv(os.path.join(self.output_dir, portfolio_file), index=False)
                print(f"📈 Portfolio history exported for {strategy}")
        
        print(f"📊 All data exported with timestamp: {timestamp}")

# ------------------------
# Usage
# ------------------------
if __name__=="__main__":
    coins = [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD",
        "XRP-USD", "DOT-USD", "AVAX-USD", "LTC-USD",
        "ATOM-USD", "LINK-USD", "NEAR-USD", "HBAR-USD",
        "ICP-USD", "AR-USD", "AAVE-USD", "SAND-USD",
        "DOGE-USD", "SHIB-USD"
    ]
    strategies = [
        "ATR_Breakout",
        "Breakout",
        "Momentum_Enhanced",
        "MA_Original",
        "MA_Fast",
        "MA_Enhanced",
        "MACD",
        "ADX_Trend",
        "MeanReversion"
    ]

    simulator = EnhancedParallelCryptoSimulatorCG(starting_cash=10000, update_interval=15)
    print("🚀 Starting 5-Hour Enhanced Crypto Trading Simulation")
    all_results = simulator.run_parallel_strategies(coins, strategies, duration_minutes=300)  # 5 hours
    simulator.enhanced_export_all_data(all_results)  # Use enhanced export
    print("✅ 5-Hour Simulation completed! All logs saved!")
