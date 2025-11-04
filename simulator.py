import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import threading
import time
import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import requests
import os
from datetime import datetime, timedelta

class EnhancedParallelCryptoSimulatorCG:
    def __init__(self, starting_cash=1000.0, update_interval=10):
        self.starting_cash = starting_cash
        self.transaction_cost = 0.001
        self.update_interval = update_interval
        self.live_data = {}
        self.is_running = False
        self.data_lock = threading.Lock()
        self.output_dir = "portfolio_logs"
        os.makedirs(self.output_dir, exist_ok=True)

        # Strategy parameters with added min_price_variation_pct, use_kama, limit_buy_offset_pct, and trailing_stop_pct
        self.params = {
            "MA_Original": {"window": 5, "max_position_pct": 0.15, "stop_loss_pct": 0.05, "take_profit_pct": 0.10, "min_price_variation_pct": 0.0003, "confirmation_periods": 1, "timeframes": ['1m', '5m']}, # Lowered min_price_variation_pct
            "MA_Fast": {"short_window": 3, "long_window": 10, "use_ema": True, "use_kama": False, "kama_fast_period": 2, "kama_slow_period": 30, "max_position_pct": 0.20, "stop_loss_pct": 0.08, "min_price_variation_pct": 0.0003, "confirmation_periods": 1, "timeframes": ['1m', '5m', '15m']}, # Lowered min_price_variation_pct, reduced confirmation_periods
            "MA_Enhanced": {"short_window": 5, "long_window": 20, "volatility_threshold": 0.0005, "max_position_pct": 0.12, "stop_loss_pct": 0.06, "take_profit_pct": 0.15, "min_price_variation_pct": 0.0005, "confirmation_periods": 1, "timeframes": ['1m', '5m']}, # Lowered volatility_threshold and min_price_variation_pct, reduced confirmation_periods
            "Momentum_Enhanced": {"period": 1, "threshold": 0.3, "smoothing": 1, "max_position_pct": 0.9, "stop_loss_pct": 0.7, "min_price_variation_pct": 0.0005, "confirmation_periods": 1, "timeframes": ['1m']}, # Lowered threshold
            "Breakout": {"period": 5, "max_position_pct": 0.10, "stop_loss_pct": 0.04, "take_profit_pct": 0.12, "min_price_variation_pct": 0.0005, "confirmation_periods": 1, "timeframes": ['1m', '15m']}, # Lowered min_price_variation_pct, reduced confirmation_periods
            "MACD": {"fast": 12, "slow": 26, "signal": 9, "max_position_pct": 0.15, "stop_loss_pct": 0.06, "min_price_variation_pct": 0.0005, "confirmation_periods": 1, "timeframes": ['1m', '5m']}, # Lowered min_price_variation_pct, reduced confirmation_periods
            "ATR_Breakout": {"period": 14, "multiplier": 1.5, "max_position_pct": 0.12, "stop_loss_pct": 2.0, "take_profit_pct": 3.0, "min_price_variation_pct": 0.0008, "confirmation_periods": 1, "timeframes": ['1m', '15m']}, # Lowered multiplier and min_price_variation_pct, reduced confirmation_periods
            "ADX_Trend": {"period": 14, "min_strength": 20, "max_position_pct": 0.10, "stop_loss_pct": 0.08, "min_price_variation_pct": 0.0005, "confirmation_periods": 1, "timeframes": ['1m', '5m']}, # Lowered min_strength and min_price_variation_pct, reduced confirmation_periods
            "MeanReversion": {"period": 20, "buy_threshold": 0.99, "sell_threshold": 1.01, "max_position_pct": 0.08, "take_profit_pct": 1.015, "min_price_variation_pct": 0.0003, "confirmation_periods": 1, "limit_buy_offset_pct": -0.003, "trailing_stop_pct": 0.02, "timeframes": ['1m', '5m']} # Adjusted thresholds, min_price_variation_pct, limit_buy_offset_pct, and trailing_stop_pct, reduced confirmation_periods
        }

        # Dictionnaire pour "traduire" nos noms en ID CoinGecko
        self.coingecko_id_map = {
         "BTC-USD": "90",
         "ETH-USD": "80",
         "XRP-USD": "58",
         "BNB-USD": "2710",
         "SOL-USD": "48543",
         "DOGE-USD": "2",
         "ADA-USD": "257",
         "LINK-USD": "2751",
         "HBAR-USD": "48555",
         "AVAX-USD": "44883",
         "LTC-USD": "1",
         "SHIB-USD": "45088",
         "DOT-USD": "45219",
         "AAVE-USD": "46018",
         "NEAR-USD": "48563",
         "ICP-USD": "47311",
         "ATOM-USD": "33830",
         "SAND-USD": "45161",
         "AR-USD": "42441"
     }
        # Dictionaries to store historical data at different granularities
        self.historical_data = {
            '1m': {},
            '5m': {},
            '15m': {}
        }
        self._last_update_time = None

    ##############################
    # CoinGecko Data Fetching - REPAIRED VERSION
    ##############################
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
                # Get valid CoinLore IDs (on réutilise coingecko_id_map pour les IDs CoinLore)
                coin_ids_for_api = []
                valid_coins = []

                for coin in self.coins:
                    if coin in self.coingecko_id_map:
                        coin_ids_for_api.append(self.coingecko_id_map[coin])
                        valid_coins.append(coin)
                    else:
                        print(f"⚠️ Warning: {coin} not found in CoinLore mapping (in coingecko_id_map)")

                if not coin_ids_for_api:
                    print("❌ Error: No valid coins to fetch.")
                    time.sleep(60)
                    continue

                # Construire URL pour CoinLore : /api/ticker/?id=ID1,ID2,...
                ids_param = ",".join(coin_ids_for_api)
                url = f"https://api.coinlore.net/api/ticker/?id={ids_param}"

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }

                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code != 200:
                    print(f"❌ API Error (CoinLore): Status code {response.status_code}")
                    print("DEBUG: response.text:", response.text)
                    time.sleep(60)
                    continue

                resp = response.json()
                # resp est une liste d’objets JSON, ex : [ { "id":"90", "symbol":"BTC", "price_usd":"12345.67", … }, … ]

                current_time = datetime.now()

                with self.data_lock:
                    updated_count = 0
                    for coin in valid_coins:
                        lore_id = self.coingecko_id_map[coin]
                        obj = None
                        for item in resp:
                            if str(item.get("id")) == str(lore_id):
                                obj = item
                                break

                        if obj is None:
                            print(f"⚠️ No JSON object for {coin} with id {lore_id}")
                            continue

                        price_usd = obj.get("price_usd")
                        if price_usd is None:
                            print(f"⚠️ price_usd missing for {coin}: {price_usd}")
                            continue

                        try:
                            price = float(price_usd)
                        except ValueError:
                            print(f"⚠️ Cannot convert price_usd to float for {coin}: {price_usd}")
                            continue

                        # print(f"🪙 {coin}: ${price:,.2f} USD (CoinLore)") # Suppress frequent logging

                        if coin not in self.live_data:
                            self.live_data[coin] = {
                                'history': pd.Series([price], index=[current_time]),
                                'price': price,
                                'timestamp': current_time
                            }
                            # Initialize historical_data for the coin
                            for timeframe in self.historical_data.keys():
                                self.historical_data[timeframe][coin] = pd.Series([price], index=[current_time])
                        else:
                            new_data = pd.Series([price], index=[current_time])
                            self.live_data[coin]['history'] = pd.concat([
                                self.live_data[coin]['history'],
                                new_data
                            ])#.tail(100) # Keep all data for downsampling
                            self.live_data[coin]['price'] = price
                            self.live_data[coin]['timestamp'] = current_time

                            # Update historical_data at different granularities
                            for timeframe, data_series_dict in self.historical_data.items():
                                # Determine the interval in minutes
                                interval_minutes = int(timeframe[:-1])
                                # Check if the current time aligns with the timeframe interval
                                if self._last_update_time is None or (current_time - self._last_update_time).total_seconds() >= interval_minutes * 60:
                                     # Append the current price if enough time has passed or it's the first update
                                     data_series_dict[coin] = pd.concat([
                                         data_series_dict.get(coin, pd.Series([], dtype=float)), # Handle initial case
                                         pd.Series([price], index=[current_time])
                                     ])


                        updated_count += 1

                    if updated_count > 0:
                         self._last_update_time = current_time # Update last update time only if data was successfully updated
                         print(f"✅ Updated {updated_count}/{len(valid_coins)} coins at {current_time.strftime('%H:%M:%S')}")


            except requests.exceptions.RequestException as e:
                print(f"❌ Network error fetching data (CoinLore): {e}")
                time.sleep(30)
            except Exception as e:
                print(f"❌ Unexpected error in data update (CoinLore): {e}")
                import traceback
                traceback.print_exc()
                time.sleep(30)

            time.sleep(self.update_interval)

    def get_current_prices(self):
        """Récupère les prix les plus récents de live_data."""
        with self.data_lock:
            # retourne un dict {coin: price}
            return {coin: data['price'] for coin, data in self.live_data.items() if 'price' in data}

    def get_price_history(self, coin, timeframe='1m', limit=None):
        """
        Récupère l’historique des prix pour un coin à une granularité donnée, limité si besoin.
        timeframe: '1m', '5m', '15m'
        """
        with self.data_lock:
            if timeframe in self.historical_data and coin in self.historical_data[timeframe]:
                history = self.historical_data[timeframe][coin]
                if limit is not None:
                    return history.tail(limit)
                return history
            # Fallback to the most granular data if requested timeframe is not available
            elif coin in self.live_data and 'history' in self.live_data[coin]:
                 print(f"⚠️ Requested timeframe '{timeframe}' not available for {coin}, using full history.")
                 history = self.live_data[coin]['history']
                 if limit is not None:
                     return history.tail(limit)
                 return history
            return pd.Series([]) # retourne une série vide si pas de données


    ##############################
    # Technical Indicators
    ##############################
    def calculate_technical_indicators(self, prices: pd.Series, strategy: str, lookback_period: int = 30):
        """
        Calculate additional technical indicators for better strategy decisions.
        `prices` : pd.Series (index temporel) des prix.
        `strategy` : (non utilisé ici, mais tu peux l’intégrer pour choisir les indicateurs à calculer).
        `lookback_period`: The number of periods to look back for calculations.
        Retourne un dict d’indicateurs calculés.
        """
        if prices is None or len(prices) < 2:
            return {}

        # Ensure we have enough data for the lookback period
        if len(prices) < lookback_period:
            # Use all available data if less than lookback_period
            prices_subset = prices
        else:
            prices_subset = prices.tail(lookback_period)

        indicators = {}

        # Volatilité & momentum
        if len(prices_subset) > 1:
            returns = prices_subset.pct_change().dropna()
            if not returns.empty:
                indicators['volatility'] = returns.std()
            else:
                 indicators['volatility'] = 0.0

            # momentum : variation sur lookback_period
            if len(prices_subset) >= lookback_period:
                 try:
                     indicators['momentum'] = (prices_subset.iloc[-1] / prices_subset.iloc[0] - 1)
                 except Exception:
                     indicators['momentum'] = 0.0
                 except IndexError: # Handle case where prices_subset might have only one element after subsetting
                    indicators['momentum'] = 0.0
            else:
                 # If not enough data for full lookback, calculate momentum over available data
                 if len(prices_subset) > 1:
                     try:
                         indicators['momentum'] = (prices_subset.iloc[-1] / prices_subset.iloc[0] - 1)
                     except Exception:
                         indicators['momentum'] = 0.0
                     except IndexError: # Handle case where prices_subset might have only one element after subsetting
                        indicators['momentum'] = 0.0
                 else:
                     indicators['momentum'] = 0.0


        # RSI (Relative Strength Index) — version “classique 14”
        rsi_period = 14 # Use standard RSI period, but ensure enough data
        if len(prices_subset) > rsi_period:
            delta = prices_subset.diff()
            # gains ; pertes
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            # moyenne sur fenêtre de 14 (simple)
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            # éviter division par zéro
            rs = avg_gain / (avg_loss + 1e-9)
            rsi_series = 100 - (100 / (1 + rs))
            # le RSI “dernier”
            last_rsi = rsi_series.iloc[-1]
            indicators['rsi'] = last_rsi
        else:
            indicators['rsi'] = 50 # Neutral RSI if not enough data

        return indicators

    ##############################
# Trading Strategies - Version corrigée et agressive
##############################
    def generate_signal(self, prices: pd.Series, strategy: str, lookback_period: int = 30, min_price_variation_pct: float = 0.0001) -> pd.Series:
        """
        Génère un signal de trading (1 = achat, -1 = vente, 0 = neutre) selon la stratégie donnée.
        Version optimisée : signaux plus réactifs, sans blocages de logique.
        """
        # Vérification de base
        if prices is None or prices.empty or len(prices) < 3:
            return pd.Series([0] * len(prices), index=prices.index if prices is not None else None)

        # Sélection de la fenêtre d'analyse
        prices_subset = prices.tail(lookback_period).copy() # Added .copy() to avoid SettingWithCopyWarning

        # Variations court terme / long terme
        short_var = (prices_subset.iloc[-1] - prices_subset.iloc[-2]) / prices_subset.iloc[-2] if len(prices_subset) >= 2 else 0
        long_var = (prices_subset.iloc[-1] / prices_subset.iloc[0] - 1) if len(prices_subset) > 1 else 0 # Changed from > 2 to > 1 for long_var calculation

        # Signal neutre initial
        sig = pd.Series(0, index=[prices_subset.index[-1]]) # Initialize with a single value for the last timestamp

        # Petit filtre anti-bruit (empêche de trader sur micro-variations)
        if abs(short_var) < min_price_variation_pct:
            return pd.Series([0] * len(prices), index=prices.index if prices is not None else None) # Return neutral signal for the whole series

        # --- LOGIQUE PAR STRATÉGIE ---
        p = self.params.get(strategy, {})
        s = 0  # signal scalaire

        # MA_Original
        if strategy == "MA_Original":
            window = min(p.get("window", 5), len(prices_subset))
            if window >= 2: # Ensure enough data for MA calculation
                ma = prices_subset.rolling(window=window).mean()
                if not ma.empty:
                    s = 1 if prices_subset.iloc[-1] > ma.iloc[-1] else -1

        # MA_Fast
        elif strategy == "MA_Fast":
            short_window = min(p.get("short_window", 5), len(prices_subset))
            long_window = min(p.get("long_window", 20), len(prices_subset))
            if short_window >= 2 and long_window >= 2: # Ensure enough data for MA calculation
                if p.get("use_ema", True):
                    short_ma = prices_subset.ewm(span=short_window, adjust=False).mean()
                    long_ma = prices_subset.ewm(span=long_window, adjust=False).mean()
                else:
                    short_ma = prices_subset.rolling(window=short_window).mean()
                    long_ma = prices_subset.rolling(window=long_window).mean()
                if not short_ma.empty and not long_ma.empty:
                    s = 1 if short_ma.iloc[-1] > long_ma.iloc[-1] else -1

        # MA_Enhanced
        elif strategy == "MA_Enhanced":
            short_window = min(p.get("short_window", 5), len(prices_subset))
            long_window = min(p.get("long_window", 20), len(prices_subset))
            if short_window >= 2 and long_window >= 2: # Ensure enough data for MA calculation
                vol_window = min(10, len(prices_subset))
                vol = prices_subset.pct_change().rolling(window=vol_window).std()
                short_ema = prices_subset.ewm(span=short_window, adjust=False).mean()
                long_ema = prices_subset.ewm(span=long_window, adjust=False).mean()
                if not vol.empty and not short_ema.empty and not long_ema.empty:
                    if vol.iloc[-1] > p.get("volatility_threshold", 0.0003):
                        s = 1 if short_ema.iloc[-1] > long_ema.iloc[-1] else -1

        # Momentum_Enhanced
        elif strategy == "Momentum_Enhanced":
            period = min(p.get("period", 1), len(prices_subset) - 1)
            if period >= 1: # Ensure enough data for momentum calculation
                mom = prices_subset.pct_change(period)
                smoothing = min(p.get("smoothing", 1), len(mom) if not mom.empty else 0)
                if smoothing > 1:
                    mom = mom.rolling(window=smoothing).mean()
                threshold = p.get("threshold", 0.2)
                if not mom.empty:
                    mom_val = mom.iloc[-1]
                    if mom_val > threshold: s = 1
                    elif mom_val < -threshold: s = -1

        # Breakout
        elif strategy == "Breakout":
            period = min(p.get("period", 5), len(prices_subset) - 1)
            if period >= 1: # Ensure enough data for breakout calculation
                res = prices_subset.rolling(window=period).max()
                sup = prices_subset.rolling(window=period).min()
                if not res.empty and not sup.empty and len(prices_subset) > 1:
                    if prices_subset.iloc[-1] > res.shift(1).iloc[-1]: s = 1
                    elif prices_subset.iloc[-1] < sup.shift(1).iloc[-1]: s = -1

        # MACD
        elif strategy == "MACD":
            fast = min(p.get("fast", 12), len(prices_subset))
            slow = min(p.get("slow", 26), len(prices_subset))
            signal_win = min(p.get("signal", 9), len(prices_subset))
            if fast >= 2 and slow >= 2 and signal_win >= 1: # Ensure enough data for MACD calculation
                exp1 = prices_subset.ewm(span=fast, adjust=False).mean()
                exp2 = prices_subset.ewm(span=slow, adjust=False).mean()
                macd = exp1 - exp2
                signal_line = macd.ewm(span=signal_win, adjust=False).mean()
                if not macd.empty and not signal_line.empty:
                    s = 1 if macd.iloc[-1] > signal_line.iloc[-1] else -1

        # ATR_Breakout
        elif strategy == "ATR_Breakout":
            period = min(p.get("period", 14), len(prices_subset) - 1)
            if period >= 1: # Ensure enough data for ATR calculation
                tr = prices_subset.diff().abs()
                atr = tr.rolling(window=period).mean()
                if not atr.empty:
                    high_band = prices_subset.rolling(window=period).max() + atr.iloc[-1] * p.get("multiplier", 1.5) # Use last ATR value
                    low_band = prices_subset.rolling(window=period).min() - atr.iloc[-1] * p.get("multiplier", 1.5) # Use last ATR value
                    if not high_band.empty and not low_band.empty and len(prices_subset) > 1:
                        if prices_subset.iloc[-1] > high_band.shift(1).iloc[-1]: s = 1
                        elif prices_subset.iloc[-1] < low_band.shift(1).iloc[-1]: s = -1

        # ADX_Trend
        elif strategy == "ADX_Trend":
            period = min(p.get("period", 14), len(prices_subset))
            if period >= 2: # Ensure enough data for ADX calculation
                diff = prices_subset.diff()
                up = diff.clip(lower=0)
                down = (-diff).clip(lower=0)
                avg_up = up.rolling(window=period).mean()
                avg_down = down.rolling(window=period).mean()
                if not avg_up.empty and not avg_down.empty:
                    denom = avg_up.iloc[-1] + avg_down.iloc[-1] + 1e-9 # Use last values
                    strength = abs(avg_up.iloc[-1] - avg_down.iloc[-1]) / denom
                    min_strength = p.get("min_strength", 20) / 100
                    dir_sig = 1 if prices_subset.iloc[-1] > prices_subset.rolling(window=period).mean().iloc[-1] else -1
                    if strength > min_strength: s = dir_sig

        # MeanReversion
        elif strategy == "MeanReversion":
            period = min(p.get("period", 20), len(prices_subset))
            if period >= 2: # Ensure enough data for Mean Reversion calculation
                rm = prices_subset.rolling(window=period).mean()
                if not rm.empty:
                    rz = prices_subset.iloc[-1] / rm.iloc[-1]
                    if rz < p.get("buy_threshold", 0.99): s = 1
                    elif rz > p.get("sell_threshold", 1.01): s = -1

        # --- COMBINAISON AVEC TENDANCE LONG TERME ---
        # Adjusted thresholds for long_var to be less strict
        if long_var > 0.002 and short_var > 0: # Lowered threshold
            s = max(s, 1)   # renforce les achats
        elif long_var < -0.002 and short_var < 0: # Lowered threshold
            s = min(s, -1)  # renforce les ventes

        sig.iloc[-1] = s
        return pd.Series([s] * len(prices), index=prices.index).ffill().fillna(0) # Return a series matching the original prices length

    def enhanced_generate_signal(self, prices: pd.Series, strategy: str, indicators: dict = None, lookback_period: int = 30, min_price_variation_pct: float = 0.0001):
        """
        Version améliorée combinant le signal de base et un filtre RSI.
        """
        if prices is None or prices.empty or len(prices) < 5:
            return 0

        # Pass min_price_variation_pct and lookback_period to generate_signal
        base_signal_series = self.generate_signal(prices, strategy, lookback_period, min_price_variation_pct)
        if base_signal_series is None or base_signal_series.empty:
            return 0

        # Dernier signal non nul
        non_zero = base_signal_series[base_signal_series != 0]
        signal_value = int(non_zero.iloc[-1]) if not non_zero.empty else 0

        # Filtre RSI si présent
        if indicators is not None:
            rsi = indicators.get("rsi")
            if rsi is not None:
                if signal_value > 0 and rsi > 70:
                    return 0
                if signal_value < 0 and rsi < 30:
                    return 0

        return signal_value

    ##############################
# Position sizing & trading
##############################
    def calculate_position_size(self, strategy: str, cash: float) -> float:
        """
        Taille de position fixe selon le pourcentage maximal défini.
        Plus agressive : minimum 20 % du capital par trade.
        """
        pct = self.params.get(strategy, {}).get("max_position_pct", 0.2)
        pct = max(0.2, pct)  # impose un plancher de 20 %
        return min(cash * pct, cash * 0.95)  # pas plus de 95 % du cash


    def calculate_dynamic_position_size(self, strategy: str, cash: float, current_volatility: float = None) -> float:
        """
        Taille de position dynamique : ajuste selon la volatilité.
        Version agressive : augmente la taille si volatilité faible.
        """
        base_pct = self.params.get(strategy, {}).get("max_position_pct", 0.15)

        if current_volatility is not None:
            if current_volatility < 0.01:
                base_pct *= 1.5  # plus agressif si volatilité faible
            elif current_volatility > 0.03:
                base_pct *= 0.5  # réduit si trop volatile

        return min(cash * base_pct, cash * 0.95)


    def calculate_portfolio_volatility(self, portfolio_history: list) -> float:
        """
        Calculer la volatilité du portefeuille sur les dernières périodes.
        """
        if portfolio_history is None or len(portfolio_history) < 5:
            return 0.0

        values = [p.get('value', 0) for p in portfolio_history[-10:] if 'value' in p]
        if len(values) < 2:
            return 0.0

        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        return float(np.std(returns)) if returns else 0.0


    def calculate_win_rate(self, trade_log: list, entry_prices: dict = None, current_prices: dict = None) -> float:
        """
        Calcul du win rate basé sur les trades effectués.
        """
        if not trade_log:
            return 0.0

        buy_trades = [t for t in trade_log if t.get('action') in ['BUY', 'LIMIT_BUY_FILLED']]
        total = len([t for t in trade_log if t.get('action') in [
            'BUY', 'SELL', 'STOP_LOSS', 'TAKE_PROFIT', 'LIMIT_BUY_FILLED', 'TRAILING_STOP_TRIGGERED'
        ]])
        return len(buy_trades) / total if total > 0 else 0.0


    def calculate_limit_buy_price(self, prices: pd.Series, strategy_params: dict) -> float or None:
        """
        Détermine le prix limite d’achat selon un offset plus agressif.
        """
        limit_offset_pct = strategy_params.get("limit_buy_offset_pct", -0.002)  # -0.2 % par défaut
        if prices is not None and not prices.empty:
            return prices.iloc[-1] * (1 + limit_offset_pct)
        return None


    def calculate_trailing_stop_price(self, peak_price: float, trailing_pct: float) -> float:
        """
        Calcul du prix de stop suiveur (trailing stop).
        Version agressive : trailing plus serré.
        """
        trailing_pct = max(trailing_pct * 0.7, 0.005)  # réduit de 30 %
        return peak_price * (1 - trailing_pct)


    ##############################
    # Run a single strategy
    ##############################
    def run_single_strategy(self, coins, strategy, duration_minutes=2, lookback_period=30):
        cash = self.starting_cash
        holdings = {coin: 0.0 for coin in coins}
        entry_prices = {}
        portfolio_history = []
        trade_log = []
        end_time = datetime.now() + timedelta(minutes=duration_minutes)

        # Initialize order management dictionaries
        open_orders = {coin: [] for coin in coins}
        # trailing_stops structure: {coin: {'stop_price': price, 'trailing_pct': pct, 'peak_price': price}}
        trailing_stops = {coin: None for coin in coins}


        consecutive_losses = 0
        max_consecutive_losses = 3

        # Dictionary to store recent signals for confirmation
        recent_signals = {coin: [] for coin in coins}
        confirmation_periods = self.params.get(strategy, {}).get("confirmation_periods", 1)
        # Get timeframes for multi-timeframe analysis
        strategy_timeframes = self.params.get(strategy, {}).get("timeframes", ['1m']) # Default to 1m if not specified


        # Assure que self.is_running vaut True (sinon la boucle ne tourne pas)
        self.is_running = True

        # Get min_price_variation_pct for the current strategy
        min_price_variation_pct = self.params.get(strategy, {}).get("min_price_variation_pct", 0.001) # Corrected parameter name


        while datetime.now() < end_time and self.is_running:
            prices = self.get_current_prices()
            # Si trop peu de données disponibles, attendre
            if prices is None or len(prices) < len(coins) // 2: # Added None check
                time.sleep(1)
                continue

            current_signals = {}
            indicators_dict = {}
            multi_timeframe_signals = {} # Store signals for each timeframe and coin

            # Process open orders (check if limit buys can be filled)
            orders_to_execute = []
            for coin in coins:
                if coin in prices:
                    current_price = prices[coin]
                    # Iterate through a copy of the list to allow modification
                    # Create a list of indices to remove after iteration
                    indices_to_remove = []
                    for i, order in enumerate(open_orders.get(coin, [])):
                        if order['type'] == 'LIMIT_BUY' and current_price <= order['price']:
                            # Limit buy condition met
                            orders_to_execute.append((coin, order))
                            indices_to_remove.append(i)

                    # Remove executed orders from the open_orders list for this coin
                    # Iterate in reverse to avoid index issues
                    for i in sorted(indices_to_remove, reverse=True):
                        del open_orders[coin][i]


            # Execute filled orders
            executed_buy_coins = set()
            for coin, order in orders_to_execute:
                # Ensure we only execute the order if we don't currently hold the coin
                if holdings.get(coin, 0) == 0:
                    amount_to_buy = order['amount']
                    cost = amount_to_buy * order['price'] * (1 + self.transaction_cost)

                    if cash >= cost:
                        holdings[coin] += amount_to_buy
                        cash -= cost
                        entry_prices[coin] = order['price'] # Entry price is the limit order price
                        trade_log.append({
                            'timestamp': datetime.now(),
                            'action': 'LIMIT_BUY_FILLED',
                            'coin': coin,
                            'amount': amount_to_buy,
                            'price': order['price']
                        })
                        print(f"✅ [{strategy}] LIMIT BUY FILLED {coin}: {amount_to_buy:.4f} at ${order['price']:.2f}")
                        executed_buy_coins.add(coin) # Mark this coin as bought via limit order

                        # Place a trailing stop after a position is opened via limit order
                        strategy_params = self.params.get(strategy, {})
                        trailing_stop_pct = strategy_params.get("trailing_stop_pct")
                        if trailing_stop_pct is not None:
                             trailing_stops[coin] = {
                                 'stop_price': entry_prices[coin] * (1 - trailing_stop_pct),
                                 'trailing_pct': trailing_stop_pct,
                                 'peak_price': entry_prices[coin] # Initialize peak price at entry
                             }
                             print(f"🅿️ [{strategy}] PLACED TRAILING STOP for {coin} at ${trailing_stops[coin]['stop_price']:.2f} (Initial Peak: ${trailing_stops[coin]['peak_price']:.2f})")


            # Process Trailing Stops
            stops_to_trigger = []
            for coin in coins:
                if holdings.get(coin, 0) > 0 and coin in prices and trailing_stops.get(coin) is not None:
                    current_price = prices[coin]
                    stop_info = trailing_stops[coin]
                    peak_price = stop_info.get('peak_price', current_price) # Get tracked peak price, default to current

                    # Update peak price if current price is higher
                    if current_price > peak_price:
                        stop_info['peak_price'] = current_price
                        # Update the stop price based on the new peak
                        new_stop_price = self.calculate_trailing_stop_price(stop_info['peak_price'], stop_info['trailing_pct'])
                        if new_stop_price > stop_info['stop_price']: # Only move stop up
                            stop_info['stop_price'] = new_stop_price
                            print(f"⬆️ [{strategy}] Trailing stop for {coin} moved up to ${stop_info['stop_price']:.2f} (New Peak: ${stop_info['peak_price']:.2f})")


                    # Check if current price has fallen below the trailing stop price
                    if current_price <= stop_info['stop_price']:
                        stops_to_trigger.append(coin)


            # Execute triggered trailing stops
            for coin in stops_to_trigger:
                 if holdings.get(coin, 0) > 0 and coin in prices: # Ensure we still hold the coin
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
                    print(f"🛑 [{strategy}] TRAILING STOP TRIGGERED {coin} at ${current_price:.2f} (Stop: ${trailing_stops[coin]['stop_price']:.2f})")
                    holdings[coin] = 0
                    entry_prices.pop(coin, None)
                    trailing_stops.pop(coin, None) # Remove the trailing stop
                    consecutive_losses += 1 # Consider a triggered stop as a loss for consecutive loss tracking
                    recent_signals[coin] = [] # Clear recent signals after selling


            # Calculate signals for each timeframe
            for coin in coins:
                if coin in prices:
                    multi_timeframe_signals[coin] = {}
                    for timeframe in strategy_timeframes:
                        ph = self.get_price_history(coin, timeframe=timeframe)
                        indicators = self.calculate_technical_indicators(ph, strategy, lookback_period)
                        # Use enhanced_generate_signal for each timeframe
                        signal = self.enhanced_generate_signal(ph, strategy, indicators, lookback_period, min_price_variation_pct)
                        multi_timeframe_signals[coin][timeframe] = signal

                    # Determine the final signal based on multi-timeframe confirmation
                    # A signal is confirmed if it appears on the largest timeframe
                    final_signal = 0
                    if strategy_timeframes:
                        largest_timeframe = strategy_timeframes[-1]
                        final_signal = multi_timeframe_signals[coin].get(largest_timeframe, 0)

                    current_signals[coin] = final_signal # Use the confirmed signal

            # Add logging for current signals - Moved outside the timeframe loop
            # This logs the FINAL confirmed signal for each coin in this timestep
            if current_signals:
                 print(f"[{strategy}] Signals at {datetime.now().strftime('%H:%M:%S')}: {current_signals}")


            # Calcul de la volatilité du portefeuille actuel
            current_volatility = self.calculate_portfolio_volatility(portfolio_history)

            # LOGIQUE DE VENTE — stop loss and take profit already handled
            # This is now primarily for placing new orders or market sells based on signals


            # Si plusieurs pertes consécutives, ralentir l’activité
            if consecutive_losses >= max_consecutive_losses:
                time.sleep(self.update_interval * 2)
                continue

            # Place New Orders / Execute Market Orders
            for coin in coins:
                if coin in current_signals and coin in prices:
                    sig = current_signals[coin]
                    current_price = prices[coin] # Get current price again for clarity
                    strategy_params = self.params.get(strategy, {})

                    # Action based on the confirmed signal (sig)
                    # Vente standard (Market Sell) - Only sell if currently holding the coin and no open sell orders (though we don't have sell orders yet)
                    # Also, don't sell if a trailing stop was just triggered for this coin in this time step
                    if sig < 0 and holdings.get(coin, 0) > 0 and coin not in stops_to_trigger:
                         # Check if there are any open buy orders for this coin, cancel them before selling
                        if coin in open_orders and open_orders[coin]:
                            print(f"Cancelling {len(open_orders[coin])} open orders for {coin} before selling.")
                            # Add cancellation to trade log
                            for order in open_orders[coin]:
                                trade_log.append({
                                    'timestamp': datetime.now(),
                                    'action': f"CANCELLED_{order['type']}_BEFORE_SELL",
                                    'coin': order['coin'],
                                    'amount': order['amount'],
                                    'price': order['price']
                                })
                            open_orders[coin] = [] # Cancel all open orders for this coin

                        # Cancel any active trailing stop for this position before selling
                        if coin in trailing_stops and trailing_stops[coin] is not None:
                            print(f"Cancelling trailing stop for {coin} before selling.")
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
                        recent_signals[coin] = [] # Clear recent signals after selling


                    # Logique d’achat — signal positif
                    # Only place a new buy order if not already holding the coin AND no open orders for this coin
                    elif sig > 0 and cash > 10 and holdings.get(coin, 0) == 0 and not open_orders.get(coin): # Added check for open orders
                         if cash < 1:
                            break

                         size = self.calculate_dynamic_position_size(strategy, cash, current_volatility)
                         amount = size / current_price # Calculate amount based on current price for size calculation

                         # Check for Limit Buy support and calculate limit price
                         limit_buy_price = self.calculate_limit_buy_price(self.get_price_history(coin), strategy_params)

                         if limit_buy_price is not None:
                             # Place a Limit Buy Order
                             open_orders[coin].append({
                                 'type': 'LIMIT_BUY',
                                 'coin': coin,
                                 'price': limit_buy_price,
                                 'amount': amount,
                                 'timestamp': datetime.now()
                             })
                             print(f"🅿️ [{strategy}] PLACING LIMIT BUY {coin}: {amount:.4f} at ${limit_buy_price:.2f}")
                             recent_signals[coin] = [] # Clear recent signals after placing order

                         else:
                             # Execute a Market Buy (original logic) - Only if no limit buy was placed
                             # Limite du nombre de positions ouvertes
                             current_positions = sum(1 for h in holdings.values() if h > 0)
                             if current_positions >= 5:
                                break

                             # Recalculate amount based on current price for market buy execution
                             amount = size / current_price

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
                             recent_signals[coin] = [] # Clear recent signals after buying

                             # Place a trailing stop after a market buy
                             strategy_params = self.params.get(strategy, {})
                             trailing_stop_pct = strategy_params.get("trailing_stop_pct")
                             if trailing_stop_pct is not None:
                                 trailing_stops[coin] = {
                                     'stop_price': entry_prices[coin] * (1 - trailing_stop_pct),
                                     'trailing_pct': trailing_stop_pct,
                                     'peak_price': entry_prices[coin] # Initialize peak price at entry
                                 }
                                 print(f"🅿️ [{strategy}] PLACED TRAILING STOP for {coin} at ${trailing_stops[coin]['stop_price']:.2f} (Initial Peak: ${trailing_stops[coin]['peak_price']:.2f})")


            # Calcul de la valeur du portefeuille
            port_value = cash + sum(holdings.get(c, 0) * prices.get(c, 0) for c in coins)
            portfolio_history.append({
                'timestamp': datetime.now(),
                'value': port_value,
                'cash': cash,
                'holdings_value': port_value - cash
            })

            # Add open orders and trailing stops to trade log for tracking (optional, for debugging/analysis)
            # for coin, orders in open_orders.items():
            #     for order in orders:
            #          trade_log.append({
            #              'timestamp': datetime.now(),
            #              'action': f"OPEN_{order['type']}",
            #              'coin': order['coin'],
            #              'amount': order['amount'],
            #              'price': order['price'],
            #              'order_time': order['timestamp']
            #          })
            # for coin, stop_info in trailing_stops.items():
            #      if stop_info:
            #          trade_log.append({
            #              'timestamp': datetime.now(),
            #              'action': 'OPEN_TRAILING_STOP',
            #              'coin': coin,
            #              'stop_price': stop_info['stop_price'],
            #              'peak_price': stop_info['peak_price']
            #          })


            time.sleep(self.update_interval)

        # At the end of the simulation, cancel any remaining open orders and trailing stops
        for coin in coins:
             if coin in open_orders and open_orders[coin]:
                 print(f"Cancelling {len(open_orders[coin])} remaining open orders for {coin} at end of simulation.")
                 for order in open_orders[coin]:
                      trade_log.append({
                          'timestamp': datetime.now(),
                          'action': f"CANCELLED_{order['type']}",
                          'coin': order['coin'],
                          'amount': order['amount'],
                          'price': order['price'], # Price at time of order placement
                          'cancel_price': prices.get(coin, order['price']) # Price at time of cancellation
                      })
                 open_orders[coin] = [] # Clear the list of open orders

             if coin in trailing_stops and trailing_stops[coin] is not None:
                 print(f"Cancelling remaining trailing stop for {coin} at end of simulation.")
                 trade_log.append({
                      'timestamp': datetime.now(),
                      'action': 'CANCELLED_TRAILING_STOP',
                      'coin': coin,
                      'stop_price': trailing_stops[coin]['stop_price'],
                      'cancel_price': prices.get(coin, trailing_stops[coin]['stop_price']) # Price at time of cancellation
                  })
                 trailing_stops[coin] = None


        # Résultat final
        final_value = portfolio_history[-1]['value'] if portfolio_history else self.starting_cash
        result = {
            'Strategy': strategy,
            'Final Value': final_value,
            'Return': (final_value - self.starting_cash) / self.starting_cash * 100,
            'Trades': len([t for t in trade_log if t['action'] in ['BUY', 'SELL', 'STOP_LOSS', 'TAKE_PROFIT', 'LIMIT_BUY_FILLED', 'TRAILING_STOP_TRIGGERED']]), # Count actual trades
            'Win Rate': self.calculate_win_rate([t for t in trade_log if t['action'] in ['BUY', 'LIMIT_BUY_FILLED']]) # Calculate win rate based on filled buys
        }
        return result, portfolio_history, trade_log

    ##############################
    # Run all strategies in parallel
    ##############################
    def run_parallel_strategies(self, coins, strategies, duration_minutes=2, lookback_period=30):
        # Démarrer la collecte des prix en arrière-plan
        self.start_real_time_data(coins)
        # attendre un peu pour que les premières données arrivent
        time.sleep(5)

        results_all = {}
        threads = []

        # fonction “worker” pour chaque stratégie
        def worker(strategy_name):
            try:
                res, ph, log = self.run_single_strategy(coins, strategy_name, duration_minutes, lookback_period)
            except Exception as e:
                print(f"❌ Error running strategy {strategy_name}: {e}")
                import traceback
                traceback.print_exc()
                # en cas d’erreur, on stocke un résultat partiel vide
                res, ph, log = None, [], []
            results_all[strategy_name] = {
                'results': res,
                'portfolio_history': ph,
                'trade_log': log
            }

        # lancer un thread pour chaque stratégie
        for strat in strategies:
            t = threading.Thread(target=worker, args=(strat,))
            threads.append(t)
            t.start()
            print(f"🚀 Running strategy: {strat}")

        # attendre que tous les threads finissent
        for t in threads:
            t.join()

        # arrêter la collecte
        self.stop_real_time_data()

        # tracer la comparaison des résultats (méthode que tu dois avoir ou définir)
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
